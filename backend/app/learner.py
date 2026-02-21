"""ML training logic for predicting next-session metrics."""
import json
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sqlalchemy.orm import Session

from .models import SessionSummary, ModelState


MIN_SESSIONS_FOR_TRAINING = 3


def prepare_training_data(
    sessions: List[SessionSummary],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Prepare training data from earth session summaries.

    Features (X): previous fta_level, previous repetition_burden
    Target (y): next-session fta_level

    Returns None if insufficient data (need at least 2 sessions to create 1 sample).
    """
    if len(sessions) < 2:
        return None

    X = []
    y = []

    for i in range(len(sessions) - 1):
        prev_session = sessions[i]
        next_session = sessions[i + 1]

        X.append([prev_session.fta_level, prev_session.repetition_burden])
        y.append(next_session.fta_level)

    return np.array(X), np.array(y)


def prepare_water_training_data(
    sessions: List[SessionSummary],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Prepare training data from water session summaries.

    Features (X): previous calmness_score
    Target (y): next-session calmness_score

    Returns None if insufficient data (need at least 2 sessions to create 1 sample).
    """
    valid = [s for s in sessions if s.calmness_score is not None]
    if len(valid) < 2:
        return None

    X = []
    y = []

    for i in range(len(valid) - 1):
        X.append([valid[i].calmness_score])
        y.append(valid[i + 1].calmness_score)

    return np.array(X), np.array(y)


def train_model(X: np.ndarray, y: np.ndarray) -> Tuple[LinearRegression, float]:
    """
    Train a LinearRegression model and compute MAE.

    Uses simple train-on-all approach for MAE computation.
    """
    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)
    mae = mean_absolute_error(y, predictions)

    return model, mae


def retrain_player_model(player_id: str, db: Session, model_name: str = "earth") -> ModelState:
    """
    Retrain the model for a specific player using their finalized sessions.

    model_name="earth" trains on earth_v1 sessions using fta_level/repetition_burden.
    model_name="water" trains on water_v1 sessions using calmness_score.

    Returns the updated ModelState record.
    """
    now_utc = datetime.now(timezone.utc).isoformat()

    # Get sessions filtered by task type
    if model_name == "water":
        sessions = (
            db.query(SessionSummary)
            .filter(
                SessionSummary.player_id == player_id,
                SessionSummary.task_version == "water_v1",
                SessionSummary.calmness_score.isnot(None),
            )
            .order_by(SessionSummary.created_ts_utc)
            .all()
        )
    else:
        sessions = (
            db.query(SessionSummary)
            .filter(
                SessionSummary.player_id == player_id,
                SessionSummary.task_version == "earth_v1",
                SessionSummary.fta_level.isnot(None),
            )
            .order_by(SessionSummary.created_ts_utc)
            .all()
        )

    n_sessions = len(sessions)

    # Check for existing model state (filtered by model_name)
    existing_state = db.query(ModelState).filter(
        ModelState.player_id == player_id,
        ModelState.model_name == model_name,
    ).first()

    # Guardrail: need at least MIN_SESSIONS_FOR_TRAINING finalized sessions
    if n_sessions < MIN_SESSIONS_FOR_TRAINING:
        if existing_state:
            existing_state.trained_ts_utc = now_utc
            existing_state.n_samples = n_sessions
            existing_state.coefficients_json = None
            existing_state.intercept = None
            existing_state.mae = None
            existing_state.status = "insufficient_data"
        else:
            existing_state = ModelState(
                player_id=player_id,
                model_name=model_name,
                trained_ts_utc=now_utc,
                n_samples=n_sessions,
                coefficients_json=None,
                intercept=None,
                mae=None,
                status="insufficient_data",
            )
            db.add(existing_state)

        db.commit()
        db.refresh(existing_state)
        return existing_state

    # Prepare training data based on model type
    if model_name == "water":
        result = prepare_water_training_data(sessions)
    else:
        result = prepare_training_data(sessions)

    if result is None:
        # Should not happen if n_sessions >= 3, but handle gracefully
        if existing_state:
            existing_state.trained_ts_utc = now_utc
            existing_state.n_samples = n_sessions
            existing_state.status = "insufficient_data"
        else:
            existing_state = ModelState(
                player_id=player_id,
                model_name=model_name,
                trained_ts_utc=now_utc,
                n_samples=n_sessions,
                status="insufficient_data",
            )
            db.add(existing_state)
        db.commit()
        db.refresh(existing_state)
        return existing_state

    X, y = result

    # Train model
    model, mae = train_model(X, y)

    # Update or create model state
    coefficients_json = json.dumps(model.coef_.tolist())

    if existing_state:
        existing_state.trained_ts_utc = now_utc
        existing_state.n_samples = n_sessions
        existing_state.coefficients_json = coefficients_json
        existing_state.intercept = float(model.intercept_)
        existing_state.mae = float(mae)
        existing_state.status = "trained"
    else:
        existing_state = ModelState(
            player_id=player_id,
            model_name=model_name,
            trained_ts_utc=now_utc,
            n_samples=n_sessions,
            coefficients_json=coefficients_json,
            intercept=float(model.intercept_),
            mae=float(mae),
            status="trained",
        )
        db.add(existing_state)

    db.commit()
    db.refresh(existing_state)
    return existing_state
