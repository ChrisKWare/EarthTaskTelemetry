"""ML training logic for predicting next-session fta_level."""
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
    Prepare training data from session summaries.

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


def retrain_player_model(player_id: str, db: Session) -> ModelState:
    """
    Retrain the model for a specific player using all their finalized sessions.

    Returns the updated ModelState record.
    """
    # Get all sessions for this player, ordered by created_ts_utc
    sessions = (
        db.query(SessionSummary)
        .filter(SessionSummary.player_id == player_id)
        .order_by(SessionSummary.created_ts_utc)
        .all()
    )

    now_utc = datetime.now(timezone.utc).isoformat()
    n_sessions = len(sessions)

    # Check for existing model state
    existing_state = db.query(ModelState).filter(
        ModelState.player_id == player_id
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

    # Prepare training data
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
