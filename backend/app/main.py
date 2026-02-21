"""FastAPI application for telemetry ingestion."""
import json
import os
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import Body, FastAPI, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .db import Base, engine, get_db, ensure_schema
from .models import RawEvent, SessionSummary, ModelState, CompanyRegistry
from .schemas import (
    AttemptSubmitted,
    FinalizeSessionRequest,
    StoredResponse,
    SessionSummaryResponse,
    ModelStateResponse,
    CompanyInfoResponse,
    CompanySummaryResponse,
    CompanyTimeseriesResponse,
    CompanyTimeseriesBucket,
    SeedCompanyRequest,
    SeedCompanyResponse,
)
from .metrics import compute_session_metrics
from .learner import retrain_player_model
from .company import compute_company_id, compute_dashboard_token, resolve_token_to_company_id
from .settings import ADMIN_KEY, MIN_COMPANY_N

# Validate schema and create tables on startup
ensure_schema()

app = FastAPI(title="Earth Task Telemetry API", version="0.1.0")

# CORS configuration
_allowed_origins_env = os.getenv("ALLOWED_ORIGINS")
if _allowed_origins_env:
    allowed_origins = [origin.strip() for origin in _allowed_origins_env.split(",")]
else:
    allowed_origins = ["http://localhost:8501"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/events", response_model=StoredResponse)
def ingest_event(event: AttemptSubmitted, db: Session = Depends(get_db)):
    """Ingest an AttemptSubmitted event."""
    # Compute company_id if company_name provided but company_id not
    company_id = event.company_id
    if event.company_name and not company_id:
        company_id = compute_company_id(event.company_name)

    raw_event = RawEvent(
        player_id=event.player_id,
        session_id=event.session_id,
        task_version=event.task_version,
        sequence_index=event.sequence_index,
        sequence_length=event.sequence_length,
        attempt_index=event.attempt_index,
        presented_json=json.dumps(event.presented),
        input_json=json.dumps(event.input),
        is_correct=event.is_correct,
        duration_ms=event.duration_ms,
        ts_utc=event.ts_utc,
        remediation_stage=event.remediation_stage,
        volts_count=event.volts_count,
        company_id=company_id,
    )
    db.add(raw_event)
    db.commit()
    return {"stored": True}


@app.post("/sessions/{session_id}/finalize", response_model=SessionSummaryResponse)
def finalize_session(
    session_id: str,
    body: Optional[FinalizeSessionRequest] = Body(None),
    db: Session = Depends(get_db),
):
    """Compute and store session summary metrics."""
    # Check if already finalized
    existing = db.query(SessionSummary).filter(
        SessionSummary.session_id == session_id
    ).first()
    if existing:
        return existing

    # Get all events for this session
    events = db.query(RawEvent).filter(RawEvent.session_id == session_id).all()
    if not events:
        raise HTTPException(status_code=404, detail="No events found for session")

    # Extract player_id and company_id from first event
    player_id = events[0].player_id
    company_id = events[0].company_id

    # Validate all events share the same player_id
    mismatched_players = [e for e in events if e.player_id != player_id]
    if mismatched_players:
        raise HTTPException(
            status_code=400,
            detail=f"Mixed player_ids in session: expected {player_id!r}, "
                   f"found {mismatched_players[0].player_id!r}",
        )

    # Validate all events share the same company_id
    mismatched = [e for e in events if e.company_id != company_id]
    if mismatched:
        raise HTTPException(
            status_code=400,
            detail=f"Mixed company_ids in session: expected {company_id!r}, "
                   f"found {mismatched[0].company_id!r}",
        )

    # Ensure company is registered (so dashboard token resolves via DB lookup)
    if company_id:
        existing_reg = db.query(CompanyRegistry).filter(
            CompanyRegistry.company_id == company_id
        ).first()
        if not existing_reg:
            db.add(CompanyRegistry(
                company_id=company_id,
                company_name=company_id,  # placeholder; admin can update via seed
                dashboard_token=compute_dashboard_token(company_id),
                created_ts_utc=datetime.now(timezone.utc).isoformat(),
            ))

    # Determine effective task version
    effective_task_version = (
        body.task_version
        if body and body.task_version
        else events[0].task_version
    )

    if effective_task_version == "water_v1":
        # Water path: require calmness_score
        if not body or body.calmness_score is None:
            raise HTTPException(
                status_code=400,
                detail="water_v1 sessions require calmness_score in request body",
            )

        summary = SessionSummary(
            player_id=player_id,
            session_id=session_id,
            task_version="water_v1",
            fta_level=None,
            fta_strict=None,
            repetition_burden=None,
            earth_score_bucket=None,
            created_ts_utc=datetime.now(timezone.utc).isoformat(),
            company_id=company_id,
            calmness_score=body.calmness_score,
        )
        db.add(summary)
        db.commit()
        db.refresh(summary)

        retrain_player_model(player_id, db, model_name="water")
    else:
        # Earth path (default): compute metrics
        metrics = compute_session_metrics(events)

        summary = SessionSummary(
            player_id=player_id,
            session_id=session_id,
            task_version=effective_task_version,
            fta_level=metrics["fta_level"],
            fta_strict=metrics["fta_strict"],
            repetition_burden=metrics["repetition_burden"],
            earth_score_bucket=metrics["earth_score_bucket"],
            created_ts_utc=datetime.now(timezone.utc).isoformat(),
            company_id=company_id,
            calmness_score=None,
        )
        db.add(summary)
        db.commit()
        db.refresh(summary)

        retrain_player_model(player_id, db, model_name="earth")

    return summary


@app.get("/players/{player_id}/sessions", response_model=List[SessionSummaryResponse])
def get_player_sessions(player_id: str, db: Session = Depends(get_db)):
    """Get all session summaries for a player, ordered by created_ts_utc."""
    summaries = (
        db.query(SessionSummary)
        .filter(SessionSummary.player_id == player_id)
        .order_by(SessionSummary.created_ts_utc)
        .all()
    )
    return summaries


@app.get("/model/state/{player_id}", response_model=ModelStateResponse)
def get_model_state(player_id: str, model_name: str = "earth", db: Session = Depends(get_db)):
    """Get the current model state for a player."""
    model_state = db.query(ModelState).filter(
        ModelState.player_id == player_id,
        ModelState.model_name == model_name,
    ).first()

    if not model_state:
        raise HTTPException(status_code=404, detail="No model state found for player")

    # Parse coefficients from JSON
    coefficients = None
    if model_state.coefficients_json:
        coefficients = json.loads(model_state.coefficients_json)

    return ModelStateResponse(
        player_id=model_state.player_id,
        model_name=model_state.model_name,
        trained_ts_utc=model_state.trained_ts_utc,
        n_samples=model_state.n_samples,
        coefficients=coefficients,
        intercept=model_state.intercept,
        mae=model_state.mae,
        status=model_state.status,
    )


# Company Dashboard Endpoints

@app.get("/dashboard/company", response_model=CompanyInfoResponse)
def get_company_info(t: str, db: Session = Depends(get_db)):
    """Validate dashboard token and return company info with player count."""
    company_id = resolve_token_to_company_id(t, db)
    if not company_id:
        raise HTTPException(status_code=403, detail="Invalid dashboard token")

    # Count unique players for this company
    n_players = db.query(SessionSummary.player_id).filter(
        SessionSummary.company_id == company_id
    ).distinct().count()

    has_sufficient = n_players >= MIN_COMPANY_N
    if has_sufficient:
        message = f"Company dashboard ready with {n_players} players."
    else:
        message = f"Insufficient players ({n_players}/{MIN_COMPANY_N}). Aggregates hidden for anonymity."

    player_ids_list = []
    if has_sufficient:
        rows = db.query(SessionSummary.player_id).filter(
            SessionSummary.company_id == company_id
        ).distinct().all()
        player_ids_list = [r[0] for r in rows]

    return CompanyInfoResponse(
        company_id=company_id,
        n_players=n_players,
        has_sufficient_data=has_sufficient,
        message=message,
        player_ids=player_ids_list,
    )


@app.get("/dashboard/summary", response_model=CompanySummaryResponse)
def get_company_summary(t: str, db: Session = Depends(get_db)):
    """Get aggregate stats for company. Blocked if insufficient players."""
    company_id = resolve_token_to_company_id(t, db)
    if not company_id:
        raise HTTPException(status_code=403, detail="Invalid dashboard token")

    # Count unique players
    n_players = db.query(SessionSummary.player_id).filter(
        SessionSummary.company_id == company_id
    ).distinct().count()

    if n_players < MIN_COMPANY_N:
        raise HTTPException(
            status_code=403,
            detail=f"Insufficient players ({n_players}/{MIN_COMPANY_N}) for anonymity"
        )

    # Get all sessions for this company
    sessions = db.query(SessionSummary).filter(
        SessionSummary.company_id == company_id
    ).all()

    n_sessions = len(sessions)
    if n_sessions == 0:
        raise HTTPException(status_code=404, detail="No sessions found for company")

    # Split by task type
    earth_sessions = [s for s in sessions if s.fta_level is not None]
    water_sessions = [s for s in sessions if s.calmness_score is not None]

    # Compute earth aggregates
    avg_brain_performance_score = None
    avg_repetition_burden = None
    avg_earth_score_bucket = None
    if earth_sessions:
        n_earth = len(earth_sessions)
        avg_brain_performance_score = sum(s.fta_level for s in earth_sessions) / n_earth
        avg_repetition_burden = sum(s.repetition_burden for s in earth_sessions) / n_earth
        avg_earth_score_bucket = sum(s.earth_score_bucket for s in earth_sessions) / n_earth

    # Compute water aggregates
    avg_calmness_score = None
    n_water = len(water_sessions)
    if water_sessions:
        avg_calmness_score = sum(s.calmness_score for s in water_sessions) / n_water

    return CompanySummaryResponse(
        company_id=company_id,
        n_players=n_players,
        n_sessions=n_sessions,
        avg_brain_performance_score=avg_brain_performance_score,
        avg_repetition_burden=avg_repetition_burden,
        avg_earth_score_bucket=avg_earth_score_bucket,
        n_water_sessions=n_water,
        avg_calmness_score=avg_calmness_score,
    )


@app.get("/dashboard/timeseries", response_model=CompanyTimeseriesResponse)
def get_company_timeseries(t: str, bucket: str = "day", db: Session = Depends(get_db)):
    """Get time-bucketed averages for company. Blocked if insufficient players."""
    if bucket not in ("day", "week"):
        raise HTTPException(status_code=400, detail="bucket must be 'day' or 'week'")

    company_id = resolve_token_to_company_id(t, db)
    if not company_id:
        raise HTTPException(status_code=403, detail="Invalid dashboard token")

    # Count unique players
    n_players = db.query(SessionSummary.player_id).filter(
        SessionSummary.company_id == company_id
    ).distinct().count()

    if n_players < MIN_COMPANY_N:
        raise HTTPException(
            status_code=403,
            detail=f"Insufficient players ({n_players}/{MIN_COMPANY_N}) for anonymity"
        )

    # Get all sessions for this company, ordered by time
    sessions = db.query(SessionSummary).filter(
        SessionSummary.company_id == company_id
    ).order_by(SessionSummary.created_ts_utc).all()

    if not sessions:
        return CompanyTimeseriesResponse(
            company_id=company_id,
            bucket_type=bucket,
            buckets=[],
        )

    # Group sessions into buckets
    from datetime import timedelta

    buckets_dict = {}
    for session in sessions:
        # Parse timestamp
        ts = datetime.fromisoformat(session.created_ts_utc.replace('Z', '+00:00'))

        # Compute bucket start
        if bucket == "day":
            bucket_start = ts.replace(hour=0, minute=0, second=0, microsecond=0)
            bucket_end = bucket_start + timedelta(days=1)
        else:  # week
            # Start of week (Monday)
            days_since_monday = ts.weekday()
            bucket_start = (ts - timedelta(days=days_since_monday)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            bucket_end = bucket_start + timedelta(days=7)

        bucket_key = bucket_start.isoformat()
        if bucket_key not in buckets_dict:
            buckets_dict[bucket_key] = {
                "bucket_start": bucket_start.isoformat(),
                "bucket_end": bucket_end.isoformat(),
                "sessions": [],
            }
        buckets_dict[bucket_key]["sessions"].append(session)

    # Compute averages for each bucket
    result_buckets = []
    for bucket_key in sorted(buckets_dict.keys()):
        bucket_data = buckets_dict[bucket_key]
        bucket_sessions = bucket_data["sessions"]
        n = len(bucket_sessions)

        earth_in_bucket = [s for s in bucket_sessions if s.fta_level is not None]
        water_in_bucket = [s for s in bucket_sessions if s.calmness_score is not None]

        avg_bps = None
        avg_burden = None
        if earth_in_bucket:
            n_earth = len(earth_in_bucket)
            avg_bps = sum(s.fta_level for s in earth_in_bucket) / n_earth
            avg_burden = sum(s.repetition_burden for s in earth_in_bucket) / n_earth

        avg_calmness = None
        n_water = len(water_in_bucket)
        if water_in_bucket:
            avg_calmness = sum(s.calmness_score for s in water_in_bucket) / n_water

        result_buckets.append(CompanyTimeseriesBucket(
            bucket_start=bucket_data["bucket_start"],
            bucket_end=bucket_data["bucket_end"],
            n_sessions=n,
            avg_brain_performance_score=avg_bps,
            avg_repetition_burden=avg_burden,
            n_water_sessions=n_water,
            avg_calmness_score=avg_calmness,
        ))

    return CompanyTimeseriesResponse(
        company_id=company_id,
        bucket_type=bucket,
        buckets=result_buckets,
    )


# Admin Endpoints

@app.post("/admin/seed_company", response_model=SeedCompanyResponse)
def seed_company(
    body: SeedCompanyRequest,
    db: Session = Depends(get_db),
    x_admin_key: str = Header(...),
):
    """Register a company for dashboard token validation.

    Requires X-ADMIN-KEY header matching the ADMIN_KEY env var.
    Upserts into CompanyRegistry so the company_id is available for
    token resolution even before gameplay data arrives.
    """
    if not ADMIN_KEY or x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing admin key")

    company_id = body.company_id or compute_company_id(body.company_name)
    dashboard_token = compute_dashboard_token(company_id)

    existing = db.query(CompanyRegistry).filter(
        CompanyRegistry.company_id == company_id
    ).first()

    if existing:
        existing.company_name = body.company_name
        existing.dashboard_token = dashboard_token
    else:
        registry_entry = CompanyRegistry(
            company_id=company_id,
            company_name=body.company_name,
            dashboard_token=dashboard_token,
            created_ts_utc=datetime.now(timezone.utc).isoformat(),
        )
        db.add(registry_entry)

    db.commit()

    return SeedCompanyResponse(
        company_id=company_id,
        company_name=body.company_name,
        dashboard_token=dashboard_token,
    )
