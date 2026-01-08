"""FastAPI application for telemetry ingestion."""
import json
from datetime import datetime, timezone
from typing import List

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

from .db import Base, engine, get_db
from .models import RawEvent, SessionSummary
from .schemas import (
    AttemptSubmitted,
    StoredResponse,
    HealthResponse,
    SessionSummaryResponse,
)
from .metrics import compute_session_metrics

# Create tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Earth Task Telemetry API", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return {"ok": True}


@app.post("/events", response_model=StoredResponse)
def ingest_event(event: AttemptSubmitted, db: Session = Depends(get_db)):
    """Ingest an AttemptSubmitted event."""
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
    )
    db.add(raw_event)
    db.commit()
    return {"stored": True}


@app.post("/sessions/{session_id}/finalize", response_model=SessionSummaryResponse)
def finalize_session(session_id: str, db: Session = Depends(get_db)):
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

    # Extract player_id and task_version from first event
    player_id = events[0].player_id
    task_version = events[0].task_version

    # Compute metrics
    metrics = compute_session_metrics(events)

    # Create summary
    summary = SessionSummary(
        player_id=player_id,
        session_id=session_id,
        task_version=task_version,
        fta_level=metrics["fta_level"],
        fta_strict=metrics["fta_strict"],
        repetition_burden=metrics["repetition_burden"],
        earth_score_bucket=metrics["earth_score_bucket"],
        created_ts_utc=datetime.now(timezone.utc).isoformat(),
    )
    db.add(summary)
    db.commit()
    db.refresh(summary)

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
