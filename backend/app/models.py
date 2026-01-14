"""SQLAlchemy ORM models."""
from sqlalchemy import Column, Integer, String, Boolean, Float, Text, DateTime
from datetime import datetime, timezone

from .db import Base


class RawEvent(Base):
    """Stores individual AttemptSubmitted events."""
    __tablename__ = "raw_events"

    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(String, index=True, nullable=False)
    session_id = Column(String, index=True, nullable=False)
    task_version = Column(String, nullable=False)
    sequence_index = Column(Integer, nullable=False)
    sequence_length = Column(Integer, nullable=False)
    attempt_index = Column(Integer, nullable=False)
    presented_json = Column(Text, nullable=False)
    input_json = Column(Text, nullable=False)
    is_correct = Column(Boolean, nullable=False)
    duration_ms = Column(Integer, nullable=False)
    ts_utc = Column(String, nullable=False)
    remediation_stage = Column(String, nullable=False)
    volts_count = Column(Integer, nullable=True)


class SessionSummary(Base):
    """Stores computed per-session metrics."""
    __tablename__ = "session_summary"

    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(String, index=True, nullable=False)
    session_id = Column(String, unique=True, nullable=False)
    task_version = Column(String, nullable=False)
    fta_level = Column(Float, nullable=False)
    fta_strict = Column(Boolean, nullable=False)
    repetition_burden = Column(Float, nullable=False)
    earth_score_bucket = Column(Integer, nullable=False)
    created_ts_utc = Column(String, nullable=False)


class ModelState(Base):
    """Stores trained model state for predicting next-session fta_level."""
    __tablename__ = "model_state"

    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(String, unique=True, index=True, nullable=False)
    trained_ts_utc = Column(String, nullable=False)
    n_samples = Column(Integer, nullable=False)
    coefficients_json = Column(Text, nullable=True)  # JSON array of coefficients
    intercept = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    status = Column(String, nullable=False)  # "trained" or "insufficient_data"
