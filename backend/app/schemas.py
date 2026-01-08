"""Pydantic schemas for request/response validation."""
from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field, ConfigDict


class AttemptSubmitted(BaseModel):
    """Schema for incoming AttemptSubmitted events."""
    player_id: str
    session_id: str
    task_version: str = "earth_v1"
    sequence_index: int = Field(..., ge=1)
    sequence_length: int = Field(..., ge=1)
    attempt_index: int = Field(..., ge=1, le=3)
    presented: List[Union[int, str]]
    input: List[Union[int, str]]
    is_correct: bool
    duration_ms: int = Field(..., ge=0)
    ts_utc: str
    remediation_stage: Literal["none", "volts", "space"]
    volts_count: Optional[int] = None


class StoredResponse(BaseModel):
    """Response for event storage."""
    stored: bool


class HealthResponse(BaseModel):
    """Response for health check."""
    ok: bool


class SessionSummaryResponse(BaseModel):
    """Response schema for session summary."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    player_id: str
    session_id: str
    task_version: str
    fta_level: float
    fta_strict: bool
    repetition_burden: float
    earth_score_bucket: int
    created_ts_utc: str
