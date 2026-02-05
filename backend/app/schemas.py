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
    company_name: Optional[str] = None  # Raw name (NOT stored, used to compute company_id)
    company_id: Optional[str] = None  # Pre-computed company ID (stored)


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


class ModelStateResponse(BaseModel):
    """Response schema for model state."""
    model_config = ConfigDict(from_attributes=True)

    player_id: str
    trained_ts_utc: str
    n_samples: int
    coefficients: Optional[List[float]] = None
    intercept: Optional[float] = None
    mae: Optional[float] = None
    status: str


class CompanyInfoResponse(BaseModel):
    """Response schema for company dashboard info."""
    company_id: str
    n_players: int
    has_sufficient_data: bool
    message: str


class CompanySummaryResponse(BaseModel):
    """Response schema for company aggregate stats."""
    company_id: str
    n_players: int
    n_sessions: int
    avg_brain_performance_score: float
    avg_repetition_burden: float
    avg_earth_score_bucket: float


class CompanyTimeseriesBucket(BaseModel):
    """Single bucket in company timeseries."""
    bucket_start: str
    bucket_end: str
    n_sessions: int
    avg_brain_performance_score: float
    avg_repetition_burden: float


class CompanyTimeseriesResponse(BaseModel):
    """Response schema for company time series data."""
    company_id: str
    bucket_type: str  # "day" or "week"
    buckets: List[CompanyTimeseriesBucket]
