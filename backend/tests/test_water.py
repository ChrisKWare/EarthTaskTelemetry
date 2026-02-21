"""Test Water task (water_v1) support."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.main import app
from backend.app.db import Base, get_db


@pytest.fixture(scope="function")
def test_db():
    """Create a fresh in-memory test database for each test."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.clear()
    engine.dispose()


@pytest.fixture
def client(test_db):
    """Test client with test database."""
    return TestClient(app)


def create_event(
    player_id: str,
    session_id: str,
    sequence_index: int,
    attempt_index: int,
    is_correct: bool,
    task_version: str = "earth_v1",
    ts_utc: str = "2024-01-15T10:00:00Z",
):
    """Helper to create test event payload."""
    return {
        "player_id": player_id,
        "session_id": session_id,
        "task_version": task_version,
        "sequence_index": sequence_index,
        "sequence_length": 4,
        "attempt_index": attempt_index,
        "presented": [1, 2, 3, 4],
        "input": [1, 2, 3, 4] if is_correct else [4, 3, 2, 1],
        "is_correct": is_correct,
        "duration_ms": 5000,
        "ts_utc": ts_utc,
        "remediation_stage": "none",
        "volts_count": None,
    }


def ingest_water_session(client, player_id, session_id, calmness_score, ts_utc="2024-01-15T10:00:00Z"):
    """Helper: ingest events for a water session and finalize it."""
    # Water sessions still need raw events in the DB (the events table stores them).
    for seq in [1, 2, 3]:
        event = create_event(
            player_id=player_id,
            session_id=session_id,
            sequence_index=seq,
            attempt_index=1,
            is_correct=True,
            task_version="water_v1",
            ts_utc=ts_utc,
        )
        resp = client.post("/events", json=event)
        assert resp.status_code == 200

    # Finalize with water body
    resp = client.post(
        f"/sessions/{session_id}/finalize",
        json={"task_version": "water_v1", "calmness_score": calmness_score},
    )
    return resp


def test_finalize_water_requires_calmness_score(client):
    """POST finalize for water session without calmness_score -> 400."""
    player_id = "player-water-no-calm"
    session_id = "session-water-no-calm"

    for seq in [1, 2, 3]:
        event = create_event(
            player_id=player_id,
            session_id=session_id,
            sequence_index=seq,
            attempt_index=1,
            is_correct=True,
            task_version="water_v1",
        )
        client.post("/events", json=event)

    # Finalize without calmness_score
    resp = client.post(
        f"/sessions/{session_id}/finalize",
        json={"task_version": "water_v1"},
    )
    assert resp.status_code == 400
    assert "calmness_score" in resp.json()["detail"]


def test_finalize_water_success(client):
    """POST events + finalize with water_v1 and calmness_score -> 200."""
    player_id = "player-water-ok"
    session_id = "session-water-ok"

    resp = ingest_water_session(client, player_id, session_id, calmness_score=0.75)
    assert resp.status_code == 200

    data = resp.json()
    assert data["task_version"] == "water_v1"
    assert data["calmness_score"] == 0.75
    assert data["fta_level"] is None
    assert data["fta_strict"] is None
    assert data["repetition_burden"] is None
    assert data["earth_score_bucket"] is None


def test_finalize_rejects_mixed_player_ids(client):
    """Events with different player_ids -> 400."""
    session_id = "session-mixed-players"

    # Insert events with two different player_ids for the same session
    event1 = create_event("player-A", session_id, 1, 1, True)
    event2 = create_event("player-B", session_id, 2, 1, True)

    client.post("/events", json=event1)
    client.post("/events", json=event2)

    resp = client.post(f"/sessions/{session_id}/finalize")
    assert resp.status_code == 400
    assert "player_id" in resp.json()["detail"].lower()


def test_earth_finalize_unchanged(client):
    """Existing earth finalization still works, calmness_score is None."""
    player_id = "player-earth-check"
    session_id = "session-earth-check"

    for seq in [1, 2, 3]:
        event = create_event(player_id, session_id, seq, 1, True)
        client.post("/events", json=event)

    resp = client.post(f"/sessions/{session_id}/finalize")
    assert resp.status_code == 200

    data = resp.json()
    assert data["task_version"] == "earth_v1"
    assert data["fta_level"] == 1.0
    assert data["fta_strict"] is True
    assert data["calmness_score"] is None


def test_model_state_two_models_per_player(client):
    """Same player has earth + water ModelState rows."""
    player_id = "player-dual-model"

    # Create 1 earth session (creates earth model state with insufficient_data)
    for seq in [1, 2, 3]:
        event = create_event(player_id, f"earth-sess-1", seq, 1, True, ts_utc="2024-01-15T10:00:00Z")
        client.post("/events", json=event)
    client.post("/sessions/earth-sess-1/finalize")

    # Create 1 water session (creates water model state with insufficient_data)
    ingest_water_session(client, player_id, "water-sess-1", 0.5, ts_utc="2024-01-15T11:00:00Z")

    # Both model states should exist
    earth_resp = client.get(f"/model/state/{player_id}?model_name=earth")
    water_resp = client.get(f"/model/state/{player_id}?model_name=water")

    assert earth_resp.status_code == 200
    assert water_resp.status_code == 200

    assert earth_resp.json()["model_name"] == "earth"
    assert water_resp.json()["model_name"] == "water"


def test_water_model_trains_on_water_sessions_only(client):
    """3 water sessions -> water model trained; earth model still insufficient."""
    player_id = "player-water-train"

    # 3 water sessions
    for i in range(3):
        ingest_water_session(
            client, player_id, f"water-sess-{i+1}",
            calmness_score=0.5 + i * 0.1,
            ts_utc=f"2024-01-{15+i}T10:00:00Z",
        )

    # Water model should be trained
    water_resp = client.get(f"/model/state/{player_id}?model_name=water")
    assert water_resp.status_code == 200
    water_data = water_resp.json()
    assert water_data["status"] == "trained"
    assert water_data["n_samples"] == 3
    assert water_data["coefficients"] is not None
    assert len(water_data["coefficients"]) == 1  # Single feature: calmness_score

    # Earth model should not exist (no earth sessions)
    earth_resp = client.get(f"/model/state/{player_id}?model_name=earth")
    assert earth_resp.status_code == 404


def test_water_model_insufficient_data(client):
    """2 water sessions -> status=insufficient_data."""
    player_id = "player-water-insuff"

    for i in range(2):
        ingest_water_session(
            client, player_id, f"water-sess-{i+1}",
            calmness_score=0.6 + i * 0.1,
            ts_utc=f"2024-01-{15+i}T10:00:00Z",
        )

    water_resp = client.get(f"/model/state/{player_id}?model_name=water")
    assert water_resp.status_code == 200
    water_data = water_resp.json()
    assert water_data["status"] == "insufficient_data"
    assert water_data["n_samples"] == 2


def test_get_model_state_with_model_name_param(client):
    """GET /model/state/{id}?model_name=water returns water model."""
    player_id = "player-water-param"

    ingest_water_session(client, player_id, "water-sess-1", 0.8)

    # Default (earth) should 404
    resp_default = client.get(f"/model/state/{player_id}")
    assert resp_default.status_code == 404

    # Explicit water should return water model
    resp_water = client.get(f"/model/state/{player_id}?model_name=water")
    assert resp_water.status_code == 200
    assert resp_water.json()["model_name"] == "water"
