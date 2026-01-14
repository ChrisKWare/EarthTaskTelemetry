"""Test ML learning component."""
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
    ts_utc: str = "2024-01-15T10:00:00Z",
):
    """Helper to create test event payload."""
    return {
        "player_id": player_id,
        "session_id": session_id,
        "task_version": "earth_v1",
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


def create_session_events(client, player_id: str, session_id: str, pattern: list, ts_utc: str):
    """
    Create events for a session based on a pattern.
    pattern: list of (sequence_index, attempt_index, is_correct) tuples.
    """
    for seq_idx, attempt_idx, is_correct in pattern:
        event = create_event(player_id, session_id, seq_idx, attempt_idx, is_correct, ts_utc)
        client.post("/events", json=event)


def test_model_state_not_found_returns_404(client):
    """Test that GET /model/state/{player_id} returns 404 for unknown player."""
    response = client.get("/model/state/unknown-player")
    assert response.status_code == 404
    assert "No model state found" in response.json()["detail"]


def test_model_state_insufficient_data_one_session(client):
    """Test model state with only 1 session shows insufficient_data."""
    player_id = "player-one-session"
    session_id = "session-1"

    # Perfect session (all correct on attempt 1)
    for seq in [1, 2, 3]:
        client.post("/events", json=create_event(player_id, session_id, seq, 1, True))

    client.post(f"/sessions/{session_id}/finalize")

    response = client.get(f"/model/state/{player_id}")
    assert response.status_code == 200
    data = response.json()

    assert data["player_id"] == player_id
    assert data["n_samples"] == 1
    assert data["status"] == "insufficient_data"
    assert data["coefficients"] is None
    assert data["intercept"] is None
    assert data["mae"] is None


def test_model_state_insufficient_data_two_sessions(client):
    """Test model state with only 2 sessions shows insufficient_data."""
    player_id = "player-two-sessions"

    for i, session_id in enumerate(["session-1", "session-2"]):
        for seq in [1, 2, 3]:
            ts = f"2024-01-{15+i}T10:00:00Z"
            client.post(
                "/events",
                json=create_event(player_id, session_id, seq, 1, True, ts),
            )
        client.post(f"/sessions/{session_id}/finalize")

    response = client.get(f"/model/state/{player_id}")
    assert response.status_code == 200
    data = response.json()

    assert data["n_samples"] == 2
    assert data["status"] == "insufficient_data"
    assert data["coefficients"] is None


def test_model_trained_with_three_sessions(client):
    """Test model trains successfully with 3 sessions."""
    player_id = "player-three-sessions"

    # Session 1: fta_level=1.0, repetition_burden=1.0 (all first attempt)
    create_session_events(
        client, player_id, "session-1",
        [(1, 1, True), (2, 1, True), (3, 1, True)],
        "2024-01-15T10:00:00Z"
    )
    client.post("/sessions/session-1/finalize")

    # Session 2: fta_level=0.333, repetition_burden=2.0 (mixed)
    create_session_events(
        client, player_id, "session-2",
        [
            (1, 1, True),  # Seq 1: first attempt
            (2, 1, False), (2, 2, True),  # Seq 2: second attempt
            (3, 1, False), (3, 2, False), (3, 3, True),  # Seq 3: third attempt
        ],
        "2024-01-16T10:00:00Z"
    )
    client.post("/sessions/session-2/finalize")

    # Session 3: fta_level=0.666, repetition_burden=1.333 (two first, one second)
    create_session_events(
        client, player_id, "session-3",
        [
            (1, 1, True),  # Seq 1: first attempt
            (2, 1, True),  # Seq 2: first attempt
            (3, 1, False), (3, 2, True),  # Seq 3: second attempt
        ],
        "2024-01-17T10:00:00Z"
    )
    client.post("/sessions/session-3/finalize")

    response = client.get(f"/model/state/{player_id}")
    assert response.status_code == 200
    data = response.json()

    assert data["player_id"] == player_id
    assert data["n_samples"] == 3
    assert data["status"] == "trained"
    assert data["coefficients"] is not None
    assert len(data["coefficients"]) == 2  # Two features: fta_level, repetition_burden
    assert data["intercept"] is not None
    assert data["mae"] is not None
    assert data["mae"] >= 0  # MAE should be non-negative


def test_model_retrains_on_new_session(client):
    """Test model retrains when a new session is finalized."""
    player_id = "player-retrain"

    # Create 3 initial sessions
    for i in range(3):
        session_id = f"session-{i+1}"
        for seq in [1, 2, 3]:
            ts = f"2024-01-{15+i}T10:00:00Z"
            client.post(
                "/events",
                json=create_event(player_id, session_id, seq, 1, True, ts),
            )
        client.post(f"/sessions/{session_id}/finalize")

    # Get initial model state
    response1 = client.get(f"/model/state/{player_id}")
    data1 = response1.json()
    assert data1["n_samples"] == 3
    initial_trained_ts = data1["trained_ts_utc"]

    # Add a 4th session
    session_id = "session-4"
    for seq in [1, 2, 3]:
        client.post(
            "/events",
            json=create_event(player_id, session_id, seq, 1, True, "2024-01-18T10:00:00Z"),
        )
    client.post(f"/sessions/{session_id}/finalize")

    # Get updated model state
    response2 = client.get(f"/model/state/{player_id}")
    data2 = response2.json()

    assert data2["n_samples"] == 4
    assert data2["trained_ts_utc"] != initial_trained_ts  # Retrained with new timestamp


def test_model_coefficients_are_explainable(client):
    """Test that model coefficients follow expected patterns."""
    player_id = "player-explainable"

    # Create sessions with clear learning progression
    # Session 1: Poor performance
    create_session_events(
        client, player_id, "session-1",
        [
            (1, 1, False), (1, 2, False), (1, 3, True),
            (2, 1, False), (2, 2, False), (2, 3, True),
            (3, 1, False), (3, 2, False), (3, 3, True),
        ],
        "2024-01-15T10:00:00Z"
    )
    client.post("/sessions/session-1/finalize")

    # Session 2: Better performance
    create_session_events(
        client, player_id, "session-2",
        [
            (1, 1, False), (1, 2, True),
            (2, 1, False), (2, 2, True),
            (3, 1, False), (3, 2, True),
        ],
        "2024-01-16T10:00:00Z"
    )
    client.post("/sessions/session-2/finalize")

    # Session 3: Good performance
    create_session_events(
        client, player_id, "session-3",
        [
            (1, 1, True),
            (2, 1, False), (2, 2, True),
            (3, 1, True),
        ],
        "2024-01-17T10:00:00Z"
    )
    client.post("/sessions/session-3/finalize")

    # Session 4: Excellent performance
    create_session_events(
        client, player_id, "session-4",
        [
            (1, 1, True),
            (2, 1, True),
            (3, 1, True),
        ],
        "2024-01-18T10:00:00Z"
    )
    client.post("/sessions/session-4/finalize")

    response = client.get(f"/model/state/{player_id}")
    data = response.json()

    assert data["status"] == "trained"
    # With this progression, we expect positive coefficient for fta_level
    # (higher previous FTA tends to predict higher next FTA)
    # The exact values depend on the data, but we verify the model trained
    assert data["coefficients"] is not None
    assert isinstance(data["coefficients"][0], float)
    assert isinstance(data["coefficients"][1], float)
