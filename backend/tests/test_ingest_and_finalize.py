"""Test event ingestion and session finalization."""
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
    # Use in-memory SQLite with StaticPool for test isolation
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
    sequence_length: int = 4,
):
    """Helper to create test event payload."""
    return {
        "player_id": player_id,
        "session_id": session_id,
        "task_version": "earth_v1",
        "sequence_index": sequence_index,
        "sequence_length": sequence_length,
        "attempt_index": attempt_index,
        "presented": [1, 2, 3, 4],
        "input": [1, 2, 3, 4] if is_correct else [4, 3, 2, 1],
        "is_correct": is_correct,
        "duration_ms": 5000,
        "ts_utc": "2024-01-15T10:00:00Z",
        "remediation_stage": "none",
        "volts_count": None,
    }


def test_ingest_event_stores_successfully(client):
    """Test that POST /events stores an event."""
    event = create_event(
        player_id="player-123",
        session_id="session-abc",
        sequence_index=1,
        attempt_index=1,
        is_correct=True,
    )
    response = client.post("/events", json=event)
    assert response.status_code == 200
    assert response.json() == {"stored": True}


def test_finalize_all_first_attempt_correct(client):
    """
    Test: All 3 sequences solved on first attempt.
    Expected: fta_level=1.0, fta_strict=True, repetition_burden=1.0, earth_score_bucket=10
    """
    player_id = "player-perfect"
    session_id = "session-perfect"

    # All sequences correct on first attempt
    for seq in [1, 2, 3]:
        event = create_event(
            player_id=player_id,
            session_id=session_id,
            sequence_index=seq,
            attempt_index=1,
            is_correct=True,
        )
        response = client.post("/events", json=event)
        assert response.status_code == 200

    # Finalize
    response = client.post(f"/sessions/{session_id}/finalize")
    assert response.status_code == 200
    data = response.json()

    assert data["fta_level"] == 1.0
    assert data["fta_strict"] is True
    assert data["repetition_burden"] == 1.0
    assert data["earth_score_bucket"] == 10


def test_finalize_mixed_attempts(client):
    """
    Test: Sequence 1 correct on attempt 1, Sequence 2 correct on attempt 2,
          Sequence 3 correct on attempt 3.
    Expected: fta_level=0.333..., fta_strict=False, repetition_burden=2.0, earth_score_bucket=5
    """
    player_id = "player-mixed"
    session_id = "session-mixed"

    # Sequence 1: correct on attempt 1
    client.post("/events", json=create_event(player_id, session_id, 1, 1, True))

    # Sequence 2: wrong on attempt 1, correct on attempt 2
    client.post("/events", json=create_event(player_id, session_id, 2, 1, False))
    client.post("/events", json=create_event(player_id, session_id, 2, 2, True))

    # Sequence 3: wrong on attempts 1 and 2, correct on attempt 3
    client.post("/events", json=create_event(player_id, session_id, 3, 1, False))
    client.post("/events", json=create_event(player_id, session_id, 3, 2, False))
    client.post("/events", json=create_event(player_id, session_id, 3, 3, True))

    # Finalize
    response = client.post(f"/sessions/{session_id}/finalize")
    assert response.status_code == 200
    data = response.json()

    # fta_level = (1 + 0 + 0) / 3 = 0.333...
    assert abs(data["fta_level"] - (1 / 3)) < 0.001
    assert data["fta_strict"] is False
    # repetition_burden = (1 + 2 + 3) / 3 = 2.0
    assert data["repetition_burden"] == 2.0
    # All solved but not all on first attempt
    assert data["earth_score_bucket"] == 5


def test_finalize_one_sequence_unsolved(client):
    """
    Test: Sequences 1 and 2 solved on first attempt, Sequence 3 never solved (all wrong).
    Expected: fta_level=0.666..., fta_strict=False, repetition_burden=5/3, earth_score_bucket=0
    """
    player_id = "player-fail"
    session_id = "session-fail"

    # Sequence 1: correct on attempt 1
    client.post("/events", json=create_event(player_id, session_id, 1, 1, True))

    # Sequence 2: correct on attempt 1
    client.post("/events", json=create_event(player_id, session_id, 2, 1, True))

    # Sequence 3: all attempts wrong
    client.post("/events", json=create_event(player_id, session_id, 3, 1, False))
    client.post("/events", json=create_event(player_id, session_id, 3, 2, False))
    client.post("/events", json=create_event(player_id, session_id, 3, 3, False))

    # Finalize
    response = client.post(f"/sessions/{session_id}/finalize")
    assert response.status_code == 200
    data = response.json()

    # fta_level = (1 + 1 + 0) / 3 = 0.666...
    assert abs(data["fta_level"] - (2 / 3)) < 0.001
    assert data["fta_strict"] is False
    # repetition_burden = (1 + 1 + 3) / 3 = 5/3 ≈ 1.666...
    assert abs(data["repetition_burden"] - (5 / 3)) < 0.001
    # Not all solved
    assert data["earth_score_bucket"] == 0


def test_get_player_sessions(client):
    """Test GET /players/{player_id}/sessions returns ordered summaries."""
    player_id = "player-multi"

    # Create two sessions
    for i, session_id in enumerate(["session-1", "session-2"]):
        for seq in [1, 2, 3]:
            client.post(
                "/events",
                json=create_event(player_id, session_id, seq, 1, True),
            )
        client.post(f"/sessions/{session_id}/finalize")

    # Get player sessions
    response = client.get(f"/players/{player_id}/sessions")
    assert response.status_code == 200
    data = response.json()

    assert len(data) == 2
    assert data[0]["session_id"] == "session-1"
    assert data[1]["session_id"] == "session-2"


def test_finalize_no_events_returns_404(client):
    """Test that finalizing a session with no events returns 404."""
    response = client.post("/sessions/nonexistent-session/finalize")
    assert response.status_code == 404


def test_finalize_idempotent(client):
    """Test that calling finalize twice returns the same result."""
    player_id = "player-idem"
    session_id = "session-idem"

    for seq in [1, 2, 3]:
        client.post("/events", json=create_event(player_id, session_id, seq, 1, True))

    # Finalize twice
    response1 = client.post(f"/sessions/{session_id}/finalize")
    response2 = client.post(f"/sessions/{session_id}/finalize")

    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response1.json()["id"] == response2.json()["id"]
