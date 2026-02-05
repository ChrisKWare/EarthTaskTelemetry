"""Tests for company dashboard endpoints."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.main import app
from backend.app.db import Base, get_db
from backend.app.company import (
    normalize_company_name,
    compute_company_id,
    compute_dashboard_token,
)


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
    company_name: str = None,
    company_id: str = None,
    sequence_length: int = 4,
):
    """Helper to create test event payload."""
    event = {
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
    if company_name:
        event["company_name"] = company_name
    if company_id:
        event["company_id"] = company_id
    return event


class TestCompanyIdentityFunctions:
    """Tests for company identity utility functions."""

    def test_normalize_company_name_basic(self):
        """Test basic normalization."""
        assert normalize_company_name("Acme Corp") == "acme corp"
        assert normalize_company_name("  ACME  CORP  ") == "acme corp"

    def test_normalize_company_name_punctuation(self):
        """Test punctuation removal."""
        assert normalize_company_name("Acme, Inc.") == "acme inc"
        assert normalize_company_name("O'Brien & Co.") == "obrien  co"

    def test_normalize_company_name_preserves_spaces(self):
        """Test multiple whitespace collapse."""
        assert normalize_company_name("Big   Company   Name") == "big company name"

    def test_compute_company_id_format(self):
        """Test company_id has correct format."""
        company_id = compute_company_id("Acme Corp")
        assert company_id.startswith("c_")
        assert len(company_id) == 14  # "c_" + 12 hex chars

    def test_compute_company_id_deterministic(self):
        """Test same input produces same output."""
        id1 = compute_company_id("Acme Corp")
        id2 = compute_company_id("Acme Corp")
        assert id1 == id2

    def test_compute_company_id_normalized(self):
        """Test company_id is based on normalized name."""
        id1 = compute_company_id("Acme Corp")
        id2 = compute_company_id("  ACME   CORP  ")
        assert id1 == id2

    def test_compute_dashboard_token_format(self):
        """Test dashboard token has correct format."""
        company_id = compute_company_id("Acme Corp")
        token = compute_dashboard_token(company_id)
        assert len(token) == 24

    def test_compute_dashboard_token_deterministic(self):
        """Test same company_id produces same token."""
        company_id = compute_company_id("Acme Corp")
        token1 = compute_dashboard_token(company_id)
        token2 = compute_dashboard_token(company_id)
        assert token1 == token2


class TestEventsWithCompany:
    """Tests for /events endpoint with company fields."""

    def test_ingest_event_with_company_name(self, client):
        """Test that company_id is computed from company_name."""
        event = create_event(
            player_id="player-1",
            session_id="session-1",
            sequence_index=1,
            attempt_index=1,
            is_correct=True,
            company_name="Acme Corp",
        )
        response = client.post("/events", json=event)
        assert response.status_code == 200
        assert response.json() == {"stored": True}

    def test_ingest_event_with_company_id_direct(self, client):
        """Test that pre-computed company_id is stored."""
        company_id = compute_company_id("Acme Corp")
        event = create_event(
            player_id="player-1",
            session_id="session-1",
            sequence_index=1,
            attempt_index=1,
            is_correct=True,
            company_id=company_id,
        )
        response = client.post("/events", json=event)
        assert response.status_code == 200


class TestCompanyDashboardEndpoints:
    """Tests for company dashboard endpoints."""

    def _create_company_sessions(self, client, company_name: str, n_players: int):
        """Helper to create sessions for a company with n unique players."""
        company_id = compute_company_id(company_name)

        for i in range(n_players):
            player_id = f"player-{company_name}-{i}"
            session_id = f"session-{company_name}-{i}"

            # Create a simple session with all first-attempt correct
            for seq in [1, 2, 3]:
                event = create_event(
                    player_id=player_id,
                    session_id=session_id,
                    sequence_index=seq,
                    attempt_index=1,
                    is_correct=True,
                    company_id=company_id,
                )
                client.post("/events", json=event)

            # Finalize the session
            client.post(f"/sessions/{session_id}/finalize")

        return company_id

    def test_dashboard_company_invalid_token(self, client):
        """Test /dashboard/company returns 403 for invalid token."""
        response = client.get("/dashboard/company", params={"t": "invalid-token"})
        assert response.status_code == 403

    def test_dashboard_company_valid_token_insufficient_players(self, client):
        """Test /dashboard/company with valid token but < 5 players."""
        company_id = self._create_company_sessions(client, "SmallCo", n_players=3)
        token = compute_dashboard_token(company_id)

        response = client.get("/dashboard/company", params={"t": token})
        assert response.status_code == 200
        data = response.json()

        assert data["company_id"] == company_id
        assert data["n_players"] == 3
        assert data["has_sufficient_data"] is False

    def test_dashboard_company_valid_token_sufficient_players(self, client):
        """Test /dashboard/company with valid token and >= 5 players."""
        company_id = self._create_company_sessions(client, "BigCo", n_players=5)
        token = compute_dashboard_token(company_id)

        response = client.get("/dashboard/company", params={"t": token})
        assert response.status_code == 200
        data = response.json()

        assert data["company_id"] == company_id
        assert data["n_players"] == 5
        assert data["has_sufficient_data"] is True

    def test_dashboard_summary_invalid_token(self, client):
        """Test /dashboard/summary returns 403 for invalid token."""
        response = client.get("/dashboard/summary", params={"t": "invalid-token"})
        assert response.status_code == 403

    def test_dashboard_summary_insufficient_players(self, client):
        """Test /dashboard/summary returns 403 with < 5 players."""
        company_id = self._create_company_sessions(client, "TinyCo", n_players=2)
        token = compute_dashboard_token(company_id)

        response = client.get("/dashboard/summary", params={"t": token})
        assert response.status_code == 403

    def test_dashboard_summary_success(self, client):
        """Test /dashboard/summary returns aggregates with >= 5 players."""
        company_id = self._create_company_sessions(client, "GoodCo", n_players=5)
        token = compute_dashboard_token(company_id)

        response = client.get("/dashboard/summary", params={"t": token})
        assert response.status_code == 200
        data = response.json()

        assert data["company_id"] == company_id
        assert data["n_players"] == 5
        assert data["n_sessions"] == 5
        assert data["avg_brain_performance_score"] == 1.0  # All first-attempt correct
        assert data["avg_repetition_burden"] == 1.0

    def test_dashboard_timeseries_invalid_token(self, client):
        """Test /dashboard/timeseries returns 403 for invalid token."""
        response = client.get("/dashboard/timeseries", params={"t": "invalid"})
        assert response.status_code == 403

    def test_dashboard_timeseries_invalid_bucket(self, client):
        """Test /dashboard/timeseries returns 400 for invalid bucket type."""
        company_id = self._create_company_sessions(client, "TestCo", n_players=5)
        token = compute_dashboard_token(company_id)

        response = client.get(
            "/dashboard/timeseries",
            params={"t": token, "bucket": "invalid"}
        )
        assert response.status_code == 400

    def test_dashboard_timeseries_success(self, client):
        """Test /dashboard/timeseries returns bucketed data."""
        company_id = self._create_company_sessions(client, "TimeseriesCo", n_players=5)
        token = compute_dashboard_token(company_id)

        response = client.get(
            "/dashboard/timeseries",
            params={"t": token, "bucket": "day"}
        )
        assert response.status_code == 200
        data = response.json()

        assert data["company_id"] == company_id
        assert data["bucket_type"] == "day"
        assert len(data["buckets"]) >= 1

        # Check bucket structure
        bucket = data["buckets"][0]
        assert "bucket_start" in bucket
        assert "bucket_end" in bucket
        assert "n_sessions" in bucket
        assert "avg_brain_performance_score" in bucket
        assert "avg_repetition_burden" in bucket


class TestCompanyIdPropagation:
    """Tests for company_id propagation through the system."""

    def test_company_id_propagates_to_session_summary(self, client):
        """Test that company_id from events propagates to session summary."""
        company_id = compute_company_id("PropagationTest")
        player_id = "player-prop"
        session_id = "session-prop"

        # Create events with company_id
        for seq in [1, 2, 3]:
            event = create_event(
                player_id=player_id,
                session_id=session_id,
                sequence_index=seq,
                attempt_index=1,
                is_correct=True,
                company_id=company_id,
            )
            client.post("/events", json=event)

        # Finalize
        response = client.post(f"/sessions/{session_id}/finalize")
        assert response.status_code == 200

        # The summary should be accessible via the company dashboard
        token = compute_dashboard_token(company_id)
        response = client.get("/dashboard/company", params={"t": token})
        assert response.status_code == 200
        assert response.json()["n_players"] == 1

    def test_company_name_converted_to_id(self, client):
        """Test that company_name is correctly converted to company_id."""
        player_id = "player-name-test"
        session_id = "session-name-test"
        company_name = "Name Test Corp"

        # Create events with company_name (not company_id)
        for seq in [1, 2, 3]:
            event = create_event(
                player_id=player_id,
                session_id=session_id,
                sequence_index=seq,
                attempt_index=1,
                is_correct=True,
                company_name=company_name,
            )
            client.post("/events", json=event)

        # Finalize
        client.post(f"/sessions/{session_id}/finalize")

        # Verify via dashboard
        expected_company_id = compute_company_id(company_name)
        token = compute_dashboard_token(expected_company_id)
        response = client.get("/dashboard/company", params={"t": token})
        assert response.status_code == 200
        assert response.json()["company_id"] == expected_company_id
