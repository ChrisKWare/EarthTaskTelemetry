"""Test automatic schema bootstrap / dev DB reset logic."""
import importlib
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from backend.app.db import check_schema, ensure_schema, Base, REQUIRED_SCHEMA

# Migration module has a leading digit — must use importlib.
_migration_002 = importlib.import_module("backend.migrations.002_water_schema")


def _create_old_schema_db(path: Path):
    """Create a SQLite DB with the pre-water schema (missing model_name,
    calmness_score, etc.)."""
    conn = sqlite3.connect(str(path))
    cursor = conn.cursor()

    # Old model_state: no model_name column, unique on player_id alone
    cursor.execute("""
        CREATE TABLE model_state (
            id INTEGER PRIMARY KEY,
            player_id TEXT NOT NULL UNIQUE,
            trained_ts_utc TEXT NOT NULL,
            n_samples INTEGER NOT NULL,
            coefficients_json TEXT,
            intercept REAL,
            mae REAL,
            status TEXT NOT NULL
        )
    """)

    # Old session_summary: no calmness_score, no company_id
    cursor.execute("""
        CREATE TABLE session_summary (
            id INTEGER PRIMARY KEY,
            player_id TEXT NOT NULL,
            session_id TEXT NOT NULL UNIQUE,
            task_version TEXT NOT NULL,
            fta_level REAL NOT NULL,
            fta_strict BOOLEAN NOT NULL,
            repetition_burden REAL NOT NULL,
            earth_score_bucket INTEGER NOT NULL,
            created_ts_utc TEXT NOT NULL
        )
    """)

    # Insert a row so we know the DB has data
    cursor.execute(
        "INSERT INTO model_state "
        "(player_id, trained_ts_utc, n_samples, status) "
        "VALUES (?, ?, ?, ?)",
        ("player-old", "2024-01-01T00:00:00Z", 1, "insufficient_data"),
    )

    conn.commit()
    conn.close()


class TestCheckSchema:
    """Tests for the pure check_schema() function."""

    def test_detects_missing_columns_on_old_db(self, tmp_path):
        db_path = tmp_path / "old.db"
        _create_old_schema_db(db_path)

        missing = check_schema(db_path)

        # model_state should report model_name missing
        assert "model_state" in missing
        assert "model_name" in missing["model_state"]

        # session_summary should report calmness_score and company_id missing
        assert "session_summary" in missing
        assert "calmness_score" in missing["session_summary"]
        assert "company_id" in missing["session_summary"]

        # company_registry table doesn't exist at all
        assert "company_registry" in missing

    def test_returns_empty_for_up_to_date_db(self, tmp_path):
        """A freshly-created DB (via create_all) should pass the check."""
        db_path = tmp_path / "fresh.db"
        from sqlalchemy import create_engine as ce
        fresh_engine = ce(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=fresh_engine)
        fresh_engine.dispose()

        missing = check_schema(db_path)
        assert missing == {}

    def test_detects_missing_table(self, tmp_path):
        """A DB with no tables at all should report everything missing."""
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db_path))
        conn.close()

        missing = check_schema(db_path)
        for table in REQUIRED_SCHEMA:
            assert table in missing


class TestEnsureSchema:
    """Tests for the ensure_schema() startup logic."""

    def test_dev_reset_backs_up_and_recreates(self, tmp_path):
        """ALLOW_DEV_DB_RESET=1 should back up the stale DB and create a
        fresh one that passes schema checks."""
        db_path = tmp_path / "telemetry.db"
        _create_old_schema_db(db_path)

        db_url = f"sqlite:///{db_path}"

        with mock.patch("backend.app.db.DATABASE_URL", db_url), \
             mock.patch("backend.app.db.engine", _make_engine(db_url)), \
             mock.patch.dict(os.environ, {"ALLOW_DEV_DB_RESET": "1"}):
            ensure_schema()

        # Old file should be renamed to .bak-*
        bak_files = list(tmp_path.glob("telemetry.db.bak-*"))
        assert len(bak_files) == 1

        # New file should exist and pass the schema check
        assert db_path.exists()
        assert check_schema(db_path) == {}

        # Backup should still have the old data
        conn = sqlite3.connect(str(bak_files[0]))
        cursor = conn.cursor()
        cursor.execute("SELECT player_id FROM model_state")
        rows = cursor.fetchall()
        conn.close()
        assert ("player-old",) in rows

    def test_no_reset_raises_clear_error(self, tmp_path):
        """Without ALLOW_DEV_DB_RESET, ensure_schema must raise RuntimeError
        listing the missing columns and the migration command."""
        db_path = tmp_path / "telemetry.db"
        _create_old_schema_db(db_path)

        db_url = f"sqlite:///{db_path}"

        with mock.patch("backend.app.db.DATABASE_URL", db_url), \
             mock.patch("backend.app.db.engine", _make_engine(db_url)), \
             mock.patch.dict(os.environ, {}, clear=False):
            # Make sure ALLOW_DEV_DB_RESET is NOT set
            os.environ.pop("ALLOW_DEV_DB_RESET", None)
            with pytest.raises(RuntimeError) as exc_info:
                ensure_schema()

        msg = str(exc_info.value)
        # Should mention the missing column
        assert "model_name" in msg
        assert "calmness_score" in msg
        # Should mention the migration command
        assert "002_water_schema" in msg

    def test_fresh_db_creates_cleanly(self, tmp_path):
        """If the DB file doesn't exist, ensure_schema creates it."""
        db_path = tmp_path / "telemetry.db"
        assert not db_path.exists()

        db_url = f"sqlite:///{db_path}"

        with mock.patch("backend.app.db.DATABASE_URL", db_url), \
             mock.patch("backend.app.db.engine", _make_engine(db_url)):
            ensure_schema()

        assert db_path.exists()
        assert check_schema(db_path) == {}

    def test_in_memory_always_works(self):
        """In-memory DBs (test path) should always succeed."""
        db_url = "sqlite:///:memory:"
        eng = _make_engine(db_url)

        with mock.patch("backend.app.db.DATABASE_URL", db_url), \
             mock.patch("backend.app.db.engine", eng):
            # Should not raise
            ensure_schema()

        eng.dispose()


class TestMigrationScript:
    """Tests for the 002_water_schema migration script."""

    def test_migration_adds_missing_columns(self, tmp_path):
        db_path = tmp_path / "telemetry.db"
        _create_old_schema_db(db_path)

        _migration_002.migrate(db_path)

        # Verify columns now exist
        missing = check_schema(db_path)
        # model_state and session_summary should now have all required cols
        assert "model_state" not in missing or "model_name" not in missing.get("model_state", [])
        assert "session_summary" not in missing or "calmness_score" not in missing.get("session_summary", [])

    def test_migration_is_idempotent(self, tmp_path):
        db_path = tmp_path / "telemetry.db"
        _create_old_schema_db(db_path)

        _migration_002.migrate(db_path)
        # Running again should not raise
        _migration_002.migrate(db_path)

        missing = check_schema(db_path)
        assert "model_state" not in missing or "model_name" not in missing.get("model_state", [])

    def test_migration_preserves_existing_data(self, tmp_path):
        db_path = tmp_path / "telemetry.db"
        _create_old_schema_db(db_path)

        _migration_002.migrate(db_path)

        # Old row should still be there with model_name defaulted to 'earth'
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT player_id, model_name FROM model_state")
        rows = cursor.fetchall()
        conn.close()

        assert ("player-old", "earth") in rows


def _make_engine(db_url: str):
    """Helper to create a disposable engine for testing."""
    from sqlalchemy import create_engine as ce
    return ce(db_url, connect_args={"check_same_thread": False})
