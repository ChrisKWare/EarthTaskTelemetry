"""Database connection and session management."""
import logging
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from .settings import DATABASE_URL

logger = logging.getLogger(__name__)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Required columns per table.  Only columns that have been added after the
# initial schema need to be listed — the check is purely additive.
REQUIRED_SCHEMA = {
    "model_state": [
        "player_id", "model_name", "trained_ts_utc", "n_samples",
        "coefficients_json", "intercept", "mae", "status",
    ],
    "session_summary": [
        "player_id", "session_id", "task_version", "fta_level", "fta_strict",
        "repetition_burden", "earth_score_bucket", "created_ts_utc",
        "company_id", "calmness_score",
    ],
    "company_registry": [
        "company_id", "company_name", "dashboard_token", "created_ts_utc",
    ],
}


def _get_sqlite_path() -> Path | None:
    """Extract the filesystem path from a sqlite:/// URL.  Returns None for
    non-file databases (e.g. :memory: or non-SQLite engines)."""
    if not DATABASE_URL.startswith("sqlite:///"):
        return None
    raw = DATABASE_URL.replace("sqlite:///", "", 1)
    if raw == ":memory:" or raw == "":
        return None
    return Path(raw)


def _get_table_columns(cursor: sqlite3.Cursor, table: str) -> set[str]:
    cursor.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cursor.fetchall()}


def check_schema(db_path: Path) -> dict[str, list[str]]:
    """Return a dict of {table: [missing_columns]} for every table that is
    either missing entirely or lacks required columns.  An empty dict means
    the schema is up to date."""
    missing: dict[str, list[str]] = {}

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = {row[0] for row in cursor.fetchall()}

        for table, required_cols in REQUIRED_SCHEMA.items():
            if table not in existing_tables:
                missing[table] = required_cols
            else:
                actual_cols = _get_table_columns(cursor, table)
                cols_missing = [c for c in required_cols if c not in actual_cols]
                if cols_missing:
                    missing[table] = cols_missing
    finally:
        conn.close()

    return missing


def ensure_schema():
    """Run at application startup.  Guarantees the SQLite file has all
    required tables and columns, or fails fast with actionable guidance.

    Behaviour depends on the ALLOW_DEV_DB_RESET env-var:
      * "1"  → back up the stale DB and recreate it from scratch.
      * unset → raise RuntimeError listing every missing column and the
                exact migration command to run.

    For in-memory or non-SQLite databases the function simply delegates to
    ``Base.metadata.create_all()`` (the normal path for tests).
    """
    db_path = _get_sqlite_path()

    # Non-file DB (in-memory for tests, or non-SQLite): just create tables.
    if db_path is None:
        Base.metadata.create_all(bind=engine)
        return

    # DB file doesn't exist yet → fresh start, create everything.
    if not db_path.exists():
        Base.metadata.create_all(bind=engine)
        logger.info("Created new database at %s", db_path)
        return

    # DB file exists → check schema.
    missing = check_schema(db_path)

    if not missing:
        # Schema is current; still call create_all for any brand-new tables
        # that have no rows yet (create_all is additive for tables).
        Base.metadata.create_all(bind=engine)
        return

    # --- Schema is stale ---
    allow_reset = os.getenv("ALLOW_DEV_DB_RESET", "") == "1"

    if allow_reset:
        # Back up, then recreate.
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = db_path.with_suffix(f".db.bak-{ts}")
        shutil.move(str(db_path), str(backup_path))
        logger.warning(
            "Schema mismatch detected.  Old DB backed up to %s.  "
            "Recreating fresh database.",
            backup_path,
        )
        Base.metadata.create_all(bind=engine)
        logger.info("Fresh database created at %s", db_path)
        return

    # Not allowed to reset → build a helpful error message.
    lines = ["Database schema is out of date.  Missing columns:"]
    for table, cols in sorted(missing.items()):
        lines.append(f"  {table}: {', '.join(cols)}")
    lines.append("")
    lines.append("To fix, run the idempotent migration:")
    lines.append("  python -m backend.migrations.002_water_schema")
    lines.append("")
    lines.append("Or set ALLOW_DEV_DB_RESET=1 to auto-backup and recreate the DB.")
    raise RuntimeError("\n".join(lines))


def get_db():
    """Dependency that provides a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
