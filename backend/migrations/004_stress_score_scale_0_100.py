"""Migration: Rescale stress_score from 0-10 to 0-100 in session_summary.

Multiplies all existing stress_score values by 10.
Idempotent — safe to run multiple times (only updates rows where stress_score <= 10).

Run with: python -m backend.migrations.004_stress_score_scale_0_100
"""
import sqlite3
import sys
from pathlib import Path

# Database path at repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = REPO_ROOT / "telemetry.db"


def column_exists(cursor: sqlite3.Cursor, table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    return column_name in columns


def table_exists(cursor: sqlite3.Cursor, table_name: str) -> bool:
    """Check if a table exists."""
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cursor.fetchone() is not None


def migrate(db_path: Path | None = None):
    """Rescale stress_score values from 0-10 to 0-100.

    Only updates rows where stress_score is not null and <= 10,
    making this idempotent (already-scaled values > 10 are untouched).
    """
    path = db_path or DB_PATH

    if not path.exists():
        print(f"Database not found at {path}")
        print("No migration needed — database will be created with the new scale on first run.")
        return

    conn = sqlite3.connect(str(path))
    cursor = conn.cursor()

    try:
        if not table_exists(cursor, "session_summary"):
            print("Table session_summary does not exist — nothing to migrate.")
            return

        if not column_exists(cursor, "session_summary", "stress_score"):
            print("Column stress_score does not exist — nothing to migrate.")
            return

        # Count rows that need updating (stress_score between 0 and 10 inclusive)
        cursor.execute(
            "SELECT COUNT(*) FROM session_summary "
            "WHERE stress_score IS NOT NULL AND stress_score <= 10"
        )
        count = cursor.fetchone()[0]

        if count == 0:
            print("No rows with stress_score <= 10 found — already migrated or no data.")
            return

        print(f"Found {count} rows with stress_score <= 10. Multiplying by 10...")

        cursor.execute(
            "UPDATE session_summary "
            "SET stress_score = stress_score * 10 "
            "WHERE stress_score IS NOT NULL AND stress_score <= 10"
        )

        conn.commit()
        print(f"  Updated {cursor.rowcount} rows.")
        print("\nMigration 004_stress_score_scale_0_100 completed successfully!")

    except Exception as e:
        conn.rollback()
        print(f"Migration failed: {e}")
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
