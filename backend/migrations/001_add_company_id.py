"""Migration: Add company_id column to raw_events and session_summary tables.

Run with: python -m backend.migrations.001_add_company_id
"""
import sqlite3
import sys
from pathlib import Path

# Database path at repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = REPO_ROOT / "telemetry.db"


def column_exists(cursor, table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    return column_name in columns


def index_exists(cursor, index_name: str) -> bool:
    """Check if an index exists."""
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
        (index_name,)
    )
    return cursor.fetchone() is not None


def migrate():
    """Run the migration."""
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        print("No migration needed - database will be created with new schema on first run.")
        return

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    try:
        # Add company_id to raw_events
        if not column_exists(cursor, "raw_events", "company_id"):
            print("Adding company_id column to raw_events...")
            cursor.execute("ALTER TABLE raw_events ADD COLUMN company_id TEXT")
            print("  Done.")
        else:
            print("Column company_id already exists in raw_events, skipping.")

        # Create index on raw_events.company_id
        if not index_exists(cursor, "ix_raw_events_company_id"):
            print("Creating index ix_raw_events_company_id...")
            cursor.execute(
                "CREATE INDEX ix_raw_events_company_id ON raw_events(company_id)"
            )
            print("  Done.")
        else:
            print("Index ix_raw_events_company_id already exists, skipping.")

        # Add company_id to session_summary
        if not column_exists(cursor, "session_summary", "company_id"):
            print("Adding company_id column to session_summary...")
            cursor.execute("ALTER TABLE session_summary ADD COLUMN company_id TEXT")
            print("  Done.")
        else:
            print("Column company_id already exists in session_summary, skipping.")

        # Create index on session_summary.company_id
        if not index_exists(cursor, "ix_session_summary_company_id"):
            print("Creating index ix_session_summary_company_id...")
            cursor.execute(
                "CREATE INDEX ix_session_summary_company_id ON session_summary(company_id)"
            )
            print("  Done.")
        else:
            print("Index ix_session_summary_company_id already exists, skipping.")

        conn.commit()
        print("\nMigration completed successfully!")

    except Exception as e:
        conn.rollback()
        print(f"Migration failed: {e}")
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
