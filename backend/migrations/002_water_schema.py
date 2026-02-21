"""Migration: Add water_v1 support columns and model_name to model_state.

Idempotent — safe to run multiple times.

Run with: python -m backend.migrations.002_water_schema
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


def index_exists(cursor: sqlite3.Cursor, index_name: str) -> bool:
    """Check if an index exists."""
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
        (index_name,),
    )
    return cursor.fetchone() is not None


def migrate(db_path: Path | None = None):
    """Run the migration against the given DB file (defaults to DB_PATH)."""
    path = db_path or DB_PATH

    if not path.exists():
        print(f"Database not found at {path}")
        print("No migration needed — database will be created with the new schema on first run.")
        return

    conn = sqlite3.connect(str(path))
    cursor = conn.cursor()

    try:
        # --- model_state ---
        if table_exists(cursor, "model_state"):
            if not column_exists(cursor, "model_state", "model_name"):
                print("Adding model_name column to model_state (default='earth')...")
                cursor.execute(
                    "ALTER TABLE model_state ADD COLUMN model_name TEXT NOT NULL DEFAULT 'earth'"
                )
                print("  Done.")
            else:
                print("Column model_name already exists in model_state, skipping.")

            # Create unique index on (player_id, model_name)
            if not index_exists(cursor, "uq_model_state_player_model"):
                print("Creating unique index uq_model_state_player_model...")
                cursor.execute(
                    "CREATE UNIQUE INDEX uq_model_state_player_model "
                    "ON model_state (player_id, model_name)"
                )
                print("  Done.")
            else:
                print("Index uq_model_state_player_model already exists, skipping.")

            # Create regular index on model_name for filtering
            if not index_exists(cursor, "ix_model_state_model_name"):
                print("Creating index ix_model_state_model_name...")
                cursor.execute(
                    "CREATE INDEX ix_model_state_model_name ON model_state (model_name)"
                )
                print("  Done.")
            else:
                print("Index ix_model_state_model_name already exists, skipping.")
        else:
            print("Table model_state does not exist — will be created on app startup.")

        # --- session_summary ---
        if table_exists(cursor, "session_summary"):
            if not column_exists(cursor, "session_summary", "calmness_score"):
                print("Adding calmness_score column to session_summary...")
                cursor.execute(
                    "ALTER TABLE session_summary ADD COLUMN calmness_score REAL"
                )
                print("  Done.")
            else:
                print("Column calmness_score already exists in session_summary, skipping.")

            if not column_exists(cursor, "session_summary", "company_id"):
                print("Adding company_id column to session_summary...")
                cursor.execute(
                    "ALTER TABLE session_summary ADD COLUMN company_id TEXT"
                )
                print("  Done.")
            else:
                print("Column company_id already exists in session_summary, skipping.")

            if not index_exists(cursor, "ix_session_summary_company_id"):
                print("Creating index ix_session_summary_company_id...")
                cursor.execute(
                    "CREATE INDEX ix_session_summary_company_id "
                    "ON session_summary (company_id)"
                )
                print("  Done.")
            else:
                print("Index ix_session_summary_company_id already exists, skipping.")
        else:
            print("Table session_summary does not exist — will be created on app startup.")

        # --- company_registry ---
        if table_exists(cursor, "company_registry"):
            if not column_exists(cursor, "company_registry", "dashboard_token"):
                print("Adding dashboard_token column to company_registry...")
                cursor.execute(
                    "ALTER TABLE company_registry ADD COLUMN dashboard_token TEXT"
                )
                print("  Done.")
            else:
                print("Column dashboard_token already exists in company_registry, skipping.")
        else:
            print("Table company_registry does not exist — will be created on app startup.")

        conn.commit()
        print("\nMigration 002_water_schema completed successfully!")

    except Exception as e:
        conn.rollback()
        print(f"Migration failed: {e}")
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
