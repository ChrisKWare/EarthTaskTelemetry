"""Migration: Rename calmness_score column to stress_score in session_summary.

Idempotent — safe to run multiple times.

Run with: python -m backend.migrations.003_rename_calmness_to_stress
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
    """Rename calmness_score -> stress_score in session_summary.

    SQLite does not support ALTER TABLE RENAME COLUMN until 3.25.0, so this
    migration uses the table-rebuild strategy for maximum compatibility:

    1. Create a new table with the correct schema.
    2. Copy all data from the old table.
    3. Drop the old table.
    4. Rename the new table.
    5. Recreate indexes.
    """
    path = db_path or DB_PATH

    if not path.exists():
        print(f"Database not found at {path}")
        print("No migration needed — database will be created with the new schema on first run.")
        return

    conn = sqlite3.connect(str(path))
    cursor = conn.cursor()

    try:
        if not table_exists(cursor, "session_summary"):
            print("Table session_summary does not exist — will be created on app startup.")
            return

        has_calmness = column_exists(cursor, "session_summary", "calmness_score")
        has_stress = column_exists(cursor, "session_summary", "stress_score")

        if has_stress and not has_calmness:
            print("Column stress_score already exists in session_summary, skipping.")
            return

        if not has_calmness and not has_stress:
            # Neither column exists — the table predates water support.
            # Just add the new column name directly.
            print("Adding stress_score column to session_summary...")
            cursor.execute(
                "ALTER TABLE session_summary ADD COLUMN stress_score REAL"
            )
            conn.commit()
            print("  Done.")
            print("\nMigration 003_rename_calmness_to_stress completed successfully!")
            return

        if has_calmness and has_stress:
            # Both columns exist (e.g. migration 002 re-added calmness_score
            # after 003 already renamed it). Drop calmness_score via rebuild.
            print("Both calmness_score and stress_score exist. Dropping calmness_score...")
        else:
            # has_calmness only — need to rename via table rebuild
            print("Renaming calmness_score -> stress_score in session_summary...")

        # Get current column info to preserve the schema
        cursor.execute("PRAGMA table_info(session_summary)")
        columns_info = cursor.fetchall()
        old_col_names = [row[1] for row in columns_info]

        # Build new column list: rename calmness_score -> stress_score,
        # and skip any duplicate calmness_score if stress_score already exists
        new_col_names = []
        for c in old_col_names:
            if c == "calmness_score":
                if has_stress:
                    # stress_score already exists, drop the calmness_score column
                    continue
                else:
                    new_col_names.append("stress_score")
            else:
                new_col_names.append(c)

        # Build column definitions for the new table (skip calmness_score if dropping)
        col_defs = []
        for row in columns_info:
            cid, name, col_type, notnull, default, pk = row
            if name == "calmness_score" and has_stress:
                continue  # skip — stress_score already covers this
            new_name = "stress_score" if name == "calmness_score" else name

            parts = [new_name, col_type or ""]
            if pk:
                parts.append("PRIMARY KEY")
            if notnull and not pk:
                parts.append("NOT NULL")
            if default is not None:
                parts.append(f"DEFAULT {default}")
            col_defs.append(" ".join(parts))

        # Also filter old_col_names to match (for the SELECT)
        select_col_names = [c for c in old_col_names if not (c == "calmness_score" and has_stress)]

        # Check for unique constraints on session_id
        create_sql = f"CREATE TABLE session_summary_new ({', '.join(col_defs)})"
        cursor.execute(create_sql)

        # Copy data
        select_cols_str = ", ".join(select_col_names)
        new_cols_str = ", ".join(new_col_names)
        cursor.execute(
            f"INSERT INTO session_summary_new ({new_cols_str}) "
            f"SELECT {select_cols_str} FROM session_summary"
        )

        # Drop old table
        cursor.execute("DROP TABLE session_summary")

        # Rename new table
        cursor.execute("ALTER TABLE session_summary_new RENAME TO session_summary")

        # Recreate indexes
        cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_session_summary_session_id "
            "ON session_summary (session_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS ix_session_summary_player_id "
            "ON session_summary (player_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS ix_session_summary_company_id "
            "ON session_summary (company_id)"
        )

        conn.commit()
        print("  Done.")
        print("\nMigration 003_rename_calmness_to_stress completed successfully!")

    except Exception as e:
        conn.rollback()
        print(f"Migration failed: {e}")
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
