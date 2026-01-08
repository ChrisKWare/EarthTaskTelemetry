"""Application settings."""
from pathlib import Path

# Database file lives at repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATABASE_URL = f"sqlite:///{REPO_ROOT / 'telemetry.db'}"
