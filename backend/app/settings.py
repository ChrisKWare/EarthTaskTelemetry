"""Application settings."""
import logging
import os
from pathlib import Path

# Database file lives at repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{REPO_ROOT / 'telemetry.db'}")


def _require_env(name: str) -> str:
    """Return the value of an environment variable or raise RuntimeError."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Required environment variable {name} is not set. "
            f"Set {name} before starting the application."
        )
    return value


# Required secrets — the application refuses to start without these
COMPANY_SALT: str = _require_env("COMPANY_SALT")
DASHBOARD_TOKEN_SALT: str = _require_env("DASHBOARD_TOKEN_SALT")
ADMIN_KEY: str = _require_env("ADMIN_KEY")

# Minimum unique players before showing company aggregates (anonymity threshold)
MIN_COMPANY_N = int(os.getenv("MIN_COMPANY_N", "1"))

logger = logging.getLogger(__name__)
logger.info(
    "ADMIN_KEY loaded: set=%s len=%s",
    bool(ADMIN_KEY),
    len(ADMIN_KEY) if ADMIN_KEY else 0,
)
