"""Application settings."""
import logging
import os
from pathlib import Path

# Database file lives at repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{REPO_ROOT / 'telemetry.db'}")

# Company identity hashing salts (MUST be changed in production)
COMPANY_SALT = os.getenv("COMPANY_SALT", "dev-company-salt-change-in-prod")
DASHBOARD_TOKEN_SALT = os.getenv("DASHBOARD_TOKEN_SALT", "dev-dashboard-token-salt-change-in-prod")

# Minimum unique players before showing company aggregates (anonymity threshold)
MIN_COMPANY_N = int(os.getenv("MIN_COMPANY_N", "5"))

# Admin API key for privileged endpoints (must be set in production)
ADMIN_KEY = os.getenv("ADMIN_KEY", "")

logger = logging.getLogger(__name__)
logger.info(
    "ADMIN_KEY loaded: set=%s len=%s",
    bool(ADMIN_KEY),
    len(ADMIN_KEY) if ADMIN_KEY else 0,
)
