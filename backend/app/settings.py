"""Application settings."""
import os
from pathlib import Path

# Database file lives at repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATABASE_URL = f"sqlite:///{REPO_ROOT / 'telemetry.db'}"

# Company identity hashing salts (MUST be changed in production)
COMPANY_SALT = os.getenv("COMPANY_SALT", "dev-company-salt-change-in-prod")
DASHBOARD_TOKEN_SALT = os.getenv("DASHBOARD_TOKEN_SALT", "dev-dashboard-token-salt-change-in-prod")

# Minimum unique players before showing company aggregates (anonymity threshold)
MIN_COMPANY_N = int(os.getenv("MIN_COMPANY_N", "5"))

# Admin API key for privileged endpoints (must be set in production)
ADMIN_KEY = os.getenv("ADMIN_KEY", "")
