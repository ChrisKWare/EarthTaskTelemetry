"""Shared pytest configuration for backend tests.

Sets required environment variables before any test module imports
backend.app.settings, which now raises RuntimeError if secrets are missing.
"""
import os

# Provide test-only secrets so settings.py can be imported safely.
# These are set at conftest import time (before collection), guaranteeing
# they exist before any module-level `from backend.app.settings import ...`.
os.environ.setdefault("COMPANY_SALT", "test-company-salt")
os.environ.setdefault("DASHBOARD_TOKEN_SALT", "test-dashboard-token-salt")
os.environ.setdefault("ADMIN_KEY", "test-admin-key")
