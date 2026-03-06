"""Tests for application settings."""
import importlib

import pytest


def test_database_url_respects_env_var(monkeypatch):
    """DATABASE_URL uses the env var when set."""
    monkeypatch.setenv("DATABASE_URL", "sqlite:////var/data/telemetry.db")

    import backend.app.settings as settings
    importlib.reload(settings)

    assert settings.DATABASE_URL == "sqlite:////var/data/telemetry.db"


def test_database_url_defaults_to_local_sqlite(monkeypatch):
    """DATABASE_URL falls back to repo-root telemetry.db when env var is unset."""
    monkeypatch.delenv("DATABASE_URL", raising=False)

    import backend.app.settings as settings
    importlib.reload(settings)

    assert settings.DATABASE_URL.startswith("sqlite:///")
    assert settings.DATABASE_URL.endswith("telemetry.db")


# --- Required secrets: COMPANY_SALT, DASHBOARD_TOKEN_SALT, ADMIN_KEY ---

@pytest.mark.parametrize("var_name", [
    "COMPANY_SALT",
    "DASHBOARD_TOKEN_SALT",
    "ADMIN_KEY",
])
def test_missing_required_secret_raises(monkeypatch, var_name):
    """Application refuses to start if a required secret is missing."""
    monkeypatch.delenv(var_name, raising=False)

    import backend.app.settings as settings

    with pytest.raises(RuntimeError, match=f"{var_name} is not set"):
        importlib.reload(settings)

    # Restore so subsequent tests can reload safely
    monkeypatch.setenv(var_name, f"test-{var_name.lower()}")
    importlib.reload(settings)


@pytest.mark.parametrize("var_name", [
    "COMPANY_SALT",
    "DASHBOARD_TOKEN_SALT",
    "ADMIN_KEY",
])
def test_empty_required_secret_raises(monkeypatch, var_name):
    """Application refuses to start if a required secret is set but empty."""
    monkeypatch.setenv(var_name, "")

    import backend.app.settings as settings

    with pytest.raises(RuntimeError, match=f"{var_name} is not set"):
        importlib.reload(settings)

    # Restore so subsequent tests can reload safely
    monkeypatch.setenv(var_name, f"test-{var_name.lower()}")
    importlib.reload(settings)
