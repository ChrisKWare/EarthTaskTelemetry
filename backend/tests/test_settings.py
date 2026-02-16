"""Tests for application settings."""
import importlib


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
