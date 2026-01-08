# Earth Task Telemetry

A telemetry and metrics system for tracking player performance in the Earth Task game prototype.

## Overview

This system:
- Ingests per-attempt telemetry events from Unreal via REST API
- Stores raw events in SQLite
- Computes session-level metrics (FTA, repetition burden, earth score)
- Displays trends in a Streamlit dashboard

## Quick Start

```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r backend\requirements.txt
pip install -r dashboard\requirements.txt

# 3. Start backend (Terminal 1)
uvicorn backend.app.main:app --reload --port 8000

# 4. Start dashboard (Terminal 2)
streamlit run dashboard/app.py
```

See [docs/RUNBOOK.md](docs/RUNBOOK.md) for detailed setup and usage instructions.

## Project Structure

```
EarthTaskTelemetry/
  backend/
    app/
      main.py         # FastAPI endpoints
      db.py           # Database connection
      models.py       # SQLAlchemy models
      schemas.py      # Pydantic schemas
      metrics.py      # Metrics computation
      settings.py     # Configuration
    tests/            # pytest tests
    requirements.txt
  dashboard/
    app.py            # Streamlit dashboard
    requirements.txt
  shared/
    event_contract.md # Event schema documentation
    sample_events.jsonl
  docs/
    RUNBOOK.md        # Setup and operations guide
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/events` | POST | Ingest AttemptSubmitted event |
| `/sessions/{session_id}/finalize` | POST | Compute session metrics |
| `/players/{player_id}/sessions` | GET | List player's session summaries |

## Metrics

- **FTA Level**: First-time accuracy rate (0.0 to 1.0)
- **FTA Strict**: True if all sequences solved on first attempt
- **Repetition Burden**: Average attempts used per sequence (1.0 to 3.0)
- **Earth Score Bucket**: 0 (failed), 5 (solved with retries), or 10 (perfect)

## License

Internal use only.
