# Earth Task Telemetry - Runbook

## Prerequisites

- Python 3.11 or higher
- Windows (commands shown for PowerShell/cmd)

## Initial Setup

### 1. Create Virtual Environment

Open a terminal in the project root (`D:\Projects\EarthTaskTelemetry`):

```powershell
# Create venv
python -m venv .venv

# Activate venv (PowerShell)
.\.venv\Scripts\Activate.ps1

# OR for Command Prompt
.\.venv\Scripts\activate.bat
```

### 2. Install Dependencies

```powershell
# Install backend dependencies
pip install -r backend\requirements.txt

# Install dashboard dependencies
pip install -r dashboard\requirements.txt
```

## Running Services

### Start Backend (Terminal 1)

```powershell
# From project root, with venv activated
uvicorn backend.app.main:app --reload --port 8000
```

The API will be available at: http://localhost:8000

- Health check: http://localhost:8000/health
- API docs: http://localhost:8000/docs

### Start Dashboard (Terminal 2)

```powershell
# From project root, with venv activated
streamlit run dashboard/app.py
```

The dashboard will open at: http://localhost:8501

## Loading Sample Data

### Using Python Script

```powershell
# From project root, with venv activated
python -c "
import requests
import json

with open('shared/sample_events.jsonl') as f:
    for line in f:
        event = json.loads(line)
        resp = requests.post('http://localhost:8000/events', json=event)
        print(f'Stored event: seq={event[\"sequence_index\"]}, attempt={event[\"attempt_index\"]}')

# Finalize the session
resp = requests.post('http://localhost:8000/sessions/sess-001-2024-01-15/finalize')
print(f'Session finalized: {resp.json()}')
"
```

### Using curl (if available)

```powershell
# Post each event
curl -X POST http://localhost:8000/events -H "Content-Type: application/json" -d "{\"player_id\": \"a1b2c3d4-e5f6-7890-abcd-ef1234567890\", \"session_id\": \"sess-001-2024-01-15\", \"task_version\": \"earth_v1\", \"sequence_index\": 1, \"sequence_length\": 4, \"attempt_index\": 1, \"presented\": [1, 2, 3, 4], \"input\": [1, 2, 3, 4], \"is_correct\": true, \"duration_ms\": 4523, \"ts_utc\": \"2024-01-15T14:30:00Z\", \"remediation_stage\": \"none\", \"volts_count\": null}"

# Finalize session
curl -X POST http://localhost:8000/sessions/sess-001-2024-01-15/finalize
```

## Running Tests

```powershell
# From project root, with venv activated
pytest backend/tests -v
```

Expected output:
```
backend/tests/test_health.py::test_health_returns_ok PASSED
backend/tests/test_ingest_and_finalize.py::test_ingest_event_stores_successfully PASSED
backend/tests/test_ingest_and_finalize.py::test_finalize_all_first_attempt_correct PASSED
backend/tests/test_ingest_and_finalize.py::test_finalize_mixed_attempts PASSED
backend/tests/test_ingest_and_finalize.py::test_finalize_one_sequence_unsolved PASSED
backend/tests/test_ingest_and_finalize.py::test_get_player_sessions PASSED
backend/tests/test_ingest_and_finalize.py::test_finalize_no_events_returns_404 PASSED
backend/tests/test_ingest_and_finalize.py::test_finalize_idempotent PASSED
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check, returns `{"ok": true}` |
| POST | `/events` | Ingest an AttemptSubmitted event |
| POST | `/sessions/{session_id}/finalize` | Compute and store session summary |
| GET | `/players/{player_id}/sessions` | List session summaries for a player |

## Sample Data Expectations

The sample events in `shared/sample_events.jsonl` represent one session with:

- **Sequence 1**: Correct on attempt 1 (FTA)
- **Sequence 2**: Wrong on attempt 1, correct on attempt 2
- **Sequence 3**: Wrong on attempts 1 and 2, correct on attempt 3

Expected metrics after finalization:
- `fta_level`: 0.333... (1/3 sequences got FTA)
- `fta_strict`: false
- `repetition_burden`: 2.0 ((1 + 2 + 3) / 3)
- `earth_score_bucket`: 5 (all solved, but not all on first attempt)

## Database

The SQLite database is stored at the project root: `telemetry.db`

To reset the database, simply delete the file:
```powershell
Remove-Item telemetry.db
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'backend'"

Make sure you're running commands from the project root directory (`D:\Projects\EarthTaskTelemetry`).

### Port already in use

Kill existing processes or use different ports:
```powershell
# Backend on different port
uvicorn backend.app.main:app --reload --port 8001

# Update API_BASE_URL in dashboard/app.py accordingly
```

### Tests failing with database errors

Delete any stale test database files:
```powershell
Remove-Item test_telemetry.db -ErrorAction SilentlyContinue
```
