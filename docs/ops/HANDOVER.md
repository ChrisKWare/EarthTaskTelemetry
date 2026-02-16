# Operational Handover - Earth Task Telemetry

## System Overview

Earth Task Telemetry is a backend + dashboard system that tracks player performance in the Earth Task game prototype. An Unreal Engine client sends per-attempt telemetry events to a FastAPI REST API, which stores them in SQLite, computes session-level cognitive metrics, trains a lightweight ML model per player, and exposes the data through two Streamlit dashboards: one per-player view and one company-aggregate view with privacy guards.

The core metrics are:

| Metric | Range | Meaning |
|--------|-------|---------|
| FTA Level | 0.0 - 1.0 | Fraction of sequences solved on first attempt |
| FTA Strict | bool | True only if all sequences solved first try |
| Repetition Burden | 1.0 - 3.0 | Average attempts needed per sequence |
| Earth Score Bucket | 0 / 5 / 10 | 0 = failed, 5 = solved with retries, 10 = perfect |

---

## Repository Structure

```
EarthTaskTelemetry/
  backend/
    app/
      main.py            # FastAPI app, all endpoints
      db.py              # SQLAlchemy engine + session factory
      models.py          # ORM models (RawEvent, SessionSummary, CompanyRegistry, ModelState)
      schemas.py         # Pydantic request/response schemas
      metrics.py         # Session metric computation
      settings.py        # Env var config (salts, thresholds, admin key)
      learner.py         # Per-player linear regression (scikit-learn)
      company.py         # Company hashing, token generation, token resolution
    migrations/
      001_add_company_id.py   # Adds company_id to existing tables
    tests/
      test_health.py
      test_ingest_and_finalize.py
      test_learner.py
      test_company_dashboard.py
    requirements.txt
  dashboard/
    app.py               # Streamlit player dashboard
    company_app.py       # Streamlit company dashboard
    requirements.txt
  shared/
    event_contract.md    # Payload schema reference
    sample_events.jsonl  # One sample session
  docs/
    RUNBOOK.md           # Local dev setup guide
    telemetry_contract.md  # Unreal <-> Backend integration spec
```

---

## Deployment (Render)

### Build command

```bash
pip install -r backend/requirements.txt
```

### Start command

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
```

Render sets `$PORT` automatically.

### Required environment variables

| Variable | Purpose |
|----------|---------|
| `COMPANY_SALT` | Salt for company_id hashing (must be random, not the dev default) |
| `DASHBOARD_TOKEN_SALT` | Salt for dashboard token generation (must be random, not the dev default) |
| `ADMIN_KEY` | Secret key for `POST /admin/seed_company` (empty = endpoint locked) |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins (e.g. `https://your-dashboard.streamlit.app`) |
| `MIN_COMPANY_N` | Minimum unique players before showing company aggregates (default: 5) |

### DATABASE_URL

The app currently builds its own SQLite path from the repo root:

```python
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATABASE_URL = f"sqlite:///{REPO_ROOT / 'telemetry.db'}"
```

On Render Free tier, the filesystem is **ephemeral** -- `telemetry.db` is wiped on every deploy and every daily restart. This is fine for demos but not for production.

### Switching to persistent disk on Render

1. Upgrade to Render **Starter** tier.
2. Add a Disk in the Render dashboard: mount path `/var/data`, minimum 1 GB.
3. Set the `DATABASE_URL` environment variable in the Render dashboard:

```
DATABASE_URL=sqlite:////var/data/telemetry.db
```

Do not edit `settings.py` in production -- the env var takes precedence because `settings.py` already reads from `os.getenv("DATABASE_URL", ...)`.

4. The app calls `Base.metadata.create_all()` on startup, so tables are created automatically on first boot.

### How to redeploy

Push to the branch Render is tracking (usually `master`). Render auto-deploys on push. For manual deploys, click "Manual Deploy" > "Deploy latest commit" in the Render dashboard.

---

## Database

### Current setup

SQLite via SQLAlchemy. The engine is created in `backend/app/db.py`:

```python
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
```

`check_same_thread=False` is required because FastAPI serves requests across threads but SQLite is single-writer by default. This is safe for a single-process deployment.

### Where telemetry.db lives

By default: `<repo_root>/telemetry.db`. On Render with a persistent disk: `/var/data/telemetry.db`.

### Free tier vs Starter tier

| | Free tier | Starter tier with disk |
|---|---|---|
| Filesystem | Ephemeral (wiped on deploy/restart) | Persistent mount at `/var/data` |
| Data survival | Lost every deploy | Survives deploys and restarts |
| Use case | Demos, testing | Production |

### Moving the DB to /var/data on Render

1. Attach a persistent disk mounted at `/var/data`.
2. Set the `DATABASE_URL` env var in the Render dashboard to `sqlite:////var/data/telemetry.db`.
3. Deploy. The app auto-creates tables on startup. No migration script needed for a fresh database.
4. If you have existing data to migrate, SSH into the Render shell and copy: `cp /opt/render/project/src/telemetry.db /var/data/telemetry.db`.

### Tables

| Table | Purpose |
|-------|---------|
| `raw_events` | Every individual attempt submitted by the game client |
| `session_summary` | Computed per-session metrics (created on finalize) |
| `company_registry` | Known companies with dashboard tokens |
| `model_state` | Per-player ML model coefficients and training metadata |

### Migration

`backend/migrations/001_add_company_id.py` adds `company_id` columns and indexes to `raw_events` and `session_summary`. It is idempotent. Run with:

```bash
python -m backend.migrations.001_add_company_id
```

Not needed if the database was created after the company feature was added (tables already include the column).

---

## Company Dashboard System

### How company_id is computed

1. The game client sends `company_name` (e.g. `"Acme Corp"`) or a pre-computed `company_id` on each event.
2. If only `company_name` is provided, the backend normalizes it (lowercase, strip punctuation, collapse whitespace) and hashes it:

```
company_id = "c_" + sha256(normalize(name) + COMPANY_SALT)[:12]
```

This is deterministic -- the same company name always produces the same ID regardless of casing or punctuation.

### How dashboard tokens work

A dashboard token grants read access to one company's aggregate data:

```
dashboard_token = sha256(company_id + DASHBOARD_TOKEN_SALT)[:24]
```

Tokens are 24 hex characters (96 bits). They are stored in the `company_registry` table and resolved via an indexed DB lookup in `resolve_token_to_company_id()`.

### Token resolution flow

1. Dashboard sends `GET /dashboard/company?t=<token>`.
2. Backend queries `CompanyRegistry` where `dashboard_token = token`.
3. If no match, returns 403.
4. If match, the resolved `company_id` scopes every subsequent query.

### How companies get registered

Two paths:

- **Admin seed**: `POST /admin/seed_company` (requires `X-ADMIN-KEY` header). Creates a `CompanyRegistry` row with the company name, computed ID, and dashboard token. Use this to provision a company before any gameplay data arrives.
- **Auto-registration on finalize**: When a session is finalized and the company has no `CompanyRegistry` row, one is created automatically. This ensures token resolution works even if the admin never explicitly seeded the company.

### Privacy guards

Company aggregate endpoints (`/dashboard/summary`, `/dashboard/timeseries`) require at least `MIN_COMPANY_N` unique players (default: 5) before returning data. Below this threshold they return 403. The info endpoint (`/dashboard/company`) always returns the player count and a status message so the dashboard can show a "not enough data yet" state.

### Session finalization and company_id

When `POST /sessions/{session_id}/finalize` is called:

1. All `RawEvent` rows for the session are fetched.
2. The `company_id` from the first event is extracted.
3. All events are validated to share the same `company_id` -- if not, returns 400.
4. The `company_id` is stored on the `SessionSummary`.
5. If no `CompanyRegistry` row exists for the company, one is auto-created.

---

## ML Model Layer

The learner module (`backend/app/learner.py`) trains a per-player linear regression model to predict next-session FTA level.

### How it works

- **Features**: Previous session's `[fta_level, repetition_burden]`.
- **Target**: Next session's `fta_level`.
- **Algorithm**: `sklearn.linear_model.LinearRegression`.
- **Minimum data**: 3 finalized sessions (which produces 2 training samples).
- **Trigger**: Called automatically after every session finalization.

### Model state

Stored in the `model_state` table:

| Field | Meaning |
|-------|---------|
| `status` | `"trained"` or `"insufficient_data"` |
| `n_samples` | Number of sessions used for training |
| `coefficients_json` | JSON array: `[coef_fta, coef_burden]` |
| `intercept` | Model intercept |
| `mae` | Mean absolute error on training data |

### Prediction

The player dashboard computes next-session FTA prediction client-side using the stored coefficients:

```
predicted_fta = intercept + (coef_fta * last_fta) + (coef_burden * last_burden)
```

Result is clamped to [0.0, 1.0].

---

## API Reference

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/health` | None | Returns `{"status": "ok"}` |
| POST | `/events` | None | Ingest an AttemptSubmitted event |
| POST | `/sessions/{session_id}/finalize` | None | Compute metrics, train model, store summary |
| GET | `/players/{player_id}/sessions` | None | List session summaries for a player |
| GET | `/model/state/{player_id}` | None | Get ML model state for a player |
| GET | `/dashboard/company?t=TOKEN` | Token | Company info + player count |
| GET | `/dashboard/summary?t=TOKEN` | Token | Company aggregate metrics (requires MIN_COMPANY_N) |
| GET | `/dashboard/timeseries?t=TOKEN&bucket=day\|week` | Token | Time-bucketed company metrics (requires MIN_COMPANY_N) |
| POST | `/admin/seed_company` | `X-ADMIN-KEY` header | Register company, returns dashboard token |

---

## Dashboards

### Player Dashboard (`dashboard/app.py`)

Run: `streamlit run dashboard/app.py`

Features: player lookup, session history table, FTA trend chart, repetition burden chart, earth score distribution, ML model state, next-session prediction, demo session generator.

Connects to the backend via `BACKEND_URL` env var or Streamlit secrets. Falls back to `http://127.0.0.1:8000`.

### Company Dashboard (`dashboard/company_app.py`)

Run: `streamlit run dashboard/company_app.py`

Access: requires `?t=TOKEN` query parameter. Shows company-level aggregates, time series charts, and session counts. Blocks display if fewer than `MIN_COMPANY_N` players.

---

## Testing

```bash
pytest backend/tests -v
```

45 tests across 4 files:

| File | Tests | Coverage |
|------|-------|----------|
| `test_health.py` | 1 | Health endpoint |
| `test_ingest_and_finalize.py` | 7 | Event ingestion, session finalization, metric computation |
| `test_learner.py` | 6 | ML model training lifecycle |
| `test_company_dashboard.py` | 31 | Company identity, dashboard endpoints, privacy guards, admin seed, company_id validation |

All tests use in-memory SQLite with fresh databases per test function.

---

## Salt Rotation Warning

**Do not change `COMPANY_SALT` or `DASHBOARD_TOKEN_SALT` on a running system unless you are prepared to invalidate all existing data relationships.**

Both salts are baked into stored identifiers:

- `COMPANY_SALT` is used to compute every `company_id`. Changing it means:
  - New events with the same `company_name` will produce a **different** `company_id`.
  - Existing `company_id` values in `raw_events`, `session_summary`, and `company_registry` will no longer match newly computed ones.
  - The company dashboard will effectively split into "old" and "new" data for the same company.

- `DASHBOARD_TOKEN_SALT` is used to compute every `dashboard_token`. Changing it means:
  - All previously issued dashboard tokens stop working immediately.
  - Every `CompanyRegistry` row has a now-stale `dashboard_token` value.
  - You must re-seed every company via `POST /admin/seed_company` to generate new tokens, and redistribute them to all company dashboard users.

If you must rotate salts (e.g. a salt was leaked), the recovery procedure is:

1. Set the new salt value in the Render env vars.
2. Delete all rows from `company_registry`.
3. Re-seed every company via `POST /admin/seed_company`.
4. Distribute new dashboard tokens to all company users.
5. For `COMPANY_SALT` rotation: existing `company_id` values in `raw_events` and `session_summary` are now orphaned. There is no automated way to re-hash them. You must either accept the data split or manually update the stored values.

---

## Ownership Transfer Procedure

Step-by-step instructions to transfer full ownership of the system to a new operator.

### 1. GitHub repository

- Transfer the repo via GitHub Settings > Danger Zone > Transfer ownership.
- Or add the new owner as a collaborator with Admin access, then remove yourself.
- Confirm the new owner can push to `master`.

### 2. Render service

- In the Render dashboard, go to the service Settings > Transfer Service and transfer to the new owner's Render account.
- Alternatively, the new owner creates a new Render Web Service pointing at the transferred GitHub repo.
- If creating fresh: set Build Command to `pip install -r backend/requirements.txt` and Start Command to `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`.

### 3. Environment variables

The new owner must set all required env vars in their Render dashboard. Provide the **names** only -- never commit values to the repo:

- `COMPANY_SALT`
- `DASHBOARD_TOKEN_SALT`
- `ADMIN_KEY`
- `ALLOWED_ORIGINS`
- `MIN_COMPANY_N`
- `DATABASE_URL` (if using persistent disk)

Transfer the actual secret values through a secure channel (e.g. password manager share, encrypted message). Do not send them via email or Slack.

### 4. Database file

If the current deployment uses a persistent disk with existing data:

- SSH into the current Render shell: `cp /var/data/telemetry.db /var/data/telemetry_backup.db`
- Download the backup via Render's shell or SCP.
- Upload it to the new owner's persistent disk at `/var/data/telemetry.db`.
- Verify the new deployment can read the data by hitting `GET /health` and a known dashboard token.

If starting fresh (no data to migrate), skip this step -- tables are auto-created on first boot.

### 5. Streamlit dashboards

If dashboards are deployed on Streamlit Community Cloud:

- Transfer the Streamlit app to the new owner's account, or have them redeploy from the transferred repo.
- Update the `BACKEND_URL` in Streamlit secrets to point to the new Render service URL.
- Update `ALLOWED_ORIGINS` on the backend to include the new Streamlit app URL.

### 6. Post-transfer verification

The new owner should confirm:

- [ ] `GET /health` returns `{"status": "ok"}`
- [ ] `pytest backend/tests -v` passes locally
- [ ] A known dashboard token resolves via `GET /dashboard/company?t=TOKEN`
- [ ] The Streamlit dashboards load and connect to the backend
- [ ] The previous owner's access has been revoked (GitHub, Render, Streamlit)

---

## Operational Checklist

### Before first production deploy

- [ ] Set `COMPANY_SALT` to a random secret (not the dev default)
- [ ] Set `DASHBOARD_TOKEN_SALT` to a different random secret
- [ ] Set `ADMIN_KEY` to a strong secret
- [ ] Set `ALLOWED_ORIGINS` to the exact dashboard origin(s)
- [ ] Attach a persistent disk on Render (Starter tier) and point `DATABASE_URL` to `/var/data/telemetry.db`
- [ ] Run `pytest backend/tests -v` and confirm all tests pass
- [ ] Seed companies via `POST /admin/seed_company` and distribute tokens

### Routine operations

- **View logs**: Render dashboard > your service > Logs
- **Reset database**: Delete `telemetry.db` (or `/var/data/telemetry.db`) and redeploy. Tables are auto-created on startup.
- **Add a new company**: `POST /admin/seed_company` with `X-ADMIN-KEY` header and `{"company_name": "..."}`. Returns the dashboard token to give to the company.
- **Change privacy threshold**: Update `MIN_COMPANY_N` env var and redeploy.
- **Run migration on existing DB**: `python -m backend.migrations.001_add_company_id` (idempotent, safe to re-run).
