# Unreal Engine to Backend Telemetry Integration Contract

## 1. Purpose

This document defines the integration contract between the Unreal Engine game client and the Earth Task Telemetry backend. The telemetry system captures player attempt data during memory sequence tasks, enabling computation of performance metrics (FTA level, repetition burden, earth score bucket) and adaptive learning model training. Game clients send individual attempt events in real-time and finalize sessions when gameplay completes.

## 2. Base URL Rules

*Operational guidance (not enforced by backend)*

| Environment | Base URL | Notes |
|-------------|----------|-------|
| Local dev (same machine) | `http://127.0.0.1:8000` | Only works when Unreal and backend run on the same machine |
| Remote dev / LAN testing | `http://<host-ip>:8000` | Use the actual IP address of the backend host |
| Production | `https://your-domain.com` | Use your hosted deployment URL |

**Important:** `127.0.0.1` and `localhost` will **not** work when the Unreal client runs on a different machine than the backend. For multi-machine development or device testing, use the backend host's actual IP address or hostname.

## 3. Endpoints

### POST /events

Ingests a single `AttemptSubmitted` event from the game client.

- **Method:** `POST`
- **Path:** `/events`
- **Request Body:** JSON (see [Event Payload Schema](#4-event-payload-schema))
- **Response:** `{"stored": true}` on success (HTTP 200)

### POST /sessions/{session_id}/finalize

Computes and persists session summary metrics after all attempts are submitted. This endpoint is idempotent; calling it multiple times for the same session returns the same result.

- **Method:** `POST`
- **Path:** `/sessions/{session_id}/finalize`
- **Path Parameter:** `session_id` - The unique session identifier
- **Request Body:** None
- **Response:** Session summary object (HTTP 200), or HTTP 404 if no events exist for the session

### GET /health (Optional)

Health check endpoint to verify backend availability before sending telemetry.

- **Method:** `GET`
- **Path:** `/health`
- **Response:** `{"ok": true}` (HTTP 200)

## 4. Event Payload Schema

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `player_id` | `string` | Yes | - | Identifier for the player |
| `session_id` | `string` | Yes | - | Identifier for the gameplay session |
| `task_version` | `string` | No | Default: `"earth_v1"` | Version identifier for the task variant |
| `sequence_index` | `integer` | Yes | >= 1 | Which sequence in the session (1, 2, 3, ...) |
| `sequence_length` | `integer` | Yes | >= 1 | Number of elements in the presented sequence |
| `attempt_index` | `integer` | Yes | 1-3 | Which attempt for this sequence (max 3) |
| `presented` | `array` | Yes | Elements: `int` or `string` | The sequence shown to the player |
| `input` | `array` | Yes | Elements: `int` or `string` | The sequence entered by the player |
| `is_correct` | `boolean` | Yes | - | Whether the player's input matched the presented sequence |
| `duration_ms` | `integer` | Yes | >= 0 | Time taken for the attempt in milliseconds |
| `ts_utc` | `string` | Yes | Recommended ISO 8601 UTC timestamp | UTC timestamp when the attempt occurred |
| `remediation_stage` | `string` | Yes | One of: `"none"`, `"volts"`, `"space"` | Current remediation stage during the attempt |
| `volts_count` | `integer` | No | Nullable | Number of volts used (if applicable) |

## 5. Example Event JSON Payload

```json
{
  "player_id": "player-123",
  "session_id": "session-abc",
  "task_version": "earth_v1",
  "sequence_index": 1,
  "sequence_length": 4,
  "attempt_index": 1,
  "presented": [1, 2, 3, 4],
  "input": [1, 2, 3, 4],
  "is_correct": true,
  "duration_ms": 5000,
  "ts_utc": "2024-01-15T10:00:00Z",
  "remediation_stage": "none",
  "volts_count": null
}
```

### Example: Incorrect Attempt

```json
{
  "player_id": "player-123",
  "session_id": "session-abc",
  "task_version": "earth_v1",
  "sequence_index": 2,
  "sequence_length": 4,
  "attempt_index": 1,
  "presented": [1, 2, 3, 4],
  "input": [4, 3, 2, 1],
  "is_correct": false,
  "duration_ms": 3200,
  "ts_utc": "2024-01-15T10:01:30Z",
  "remediation_stage": "none",
  "volts_count": null
}
```
