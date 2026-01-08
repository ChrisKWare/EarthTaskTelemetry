# Event Contract

## AttemptSubmitted

Sent once per attempt when a player submits their answer for a sequence.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `player_id` | string (UUID) | Yes | Unique identifier for the player |
| `session_id` | string (UUID) | Yes | Unique identifier for the session |
| `task_version` | string | No | Task version identifier (default: "earth_v1") |
| `sequence_index` | int | Yes | Which sequence in the session (1, 2, 3) |
| `sequence_length` | int | Yes | Number of items in the sequence (4, 5, 6) |
| `attempt_index` | int | Yes | Which attempt for this sequence (1, 2, or 3) |
| `presented` | array | Yes | Gem IDs shown to player (int or string) |
| `input` | array | Yes | Gem IDs entered by player (int or string) |
| `is_correct` | bool | Yes | Whether the input matched presented |
| `duration_ms` | int | Yes | Time taken for this attempt in milliseconds |
| `ts_utc` | string | Yes | ISO-8601 timestamp of submission |
| `remediation_stage` | enum | Yes | One of: "none", "volts", "space" |
| `volts_count` | int \| null | No | Number of volts used (null if none) |

### Example

```json
{
  "player_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "session_id": "sess-001-2024-01-15",
  "task_version": "earth_v1",
  "sequence_index": 1,
  "sequence_length": 4,
  "attempt_index": 1,
  "presented": [1, 2, 3, 4],
  "input": [1, 2, 3, 4],
  "is_correct": true,
  "duration_ms": 4523,
  "ts_utc": "2024-01-15T14:30:00Z",
  "remediation_stage": "none",
  "volts_count": null
}
```

## Metrics Definitions

### Per-Sequence Metrics

| Metric | Definition |
|--------|------------|
| `solved` | Any attempt has `is_correct == true` within attempts 1-3 |
| `fta_seq` | 1 if attempt 1 is correct, else 0 |
| `attempts_used` | attempt_index of first correct attempt if solved, else 3 |

### Session-Level Metrics

| Metric | Definition |
|--------|------------|
| `fta_level` | Mean of `fta_seq` across sequences 1-3 (range: 0.0 to 1.0) |
| `fta_strict` | True if all three sequences have `fta_seq == 1` |
| `repetition_burden` | Mean of `attempts_used` across sequences 1-3 |
| `earth_score_bucket` | 0 if any sequence unsolved; 10 if all solved on attempt 1; else 5 |
