"""Metrics computation for session summaries."""
from typing import List, Dict, Any
from collections import defaultdict

from .models import RawEvent


def compute_session_metrics(events: List[RawEvent]) -> Dict[str, Any]:
    """
    Compute session-level metrics from raw events.

    Per sequence:
    - solved = any attempt is_correct == true within attempts 1..3
    - fta_seq = 1 if attempt 1 is correct else 0
    - attempts_used = attempt_index of first correct attempt if solved else 3

    Session:
    - fta_level = mean(fta_seq for sequences 1..3)
    - fta_strict = 1 if all three sequences have fta_seq==1 else 0
    - repetition_burden = mean(attempts_used across sequences 1..3)
    - earth_score_bucket:
        - if any sequence not solved -> 0
        - else if all sequences solved on attempt 1 -> 10
        - else -> 5
    """
    if not events:
        return {
            "fta_level": 0.0,
            "fta_strict": False,
            "repetition_burden": 3.0,
            "earth_score_bucket": 0,
        }

    # Group events by sequence_index
    sequences: Dict[int, List[RawEvent]] = defaultdict(list)
    for event in events:
        sequences[event.sequence_index].append(event)

    # Compute per-sequence metrics
    sequence_metrics = {}
    for seq_idx, seq_events in sequences.items():
        # Sort by attempt_index
        seq_events.sort(key=lambda e: e.attempt_index)

        # Find first correct attempt
        first_correct_attempt = None
        for event in seq_events:
            if event.is_correct:
                first_correct_attempt = event.attempt_index
                break

        solved = first_correct_attempt is not None
        fta_seq = 1 if (first_correct_attempt == 1) else 0
        attempts_used = first_correct_attempt if solved else 3

        sequence_metrics[seq_idx] = {
            "solved": solved,
            "fta_seq": fta_seq,
            "attempts_used": attempts_used,
        }

    # Compute session-level metrics (assume sequences 1..3)
    expected_sequences = [1, 2, 3]

    fta_values = []
    attempts_values = []
    all_solved = True

    for seq_idx in expected_sequences:
        if seq_idx in sequence_metrics:
            metrics = sequence_metrics[seq_idx]
            fta_values.append(metrics["fta_seq"])
            attempts_values.append(metrics["attempts_used"])
            if not metrics["solved"]:
                all_solved = False
        else:
            # Missing sequence treated as not solved
            fta_values.append(0)
            attempts_values.append(3)
            all_solved = False

    fta_level = sum(fta_values) / len(fta_values) if fta_values else 0.0
    fta_strict = all(fta == 1 for fta in fta_values)
    repetition_burden = sum(attempts_values) / len(attempts_values) if attempts_values else 3.0

    # Earth score bucket
    if not all_solved:
        earth_score_bucket = 0
    elif all(att == 1 for att in attempts_values):
        earth_score_bucket = 10
    else:
        earth_score_bucket = 5

    return {
        "fta_level": fta_level,
        "fta_strict": fta_strict,
        "repetition_burden": repetition_burden,
        "earth_score_bucket": earth_score_bucket,
    }
