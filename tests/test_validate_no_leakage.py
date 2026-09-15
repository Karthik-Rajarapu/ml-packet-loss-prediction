"""Independent verification that a dataset's target column is genuinely
leakage-safe -- separate from (and a cross-check on) targets.py's own
construction logic. Deliberately tries to construct datasets that WOULD
leak if validate_no_leakage() had a hole in it.
"""

import pytest

from network.targets import add_next_interval_target
from network.validation import SchemaValidationError, validate_no_leakage


def _row(experiment_id, interval_index, loss):
    return {"experiment_id": experiment_id, "interval_index": interval_index, "packet_loss_pct": loss}


def test_accepts_correctly_shifted_target():
    rows = [_row("e1", 0, 1.0), _row("e1", 1, 2.0), _row("e1", 2, 3.0)]
    shifted = add_next_interval_target(rows)
    validate_no_leakage(shifted)  # should not raise


def test_rejects_target_taken_from_wrong_row():
    rows = [_row("e1", 0, 1.0), _row("e1", 1, 2.0), _row("e1", 2, 3.0)]
    shifted = add_next_interval_target(rows)
    # Corrupt row 0's target so it no longer matches row 1's packet_loss_pct.
    for row in shifted:
        if row["interval_index"] == 0:
            row["target_next_packet_loss_pct"] = 999.0
    with pytest.raises(SchemaValidationError):
        validate_no_leakage(shifted)


def test_rejects_target_leaked_from_a_different_experiment():
    """The scenario the whole check exists to catch: a bug that let
    experiment A's last row pick up experiment B's first value."""
    rows = [_row("e1", 0, 1.0), _row("e1", 1, 2.0), _row("e2", 0, 999.0)]
    shifted = add_next_interval_target(rows)
    for row in shifted:
        if row["experiment_id"] == "e1" and row["interval_index"] == 1:
            row["target_next_packet_loss_pct"] = 999.0  # simulate the leak
    with pytest.raises(SchemaValidationError):
        validate_no_leakage(shifted)


def test_rejects_nonnull_target_on_last_row_of_experiment():
    rows = [_row("e1", 0, 1.0), _row("e1", 1, 2.0)]
    shifted = add_next_interval_target(rows)
    for row in shifted:
        if row["interval_index"] == 1:
            row["target_next_packet_loss_pct"] = 5.0  # should be None -- no future data exists
    with pytest.raises(SchemaValidationError):
        validate_no_leakage(shifted)


def test_rejects_null_target_when_future_data_exists():
    rows = [_row("e1", 0, 1.0), _row("e1", 1, 2.0)]
    shifted = add_next_interval_target(rows)
    for row in shifted:
        if row["interval_index"] == 0:
            row["target_next_packet_loss_pct"] = None  # future data exists -- should not be null
    with pytest.raises(SchemaValidationError):
        validate_no_leakage(shifted)


def test_accepts_horizon_greater_than_one():
    rows = [_row("e1", 0, 1.0), _row("e1", 1, 2.0), _row("e1", 2, 3.0)]
    shifted = add_next_interval_target(rows, horizon=2)
    validate_no_leakage(shifted, horizon=2)  # should not raise


def test_multi_experiment_dataset_passes_when_correct():
    rows = [
        _row("e1", 0, 1.0), _row("e1", 1, 2.0), _row("e1", 2, 3.0),
        _row("e2", 0, 10.0), _row("e2", 1, 20.0),
    ]
    shifted = add_next_interval_target(rows)
    validate_no_leakage(shifted)  # should not raise
