"""Tests for the leakage-safe target construction -- the most safety
critical piece of Phase 1. These specifically try to break the
group/order boundaries, since that's exactly where a leakage bug would
hide (PROJECT_PLAN.md Section 16-17).
"""

import pytest

from network.targets import add_next_interval_target, drop_rows_without_target


def _row(experiment_id, interval_index, loss):
    return {"experiment_id": experiment_id, "interval_index": interval_index, "packet_loss_pct": loss}


def test_target_is_next_interval_within_same_experiment():
    rows = [_row("e1", 0, 1.0), _row("e1", 1, 2.0), _row("e1", 2, 3.0)]
    shifted = add_next_interval_target(rows)
    by_index = {r["interval_index"]: r for r in shifted}
    assert by_index[0]["target_next_packet_loss_pct"] == 2.0
    assert by_index[1]["target_next_packet_loss_pct"] == 3.0


def test_last_row_of_each_experiment_has_no_target():
    rows = [_row("e1", 0, 1.0), _row("e1", 1, 2.0)]
    shifted = add_next_interval_target(rows)
    by_index = {r["interval_index"]: r for r in shifted}
    assert by_index[1]["target_next_packet_loss_pct"] is None


def test_target_never_crosses_experiment_boundary():
    """The core leakage test: e1's last row must NOT pick up e2's first row."""
    rows = [_row("e1", 0, 1.0), _row("e1", 1, 2.0), _row("e2", 0, 999.0), _row("e2", 1, 888.0)]
    shifted = add_next_interval_target(rows)
    e1_last = next(r for r in shifted if r["experiment_id"] == "e1" and r["interval_index"] == 1)
    assert e1_last["target_next_packet_loss_pct"] is None
    assert e1_last["target_next_packet_loss_pct"] != 999.0


def test_horizon_greater_than_one():
    rows = [_row("e1", 0, 1.0), _row("e1", 1, 2.0), _row("e1", 2, 3.0)]
    shifted = add_next_interval_target(rows, horizon=2)
    by_index = {r["interval_index"]: r for r in shifted}
    assert by_index[0]["target_next_packet_loss_pct"] == 3.0
    assert by_index[1]["target_next_packet_loss_pct"] is None  # 1+2=3 doesn't exist


def test_horizon_zero_rejected():
    with pytest.raises(ValueError):
        add_next_interval_target([_row("e1", 0, 1.0)], horizon=0)


def test_unordered_input_still_shifts_correctly():
    """Rows may arrive out of order (e.g. from a dict/CSV read); output must
    still reflect true temporal order, not input order."""
    rows = [_row("e1", 2, 3.0), _row("e1", 0, 1.0), _row("e1", 1, 2.0)]
    shifted = add_next_interval_target(rows)
    by_index = {r["interval_index"]: r for r in shifted}
    assert by_index[0]["target_next_packet_loss_pct"] == 2.0


def test_drop_rows_without_target_removes_only_unlabeled_rows():
    rows = [_row("e1", 0, 1.0), _row("e1", 1, 2.0)]
    shifted = add_next_interval_target(rows)
    kept = drop_rows_without_target(shifted)
    assert len(kept) == 1
    assert kept[0]["interval_index"] == 0


def test_missing_intermediate_interval_does_not_fabricate_target():
    """If interval 1 is missing (e.g. a dropped sample), interval 0's
    target must be None, not silently taken from interval 2."""
    rows = [_row("e1", 0, 1.0), _row("e1", 2, 3.0)]
    shifted = add_next_interval_target(rows)
    by_index = {r["interval_index"]: r for r in shifted}
    assert by_index[0]["target_next_packet_loss_pct"] is None
