"""Leakage-safe construction of the prediction target.

Design rule (see PROJECT_PLAN.md Section 16-17 and
docs/PHASE_1_NETWORK_EXPERIMENT.md): the live collector (experiment.py)
NEVER writes target_next_packet_loss_pct itself. It only ever writes
packet_loss_pct for the interval it just measured. This module adds the
target column afterwards, as an explicit forward shift, so it is
structurally impossible for the raw collection step to embed future
information -- the shift only happens here, once, in one place.
"""

from __future__ import annotations

from typing import Any

from network.schema import TARGET_COLUMN


def add_next_interval_target(
    rows: list[dict[str, Any]],
    horizon: int = 1,
    group_key: str = "experiment_id",
    order_key: str = "interval_index",
    source_column: str = "packet_loss_pct",
) -> list[dict[str, Any]]:
    """Return new rows with `target_next_packet_loss_pct` added.

    For each row at interval t within a group, the target is
    `source_column` measured at interval t + horizon in the SAME group.
    Rows for which t + horizon falls outside their group's observed
    intervals get target=None and should be dropped before training
    (there is no future data to predict against).

    Grouping by `group_key` and ordering by `order_key` is what prevents
    the shift from crossing experiment boundaries -- e.g. the last
    interval of experiment A must never pick up the first interval of
    experiment B as its "next" value.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1 (predicting the past is not the task)")

    groups: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row[group_key], []).append(row)

    result: list[dict[str, Any]] = []
    for group_rows in groups.values():
        ordered = sorted(group_rows, key=lambda r: r[order_key])
        indices = {row[order_key]: i for i, row in enumerate(ordered)}
        for row in ordered:
            new_row = dict(row)
            future_interval_index = row[order_key] + horizon
            future_pos = indices.get(future_interval_index)
            if future_pos is None:
                new_row[TARGET_COLUMN] = None
            else:
                new_row[TARGET_COLUMN] = ordered[future_pos][source_column]
            result.append(new_row)

    result.sort(key=lambda r: (r[group_key], r[order_key]))
    return result


def drop_rows_without_target(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove rows whose target could not be computed (end of each experiment)."""
    return [r for r in rows if r.get(TARGET_COLUMN) is not None]
