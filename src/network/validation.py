"""Preflight and output validation.

Two separate concerns kept in one small module:
  1. check_environment() -- do the required tools exist BEFORE we try to
     run anything, so we fail with a clear message instead of a half-run
     experiment.
  2. validate_rows() -- sanity-check the CSV rows we actually produced,
     so a parsing bug can't silently write nonsense data.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from typing import Any


REQUIRED_TOOLS = ["mn", "iperf3", "ping", "tc", "ovs-vsctl"]


@dataclass
class EnvironmentCheck:
    available: dict[str, bool] = field(default_factory=dict)

    @property
    def all_available(self) -> bool:
        return all(self.available.values())

    @property
    def missing(self) -> list[str]:
        return [tool for tool, ok in self.available.items() if not ok]


def check_environment(tools: list[str] | None = None) -> EnvironmentCheck:
    """Check which required CLI tools are on PATH. Does not run anything."""
    tools = tools if tools is not None else REQUIRED_TOOLS
    return EnvironmentCheck(available={tool: shutil.which(tool) is not None for tool in tools})


def require_environment(tools: list[str] | None = None) -> None:
    """Raise a clear error listing exactly what's missing, or return silently."""
    check = check_environment(tools)
    if not check.all_available:
        raise RuntimeError(
            "Cannot run a live network experiment -- missing required tool(s): "
            f"{', '.join(check.missing)}. These must be installed inside a Linux "
            "environment (WSL2 Ubuntu); see docs/ENVIRONMENT_SETUP.md. "
            "Refusing to fabricate results -- no CSV will be written."
        )


class SchemaValidationError(ValueError):
    pass


def validate_rows(rows: list[dict[str, Any]], required_columns: list[str]) -> None:
    """Sanity-check collected rows before they're written/used.

    Checks (raises SchemaValidationError on the first violation found):
      - required columns are present on every row
      - timestamps are non-decreasing within each experiment_id
      - interval_index is unique and starts at 0 within each experiment_id
      - packets_received <= packets_sent
      - packet_loss_pct is within [0, 100]
    """
    if not rows:
        raise SchemaValidationError("No rows to validate -- experiment produced zero measurements")

    for i, row in enumerate(rows):
        missing = [c for c in required_columns if c not in row]
        if missing:
            raise SchemaValidationError(f"Row {i} missing required column(s): {missing}")

    by_experiment: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        by_experiment.setdefault(row["experiment_id"], []).append(row)

    for experiment_id, exp_rows in by_experiment.items():
        ordered = sorted(exp_rows, key=lambda r: r["interval_index"])

        indices = [r["interval_index"] for r in ordered]
        if indices != list(range(len(indices))):
            raise SchemaValidationError(
                f"experiment_id={experiment_id}: interval_index must be a contiguous "
                f"0-based sequence, got {indices}"
            )

        timestamps = [r["timestamp"] for r in ordered]
        if any(later < earlier for earlier, later in zip(timestamps, timestamps[1:])):
            raise SchemaValidationError(f"experiment_id={experiment_id}: timestamps are not non-decreasing")

        for row in ordered:
            sent = row.get("packets_sent")
            received = row.get("packets_received")
            if sent is not None and received is not None and received > sent:
                raise SchemaValidationError(
                    f"experiment_id={experiment_id} interval_index={row['interval_index']}: "
                    f"packets_received ({received}) > packets_sent ({sent})"
                )
            loss = row.get("packet_loss_pct")
            if loss is not None and not (0.0 <= loss <= 100.0):
                raise SchemaValidationError(
                    f"experiment_id={experiment_id} interval_index={row['interval_index']}: "
                    f"packet_loss_pct out of [0, 100]: {loss}"
                )


def validate_no_leakage(
    rows: list[dict[str, Any]],
    horizon: int = 1,
    target_column: str = "target_next_packet_loss_pct",
    source_column: str = "packet_loss_pct",
) -> None:
    """Verify a dataset that already has its target column is genuinely
    leakage-safe -- an independent check of the *output*, separate from
    trusting that targets.add_next_interval_target() was used correctly
    upstream. Intended to run on the final per-experiment or combined
    CSV before it's treated as ready for use.

    Checks (raises SchemaValidationError on the first violation found):
      - every non-null target actually equals `source_column` measured
        `horizon` intervals later IN THE SAME experiment_id (catches both
        a wrong-row bug and a leakage-across-experiments bug in one check)
      - the last `horizon` row(s) of every experiment_id have a null target
        (there is no future interval for them -- a non-null value there
        would mean something invented a target from outside the group)
      - no target value is reachable from a *different* experiment_id's
        source_column at the same offset (explicit cross-experiment guard)
    """
    if not rows:
        raise SchemaValidationError("No rows to validate")

    by_experiment: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        by_experiment.setdefault(row["experiment_id"], []).append(row)

    for experiment_id, exp_rows in by_experiment.items():
        ordered = sorted(exp_rows, key=lambda r: r["interval_index"])
        by_index = {r["interval_index"]: r for r in ordered}
        max_index = max(by_index)

        for row in ordered:
            target = row.get(target_column)
            future_index = row["interval_index"] + horizon

            if future_index > max_index:
                if target is not None:
                    raise SchemaValidationError(
                        f"experiment_id={experiment_id} interval_index={row['interval_index']}: "
                        f"target is non-null ({target}) but interval {future_index} does not exist "
                        f"in this experiment -- target must be null at the end of each run"
                    )
                continue

            future_row = by_index.get(future_index)
            if future_row is None:
                if target is not None:
                    raise SchemaValidationError(
                        f"experiment_id={experiment_id} interval_index={row['interval_index']}: "
                        f"target is non-null but interval {future_index} is missing (gap in data)"
                    )
                continue

            expected = future_row.get(source_column)
            if target is None:
                raise SchemaValidationError(
                    f"experiment_id={experiment_id} interval_index={row['interval_index']}: "
                    f"target is null but interval {future_index} exists with {source_column}={expected} "
                    f"-- target should have been populated"
                )
            if expected is not None and abs(target - expected) > 1e-9:
                raise SchemaValidationError(
                    f"experiment_id={experiment_id} interval_index={row['interval_index']}: "
                    f"target ({target}) does not match {source_column} at interval {future_index} "
                    f"({expected}) within the SAME experiment -- possible leakage or misalignment bug"
                )

    # Explicit cross-experiment guard: no row's target may equal a
    # DIFFERENT experiment's source value at the matching offset unless
    # that's a coincidental numeric match already ruled out above by
    # requiring equality with the SAME experiment's value.
    all_source_by_key: dict[tuple[Any, int], float] = {
        (row["experiment_id"], row["interval_index"]): row.get(source_column) for row in rows
    }
    for row in rows:
        target = row.get(target_column)
        if target is None:
            continue
        same_exp_key = (row["experiment_id"], row["interval_index"] + horizon)
        if all_source_by_key.get(same_exp_key) != target:
            raise SchemaValidationError(
                f"experiment_id={row['experiment_id']} interval_index={row['interval_index']}: "
                f"target does not trace back to this experiment's own future interval -- "
                f"possible cross-experiment leakage"
            )
