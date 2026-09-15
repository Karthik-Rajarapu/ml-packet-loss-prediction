"""Phase 2 dataset-generation orchestration.

Reuses Phase 1's run_experiment() unchanged (via dependency injection,
so tests can substitute a fake without touching Mininet) to run every
ExperimentConfig produced by sweep.expand_configs(), and records exactly
what happened in a manifest. Never invents a row: a failed experiment is
recorded as failed with its error message, not silently dropped, and
row counts come from actually reading back the CSV Phase 1 wrote.
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
from typing import Any, Callable

from network.config import ExperimentConfig
from network.experiment import run_experiment as _default_run_experiment
from network.manifest import ExperimentResult, build_manifest, write_manifest
from network.sweep import SweepConfig, expand_configs
from network.validation import require_environment

RunExperimentFn = Callable[[ExperimentConfig, Path], Path]


def describe_sweep(sweep: SweepConfig, output_dir: Path) -> dict[str, Any]:
    """Pure, side-effect-free summary of what a sweep WOULD do. Used by
    both --dry-run's printed output and its tests -- never touches the
    filesystem or the network.
    """
    configs = expand_configs(sweep)
    estimated_seconds = len(configs) * sweep.duration_s
    return {
        "sweep_id": sweep.sweep_id,
        "n_combinations": sweep.n_combinations,
        "repetitions": sweep.repetitions,
        "n_experiments": len(configs),
        "estimated_duration_s": estimated_seconds,
        "estimated_duration_human": f"{estimated_seconds / 60:.1f} min",
        "output_dir": str(output_dir),
        "experiment_ids": [c.experiment_id for c in configs],
        "parameters": {
            "bottleneck_bw_mbps_values": sweep.bottleneck_bw_mbps_values,
            "bottleneck_delay_ms_values": sweep.bottleneck_delay_ms_values,
            "offered_load_factors": sweep.offered_load_factors,
            "n_flows_values": sweep.n_flows_values,
            "traffic_type": sweep.traffic_type,
            "sample_interval_s": sweep.sample_interval_s,
            "duration_s": sweep.duration_s,
        },
    }


def _count_rows(csv_path: Path) -> int:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def run_sweep(
    sweep: SweepConfig,
    output_dir: Path,
    run_experiment_fn: RunExperimentFn = _default_run_experiment,
    stop_on_first_failure: bool = False,
) -> dict[str, Any]:
    """Run every experiment in `sweep` and return the manifest dict.

    Does NOT check tool availability itself when `run_experiment_fn` is
    the real Phase 1 run_experiment -- that function already calls
    require_environment() per experiment. Callers that want a single
    up-front check (e.g. the CLI, so failure is immediate rather than
    after experiment 1 of N) should call require_environment() before
    this, which scripts/generate_dataset.py does.

    A failed experiment is recorded, loudly printed, and the sweep
    CONTINUES to the next combination by default (stop_on_first_failure
    controls this) -- "don't silently skip" is satisfied by recording
    and reporting every failure in the manifest, not by aborting.
    """
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    configs = expand_configs(sweep)
    results: list[ExperimentResult] = []

    for config in configs:
        try:
            csv_path = run_experiment_fn(config, output_dir)
            row_count = _count_rows(csv_path)
            results.append(ExperimentResult(
                experiment_id=config.experiment_id, status="success",
                csv_path=str(csv_path), row_count=row_count,
            ))
            print(f"[OK]     {config.experiment_id} -- {row_count} rows -> {csv_path}")
        except Exception as exc:  # noqa: BLE001 -- must record every failure, not propagate and abort by default
            results.append(ExperimentResult(
                experiment_id=config.experiment_id, status="failed", error=str(exc),
            ))
            print(f"[FAILED] {config.experiment_id} -- {exc}")
            if stop_on_first_failure:
                break

    finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest = build_manifest(sweep, results, started_at, finished_at, output_dir)
    return manifest


def run_sweep_and_write_manifest(
    sweep: SweepConfig,
    output_dir: Path,
    manifest_dir: Path,
    run_experiment_fn: RunExperimentFn = _default_run_experiment,
) -> tuple[dict[str, Any], Path]:
    manifest = run_sweep(sweep, output_dir, run_experiment_fn)
    manifest_path = manifest_dir / f"{sweep.sweep_id}_manifest.json"
    write_manifest(manifest, manifest_path)
    return manifest, manifest_path
