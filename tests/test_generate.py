"""Orchestration tests for Phase 2's dataset generator.

None of these touch Mininet: run_experiment is replaced with a fake via
dependency injection (network.generate.run_sweep's run_experiment_fn
parameter), exactly as the Phase 2 spec requires ("use mocks/fakes only
for testing orchestration behavior... do not create fake network
measurements and call them real data" -- these fakes are clearly test
fixtures for orchestration logic, never written to data/raw/ as if real).
"""

import csv
from pathlib import Path

from network.config import ExperimentConfig
from network.generate import describe_sweep, run_sweep
from network.schema import CSV_COLUMNS
from network.sweep import SweepConfig, expand_configs


def _fake_row(experiment_id: str, interval_index: int) -> dict:
    return {col: "" for col in CSV_COLUMNS} | {
        "experiment_id": experiment_id,
        "run_id": experiment_id,
        "interval_index": interval_index,
        "timestamp": 1000.0 + interval_index,
        "packet_loss_pct": 1.0,
        "target_next_packet_loss_pct": 2.0 if interval_index == 0 else "",
    }


def _make_fake_run_experiment(fail_for: set[str] | None = None):
    fail_for = fail_for or set()

    def fake_run_experiment(config: ExperimentConfig, output_dir: Path) -> Path:
        if config.experiment_id in fail_for:
            raise RuntimeError(f"simulated failure for {config.experiment_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{config.experiment_id}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerow(_fake_row(config.experiment_id, 0))
            writer.writerow(_fake_row(config.experiment_id, 1))
        return path

    return fake_run_experiment


def _small_sweep(**overrides) -> SweepConfig:
    defaults = dict(
        bottleneck_bw_mbps_values=[1.0, 2.0],
        bottleneck_delay_ms_values=[0.0],
        offered_load_factors=[1.0],
        n_flows_values=[1],
        repetitions=1,
        sweep_id="test-sweep",
    )
    defaults.update(overrides)
    return SweepConfig(**defaults)


def test_run_sweep_all_succeed(tmp_path):
    sweep = _small_sweep()
    manifest = run_sweep(sweep, tmp_path, run_experiment_fn=_make_fake_run_experiment())
    assert manifest["n_experiments_planned"] == 2
    assert manifest["n_experiments_succeeded"] == 2
    assert manifest["n_experiments_failed"] == 0
    assert manifest["total_rows_written"] == 4  # 2 rows x 2 experiments


def test_run_sweep_records_failures_without_aborting(tmp_path):
    sweep = _small_sweep()
    configs_ids = [c.experiment_id for c in expand_configs(sweep)]
    fail_id = configs_ids[0]
    manifest = run_sweep(sweep, tmp_path, run_experiment_fn=_make_fake_run_experiment(fail_for={fail_id}))
    assert manifest["n_experiments_planned"] == 2
    assert manifest["n_experiments_succeeded"] == 1
    assert manifest["n_experiments_failed"] == 1
    failed_entries = [e for e in manifest["experiments"] if e["status"] == "failed"]
    assert failed_entries[0]["experiment_id"] == fail_id
    assert "simulated failure" in failed_entries[0]["error"]


def test_run_sweep_failure_does_not_fabricate_row_count(tmp_path):
    sweep = _small_sweep()
    configs_ids = [c.experiment_id for c in expand_configs(sweep)]
    manifest = run_sweep(sweep, tmp_path, run_experiment_fn=_make_fake_run_experiment(fail_for=set(configs_ids)))
    assert manifest["n_experiments_succeeded"] == 0
    assert manifest["total_rows_written"] == 0
    assert all(e["row_count"] is None for e in manifest["experiments"])


def test_describe_sweep_touches_no_filesystem(tmp_path):
    sweep = _small_sweep()
    nonexistent_output_dir = tmp_path / "does_not_exist_yet"
    summary = describe_sweep(sweep, nonexistent_output_dir)
    assert not nonexistent_output_dir.exists(), "dry-run description must not create any directory"
    assert summary["n_experiments"] == 2
    assert len(summary["experiment_ids"]) == 2


def test_describe_sweep_estimated_duration_matches_experiment_count():
    sweep = _small_sweep(duration_s=20.0)
    summary = describe_sweep(sweep, Path("data/raw"))
    assert summary["estimated_duration_s"] == 2 * 20.0
