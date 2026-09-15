import json
from pathlib import Path

from network.manifest import ExperimentResult, build_manifest, environment_snapshot, write_manifest
from network.sweep import SweepConfig


def test_environment_snapshot_never_raises_even_without_tools():
    snapshot = environment_snapshot()
    assert "python_version" in snapshot
    assert "tool_versions" in snapshot
    # Tools missing on this machine (e.g. Windows lacking mn/iperf3) must
    # come back as None, not raise or crash manifest generation.
    assert set(snapshot["tool_versions"]) == {"mn", "iperf3", "tc", "ovs-vsctl", "ping"}


def test_build_manifest_counts_success_and_failure():
    sweep = SweepConfig(repetitions=1)
    results = [
        ExperimentResult(experiment_id="a", status="success", csv_path="data/raw/a.csv", row_count=9),
        ExperimentResult(experiment_id="b", status="failed", error="iperf3 timed out"),
    ]
    manifest = build_manifest(sweep, results, "2026-01-01T00:00:00Z", "2026-01-01T00:05:00Z", Path("data/raw"))
    assert manifest["n_experiments_planned"] == 2
    assert manifest["n_experiments_succeeded"] == 1
    assert manifest["n_experiments_failed"] == 1
    assert manifest["total_rows_written"] == 9
    assert manifest["sweep_id"] == sweep.sweep_id


def test_build_manifest_never_counts_rows_for_failed_experiments():
    sweep = SweepConfig(repetitions=1)
    results = [ExperimentResult(experiment_id="a", status="failed", error="boom", row_count=None)]
    manifest = build_manifest(sweep, results, "t0", "t1", Path("data/raw"))
    assert manifest["total_rows_written"] == 0


def test_build_manifest_includes_feature_and_target_columns():
    sweep = SweepConfig(repetitions=1)
    manifest = build_manifest(sweep, [], "t0", "t1", Path("data/raw"))
    assert "target_next_packet_loss_pct" == manifest["target_column"]
    assert len(manifest["feature_columns"]) > 0


def test_write_manifest_roundtrips_to_valid_json(tmp_path):
    sweep = SweepConfig(repetitions=1)
    manifest = build_manifest(sweep, [], "t0", "t1", Path("data/raw"))
    out_path = write_manifest(manifest, tmp_path / "nested" / "m.json")
    assert out_path.exists()
    reloaded = json.loads(out_path.read_text())
    assert reloaded["sweep_id"] == sweep.sweep_id


def test_dry_run_flag_is_recorded():
    sweep = SweepConfig(repetitions=1)
    manifest = build_manifest(sweep, [], "t0", "t1", Path("data/raw"), dry_run=True)
    assert manifest["dry_run"] is True
