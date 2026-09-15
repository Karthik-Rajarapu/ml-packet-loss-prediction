"""Tests for scripts/predict_packet_loss.py's CLI logic (Step 10-13).

`scripts/` has no __init__.py (matches every other script in this repo),
so the module is loaded directly from its file path rather than via a
package import -- this mirrors how the script is actually invoked
(`python3 scripts/predict_packet_loss.py`), just without spawning a
subprocess, so tests stay fast and can capture stdout/return codes.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "predict_packet_loss_cli", REPO_ROOT / "scripts" / "predict_packet_loss.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Register before exec: see tests/test_system_check.py for why this
    # matters whenever the loaded module defines its own dataclasses.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cli = _load_cli_module()


def test_help_does_not_crash(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.parse_args(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "predict_packet_loss.py" in captured.out or "usage" in captured.out.lower()


def test_requires_exactly_one_input_source():
    with pytest.raises(SystemExit):
        cli.parse_args([])  # none of --demo/--metrics/--metrics-json given


def test_demo_baseline_succeeds(capsys):
    args = cli.parse_args(["--demo", "--baseline"])
    exit_code = cli.run(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "NAIVE PERSISTENCE BASELINE" in out
    assert "Predicted next-interval packet loss" in out
    assert "Risk:" in out


def test_demo_test_fixture_succeeds_and_is_clearly_labeled(capsys):
    args = cli.parse_args(["--demo", "--use-test-fixture"])
    exit_code = cli.run(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "TEST FIXTURE DEMONSTRATION -- NOT REAL NETWORK PERFORMANCE" in out
    assert "NOT TRAINED ON REAL NETWORK DATA" in out


def test_production_mode_fails_gracefully_with_no_model(tmp_path, capsys):
    args = cli.parse_args(["--demo", "--models-dir", str(tmp_path)])
    exit_code = cli.run(args)
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "No production model is available" in out


def test_production_mode_never_silently_uses_test_fixture(tmp_path, capsys):
    """Even with --models-dir pointing at an empty directory, the CLI must
    NOT fall back to a test-fixture model on its own -- it must fail."""
    args = cli.parse_args(["--demo", "--models-dir", str(tmp_path)])
    exit_code = cli.run(args)
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "TEST FIXTURE" not in out


def test_invalid_inline_metrics_json_fails_gracefully(capsys):
    args = cli.parse_args(["--metrics", "not valid json{", "--baseline"])
    exit_code = cli.run(args)
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "Failed to read input metrics" in out


def test_metrics_missing_required_feature_fails_gracefully(capsys):
    incomplete = {k: v for k, v in cli.DEMO_CURRENT_METRICS.items() if k != "current_rtt_ms"}
    args = cli.parse_args(["--metrics", json.dumps(incomplete), "--baseline"])
    exit_code = cli.run(args)
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "Invalid input metrics" in out


def test_metrics_json_file_input_works(tmp_path, capsys):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(cli.DEMO_CURRENT_METRICS))
    args = cli.parse_args(["--metrics-json", str(metrics_path), "--baseline"])
    exit_code = cli.run(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Predicted next-interval packet loss" in out


def test_production_mode_succeeds_with_a_valid_tmp_model(tmp_path, capsys):
    """Proves the production PATH works mechanically, using a model that
    is production-SHAPED (is_test_fixture=False) but trained on fixture
    data purely for test speed -- written only under tmp_path, never the
    real repo models/ directory, and never claimed as a real result."""
    from ml.artifacts import ModelMetadata, save_metadata, save_model
    from ml.models import MODEL_REGISTRY
    from network.schema import FEATURE_COLUMNS, TARGET_COLUMN
    from ml_fixtures import make_labeled_fixture_dataframe

    df = make_labeled_fixture_dataframe(n_experiments=3, n_intervals=5)
    pipeline = MODEL_REGISTRY["decision_tree"]()
    pipeline.fit(df[FEATURE_COLUMNS], df[TARGET_COLUMN])
    save_model(pipeline, tmp_path / "decision_tree.joblib")
    save_metadata(
        ModelMetadata(model_name="decision_tree", feature_columns=FEATURE_COLUMNS,
                       target_column=TARGET_COLUMN, is_test_fixture=False),
        tmp_path / "decision_tree_metadata.json",
    )

    args = cli.parse_args(["--demo", "--models-dir", str(tmp_path)])
    exit_code = cli.run(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "PRODUCTION MODEL PREDICTION" in out
    assert "TEST FIXTURE" not in out
