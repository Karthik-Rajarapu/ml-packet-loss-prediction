import pytest

from network.config import ExperimentConfig
from network.validation import SchemaValidationError, check_environment, require_environment, validate_rows


def _row(experiment_id, idx, **overrides):
    row = {
        "experiment_id": experiment_id,
        "interval_index": idx,
        "timestamp": 1000.0 + idx,
        "packets_sent": 100,
        "packets_received": 95,
        "packet_loss_pct": 5.0,
    }
    row.update(overrides)
    return row


def test_check_environment_reports_missing_tools_without_raising():
    result = check_environment(tools=["definitely-not-a-real-command-xyz"])
    assert result.all_available is False
    assert "definitely-not-a-real-command-xyz" in result.missing


def test_validate_rows_accepts_well_formed_rows():
    rows = [_row("e1", 0), _row("e1", 1)]
    validate_rows(rows, ["experiment_id", "interval_index", "timestamp"])  # should not raise


def test_validate_rows_rejects_non_contiguous_interval_index():
    rows = [_row("e1", 0), _row("e1", 2)]
    with pytest.raises(SchemaValidationError):
        validate_rows(rows, ["experiment_id", "interval_index", "timestamp"])


def test_validate_rows_rejects_decreasing_timestamp():
    rows = [_row("e1", 0, timestamp=100.0), _row("e1", 1, timestamp=50.0)]
    with pytest.raises(SchemaValidationError):
        validate_rows(rows, ["experiment_id", "interval_index", "timestamp"])


def test_validate_rows_rejects_received_greater_than_sent():
    rows = [_row("e1", 0, packets_sent=10, packets_received=50)]
    with pytest.raises(SchemaValidationError):
        validate_rows(rows, ["experiment_id", "interval_index", "timestamp"])


def test_validate_rows_rejects_out_of_range_loss_pct():
    rows = [_row("e1", 0, packet_loss_pct=150.0)]
    with pytest.raises(SchemaValidationError):
        validate_rows(rows, ["experiment_id", "interval_index", "timestamp"])


def test_validate_rows_rejects_missing_required_column():
    rows = [{"experiment_id": "e1", "interval_index": 0}]
    with pytest.raises(SchemaValidationError):
        validate_rows(rows, ["experiment_id", "interval_index", "timestamp"])


def test_validate_rows_rejects_empty_input():
    with pytest.raises(SchemaValidationError):
        validate_rows([], ["experiment_id"])


def test_experiment_config_rejects_more_than_four_hosts():
    with pytest.raises(ValueError):
        ExperimentConfig(n_left_hosts=3, n_right_hosts=3)


def test_experiment_config_rejects_too_short_duration():
    with pytest.raises(ValueError):
        ExperimentConfig(duration_s=1.0, sample_interval_s=2.0)


def test_experiment_config_default_is_valid():
    cfg = ExperimentConfig()
    assert cfg.experiment_id.startswith("exp-")


def test_require_environment_raises_clearly_when_tools_missing():
    """This is the exact failure path scripts/run_experiment.py and
    scripts/generate_dataset.py rely on to refuse running (and refuse
    writing any CSV/manifest) when Mininet/iperf3/tc aren't available --
    see PHASE_1_NETWORK_EXPERIMENT.md Section 11 and
    PHASE_2_DATA_GENERATION.md."""
    with pytest.raises(RuntimeError, match="definitely-not-a-real-tool-xyz"):
        require_environment(tools=["definitely-not-a-real-tool-xyz"])


def test_require_environment_passes_silently_when_tools_present():
    require_environment(tools=[])  # empty requirement list is trivially satisfied, should not raise
