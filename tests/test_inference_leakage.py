"""Dedicated leakage-protection tests for the inference engine (Phase 6
Step 5). The core rule: the inference engine must only ever consume
information available at or before the prediction cutoff T -- never
anything from the target interval T+1 (packet loss, packets sent/received,
throughput, retransmissions during T+1, or the target value itself).
"""

import copy

import pytest

from ml.demo import DEMO_CURRENT_METRICS, build_test_fixture_model
from ml.inference import FeatureValidationError, InferenceEngine, validate_feature_input
from network.schema import FEATURE_COLUMNS, TARGET_COLUMN


def _valid_metrics() -> dict:
    return copy.deepcopy(DEMO_CURRENT_METRICS)


def test_target_column_is_not_a_valid_input_feature():
    """The feature contract itself must never include the target."""
    assert TARGET_COLUMN not in FEATURE_COLUMNS


def test_supplying_the_target_column_as_input_is_rejected_with_leakage_message():
    metrics = _valid_metrics()
    metrics[TARGET_COLUMN] = 5.0  # attempting to hand the answer to the model
    with pytest.raises(FeatureValidationError, match="target/future-interval"):
        validate_feature_input(metrics)


@pytest.mark.parametrize("leakage_field", [
    "target_next_packet_loss_pct",
    "next_interval_packet_loss",
    "next_packet_loss_pct",
    "future_packet_loss",
    "future_rtt_ms",
])
def test_various_future_looking_field_names_are_flagged_specifically(leakage_field):
    metrics = _valid_metrics()
    metrics[leakage_field] = 1.23
    with pytest.raises(FeatureValidationError, match="target/future-interval"):
        validate_feature_input(metrics)


def test_unrelated_extra_field_gets_generic_message_not_leakage_message():
    """A typo'd/unknown field that does NOT look future-related should
    still be rejected, but with the generic message -- the specific
    leakage wording is reserved for genuinely suspicious names so it
    stays meaningful."""
    metrics = _valid_metrics()
    metrics["extra_unrelated_column"] = 1.0
    with pytest.raises(FeatureValidationError, match="Unexpected feature") as exc_info:
        validate_feature_input(metrics)
    assert "target/future-interval" not in str(exc_info.value)


def test_feature_columns_contain_no_leakage_suspect_names():
    """A structural guard on the schema itself: none of the legitimate
    FEATURE_COLUMNS should accidentally look like target/future
    information (which would mean the schema itself has a leak, not just
    the inference validator)."""
    suspect_substrings = ("target", "next_interval", "next_packet_loss", "future")
    for name in FEATURE_COLUMNS:
        assert not any(s in name.lower() for s in suspect_substrings), (
            f"FEATURE_COLUMNS contains a leakage-suspect name: {name!r}"
        )


def test_engine_prediction_does_not_require_or_accept_future_information(tmp_path):
    """End-to-end: a full valid prediction succeeds using only
    current/past information, and supplying future information alongside
    it is rejected rather than silently ignored or used."""
    handle = build_test_fixture_model(tmp_path)
    engine = InferenceEngine(handle)

    clean_metrics = _valid_metrics()
    result = engine.predict(clean_metrics)  # must succeed without any T+1 information
    assert result.predicted_packet_loss_pct is not None

    contaminated_metrics = _valid_metrics()
    contaminated_metrics["target_next_packet_loss_pct"] = 99.0
    with pytest.raises(FeatureValidationError):
        engine.predict(contaminated_metrics)


def test_current_packet_loss_pct_is_the_only_loss_related_feature():
    """packet_loss_pct (CURRENT interval, safe) must be present, and no
    other loss-related feature name should exist that could smuggle in
    future-interval loss information under a different name."""
    loss_related = [c for c in FEATURE_COLUMNS if "loss" in c.lower()]
    assert loss_related == ["packet_loss_pct"]
