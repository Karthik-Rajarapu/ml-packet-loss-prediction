import pandas as pd

from ml.column_mapping import (
    CONSTANT_ELIGIBLE_FEATURES,
    FEATURE_ALIASES,
    apply_column_mapping,
    detect_column_mapping,
)
from network.schema import FEATURE_COLUMNS


def test_every_feature_column_has_an_alias_entry():
    """A feature with no alias list could never be detected -- this
    would be a silent gap, so it's enforced as a hard test failure."""
    missing = [f for f in FEATURE_COLUMNS if f not in FEATURE_ALIASES]
    assert missing == [], f"FEATURE_ALIASES is missing entries for: {missing}"


def test_exact_canonical_name_is_high_confidence():
    detections = detect_column_mapping(["current_rtt_ms", "packet_loss_pct"])
    assert detections["current_rtt_ms"].confidence == "high"
    assert detections["current_rtt_ms"].detected_column == "current_rtt_ms"
    assert detections["packet_loss_pct"].confidence == "high"


def test_known_alias_is_high_confidence():
    detections = detect_column_mapping(["rtt", "loss", "jitter"])
    assert detections["current_rtt_ms"].confidence == "high"
    assert detections["current_rtt_ms"].detected_column == "rtt"
    assert detections["packet_loss_pct"].confidence == "high"
    assert detections["packet_loss_pct"].detected_column == "loss"
    assert detections["current_jitter_ms"].confidence == "high"


def test_case_and_separator_insensitive_matching():
    # "Round Trip Time" normalizes to "round_trip_time", a known alias
    detections = detect_column_mapping(["Round Trip Time", "Packet-Loss"])
    assert detections["current_rtt_ms"].confidence == "high"
    assert detections["current_rtt_ms"].detected_column == "Round Trip Time"


def test_unrelated_column_name_is_none_confidence():
    detections = detect_column_mapping(["banana", "spaceship_id"])
    assert detections["current_rtt_ms"].confidence == "none"
    assert detections["current_rtt_ms"].detected_column is None


def test_partial_match_is_low_confidence_not_high():
    detections = detect_column_mapping(["my_custom_rtt_measurement_field"])
    result = detections["current_rtt_ms"]
    assert result.confidence == "low"
    assert result.detected_column == "my_custom_rtt_measurement_field"


def test_low_confidence_is_never_silently_treated_as_high():
    """The core Phase 4 safety property: ambiguous matches must be
    distinguishable from confident ones so a caller can gate on it."""
    detections = detect_column_mapping(["my_custom_rtt_measurement_field"])
    assert detections["current_rtt_ms"].confidence != "high"


def test_config_features_are_constant_eligible():
    for feature in ("bottleneck_bw_mbps", "bottleneck_delay_ms", "bottleneck_queue_pkts",
                     "active_connections", "traffic_type"):
        assert feature in CONSTANT_ELIGIBLE_FEATURES


def test_measured_features_are_not_constant_eligible():
    for feature in ("current_rtt_ms", "throughput_mbps", "packet_loss_pct"):
        assert feature not in CONSTANT_ELIGIBLE_FEATURES


def test_apply_column_mapping_renames_columns():
    df = pd.DataFrame({"rtt": [1.0, 2.0], "loss": [0.0, 1.0], "unrelated": ["a", "b"]})
    mapped = apply_column_mapping(df, {"current_rtt_ms": "rtt", "packet_loss_pct": "loss"})
    assert "current_rtt_ms" in mapped.columns
    assert "packet_loss_pct" in mapped.columns
    assert "unrelated" in mapped.columns  # unmapped columns are preserved, not dropped
    assert list(mapped["current_rtt_ms"]) == [1.0, 2.0]


def test_apply_column_mapping_adds_constant_columns():
    df = pd.DataFrame({"rtt": [1.0, 2.0]})
    mapped = apply_column_mapping(df, {"current_rtt_ms": "rtt"}, constant_values={"bottleneck_bw_mbps": 5.0})
    assert (mapped["bottleneck_bw_mbps"] == 5.0).all()


def test_apply_column_mapping_does_not_mutate_input_dataframe():
    df = pd.DataFrame({"rtt": [1.0, 2.0]})
    original_columns = list(df.columns)
    apply_column_mapping(df, {"current_rtt_ms": "rtt"})
    assert list(df.columns) == original_columns
