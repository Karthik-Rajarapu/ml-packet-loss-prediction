"""Tests for ml.inference -- input validation, feature contract, the
InferenceEngine, and the naive baseline. All models here are TEST
FIXTURE models trained on synthetic data (ml.demo / ml_fixtures), never
presented as real performance.
"""

import copy
import math

import pytest

from ml.demo import DEMO_CURRENT_METRICS, build_test_fixture_model
from ml.inference import (
    FeatureValidationError,
    InferenceEngine,
    PredictionResult,
    build_feature_frame,
    predict_naive_baseline,
    validate_feature_input,
)
from network.schema import FEATURE_COLUMNS, TARGET_COLUMN


def _valid_metrics() -> dict:
    return copy.deepcopy(DEMO_CURRENT_METRICS)


# --------------------------------------------------------------------
# Step 4: input validation
# --------------------------------------------------------------------

def test_valid_metrics_pass_validation():
    validated = validate_feature_input(_valid_metrics())
    assert set(validated.keys()) == set(FEATURE_COLUMNS)


def test_missing_feature_rejected():
    metrics = _valid_metrics()
    del metrics["current_rtt_ms"]
    with pytest.raises(FeatureValidationError, match="Missing required feature"):
        validate_feature_input(metrics)


def test_extra_unexpected_feature_rejected():
    metrics = _valid_metrics()
    metrics["totally_made_up_field"] = 1.0
    with pytest.raises(FeatureValidationError, match="Unexpected feature"):
        validate_feature_input(metrics)


def test_nan_value_rejected():
    metrics = _valid_metrics()
    metrics["current_rtt_ms"] = float("nan")
    with pytest.raises(FeatureValidationError, match="NaN"):
        validate_feature_input(metrics)


def test_infinite_value_rejected():
    metrics = _valid_metrics()
    metrics["throughput_mbps"] = math.inf
    with pytest.raises(FeatureValidationError, match="infinite"):
        validate_feature_input(metrics)


def test_negative_value_rejected():
    metrics = _valid_metrics()
    metrics["queue_length"] = -1.0
    with pytest.raises(FeatureValidationError, match="cannot be negative"):
        validate_feature_input(metrics)


def test_out_of_range_percentage_rejected():
    metrics = _valid_metrics()
    metrics["packet_loss_pct"] = 150.0
    with pytest.raises(FeatureValidationError, match=r"\[0, 100\]"):
        validate_feature_input(metrics)


def test_wrong_type_string_for_numeric_rejected():
    metrics = _valid_metrics()
    metrics["current_rtt_ms"] = "twenty"
    with pytest.raises(FeatureValidationError, match="must be numeric"):
        validate_feature_input(metrics)


def test_bool_rejected_for_numeric_feature():
    """bool is a subclass of int in Python -- must not silently pass as numeric."""
    metrics = _valid_metrics()
    metrics["retransmissions"] = True
    with pytest.raises(FeatureValidationError, match="must be numeric"):
        validate_feature_input(metrics)


def test_invalid_traffic_type_rejected():
    metrics = _valid_metrics()
    metrics["traffic_type"] = "quic"
    with pytest.raises(FeatureValidationError, match="traffic_type"):
        validate_feature_input(metrics)


def test_non_string_traffic_type_rejected():
    metrics = _valid_metrics()
    metrics["traffic_type"] = 1
    with pytest.raises(FeatureValidationError, match="must be a string"):
        validate_feature_input(metrics)


def test_non_dict_input_rejected():
    with pytest.raises(FeatureValidationError, match="must be a dict"):
        validate_feature_input([1, 2, 3])


def test_malformed_input_none_rejected():
    with pytest.raises(FeatureValidationError):
        validate_feature_input(None)


# --------------------------------------------------------------------
# Step 3: feature contract (ordering / consistency)
# --------------------------------------------------------------------

def test_build_feature_frame_uses_canonical_column_order():
    validated = validate_feature_input(_valid_metrics())
    frame = build_feature_frame(validated)
    assert list(frame.columns) == list(FEATURE_COLUMNS)


def test_build_feature_frame_order_independent_of_input_dict_order():
    metrics_a = _valid_metrics()
    metrics_b = dict(reversed(list(metrics_a.items())))  # same content, different insertion order
    frame_a = build_feature_frame(validate_feature_input(metrics_a))
    frame_b = build_feature_frame(validate_feature_input(metrics_b))
    assert frame_a.columns.tolist() == frame_b.columns.tolist()
    assert frame_a.equals(frame_b)


def test_target_column_never_in_feature_columns():
    assert TARGET_COLUMN not in FEATURE_COLUMNS


# --------------------------------------------------------------------
# Steps 2, 7, 8: engine, baseline, risk
# --------------------------------------------------------------------

def test_inference_engine_predict_returns_structured_result(tmp_path):
    handle = build_test_fixture_model(tmp_path)
    engine = InferenceEngine(handle)
    result = engine.predict(_valid_metrics())

    assert isinstance(result, PredictionResult)
    assert isinstance(result.predicted_packet_loss_pct, float)
    assert result.risk_level in ("LOW", "MODERATE", "HIGH")
    assert result.model_kind == "ml_model"
    assert result.feature_names == list(FEATURE_COLUMNS)
    assert result.is_test_fixture is True
    assert result.model_metadata is not None


def test_inference_engine_raises_on_invalid_input(tmp_path):
    handle = build_test_fixture_model(tmp_path)
    engine = InferenceEngine(handle)
    bad_metrics = _valid_metrics()
    del bad_metrics["current_rtt_ms"]
    with pytest.raises(FeatureValidationError):
        engine.predict(bad_metrics)


def test_inference_engine_deterministic_given_same_model_and_input(tmp_path):
    handle = build_test_fixture_model(tmp_path)
    engine = InferenceEngine(handle)
    metrics = _valid_metrics()
    result_a = engine.predict(metrics)
    result_b = engine.predict(metrics)
    assert result_a.predicted_packet_loss_pct == result_b.predicted_packet_loss_pct
    assert result_a.risk_level == result_b.risk_level


def test_naive_baseline_predicts_current_loss_exactly():
    metrics = _valid_metrics()
    result = predict_naive_baseline(metrics)
    assert result.predicted_packet_loss_pct == metrics["packet_loss_pct"]
    assert result.model_kind == "naive_baseline"
    assert result.model_name == "naive_persistence"
    assert result.is_test_fixture is False
    assert result.model_metadata is None


def test_naive_baseline_and_ml_model_are_clearly_distinguished(tmp_path):
    metrics = _valid_metrics()
    baseline_result = predict_naive_baseline(metrics)
    handle = build_test_fixture_model(tmp_path)
    ml_result = InferenceEngine(handle).predict(metrics)
    assert baseline_result.model_kind != ml_result.model_kind
    assert baseline_result.model_name != ml_result.model_name


def test_naive_baseline_also_validates_input():
    bad_metrics = _valid_metrics()
    bad_metrics["packet_loss_pct"] = -5.0
    with pytest.raises(FeatureValidationError):
        predict_naive_baseline(bad_metrics)


def test_prediction_result_has_no_confidence_field():
    """Step 9: never invent a confidence/uncertainty score."""
    fields = PredictionResult.__dataclass_fields__
    assert not any("confidence" in name.lower() for name in fields)
    assert not any("uncertainty" in name.lower() for name in fields)
