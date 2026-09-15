"""Tests for dashboard.pages.predict -- specifically _run_one_prediction,
the routing function that replaced app.py's old run_prediction() when
the dashboard was restructured into multiple pages (Phase 9). This is
the most safety-critical piece of page logic: it must never let
PRODUCTION mode silently fall back to the test fixture, and must keep
the baseline/ML-model/demo paths cleanly separated.
"""

import copy

import pytest

from dashboard.model_status import check_production_model
from dashboard.pages.predict import _record_history, _run_one_prediction
from dashboard.session import AppSession
from ml.demo import DEMO_CURRENT_METRICS, build_test_fixture_model


def _valid_metrics() -> dict:
    return copy.deepcopy(DEMO_CURRENT_METRICS)


def _get_fixture_handle(tmp_path):
    handle = build_test_fixture_model(tmp_path)
    return lambda: handle


def test_baseline_path_needs_no_model(tmp_path):
    model_status = check_production_model(tmp_path)  # unavailable
    result = _run_one_prediction("PRODUCTION", model_status, True, _valid_metrics(), _get_fixture_handle(tmp_path))
    assert result is not None
    assert result.model_kind == "naive_baseline"


def test_demonstration_mode_always_uses_test_fixture_even_if_production_available(tmp_path):
    from ml.artifacts import ModelMetadata, save_metadata, save_model
    from ml.models import MODEL_REGISTRY
    from ml_fixtures import make_labeled_fixture_dataframe
    from network.schema import FEATURE_COLUMNS, TARGET_COLUMN

    df = make_labeled_fixture_dataframe(n_experiments=3, n_intervals=5)
    pipeline = MODEL_REGISTRY["decision_tree"]()
    pipeline.fit(df[FEATURE_COLUMNS], df[TARGET_COLUMN])
    save_model(pipeline, tmp_path / "decision_tree.joblib")
    save_metadata(
        ModelMetadata(model_name="decision_tree", feature_columns=FEATURE_COLUMNS,
                       target_column=TARGET_COLUMN, is_test_fixture=False),
        tmp_path / "decision_tree_metadata.json",
    )
    model_status = check_production_model(tmp_path)
    assert model_status.available is True

    fixture_dir = tmp_path / "fixture_only"
    fixture_dir.mkdir()
    result = _run_one_prediction(
        "DEMONSTRATION", model_status, False, _valid_metrics(), _get_fixture_handle(fixture_dir)
    )
    assert result.is_test_fixture is True


def test_production_mode_returns_none_when_unavailable(tmp_path):
    model_status = check_production_model(tmp_path)
    result = _run_one_prediction("PRODUCTION", model_status, False, _valid_metrics(), _get_fixture_handle(tmp_path))
    assert result is None, "production mode must never silently substitute a test fixture"


def test_production_mode_uses_real_handle_when_available(tmp_path):
    from ml.artifacts import ModelMetadata, save_metadata, save_model
    from ml.models import MODEL_REGISTRY
    from ml_fixtures import make_labeled_fixture_dataframe
    from network.schema import FEATURE_COLUMNS, TARGET_COLUMN

    df = make_labeled_fixture_dataframe(n_experiments=3, n_intervals=5)
    pipeline = MODEL_REGISTRY["decision_tree"]()
    pipeline.fit(df[FEATURE_COLUMNS], df[TARGET_COLUMN])
    save_model(pipeline, tmp_path / "decision_tree.joblib")
    save_metadata(
        ModelMetadata(model_name="decision_tree", feature_columns=FEATURE_COLUMNS,
                       target_column=TARGET_COLUMN, is_test_fixture=False),
        tmp_path / "decision_tree_metadata.json",
    )
    model_status = check_production_model(tmp_path)
    result = _run_one_prediction("PRODUCTION", model_status, False, _valid_metrics(), _get_fixture_handle(tmp_path))
    assert result is not None
    assert result.is_test_fixture is False
    assert result.model_kind == "ml_model"


def test_invalid_input_returns_none_not_an_exception(tmp_path):
    model_status = check_production_model(tmp_path)
    bad_metrics = _valid_metrics()
    del bad_metrics["current_rtt_ms"]
    result = _run_one_prediction("PRODUCTION", model_status, True, bad_metrics, _get_fixture_handle(tmp_path))
    assert result is None


def test_record_history_appends_entry_with_correct_shape(tmp_path):
    model_status = check_production_model(tmp_path)
    metrics = _valid_metrics()
    result = _run_one_prediction("PRODUCTION", model_status, True, metrics, _get_fixture_handle(tmp_path))
    session = AppSession()
    _record_history(session, result, metrics)
    assert len(session.history) == 1
    entry = session.history.entries[0]
    assert entry.model_kind == "naive_baseline"
    assert entry.predicted_packet_loss_pct == result.predicted_packet_loss_pct
