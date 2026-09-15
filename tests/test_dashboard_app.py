"""Tests for app.py itself (Phase 7 Step 15): imports cleanly, and its
prediction-orchestration function (run_prediction) correctly integrates
with Phase 6's inference engine and keeps PRODUCTION/DEMONSTRATION modes
separated. app.py's `if __name__ == "__main__": main()` guard means a
plain import here never renders the UI or touches Streamlit's script-run
machinery -- see app.py's module docstring.
"""

import copy
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_app_module():
    spec = importlib.util.spec_from_file_location("packet_loss_dashboard_app", REPO_ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    # Register before exec: see tests/test_system_check.py for why this
    # matters whenever the loaded module defines its own dataclasses.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


app = _load_app_module()


def _valid_metrics() -> dict:
    return copy.deepcopy(app.DEMO_CURRENT_METRICS)


def test_application_imports_without_executing_main():
    """The core Step 15 requirement: importing app.py must not render
    any UI or raise -- proven simply by having reached this line."""
    assert callable(app.main)
    assert callable(app.run_prediction)


def test_paths_point_at_the_real_repo_locations():
    assert app.MODELS_DIR == REPO_ROOT / "models"
    assert app.REPORTS_DIR == REPO_ROOT / "reports" / "modeling"


def test_feature_mapping_matches_schema():
    from network.schema import FEATURE_COLUMNS
    specs = app.build_feature_input_specs(app.DEMO_CURRENT_METRICS)
    assert [s.name for s in specs] == list(FEATURE_COLUMNS)


def test_run_prediction_baseline_path_does_not_need_a_model(tmp_path):
    model_status = app.check_production_model(tmp_path)  # unavailable
    result = app.run_prediction("PRODUCTION", model_status, use_baseline=True, metrics=_valid_metrics())
    assert result is not None
    assert result.model_kind == "naive_baseline"


def test_run_prediction_demonstration_mode_always_uses_test_fixture(tmp_path):
    """Even if a production model happens to be available, DEMONSTRATION
    mode must still use the test fixture -- modes never bleed into each
    other."""
    model_status = app.check_production_model(tmp_path)  # unavailable, but irrelevant here
    result = app.run_prediction("DEMONSTRATION", model_status, use_baseline=False, metrics=_valid_metrics())
    assert result is not None
    assert result.is_test_fixture is True


def test_run_prediction_production_mode_returns_none_when_unavailable(tmp_path):
    model_status = app.check_production_model(tmp_path)
    assert model_status.available is False
    result = app.run_prediction("PRODUCTION", model_status, use_baseline=False, metrics=_valid_metrics())
    assert result is None, "production mode must never silently substitute a test fixture"


def test_run_prediction_production_mode_uses_real_handle_when_available(tmp_path):
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
    model_status = app.check_production_model(tmp_path)
    assert model_status.available is True

    result = app.run_prediction("PRODUCTION", model_status, use_baseline=False, metrics=_valid_metrics())
    assert result is not None
    assert result.is_test_fixture is False
    assert result.model_kind == "ml_model"


def test_run_prediction_invalid_input_returns_none_not_an_exception(tmp_path):
    model_status = app.check_production_model(tmp_path)
    bad_metrics = _valid_metrics()
    del bad_metrics["current_rtt_ms"]
    result = app.run_prediction("PRODUCTION", model_status, use_baseline=True, metrics=bad_metrics)
    assert result is None


def test_init_session_state_does_not_raise():
    """Exercises the session-state bootstrap path outside a live
    Streamlit session (bare-mode session_state is a documented Streamlit
    limitation, not an error) -- must not raise."""
    app._init_session_state()


def test_history_entry_round_trips_through_prediction_history():
    from dashboard.history import PredictionHistory, PredictionHistoryEntry
    model_status = app.check_production_model(Path("/does/not/exist"))
    result = app.run_prediction("DEMONSTRATION", model_status, use_baseline=False, metrics=_valid_metrics())
    history = PredictionHistory()
    history.add(PredictionHistoryEntry(
        timestamp=result.predicted_at, model_name=result.model_name, model_kind=result.model_kind,
        is_test_fixture=result.is_test_fixture, predicted_packet_loss_pct=result.predicted_packet_loss_pct,
        risk_level=result.risk_level, input_features=_valid_metrics(),
    ))
    assert len(history) == 1
    assert history.to_dataframe().iloc[0]["is_test_fixture"] == True  # noqa: E712
