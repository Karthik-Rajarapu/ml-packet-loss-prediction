"""Bare-mode rendering tests for src/dashboard/pages/*.

Streamlit safely no-ops widget calls when there is no live
ScriptRunContext (documented "bare mode" behavior, empirically verified
throughout this project since Phase 7/8/9's app.main() bare-mode tests).
Calling each render_* function directly, with a session constructed in
various real states, proves the rendering code path itself is free of
Python exceptions across every conditional branch it can reach --
it does not prove pixel-perfect visual output, which was verified
separately via a live headless `streamlit run` this phase.
"""

import pandas as pd

from dashboard.model_status import check_production_model
from dashboard.pages.history_page import render_history_page
from dashboard.pages.home import render_home
from dashboard.pages.prepare import render_prepare
from dashboard.pages.train import render_train
from dashboard.pages.upload import render_upload
from dashboard.session import AppSession
from ml_fixtures import make_labeled_fixture_dataframe


def test_home_page_renders_with_no_model(tmp_path):
    status = check_production_model(tmp_path)
    render_home(status)  # must not raise


def test_home_page_renders_with_available_model(tmp_path):
    from ml.artifacts import ModelMetadata, save_metadata, save_model
    from ml.models import MODEL_REGISTRY
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
    status = check_production_model(tmp_path)
    render_home(status)  # must not raise


def test_upload_page_renders_empty_state():
    session = AppSession()
    render_upload(session)  # must not raise -- no file uploaded in bare mode


def test_upload_page_renders_with_a_raw_dataset_loaded():
    from ml.dataset_upload import summarize_raw_dataset
    df = pd.DataFrame({"rtt": [1.0, 2.0, 3.0, 4.0], "loss": [0.0, 1.0, 0.0, 1.0]})
    session = AppSession()
    session.raw_df = df
    session.upload_filename = "test.csv"
    session.raw_summary = summarize_raw_dataset(df, "test.csv")
    render_upload(session)  # exercises the Dataset Health + mapping UI code path


def test_prepare_page_renders_empty_state():
    session = AppSession()
    render_prepare(session)


def test_prepare_page_renders_with_mapping_confirmed_but_not_yet_prepared():
    session = AppSession()
    session.raw_df = pd.DataFrame({"rtt": [1.0] * 8})
    session.column_mapping = {"current_rtt_ms": "rtt"}
    render_prepare(session)


def test_train_page_renders_empty_state(tmp_path):
    session = AppSession()
    render_train(session, tmp_path / "models", tmp_path / "reports")


def test_train_page_renders_with_a_trained_result(tmp_path):
    from ml.training_orchestration import run_training

    df = make_labeled_fixture_dataframe(n_experiments=6, n_intervals=5)
    session = AppSession()
    session.prepared_df = df
    session.upload_filename = "fixture.csv"
    session.training_result = run_training(df)
    render_train(session, tmp_path / "models", tmp_path / "reports")  # must not raise


def test_history_page_renders_empty_state():
    session = AppSession()
    render_history_page(session)


def test_history_page_renders_with_entries_and_no_actuals():
    from dashboard.history import PredictionHistoryEntry
    session = AppSession()
    session.history.add(PredictionHistoryEntry(
        timestamp="2026-01-01T00:00:00+00:00", model_name="naive_persistence", model_kind="naive_baseline",
        is_test_fixture=False, predicted_packet_loss_pct=3.0, risk_level="MODERATE",
        input_features={"packet_loss_pct": 3.0},
    ))
    render_history_page(session)


def test_history_page_renders_with_a_recorded_actual():
    from dashboard.history import PredictionHistoryEntry
    session = AppSession()
    session.history.add(PredictionHistoryEntry(
        timestamp="2026-01-01T00:00:00+00:00", model_name="naive_persistence", model_kind="naive_baseline",
        is_test_fixture=False, predicted_packet_loss_pct=3.0, risk_level="MODERATE",
        input_features={"packet_loss_pct": 3.0},
    ))
    session.history.record_actual(0, 2.5)
    render_history_page(session)  # exercises the actual-vs-predicted chart path
