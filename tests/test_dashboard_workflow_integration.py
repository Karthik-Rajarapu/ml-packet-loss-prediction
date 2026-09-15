"""Phase 9 Step 22 verification, as a test: chains the ENTIRE new
product workflow using a clearly labeled TEST FIXTURE dataset --
upload -> detect/confirm column mapping -> prepare (leakage-safe target)
-> train -> save is deliberately SKIPPED (never written to the real
models/ directory by this test) -> predict, using the in-memory trained
pipeline as if it were production, then renders every page against the
resulting session state.

    TEST FIXTURE — NOT REAL NETWORK DATA. Software verification only;
    none of this is, or may be presented as, an empirical result.
"""

from __future__ import annotations

from pathlib import Path

from dashboard.pages.history_page import render_history_page
from dashboard.pages.predict import _record_history, _run_one_prediction
from dashboard.pages.prepare import render_prepare
from dashboard.pages.train import render_train
from dashboard.pages.upload import render_upload
from dashboard.session import AppSession
from ml.artifacts import ModelMetadata
from ml.column_mapping import apply_column_mapping, detect_column_mapping
from ml.dataset_adapter import prepare_canonical_dataset
from ml.dataset_upload import read_uploaded_csv, summarize_raw_dataset
from ml.inference import InferenceEngine, ModelHandle
from ml.training_orchestration import run_training
from ml_fixtures import make_fixture_rows
from network.schema import FEATURE_COLUMNS, TARGET_COLUMN


def _test_fixture_csv_bytes() -> bytes:
    """A CSV using GENERIC (non-canonical) column names on purpose, so
    this exercises the alias-detection/mapping layer honestly rather
    than handing the pipeline already-canonical names."""
    import pandas as pd

    rows = make_fixture_rows(n_experiments=6, n_intervals=6, id_prefix="workflow-e2e")
    df = pd.DataFrame(rows)
    # rename a few canonical columns to realistic real-world aliases
    df = df.rename(columns={
        "current_rtt_ms": "rtt",
        "current_jitter_ms": "jitter",
        "packet_loss_pct": "loss",
        "throughput_mbps": "throughput",
    })
    return df.to_csv(index=False).encode("utf-8")


def test_full_dashboard_workflow_upload_through_predict(tmp_path):
    session = AppSession()

    # ---- Upload ----
    raw_df = read_uploaded_csv(_test_fixture_csv_bytes(), "test_fixture_workflow.csv")
    session.raw_df = raw_df
    session.upload_filename = "test_fixture_workflow.csv"
    session.raw_summary = summarize_raw_dataset(raw_df, "test_fixture_workflow.csv")
    render_upload(session)  # renders Dataset Health + mapping UI for this real data, must not raise

    # ---- Confirm mapping (what clicking "Confirm Mapping" would do) ----
    detections = detect_column_mapping(session.raw_summary.columns)
    confirmed = {
        feature: det.detected_column
        for feature, det in detections.items()
        if det.detected_column is not None
    }
    # anything still undetected must be supplied as a constant (config-role features)
    missing = [f for f in FEATURE_COLUMNS if f not in confirmed]
    constants = {f: 0.0 if f != "traffic_type" else "udp" for f in missing}
    session.column_mapping = confirmed
    session.constant_values = constants
    session.group_column = "experiment_id"
    session.timestamp_column = "timestamp"

    # ---- Prepare ----
    mapped_df = apply_column_mapping(session.raw_df, session.column_mapping, session.constant_values)
    preparation = prepare_canonical_dataset(mapped_df, group_column="experiment_id", timestamp_column="timestamp")
    session.prepared_df = preparation.prepared_df
    session.preparation_result = preparation
    assert TARGET_COLUMN in session.prepared_df.columns
    assert session.prepared_df[TARGET_COLUMN].isna().sum() == 0
    render_prepare(session)  # must not raise

    # ---- Train (never saved to the real models/ directory) ----
    session.training_result = run_training(session.prepared_df)
    render_train(session, tmp_path / "models", tmp_path / "reports")  # must not raise
    assert not (tmp_path / "models").exists() or not list((tmp_path / "models").glob("*.joblib")), (
        "rendering the Train page must not itself save anything -- only an explicit button click does"
    )

    # ---- Predict, using the just-trained (in-memory only) pipeline as if it were production ----
    best_name = session.training_result.best_model_name
    pipeline = session.training_result.fitted_models[best_name]
    in_memory_handle = ModelHandle(
        pipeline=pipeline,
        metadata=ModelMetadata(model_name=best_name, feature_columns=FEATURE_COLUMNS,
                                 target_column=TARGET_COLUMN, is_test_fixture=False),
        source_path=Path("in-memory-test-only"),
        is_test_fixture=False,
    )

    class _FakeModelStatus:
        available = True
        handle = in_memory_handle

    sample_row = session.prepared_df.iloc[0]
    metrics = {name: (sample_row[name].item() if hasattr(sample_row[name], "item") else sample_row[name])
               for name in FEATURE_COLUMNS}

    result = _run_one_prediction("PRODUCTION", _FakeModelStatus(), False, metrics, lambda: in_memory_handle)
    assert result is not None
    assert result.is_test_fixture is False  # this "production" path correctly reports itself as non-fixture
    _record_history(session, result, metrics)

    # ---- History ----
    assert len(session.history) == 1
    session.history.record_actual(0, actual_packet_loss_pct=metrics["packet_loss_pct"])
    render_history_page(session)  # exercises the actual-vs-predicted chart path too, must not raise
