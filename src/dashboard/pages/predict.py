"""Predict page: Batch Prediction (CSV, primary workflow) + Manual
Prediction (secondary tab -- the original Phase 7 form, relocated here
unchanged, per Phase 9 Phase 15: "refactor it into 'Manual Prediction'
inside the Prediction workflow").

All prediction logic is a direct call into ml.inference (Phase 6,
unmodified); nothing here re-implements validation, preprocessing, or
risk classification.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
import streamlit as st

from dashboard.feature_inputs import build_feature_input_specs
from dashboard.history import PredictionHistoryEntry
from dashboard.model_status import ModelStatus
from dashboard.session import AppSession
from ml.dataset_upload import DatasetUploadError, read_uploaded_csv
from ml.demo import DEMO_CURRENT_METRICS
from ml.inference import FeatureValidationError, InferenceEngine, ModelHandle, PredictionResult, predict_naive_baseline
from network.schema import FEATURE_COLUMNS

RISK_ICONS = {"LOW": "\U0001F7E2", "MODERATE": "\U0001F7E1", "HIGH": "\U0001F534"}


def render_predict(
    session: AppSession,
    mode: str,
    model_status: ModelStatus,
    use_baseline: bool,
    get_test_fixture_handle: Callable[[], ModelHandle],
) -> None:
    st.title("Predict")
    if mode == "DEMONSTRATION":
        st.warning("**DEMONSTRATION MODE — TEST FIXTURE — NOT REAL NETWORK PERFORMANCE.** "
                    "Predictions here use a small model trained on synthetic data, not your uploaded dataset.")
    elif not model_status.available and not use_baseline:
        st.info("No trained production model is available yet. Upload a dataset and train a model on the "
                 "earlier pages, or use the naive baseline / switch to Demonstration mode in the sidebar.")
    elif mode == "PRODUCTION" and model_status.available:
        meta = model_status.handle.metadata
        with st.expander(f"Active production model: {meta.model_name}"):
            st.write(f"Trained: {meta.training_timestamp or 'unknown'}")
            if meta.training_config:
                st.write("Training configuration:", meta.training_config)
            if meta.metrics:
                st.write("Test-set metrics at training time:", meta.metrics)

    tab_batch, tab_manual = st.tabs(["Batch Prediction (CSV)", "Manual Prediction"])
    with tab_batch:
        _render_batch_tab(session, mode, model_status, use_baseline, get_test_fixture_handle)
    with tab_manual:
        _render_manual_tab(session, mode, model_status, use_baseline, get_test_fixture_handle)


def _run_one_prediction(
    mode: str, model_status: ModelStatus, use_baseline: bool, metrics: dict,
    get_test_fixture_handle: Callable[[], ModelHandle],
) -> PredictionResult | None:
    try:
        if use_baseline:
            return predict_naive_baseline(metrics)
        if mode == "DEMONSTRATION":
            return InferenceEngine(get_test_fixture_handle()).predict(metrics)
        if not model_status.available:
            return None
        return InferenceEngine(model_status.handle).predict(metrics)
    except FeatureValidationError as exc:
        st.error(f"Your input has a problem: {exc}")
        return None


def _record_history(session: AppSession, result: PredictionResult, metrics: dict) -> None:
    session.history.add(PredictionHistoryEntry(
        timestamp=result.predicted_at, model_name=result.model_name, model_kind=result.model_kind,
        is_test_fixture=result.is_test_fixture, predicted_packet_loss_pct=result.predicted_packet_loss_pct,
        risk_level=result.risk_level, input_features=dict(metrics),
    ))


def _render_result(result: PredictionResult, current_loss: float) -> None:
    if result.is_test_fixture:
        st.warning("**TEST FIXTURE — NOT REAL NETWORK PERFORMANCE**")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Predicted Next-Interval Packet Loss", f"{result.predicted_packet_loss_pct:.2f}%")
    col_b.metric("Risk", f"{RISK_ICONS.get(result.risk_level, '')} {result.risk_level}")
    change = result.predicted_packet_loss_pct - current_loss
    col_c.metric("Predicted Change", f"{change:+.2f} pts")
    st.caption(f"Current packet loss (this interval): {current_loss:.2f}%  —  "
                f"Model: {result.model_name} ({result.model_kind})")


def _render_manual_tab(session, mode, model_status, use_baseline, get_test_fixture_handle) -> None:
    st.caption("Enter current network conditions for a single observation.")
    specs = build_feature_input_specs(DEMO_CURRENT_METRICS)
    with st.form("manual_prediction_form"):
        values: dict = {}
        cols = st.columns(2)
        for i, spec in enumerate(specs):
            target_col = cols[i % 2]
            label = f"{spec.name} ({spec.unit})"
            with target_col:
                if spec.widget == "select":
                    options = list(spec.options)
                    values[spec.name] = st.selectbox(label, options=options, index=options.index(spec.default),
                                                       help=spec.description)
                else:
                    values[spec.name] = st.number_input(
                        label, min_value=spec.min_value, max_value=spec.max_value,
                        value=spec.default, step=spec.step, help=spec.description,
                    )
        submitted = st.form_submit_button("Predict Next Interval", type="primary")

    if submitted:
        if mode == "PRODUCTION" and not use_baseline and not model_status.available:
            st.error("No trained production model is available yet. Upload a dataset and train a model "
                       "first, or use the naive baseline / Demonstration mode.")
            return
        result = _run_one_prediction(mode, model_status, use_baseline, values, get_test_fixture_handle)
        if result is not None:
            _render_result(result, current_loss=float(values["packet_loss_pct"]))
            _record_history(session, result, values)


def _render_batch_tab(session, mode, model_status, use_baseline, get_test_fixture_handle) -> None:
    st.caption("Upload one or more current observations as a CSV. Columns are matched using the same "
                "mapping confirmed when you uploaded your training dataset, where possible.")
    uploaded = st.file_uploader("Current network measurements (CSV)", type=["csv"], key="predict_batch_upload")
    if uploaded is None:
        st.info("Upload a CSV to predict packet loss for one or more observations.")
        return

    try:
        raw_df = read_uploaded_csv(uploaded.getvalue(), uploaded.name)
    except DatasetUploadError as exc:
        st.error(str(exc))
        return

    mapped_df = raw_df.copy()
    if session.column_mapping:
        rename_map = {uploaded_col: feature for feature, uploaded_col in session.column_mapping.items()
                      if uploaded_col in raw_df.columns}
        mapped_df = mapped_df.rename(columns=rename_map)
    for feature, value in (session.constant_values or {}).items():
        if feature not in mapped_df.columns:
            mapped_df[feature] = value

    missing = [f for f in FEATURE_COLUMNS if f not in mapped_df.columns]
    if missing:
        st.error(f"Your prediction data is missing required network measurements: {', '.join(missing)}.")
        return

    if mode == "PRODUCTION" and not use_baseline and not model_status.available:
        st.error("No trained production model is available yet. Upload a dataset and train a model first, "
                   "or use the naive baseline / Demonstration mode.")
        return

    results = []
    for idx, row in mapped_df[FEATURE_COLUMNS].reset_index(drop=True).iterrows():
        metrics = row.to_dict()
        result = _run_one_prediction(mode, model_status, use_baseline, metrics, get_test_fixture_handle)
        if result is None:
            continue
        results.append({
            "row": idx,
            "predicted_packet_loss_pct": result.predicted_packet_loss_pct,
            "risk_level": result.risk_level,
            "current_packet_loss_pct": metrics["packet_loss_pct"],
        })
        _record_history(session, result, metrics)

    if results:
        st.subheader("Predictions")
        st.dataframe(pd.DataFrame(results), use_container_width=True)
    else:
        st.warning("No rows could be predicted — see any error messages above.")
