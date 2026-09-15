"""Phase 7: Streamlit dashboard for next-interval packet-loss prediction.

    streamlit run app.py

PRESENTATION LAYER ONLY. Every prediction, validation, model-loading, and
risk-classification call here is a direct call into Phase 6's
src/ml/inference.py (plus Phase 4's src/ml/risk.py and the saved report
files src/dashboard/reports.py reads). Nothing in this file re-implements
any of that logic -- see docs/PHASE_7_STREAMLIT_DASHBOARD.md Section 1.

No production model exists on this development machine as of Phase 7
(Phase 5 is blocked -- see docs/PHASE_5_REAL_DATA_VALIDATION.md). This
dashboard therefore starts with PRODUCTION mode unavailable and only
DEMONSTRATION mode (an explicitly labeled TEST FIXTURE model) usable.

Future Phase 8 integration point: `run_prediction()` below accepts a
plain `metrics: dict` -- a future live network-metrics collector only
needs to produce that same dict shape (network.schema.FEATURE_COLUMNS
keys) and can call straight into the same function; no dashboard code
needs to change.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import streamlit as st  # noqa: E402

from dashboard.feature_inputs import FeatureInputSpec, build_feature_input_specs  # noqa: E402
from dashboard.history import PredictionHistory, PredictionHistoryEntry  # noqa: E402
from dashboard.model_status import ModelStatus, check_production_model  # noqa: E402
from dashboard.reports import (  # noqa: E402
    find_actual_vs_predicted_plot,
    find_feature_importance,
    find_model_comparison,
    find_permutation_importance,
    find_residuals_plot,
)
from ml.demo import DEMO_CURRENT_METRICS, build_test_fixture_model  # noqa: E402
from ml.inference import (  # noqa: E402
    FeatureValidationError,
    InferenceEngine,
    ModelArtifactError,
    PredictionResult,
    predict_naive_baseline,
)
from network.schema import FEATURE_COLUMNS  # noqa: E402

MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports" / "modeling"

RISK_ICONS = {"LOW": "\U0001F7E2", "MODERATE": "\U0001F7E1", "HIGH": "\U0001F534"}


def _init_session_state() -> None:
    if "history" not in st.session_state:
        st.session_state["history"] = PredictionHistory()


@st.cache_resource(show_spinner="Building TEST FIXTURE demonstration model...")
def _cached_test_fixture_handle():
    """Built once per Streamlit process, not once per rerun. Writes into
    a fresh temp directory -- NEVER models/ -- via ml.demo (Phase 6),
    unmodified here."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="packet_loss_dashboard_fixture_"))
    return build_test_fixture_model(tmp_dir)


@st.cache_resource(show_spinner=False)
def _cached_model_status(models_dir: Path) -> ModelStatus:
    """Phase 8 performance fix: every widget interaction triggers a full
    Streamlit script rerun, so without caching this re-ran
    load_production_model() -- including a fresh joblib.load() off disk
    -- on every single click. Cached per Streamlit process instead.

    Trade-off: if a production model is trained/replaced WHILE the
    dashboard is running, this stays stale until cleared -- the sidebar's
    "Refresh model status" button (see main()) calls
    _cached_model_status.clear() to handle that explicitly rather than
    re-checking disk on every rerun "just in case".
    """
    return check_production_model(models_dir)


def render_system_status(model_status: ModelStatus, mode: str) -> None:
    st.subheader("System Status")
    cols = st.columns(3)
    with cols[0]:
        if mode == "PRODUCTION" and model_status.available:
            st.metric("Model", model_status.handle.metadata.model_name)
        elif mode == "DEMONSTRATION":
            st.metric("Model", "TEST_FIXTURE_decision_tree")
        else:
            st.metric("Model", "unavailable")
    with cols[1]:
        st.metric("Mode", mode)
    with cols[2]:
        if mode == "PRODUCTION":
            st.metric("Model status", "AVAILABLE" if model_status.available else "NOT AVAILABLE")
        else:
            st.metric("Model status", "TEST FIXTURE")

    if mode == "PRODUCTION" and not model_status.available:
        st.warning(
            "**Production model unavailable.**\n\n"
            "Phase 5 real-network dataset generation has not yet been completed.\n\n"
            f"Detail: {model_status.message}\n\n"
            "The dashboard interface is available in demonstration mode. "
            "Production predictions require a validated model trained on real "
            "network experiments."
        )


def render_feature_inputs(specs: list[FeatureInputSpec]) -> dict:
    st.subheader("Current Network Conditions")
    st.caption("These describe the CURRENT interval. The model predicts packet loss for the NEXT interval.")
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
    return values


def run_prediction(mode: str, model_status: ModelStatus, use_baseline: bool, metrics: dict) -> PredictionResult | None:
    """The one place app.py calls into the Phase 6 inference engine.
    Never lets PRODUCTION mode silently fall back to the test fixture."""
    try:
        if use_baseline:
            return predict_naive_baseline(metrics)

        if mode == "DEMONSTRATION":
            handle = _cached_test_fixture_handle()
            return InferenceEngine(handle).predict(metrics)

        if not model_status.available:
            st.error(model_status.message)
            return None
        return InferenceEngine(model_status.handle).predict(metrics)

    except FeatureValidationError as exc:
        st.error(f"Invalid input: {exc}")
        return None
    except ModelArtifactError as exc:
        st.error(str(exc))
        return None
    except Exception as exc:  # noqa: BLE001 -- last-resort guard; never show a raw traceback to normal users
        st.error("An unexpected error occurred while generating the prediction.")
        with st.expander("Technical details (for developers)"):
            st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        return None


def render_prediction_result(result: PredictionResult, current_loss: float) -> None:
    st.subheader("Next-Interval Prediction")
    if result.is_test_fixture:
        st.warning("**DEMONSTRATION MODE — TEST FIXTURE — NOT REAL NETWORK PERFORMANCE**")

    col_current, col_predicted = st.columns(2)
    with col_current:
        st.markdown("**CURRENT OBSERVATION**")
        st.metric("Current packet loss (this interval)", f"{current_loss:.2f}%")
    with col_predicted:
        st.markdown("**PREDICTED NEXT INTERVAL**")
        st.metric("Predicted next-interval packet loss", f"{result.predicted_packet_loss_pct:.2f}%")

    risk_icon = RISK_ICONS.get(result.risk_level, "")
    st.markdown(f"### Risk: {risk_icon} {result.risk_level}")
    st.caption(f"Model: {result.model_name} ({result.model_kind}) — predicted at {result.predicted_at}")


def render_history(history: PredictionHistory) -> None:
    st.subheader("Prediction History (this session)")
    st.caption(
        "In-memory only for this browser session -- never written to disk. No 'actual packet loss' "
        "column is shown: a next-interval outcome is not yet known at prediction time in this "
        "manual-input dashboard (see docs/PHASE_7_STREAMLIT_DASHBOARD.md Section 8)."
    )
    df = history.to_dataframe()
    if df.empty:
        st.info("No predictions yet this session.")
        return
    st.dataframe(df, use_container_width=True)
    if st.button("Clear history"):
        history.clear()
        st.rerun()


def render_model_information(mode: str, model_status: ModelStatus) -> None:
    st.subheader("Model Information")
    st.write(f"Feature count: {len(FEATURE_COLUMNS)}")
    with st.expander("Feature schema"):
        st.write(list(FEATURE_COLUMNS))

    if mode == "PRODUCTION" and model_status.available:
        meta = model_status.handle.metadata
        st.write(f"Model type: {meta.model_name}")
        st.write(f"Training timestamp: {meta.training_timestamp or 'unknown'}")
        if meta.training_config:
            st.write("Training configuration:", meta.training_config)
    else:
        st.info(
            "Model evaluation metrics unavailable because no production model has been trained on "
            "real network data."
        )


def render_feature_importance() -> None:
    st.subheader("Feature Importance")
    importance_df = find_feature_importance(REPORTS_DIR)
    if importance_df is None:
        st.info(
            "Feature importance unavailable: no production-trained model with saved feature "
            "importance exists yet (requires Phase 5 real data + scripts/train_models.py)."
        )
        return
    st.bar_chart(importance_df.set_index("feature")["importance"])
    perm_df = find_permutation_importance(REPORTS_DIR)
    if perm_df is not None:
        with st.expander("Permutation importance (real held-out test set)"):
            st.dataframe(perm_df, use_container_width=True)


def render_actual_vs_predicted() -> None:
    st.subheader("Actual vs Predicted")
    plot_path = find_actual_vs_predicted_plot(REPORTS_DIR)
    if plot_path is None:
        st.info(
            "Actual-vs-predicted visualization will be available after real network experiments "
            "and production-model training."
        )
        return
    st.image(str(plot_path), caption="Actual vs Predicted (real held-out test set)")
    residuals_path = find_residuals_plot(REPORTS_DIR)
    if residuals_path is not None:
        st.image(str(residuals_path), caption="Residual distribution (real held-out test set)")


def render_model_metrics() -> None:
    st.subheader("Model Evaluation Metrics")
    table = find_model_comparison(REPORTS_DIR)
    if table is None:
        st.info(
            "Model evaluation metrics unavailable because no production model has been trained on "
            "real network data."
        )
        return
    st.caption("Real Model | MAE | RMSE | R² comparison, including the naive persistence baseline.")
    st.dataframe(table, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Packet Loss Prediction Dashboard", layout="wide")
    _init_session_state()

    st.title("Packet Loss Prediction Dashboard")
    st.caption("Machine Learning-Based Prediction of Packet Loss in Computer Networks")

    model_status = _cached_model_status(MODELS_DIR)

    default_mode = "PRODUCTION" if model_status.available else "DEMONSTRATION"
    mode_options = ["PRODUCTION", "DEMONSTRATION"]
    mode = st.sidebar.radio("Mode", mode_options, index=mode_options.index(default_mode))
    use_baseline = st.sidebar.checkbox("Use naive persistence baseline instead of the ML model", value=False)
    if st.sidebar.button("Refresh model status"):
        _cached_model_status.clear()
        st.rerun()

    if mode == "DEMONSTRATION":
        st.info(
            "**DEMONSTRATION MODE — TEST FIXTURE — NOT REAL NETWORK PERFORMANCE.** "
            "This mode exists only to verify the dashboard/inference integration."
        )

    render_system_status(model_status, mode)

    specs = build_feature_input_specs(DEMO_CURRENT_METRICS)
    with st.form("prediction_form"):
        metrics = render_feature_inputs(specs)
        submitted = st.form_submit_button("PREDICT NEXT INTERVAL")

    if submitted:
        result = run_prediction(mode, model_status, use_baseline, metrics)
        if result is not None:
            render_prediction_result(result, current_loss=float(metrics["packet_loss_pct"]))
            st.session_state["history"].add(PredictionHistoryEntry(
                timestamp=result.predicted_at,
                model_name=result.model_name,
                model_kind=result.model_kind,
                is_test_fixture=result.is_test_fixture,
                predicted_packet_loss_pct=result.predicted_packet_loss_pct,
                risk_level=result.risk_level,
                input_features=dict(metrics),
            ))

    render_history(st.session_state["history"])
    render_model_information(mode, model_status)
    render_feature_importance()
    render_actual_vs_predicted()
    render_model_metrics()


if __name__ == "__main__":
    main()
