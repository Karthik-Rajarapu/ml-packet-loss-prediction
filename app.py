"""Packet Loss Prediction -- product-style Streamlit application.

    streamlit run app.py

PRESENTATION LAYER ONLY. This file is a thin navigation shell: every
page module under src/dashboard/pages/ calls directly into the existing,
unmodified project logic (src/network/schema.py for the one feature
contract, src/ml/dataset_adapter.py + src/network for leakage-safe
target construction and validation, src/ml/training_orchestration.py for
the existing split/train/evaluate/artifact pipeline, src/ml/inference.py
for the inference engine, src/ml/risk.py for risk classification).
Nothing is re-implemented here.

User journey: Home -> Upload Data -> Prepare Dataset -> Train Model ->
Predict -> History. The original Phase 7 manual-entry form still exists,
relocated (unchanged) into Predict's "Manual Prediction" tab.

No production model exists on this development machine as of this
writing (real Mininet data generation is blocked -- see
docs/PHASE_5_REAL_DATA_VALIDATION.md). A user can still train a real
(non-fixture) production model here by uploading their OWN dataset --
the application never claims real-world network accuracy beyond
whatever a genuine uploaded dataset and its own held-out evaluation
actually support.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import streamlit as st  # noqa: E402

from dashboard.model_status import ModelStatus, check_production_model  # noqa: E402
from dashboard.nav import NAV_HISTORY, NAV_HOME, NAV_ORDER, NAV_PREDICT, NAV_PREPARE, NAV_TRAIN, NAV_UPLOAD, nav_label  # noqa: E402
from dashboard.pages.history_page import render_history_page  # noqa: E402
from dashboard.pages.home import render_home  # noqa: E402
from dashboard.pages.predict import render_predict  # noqa: E402
from dashboard.pages.prepare import render_prepare  # noqa: E402
from dashboard.pages.train import render_train  # noqa: E402
from dashboard.pages.upload import render_upload  # noqa: E402
from dashboard.session import get_session  # noqa: E402
from ml.demo import build_test_fixture_model  # noqa: E402

MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports" / "modeling"


@st.cache_resource(show_spinner="Building TEST FIXTURE demonstration model...")
def _cached_test_fixture_handle():
    """Built once per Streamlit process, not once per rerun. Writes into
    a fresh temp directory -- NEVER models/ -- via ml.demo, unmodified."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="packet_loss_dashboard_fixture_"))
    return build_test_fixture_model(tmp_dir)


@st.cache_resource(show_spinner=False)
def _cached_model_status(models_dir: Path) -> ModelStatus:
    """Avoids reloading the production model artifact from disk on every
    widget interaction. Invalidated explicitly (st.cache_resource.clear())
    whenever the Train page saves a new production model, or via the
    sidebar's 'Refresh model status' button."""
    return check_production_model(models_dir)


def main() -> None:
    st.set_page_config(page_title="Packet Loss Prediction", layout="wide")
    session = get_session()

    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = NAV_HOME

    model_status = _cached_model_status(MODELS_DIR)

    st.sidebar.title("Packet Loss Prediction")
    nav_choice = st.sidebar.radio(
        "Navigate", NAV_ORDER, format_func=nav_label,
        index=NAV_ORDER.index(st.session_state["nav_page"]),
    )
    st.session_state["nav_page"] = nav_choice

    st.sidebar.divider()
    use_baseline = st.sidebar.checkbox("Use naive persistence baseline instead of the ML model", value=False)
    mode_options = ["PRODUCTION", "DEMONSTRATION"]
    default_mode_index = 0 if model_status.available else 1
    mode = st.sidebar.radio("Prediction mode", mode_options, index=default_mode_index)
    if st.sidebar.button("Refresh model status"):
        _cached_model_status.clear()
        st.rerun()

    if nav_choice == NAV_HOME:
        render_home(model_status)
    elif nav_choice == NAV_UPLOAD:
        render_upload(session)
    elif nav_choice == NAV_PREPARE:
        render_prepare(session)
    elif nav_choice == NAV_TRAIN:
        render_train(session, MODELS_DIR, REPORTS_DIR)
    elif nav_choice == NAV_PREDICT:
        render_predict(session, mode, model_status, use_baseline, _cached_test_fixture_handle)
    elif nav_choice == NAV_HISTORY:
        render_history_page(session)


if __name__ == "__main__":
    main()
