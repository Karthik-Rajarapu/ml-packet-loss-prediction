"""Home / landing page. No implementation detail is exposed here --
no mention of pipelines, models by name, feature contracts, or file
paths; just the product framing and one honest, plain-language status
line."""

from __future__ import annotations

import streamlit as st

from dashboard.model_status import ModelStatus
from dashboard.nav import NAV_UPLOAD


def render_home(model_status: ModelStatus) -> None:
    st.title("Packet Loss Prediction")
    st.caption("Predict future network packet loss using network performance measurements.")
    st.write("")

    cards = [
        ("\U0001F4E4", "Upload Network Data", "Upload your network measurement dataset."),
        ("\U0001F3AF", "Train & Validate",
         "Prepare the data, evaluate candidate models, and select the best validated model."),
        ("\U0001F52E", "Predict",
         "Use current network measurements to predict packet loss for the next interval."),
    ]
    cols = st.columns(3)
    for col, (icon, title, desc) in zip(cols, cards):
        with col, st.container(border=True):
            st.markdown(f"#### {icon} {title}")
            st.write(desc)

    st.write("")
    _, center, _ = st.columns([1, 1, 1])
    with center:
        if st.button("Get Started →", type="primary", use_container_width=True):
            st.session_state["nav_page"] = NAV_UPLOAD
            st.rerun()

    st.write("")
    st.divider()
    status_text = "A trained model is available." if model_status.available else "No trained model yet — you'll train one after uploading data."
    st.caption(status_text)
