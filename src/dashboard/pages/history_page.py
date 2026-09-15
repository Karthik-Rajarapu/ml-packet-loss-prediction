"""History page: prediction history table, an optional 'record actual
outcome' input, and an actual-vs-predicted chart shown ONLY once the
user has genuinely recorded at least one real outcome (never fabricated
-- see dashboard.history's module docstring).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.session import AppSession  # noqa: E402


def render_history_page(session: AppSession) -> None:
    st.title("History")
    st.caption("Predictions made this session. In-memory only — never written to disk, cleared when the "
                "browser session ends.")

    df = session.history.to_dataframe()
    if df.empty:
        st.info("No predictions yet. Make a prediction on the Predict page to see it here.")
        return

    st.dataframe(df, use_container_width=True)
    if st.button("Clear History"):
        session.history.clear()
        st.rerun()

    st.divider()
    st.subheader("Record an Actual Outcome")
    st.caption("If you later observe the real packet loss that occurred, record it here to compare against "
                "what was predicted. This value is never filled in automatically.")
    n = len(session.history)
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        index = st.number_input("Row index", min_value=0, max_value=max(n - 1, 0), value=0, step=1)
    with col_b:
        actual_value = st.number_input("Actual packet loss (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    with col_c:
        st.write("")
        st.write("")
        if st.button("Record"):
            session.history.record_actual(int(index), float(actual_value))
            st.success(f"Recorded actual outcome for row {int(index)}.")
            st.rerun()

    with_actuals = session.history.entries_with_actuals()
    st.divider()
    st.subheader("Actual vs Predicted")
    if with_actuals.empty:
        st.info("This will appear once you record at least one real outcome above.")
        return

    st.dataframe(with_actuals, use_container_width=True)

    fig, ax = plt.subplots()
    ax.scatter(with_actuals["actual_packet_loss_pct"], with_actuals["predicted_packet_loss_pct"])
    lo = min(with_actuals["actual_packet_loss_pct"].min(), with_actuals["predicted_packet_loss_pct"].min())
    hi = max(with_actuals["actual_packet_loss_pct"].max(), with_actuals["predicted_packet_loss_pct"].max())
    if hi > lo:
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="Perfect prediction")
        ax.legend()
    ax.set_xlabel("Actual packet loss (%)")
    ax.set_ylabel("Predicted packet loss (%)")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Based on outcomes you recorded manually in this session — not a held-out test-set "
                "evaluation.")
