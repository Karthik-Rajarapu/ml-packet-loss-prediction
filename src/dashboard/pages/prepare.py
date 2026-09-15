"""Prepare Dataset page: applies the confirmed column mapping, builds
the canonical schema, and constructs the leakage-safe next-interval
target -- entirely via ml.dataset_adapter, which itself calls the
project's existing, unmodified target/validation functions.
"""

from __future__ import annotations

import streamlit as st

from dashboard.nav import NAV_UPLOAD
from dashboard.session import AppSession
from ml.column_mapping import apply_column_mapping
from ml.data_quality import build_data_quality_report
from ml.dataset_adapter import DatasetPreparationError, prepare_canonical_dataset


def render_prepare(session: AppSession) -> None:
    st.title("Prepare Dataset")
    st.caption("We use current network conditions to predict packet loss in the next interval.")

    if session.raw_df is None or (not session.column_mapping and not session.constant_values):
        st.info("Upload and map a dataset first.")
        if st.button("Go to Upload Data"):
            st.session_state["nav_page"] = NAV_UPLOAD
            st.rerun()
        return

    if st.button("Prepare Dataset", type="primary"):
        session.preparation_error = None
        try:
            mapped_df = apply_column_mapping(session.raw_df, session.column_mapping, session.constant_values)
            result = prepare_canonical_dataset(
                mapped_df, group_column=session.group_column, timestamp_column=session.timestamp_column,
            )
        except DatasetPreparationError as exc:
            session.preparation_error = str(exc)
            session.prepared_df = None
            session.preparation_result = None
        else:
            session.prepared_df = result.prepared_df
            session.preparation_result = result
            session.training_result = None  # a re-prepared dataset invalidates any prior training run

    if session.preparation_error:
        st.error(session.preparation_error)
        return

    if session.preparation_result is None:
        st.info("Click “Prepare Dataset” to validate your data and build the prediction target.")
        return

    result = session.preparation_result
    st.success("Dataset prepared successfully.")

    group_desc = "using your selected column" if result.group_column_source == "detected" else "treated as one dataset"
    ts_desc = "using your selected timestamp column" if result.timestamp_column_source == "detected" else "using row order"
    for item in [
        "Dataset loaded",
        "Columns identified",
        f"Grouped into {result.n_experiments} experiment(s)/group(s) ({group_desc})",
        f"Row order established ({ts_desc})",
        "Target constructed: next-interval packet loss",
        "Leakage checks passed",
    ]:
        st.markdown(f"✅ {item}")

    st.divider()
    report = build_data_quality_report(session.prepared_df)
    st.subheader("Target Distribution")
    cols = st.columns(4)
    cols[0].metric("Rows ready for training", f"{report.overview.total_rows:,}")
    cols[1].metric("Mean packet loss", f"{report.target_summary.mean:.2f}%")
    cols[2].metric("Zero-loss intervals", f"{report.target_summary.zero_loss_pct:.1f}%")
    cols[3].metric("Non-zero-loss intervals", f"{report.target_summary.nonzero_loss_pct:.1f}%")

    if report.target_summary.nonzero_loss_pct < 1.0:
        st.warning("Almost all intervals in this dataset have zero packet loss. The model will have very "
                    "little signal to learn from — consider a dataset that includes congested/loss "
                    "conditions.")

    if report.issues.duplicated_experiment_interval_pairs or report.issues.out_of_range_loss_rows:
        st.warning(f"Data quality notes: {report.issues.duplicated_experiment_interval_pairs} duplicated "
                    f"row pair(s), {report.issues.out_of_range_loss_rows} out-of-range loss value(s).")

    st.caption("Continue to “Train Model” in the sidebar.")
