"""Upload Data page: CSV upload, generic dataset health, and column
mapping (Phase 9 Phases 3-4). Presentation only -- all real logic is in
ml.dataset_upload / ml.column_mapping.
"""

from __future__ import annotations

import streamlit as st

from dashboard.session import AppSession
from ml.column_mapping import CONSTANT_ELIGIBLE_FEATURES, detect_column_mapping
from ml.dataset_upload import DatasetUploadError, read_uploaded_csv, summarize_raw_dataset
from network.schema import FEATURE_COLUMNS, SCHEMA


def _reset_downstream_state(session: AppSession) -> None:
    """A fresh upload invalidates any previous mapping/preparation/training
    -- never silently carry stale state forward to a new dataset."""
    session.column_mapping = {}
    session.constant_values = {}
    session.group_column = None
    session.timestamp_column = None
    session.prepared_df = None
    session.preparation_result = None
    session.preparation_error = None
    session.training_result = None


def render_upload(session: AppSession) -> None:
    st.title("Upload Network Data")
    st.caption("Upload a CSV of network performance measurements.")

    uploaded_file = st.file_uploader("Network measurement dataset (CSV)", type=["csv"])
    if uploaded_file is not None and uploaded_file.name != session.upload_filename:
        try:
            df = read_uploaded_csv(uploaded_file.getvalue(), uploaded_file.name)
        except DatasetUploadError as exc:
            st.error(str(exc))
            return
        session.raw_df = df
        session.upload_filename = uploaded_file.name
        session.raw_summary = summarize_raw_dataset(df, uploaded_file.name)
        _reset_downstream_state(session)

    if session.raw_df is None or session.raw_summary is None:
        st.info("Upload a network measurement dataset to begin.")
        return

    summary = session.raw_summary

    st.subheader("Dataset Health")
    row1 = st.columns(3)
    row1[0].metric("Rows", f"{summary.n_rows:,}")
    row1[1].metric("Features", summary.n_columns)
    row1[2].metric("Missing Values", f"{summary.total_missing_values:,}")
    row2 = st.columns(3)
    row2[0].metric("Duplicate Rows", f"{summary.duplicate_row_count:,}")
    row2[1].metric("Time Column", "Detected" if summary.likely_timestamp_column else "Not detected")
    row2[2].metric("Target Column", "Detected" if summary.likely_target_column else "Not detected")

    with st.expander("Preview data"):
        st.dataframe(summary.preview, use_container_width=True)

    if summary.duplicate_row_count > 0:
        st.warning(f"{summary.duplicate_row_count} duplicate row(s) were found. They will remain in the "
                    "dataset unless you remove them from the source file.")
    if summary.total_missing_values > 0:
        st.info(f"{summary.total_missing_values} missing value(s) were found across the dataset. "
                 "Missing measurements in mapped feature columns will be handled during training "
                 "(median imputation, fit on training data only).")

    st.divider()
    st.subheader("Map Columns")
    st.caption("Detected feature → your column. Review each one; low-confidence or missing matches need your input.")

    detections = detect_column_mapping(summary.columns)
    schema_by_name = {c.name: c for c in SCHEMA}
    column_options = ["— Not in dataset —"] + summary.columns

    confirmed_mapping: dict[str, str] = {}
    constant_values: dict[str, object] = {}

    for feature in FEATURE_COLUMNS:
        detection = detections[feature]
        spec = schema_by_name[feature]
        label_col, input_col = st.columns([2, 3])
        with label_col:
            st.markdown(f"**{feature}**")
            st.caption(f"{spec.description} ({spec.unit})")
        with input_col:
            use_constant = False
            if feature in CONSTANT_ELIGIBLE_FEATURES:
                use_constant = st.checkbox(
                    "Use one fixed value for the whole dataset", key=f"const_toggle_{feature}"
                )
            if use_constant:
                if feature == "traffic_type":
                    constant_values[feature] = st.selectbox("Fixed value", ["udp", "tcp"], key=f"const_val_{feature}")
                else:
                    constant_values[feature] = st.number_input("Fixed value", value=0.0, key=f"const_val_{feature}")
            else:
                default_index = 0
                if detection.confidence == "high" and detection.detected_column in summary.columns:
                    default_index = column_options.index(detection.detected_column)
                chosen = st.selectbox(
                    f"Column for {feature}", column_options, index=default_index,
                    key=f"map_{feature}", label_visibility="collapsed",
                )
                if detection.confidence == "high":
                    st.caption(f"✓ Detected: '{detection.detected_column}'")
                elif detection.confidence == "low":
                    st.caption(f"⚠ Possible match: '{detection.detected_column}' — please confirm")
                else:
                    st.caption("No confident match found — please select manually")
                if chosen != column_options[0]:
                    confirmed_mapping[feature] = chosen
        st.divider()

    st.subheader("Identify Rows")
    id_col1, id_col2 = st.columns(2)
    with id_col1:
        group_options = ["— Treat as one dataset —"] + summary.columns
        group_choice = st.selectbox("Experiment / group column (optional)", group_options,
                                     help="If your data comes from multiple separate runs, select the "
                                          "column that identifies which run each row belongs to.")
        group_column = None if group_choice == group_options[0] else group_choice
    with id_col2:
        ts_options = ["— Use row order —"] + summary.columns
        default_ts_index = 0
        if summary.likely_timestamp_column and summary.likely_timestamp_column in summary.columns:
            default_ts_index = ts_options.index(summary.likely_timestamp_column)
        ts_choice = st.selectbox("Timestamp column (optional)", ts_options, index=default_ts_index)
        timestamp_column = None if ts_choice == ts_options[0] else ts_choice

    missing = [f for f in FEATURE_COLUMNS if f not in confirmed_mapping and f not in constant_values]
    if missing:
        st.warning(f"Not yet mapped: {', '.join(missing)}. Map these columns or provide a fixed value to continue.")

    if st.button("Confirm Mapping", type="primary", disabled=bool(missing)):
        session.column_mapping = confirmed_mapping
        session.constant_values = constant_values
        session.group_column = group_column
        session.timestamp_column = timestamp_column
        session.prepared_df = None
        session.preparation_result = None
        session.training_result = None
        st.success("Mapping confirmed. Continue to “Prepare Dataset” in the sidebar.")
