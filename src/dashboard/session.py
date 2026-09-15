"""Central Streamlit session-state container for the product workflow.

Streamlit's st.session_state is already per-browser-session (it cannot
leak between users) -- this module exists only to avoid scattering ad
hoc string keys across every page module, not to solve a session-
isolation problem that doesn't exist. get_session() lazily creates one
AppSession per session, stored under a single key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from dashboard.history import PredictionHistory


@dataclass
class AppSession:
    # Upload stage
    raw_df: pd.DataFrame | None = None
    upload_filename: str | None = None
    raw_summary: Any = None  # ml.dataset_upload.RawDatasetSummary

    # Mapping stage
    column_mapping: dict[str, str] = field(default_factory=dict)      # canonical_feature -> uploaded_column
    constant_values: dict[str, Any] = field(default_factory=dict)      # canonical_feature -> fixed value
    group_column: str | None = None
    timestamp_column: str | None = None

    # Prepare stage
    prepared_df: pd.DataFrame | None = None
    preparation_result: Any = None  # ml.dataset_adapter.PreparationResult
    preparation_error: str | None = None

    # Train stage
    training_result: Any = None  # ml.training_orchestration.TrainingRunResult
    production_model_saved_this_session: bool = False

    # Predict stage
    history: PredictionHistory = field(default_factory=PredictionHistory)


def get_session() -> AppSession:
    import streamlit as st

    if "app_session" not in st.session_state:
        st.session_state["app_session"] = AppSession()
    return st.session_state["app_session"]
