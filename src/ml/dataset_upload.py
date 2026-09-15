"""Safe, generic handling of an arbitrary user-uploaded CSV, BEFORE any
mapping to the project's canonical schema (network.schema). Pure
Python/pandas, no Streamlit dependency -- fully unit-testable.

Untrusted-input handling: file type/size/shape are validated before
pandas ever touches the bytes; parsing failures are converted into a
human-readable DatasetUploadError rather than a raw pandas exception.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import pandas as pd

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB -- generous for a course-project CSV
MAX_ROWS = 500_000
MIN_ROWS_FOR_TRAINING = 4  # need at least a couple of intervals per experiment/group

_TIMESTAMP_NAME_HINTS = ("timestamp", "time", "date", "datetime")
_TARGET_NAME_HINTS = ("packet_loss", "loss", "packetloss")


class DatasetUploadError(ValueError):
    """Raised for anything wrong with the raw upload itself, before any
    schema mapping is attempted. Messages are written for an end user,
    not a developer."""


@dataclass
class RawDatasetSummary:
    filename: str
    n_rows: int
    n_columns: int
    memory_bytes: int
    columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    missing_value_counts: dict[str, int]
    total_missing_values: int
    duplicate_row_count: int
    likely_timestamp_column: str | None
    likely_target_column: str | None
    preview: pd.DataFrame


def read_uploaded_csv(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse untrusted CSV bytes. Never executes anything from the file --
    pandas' CSV reader only ever produces tabular data, never code."""
    if not filename.lower().endswith(".csv"):
        raise DatasetUploadError(f"Only .csv files are supported (got: {filename!r}).")
    if len(file_bytes) == 0:
        raise DatasetUploadError("The uploaded file is empty.")
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise DatasetUploadError(
            f"File is too large ({len(file_bytes) / 1e6:.1f} MB). "
            f"Maximum supported size is {MAX_FILE_SIZE_BYTES / 1e6:.0f} MB."
        )
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:  # noqa: BLE001 -- pandas raises many exception types for malformed CSV
        raise DatasetUploadError(
            "This file could not be read as a CSV. Please check that it is a valid, comma-separated file."
        ) from exc

    if df.shape[1] == 0:
        raise DatasetUploadError("No columns were found in this file.")
    if len(df) > MAX_ROWS:
        raise DatasetUploadError(f"File has too many rows ({len(df):,}). Maximum supported is {MAX_ROWS:,}.")
    if len(df) < MIN_ROWS_FOR_TRAINING:
        raise DatasetUploadError(
            f"This dataset has too few rows ({len(df)}) to train a next-interval prediction model. "
            f"At least {MIN_ROWS_FOR_TRAINING} rows are needed."
        )
    return df


def _detect_by_name_hint(columns: list[str], hints: tuple[str, ...]) -> str | None:
    for col in columns:
        name = col.lower().strip()
        if any(hint in name for hint in hints):
            return col
    return None


def summarize_raw_dataset(df: pd.DataFrame, filename: str, preview_rows: int = 10) -> RawDatasetSummary:
    """Generic health stats on the RAW (pre-mapping) upload -- this is
    deliberately schema-agnostic: it describes whatever the user actually
    uploaded, using real computed values only, never a hardcoded example."""
    numeric_columns = list(df.select_dtypes(include="number").columns)
    categorical_columns = [c for c in df.columns if c not in numeric_columns]
    missing_counts = {c: int(df[c].isna().sum()) for c in df.columns}
    columns = list(df.columns)
    return RawDatasetSummary(
        filename=filename,
        n_rows=len(df),
        n_columns=df.shape[1],
        memory_bytes=int(df.memory_usage(deep=True).sum()),
        columns=columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        missing_value_counts=missing_counts,
        total_missing_values=int(sum(missing_counts.values())),
        duplicate_row_count=int(df.duplicated().sum()),
        likely_timestamp_column=_detect_by_name_hint(columns, _TIMESTAMP_NAME_HINTS),
        likely_target_column=_detect_by_name_hint(columns, _TARGET_NAME_HINTS),
        preview=df.head(preview_rows),
    )
