"""Turns a column-mapped upload into the project's canonical
network.schema.CSV_COLUMNS shape and constructs the leakage-safe target.

This module reuses network.targets.add_next_interval_target and
network.validation's validators COMPLETELY UNCHANGED -- it only builds
the input shape those functions already expect (adding the identifier
columns: experiment_id, run_id, interval_index, timestamp, source_host,
destination_host, since an arbitrary upload won't already have Mininet
experiment identifiers). No leakage-relevant logic is reimplemented here.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from network.schema import CSV_COLUMNS, FEATURE_COLUMNS
from network.targets import add_next_interval_target, drop_rows_without_target
from network.validation import SchemaValidationError, validate_no_leakage, validate_rows


class DatasetPreparationError(ValueError):
    """Human-readable preparation failure -- never a raw validator
    exception reaches the caller."""


@dataclass
class PreparationResult:
    prepared_df: pd.DataFrame
    n_experiments: int
    n_rows_before_target_drop: int
    n_rows_after_target_drop: int
    group_column_source: str  # "detected" | "single_experiment"
    timestamp_column_source: str  # "detected" | "row_order"


def prepare_canonical_dataset(
    mapped_df: pd.DataFrame,
    group_column: str | None = None,
    timestamp_column: str | None = None,
    target_horizon: int = 1,
) -> PreparationResult:
    """`mapped_df` must already have canonical FEATURE_COLUMNS names
    (via column_mapping.apply_column_mapping). Adds identifier columns,
    then calls the EXISTING leakage-safe target construction and
    validators, unmodified.
    """
    missing_features = [f for f in FEATURE_COLUMNS if f not in mapped_df.columns]
    if missing_features:
        raise DatasetPreparationError(
            "Your dataset is missing required network measurements: "
            f"{', '.join(missing_features)}. Please map these columns (or, for link-configuration "
            "fields, provide a constant value) and try again."
        )

    working = mapped_df.copy().reset_index(drop=True)

    if group_column and group_column in working.columns:
        working["experiment_id"] = working[group_column].astype(str)
        group_source = "detected"
    else:
        working["experiment_id"] = "uploaded-dataset"
        group_source = "single_experiment"
    working["run_id"] = working["experiment_id"]

    if timestamp_column and timestamp_column in working.columns:
        numeric_ts = pd.to_numeric(working[timestamp_column], errors="coerce")
        if numeric_ts.isna().any():
            raise DatasetPreparationError(
                f"The selected timestamp column ('{timestamp_column}') contains values that aren't "
                "valid numbers. Choose a different column, or leave timestamp unselected to use row order."
            )
        working["timestamp"] = numeric_ts
        ts_source = "detected"
    else:
        working["timestamp"] = working.groupby("experiment_id").cumcount().astype(float)
        ts_source = "row_order"

    working = working.sort_values(["experiment_id", "timestamp"], kind="stable").reset_index(drop=True)
    working["interval_index"] = working.groupby("experiment_id").cumcount()
    working["source_host"] = "uploaded"
    working["destination_host"] = "uploaded"

    n_experiments = int(working["experiment_id"].nunique())
    if len(working) < n_experiments * 2:
        raise DatasetPreparationError(
            "At least two intervals per experiment/group are required to construct a next-interval "
            "target, and this dataset does not have enough rows per group."
        )

    identifier_and_feature_columns = CSV_COLUMNS[:-1]  # everything except the (not-yet-computed) target
    rows = working[identifier_and_feature_columns].to_dict("records")

    try:
        rows_with_target = add_next_interval_target(rows, horizon=target_horizon)
    except ValueError as exc:
        raise DatasetPreparationError(str(exc)) from exc

    try:
        validate_rows(rows_with_target, identifier_and_feature_columns)
        validate_no_leakage(rows_with_target, horizon=target_horizon)
    except SchemaValidationError as exc:
        raise DatasetPreparationError(f"Dataset validation failed: {exc}") from exc

    n_before_drop = len(rows_with_target)
    labeled_rows = drop_rows_without_target(rows_with_target)
    if not labeled_rows:
        raise DatasetPreparationError(
            "After removing rows with no future interval to compare against (the last interval of "
            "each experiment/group), no usable rows remained."
        )

    prepared_df = pd.DataFrame(labeled_rows, columns=CSV_COLUMNS)
    return PreparationResult(
        prepared_df=prepared_df,
        n_experiments=n_experiments,
        n_rows_before_target_drop=n_before_drop,
        n_rows_after_target_drop=len(labeled_rows),
        group_column_source=group_source,
        timestamp_column_source=ts_source,
    )
