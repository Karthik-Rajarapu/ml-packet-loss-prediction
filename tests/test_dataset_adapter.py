"""Tests for ml.dataset_adapter -- the piece that turns a column-mapped
upload into the canonical schema and builds the leakage-safe target by
calling network.targets/network.validation UNCHANGED. These tests
specifically try to catch a leakage regression, since this is new glue
code sitting directly in front of the project's most safety-critical
logic.
"""

import pandas as pd
import pytest

from ml.dataset_adapter import DatasetPreparationError, prepare_canonical_dataset
from network.schema import FEATURE_COLUMNS, TARGET_COLUMN


def _mapped_df(n_groups=3, n_per_group=6, include_group_col=True, include_ts_col=True) -> pd.DataFrame:
    rows = []
    for g in range(n_groups):
        for i in range(n_per_group):
            row = {name: 0.0 for name in FEATURE_COLUMNS}
            row["traffic_type"] = "udp"
            row["current_rtt_ms"] = 5.0 + i
            row["packet_loss_pct"] = float(i)  # rising within each group -> real variation
            if include_group_col:
                row["exp"] = f"group-{g}"
            if include_ts_col:
                row["ts"] = g * 100.0 + i
            rows.append(row)
    return pd.DataFrame(rows)


def test_prepare_canonical_dataset_happy_path():
    df = _mapped_df()
    result = prepare_canonical_dataset(df, group_column="exp", timestamp_column="ts")
    assert TARGET_COLUMN in result.prepared_df.columns
    assert result.prepared_df[TARGET_COLUMN].isna().sum() == 0
    assert result.n_experiments == 3
    assert result.group_column_source == "detected"
    assert result.timestamp_column_source == "detected"


def test_prepare_canonical_dataset_without_group_column_uses_single_experiment():
    df = _mapped_df(n_groups=1, n_per_group=8, include_group_col=False)
    result = prepare_canonical_dataset(df, group_column=None, timestamp_column=None)
    assert result.group_column_source == "single_experiment"
    assert result.n_experiments == 1
    assert set(result.prepared_df["experiment_id"]) == {"uploaded-dataset"}


def test_prepare_canonical_dataset_without_timestamp_uses_row_order():
    df = _mapped_df(include_ts_col=False)
    result = prepare_canonical_dataset(df, group_column="exp", timestamp_column=None)
    assert result.timestamp_column_source == "row_order"


def test_missing_required_feature_raises_human_readable_error():
    df = _mapped_df().drop(columns=["current_rtt_ms"])
    with pytest.raises(DatasetPreparationError, match="missing required network measurements"):
        prepare_canonical_dataset(df, group_column="exp", timestamp_column="ts")


def test_invalid_timestamp_column_raises_human_readable_error():
    df = _mapped_df()
    df["ts"] = ["not", "a", "number"] * (len(df) // 3)
    with pytest.raises(DatasetPreparationError, match="valid numbers"):
        prepare_canonical_dataset(df, group_column="exp", timestamp_column="ts")


def test_target_never_crosses_experiment_boundary():
    """The core leakage-safety property this adapter must preserve --
    exercised through the adapter's own group construction, not just
    the underlying targets.py unit tests."""
    df = _mapped_df(n_groups=2, n_per_group=4)
    # make the two groups' packet_loss_pct ranges obviously distinct
    df.loc[df["exp"] == "group-0", "packet_loss_pct"] = [1.0, 2.0, 3.0, 4.0]
    df.loc[df["exp"] == "group-1", "packet_loss_pct"] = [90.0, 91.0, 92.0, 93.0]

    result = prepare_canonical_dataset(df, group_column="exp", timestamp_column="ts")
    prepared = result.prepared_df

    group0 = prepared[prepared["experiment_id"] == "group-0"].sort_values("interval_index")
    # last row of group-0 must have been dropped (no future data) or, if
    # present, its target must never equal a group-1 value
    for _, row in group0.iterrows():
        assert row[TARGET_COLUMN] < 50.0, "group-0's target leaked a group-1 value"


def test_too_few_rows_per_experiment_raises():
    df = _mapped_df(n_groups=5, n_per_group=1, include_ts_col=True)
    with pytest.raises(DatasetPreparationError, match="two intervals"):
        prepare_canonical_dataset(df, group_column="exp", timestamp_column="ts")


def test_prepared_dataframe_drops_end_of_experiment_rows():
    df = _mapped_df(n_groups=1, n_per_group=6, include_group_col=False)
    result = prepare_canonical_dataset(df, group_column=None, timestamp_column=None)
    assert result.n_rows_after_target_drop == result.n_rows_before_target_drop - 1
