"""Tests for ml.split -- the group-aware temporal/random splits. These are
the most safety-critical tests in Phase 4: a bug here would let the same
kind of leakage the whole project has been careful to avoid back in
through the ML side instead of the data-generation side.
"""

import pandas as pd
import pytest

from ml.split import SplitError, assert_no_group_overlap, chronological_split, random_group_split
from ml_fixtures import make_fixture_dataframe


def _df(n_experiments=6, n_intervals=5):
    return make_fixture_dataframe(n_experiments=n_experiments, n_intervals=n_intervals)


def test_chronological_split_has_no_group_overlap():
    train_df, test_df = chronological_split(_df(), test_fraction=0.34)
    assert_no_group_overlap(train_df, test_df)  # should not raise


def test_random_group_split_has_no_group_overlap():
    train_df, test_df = random_group_split(_df(), test_fraction=0.34, random_seed=1)
    assert_no_group_overlap(train_df, test_df)  # should not raise


def test_chronological_split_test_set_is_strictly_later():
    train_df, test_df = chronological_split(_df(n_experiments=6), test_fraction=0.34)
    latest_train_start = train_df.groupby("experiment_id")["timestamp"].min().max()
    earliest_test_start = test_df.groupby("experiment_id")["timestamp"].min().min()
    assert earliest_test_start > latest_train_start


def test_random_group_split_is_deterministic_given_seed():
    df = _df()
    train_a, test_a = random_group_split(df, random_seed=7)
    train_b, test_b = random_group_split(df, random_seed=7)
    assert set(test_a["experiment_id"]) == set(test_b["experiment_id"])


def test_random_group_split_seed_changes_partition():
    df = _df(n_experiments=8)
    _, test_a = random_group_split(df, random_seed=1)
    _, test_b = random_group_split(df, random_seed=999)
    assert set(test_a["experiment_id"]) != set(test_b["experiment_id"])


def test_both_splits_use_every_row_exactly_once():
    df = _df()
    train_df, test_df = chronological_split(df, test_fraction=0.3)
    assert len(train_df) + len(test_df) == len(df)
    # Both frames get their positional index reset by design (_apply_group_split),
    # so identity must be compared by actual row content, not .index.
    train_keys = set(zip(train_df["experiment_id"], train_df["interval_index"]))
    test_keys = set(zip(test_df["experiment_id"], test_df["interval_index"]))
    assert train_keys & test_keys == set()
    assert len(train_keys) + len(test_keys) == len(df)


def test_split_rejects_too_few_groups():
    df = make_fixture_dataframe(n_experiments=1, n_intervals=4)
    with pytest.raises(SplitError):
        chronological_split(df)


def test_split_rejects_invalid_test_fraction():
    df = _df()
    with pytest.raises(SplitError):
        chronological_split(df, test_fraction=0.0)
    with pytest.raises(SplitError):
        chronological_split(df, test_fraction=1.0)


def test_assert_no_group_overlap_catches_a_deliberately_broken_split():
    df = _df()
    train_df, test_df = chronological_split(df, test_fraction=0.34)
    # Deliberately reintroduce one train row's experiment into "test" to
    # prove the guard actually detects an overlap rather than trivially passing.
    contaminated_test = pd.concat([test_df, train_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(SplitError):
        assert_no_group_overlap(train_df, contaminated_test)
