"""Group-aware train/test splitting.

Every split here operates on WHOLE experiment_id groups, never on
individual rows -- PROJECT_PLAN.md Section 17 and this project's
recurring leakage rule both require this, since consecutive intervals
within one experiment are highly autocorrelated (same configured
conditions, evolving queue state). A row-level `train_test_split(...,
random_state=...)` would let a model see rows from the same run's local
trajectory on both sides of the split -- exactly what this module exists
to prevent.
"""

from __future__ import annotations

import random

import pandas as pd


class SplitError(ValueError):
    pass


def chronological_split(
    df: pd.DataFrame,
    group_col: str = "experiment_id",
    order_col: str = "timestamp",
    test_fraction: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on earlier experiments, test on later ones (by each group's
    earliest timestamp) -- simulates deploying on past data and predicting
    on future data, closer to real-world use than a random split.
    """
    _validate_fraction(test_fraction)
    group_starts = df.groupby(group_col)[order_col].min().sort_values()
    n_groups = len(group_starts)
    if n_groups < 2:
        raise SplitError(f"Need at least 2 experiment groups to split, got {n_groups}")

    n_test = max(1, round(n_groups * test_fraction))
    n_test = min(n_test, n_groups - 1)  # always leave at least one group for training
    test_groups = set(group_starts.index[-n_test:])
    return _apply_group_split(df, group_col, test_groups)


def random_group_split(
    df: pd.DataFrame,
    group_col: str = "experiment_id",
    test_fraction: float = 0.3,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Randomly hold out whole experiment groups (fixed seed for
    reproducibility) -- tests generalization across configurations rather
    than across time.
    """
    _validate_fraction(test_fraction)
    groups = sorted(df[group_col].unique())
    n_groups = len(groups)
    if n_groups < 2:
        raise SplitError(f"Need at least 2 experiment groups to split, got {n_groups}")

    rng = random.Random(random_seed)
    shuffled = groups[:]
    rng.shuffle(shuffled)

    n_test = max(1, round(n_groups * test_fraction))
    n_test = min(n_test, n_groups - 1)
    test_groups = set(shuffled[:n_test])
    return _apply_group_split(df, group_col, test_groups)


def _validate_fraction(test_fraction: float) -> None:
    if not (0.0 < test_fraction < 1.0):
        raise SplitError(f"test_fraction must be in (0, 1), got {test_fraction}")


def _apply_group_split(df: pd.DataFrame, group_col: str, test_groups: set) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_mask = df[group_col].isin(test_groups)
    train_df = df.loc[~test_mask].reset_index(drop=True)
    test_df = df.loc[test_mask].reset_index(drop=True)
    assert_no_group_overlap(train_df, test_df, group_col)
    return train_df, test_df


def assert_no_group_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame, group_col: str = "experiment_id") -> None:
    """Defense-in-depth: verify zero experiment_id overlap between the two
    sets. Cheap to check, and this is the single property that makes the
    split leakage-safe -- worth asserting explicitly rather than trusting
    the set-difference logic above silently.
    """
    overlap = set(train_df[group_col]) & set(test_df[group_col])
    if overlap:
        raise SplitError(f"Train/test split leaked group(s) across the boundary: {overlap}")
