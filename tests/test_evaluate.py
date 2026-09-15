"""Tests for ml.evaluate -- metric math against known values, comparison
table sorting, and feature importance (native RF + permutation) shape
and correctness on TEST FIXTURE data.
"""

import numpy as np

from ml.evaluate import (
    comparison_table,
    compute_regression_metrics,
    permutation_feature_importance,
    random_forest_feature_importance,
    residual_stats,
)
from ml.models import MODEL_REGISTRY
from ml.preprocessing import CATEGORICAL_FEATURE_COLUMNS
from ml_fixtures import make_labeled_fixture_dataframe
from network.schema import FEATURE_COLUMNS, TARGET_COLUMN


def test_compute_regression_metrics_known_values():
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.0, 2.0, 3.0, 4.0]
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["R2"] == 1.0


def test_compute_regression_metrics_known_error():
    y_true = [0.0, 0.0, 0.0, 0.0]
    y_pred = [1.0, -1.0, 1.0, -1.0]
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["MAE"] == 1.0
    assert metrics["RMSE"] == 1.0


def test_comparison_table_sorted_by_mae_ascending():
    results = {
        "worse": {"MAE": 5.0, "RMSE": 6.0, "R2": 0.1},
        "better": {"MAE": 1.0, "RMSE": 2.0, "R2": 0.9},
    }
    table = comparison_table(results)
    assert list(table["Model"]) == ["better", "worse"]
    assert list(table.columns) == ["Model", "MAE", "RMSE", "R2"]


def test_residual_stats_reports_expected_keys_and_symmetry():
    y_true = [10.0, 10.0, 10.0]
    y_pred = [8.0, 12.0, 10.0]  # residuals: +2, -2, 0 -> abs residuals sorted: 0, 2, 2
    stats = residual_stats(y_true, y_pred)
    assert stats["mean_residual"] == 0.0
    assert stats["median_abs_error"] == 2.0
    assert stats["max_abs_error"] == 2.0
    assert set(stats) == {
        "mean_residual", "std_residual", "median_abs_error", "max_abs_error",
        "pct_within_1pt", "pct_within_5pt",
    }


def _fitted_random_forest_and_test_split():
    df = make_labeled_fixture_dataframe(n_experiments=6, n_intervals=5)
    experiments = sorted(df["experiment_id"].unique())
    train_df = df[df["experiment_id"].isin(experiments[:4])]
    test_df = df[df["experiment_id"].isin(experiments[4:])]
    pipeline = MODEL_REGISTRY["random_forest"]()
    pipeline.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])
    return pipeline, test_df


def test_random_forest_feature_importance_shape_and_range():
    pipeline, _ = _fitted_random_forest_and_test_split()
    table = random_forest_feature_importance(pipeline)
    assert set(table.columns) == {"feature", "importance"}
    assert len(table) >= len(FEATURE_COLUMNS) - len(CATEGORICAL_FEATURE_COLUMNS)  # one-hot expands categoricals
    assert np.all(table["importance"] >= 0)
    assert np.isclose(table["importance"].sum(), 1.0, atol=1e-6)
    # sorted descending
    assert list(table["importance"]) == sorted(table["importance"], reverse=True)


def test_random_forest_feature_importance_rejects_non_tree_model():
    pipeline = MODEL_REGISTRY["linear_regression"]()
    df = make_labeled_fixture_dataframe(n_experiments=2, n_intervals=4)
    pipeline.fit(df[FEATURE_COLUMNS], df[TARGET_COLUMN])
    try:
        random_forest_feature_importance(pipeline)
        raised = False
    except TypeError:
        raised = True
    assert raised


def test_permutation_importance_runs_on_held_out_test_only():
    pipeline, test_df = _fitted_random_forest_and_test_split()
    table = permutation_feature_importance(
        pipeline, test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN], n_repeats=3, random_state=0,
    )
    assert set(table["feature"]) == set(FEATURE_COLUMNS)
    assert len(table) == len(FEATURE_COLUMNS)
