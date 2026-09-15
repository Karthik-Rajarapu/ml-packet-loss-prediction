"""Metric computation, model comparison, and feature importance.

Kept separate from train.py so metric math can be unit-tested against
known values independent of any model-fitting machinery.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


def compute_regression_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE, R^2 -- the three primary metrics this project standardizes on."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def comparison_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """{"model_name": {"MAE":..., "RMSE":..., "R2":...}} -> a sorted-by-MAE
    DataFrame with columns Model, MAE, RMSE, R2. Sorted by MAE (not R2)
    as the default ordering -- "do not select a model using only R2" --
    callers deciding the FINAL model should still weigh the other
    documented criteria (Section 12), this ordering is a starting point.
    """
    table = pd.DataFrame.from_dict(results, orient="index")
    table.index.name = "Model"
    table = table.reset_index().sort_values("MAE").reset_index(drop=True)
    return table[["Model", "MAE", "RMSE", "R2"]]


def residual_stats(y_true, y_pred) -> dict[str, float]:
    """Distribution of prediction errors, not just the aggregate metrics --
    important here because packet loss is often near-zero (many
    low-loss observations), so R2/MAE alone can hide how errors are
    distributed (e.g. mostly tiny with a few large misses vs. uniformly
    mediocre).
    """
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    return {
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals)),
        "median_abs_error": float(np.median(np.abs(residuals))),
        "max_abs_error": float(np.max(np.abs(residuals))),
        "pct_within_1pt": float(np.mean(np.abs(residuals) <= 1.0) * 100.0),
        "pct_within_5pt": float(np.mean(np.abs(residuals) <= 5.0) * 100.0),
    }


def random_forest_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Native impurity-based importance from a fitted RandomForestRegressor
    pipeline. Uses the preprocessor's own get_feature_names_out() so
    importances are correctly labeled in POST-transform space (e.g. the
    one-hot-expanded traffic_type columns), not silently misattributed to
    the original raw column list.
    """
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        raise TypeError(f"{type(model).__name__} has no feature_importances_ -- not a tree-based model")

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    table = pd.DataFrame({"feature": feature_names, "importance": importances})
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def permutation_feature_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Permutation importance computed on already-held-out TEST data only
    (X_test/y_test come from a group-aware split -- see split.py), so this
    never touches training-only or future-interval information. Operates
    on the RAW feature columns (not the one-hot-expanded ones), since
    sklearn shuffles each input column of X_test before it reaches the
    pipeline's own preprocessing step.
    """
    result = permutation_importance(
        pipeline, X_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=-1,
    )
    table = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)
