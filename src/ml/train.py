"""Training orchestration: fit every candidate model (plus the naive
baseline) on TRAIN only, predict on TEST only, compute metrics. This is
the function both scripts/train_models.py and the test suite call --
scripts/train_models.py with a real, validated dataset; tests with tiny
labeled fixtures (never presented as real performance).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml.baselines import NaivePersistenceBaseline
from ml.evaluate import compute_regression_metrics
from ml.models import MODEL_REGISTRY, ModelFactory
from network.schema import FEATURE_COLUMNS, TARGET_COLUMN


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_factories: dict[str, ModelFactory] | None = None,
    include_naive_baseline: bool = True,
    feature_columns: list[str] = FEATURE_COLUMNS,
    target_column: str = TARGET_COLUMN,
) -> dict[str, Any]:
    """Returns {
        "metrics": {model_name: {"MAE":..., "RMSE":..., "R2":...}},
        "fitted_models": {model_name: fitted_pipeline_or_baseline},
        "X_test": DataFrame, "y_test": Series, "y_pred": {model_name: array},
    }

    Every model factory is called fresh (see models.py) and .fit() is
    called on X_train/y_train ONLY; predictions come from .predict(X_test)
    ONLY -- the test set is never touched during fitting for any model,
    including the naive baseline (which is stateless but still follows
    the same fit-then-predict contract for a uniform evaluation loop).
    """
    model_factories = model_factories if model_factories is not None else MODEL_REGISTRY

    X_train, y_train = train_df[feature_columns], train_df[target_column]
    X_test, y_test = test_df[feature_columns], test_df[target_column]

    metrics: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, Any] = {}
    predictions: dict[str, Any] = {}

    if include_naive_baseline:
        baseline = NaivePersistenceBaseline().fit(X_train, y_train)
        y_pred = baseline.predict(X_test)
        metrics["naive_persistence"] = compute_regression_metrics(y_test, y_pred)
        fitted_models["naive_persistence"] = baseline
        predictions["naive_persistence"] = y_pred

    for name, factory in model_factories.items():
        pipeline = factory()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics[name] = compute_regression_metrics(y_test, y_pred)
        fitted_models[name] = pipeline
        predictions[name] = y_pred

    return {
        "metrics": metrics,
        "fitted_models": fitted_models,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
    }
