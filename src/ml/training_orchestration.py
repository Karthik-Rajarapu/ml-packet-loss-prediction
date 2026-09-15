"""Orchestrates training the EXISTING model registry (ml.models,
ml.train, ml.split, ml.evaluate) against a prepared (canonical,
leakage-validated) dataframe, and -- only on an explicit caller action,
never automatically -- promotes the result to the real production
model location, archiving any prior production artifact rather than
overwriting it blindly.

No new model, splitting, or evaluation logic lives here: this module
only sequences the existing Phase 4 functions and handles artifact
placement.
"""

from __future__ import annotations

import datetime as dt
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.artifacts import ModelMetadata, save_metadata, save_model
from ml.evaluate import comparison_table
from ml.split import SplitError, chronological_split, random_group_split
from ml.train import train_and_evaluate
from network.schema import FEATURE_COLUMNS, TARGET_COLUMN


class TrainingError(ValueError):
    """Human-readable training-pipeline failure."""


@dataclass
class TrainingRunResult:
    comparison: pd.DataFrame
    best_model_name: str
    metrics: dict[str, dict[str, float]]
    fitted_models: dict[str, Any]
    X_test: pd.DataFrame
    y_test: pd.Series
    predictions: dict[str, Any]
    split_method: str
    n_train_experiments: int
    n_test_experiments: int
    n_train_rows: int
    n_test_rows: int


def run_training(
    prepared_df: pd.DataFrame,
    split_method: str = "chronological",
    test_fraction: float = 0.3,
    random_seed: int = 42,
) -> TrainingRunResult:
    """Group-aware split (never row-level) + train_and_evaluate, exactly
    as Phase 4 defines both -- unmodified."""
    if prepared_df["experiment_id"].nunique() < 2:
        raise TrainingError(
            "At least 2 experiments/groups are required for a leakage-safe train/test split -- this "
            "dataset only has 1. Upload a dataset with multiple experiments/groups, or map a column "
            "that identifies separate groups of measurements."
        )

    try:
        if split_method == "chronological":
            train_df, test_df = chronological_split(prepared_df, test_fraction=test_fraction)
        else:
            train_df, test_df = random_group_split(prepared_df, test_fraction=test_fraction, random_seed=random_seed)
    except SplitError as exc:
        raise TrainingError(str(exc)) from exc

    result = train_and_evaluate(train_df, test_df)
    table = comparison_table(result["metrics"])
    best_name = table.iloc[0]["Model"]

    return TrainingRunResult(
        comparison=table,
        best_model_name=best_name,
        metrics=result["metrics"],
        fitted_models=result["fitted_models"],
        X_test=result["X_test"],
        y_test=result["y_test"],
        predictions=result["predictions"],
        split_method=split_method,
        n_train_experiments=int(train_df["experiment_id"].nunique()),
        n_test_experiments=int(test_df["experiment_id"].nunique()),
        n_train_rows=len(train_df),
        n_test_rows=len(test_df),
    )


def save_as_production_model(
    training_result: TrainingRunResult,
    models_dir: Path,
    dataset_filename: str,
    n_rows: int,
) -> Path:
    """Archives any existing production artifact (moved to
    models_dir/archive/, never deleted or silently overwritten), then
    saves the best-by-MAE model as THE production model.

    Always writes is_test_fixture=False -- this function must only ever
    be called from a real upload-and-train flow, never from the
    demonstration/test-fixture path (ml.demo is never imported here).
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    existing_joblib = sorted(models_dir.glob("*.joblib"))
    if existing_joblib:
        archive_dir = models_dir / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for path in existing_joblib:
            shutil.move(str(path), str(archive_dir / f"{stamp}_{path.name}"))
            metadata_path = path.with_name(path.stem + "_metadata.json")
            if metadata_path.exists():
                shutil.move(str(metadata_path), str(archive_dir / f"{stamp}_{metadata_path.name}"))

    best_name = training_result.best_model_name
    pipeline = training_result.fitted_models[best_name]
    joblib_path = save_model(pipeline, models_dir / f"{best_name}.joblib")
    save_metadata(
        ModelMetadata(
            model_name=best_name,
            feature_columns=FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
            is_test_fixture=False,
            training_timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
            training_config={
                "split_method": training_result.split_method,
                "n_train_experiments": training_result.n_train_experiments,
                "n_test_experiments": training_result.n_test_experiments,
                "source_dataset_filename": dataset_filename,
                "source_dataset_rows": n_rows,
            },
            metrics=training_result.metrics[best_name],
        ),
        models_dir / f"{best_name}_metadata.json",
    )
    return joblib_path
