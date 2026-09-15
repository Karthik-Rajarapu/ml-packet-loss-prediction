#!/usr/bin/env python3
"""Phase 4 CLI: train and evaluate regression models on a REAL Phase 1/2
dataset. Refuses to run -- writes nothing -- if no real dataset exists.

    python3 scripts/train_models.py --input data/raw

    python3 scripts/train_models.py --input data/raw --split-method random_group --test-fraction 0.3

This script never trains on synthetic/fixture data. If data/raw/ has no
CSVs (the case on this development machine as of Phase 4), it prints a
clear message and exits non-zero without producing any comparison table,
plot, or model artifact.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ml.artifacts import ModelMetadata, save_metadata, save_model  # noqa: E402
from ml.dataset import DatasetError, discover_csv_files, load_dataset  # noqa: E402
from ml.evaluate import (  # noqa: E402
    comparison_table,
    permutation_feature_importance,
    random_forest_feature_importance,
    residual_stats,
)
from ml.risk import DEFAULT_RISK_THRESHOLDS, classify_risk  # noqa: E402
from ml.split import SplitError, chronological_split, random_group_split  # noqa: E402
from ml.train import train_and_evaluate  # noqa: E402
from ml.visualize import (  # noqa: E402
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_residual_distribution,
)
from network.schema import FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data" / "raw",
                         help="Directory of experiment CSVs, or a single CSV file")
    parser.add_argument("--split-method", choices=["chronological", "random_group"], default="chronological")
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--models-dir", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--reports-dir", type=Path, default=REPO_ROOT / "reports" / "modeling")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    csv_paths = discover_csv_files(args.input)
    if not csv_paths:
        print(f"No CSV files found under {args.input}.")
        print("No real Mininet-generated dataset exists yet -- refusing to train on nothing.")
        print("Run scripts/generate_dataset.py against a real Mininet environment first "
              "(see docs/PHASE_2_DATA_GENERATION.md and docs/ENVIRONMENT_SETUP.md).")
        return 1

    try:
        df = load_dataset(csv_paths)
    except DatasetError as exc:
        print(f"Dataset failed to load: {exc}")
        return 1

    print(f"Loaded {len(df)} labeled rows from {len(csv_paths)} experiment file(s), "
          f"{df['experiment_id'].nunique()} experiment(s).")

    split_fn = chronological_split if args.split_method == "chronological" else random_group_split
    try:
        if args.split_method == "chronological":
            train_df, test_df = split_fn(df, test_fraction=args.test_fraction)
        else:
            train_df, test_df = split_fn(df, test_fraction=args.test_fraction, random_seed=args.random_seed)
    except SplitError as exc:
        print(f"Cannot split dataset: {exc}")
        return 1

    print(f"Split ({args.split_method}): {train_df['experiment_id'].nunique()} train experiments "
          f"({len(train_df)} rows), {test_df['experiment_id'].nunique()} test experiments ({len(test_df)} rows).")

    result = train_and_evaluate(train_df, test_df)
    table = comparison_table(result["metrics"])
    print()
    print(table.to_string(index=False))

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.reports_dir / "model_comparison.csv", index=False)

    best_name = table.iloc[0]["Model"]
    print(f"\nLowest-MAE model: {best_name} (see docs/PHASE_4_ML_MODELING.md Section 12 -- "
          f"this is a starting point, not an automatic final selection).")

    y_test = result["y_test"]
    y_pred_best = result["predictions"][best_name]
    print("Residual stats (best-by-MAE model):", residual_stats(y_test, y_pred_best))

    plot_actual_vs_predicted(y_test, y_pred_best, args.reports_dir / f"{best_name}_actual_vs_predicted.png")
    plot_residual_distribution(y_test, y_pred_best, args.reports_dir / f"{best_name}_residuals.png")

    best_pipeline = result["fitted_models"][best_name]
    if hasattr(best_pipeline.named_steps.get("model"), "feature_importances_"):
        importance_df = random_forest_feature_importance(best_pipeline)
        importance_df.to_csv(args.reports_dir / f"{best_name}_feature_importance.csv", index=False)
        plot_feature_importance(importance_df, args.reports_dir / f"{best_name}_feature_importance.png")

        perm_df = permutation_feature_importance(best_pipeline, result["X_test"], y_test, random_state=args.random_seed)
        perm_df.to_csv(args.reports_dir / f"{best_name}_permutation_importance.csv", index=False)

    risk_labels = classify_risk(y_pred_best)
    print(f"Predicted risk distribution (best model, thresholds={DEFAULT_RISK_THRESHOLDS}): "
          f"{risk_labels.value_counts().to_dict()}")

    save_model(best_pipeline, args.models_dir / f"{best_name}.joblib")
    save_metadata(
        ModelMetadata(
            model_name=best_name,
            feature_columns=FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
            is_test_fixture=False,
            training_timestamp=datetime.now(timezone.utc).isoformat(),
            training_config={
                "split_method": args.split_method,
                "test_fraction": args.test_fraction,
                "random_seed": args.random_seed,
                "n_train_experiments": int(train_df["experiment_id"].nunique()),
                "n_test_experiments": int(test_df["experiment_id"].nunique()),
            },
            metrics=result["metrics"][best_name],
        ),
        args.models_dir / f"{best_name}_metadata.json",
    )
    print(f"\nSaved model + metadata to {args.models_dir}, reports to {args.reports_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
