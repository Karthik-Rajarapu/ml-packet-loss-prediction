#!/usr/bin/env python3
"""Phase 6 CLI: run one next-interval packet-loss prediction from a
current-network-metrics input.

    # Naive baseline (no trained model needed):
    python3 scripts/predict_packet_loss.py --demo --baseline

    # Production ML model (requires a real model in models/, trained on
    # real Mininet data -- fails clearly, not silently, if none exists):
    python3 scripts/predict_packet_loss.py --metrics-json my_metrics.json

    # TEST FIXTURE demonstration (explicitly labeled in the output,
    # never presented as a real project result):
    python3 scripts/predict_packet_loss.py --demo --use-test-fixture

This script never trains a model and never fabricates a prediction --
see src/ml/inference.py and docs/PHASE_6_INFERENCE_ENGINE.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ml.demo import DEMO_CURRENT_METRICS, build_test_fixture_model  # noqa: E402
from ml.inference import (  # noqa: E402
    FeatureValidationError,
    InferenceEngine,
    ModelArtifactError,
    load_production_model,
    predict_naive_baseline,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--demo", action="store_true",
                              help="Use a small built-in, clearly-synthetic example metrics dict")
    input_group.add_argument("--metrics-json", type=Path, help="Path to a JSON file of {feature_name: value}")
    input_group.add_argument("--metrics", type=str, help="Inline JSON string of {feature_name: value}")

    parser.add_argument("--models-dir", type=Path, default=REPO_ROOT / "models",
                         help="Production model directory (default: models/)")
    parser.add_argument("--baseline", action="store_true",
                         help="Use the naive persistence baseline instead of a trained ML model")
    parser.add_argument("--use-test-fixture", action="store_true",
                         help="Use a freshly-built TEST FIXTURE model instead of a production model "
                              "(output is explicitly labeled as such, never a real project result)")
    return parser.parse_args(argv)


def load_metrics(args: argparse.Namespace) -> dict:
    if args.demo:
        return dict(DEMO_CURRENT_METRICS)
    if args.metrics_json is not None:
        return json.loads(args.metrics_json.read_text())
    return json.loads(args.metrics)


def run(args: argparse.Namespace) -> int:
    try:
        metrics = load_metrics(args)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Failed to read input metrics: {exc}")
        return 1

    if args.baseline:
        try:
            result = predict_naive_baseline(metrics)
        except FeatureValidationError as exc:
            print(f"Invalid input metrics: {exc}")
            return 1
        _print_result(result, header="NAIVE PERSISTENCE BASELINE (not the primary ML path)")
        return 0

    if args.use_test_fixture:
        with tempfile.TemporaryDirectory(prefix="packet_loss_test_fixture_model_") as tmp_dir:
            handle = build_test_fixture_model(Path(tmp_dir))
            engine = InferenceEngine(handle)
            try:
                result = engine.predict(metrics)
            except FeatureValidationError as exc:
                print(f"Invalid input metrics: {exc}")
                return 1
        _print_result(result, header="TEST FIXTURE DEMONSTRATION -- NOT REAL NETWORK PERFORMANCE")
        return 0

    try:
        handle = load_production_model(args.models_dir)
    except ModelArtifactError as exc:
        print(str(exc))
        return 1

    engine = InferenceEngine(handle)
    try:
        result = engine.predict(metrics)
    except FeatureValidationError as exc:
        print(f"Invalid input metrics: {exc}")
        return 1

    _print_result(result, header="PRODUCTION MODEL PREDICTION")
    return 0


def _print_result(result, header: str) -> None:
    print("=" * 70)
    print(header)
    print("=" * 70)
    if result.is_test_fixture:
        print("*** TEST FIXTURE MODEL -- NOT TRAINED ON REAL NETWORK DATA ***")
    print(f"Predicted next-interval packet loss: {result.predicted_packet_loss_pct:.4f}%")
    print(f"Risk: {result.risk_level}")
    print(f"Model: {result.model_name} ({result.model_kind})")
    print(f"Predicted at: {result.predicted_at}")
    if result.model_metadata:
        metrics = result.model_metadata.get("metrics")
        if metrics:
            print(f"Model's own reported evaluation metrics: {metrics}")
        training_timestamp = result.model_metadata.get("training_timestamp")
        if training_timestamp:
            print(f"Model trained at: {training_timestamp}")


def main() -> int:
    args = parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
