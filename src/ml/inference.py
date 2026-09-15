"""Real-time-ready inference layer (Phase 6).

    current network metrics
            |
            v
    validate_feature_input()   -- reject anything malformed or leakage-suspect
            |
            v
    build_feature_frame()      -- single-row DataFrame, canonical FEATURE_COLUMNS order
            |
            v
    InferenceEngine.predict()  -- pipeline.predict() (preprocessing + model, Phase 4, unchanged)
            |
            v
    classify_risk()            -- Phase 4's existing, unmodified risk layer
            |
            v
    PredictionResult

This module NEVER trains a model -- it only loads an artifact already
produced by scripts/train_models.py (Phase 4) and runs it forward. It
never fabricates a prediction, never invents a confidence score, and
never lets a test-fixture model be mistaken for -- or silently substituted
for -- a production one. See docs/PHASE_6_INFERENCE_ENGINE.md.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.artifacts import ModelMetadata, load_metadata, load_model
from ml.baselines import NaivePersistenceBaseline
from ml.risk import DEFAULT_RISK_THRESHOLDS, RiskThresholds, classify_risk
from network.schema import FEATURE_COLUMNS, TARGET_COLUMN

# --------------------------------------------------------------------------
# Step 4: input validation
# --------------------------------------------------------------------------

CATEGORICAL_FEATURES = {"traffic_type"}
VALID_TRAFFIC_TYPES = {"udp", "tcp"}
PERCENT_BOUNDED_FEATURES = {"packet_loss_pct"}  # a percentage of packets: must be in [0, 100]

# Step 5 (leakage safety): if a caller's "extra" field looks like it might
# be target/future information, say so explicitly rather than reporting a
# generic "unexpected feature" -- this is the inference-side backstop for
# the project's core leakage rule (PROJECT_PLAN.md Section 16-17,
# network.targets.add_next_interval_target).
_LEAKAGE_SUSPECT_SUBSTRINGS = ("target", "next_interval", "next_packet_loss", "future")


class FeatureValidationError(ValueError):
    """Raised for any malformed/incomplete/leakage-suspect input. The
    inference engine never silently coerces bad input into a plausible
    value -- see docs/PHASE_6_INFERENCE_ENGINE.md Section 3-4."""


def validate_feature_input(metrics: Any) -> dict[str, Any]:
    """Validate a raw `{feature_name: value}` dict against the EXACT
    feature contract the trained model expects (network.schema.FEATURE_COLUMNS).

    Returns a new dict with values coerced to plain float/str (never
    mutates the caller's input). Raises FeatureValidationError identifying
    the specific problem feature(s) on any violation -- missing feature,
    unexpected/extra feature (including a specific message for anything
    that looks like target/future information), wrong type, NaN, infinite,
    negative where physically impossible, or a percentage outside [0, 100].
    """
    if not isinstance(metrics, dict):
        raise FeatureValidationError(f"metrics must be a dict of {{feature_name: value}}, got {type(metrics).__name__}")

    provided = set(metrics.keys())
    required = set(FEATURE_COLUMNS)

    missing = required - provided
    if missing:
        raise FeatureValidationError(f"Missing required feature(s): {sorted(missing)}")

    extra = provided - required
    if extra:
        leakage_suspects = sorted(f for f in extra if any(s in f.lower() for s in _LEAKAGE_SUSPECT_SUBSTRINGS))
        if leakage_suspects:
            raise FeatureValidationError(
                f"Refusing input: field(s) {leakage_suspects} look like target/future-interval "
                "information, which must never be supplied as a prediction input -- the inference "
                "engine only ever consumes information available at or before the prediction cutoff. "
                "See docs/PHASE_6_INFERENCE_ENGINE.md Section 5-6."
            )
        raise FeatureValidationError(f"Unexpected feature(s) not in the model's feature contract: {sorted(extra)}")

    validated: dict[str, Any] = {}
    for name in FEATURE_COLUMNS:
        value = metrics[name]

        if name in CATEGORICAL_FEATURES:
            if not isinstance(value, str):
                raise FeatureValidationError(f"'{name}' must be a string, got {type(value).__name__}: {value!r}")
            if name == "traffic_type" and value not in VALID_TRAFFIC_TYPES:
                raise FeatureValidationError(f"'{name}' must be one of {sorted(VALID_TRAFFIC_TYPES)}, got {value!r}")
            validated[name] = value
            continue

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise FeatureValidationError(f"'{name}' must be numeric, got {type(value).__name__}: {value!r}")
        numeric_value = float(value)
        if math.isnan(numeric_value):
            raise FeatureValidationError(f"'{name}' is NaN -- not a valid measurement")
        if math.isinf(numeric_value):
            raise FeatureValidationError(f"'{name}' is infinite -- not a valid measurement")
        if numeric_value < 0:
            raise FeatureValidationError(f"'{name}' cannot be negative (got {numeric_value}) -- not physically possible")
        if name in PERCENT_BOUNDED_FEATURES and numeric_value > 100.0:
            raise FeatureValidationError(f"'{name}' must be within [0, 100] (got {numeric_value})")
        validated[name] = numeric_value

    return validated


def build_feature_frame(validated_metrics: dict[str, Any]) -> pd.DataFrame:
    """Single-row DataFrame in CANONICAL FEATURE_COLUMNS order.

    sklearn's ColumnTransformer (ml.preprocessing.build_preprocessor)
    selects columns by NAME, not position, so this ordering is not
    strictly required for correctness today -- but building it explicitly
    keeps the feature contract self-evident and protects against a future
    change to positional selection. validate_feature_input() must always
    run first; this function assumes its input is already validated.
    """
    return pd.DataFrame([[validated_metrics[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)


# --------------------------------------------------------------------------
# Step 6: model artifact loading
# --------------------------------------------------------------------------

class ModelArtifactError(RuntimeError):
    """Raised for a missing, ambiguous, or incompatible model artifact --
    covers both 'no production model exists' and 'this artifact doesn't
    match the current feature/target contract'."""


PRODUCTION_MODEL_UNAVAILABLE_MESSAGE = (
    "No production model is available. Train a model using a validated real "
    "network dataset first (see docs/PHASE_5_REAL_DATA_VALIDATION.md and "
    "run scripts/train_models.py against real Mininet-generated data)."
)


@dataclass(frozen=True)
class ModelHandle:
    pipeline: Any
    metadata: ModelMetadata
    source_path: Path
    is_test_fixture: bool = False


def _validate_feature_compatibility(metadata: ModelMetadata) -> None:
    if list(metadata.feature_columns) != list(FEATURE_COLUMNS):
        raise ModelArtifactError(
            "Model artifact's feature_columns do not match the current schema.FEATURE_COLUMNS -- "
            "this model was trained against a different feature contract and must not be used for "
            f"inference without retraining. model={metadata.feature_columns!r} current={FEATURE_COLUMNS!r}"
        )
    if metadata.target_column != TARGET_COLUMN:
        raise ModelArtifactError(
            f"Model artifact's target_column ({metadata.target_column!r}) does not match the current "
            f"target definition ({TARGET_COLUMN!r})."
        )


def load_model_from_path(joblib_path: Path, expected_test_fixture: bool) -> ModelHandle:
    """Load one specific model + its metadata sidecar, enforcing:
      - the artifact and its metadata both exist
      - the feature/target contract matches the current schema
      - the metadata's own is_test_fixture flag matches what the CALLER
        expects to be loading (production loader passes False, test-fixture
        loader passes True) -- this is the structural guard against ever
        mixing the two up (Strict Rule #4/#8).
    """
    if not joblib_path.exists():
        raise ModelArtifactError(f"Model artifact not found: {joblib_path}")

    metadata_path = joblib_path.with_name(joblib_path.stem + "_metadata.json")
    if not metadata_path.exists():
        raise ModelArtifactError(f"Model metadata not found: {metadata_path} (expected alongside {joblib_path})")

    metadata = load_metadata(metadata_path)
    _validate_feature_compatibility(metadata)

    if metadata.is_test_fixture != expected_test_fixture:
        if expected_test_fixture:
            raise ModelArtifactError(
                f"{joblib_path} is not marked is_test_fixture=True in its metadata -- refusing to load "
                "it through the test-fixture loader."
            )
        raise ModelArtifactError(
            f"{joblib_path} is marked is_test_fixture=True in its metadata -- refusing to load it as a "
            "production model. Test fixtures must never be used for real predictions (Strict Rule #4)."
        )

    pipeline = load_model(joblib_path)
    return ModelHandle(pipeline=pipeline, metadata=metadata, source_path=joblib_path,
                        is_test_fixture=metadata.is_test_fixture)


def load_production_model(models_dir: Path) -> ModelHandle:
    """Load THE production model from `models_dir` (normally the repo's
    models/ directory, i.e. scripts/train_models.py's output).

    NEVER falls back to a test fixture. If no production artifact exists,
    raises ModelArtifactError with the exact message specified in the
    Phase 6 brief -- callers (the CLI) should catch this and print it,
    not treat it as an unexpected crash.
    """
    if not models_dir.exists():
        raise ModelArtifactError(PRODUCTION_MODEL_UNAVAILABLE_MESSAGE)

    joblib_paths = sorted(models_dir.glob("*.joblib"))
    if not joblib_paths:
        raise ModelArtifactError(PRODUCTION_MODEL_UNAVAILABLE_MESSAGE)
    if len(joblib_paths) > 1:
        raise ModelArtifactError(
            f"Multiple candidate model artifacts found in {models_dir}: "
            f"{[p.name for p in joblib_paths]}. Load one explicitly via load_model_from_path() instead."
        )

    return load_model_from_path(joblib_paths[0], expected_test_fixture=False)


def load_test_fixture_model(joblib_path: Path) -> ModelHandle:
    """Load a model explicitly expected to be a labeled TEST FIXTURE.
    Raises if the artifact is not actually marked as one -- see Strict
    Rule #4: a production-looking artifact can never be loaded through
    this path by accident."""
    return load_model_from_path(joblib_path, expected_test_fixture=True)


# --------------------------------------------------------------------------
# Steps 2, 7, 8, 9: prediction result + engine + baseline
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PredictionResult:
    predicted_packet_loss_pct: float
    risk_level: str
    model_name: str
    model_kind: str  # "ml_model" | "naive_baseline"
    feature_names: list[str]
    predicted_at: str
    is_test_fixture: bool = False
    model_metadata: dict[str, Any] | None = None
    # Deliberately NO confidence/uncertainty field -- Step 9: this model
    # family provides no calibrated uncertainty, so none is invented.

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class InferenceEngine:
    """Wraps one loaded ModelHandle (Step 6) and exposes the primary ML
    prediction path (Step 2). The naive-baseline path (Step 7) is
    deliberately a separate module-level function below, not a method
    here, since it needs no trained artifact at all -- keeping the two
    call shapes visibly different helps prevent accidentally mixing them.
    """

    def __init__(self, model_handle: ModelHandle, risk_thresholds: RiskThresholds = DEFAULT_RISK_THRESHOLDS):
        self._handle = model_handle
        self._risk_thresholds = risk_thresholds

    @property
    def model_handle(self) -> ModelHandle:
        return self._handle

    def predict(self, current_network_metrics: dict[str, Any]) -> PredictionResult:
        """predict_next_packet_loss(current_network_metrics), conceptually.
        Raises FeatureValidationError on bad input -- never silently
        substitutes a plausible-looking value for a bad measurement."""
        validated = validate_feature_input(current_network_metrics)
        frame = build_feature_frame(validated)
        predicted = float(self._handle.pipeline.predict(frame)[0])
        risk = str(classify_risk([predicted], thresholds=self._risk_thresholds).iloc[0])
        return PredictionResult(
            predicted_packet_loss_pct=predicted,
            risk_level=risk,
            model_name=self._handle.metadata.model_name,
            model_kind="ml_model",
            feature_names=list(FEATURE_COLUMNS),
            predicted_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            is_test_fixture=self._handle.is_test_fixture,
            model_metadata=asdict(self._handle.metadata),
        )


def predict_naive_baseline(
    current_network_metrics: dict[str, Any],
    risk_thresholds: RiskThresholds = DEFAULT_RISK_THRESHOLDS,
) -> PredictionResult:
    """Naive persistence baseline: predicted_next_loss = current
    packet_loss_pct (ml.baselines.NaivePersistenceBaseline, unchanged).

    This is the simple temporal reference point, NOT the primary ML
    inference path (Step 7) -- callers must not confuse
    model_kind="naive_baseline" results with model_kind="ml_model" ones.
    Needs no trained artifact at all, so it is always available even
    when NO production model exists.
    """
    validated = validate_feature_input(current_network_metrics)
    frame = build_feature_frame(validated)
    baseline = NaivePersistenceBaseline()
    predicted = float(baseline.predict(frame).iloc[0])
    risk = str(classify_risk([predicted], thresholds=risk_thresholds).iloc[0])
    return PredictionResult(
        predicted_packet_loss_pct=predicted,
        risk_level=risk,
        model_name="naive_persistence",
        model_kind="naive_baseline",
        feature_names=list(FEATURE_COLUMNS),
        predicted_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        is_test_fixture=False,
        model_metadata=None,
    )
