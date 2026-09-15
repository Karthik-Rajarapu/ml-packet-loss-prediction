"""Model artifact persistence: the fitted pipeline (joblib) plus a JSON
metadata sidecar (feature list, target, training config, evaluation
metrics). No secrets or environment variables are ever written here --
metadata is limited to the fields listed in ModelMetadata below.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline


@dataclass
class ModelMetadata:
    model_name: str
    feature_columns: list[str]
    target_column: str
    training_config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


def save_model(pipeline: Pipeline, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_model(path: Path) -> Pipeline:
    return joblib.load(path)


def save_metadata(metadata: ModelMetadata, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(metadata), f, indent=2)
    return path


def load_metadata(path: Path) -> ModelMetadata:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return ModelMetadata(**data)
