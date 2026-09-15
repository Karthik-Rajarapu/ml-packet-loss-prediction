"""Tests for ml.artifacts -- save/load round-trips for both the fitted
pipeline (joblib) and its metadata (JSON), verified with TEST FIXTURE
data, and a check that no unexpected (e.g. secret-shaped) keys sneak
into the saved metadata.
"""

import numpy as np

from ml.artifacts import ModelMetadata, load_metadata, load_model, save_metadata, save_model
from ml.models import MODEL_REGISTRY
from ml_fixtures import make_labeled_fixture_dataframe
from network.schema import FEATURE_COLUMNS, TARGET_COLUMN


def _fitted_pipeline_and_test_data():
    df = make_labeled_fixture_dataframe(n_experiments=4, n_intervals=5)
    experiments = sorted(df["experiment_id"].unique())
    train_df = df[df["experiment_id"].isin(experiments[:3])]
    test_df = df[df["experiment_id"].isin(experiments[3:])]
    pipeline = MODEL_REGISTRY["decision_tree"]()
    pipeline.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])
    return pipeline, test_df


def test_save_and_load_model_predictions_match(tmp_path):
    pipeline, test_df = _fitted_pipeline_and_test_data()
    original_predictions = pipeline.predict(test_df[FEATURE_COLUMNS])

    path = save_model(pipeline, tmp_path / "model.joblib")
    assert path.exists()

    reloaded = load_model(path)
    reloaded_predictions = reloaded.predict(test_df[FEATURE_COLUMNS])

    assert np.array_equal(original_predictions, reloaded_predictions), \
        "a reloaded model must produce IDENTICAL predictions to the original"


def test_save_and_load_metadata_roundtrip(tmp_path):
    metadata = ModelMetadata(
        model_name="decision_tree",
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
        training_config={"split_method": "chronological", "test_fraction": 0.3, "random_seed": 42},
        metrics={"MAE": 1.23, "RMSE": 2.34, "R2": 0.5},
    )
    path = save_metadata(metadata, tmp_path / "meta.json")
    reloaded = load_metadata(path)

    assert reloaded == metadata


def test_saved_metadata_contains_no_unexpected_keys(tmp_path):
    """A crude guard against accidentally serializing something outside
    the documented ModelMetadata fields (e.g. an env var or secret)."""
    metadata = ModelMetadata(
        model_name="decision_tree", feature_columns=["a", "b"], target_column="t",
    )
    path = save_metadata(metadata, tmp_path / "meta.json")
    import json
    raw = json.loads(path.read_text())
    assert set(raw.keys()) == {
        "model_name", "feature_columns", "target_column", "training_config", "metrics",
        "is_test_fixture", "training_timestamp",
    }
