"""Tests for ml.inference's model-artifact loading -- Step 6 and the
strict rule "never silently replace a missing production model with a
test model." Every "production model" constructed in these tests lives
under tmp_path, never the real repo models/ directory.
"""

from pathlib import Path

import pytest

from ml.artifacts import ModelMetadata, save_metadata, save_model
from ml.demo import build_test_fixture_model
from ml.inference import (
    PRODUCTION_MODEL_UNAVAILABLE_MESSAGE,
    ModelArtifactError,
    load_model_from_path,
    load_production_model,
    load_test_fixture_model,
)
from ml.models import MODEL_REGISTRY
from ml_fixtures import make_labeled_fixture_dataframe
from network.schema import FEATURE_COLUMNS, TARGET_COLUMN


def _train_and_save_fake_production_model(models_dir: Path, name: str = "decision_tree") -> None:
    """A model that LOOKS production-shaped (is_test_fixture=False) for
    testing the loader mechanics only -- trained on the synthetic fixture
    for speed, never claimed anywhere as a real result. Always written
    under a tmp_path 'models_dir', never the real repo models/."""
    df = make_labeled_fixture_dataframe(n_experiments=3, n_intervals=5)
    pipeline = MODEL_REGISTRY[name]()
    pipeline.fit(df[FEATURE_COLUMNS], df[TARGET_COLUMN])
    save_model(pipeline, models_dir / f"{name}.joblib")
    save_metadata(
        ModelMetadata(model_name=name, feature_columns=FEATURE_COLUMNS, target_column=TARGET_COLUMN,
                       is_test_fixture=False),
        models_dir / f"{name}_metadata.json",
    )


def test_load_production_model_empty_dir_raises_exact_message(tmp_path):
    with pytest.raises(ModelArtifactError, match="No production model is available"):
        load_production_model(tmp_path)
    # exact text match too, since the CLI prints this verbatim
    try:
        load_production_model(tmp_path)
    except ModelArtifactError as exc:
        assert str(exc) == PRODUCTION_MODEL_UNAVAILABLE_MESSAGE


def test_load_production_model_nonexistent_dir_raises_same_message(tmp_path):
    with pytest.raises(ModelArtifactError, match="No production model is available"):
        load_production_model(tmp_path / "does_not_exist")


def test_load_production_model_succeeds_with_one_valid_artifact(tmp_path):
    _train_and_save_fake_production_model(tmp_path)
    handle = load_production_model(tmp_path)
    assert handle.is_test_fixture is False
    assert handle.metadata.model_name == "decision_tree"


def test_load_production_model_rejects_multiple_candidates(tmp_path):
    _train_and_save_fake_production_model(tmp_path, name="decision_tree")
    _train_and_save_fake_production_model(tmp_path, name="random_forest")
    with pytest.raises(ModelArtifactError, match="Multiple candidate"):
        load_production_model(tmp_path)


def test_load_production_model_never_falls_back_to_a_fixture_marked_artifact(tmp_path):
    """If the only artifact present is explicitly marked as a test
    fixture, the production loader must still refuse -- never silently
    treat a fixture as production."""
    build_test_fixture_model(tmp_path)
    with pytest.raises(ModelArtifactError, match="marked is_test_fixture=True"):
        load_production_model(tmp_path)


def test_load_test_fixture_model_rejects_a_production_marked_artifact(tmp_path):
    """The inverse guard: the fixture loader must refuse to load
    something NOT marked as a fixture, even if it exists."""
    _train_and_save_fake_production_model(tmp_path)
    joblib_path = tmp_path / "decision_tree.joblib"
    with pytest.raises(ModelArtifactError, match="not marked is_test_fixture=True"):
        load_test_fixture_model(joblib_path)


def test_load_test_fixture_model_succeeds_for_an_actual_fixture(tmp_path):
    handle = build_test_fixture_model(tmp_path)
    reloaded = load_test_fixture_model(handle.source_path)
    assert reloaded.is_test_fixture is True


def test_load_model_from_path_missing_artifact_raises(tmp_path):
    with pytest.raises(ModelArtifactError, match="not found"):
        load_model_from_path(tmp_path / "nope.joblib", expected_test_fixture=False)


def test_load_model_from_path_missing_metadata_sidecar_raises(tmp_path):
    _train_and_save_fake_production_model(tmp_path)
    (tmp_path / "decision_tree_metadata.json").unlink()
    with pytest.raises(ModelArtifactError, match="metadata not found"):
        load_model_from_path(tmp_path / "decision_tree.joblib", expected_test_fixture=False)


def test_load_model_from_path_rejects_incompatible_feature_columns(tmp_path):
    df = make_labeled_fixture_dataframe(n_experiments=2, n_intervals=4)
    pipeline = MODEL_REGISTRY["decision_tree"]()
    pipeline.fit(df[FEATURE_COLUMNS], df[TARGET_COLUMN])
    save_model(pipeline, tmp_path / "stale.joblib")
    save_metadata(
        ModelMetadata(
            model_name="stale", feature_columns=["some", "old", "feature", "list"],
            target_column=TARGET_COLUMN, is_test_fixture=False,
        ),
        tmp_path / "stale_metadata.json",
    )
    with pytest.raises(ModelArtifactError, match="feature_columns do not match"):
        load_model_from_path(tmp_path / "stale.joblib", expected_test_fixture=False)


def test_load_model_from_path_rejects_incompatible_target_column(tmp_path):
    df = make_labeled_fixture_dataframe(n_experiments=2, n_intervals=4)
    pipeline = MODEL_REGISTRY["decision_tree"]()
    pipeline.fit(df[FEATURE_COLUMNS], df[TARGET_COLUMN])
    save_model(pipeline, tmp_path / "stale2.joblib")
    save_metadata(
        ModelMetadata(
            model_name="stale2", feature_columns=FEATURE_COLUMNS,
            target_column="some_old_target_name", is_test_fixture=False,
        ),
        tmp_path / "stale2_metadata.json",
    )
    with pytest.raises(ModelArtifactError, match="target_column"):
        load_model_from_path(tmp_path / "stale2.joblib", expected_test_fixture=False)


def test_model_handle_exposes_name_and_source_path(tmp_path):
    _train_and_save_fake_production_model(tmp_path)
    handle = load_production_model(tmp_path)
    assert handle.metadata.model_name == "decision_tree"
    assert handle.source_path == tmp_path / "decision_tree.joblib"
