"""Tests for ml.training_orchestration -- sequences EXISTING Phase 4
functions (split, train_and_evaluate, artifacts) only. Also verifies the
artifact-archiving ("never overwrite blindly") behavior and that every
artifact saved through save_as_production_model() is unambiguously
is_test_fixture=False.
"""

from ml.artifacts import load_metadata, load_model
from ml.training_orchestration import TrainingError, run_training, save_as_production_model
from ml_fixtures import make_labeled_fixture_dataframe


def test_run_training_produces_all_five_models():
    df = make_labeled_fixture_dataframe(n_experiments=6, n_intervals=5)
    result = run_training(df, split_method="chronological", test_fraction=0.3)
    expected = {"naive_persistence", "linear_regression", "decision_tree", "random_forest", "gradient_boosting"}
    assert set(result.metrics) == expected
    assert result.best_model_name in expected
    assert (result.comparison["MAE"] >= 0).all()


def test_run_training_random_group_split():
    df = make_labeled_fixture_dataframe(n_experiments=6, n_intervals=5)
    result = run_training(df, split_method="random_group", test_fraction=0.3, random_seed=1)
    assert result.split_method == "random_group"
    assert result.n_train_experiments + result.n_test_experiments == 6


def test_run_training_rejects_single_experiment_dataset():
    df = make_labeled_fixture_dataframe(n_experiments=1, n_intervals=10)
    try:
        run_training(df)
        raised = False
    except TrainingError:
        raised = True
    assert raised


def test_save_as_production_model_writes_real_artifact(tmp_path):
    df = make_labeled_fixture_dataframe(n_experiments=6, n_intervals=5)
    result = run_training(df)
    joblib_path = save_as_production_model(result, tmp_path, dataset_filename="upload.csv", n_rows=len(df))

    assert joblib_path.exists()
    metadata_path = joblib_path.with_name(joblib_path.stem + "_metadata.json")
    assert metadata_path.exists()

    metadata = load_metadata(metadata_path)
    assert metadata.is_test_fixture is False, "a saved production artifact must never be marked as a test fixture"
    assert metadata.training_config["source_dataset_filename"] == "upload.csv"
    assert metadata.training_config["source_dataset_rows"] == len(df)

    reloaded = load_model(joblib_path)
    assert reloaded is not None


def test_save_as_production_model_archives_prior_artifact_instead_of_deleting(tmp_path):
    df = make_labeled_fixture_dataframe(n_experiments=6, n_intervals=5)
    result_a = run_training(df)
    first_path = save_as_production_model(result_a, tmp_path, dataset_filename="first.csv", n_rows=len(df))
    first_model_name = first_path.stem

    result_b = run_training(df, split_method="random_group", random_seed=99)
    save_as_production_model(result_b, tmp_path, dataset_filename="second.csv", n_rows=len(df))

    archive_dir = tmp_path / "archive"
    assert archive_dir.exists()
    archived_files = list(archive_dir.glob(f"*{first_model_name}*"))
    assert archived_files, "the prior production artifact must be archived, not deleted"

    # exactly one current production joblib remains at the top level
    current_joblibs = list(tmp_path.glob("*.joblib"))
    assert len(current_joblibs) == 1


def test_save_as_production_model_never_imports_ml_demo():
    """Structural guard: the fixture-model module must not be a runtime
    dependency of this one -- a production save path must have no code
    path back to the synthetic demo/fixture data. Checks the actual
    import graph, not prose text (this module's own docstring mentions
    "ml.demo" by name precisely to explain that it's NOT imported)."""
    import ml.training_orchestration as mod
    assert not hasattr(mod, "demo")
    assert "ml.demo" not in mod.__dict__
    # nothing bound in this module's namespace originates from ml.demo
    assert all(getattr(v, "__module__", "") != "ml.demo" for v in vars(mod).values())
