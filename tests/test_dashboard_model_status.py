from dashboard.model_status import check_production_model
from ml.artifacts import ModelMetadata, save_metadata, save_model
from ml.demo import build_test_fixture_model
from ml.inference import PRODUCTION_MODEL_UNAVAILABLE_MESSAGE
from ml.models import MODEL_REGISTRY
from ml_fixtures import make_labeled_fixture_dataframe
from network.schema import FEATURE_COLUMNS, TARGET_COLUMN


def test_no_model_reports_unavailable_with_exact_message(tmp_path):
    status = check_production_model(tmp_path)
    assert status.available is False
    assert status.handle is None
    assert status.message == PRODUCTION_MODEL_UNAVAILABLE_MESSAGE


def test_nonexistent_dir_reports_unavailable(tmp_path):
    status = check_production_model(tmp_path / "does_not_exist")
    assert status.available is False


def test_valid_production_model_reports_available(tmp_path):
    df = make_labeled_fixture_dataframe(n_experiments=3, n_intervals=5)
    pipeline = MODEL_REGISTRY["decision_tree"]()
    pipeline.fit(df[FEATURE_COLUMNS], df[TARGET_COLUMN])
    save_model(pipeline, tmp_path / "decision_tree.joblib")
    save_metadata(
        ModelMetadata(model_name="decision_tree", feature_columns=FEATURE_COLUMNS,
                       target_column=TARGET_COLUMN, is_test_fixture=False),
        tmp_path / "decision_tree_metadata.json",
    )
    status = check_production_model(tmp_path)
    assert status.available is True
    assert status.handle is not None
    assert status.message is None


def test_fixture_marked_model_never_reported_as_available_production(tmp_path):
    """The dashboard's status check must never treat a labeled test
    fixture as a usable production model."""
    build_test_fixture_model(tmp_path)
    status = check_production_model(tmp_path)
    assert status.available is False


def test_never_raises_on_bad_directory(tmp_path):
    """check_production_model must always return a ModelStatus, never
    propagate an exception the UI would have to catch separately."""
    status = check_production_model(tmp_path / "a" / "b" / "c")
    assert status.available is False
    assert isinstance(status.message, str)
