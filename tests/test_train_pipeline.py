"""End-to-end integration test of the Phase 4 pipeline: dataset -> split
-> train_and_evaluate -> comparison table -> feature importance -> risk
classification -> artifact save/load -> plots.

Uses ONLY the synthetic TEST FIXTURE from ml_fixtures.py, written to a
tmp_path CSV and reloaded through the real ml.dataset.load_dataset(), so
this exercises the exact code path scripts/train_models.py uses -- minus
the "is this a real Mininet dataset" question, which is out of scope for
a fixture. NOTHING here asserts a performance threshold (e.g. "MAE < X")
-- fixture data has no real-world meaning, only structural correctness
is checked (shapes, types, no NaNs, files exist). See
docs/PHASE_4_ML_MODELING.md Section 1 and 15.
"""

from ml.dataset import load_dataset
from ml.evaluate import comparison_table, random_forest_feature_importance, residual_stats
from ml.risk import classify_risk
from ml.split import chronological_split
from ml.train import train_and_evaluate
from ml.visualize import plot_actual_vs_predicted, plot_feature_importance, plot_residual_distribution
from ml_fixtures import make_fixture_dataframe
from network.schema import FEATURE_COLUMNS, TARGET_COLUMN


def test_full_pipeline_runs_end_to_end_on_fixture_data(tmp_path):
    fixture_csv = tmp_path / "fixture_dataset.csv"
    make_fixture_dataframe(n_experiments=8, n_intervals=6).to_csv(fixture_csv, index=False)

    df = load_dataset([fixture_csv])
    assert TARGET_COLUMN in df.columns
    assert df[TARGET_COLUMN].isna().sum() == 0

    train_df, test_df = chronological_split(df, test_fraction=0.3)
    assert len(train_df) > 0 and len(test_df) > 0

    result = train_and_evaluate(train_df, test_df)

    # every model (including the naive baseline) produced a metrics row
    expected_models = {"naive_persistence", "linear_regression", "decision_tree",
                        "random_forest", "gradient_boosting"}
    assert set(result["metrics"]) == expected_models

    table = comparison_table(result["metrics"])
    assert len(table) == len(expected_models)
    assert table["MAE"].is_monotonic_increasing  # sorted ascending by MAE
    assert (table["MAE"] >= 0).all()
    assert (table["RMSE"] >= 0).all()

    best_name = table.iloc[0]["Model"]
    y_test = result["y_test"]
    y_pred = result["predictions"][best_name]
    assert len(y_pred) == len(y_test)

    stats = residual_stats(y_test, y_pred)
    assert "mean_residual" in stats

    risk_labels = classify_risk(y_pred)
    assert set(risk_labels).issubset({"LOW", "MODERATE", "HIGH"})

    # feature importance for a tree model
    rf_pipeline = result["fitted_models"]["random_forest"]
    importance_table = random_forest_feature_importance(rf_pipeline)
    assert len(importance_table) > 0

    # plots write real files (to tmp_path, never reports/modeling/ for fixture data)
    scatter_path = plot_actual_vs_predicted(y_test, y_pred, tmp_path / "scatter.png")
    residual_path = plot_residual_distribution(y_test, y_pred, tmp_path / "residuals.png")
    importance_path = plot_feature_importance(importance_table, tmp_path / "importance.png")
    for path in (scatter_path, residual_path, importance_path):
        assert path.exists()
        assert path.stat().st_size > 0


def test_pipeline_never_uses_target_column_as_a_feature():
    """A cheap but important guard: TARGET_COLUMN must never appear in
    FEATURE_COLUMNS, which train_and_evaluate relies on to build X."""
    assert TARGET_COLUMN not in FEATURE_COLUMNS
