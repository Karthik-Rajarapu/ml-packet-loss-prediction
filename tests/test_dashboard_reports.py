"""Tests for dashboard.reports -- must return None/[] cleanly when no
real report exists (the current state of this repo's reports/modeling/),
and correctly find a real one when present. Never fabricates a report.
"""

import pandas as pd

from dashboard.reports import (
    find_actual_vs_predicted_plot,
    find_feature_importance,
    find_model_comparison,
    find_permutation_importance,
    find_residuals_plot,
)


def test_all_finders_return_none_on_empty_directory(tmp_path):
    assert find_model_comparison(tmp_path) is None
    assert find_feature_importance(tmp_path) is None
    assert find_permutation_importance(tmp_path) is None
    assert find_actual_vs_predicted_plot(tmp_path) is None
    assert find_residuals_plot(tmp_path) is None


def test_find_model_comparison_reads_real_csv(tmp_path):
    pd.DataFrame({"Model": ["naive_persistence", "random_forest"], "MAE": [2.0, 1.0],
                  "RMSE": [3.0, 1.5], "R2": [0.0, 0.5]}).to_csv(tmp_path / "model_comparison.csv", index=False)
    table = find_model_comparison(tmp_path)
    assert table is not None
    assert list(table["Model"]) == ["naive_persistence", "random_forest"]


def test_find_feature_importance_reads_real_csv(tmp_path):
    pd.DataFrame({"feature": ["current_rtt_ms", "queue_length"], "importance": [0.6, 0.4]}).to_csv(
        tmp_path / "random_forest_feature_importance.csv", index=False
    )
    table = find_feature_importance(tmp_path)
    assert table is not None
    assert "importance" in table.columns


def test_find_actual_vs_predicted_plot_finds_real_png(tmp_path):
    (tmp_path / "random_forest_actual_vs_predicted.png").write_bytes(b"not a real png but a real file")
    path = find_actual_vs_predicted_plot(tmp_path)
    assert path is not None
    assert path.name == "random_forest_actual_vs_predicted.png"


def test_find_residuals_plot_finds_real_png(tmp_path):
    (tmp_path / "random_forest_residuals.png").write_bytes(b"placeholder")
    path = find_residuals_plot(tmp_path)
    assert path is not None


def test_finders_ignore_unrelated_files(tmp_path):
    (tmp_path / "notes.txt").write_text("irrelevant")
    (tmp_path / "some_other_report.csv").write_text("a,b\n1,2\n")
    assert find_model_comparison(tmp_path) is None
    assert find_feature_importance(tmp_path) is None
