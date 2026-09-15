"""Tests for ml.dataset -- the CSV-combining/validating step Phase 4 had
to add itself since no Phase 3 component existed to reuse (see
docs/PHASE_4_ML_MODELING.md Section 1). Uses tiny TEST FIXTURE CSVs
written to tmp_path, never real experiment data.
"""

import pandas as pd
import pytest

from ml.dataset import DatasetError, discover_csv_files, load_dataset
from network.schema import CSV_COLUMNS, TARGET_COLUMN
from ml_fixtures import make_fixture_dataframe


def _write_fixture_csv(tmp_path, filename="fixture.csv", n_experiments=2, n_intervals=4, id_prefix="fixture-exp"):
    df = make_fixture_dataframe(n_experiments=n_experiments, n_intervals=n_intervals, id_prefix=id_prefix)
    path = tmp_path / filename
    df.to_csv(path, index=False)
    return path


def test_discover_csv_files_from_directory(tmp_path):
    _write_fixture_csv(tmp_path, "a.csv")
    _write_fixture_csv(tmp_path, "b.csv")
    (tmp_path / "not_a_csv.txt").write_text("ignore me")
    found = discover_csv_files(tmp_path)
    assert sorted(p.name for p in found) == ["a.csv", "b.csv"]


def test_discover_csv_files_single_file(tmp_path):
    path = _write_fixture_csv(tmp_path)
    assert discover_csv_files(path) == [path]


def test_discover_csv_files_missing_path_returns_empty(tmp_path):
    assert discover_csv_files(tmp_path / "does_not_exist") == []


def test_load_dataset_no_files_raises_with_clear_message():
    with pytest.raises(DatasetError, match="No CSV files provided"):
        load_dataset([])


def test_load_dataset_combines_multiple_files(tmp_path):
    # Distinct id_prefix per file -- mirrors how two real Phase 2 sweep
    # outputs would never share an experiment_id -- so this tests genuine
    # multi-file combination, not two files worth of duplicate rows.
    path_a = _write_fixture_csv(tmp_path, "a.csv", n_experiments=2, id_prefix="sweep-a")
    path_b = _write_fixture_csv(tmp_path, "b.csv", n_experiments=2, id_prefix="sweep-b")
    df = load_dataset([path_a, path_b])
    assert set(df.columns) == set(CSV_COLUMNS)
    assert df["experiment_id"].nunique() == 4
    assert len(df) > 0


def test_load_dataset_drops_end_of_experiment_rows_without_target(tmp_path):
    path = _write_fixture_csv(tmp_path, n_experiments=1, n_intervals=5)
    df = load_dataset([path])
    assert df[TARGET_COLUMN].isna().sum() == 0


def test_load_dataset_missing_column_raises(tmp_path):
    df = make_fixture_dataframe(n_experiments=1, n_intervals=4)
    df = df.drop(columns=["current_rtt_ms"])
    path = tmp_path / "broken.csv"
    df.to_csv(path, index=False)
    with pytest.raises(DatasetError, match="missing required column"):
        load_dataset([path])


def test_load_dataset_rejects_leakage_across_experiments(tmp_path):
    df = make_fixture_dataframe(n_experiments=2, n_intervals=4)
    # Corrupt one row's target to point at a different experiment's value.
    other_exp_value = df.loc[df["experiment_id"] != df.iloc[0]["experiment_id"], "packet_loss_pct"].iloc[0]
    df.loc[0, TARGET_COLUMN] = other_exp_value + 999.0  # guaranteed mismatch
    path = tmp_path / "corrupted.csv"
    df.to_csv(path, index=False)
    with pytest.raises(DatasetError, match="failed validation"):
        load_dataset([path])
