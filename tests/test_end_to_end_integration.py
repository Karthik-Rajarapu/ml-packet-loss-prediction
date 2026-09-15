"""Phase 8 Step 3: full end-to-end integration test.

    TEST FIXTURE — NOT REAL NETWORK DATA

Exercises the ENTIRE pipeline in one test, chaining every phase's real
code (never re-implemented here):

    synthetic CSVs (shaped exactly like Phase 1 experiment output)
        -> ml.dataset.load_dataset            (Phase 4: combine + validate + leakage-check)
        -> ml.data_quality.build_data_quality_report  (Phase 5)
        -> ml.split.chronological_split / random_group_split  (Phase 4)
        -> ml.train.train_and_evaluate         (Phase 4: baseline + 4 models)
        -> ml.artifacts.save_model/save_metadata      (Phase 4/6)
        -> ml.inference.load_test_fixture_model + InferenceEngine  (Phase 6)
        -> ml.risk.classify_risk (via InferenceEngine)  (Phase 4)
        -> dashboard.history / dashboard.feature_inputs  (Phase 7 data contract)

The model artifact this test saves is explicitly marked
is_test_fixture=True and written to tmp_path -- NEVER models/ -- and is
loaded back exclusively through load_test_fixture_model(), which refuses
anything not marked that way. Nothing here is presented as, or could be
mistaken for, a production result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dashboard.feature_inputs import build_feature_input_specs
from dashboard.history import PredictionHistory, PredictionHistoryEntry
from ml.artifacts import ModelMetadata, save_metadata, save_model
from ml.data_quality import build_data_quality_report
from ml.dataset import load_dataset
from ml.evaluate import comparison_table
from ml.inference import InferenceEngine, load_test_fixture_model
from ml.models import MODEL_REGISTRY
from ml.risk import classify_risk
from ml.split import assert_no_group_overlap, chronological_split, random_group_split
from ml.train import train_and_evaluate
from ml_fixtures import make_fixture_dataframe
from network.schema import FEATURE_COLUMNS, TARGET_COLUMN

TEST_FIXTURE_MARK = "TEST FIXTURE — NOT REAL NETWORK DATA"


@pytest.fixture()
def synthetic_dataset_csvs(tmp_path: Path) -> list[Path]:
    """8 synthetic 'experiments' spread across two 'files' -- mirrors
    Phase 2's multi-CSV data/raw/ layout without touching it."""
    paths = []
    for i, prefix in enumerate(["e2e-sweep-a", "e2e-sweep-b"]):
        df = make_fixture_dataframe(n_experiments=4, n_intervals=6, id_prefix=prefix)
        path = tmp_path / f"{prefix}.csv"
        df.to_csv(path, index=False)
        paths.append(path)
    return paths


def test_full_pipeline_end_to_end(tmp_path, synthetic_dataset_csvs):
    # ---- 1. Load + validate (reuses Phase 4's exact loader, which itself
    #         reuses Phase 2's validate_rows/validate_no_leakage) ----
    df = load_dataset(synthetic_dataset_csvs)
    assert df["experiment_id"].nunique() == 8
    assert TARGET_COLUMN in df.columns
    assert df[TARGET_COLUMN].isna().sum() == 0

    # ---- 2. Data quality report (Phase 5) -- confirm the fixture has
    #         SOME loss variation, otherwise the rest of this test would
    #         be exercising a degenerate case ----
    report = build_data_quality_report(df, source_files=[str(p) for p in synthetic_dataset_csvs])
    assert report.overview.total_rows == len(df)
    assert report.target_summary.max > 0, f"{TEST_FIXTURE_MARK}: fixture must contain non-zero loss to be useful"
    assert report.issues.duplicated_experiment_interval_pairs == 0
    assert report.issues.out_of_range_loss_rows == 0

    # ---- 3. Group-aware split, both strategies (Phase 4) ----
    chrono_train, chrono_test = chronological_split(df, test_fraction=0.3)
    assert_no_group_overlap(chrono_train, chrono_test)
    random_train, random_test = random_group_split(df, test_fraction=0.3, random_seed=42)
    assert_no_group_overlap(random_train, random_test)
    # both splits must partition every row exactly once
    assert len(chrono_train) + len(chrono_test) == len(df)
    assert len(random_train) + len(random_test) == len(df)

    # ---- 4. Leakage audit at the split boundary: no feature column may
    #         equal the target column, and the split must never leak a
    #         group across the train/test boundary (checked above) ----
    assert TARGET_COLUMN not in FEATURE_COLUMNS

    # ---- 5. Train baseline + all 4 models (Phase 4, unmodified) ----
    result = train_and_evaluate(chrono_train, chrono_test)
    expected_models = {"naive_persistence", "linear_regression", "decision_tree",
                        "random_forest", "gradient_boosting"}
    assert set(result["metrics"]) == expected_models
    table = comparison_table(result["metrics"])
    assert (table["MAE"] >= 0).all() and (table["RMSE"] >= 0).all()

    # ---- 6. Save the best model as an EXPLICIT TEST FIXTURE artifact
    #         (never models/, never is_test_fixture=False) ----
    best_name = table.iloc[0]["Model"]
    best_pipeline = result["fitted_models"][best_name]
    fixture_models_dir = tmp_path / "test_fixture_artifacts"  # NOT the real models/ dir
    joblib_path = save_model(best_pipeline, fixture_models_dir / f"E2E_TEST_FIXTURE_{best_name}.joblib")
    save_metadata(
        ModelMetadata(
            model_name=f"E2E_TEST_FIXTURE_{best_name}",
            feature_columns=FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
            is_test_fixture=True,
            training_config={"note": TEST_FIXTURE_MARK},
            metrics=result["metrics"][best_name],
        ),
        fixture_models_dir / f"E2E_TEST_FIXTURE_{best_name}_metadata.json",
    )

    # ---- 7. Load it back EXCLUSIVELY through the test-fixture loader --
    #         load_production_model() is deliberately never called here ----
    handle = load_test_fixture_model(joblib_path)
    assert handle.is_test_fixture is True
    assert handle.metadata.feature_columns == list(FEATURE_COLUMNS)

    # ---- 8. Run inference through the real Phase 6 engine ----
    engine = InferenceEngine(handle)
    sample_row = chrono_test.iloc[0]
    metrics = {name: sample_row[name] for name in FEATURE_COLUMNS}
    # normalize numpy scalar types to plain python, as a live caller would supply
    metrics = {k: (v.item() if hasattr(v, "item") else v) for k, v in metrics.items()}
    prediction = engine.predict(metrics)

    assert prediction.is_test_fixture is True
    assert isinstance(prediction.predicted_packet_loss_pct, float)
    assert prediction.risk_level in ("LOW", "MODERATE", "HIGH")

    # ---- 9. Risk classification consistency: InferenceEngine's risk must
    #         match an independent call to the same Phase 4 risk function ----
    independent_risk = str(classify_risk([prediction.predicted_packet_loss_pct]).iloc[0])
    assert prediction.risk_level == independent_risk

    # ---- 10. Leakage check on the actual inference call: the metrics
    #          dict handed to the engine must never have contained the
    #          target column or any future-interval information ----
    assert TARGET_COLUMN not in metrics

    # ---- 11. Dashboard data contract (Phase 7): PredictionResult must
    #          flow cleanly into PredictionHistory and the same
    #          FEATURE_COLUMNS must drive both the training features and
    #          the dashboard's own input-widget spec list ----
    history = PredictionHistory()
    history.add(PredictionHistoryEntry(
        timestamp=prediction.predicted_at, model_name=prediction.model_name,
        model_kind=prediction.model_kind, is_test_fixture=prediction.is_test_fixture,
        predicted_packet_loss_pct=prediction.predicted_packet_loss_pct, risk_level=prediction.risk_level,
        input_features=metrics,
    ))
    history_df = history.to_dataframe()
    assert len(history_df) == 1
    assert history_df.iloc[0]["is_test_fixture"] == True  # noqa: E712

    specs = build_feature_input_specs(metrics)
    assert [s.name for s in specs] == list(FEATURE_COLUMNS), (
        "the dashboard's input-widget contract must match the exact feature "
        "list the model was trained and inferred on"
    )


def test_naive_baseline_integrates_alongside_ml_models(synthetic_dataset_csvs):
    """The baseline must sit in the same evaluation/inference contract
    as the ML models, not a special-cased side path."""
    df = load_dataset(synthetic_dataset_csvs)
    train_df, test_df = chronological_split(df, test_fraction=0.3)
    result = train_and_evaluate(train_df, test_df)
    assert "naive_persistence" in result["metrics"]
    assert "naive_persistence" in result["fitted_models"]
    baseline_pred = result["predictions"]["naive_persistence"]
    # persistence baseline must equal the CURRENT interval's own packet_loss_pct
    assert (baseline_pred.values == test_df["packet_loss_pct"].values).all()
