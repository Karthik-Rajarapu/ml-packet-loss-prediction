"""Train Model page: reuses the EXISTING split/train/evaluate/artifact
functions via ml.training_orchestration -- no new model or evaluation
logic lives here. Saving as production is an explicit action, never
automatic, and archives any prior production artifact.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from dashboard.nav import NAV_PREPARE
from dashboard.reports import find_actual_vs_predicted_plot, find_feature_importance, find_model_comparison
from dashboard.session import AppSession
from ml.evaluate import random_forest_feature_importance, residual_stats
from ml.training_orchestration import TrainingError, run_training, save_as_production_model


def render_train(session: AppSession, models_dir: Path, reports_dir: Path) -> None:
    st.title("Train Model")

    cli_comparison = find_model_comparison(reports_dir)
    if cli_comparison is not None:
        with st.expander("Reports from a command-line training run (scripts/train_models.py), if any"):
            st.caption("These come from data/raw/ via the CLI pipeline, separate from any dataset you "
                        "upload and train here in the dashboard.")
            st.dataframe(cli_comparison, use_container_width=True)
            cli_importance = find_feature_importance(reports_dir)
            if cli_importance is not None:
                st.bar_chart(cli_importance.set_index("feature")["importance"].head(10))
            cli_plot = find_actual_vs_predicted_plot(reports_dir)
            if cli_plot is not None:
                st.image(str(cli_plot), caption="Actual vs Predicted (from the CLI training run)")

    if session.prepared_df is None:
        st.info("Prepare a dataset first.")
        if st.button("Go to Prepare Dataset"):
            st.session_state["nav_page"] = NAV_PREPARE
            st.rerun()
        return

    col_a, col_b = st.columns(2)
    with col_a:
        split_label = st.radio(
            "Evaluation split",
            ["Chronological (train on earlier data, test on later)",
             "Random group (random hold-out of whole experiments)"],
        )
        split_method = "chronological" if split_label.startswith("Chronological") else "random_group"
    with col_b:
        test_fraction = st.slider("Test fraction (by experiment/group)", 0.1, 0.5, 0.3, 0.05)

    if st.button("Train Models", type="primary"):
        with st.spinner("Training prediction models..."):
            try:
                session.training_result = run_training(
                    session.prepared_df, split_method=split_method, test_fraction=test_fraction,
                )
                session.production_model_saved_this_session = False
            except TrainingError as exc:
                st.error(str(exc))
                session.training_result = None

    if session.training_result is None:
        st.info("Click “Train Models” to evaluate candidate models on your prepared dataset.")
        return

    result = session.training_result
    st.subheader("Model Performance")
    st.caption(f"Evaluated on a held-out split: {result.n_train_experiments} training "
                f"experiment(s)/group(s) ({result.n_train_rows} rows), {result.n_test_experiments} held out "
                f"for testing ({result.n_test_rows} rows).")
    st.dataframe(result.comparison, use_container_width=True)

    best_row = result.comparison.iloc[0]
    best_name = best_row["Model"]
    st.subheader("Recommended Model")
    st.markdown(f"### {best_name}")
    st.caption(f"Selected because it achieved the lowest MAE ({best_row['MAE']:.4f}) on this dataset's "
                "held-out test split — the evaluation basis used for selection.")

    baseline_rows = result.comparison[result.comparison["Model"] == "naive_persistence"]
    if not baseline_rows.empty and best_name != "naive_persistence":
        baseline_mae = float(baseline_rows.iloc[0]["MAE"])
        if baseline_mae > 0:
            improvement_pct = (baseline_mae - float(best_row["MAE"])) / baseline_mae * 100
            if improvement_pct > 0:
                st.success(f"This model's MAE is {improvement_pct:.1f}% lower than the naive "
                            "“loss doesn't change” baseline on this dataset's test split.")
            else:
                st.warning("This model did not outperform the naive baseline on this dataset's test split.")
    elif best_name == "naive_persistence":
        st.warning("The naive baseline performed best on this dataset — none of the ML models beat "
                    "simply predicting that loss stays the same.")

    with st.expander("Residual detail (best model)"):
        st.write(residual_stats(result.y_test, result.predictions[best_name]))

    best_pipeline = result.fitted_models[best_name]
    if hasattr(best_pipeline.named_steps.get("model"), "feature_importances_"):
        st.subheader("Why this model? (feature importance)")
        importance_df = random_forest_feature_importance(best_pipeline)
        st.bar_chart(importance_df.set_index("feature")["importance"].head(10))
    else:
        st.info("Feature explanation is not available for this model.")

    st.divider()
    st.subheader("Save Model")
    st.caption("Saving replaces the model the Predict page uses in Production mode. Any existing "
                "production model is archived, never deleted or silently overwritten.")
    if st.button("Save as Production Model", type="primary"):
        save_as_production_model(
            result, models_dir, dataset_filename=session.upload_filename or "uploaded.csv",
            n_rows=len(session.prepared_df),
        )
        session.production_model_saved_this_session = True
        st.cache_resource.clear()
        st.success(f"Saved “{best_name}” as the production model.")

    if session.production_model_saved_this_session:
        st.caption("✓ This model is now used by Predict → Production mode.")
