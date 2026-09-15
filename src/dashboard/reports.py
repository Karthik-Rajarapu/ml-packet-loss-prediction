"""Locates existing Phase 4/5 report artifacts on disk -- the dashboard
reads these, it never recomputes metrics/importance/plots itself (Step
10/11: "Reuse Phase 4 outputs... do not recompute inside Streamlit").

If reports/modeling/ has nothing in it (the current state of this repo --
no real dataset has been trained on yet), every finder here returns None
or an empty list, and app.py is responsible for showing the honest
"unavailable" message rather than fabricating anything.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def find_model_comparison(reports_dir: Path) -> pd.DataFrame | None:
    """reports/modeling/model_comparison.csv, written by
    scripts/train_models.py -- the real Model|MAE|RMSE|R2 table,
    including the naive_persistence baseline row, when it exists."""
    path = reports_dir / "model_comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def find_feature_importance(reports_dir: Path) -> pd.DataFrame | None:
    """The best-model's *_feature_importance.csv, if scripts/train_models.py
    has produced one. Returns the first match (there should be at most
    one per training run in this project's current design)."""
    matches = sorted(reports_dir.glob("*_feature_importance.csv"))
    if not matches:
        return None
    return pd.read_csv(matches[0])


def find_permutation_importance(reports_dir: Path) -> pd.DataFrame | None:
    matches = sorted(reports_dir.glob("*_permutation_importance.csv"))
    if not matches:
        return None
    return pd.read_csv(matches[0])


def find_actual_vs_predicted_plot(reports_dir: Path) -> Path | None:
    matches = sorted(reports_dir.glob("*_actual_vs_predicted.png"))
    return matches[0] if matches else None


def find_residuals_plot(reports_dir: Path) -> Path | None:
    matches = sorted(reports_dir.glob("*_residuals.png"))
    return matches[0] if matches else None
