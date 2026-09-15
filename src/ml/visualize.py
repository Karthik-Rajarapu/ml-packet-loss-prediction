"""Plotting for model evaluation. Uses the non-interactive 'Agg' backend
so this works on a machine with no display (WSL2 without X, CI, etc.).

Every function here just draws whatever arrays it's given -- it has no
way to know if the data is real or a fixture. The caller is responsible
for only pointing these at reports/modeling/ when the data is real (see
scripts/train_models.py); tests must save to a tmp_path instead.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def plot_actual_vs_predicted(y_true, y_pred, path: Path, title: str = "Actual vs Predicted") -> Path:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=15)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual next-interval packet loss (%)")
    ax.set_ylabel("Predicted next-interval packet loss (%)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_residual_distribution(y_true, y_pred, path: Path, title: str = "Residual Distribution") -> Path:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=30, color="steelblue", edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual (actual - predicted), percentage points")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_feature_importance(importance_df: pd.DataFrame, path: Path, top_n: int = 15,
                             title: str = "Feature Importance") -> Path:
    top = importance_df.head(top_n).iloc[::-1]
    value_col = "importance" if "importance" in top.columns else "importance_mean"
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(top))))
    ax.barh(top["feature"], top[value_col], color="seagreen")
    ax.set_xlabel(value_col)
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_prediction_timeline(timestamps, y_true, y_pred, path: Path,
                              title: str = "Prediction Timeline (one experiment)") -> Path:
    """Optional: actual vs predicted over time for a SINGLE experiment_id
    (multiple experiments on one time axis would be misleading, since
    each has its own independent clock)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(timestamps, y_true, label="Actual", marker="o", markersize=3)
    ax.plot(timestamps, y_pred, label="Predicted", marker="x", markersize=3)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Packet loss (%)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
