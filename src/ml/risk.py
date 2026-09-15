"""Derived risk classification layer -- LOW / MODERATE / HIGH, computed
FROM the regression model's predicted packet-loss percentage. This never
replaces the regression target; it is a presentation-layer label applied
after prediction (see PROJECT_PLAN.md Section 12 and 20).

Thresholds are fixed, a priori domain values, NOT fit or tuned against
any dataset (train or test) -- doing so would let the "risk" boundary
quietly leak information about the test set's own distribution back into
what counts as a project design decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskThresholds:
    """low_max: predicted loss <= this is LOW.
    moderate_max: predicted loss <= this (and > low_max) is MODERATE.
    Anything above moderate_max is HIGH.

    Defaults (0.1%, 1.0%) follow widely used real-time-traffic quality
    guidance (e.g. ITU-T-style voice/video quality thresholds): loss
    under ~0.1% is generally imperceptible for interactive traffic,
    0.1-1% causes noticeable but often tolerable degradation, and above
    ~1% commonly degrades VoIP/video-conferencing quality enough to be
    considered a real risk. These are ROUND, defensible starting values
    for a course project's report, not a fitted or optimized boundary --
    document any change to these numbers with its own justification.
    """

    low_max: float = 0.1
    moderate_max: float = 1.0

    def __post_init__(self) -> None:
        if not (0 <= self.low_max < self.moderate_max):
            raise ValueError("Require 0 <= low_max < moderate_max")


DEFAULT_RISK_THRESHOLDS = RiskThresholds()


def classify_risk(predicted_loss_pct, thresholds: RiskThresholds = DEFAULT_RISK_THRESHOLDS) -> pd.Series:
    """Vectorized LOW/MODERATE/HIGH classification of predicted packet loss."""
    values = np.asarray(predicted_loss_pct, dtype=float)
    labels = np.where(
        values <= thresholds.low_max, "LOW",
        np.where(values <= thresholds.moderate_max, "MODERATE", "HIGH"),
    )
    return pd.Series(labels, name="risk_level")
