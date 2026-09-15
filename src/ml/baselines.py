"""Naive temporal baseline: the ML models must beat THIS, not just each
other. "predict the previous interval's packet loss" is the standard
reference point for one-step-ahead forecasting -- if a model can't beat
"loss doesn't change from now", it isn't learning anything about network
dynamics (PROJECT_PLAN.md Section 14).
"""

from __future__ import annotations

import pandas as pd


class NaivePersistenceBaseline:
    """predict(X)[i] = X['packet_loss_pct'][i] -- i.e. "next interval's
    loss will equal this interval's measured loss". Implements a minimal
    fit/predict interface (not a full sklearn estimator) so it can sit in
    the same evaluation loop as the real models without needing to be
    fitted or hold any parameters.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaivePersistenceBaseline":
        return self  # stateless: nothing to learn from training data

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X["packet_loss_pct"]
