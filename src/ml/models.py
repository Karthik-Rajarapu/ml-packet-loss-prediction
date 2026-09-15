"""Candidate regression models.

Each entry is a zero-arg factory (not a shared instance) so every call
to build a fresh, unfitted Pipeline -- reusing one fitted estimator
across experiments/tests would silently carry over state between runs.
Every model with a random component uses RANDOM_SEED so results are
reproducible run-to-run given the same data and split.

XGBoost is deliberately NOT included: the brief says to consider it only
if existing results justify it, and with no real dataset trained yet
(Section 1), there is nothing to justify it against. GradientBoostingRegressor
(scikit-learn, already a dependency) covers the same "boosted trees" family
for now -- see docs/PHASE_4_ML_MODELING.md.
"""

from __future__ import annotations

from typing import Callable

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from ml.preprocessing import build_preprocessor

RANDOM_SEED = 42

ModelFactory = Callable[[], Pipeline]


def _linear_regression() -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor(scale_numeric=True)),
        ("model", LinearRegression()),
    ])


def _decision_tree() -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor(scale_numeric=False)),
        ("model", DecisionTreeRegressor(random_state=RANDOM_SEED, max_depth=8)),
    ])


def _random_forest() -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor(scale_numeric=False)),
        ("model", RandomForestRegressor(random_state=RANDOM_SEED, n_estimators=200, max_depth=None, n_jobs=-1)),
    ])


def _gradient_boosting() -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor(scale_numeric=False)),
        ("model", GradientBoostingRegressor(random_state=RANDOM_SEED)),
    ])


MODEL_REGISTRY: dict[str, ModelFactory] = {
    "linear_regression": _linear_regression,
    "decision_tree": _decision_tree,
    "random_forest": _random_forest,
    "gradient_boosting": _gradient_boosting,
}
