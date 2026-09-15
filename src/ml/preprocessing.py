"""Preprocessing, kept separate from model definitions (models.py) so the
same preprocessing logic can be reused/inspected independently of which
estimator it feeds.

Built as an sklearn ColumnTransformer/Pipeline specifically so it is
fitted ONLY on training data: calling code always does
`pipeline.fit(X_train, y_train)` then `pipeline.predict(X_test)` --
sklearn's Pipeline contract guarantees the preprocessing statistics
(imputation medians, scaler mean/std, one-hot categories) come from
X_train alone and are only ever *applied* (never re-fit) to X_test.
That is what prevents preprocessing leakage here, not a manual split
of "fit vs transform" code.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from network.schema import FEATURE_COLUMNS

CATEGORICAL_FEATURE_COLUMNS = ["traffic_type"]
NUMERIC_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURE_COLUMNS]


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Impute missing values for every feature; additionally scale numeric
    features only when requested.

    scale_numeric=True is for Linear Regression, which is sensitive to
    feature magnitude differences (e.g. bottleneck_bw_mbps ~1-2 vs
    current_rtt_ms ~tens of ms). Tree-based models (Decision Tree, Random
    Forest, Gradient Boosting) split on raw thresholds and are invariant
    to monotonic rescaling, so they use scale_numeric=False -- scaling
    them would add computation and an extra fitted artifact for no
    modeling benefit ("do not blindly scale everything").
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURE_COLUMNS),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    # sparse_output=False: keeps downstream code (feature
                    # importance mapping, tests, plotting) working with plain
                    # numpy arrays instead of needing to branch on sparse vs
                    # dense output.
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                CATEGORICAL_FEATURE_COLUMNS,
            ),
        ],
        remainder="drop",
    )
