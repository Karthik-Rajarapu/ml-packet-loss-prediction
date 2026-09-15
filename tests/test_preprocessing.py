"""Tests for ml.preprocessing -- specifically that fitting only touches
training data (the core leakage-prevention property sklearn's Pipeline
contract gives us, verified here rather than just assumed).
"""

import numpy as np
import pandas as pd

from ml.preprocessing import CATEGORICAL_FEATURE_COLUMNS, NUMERIC_FEATURE_COLUMNS, build_preprocessor
from ml_fixtures import make_fixture_dataframe
from network.schema import FEATURE_COLUMNS


def test_numeric_and_categorical_columns_partition_feature_columns():
    assert set(NUMERIC_FEATURE_COLUMNS) | set(CATEGORICAL_FEATURE_COLUMNS) == set(FEATURE_COLUMNS)
    assert set(NUMERIC_FEATURE_COLUMNS) & set(CATEGORICAL_FEATURE_COLUMNS) == set()


def _named_transformer(pre, name):
    # ColumnTransformer.transformers is a list of (name, transformer, columns)
    # 3-tuples, not 2-tuples -- dict(pre.transformers) is not valid.
    return next(transformer for tname, transformer, _cols in pre.transformers if tname == name)


def test_scale_numeric_false_omits_scaler_step():
    pre = build_preprocessor(scale_numeric=False)
    numeric_pipeline = _named_transformer(pre, "numeric")
    assert "scaler" not in dict(numeric_pipeline.steps)


def test_scale_numeric_true_includes_scaler_step():
    pre = build_preprocessor(scale_numeric=True)
    numeric_pipeline = _named_transformer(pre, "numeric")
    assert "scaler" in dict(numeric_pipeline.steps)


def test_imputer_statistics_come_from_train_only():
    """Fit on train (all current_rtt_ms = 100), transform a test row with a
    DIFFERENT missing-value context -- the imputed value must reflect
    TRAIN's median (100), never anything derived from the test row."""
    train = make_fixture_dataframe(n_experiments=2, n_intervals=4).copy()
    train["current_rtt_ms"] = 100.0  # uniform train value -> median is unambiguous

    test = make_fixture_dataframe(n_experiments=1, n_intervals=4).copy()
    test["current_rtt_ms"] = np.nan  # entirely missing in test

    pre = build_preprocessor(scale_numeric=False)
    pre.fit(train[FEATURE_COLUMNS])
    transformed_test = pre.transform(test[FEATURE_COLUMNS])

    feature_names = list(pre.get_feature_names_out())
    rtt_col_idx = [i for i, name in enumerate(feature_names) if "current_rtt_ms" in name][0]
    imputed_values = transformed_test[:, rtt_col_idx]
    assert np.allclose(imputed_values, 100.0), "imputed value must come from TRAIN's median, not test data"


def test_transform_never_refits_on_test_data():
    """Calling .transform() (not .fit_transform()) on test data must not
    change the fitted statistics -- verified by transforming twice with
    wildly different test data and confirming identical output for a
    fixed input row."""
    train = make_fixture_dataframe(n_experiments=3, n_intervals=4)
    pre = build_preprocessor(scale_numeric=True)
    pre.fit(train[FEATURE_COLUMNS])

    probe_row = train[FEATURE_COLUMNS].iloc[[0]]
    first = pre.transform(probe_row)

    # Transform a large, differently-distributed batch in between.
    other = make_fixture_dataframe(n_experiments=2, n_intervals=6)
    other["current_rtt_ms"] = other["current_rtt_ms"] * 1000.0
    pre.transform(other[FEATURE_COLUMNS])

    second = pre.transform(probe_row)
    assert np.allclose(first, second), "transform() must not mutate fitted preprocessor state"


def test_unknown_categorical_value_in_test_does_not_raise():
    """handle_unknown='ignore' must let a traffic_type value never seen in
    train pass through test without crashing (encoded as all-zeros)."""
    train = make_fixture_dataframe(n_experiments=1, n_intervals=4)
    train["traffic_type"] = "udp"

    test = make_fixture_dataframe(n_experiments=1, n_intervals=4)
    test["traffic_type"] = "quic"  # never seen during fit

    pre = build_preprocessor(scale_numeric=False)
    pre.fit(train[FEATURE_COLUMNS])
    transformed = pre.transform(test[FEATURE_COLUMNS])  # should not raise
    assert transformed.shape[0] == len(test)
