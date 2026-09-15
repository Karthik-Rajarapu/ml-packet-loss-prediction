"""Tests for ml.models -- every registered model fits/predicts on TEST
FIXTURE data without error, factories are independent (no shared fitted
state), and results are reproducible given the fixed RANDOM_SEED.
"""

import numpy as np

from ml.models import MODEL_REGISTRY, RANDOM_SEED
from ml_fixtures import make_labeled_fixture_dataframe
from network.schema import FEATURE_COLUMNS, TARGET_COLUMN


def _train_test_frames():
    df = make_labeled_fixture_dataframe(n_experiments=6, n_intervals=5)
    experiments = sorted(df["experiment_id"].unique())
    train_ids, test_ids = experiments[:4], experiments[4:]
    train_df = df[df["experiment_id"].isin(train_ids)]
    test_df = df[df["experiment_id"].isin(test_ids)]
    return train_df, test_df


def test_every_registered_model_fits_and_predicts():
    train_df, test_df = _train_test_frames()
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]

    for name, factory in MODEL_REGISTRY.items():
        pipeline = factory()
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        assert len(predictions) == len(X_test), f"{name}: prediction length mismatch"
        assert np.all(np.isfinite(predictions)), f"{name}: produced non-finite predictions"


def test_factories_produce_independent_unfitted_instances():
    """Calling a factory twice must not share fitted state -- each call
    should return a brand-new, unfitted Pipeline."""
    train_df, _ = _train_test_frames()
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]

    factory = MODEL_REGISTRY["random_forest"]
    pipeline_a = factory()
    pipeline_a.fit(X_train, y_train)

    pipeline_b = factory()
    assert pipeline_a is not pipeline_b
    assert not hasattr(pipeline_b.named_steps["model"], "estimators_"), \
        "a freshly-built pipeline must not already be fitted"


def test_random_forest_predictions_are_reproducible_given_fixed_seed():
    train_df, test_df = _train_test_frames()
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]

    pipeline_a = MODEL_REGISTRY["random_forest"]()
    pipeline_a.fit(X_train, y_train)
    preds_a = pipeline_a.predict(X_test)

    pipeline_b = MODEL_REGISTRY["random_forest"]()
    pipeline_b.fit(X_train, y_train)
    preds_b = pipeline_b.predict(X_test)

    assert np.array_equal(preds_a, preds_b), "same data + same RANDOM_SEED must give identical predictions"


def test_gradient_boosting_predictions_are_reproducible_given_fixed_seed():
    train_df, test_df = _train_test_frames()
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]

    preds = []
    for _ in range(2):
        pipeline = MODEL_REGISTRY["gradient_boosting"]()
        pipeline.fit(X_train, y_train)
        preds.append(pipeline.predict(X_test))
    assert np.array_equal(preds[0], preds[1])


def test_random_seed_is_fixed_not_left_to_chance():
    assert isinstance(RANDOM_SEED, int)
