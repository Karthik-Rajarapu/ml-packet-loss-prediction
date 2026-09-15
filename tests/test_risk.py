import pytest

from ml.risk import RiskThresholds, classify_risk


def test_default_thresholds_boundaries():
    values = [0.0, 0.1, 0.1001, 1.0, 1.0001, 50.0]
    labels = list(classify_risk(values))
    assert labels == ["LOW", "LOW", "MODERATE", "MODERATE", "HIGH", "HIGH"]


def test_custom_thresholds_are_respected():
    thresholds = RiskThresholds(low_max=2.0, moderate_max=10.0)
    labels = list(classify_risk([1.0, 5.0, 20.0], thresholds=thresholds))
    assert labels == ["LOW", "MODERATE", "HIGH"]


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        RiskThresholds(low_max=5.0, moderate_max=1.0)  # low_max must be < moderate_max
    with pytest.raises(ValueError):
        RiskThresholds(low_max=-1.0, moderate_max=1.0)  # must be non-negative


def test_classify_risk_returns_series_named_risk_level():
    result = classify_risk([0.0])
    assert result.name == "risk_level"
    assert len(result) == 1
