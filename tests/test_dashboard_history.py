import pytest

from dashboard.history import PredictionHistory, PredictionHistoryEntry


def _entry(loss=5.0, risk="MODERATE", is_test_fixture=True):
    return PredictionHistoryEntry(
        timestamp="2026-01-01T00:00:00+00:00", model_name="TEST_FIXTURE_decision_tree",
        model_kind="ml_model", is_test_fixture=is_test_fixture,
        predicted_packet_loss_pct=loss, risk_level=risk,
        input_features={"packet_loss_pct": 4.0},
    )


def test_empty_history_has_zero_length_and_empty_dataframe():
    history = PredictionHistory()
    assert len(history) == 0
    df = history.to_dataframe()
    assert df.empty
    assert "predicted_packet_loss_pct" in df.columns


def test_add_increases_length_and_appears_in_dataframe():
    history = PredictionHistory()
    history.add(_entry(loss=7.5, risk="HIGH"))
    assert len(history) == 1
    df = history.to_dataframe()
    assert len(df) == 1
    assert df.iloc[0]["predicted_packet_loss_pct"] == 7.5
    assert df.iloc[0]["risk_level"] == "HIGH"


def test_actual_loss_column_exists_but_starts_unset():
    """Superseded by the Phase 9 'record actual outcome' feature: the
    column now exists (so a real, user-reported outcome can be attached
    later), but the strict no-fabrication rule still holds -- it must
    never be auto-populated. Only an explicit record_actual() call may
    set it (see the next tests)."""
    history = PredictionHistory()
    history.add(_entry())
    df = history.to_dataframe()
    assert "actual_packet_loss_pct" in df.columns
    assert df.iloc[0]["actual_packet_loss_pct"] is None
    assert df.iloc[0]["absolute_error"] is None


def test_record_actual_sets_value_and_absolute_error():
    history = PredictionHistory()
    history.add(_entry(loss=6.0))
    history.record_actual(0, actual_packet_loss_pct=5.0)
    df = history.to_dataframe()
    assert df.iloc[0]["actual_packet_loss_pct"] == 5.0
    assert df.iloc[0]["absolute_error"] == pytest.approx(1.0)


def test_record_actual_invalid_index_raises():
    history = PredictionHistory()
    with pytest.raises(IndexError):
        history.record_actual(0, actual_packet_loss_pct=1.0)


def test_entries_with_actuals_only_includes_recorded_rows():
    history = PredictionHistory()
    history.add(_entry(loss=1.0))
    history.add(_entry(loss=2.0))
    history.record_actual(1, actual_packet_loss_pct=2.5)
    subset = history.entries_with_actuals()
    assert len(subset) == 1
    assert subset.iloc[0]["predicted_packet_loss_pct"] == 2.0


def test_clear_empties_the_history():
    history = PredictionHistory()
    history.add(_entry())
    history.add(_entry())
    assert len(history) == 2
    history.clear()
    assert len(history) == 0
    assert history.to_dataframe().empty


def test_entries_preserves_order_and_is_a_copy():
    history = PredictionHistory()
    history.add(_entry(loss=1.0))
    history.add(_entry(loss=2.0))
    entries = history.entries
    assert [e.predicted_packet_loss_pct for e in entries] == [1.0, 2.0]
    entries.append(_entry(loss=3.0))
    assert len(history) == 2, "history.entries must return a copy, not the live internal list"


def test_history_bounds_unbounded_growth():
    history = PredictionHistory(max_entries=10)
    for i in range(25):
        history.add(_entry(loss=float(i)))
    assert len(history) == 10
    # oldest entries dropped, most recent kept
    remaining = [e.predicted_packet_loss_pct for e in history.entries]
    assert remaining == [float(i) for i in range(15, 25)]


def test_multiple_entries_preserve_insertion_order_in_dataframe():
    history = PredictionHistory()
    for i in range(5):
        history.add(_entry(loss=float(i)))
    df = history.to_dataframe()
    assert list(df["predicted_packet_loss_pct"]) == [0.0, 1.0, 2.0, 3.0, 4.0]
