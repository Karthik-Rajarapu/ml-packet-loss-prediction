"""Tests for dashboard.session.AppSession -- the dataclass itself is
plain Python (no Streamlit needed to construct/inspect it); get_session()
needs a Streamlit session_state, exercised via app.py's own bare-mode
tests instead (tests/test_dashboard_app.py)."""

from dashboard.history import PredictionHistory
from dashboard.session import AppSession


def test_app_session_defaults_are_empty_not_none_for_containers():
    session = AppSession()
    assert session.raw_df is None
    assert session.column_mapping == {}
    assert session.constant_values == {}
    assert isinstance(session.history, PredictionHistory)
    assert len(session.history) == 0


def test_app_session_instances_do_not_share_mutable_defaults():
    """A classic Python dataclass-mutable-default bug would make every
    AppSession share the same dict/history -- must not happen here."""
    a = AppSession()
    b = AppSession()
    a.column_mapping["current_rtt_ms"] = "rtt"
    assert b.column_mapping == {}
    assert a.history is not b.history
