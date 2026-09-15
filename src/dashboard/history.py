"""In-session prediction history.

Pure Python container -- app.py stores ONE instance of PredictionHistory
inside st.session_state so it survives Streamlit reruns within a browser
session, but nothing here imports Streamlit or touches disk (Phase 7
Step 8: "Do not persist fake historical data to disk").

Each entry deliberately has NO "actual packet loss" field: a next-interval
prediction's true outcome is, by definition, not known at prediction
time in this manual-input dashboard (there is no live feed replaying
ground truth yet -- that is Phase 8 territory, see
docs/PHASE_7_STREAMLIT_DASHBOARD.md Section 8). Inventing an "actual"
column here would mean fabricating a network measurement, which Strict
Rule #1 forbids outright.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class PredictionHistoryEntry:
    timestamp: str
    model_name: str
    model_kind: str  # "ml_model" | "naive_baseline"
    is_test_fixture: bool
    predicted_packet_loss_pct: float
    risk_level: str
    input_features: dict[str, Any]


class PredictionHistory:
    def __init__(self, max_entries: int = 500) -> None:
        """max_entries bounds in-memory growth for a long-running browser
        session (Phase 8 Step 9/11: 'uncontrolled history growth') --
        oldest entries are dropped once the cap is exceeded, since this
        is a live demo aid, not a persisted record (Step 8: never
        persisted to disk in the first place)."""
        self._entries: list[PredictionHistoryEntry] = []
        self._max_entries = max_entries

    def add(self, entry: PredictionHistoryEntry) -> None:
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    @property
    def entries(self) -> list[PredictionHistoryEntry]:
        return list(self._entries)

    def to_dataframe(self) -> pd.DataFrame:
        columns = ["timestamp", "model_name", "model_kind", "is_test_fixture",
                   "predicted_packet_loss_pct", "risk_level"]
        if not self._entries:
            return pd.DataFrame(columns=columns)
        rows = [{
            "timestamp": e.timestamp,
            "model_name": e.model_name,
            "model_kind": e.model_kind,
            "is_test_fixture": e.is_test_fixture,
            "predicted_packet_loss_pct": e.predicted_packet_loss_pct,
            "risk_level": e.risk_level,
        } for e in self._entries]
        return pd.DataFrame(rows, columns=columns)
