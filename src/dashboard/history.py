"""In-session prediction history.

Pure Python container -- app.py stores ONE instance of PredictionHistory
inside st.session_state so it survives Streamlit reruns within a browser
session, but nothing here imports Streamlit or touches disk (Phase 7
Step 8: "Do not persist fake historical data to disk").

`actual_packet_loss_pct` is optional and defaults to None: a
next-interval prediction's true outcome is not known at prediction time
in this manual/batch-upload dashboard -- there is no live feed replaying
ground truth. It can ONLY ever be set afterward by an explicit user
action (record_actual(), Phase 9 Step 13's "if actual future packet loss
becomes available") -- the software never fills it in on its own.
Inventing a value here would mean fabricating a network measurement,
which Strict Rule #1 forbids outright.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    actual_packet_loss_pct: float | None = None  # user-supplied only, see module docstring


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

    def record_actual(self, index: int, actual_packet_loss_pct: float) -> None:
        """Attach a user-reported actual outcome to entry `index` (as
        returned by `entries`/`to_dataframe` row order). The ONLY way an
        `actual_packet_loss_pct` value is ever set -- always an explicit
        call with a caller-supplied number, never inferred or guessed."""
        if not (0 <= index < len(self._entries)):
            raise IndexError(f"No history entry at index {index}")
        self._entries[index] = replace(self._entries[index], actual_packet_loss_pct=float(actual_packet_loss_pct))

    def to_dataframe(self) -> pd.DataFrame:
        columns = ["timestamp", "model_name", "model_kind", "is_test_fixture",
                   "predicted_packet_loss_pct", "risk_level", "actual_packet_loss_pct", "absolute_error"]
        if not self._entries:
            return pd.DataFrame(columns=columns)
        rows = []
        for e in self._entries:
            abs_error = (
                abs(e.predicted_packet_loss_pct - e.actual_packet_loss_pct)
                if e.actual_packet_loss_pct is not None else None
            )
            rows.append({
                "timestamp": e.timestamp,
                "model_name": e.model_name,
                "model_kind": e.model_kind,
                "is_test_fixture": e.is_test_fixture,
                "predicted_packet_loss_pct": e.predicted_packet_loss_pct,
                "risk_level": e.risk_level,
                "actual_packet_loss_pct": e.actual_packet_loss_pct,
                "absolute_error": abs_error,
            })
        return pd.DataFrame(rows, columns=columns)

    def entries_with_actuals(self) -> pd.DataFrame:
        """Subset of history rows that have a user-recorded actual value
        -- the only rows an actual-vs-predicted chart may ever be built
        from (Phase 9 Step 13: only when actual values genuinely exist)."""
        df = self.to_dataframe()
        return df[df["actual_packet_loss_pct"].notna()].reset_index(drop=True)
