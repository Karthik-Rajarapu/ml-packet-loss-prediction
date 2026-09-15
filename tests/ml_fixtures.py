"""Synthetic TEST FIXTURES for the Phase 4 ML pipeline tests.

These rows are NOT real Mininet measurements -- they are small,
deterministic, hand-constructed values used only to check that the
pipeline's plumbing (split, preprocess, fit, predict, save/load,
metrics) is wired correctly. No test in this repo may use this fixture
to report or imply real-world model performance; see
docs/PHASE_4_ML_MODELING.md Section 1.
"""

from __future__ import annotations

import pandas as pd

from network.schema import CSV_COLUMNS
from network.targets import add_next_interval_target


def make_fixture_rows(n_experiments: int = 4, n_intervals: int = 6, id_prefix: str = "fixture-exp") -> list[dict]:
    """Deterministic synthetic rows across `n_experiments` experiments,
    each with `n_intervals` intervals. packet_loss_pct rises slightly
    within each experiment so there is SOME learnable signal for
    plumbing tests (e.g. "predictions vary", "metrics aren't NaN") --
    still not meant to represent anything about real network behavior.

    `id_prefix` lets tests that combine multiple fixture "files" give each
    one distinct experiment_ids (real Phase 2 sweeps always do; two
    fixture calls with the default prefix would otherwise collide).
    """
    rows = []
    for exp_i in range(n_experiments):
        experiment_id = f"{id_prefix}-{exp_i}"
        base_bw = 1.0 + exp_i  # 1, 2, 3, 4 Mbps
        for interval_i in range(n_intervals):
            loss = min(20.0, interval_i * (1.0 + exp_i * 0.5))
            rows.append({
                "experiment_id": experiment_id,
                "run_id": experiment_id,
                "interval_index": interval_i,
                "timestamp": exp_i * 1000.0 + interval_i * 2.0,
                "source_host": "h1",
                "destination_host": "h3",
                "bottleneck_bw_mbps": base_bw,
                "bottleneck_delay_ms": 10.0 * exp_i,
                "bottleneck_queue_pkts": 20,
                "traffic_type": "udp" if exp_i % 2 == 0 else "tcp",
                "active_connections": 1,
                "current_rtt_ms": 5.0 + interval_i * 0.5,
                "current_jitter_ms": 0.5 + interval_i * 0.1,
                "throughput_mbps": base_bw * 0.8,
                "bandwidth_utilization_pct": 80.0,
                "packet_rate_pps": 300.0 + interval_i * 10,
                "queue_length": interval_i,
                "retransmissions": 0,
                "packets_sent": 700,
                "packets_received": 700 - int(loss * 7),
                "packet_loss_pct": loss,
            })
    return add_next_interval_target(rows)


def make_fixture_dataframe(n_experiments: int = 4, n_intervals: int = 6, id_prefix: str = "fixture-exp") -> pd.DataFrame:
    """Mirrors the exact shape of a real Phase 1 CSV (network.experiment's
    _write_csv): includes the trailing null-target row for each
    experiment_id, NOT pre-dropped. Use this for tests that exercise
    ml.dataset.load_dataset() itself, which is the component responsible
    for validating-then-dropping those rows -- a fixture that pre-drops
    them would no longer look like real input and would trip
    validate_no_leakage's own end-of-experiment check for the wrong
    reason (this was an actual bug here, caught by running the tests)."""
    rows = make_fixture_rows(n_experiments, n_intervals, id_prefix=id_prefix)
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def make_labeled_fixture_dataframe(n_experiments: int = 4, n_intervals: int = 6, id_prefix: str = "fixture-exp") -> pd.DataFrame:
    """Same data as make_fixture_dataframe, but with the trailing
    null-target rows already dropped -- i.e. what load_dataset() would
    hand back. Use this for tests that fit a model directly (they need
    every row to have a usable, non-null target) and don't care about
    exercising the CSV-loading/validation step itself."""
    from network.targets import drop_rows_without_target
    rows = drop_rows_without_target(make_fixture_rows(n_experiments, n_intervals, id_prefix=id_prefix))
    return pd.DataFrame(rows, columns=CSV_COLUMNS)
