"""TEST FIXTURE model builder + example input for demos/tests.

Everything in this module is explicitly synthetic. It exists so
scripts/predict_packet_loss.py --use-test-fixture and the test suite can
exercise the full inference path (loading, validation, prediction, risk
classification) without a real trained production model -- see Strict
Rule #4: "NEVER call a test fixture a production model." Every artifact
this module produces has is_test_fixture=True baked into its metadata,
and ml.inference.load_test_fixture_model() refuses to load anything that
isn't marked that way.

This is intentionally self-contained (does not import tests/ml_fixtures.py):
src/ never depends on tests/, so the small synthetic-row generator below
is a separate, deliberately tiny duplication rather than a backwards
src-depends-on-tests import.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.artifacts import ModelMetadata, save_metadata, save_model
from ml.inference import ModelHandle
from ml.models import MODEL_REGISTRY
from network.schema import CSV_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN
from network.targets import add_next_interval_target, drop_rows_without_target

TEST_FIXTURE_MODEL_NAME = "TEST_FIXTURE_decision_tree"

# One illustrative example of "current network metrics" for --demo. These
# are hand-picked, clearly-synthetic example values -- not a real
# measurement -- used only to exercise the CLI end-to-end.
DEMO_CURRENT_METRICS: dict[str, float | str] = {
    "bottleneck_bw_mbps": 2.0,
    "bottleneck_delay_ms": 20.0,
    "bottleneck_queue_pkts": 20,
    "traffic_type": "udp",
    "active_connections": 1,
    "current_rtt_ms": 25.0,
    "current_jitter_ms": 3.0,
    "throughput_mbps": 2.4,
    "bandwidth_utilization_pct": 120.0,
    "packet_rate_pps": 350.0,
    "queue_length": 18,
    "retransmissions": 0,
    "packets_sent": 700,
    "packets_received": 640,
    "packet_loss_pct": 8.5,
}


def _tiny_synthetic_rows() -> list[dict]:
    """A handful of small, deterministic, clearly-synthetic rows -- enough
    for a DecisionTreeRegressor to fit and produce a real (if
    meaningless) prediction. NOT real network data."""
    rows = []
    for exp_i in range(3):
        experiment_id = f"demo-fixture-exp-{exp_i}"
        for interval_i in range(6):
            loss = min(20.0, interval_i * (1.0 + exp_i))
            rows.append({
                "experiment_id": experiment_id, "run_id": experiment_id,
                "interval_index": interval_i, "timestamp": exp_i * 100.0 + interval_i * 2.0,
                "source_host": "h1", "destination_host": "h3",
                "bottleneck_bw_mbps": 1.0 + exp_i, "bottleneck_delay_ms": 5.0 * exp_i,
                "bottleneck_queue_pkts": 20, "traffic_type": "udp", "active_connections": 1,
                "current_rtt_ms": 5.0 + interval_i, "current_jitter_ms": 0.5 + interval_i * 0.2,
                "throughput_mbps": (1.0 + exp_i) * 0.8, "bandwidth_utilization_pct": 80.0,
                "packet_rate_pps": 300.0 + interval_i * 10, "queue_length": interval_i,
                "retransmissions": 0, "packets_sent": 700,
                "packets_received": 700 - int(loss * 7), "packet_loss_pct": loss,
            })
    return rows


def build_test_fixture_model(output_dir: Path) -> ModelHandle:
    """Trains a tiny DecisionTreeRegressor on synthetic data and saves it
    to `output_dir` with is_test_fixture=True. Returns a ModelHandle ready
    to use immediately (no need to reload from disk).

    `output_dir` must NEVER be the real models/ directory -- callers
    (scripts/predict_packet_loss.py) are responsible for pointing this at
    a clearly separate location (e.g. a temp directory), never at
    production model storage.
    """
    rows = drop_rows_without_target(add_next_interval_target(_tiny_synthetic_rows()))
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)

    pipeline = MODEL_REGISTRY["decision_tree"]()
    pipeline.fit(df[FEATURE_COLUMNS], df[TARGET_COLUMN])

    joblib_path = output_dir / f"{TEST_FIXTURE_MODEL_NAME}.joblib"
    metadata_path = output_dir / f"{TEST_FIXTURE_MODEL_NAME}_metadata.json"
    save_model(pipeline, joblib_path)
    metadata = ModelMetadata(
        model_name=TEST_FIXTURE_MODEL_NAME,
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
        is_test_fixture=True,
        training_config={"note": "synthetic fixture data, not a real Mininet dataset"},
        metrics={},
    )
    save_metadata(metadata, metadata_path)

    return ModelHandle(pipeline=pipeline, metadata=metadata, source_path=joblib_path, is_test_fixture=True)
