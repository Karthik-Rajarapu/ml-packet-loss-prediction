"""Phase 8 Step 10: lightweight latency regression guard using the TEST
FIXTURE model. This is a SOFTWARE latency check, not a network
prediction latency claim -- see scripts/benchmark_inference.py for the
full benchmark (real numbers reported in
docs/PHASE_8_INTEGRATION_AND_PERFORMANCE.md, not fabricated here).

The threshold below is deliberately generous (an order of magnitude
above what was actually measured on the dev machine) -- this test exists
to catch a catastrophic regression (e.g. accidentally rebuilding the
model on every call), not to enforce a specific performance target.
"""

import statistics
import time

from ml.demo import DEMO_CURRENT_METRICS, build_test_fixture_model
from ml.inference import InferenceEngine

GENEROUS_MEDIAN_LATENCY_MS = 500.0  # measured ~12ms on dev hardware -- see PHASE_8 docs


def test_sequential_inference_latency_is_reasonable(tmp_path):
    handle = build_test_fixture_model(tmp_path)
    engine = InferenceEngine(handle)

    for _ in range(3):  # warm-up
        engine.predict(DEMO_CURRENT_METRICS)

    latencies_ms = []
    for _ in range(30):
        start = time.perf_counter()
        engine.predict(DEMO_CURRENT_METRICS)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    median = statistics.median(latencies_ms)
    assert median < GENEROUS_MEDIAN_LATENCY_MS, (
        f"median inference latency ({median:.1f}ms) far exceeds the generous regression "
        f"threshold ({GENEROUS_MEDIAN_LATENCY_MS}ms) -- likely a real performance regression, "
        f"e.g. the model being rebuilt/reloaded on every call"
    )


def test_repeated_predictions_do_not_accumulate_state():
    """A cheap check that repeated calls don't leak growing state into
    the pipeline/engine itself (as opposed to the dashboard's own
    explicitly-bounded PredictionHistory, tested separately)."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_dir:
        handle = build_test_fixture_model(Path(tmp_dir))
    engine = InferenceEngine(handle)

    results = [engine.predict(DEMO_CURRENT_METRICS) for _ in range(20)]
    predictions = {r.predicted_packet_loss_pct for r in results}
    assert len(predictions) == 1, "identical repeated input must keep producing identical output"
