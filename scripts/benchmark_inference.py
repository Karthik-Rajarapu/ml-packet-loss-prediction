#!/usr/bin/env python3
"""Phase 8 Step 10/15: SOFTWARE inference-latency and load benchmark,
using the TEST FIXTURE model (ml.demo).

    python3 scripts/benchmark_inference.py --n 200
    python3 scripts/benchmark_inference.py --n 200 --concurrency 4

This measures how fast this machine's Python process can run
InferenceEngine.predict() in a loop. It is NOT network prediction
latency, NOT a claim about production model performance, and NOT a
claim about how many real concurrent users the system could serve --
see docs/PHASE_8_INTEGRATION_AND_PERFORMANCE.md Section 8.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ml.demo import DEMO_CURRENT_METRICS, build_test_fixture_model  # noqa: E402
from ml.inference import InferenceEngine  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=200, help="Number of sequential predictions to time")
    parser.add_argument("--concurrency", type=int, default=1,
                         help="If > 1, also run this many predictions concurrently via a thread pool")
    return parser.parse_args()


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = min(len(sorted_values) - 1, int(round(pct / 100 * (len(sorted_values) - 1))))
    return sorted_values[idx]


def run_sequential(engine: InferenceEngine, n: int) -> list[float]:
    latencies_ms = []
    for _ in range(n):
        start = time.perf_counter()
        engine.predict(DEMO_CURRENT_METRICS)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    return latencies_ms


def run_concurrent(engine: InferenceEngine, n: int, concurrency: int) -> float:
    def _one():
        engine.predict(DEMO_CURRENT_METRICS)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(lambda _: _one(), range(n)))
    total_s = time.perf_counter() - start
    return n / total_s if total_s > 0 else float("inf")


def main() -> int:
    args = parse_args()

    print("=" * 70)
    print("SOFTWARE INFERENCE BENCHMARK -- TEST FIXTURE MODEL ONLY")
    print("Not network prediction latency. Not a production performance claim.")
    print("=" * 70)

    with tempfile.TemporaryDirectory(prefix="packet_loss_benchmark_fixture_") as tmp_dir:
        handle = build_test_fixture_model(Path(tmp_dir))
        engine = InferenceEngine(handle)

        # warm-up (exclude JIT/first-call overhead from the reported numbers)
        for _ in range(5):
            engine.predict(DEMO_CURRENT_METRICS)

        latencies_ms = run_sequential(engine, args.n)
        latencies_ms.sort()

        print(f"\nSequential predictions: n={args.n}")
        print(f"  mean:   {statistics.mean(latencies_ms):.3f} ms")
        print(f"  median: {statistics.median(latencies_ms):.3f} ms")
        print(f"  p95:    {_percentile(latencies_ms, 95):.3f} ms")
        print(f"  min:    {latencies_ms[0]:.3f} ms")
        print(f"  max:    {latencies_ms[-1]:.3f} ms")

        if args.concurrency > 1:
            throughput = run_concurrent(engine, args.n, args.concurrency)
            print(f"\nConcurrent throughput (concurrency={args.concurrency}, n={args.n}):")
            print(f"  {throughput:.1f} predictions/sec (thread pool -- engineering benchmark only, "
                  f"NOT a claim about real concurrent network users)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
