#!/usr/bin/env python3
"""Phase 8 Step 16: lightweight system/health validation.

    python3 scripts/system_check.py

Checks Python dependencies, project imports, expected directories, model
availability, inference availability, dashboard importability, and
network-tool availability -- and clearly separates what's REQUIRED for
the software system to run from what's REQUIRED FOR LIVE EXPERIMENTS
only (Mininet/iperf3/tc/Open vSwitch). Exits non-zero ONLY if a REQUIRED
software check fails; a missing Mininet environment never fails this
check on its own -- see docs/ENVIRONMENT_SETUP.md for that separate
concern.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

STATUS_OK = "AVAILABLE"
STATUS_MISSING = "NOT AVAILABLE"


@dataclass
class CheckResult:
    name: str
    category: str  # "REQUIRED" | "OPTIONAL" | "REQUIRED FOR LIVE EXPERIMENTS" | "INFORMATIONAL"
    status: str
    detail: str = ""

    @property
    def is_required_failure(self) -> bool:
        return self.category == "REQUIRED" and self.status == STATUS_MISSING


def check_python_version() -> CheckResult:
    ok = sys.version_info >= (3, 10)
    return CheckResult("Python >= 3.10", "REQUIRED", STATUS_OK if ok else STATUS_MISSING,
                        f"found {sys.version.split()[0]}")


def check_package(name: str, import_name: str | None = None) -> CheckResult:
    try:
        mod = importlib.import_module(import_name or name)
        version = getattr(mod, "__version__", "unknown version")
        return CheckResult(f"Python package: {name}", "REQUIRED", STATUS_OK, version)
    except ImportError as exc:
        return CheckResult(f"Python package: {name}", "REQUIRED", STATUS_MISSING, str(exc))


def check_project_imports() -> list[CheckResult]:
    modules = [
        "network.schema", "network.config", "network.topology", "network.experiment",
        "network.validation", "network.targets", "network.collectors", "network.sweep",
        "network.generate", "network.manifest",
        "ml.dataset", "ml.split", "ml.preprocessing", "ml.baselines", "ml.models",
        "ml.evaluate", "ml.risk", "ml.artifacts", "ml.train", "ml.inference", "ml.demo",
        "ml.data_quality",
        "dashboard.feature_inputs", "dashboard.model_status", "dashboard.history", "dashboard.reports",
    ]
    results = []
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            results.append(CheckResult(f"import {module_name}", "REQUIRED", STATUS_OK))
        except Exception as exc:  # noqa: BLE001 -- want to report every broken import, not stop at the first
            results.append(CheckResult(f"import {module_name}", "REQUIRED", STATUS_MISSING, str(exc)))
    return results


def check_dashboard_importable() -> CheckResult:
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("packet_loss_dashboard_app_check", REPO_ROOT / "app.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert callable(module.main)
        return CheckResult("app.py importable", "REQUIRED", STATUS_OK)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("app.py importable", "REQUIRED", STATUS_MISSING, str(exc))


def check_directories() -> list[CheckResult]:
    expected = ["data/raw", "data/manifests", "models", "reports/modeling", "configs", "docs", "tests"]
    results = []
    for rel_path in expected:
        path = REPO_ROOT / rel_path
        results.append(CheckResult(f"directory: {rel_path}", "REQUIRED",
                                    STATUS_OK if path.is_dir() else STATUS_MISSING))
    return results


def check_model_availability() -> CheckResult:
    from dashboard.model_status import check_production_model
    status = check_production_model(REPO_ROOT / "models")
    return CheckResult("production model", "INFORMATIONAL",
                        STATUS_OK if status.available else STATUS_MISSING,
                        status.message or f"model: {status.handle.metadata.model_name}")


def check_inference_availability() -> CheckResult:
    """Proves the inference ENGINE works right now, using the TEST
    FIXTURE model -- this is a software capability check, not a claim
    about production predictions."""
    try:
        import tempfile
        from ml.demo import DEMO_CURRENT_METRICS, build_test_fixture_model
        from ml.inference import InferenceEngine
        with tempfile.TemporaryDirectory() as tmp_dir:
            handle = build_test_fixture_model(Path(tmp_dir))
            InferenceEngine(handle).predict(DEMO_CURRENT_METRICS)
        return CheckResult("inference engine (test fixture)", "REQUIRED", STATUS_OK)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("inference engine (test fixture)", "REQUIRED", STATUS_MISSING, str(exc))


def check_network_tools() -> list[CheckResult]:
    from network.validation import check_environment
    env = check_environment()
    return [
        CheckResult(f"network tool: {tool}", "REQUIRED FOR LIVE EXPERIMENTS",
                    STATUS_OK if available else STATUS_MISSING)
        for tool, available in env.available.items()
    ]


def main() -> int:
    checks: list[CheckResult] = []
    checks.append(check_python_version())
    for pkg, import_name in [("pandas", None), ("numpy", None), ("scikit-learn", "sklearn"),
                              ("matplotlib", None), ("joblib", None), ("streamlit", None), ("pytest", None)]:
        checks.append(check_package(pkg, import_name))
    checks.extend(check_project_imports())
    checks.append(check_dashboard_importable())
    checks.extend(check_directories())
    checks.append(check_model_availability())
    checks.append(check_inference_availability())
    checks.extend(check_network_tools())

    by_category: dict[str, list[CheckResult]] = {}
    for check in checks:
        by_category.setdefault(check.category, []).append(check)

    print("=" * 70)
    print("SYSTEM HEALTH CHECK")
    print("=" * 70)
    for category in ["REQUIRED", "INFORMATIONAL", "REQUIRED FOR LIVE EXPERIMENTS", "OPTIONAL"]:
        results = by_category.get(category, [])
        if not results:
            continue
        print(f"\n[{category}]")
        for r in results:
            marker = "OK  " if r.status == STATUS_OK else "MISS"
            detail = f" -- {r.detail}" if r.detail else ""
            print(f"  [{marker}] {r.name}: {r.status}{detail}")

    required_failures = [c for c in checks if c.is_required_failure]
    print()
    print("=" * 70)
    if required_failures:
        print(f"SOFTWARE HEALTH CHECK: FAIL ({len(required_failures)} required check(s) failed)")
    else:
        print("SOFTWARE HEALTH CHECK: PASS")
    live_missing = [c for c in checks if c.category == "REQUIRED FOR LIVE EXPERIMENTS" and c.status == STATUS_MISSING]
    if live_missing:
        print(f"LIVE MININET EXPERIMENTS: NOT AVAILABLE ({len(live_missing)} tool(s) missing) -- "
              "this does NOT fail the software health check; see docs/ENVIRONMENT_SETUP.md")
    print("=" * 70)

    return 1 if required_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
