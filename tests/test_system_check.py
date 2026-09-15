"""Tests for scripts/system_check.py -- loaded by file path like the
other scripts/ CLIs (no __init__.py in scripts/). Must never fail the
overall check merely because Mininet is unavailable (Phase 8 Step 16's
explicit requirement).
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location("system_check_module", REPO_ROOT / "scripts" / "system_check.py")
    module = importlib.util.module_from_spec(spec)
    # Must be registered in sys.modules BEFORE exec_module: system_check.py
    # defines @dataclass classes under `from __future__ import annotations`,
    # and Python's dataclass machinery resolves string annotations via
    # sys.modules[cls.__module__] -- without this line that lookup returns
    # None and dataclass() raises AttributeError. Found by actually running
    # this test, not by inspection.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sc = _load_module()


def test_python_version_check_passes_on_this_interpreter():
    result = sc.check_python_version()
    assert result.status == sc.STATUS_OK


def test_all_project_imports_succeed():
    results = sc.check_project_imports()
    failures = [r for r in results if r.status == sc.STATUS_MISSING]
    assert not failures, f"broken project import(s): {failures}"


def test_dashboard_importable_check_passes():
    result = sc.check_dashboard_importable()
    assert result.status == sc.STATUS_OK


def test_all_expected_directories_exist():
    results = sc.check_directories()
    failures = [r for r in results if r.status == sc.STATUS_MISSING]
    assert not failures, f"missing expected directories: {failures}"


def test_inference_availability_check_passes_via_test_fixture():
    result = sc.check_inference_availability()
    assert result.status == sc.STATUS_OK


def test_model_availability_is_informational_not_required():
    """Even with no production model on this machine, this check must be
    categorized INFORMATIONAL, never REQUIRED -- it must not be able to
    fail the overall health check."""
    result = sc.check_model_availability()
    assert result.category == "INFORMATIONAL"
    assert result.is_required_failure is False


def test_network_tools_are_categorized_for_live_experiments_only():
    results = sc.check_network_tools()
    assert results
    for r in results:
        assert r.category == "REQUIRED FOR LIVE EXPERIMENTS"
        assert r.is_required_failure is False, (
            "a missing network tool must never count as a REQUIRED software failure"
        )


def test_main_exits_zero_even_though_mininet_and_model_are_unavailable():
    """The real, current state of this machine: no Mininet, no
    production model. main() must still return 0 (PASS) since every
    REQUIRED software check succeeds."""
    exit_code = sc.main()
    assert exit_code == 0
