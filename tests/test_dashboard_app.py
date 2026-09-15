"""Tests for app.py (Phase 9 redesign): the file is now a thin
navigation shell -- routing logic and page rendering live in
src/dashboard/pages/*, tested separately (test_dashboard_pages_*.py).
This file only checks the shell itself: it imports cleanly, its cached
helpers work, and its paths/navigation wiring are correct.

app.py's `if __name__ == "__main__": main()` guard means a plain import
here never renders the UI or touches Streamlit's script-run machinery.
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_app_module():
    spec = importlib.util.spec_from_file_location("packet_loss_dashboard_app", REPO_ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


app = _load_app_module()


def test_application_imports_without_executing_main():
    assert callable(app.main)


def test_paths_point_at_the_real_repo_locations():
    assert app.MODELS_DIR == REPO_ROOT / "models"
    assert app.REPORTS_DIR == REPO_ROOT / "reports" / "modeling"


def test_nav_order_covers_every_page():
    from dashboard.nav import NAV_HISTORY, NAV_HOME, NAV_ORDER, NAV_PREDICT, NAV_PREPARE, NAV_TRAIN, NAV_UPLOAD
    assert NAV_ORDER == [NAV_HOME, NAV_UPLOAD, NAV_PREPARE, NAV_TRAIN, NAV_PREDICT, NAV_HISTORY]


def test_cached_model_status_reports_unavailable_for_empty_dir(tmp_path):
    status = app._cached_model_status(tmp_path)
    assert status.available is False


def test_cached_test_fixture_handle_builds_a_usable_handle():
    handle = app._cached_test_fixture_handle()
    assert handle.is_test_fixture is True
    assert handle.pipeline is not None


def test_main_does_not_run_on_plain_import():
    """A regression guard: main() must stay behind the __main__ guard --
    if it ran on import, every test importing app.py would try to touch
    Streamlit's full script-run machinery."""
    import inspect
    source = inspect.getsource(app)
    assert 'if __name__ == "__main__":' in source
    assert source.rstrip().endswith("main()")
