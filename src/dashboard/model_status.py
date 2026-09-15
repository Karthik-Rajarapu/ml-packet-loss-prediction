"""Production-model availability check for the dashboard.

Pure Python (no Streamlit import) so it's directly unit-testable, and so
app.py never has to touch ml.inference's loading logic itself -- it only
ever reads the ModelStatus this module produces. NEVER falls back to a
test fixture: check_production_model() either returns a real,
production-shaped handle or explains exactly why not.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ml.inference import ModelArtifactError, ModelHandle, load_production_model


@dataclass(frozen=True)
class ModelStatus:
    available: bool
    handle: ModelHandle | None
    message: str | None


def check_production_model(models_dir: Path) -> ModelStatus:
    """Never raises -- callers (app.py) always get a ModelStatus to
    branch on, whether or not a production model exists."""
    try:
        handle = load_production_model(models_dir)
        return ModelStatus(available=True, handle=handle, message=None)
    except ModelArtifactError as exc:
        return ModelStatus(available=False, handle=None, message=str(exc))
