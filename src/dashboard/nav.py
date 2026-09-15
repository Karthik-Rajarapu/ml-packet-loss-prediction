"""Shared navigation labels -- one source of truth so app.py's sidebar
and any page that programmatically switches pages (e.g. Home's "Get
Started" button) never drift out of sync."""

NAV_HOME = "Home"
NAV_UPLOAD = "Upload Data"
NAV_PREPARE = "Prepare Dataset"
NAV_TRAIN = "Train Model"
NAV_PREDICT = "Predict"
NAV_HISTORY = "History"

NAV_ICONS = {
    NAV_HOME: "\U0001F3E0",
    NAV_UPLOAD: "\U0001F4E4",
    NAV_PREPARE: "\U0001F9EA",
    NAV_TRAIN: "\U0001F3AF",
    NAV_PREDICT: "\U0001F52E",
    NAV_HISTORY: "\U0001F553",
}

NAV_ORDER = [NAV_HOME, NAV_UPLOAD, NAV_PREPARE, NAV_TRAIN, NAV_PREDICT, NAV_HISTORY]


def nav_label(page: str) -> str:
    return f"{NAV_ICONS.get(page, '')}  {page}"
