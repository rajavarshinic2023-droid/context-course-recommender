"""Environment-based configuration. Secrets are only ever read from the environment."""
import os

try:  # .env is optional; the app also works with real environment variables
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
_PLACEHOLDERS = {"your-gemini-api-key-here", "changeme", "your-key-here"}


def get_setting(name, default=None):
    """Return a stripped environment value, or default when unset/empty/placeholder."""
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    if not value or value.lower() in _PLACEHOLDERS:
        return default
    return value


def get_float_setting(name, default):
    try:
        return float(get_setting(name, default))
    except (TypeError, ValueError):
        return float(default)
