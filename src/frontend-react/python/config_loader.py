import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def get_config() -> dict:
    """
    Load Pulumi-generated frontend configuration if available.

    Returns:
        Dict with configuration values or empty dict when the file is absent.
    """
    config_path = Path(__file__).resolve().parents[1] / ".pulumi-config.json"
    if not config_path.exists():
        return {}

    try:
        with config_path.open() as f:
            return json.load(f)
    except Exception:
        # Corrupt/partial file should not block the frontend – fall back to defaults.
        return {}
