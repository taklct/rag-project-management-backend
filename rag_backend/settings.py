"""Application configuration and shared constants."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:  # pragma: no cover - defensive import guard
    import config  # type: ignore
except Exception:  # pragma: no cover - defensive
    config = None  # type: ignore


@dataclass(frozen=True)
class AzureSettings:
    """Configuration required to communicate with Azure OpenAI."""

    endpoint_url: str
    chat_deployment: str
    embedding_deployment: str
    api_key: str
    api_version: str = "2025-01-01-preview"


ENVIRONMENT_VARIABLES: Dict[str, str] = {
    "ENDPOINT_URL": "endpoint_url",
    "DEPLOYMENT_NAME": "chat_deployment",
    "EMBEDDING_DEPLOYMENT_NAME": "embedding_deployment",
    "AZURE_OPENAI_API_KEY": "api_key",
}


def _lookup_setting(name: str) -> Optional[Any]:
    """Fetch a configuration value from the environment or config module.

    Returns the first match from environment or ``config.py``. Values from
    environment are always strings; values from ``config.py`` may be any type.
    """

    env_value = os.getenv(name)
    if env_value:
        return env_value

    if config is not None and hasattr(config, name):
        value = getattr(config, name)
        if value is not None:
            return value
    return None


def load_azure_settings() -> AzureSettings:
    """Load Azure OpenAI credentials, raising an informative error when missing."""

    # Collect required string settings
    values: Dict[str, Optional[str]] = {}
    for env_name, alias in ENVIRONMENT_VARIABLES.items():
        raw = _lookup_setting(env_name)
        values[alias] = raw if isinstance(raw, str) and raw else None

    missing = [name for name, value in values.items() if not value]
    if missing:
        missing_display = ", ".join(sorted(missing))
        raise RuntimeError(
            "Please set ENDPOINT_URL, DEPLOYMENT_NAME, AZURE_OPENAI_API_KEY, "
            "EMBEDDING_DEPLOYMENT_NAME (env or config.py). Missing: "
            f"{missing_display}."
        )

    # Optional API version override from env/config
    api_version_raw = _lookup_setting("API_VERSION")
    if isinstance(api_version_raw, str) and api_version_raw:
        values["api_version"] = api_version_raw

    return AzureSettings(**values)  # type: ignore[arg-type]


AZURE_SETTINGS = load_azure_settings()


# Read remaining config directly from config.py
SOURCE_DIR = config.SOURCE_DIR  # type: ignore[attr-defined]
DEFAULT_TOP_K = config.DEFAULT_TOP_K  # type: ignore[attr-defined]
LOG_DIR = config.LOG_DIR  # type: ignore[attr-defined]
LOG_PATH = config.LOG_PATH  # type: ignore[attr-defined]

# Ensure directories exist at runtime
os.makedirs(SOURCE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
