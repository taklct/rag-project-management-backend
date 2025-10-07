"""Application configuration and shared constants."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

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


ENVIRONMENT_VARIABLES = {
    "ENDPOINT_URL": "endpoint_url",
    "DEPLOYMENT_NAME": "chat_deployment",
    "EMBEDDING_DEPLOYMENT_NAME": "embedding_deployment",
    "AZURE_OPENAI_API_KEY": "api_key",
}


def _lookup_setting(name: str) -> Optional[str]:
    """Fetch a configuration value from the environment or config module."""

    env_value = os.getenv(name)
    if env_value:
        return env_value

    if config is not None and hasattr(config, name):
        value = getattr(config, name)
        if isinstance(value, str) and value:
            return value
    return None


def load_azure_settings() -> AzureSettings:
    """Load Azure OpenAI credentials, raising an informative error when missing."""

    values: dict[str, Optional[str]] = {
        alias: _lookup_setting(env_name)
        for env_name, alias in ENVIRONMENT_VARIABLES.items()
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        missing_display = ", ".join(sorted(missing))
        raise RuntimeError(
            "Please set ENDPOINT_URL, DEPLOYMENT_NAME, AZURE_OPENAI_API_KEY, "
            "EMBEDDING_DEPLOYMENT_NAME (env or config.py). Missing: "
            f"{missing_display}."
        )

    return AzureSettings(**values)  # type: ignore[arg-type]


AZURE_SETTINGS = load_azure_settings()

SOURCE_DIR = "./data_sources"
DEFAULT_TOP_K = 3
LOG_DIR = "./logs"
LOG_PATH = os.path.join(LOG_DIR, "query_logs.csv")

os.makedirs(SOURCE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
