"""Azure OpenAI client factory."""

from __future__ import annotations

from openai import AzureOpenAI

from .settings import AZURE_SETTINGS


def create_client() -> AzureOpenAI:
    """Create an Azure OpenAI client using the configured settings."""

    return AzureOpenAI(
        azure_endpoint=AZURE_SETTINGS.endpoint_url,
        api_key=AZURE_SETTINGS.api_key,
        api_version=AZURE_SETTINGS.api_version,
    )


AZURE_CLIENT = create_client()
