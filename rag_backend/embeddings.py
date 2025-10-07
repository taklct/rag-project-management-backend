"""Embedding helpers that wrap Azure OpenAI operations."""

from __future__ import annotations

import math
from typing import Iterable, List

from openai import AzureOpenAI


def embed_texts(client: AzureOpenAI, model: str, texts: Iterable[str]) -> List[List[float]]:
    """Generate embeddings for a collection of texts."""

    text_list = list(texts)
    if not text_list:
        return []

    embeddings: List[List[float]] = []
    batch_size = 16
    for start in range(0, len(text_list), batch_size):
        batch = text_list[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        embeddings.extend([item.embedding for item in response.data])
    return embeddings


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
