"""Retrieval helpers for answering questions with Azure OpenAI."""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

from openai import AzureOpenAI

from .embeddings import cosine_similarity


def retrieve_top_k(
    client: AzureOpenAI,
    embedding_model: str,
    question: str,
    all_chunks: List[str],
    all_embeddings: List[List[float]],
    k: int,
) -> List[Tuple[int, float]]:
    question_embedding = (
        client.embeddings.create(model=embedding_model, input=[question]).data[0].embedding
    )
    similarities = [
        (index, cosine_similarity(question_embedding, embedding))
        for index, embedding in enumerate(all_embeddings)
    ]
    similarities.sort(key=lambda item: item[1], reverse=True)
    return similarities[:k]


def build_messages(retrieved_chunks: List[str], question: str) -> List[Dict[str, str]]:
    context_block = "\n\n".join(
        [f"[Chunk {index + 1}]\n{chunk}" for index, chunk in enumerate(retrieved_chunks)]
    )
    system_text = (
        "You are a helpful AI assistant. Use ONLY the provided context to answer the question. "
        "If the answer is not in the context, say you don't know and suggest what would help."
    )
    user_text = (
        "Here is the context from the document(s):\n"
        f"{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer clearly. If relevant, quote short snippets."
    )
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]


def ask_llm(
    client: AzureOpenAI,
    chat_deployment: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float = 1,
    max_completion_tokens: int = 10000,
) -> Tuple[str, Dict[str, int | None], float]:
    start_time = time.time()
    completion = client.chat.completions.create(
        model=chat_deployment,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
    )
    duration = time.time() - start_time
    message = completion.choices[0].message
    usage = getattr(completion, "usage", None)
    tokens: Dict[str, int | None] = (
        {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
        if usage
        else {}
    )
    return (message.content or "").strip(), tokens, duration
