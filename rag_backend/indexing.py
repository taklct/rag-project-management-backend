"""Index management for workbook documents."""

from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import dataclass
from typing import List, Optional

from openai import AzureOpenAI

from .chunking import chunk_text
from .documents import load_document
from .embeddings import embed_texts


@dataclass
class IndexFile:
    pkl_path: str
    chunks: List[str]
    embeddings: List[List[float]]
    signature: str


def file_signature(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)
    stat = os.stat(path)
    hasher.update(str(stat.st_size).encode())
    hasher.update(str(int(stat.st_mtime)).encode())
    return hasher.hexdigest()


def pkl_path_for(doc_path: str, embedding_model: str) -> str:
    base = os.path.basename(doc_path)
    return os.path.join(
        os.path.dirname(doc_path), f"{base}.{embedding_model}.idx.pkl"
    )


def load_index_if_valid(doc_path: str, embedding_model: str) -> Optional[IndexFile]:
    pkl_path = pkl_path_for(doc_path, embedding_model)
    if not os.path.exists(pkl_path):
        return None
    try:
        with open(pkl_path, "rb") as file:
            payload = pickle.load(file)
    except Exception:  # pragma: no cover - defensive
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("embedding_model") != embedding_model:
        return None
    if payload.get("doc_signature") != file_signature(doc_path):
        return None

    return IndexFile(
        pkl_path=pkl_path,
        chunks=payload["chunks"],
        embeddings=payload["embeddings"],
        signature=payload["doc_signature"],
    )


def build_index(
    doc_path: str,
    client: AzureOpenAI,
    embedding_model: str,
) -> IndexFile:
    raw_text = load_document(doc_path)
    chunks = chunk_text(raw_text, 1500, 150)
    if not chunks:
        raise RuntimeError(f"No text extracted from {doc_path}")

    embeddings = embed_texts(client, embedding_model, chunks)
    signature = file_signature(doc_path)
    payload = {
        "doc_signature": signature,
        "embedding_model": embedding_model,
        "chunks": chunks,
        "embeddings": embeddings,
    }
    pkl_path = pkl_path_for(doc_path, embedding_model)
    with open(pkl_path, "wb") as file:
        pickle.dump(payload, file)
    return IndexFile(
        pkl_path=pkl_path,
        chunks=chunks,
        embeddings=embeddings,
        signature=signature,
    )


def ensure_index(
    doc_path: str,
    client: AzureOpenAI,
    embedding_model: str,
    rebuild: bool = False,
) -> IndexFile:
    if not rebuild:
        cached = load_index_if_valid(doc_path, embedding_model)
        if cached:
            return cached
    return build_index(doc_path, client, embedding_model)


def list_xlsx(source_dir: str) -> List[str]:
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source dir not found: {source_dir}")
    return sorted(
        os.path.join(source_dir, name)
        for name in os.listdir(source_dir)
        if name.lower().endswith((".xlsx", ".xls"))
    )
