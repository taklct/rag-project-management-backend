"""Service layer orchestrating build and ask operations."""

from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple

from .api_models import (
    AskBody,
    AskErrorResponse,
    AskSuccessResponse,
    BuildBody,
    BuildResponse,
    IndexedFileSummary,
    ProjectDashboardStatus,
    to_token_usage,
)
from .azure import AZURE_CLIENT
from .indexing import ensure_index, list_xlsx
from .logging_utils import log_query
from .retrieval import ask_llm, build_messages, retrieve_top_k
from .embeddings import cosine_similarity
from .settings import AZURE_SETTINGS, DEFAULT_TOP_K, LOG_PATH, SOURCE_DIR
from project_dashboard import load_project_tasks


def _count_workbook_rows(path: str) -> Optional[int]:
    """Return the number of non-empty rows in an Excel workbook when possible."""

    lower_path = path.lower()
    if not lower_path.endswith((".xlsx", ".xls")):
        return None

    try:  # Import lazily so that non-Excel builds do not require pandas.
        import pandas as pd
    except ImportError:  # pragma: no cover - dependency guard
        return None

    try:
        dataframes = pd.read_excel(path, sheet_name=None)
    except Exception:  # pragma: no cover - defensive against malformed workbooks
        return None

    row_count = 0
    for dataframe in dataframes.values():
        try:
            row_count += len(dataframe.dropna(how="all"))
        except Exception:  # pragma: no cover - defensive
            continue

    return row_count


def _load_dashboard_status() -> ProjectDashboardStatus:
    try:
        dashboard = load_project_tasks()
    except FileNotFoundError:
        dashboard = {"ok": False, "error": "Project tasks workbook not found."}
    except Exception as exc:  # pragma: no cover - defensive
        dashboard = {"ok": False, "error": str(exc)}
    return ProjectDashboardStatus(**dashboard)


def handle_build_request(request: BuildBody) -> BuildResponse:
    xlsx_list = list_xlsx(SOURCE_DIR)
    if not xlsx_list:
        return BuildResponse(
            built=[],
            message="No .xlsx/.xls files found in ./data_sources.",
            project_dashboard=_load_dashboard_status(),
        )

    built: List[IndexedFileSummary] = []
    for path in xlsx_list:
        index_file = ensure_index(
            path,
            AZURE_CLIENT,
            AZURE_SETTINGS.embedding_deployment,
            rebuild=request.rebuild,
        )
        row_count = _count_workbook_rows(path)
        built.append(
            IndexedFileSummary(
                xlsx=os.path.basename(path),
                pkl=os.path.basename(index_file.pkl_path),
                chunks=len(index_file.chunks),
                signature=index_file.signature[:16],
                rows=row_count,
            )
        )

    return BuildResponse(
        built=built,
        project_dashboard=_load_dashboard_status(),
    )


def handle_ask_request(request: AskBody) -> AskSuccessResponse | AskErrorResponse:
    top_k = request.top_k or DEFAULT_TOP_K
    print("request.top_k "+str(request.top_k))
    print("DEFAULT_TOP_K "+str(DEFAULT_TOP_K))
    temperature_override = request.temperature
    if temperature_override is not None and temperature_override != 1:
        # Some Azure OpenAI chat deployments only support the default temperature of 1.
        # Falling back to ``None`` avoids sending an unsupported value while still letting
        # the request complete successfully.
        temperature_override = None

    try:
        xlsx_list = list_xlsx(SOURCE_DIR)
    except FileNotFoundError:
        return AskErrorResponse(
            error="No .xlsx/.xls files in ./data_sources. Build first via /build."
        )

    if not xlsx_list:
        return AskErrorResponse(
            error="No .xlsx/.xls files in ./data_sources. Build first via /build."
        )

    all_chunks: List[str] = []
    all_embeddings: List[List[float]] = []
    for path in xlsx_list:
        index_file = ensure_index(
            path,
            AZURE_CLIENT,
            AZURE_SETTINGS.embedding_deployment,
            rebuild=False,
        )
        all_chunks.extend(index_file.chunks)
        all_embeddings.extend(index_file.embeddings)

    if not all_chunks:
        return AskErrorResponse(
            error="No indexed data available. Call /build to generate embeddings first."
        )
    # Hybrid retrieval: prefer chunks that match intent and entities from the question
    # (e.g., names like "John") and prioritize the Summary sheet for count-like queries,
    # then fill up to k by embedding similarity.
    q_text = request.question
    q_lower = q_text.lower()

    # Very light heuristic to extract potential entity tokens from the question.
    # We only use tokens that appear in the data as "Assignee=<token>" to avoid noise.
    import re

    tokens = set(re.findall(r"[a-zA-Z][a-zA-Z\-']+", q_lower))
    stop = {
        "what",
        "which",
        "who",
        "how",
        "many",
        "much",
        "task",
        "tasks",
        "does",
        "do",
        "have",
        "for",
        "of",
        "the",
        "a",
        "an",
        "this",
        "that",
        "current",
        "sprint",
        "today",
        "is",
        "are",
        "in",
        "on",
        "to",
        "by",
    }
    candidates = [t for t in tokens if len(t) >= 3 and t not in stop]

    # Detect counting intent to favor the Summary sheet
    is_count_query = bool(
        re.search(
            r"\b(count|total|how\s+many|how\s+much|number\s+of|sum|total\s+tasks)\b",
            q_lower,
        )
    )

    lexical_indices: List[int] = []
    if candidates:
        patterns = [f"assignee={t}" for t in candidates]
        for idx, chunk in enumerate(all_chunks):
            text = chunk.lower()
            if any(p in text for p in patterns) or any(t in text for t in candidates):
                lexical_indices.append(idx)

    # Identify Summary sheet chunks when the question looks like a counting query
    summary_indices: List[int] = []
    if is_count_query:
        for idx, chunk in enumerate(all_chunks):
            if "--- sheet: summary ---" in chunk.lower():
                summary_indices.append(idx)

    # Compute similarities for all chunks once so we can score lexical matches.
    question_embedding = (
        AZURE_CLIENT.embeddings.create(
            model=AZURE_SETTINGS.embedding_deployment, input=[q_text]
        ).data[0].embedding
    )
    all_scores: List[Tuple[int, float]] = [
        (i, cosine_similarity(question_embedding, emb)) for i, emb in enumerate(all_embeddings)
    ]
    all_scores.sort(key=lambda item: item[1], reverse=True)

    ordered_indices: List[int] = []
    # First, prioritize Summary sheet chunks if applicable
    if summary_indices:
        summary_set = set(summary_indices)
        summary_scored = [(i, s) for i, s in all_scores if i in summary_set]
        ordered_indices.extend([i for i, _ in summary_scored])

    # Next, add lexical matches ordered by similarity
    if lexical_indices:
        lexical_set = set(lexical_indices)
        lexical_scored = [(i, s) for i, s in all_scores if i in lexical_set]
        for i, _ in lexical_scored:
            if i not in ordered_indices:
                ordered_indices.append(i)

    # Finally, fill the rest with remaining top-similarity chunks
    for i, _ in all_scores:
        if len(ordered_indices) >= top_k:
            break
        if i not in ordered_indices:
            ordered_indices.append(i)

    # Truncate to k and compute the aligned scores
    ordered_indices = ordered_indices[:top_k]
    score_map = {i: s for i, s in all_scores}
    retrieved_chunks = [all_chunks[i] for i in ordered_indices]
    top_scores = [round(score_map[i], 4) for i in ordered_indices]

    messages = build_messages(retrieved_chunks, request.question)
    answer, usage, duration = ask_llm(
        AZURE_CLIENT,
        AZURE_SETTINGS.chat_deployment,
        messages,
        temperature=temperature_override,
        max_completion_tokens=10000,
    )

    now_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_query(
        LOG_PATH,
        {
            "timestamp": now_iso,
            "question": request.question,
            "top_k": top_k,
            "temperature_requested": request.temperature,
            "temperature_sent": temperature_override,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "duration_sec": round(duration, 3),
        },
    )

    return AskSuccessResponse(
        answer=answer,
        usage=to_token_usage(usage),
        duration_sec=round(duration, 3),
        top_scores=top_scores,
        indexed_files=[os.path.basename(path) for path in xlsx_list],
    )
