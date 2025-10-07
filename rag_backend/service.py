"""Service layer orchestrating build and ask operations."""

from __future__ import annotations

import os
import time
from typing import List

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
from .settings import AZURE_SETTINGS, DEFAULT_TOP_K, LOG_PATH, SOURCE_DIR
from project_dashboard import load_project_tasks


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
        built.append(
            IndexedFileSummary(
                xlsx=os.path.basename(path),
                pkl=os.path.basename(index_file.pkl_path),
                chunks=len(index_file.chunks),
                signature=index_file.signature[:16],
            )
        )

    return BuildResponse(
        built=built,
        project_dashboard=_load_dashboard_status(),
    )


def handle_ask_request(request: AskBody) -> AskSuccessResponse | AskErrorResponse:
    top_k = request.top_k or DEFAULT_TOP_K
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

    top = retrieve_top_k(
        AZURE_CLIENT,
        AZURE_SETTINGS.embedding_deployment,
        request.question,
        all_chunks,
        all_embeddings,
        k=top_k,
    )
    retrieved_chunks = [all_chunks[index] for index, _ in top]
    top_scores = [round(score, 4) for _, score in top]

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
