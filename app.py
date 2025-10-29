"""FastAPI application entry point."""

from __future__ import annotations

import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from project_dashboard import router as project_dashboard_router

try:
    from rag_backend.api_models import (
        AskBody,
        AskErrorResponse,
        AskSuccessResponse,
        BuildBody,
        BuildResponse,
    )
    from rag_backend.service import handle_ask_request, handle_build_request
except RuntimeError as exc:  # pragma: no cover - configuration validation
    print(exc)
    sys.exit(1)

app = FastAPI(
    title="RAG Project Management Backend",
    description=(
        "Backend APIs for building retrieval-augmented generation (RAG) indexes "
        "and querying project management insights. Use the `/docs` endpoint to "
        "explore the automatically generated Swagger UI."
    ),
    version="1.0.0",
    openapi_tags=[
        {
            "name": "Indexing",
            "description": "Manage retrieval indexes generated from uploaded workbooks.",
        },
        {
            "name": "Q&A",
            "description": "Ask questions over the indexed project documents using RAG.",
        },
        {
            "name": "Project Dashboard",
            "description": "Access metrics and task listings extracted from project workbooks.",
        },
    ],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(project_dashboard_router)


@app.on_event("startup")
def _auto_build_on_startup() -> None:
    """Trigger a build on app startup with body {"rebuild": false}."""
    try:
        # Explicitly set rebuild to False as requested
        handle_build_request(BuildBody(rebuild=False))
    except Exception as exc:  # pragma: no cover - defensive
        # Log and continue so the app still starts
        print(f"Startup /build failed: {exc}")


@app.post(
    "/build",
    response_model=BuildResponse,
    tags=["Indexing"],
    summary="Generate embeddings for available workbooks",
    response_description="Details about the index build and project dashboard cache.",
)
def build_indexes(body: BuildBody) -> BuildResponse:
    return handle_build_request(body)


@app.post(
    "/ask",
    response_model=AskSuccessResponse | AskErrorResponse,
    tags=["Q&A"],
    summary="Ask a question using retrieval augmented generation",
    response_description="Generated answer or an error describing why the request failed.",
)
def ask_question(body: AskBody) -> AskSuccessResponse | AskErrorResponse:
    return handle_ask_request(body)


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
