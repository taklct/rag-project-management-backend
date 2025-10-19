"""Pydantic models shared across API routes."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class BuildBody(BaseModel):
    """Request payload for `/build` to trigger index creation."""

    rebuild: bool = Field(
        False,
        description=(
            "Recreate indexes even if cached embeddings already exist for the source "
            "workbooks."
        ),
    )


class IndexedFileSummary(BaseModel):
    """Summary of an indexed workbook and its embedding cache."""

    xlsx: str = Field(..., description="Workbook filename that was processed.")
    pkl: str = Field(..., description="Generated embedding cache filename.")
    chunks: int = Field(..., description="Number of text chunks extracted from the workbook.")
    signature: str = Field(
        ..., description="Short hash of the workbook contents used for cache validation."
    )
    rows: int | None = Field(
        None,
        description=(
            "Number of non-empty rows detected across all sheets in the workbook."
        ),
        ge=0,
    )


class ProjectDashboardStatus(BaseModel):
    """Status information about the project dashboard cache."""

    ok: bool = Field(..., description="Whether project dashboard data is available.")
    count: Optional[int] = Field(
        None, description="Number of cached project tasks when available."
    )
    error: Optional[str] = Field(
        None, description="Details when the dashboard data could not be loaded."
    )


class BuildResponse(BaseModel):
    """Response returned by the `/build` endpoint."""

    ok: Literal[True] = Field(True, description="Whether the build completed successfully.")
    built: List[IndexedFileSummary] = Field(
        default_factory=list,
        description="Details for each workbook that was indexed.",
    )
    message: Optional[str] = Field(
        None, description="Helpful message when no workbooks were available to index."
    )
    project_dashboard: ProjectDashboardStatus = Field(
        ..., description="Status of the project dashboard cache after the build runs."
    )


class AskBody(BaseModel):
    """Request payload for `/ask` to run a RAG-powered query."""

    question: str = Field(..., description="Natural language question to ask over the data.")
    top_k: Optional[int] = Field(
        None,
        description="Number of most similar chunks to retrieve from the index (defaults to 3).",
        ge=1,
    )
    temperature: Optional[float] = Field(
        None,
        description=(
            "Sampling temperature for the language model. Leave unset to use the "
            "default configuration. Some Azure deployments only allow the default "
            "value."
        ),
        ge=0.0,
    )


class TokenUsage(BaseModel):
    """Token accounting metadata returned by Azure OpenAI."""

    prompt_tokens: Optional[int] = Field(
        None, description="Tokens consumed by the prompt portion of the request."
    )
    completion_tokens: Optional[int] = Field(
        None, description="Tokens generated in the completion response."
    )
    total_tokens: Optional[int] = Field(
        None, description="Total tokens counted for the request."
    )


class AskSuccessResponse(BaseModel):
    """Successful response for the `/ask` endpoint."""

    ok: Literal[True] = Field(True, description="Indicates the question was processed.")
    answer: str = Field(..., description="Generated answer from the language model.")
    usage: Optional[TokenUsage] = Field(
        None, description="Token usage metrics reported by the language model."
    )
    duration_sec: float = Field(..., description="Time taken to build the response in seconds.")
    top_scores: List[float] = Field(
        ..., description="Similarity scores for the retrieved context chunks."
    )
    indexed_files: List[str] = Field(
        ..., description="Workbook filenames that contributed to the retrieval context."
    )


class AskErrorResponse(BaseModel):
    """Error payload returned when the `/ask` endpoint cannot run."""

    ok: Literal[False] = Field(False, description="Indicates the question could not be run.")
    error: str = Field(..., description="Reason why the request failed.")


def to_token_usage(payload: Dict[str, Any] | None) -> Optional[TokenUsage]:
    """Convert a usage dictionary into a :class:`TokenUsage` instance."""

    if not payload:
        return None
    return TokenUsage(**payload)
