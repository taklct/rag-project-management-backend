"""APIs and caching helpers for the project dashboard."""

from __future__ import annotations

import os
import re
from datetime import date
from numbers import Number
from typing import Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

SOURCE_DIR = "./data_sources"
TASKS_FILENAME_CANDIDATES = (
    "jira_project_project_management.xlsx",
    "jira_project-management.xlsx",
)

router = APIRouter(
    prefix="/project-dashboard",
    tags=["Project Dashboard"],
    responses={
        503: {
            "description": "Project dashboard cache is not yet available.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Project tasks have not been loaded. Call /build first.",
                    }
                }
            },
        }
    },
)

_tasks_cache: List[Dict[str, object]] = []
_cached_sprint_numbers: List[int] = []


def _resolve_tasks_path() -> Optional[str]:
    for name in TASKS_FILENAME_CANDIDATES:
        path = os.path.join(SOURCE_DIR, name)
        if os.path.exists(path):
            return path
    return None


def load_project_tasks() -> Dict[str, object]:
    """Load and cache project tasks from the configured Excel workbook."""
    path = _resolve_tasks_path()
    if path is None:
        raise FileNotFoundError(
            "Project tasks workbook not found in ./data_sources."
        )

    df = pd.read_excel(path, sheet_name="Tasks")
    # Drop rows that are entirely empty (if any trailing blank rows exist)
    df = df.dropna(how="all")

    # Normalise date columns to date objects for easier comparisons later on.
    for column in ("Task start date", "Task end date"):
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce").dt.date

    # Ensure sprint numbers are integers when possible.
    if "Sprint Number" in df.columns:
        df["Sprint Number"] = pd.to_numeric(df["Sprint Number"], errors="coerce").astype("Int64")

    global _tasks_cache, _cached_sprint_numbers
    _tasks_cache = df.to_dict(orient="records")
    _cached_sprint_numbers = sorted(
        {
            int(row_value)
            for row_value in (
                task.get("Sprint Number") for task in _tasks_cache
            )
            if isinstance(row_value, Number) and not pd.isna(row_value)
        }
    )

    return {"ok": True, "count": len(_tasks_cache)}


def _ensure_cache() -> None:
    if not _tasks_cache:
        raise HTTPException(
            status_code=503,
            detail="Project tasks have not been loaded. Call /build first.",
        )


def _parse_date(value: object) -> Optional[date]:
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def _today() -> date:
    return date.today()


class TaskSummaryResponse(BaseModel):
    """Aggregated project metrics for the dashboard widgets."""

    completed_today: int = Field(
        ..., description="Number of tasks completed with an end date equal to today."
    )
    updated_today: int = Field(
        ..., description="Tasks updated or ending today regardless of status."
    )
    created_today: int = Field(
        ..., description="Tasks that have a start date equal to today."
    )
    overdue: int = Field(
        ..., description="Tasks past their end date that are not yet marked as done."
    )


class SprintTask(BaseModel):
    """Minimal representation of a task for sprint-based listings."""

    assignee: Optional[str] = Field(
        None, alias="Assignee", description="Primary owner assigned to the task."
    )
    team: Optional[str] = Field(
        None, alias="Team", description="Team responsible for the task."
    )
    task_title: Optional[str] = Field(
        None,
        alias="Task Title",
        description="Short, descriptive name summarising the work item.",
    )
    task_description: Optional[str] = Field(
        None,
        alias="Task Description",
        description="Additional context or notes for the work item.",
    )
    status: Optional[str] = Field(
        None, alias="Status", description="Current workflow status of the task."
    )
    priority: Optional[str] = Field(
        None, alias="Priority", description="Relative priority for the work item."
    )
    story_point: Optional[float] = Field(
        None,
        alias="Story Point",
        description="Effort estimate measured in story points if available.",
    )

    class Config:
        allow_population_by_field_name = True


class SprintTasksResponse(BaseModel):
    """Tasks filtered by sprint for board-style views."""

    sprint: Optional[int] = Field(
        None, description="Sprint number that was requested or automatically selected."
    )
    count: int = Field(..., description="Number of tasks included in the response.")
    tasks: List[SprintTask] = Field(
        default_factory=list,
        description="Tasks that belong to the requested sprint.",
    )


class TaskItem(BaseModel):
    """Detailed representation of a task for status-based listings."""

    task_number: Optional[str] = Field(
        None,
        alias="Task Number",
        description="Unique identifier assigned to the task in the source system.",
    )
    task_title: Optional[str] = Field(
        None,
        alias="Task Title",
        description="Short, descriptive name summarising the work item.",
    )
    status: Optional[str] = Field(
        None, alias="Status", description="Current workflow status of the task."
    )
    assignee: Optional[str] = Field(
        None, alias="Assignee", description="Primary owner assigned to the task."
    )
    team: Optional[str] = Field(
        None, alias="Team", description="Team responsible for the task."
    )
    priority: Optional[str] = Field(
        None, alias="Priority", description="Relative priority for the work item."
    )
    task_start_date: Optional[date] = Field(
        None,
        alias="Task start date",
        description="Date the task is scheduled to begin.",
    )
    task_end_date: Optional[date] = Field(
        None,
        alias="Task end date",
        description="Date the task is scheduled to be completed.",
    )

    class Config:
        allow_population_by_field_name = True


class TaskListResponse(BaseModel):
    """Tasks filtered by status and due date criteria."""

    count: int = Field(..., description="Number of tasks included in the response.")
    tasks: List[TaskItem] = Field(
        default_factory=list,
        description="Tasks that match the requested filter criteria.",
    )


@router.get(
    "/task-summary",
    response_model=TaskSummaryResponse,
    summary="Get task metrics for dashboard widgets",
    response_description="Aggregated counts for key task indicators.",
)
def task_summary() -> TaskSummaryResponse:
    """Return summary counts for key task metrics."""
    _ensure_cache()
    today = _today()

    completed_today = 0
    updated_today = 0
    created_today = 0
    overdue = 0

    for task in _tasks_cache:
        status = str(task.get("Status", "")).strip().lower()
        start_date = _parse_date(task.get("Task start date"))
        end_date = _parse_date(task.get("Task end date"))

        if status == "done" and end_date == today:
            completed_today += 1

        if end_date == today:
            updated_today += 1

        if start_date == today:
            created_today += 1

        if end_date and end_date < today and status != "done":
            overdue += 1

    return TaskSummaryResponse(
        completed_today=completed_today,
        updated_today=updated_today,
        created_today=created_today,
        overdue=overdue,
    )


def _select_sprint_number(requested: Optional[int]) -> Optional[int]:
    if requested is not None:
        return requested
    if _cached_sprint_numbers:
        return _cached_sprint_numbers[-1]
    return None


@router.get(
    "/tasks-of-sprint",
    response_model=SprintTasksResponse,
    summary="List tasks associated with a sprint",
    response_description="Tasks filtered by the requested or latest sprint.",
)
def tasks_of_sprint(
    sprint: Optional[int] = Query(
        None, description="Sprint number to filter by. Defaults to the latest sprint."
    ),
) -> SprintTasksResponse:
    """Return the tasks for the requested sprint."""
    _ensure_cache()

    sprint_number = _select_sprint_number(sprint)

    fields = [
        "Assignee",
        "Team",
        "Task Title",
        "Task Description",
        "Status",
        "Priority",
        "Story Point",
    ]

    tasks: List[Dict[str, object]] = []
    for task in _tasks_cache:
        task_sprint = task.get("Sprint Number")
        if sprint_number is not None:
            try:
                if int(task_sprint) != sprint_number:
                    continue
            except (TypeError, ValueError):
                continue
        tasks.append({field: task.get(field) for field in fields})

    return SprintTasksResponse(sprint=sprint_number, count=len(tasks), tasks=tasks)


def _normalise_status(value: object) -> str:
    if isinstance(value, str):
        return value.strip().casefold()
    return ""


def _is_blocked_status(value: object) -> bool:
    """Return ``True`` when the provided status represents a blocked item."""

    status = _normalise_status(value)
    if not status:
        return False
    return bool(re.search(r"\bblocked\b", status))


def _is_overdue(task: Dict[str, object], today: date) -> bool:
    end_date = _parse_date(task.get("Task end date"))
    if end_date is None or end_date >= today:
        return False
    status = _normalise_status(task.get("Status"))
    return status != "done"


@router.get(
    "/blocked-item",
    response_model=TaskListResponse,
    summary="List tasks currently blocked",
    response_description="Tasks whose workflow status is marked as blocked.",
)
def blocked_tasks() -> TaskListResponse:
    """Return tasks that are currently blocked."""
    _ensure_cache()

    blocked = [
        TaskItem(**task)
        for task in _tasks_cache
        if _is_blocked_status(task.get("Status"))
    ]

    return TaskListResponse(count=len(blocked), tasks=blocked)


@router.get(
    "/overdue-item",
    response_model=TaskListResponse,
    summary="List tasks that are overdue",
    response_description=(
        "Tasks past their end date that are not yet marked as complete."
    ),
)
def overdue_tasks() -> TaskListResponse:
    """Return tasks that are overdue and not completed."""
    _ensure_cache()
    today = _today()

    overdue_tasks = [
        TaskItem(**task)
        for task in _tasks_cache
        if _is_overdue(task, today)
    ]

    return TaskListResponse(count=len(overdue_tasks), tasks=overdue_tasks)
