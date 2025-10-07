"""APIs and caching helpers for the project dashboard."""

from __future__ import annotations

import os
from datetime import date
from numbers import Number
from typing import Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

SOURCE_DIR = "./data_sources"
TASKS_FILENAME_CANDIDATES = (
    "jira_project_project_management.xlsx",
    "jira_project-management.xlsx",
)

router = APIRouter(prefix="/project-dashboard", tags=["Project Dashboard"])

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


@router.get("/task-summary")
def task_summary() -> Dict[str, int]:
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

    return {
        "completed_today": completed_today,
        "updated_today": updated_today,
        "created_today": created_today,
        "overdue": overdue,
    }


def _select_sprint_number(requested: Optional[int]) -> Optional[int]:
    if requested is not None:
        return requested
    if _cached_sprint_numbers:
        return _cached_sprint_numbers[-1]
    return None


@router.get("/tasks-of-sprint")
def tasks_of_sprint(
    sprint: Optional[int] = Query(
        None, description="Sprint number to filter by. Defaults to the latest sprint."
    ),
) -> Dict[str, object]:
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

    return {"sprint": sprint_number, "count": len(tasks), "tasks": tasks}
