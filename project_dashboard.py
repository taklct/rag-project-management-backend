"""APIs and caching helpers for the project dashboard."""

from __future__ import annotations

import os
import re
from datetime import date, datetime
from numbers import Number
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import json
import base64

try:  # Fallback to config.py for JIRA config if present
    import config  # type: ignore
except Exception:  # pragma: no cover - defensive
    config = None  # type: ignore

SOURCE_DIR = "./data_sources"
TASKS_FILENAME_CANDIDATES = (
    "jira-issues.xlsx",
    "jira-tasks.xlsx",  # common alternate name
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
_summary_cache: List[Dict[str, object]] = []
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

    # Load required sheets. Prefer a sheet named "Tasks"; if missing, fall back to the first sheet.
    try:
        df = pd.read_excel(path, sheet_name="Tasks")
    except Exception:
        # Fall back to first available sheet
        all_sheets = pd.read_excel(path, sheet_name=None)
        if not all_sheets:
            raise FileNotFoundError("No sheets found in project tasks workbook.")
        # Pick the first sheet deterministically
        first_name = next(iter(all_sheets.keys()))
        df = all_sheets[first_name]

    # Try to load an optional summary sheet when present
    try:
        df_summary = pd.read_excel(path, sheet_name="Summary")
    except Exception:
        df_summary = None
    # Drop rows that are entirely empty (if any trailing blank rows exist)
    df = df.dropna(how="all")

    # Normalise date columns to date objects for easier comparisons later on.
    for column in ("Task start date", "Task end date"):
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce").dt.date

    # Replace any remaining NaN/NaT values with None so response models
    # serialise cleanly and date fields can be optional.
    df = df.where(pd.notnull(df), None)

    # Ensure sprint numbers are integers when possible.
    if "Sprint Number" in df.columns:
        df["Sprint Number"] = pd.to_numeric(df["Sprint Number"], errors="coerce").astype("Int64")

    global _tasks_cache, _summary_cache, _cached_sprint_numbers
    _tasks_cache = df.to_dict(orient="records")
    if df_summary is not None:
        _summary_cache = df_summary.fillna("").to_dict(orient="records")
    else:
        _summary_cache = []
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


# ---------------------------- JIRA helpers ----------------------------


def _cfg_lookup(name: str) -> Optional[str]:
    env = os.getenv(name)
    if env:
        return env
    if config is not None and hasattr(config, name):
        value = getattr(config, name)
        if isinstance(value, str) and value:
            return value
    return None


def _jira_auth_header() -> Tuple[str, Dict[str, str]]:
    # Prefer explicit API base (v3); fall back to endpoint URL for backward compatibility
    base_url = _cfg_lookup("JIRA_API_URL")
    if not base_url:
        raise HTTPException(status_code=500, detail="Missing JIRA_API_URL (env or config.py)")

    username = _cfg_lookup("JIRA_USERNAME")
    token = _cfg_lookup("JIRA_API_TOKEN")
    if not username or not token:
        raise HTTPException(
            status_code=500,
            detail=(
                "Missing JIRA_USERNAME/JIRA_API_TOKEN. Set as env vars or in config.py "
                "to enable JIRA API access."
            ),
        )
    basic = base64.b64encode(f"{username}:{token}".encode("utf-8")).decode("ascii")
    headers = {
        "Authorization": f"Basic {basic}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    return base_url.rstrip("/"), headers


def _adf_text(value: Any) -> str:
    # Jira Cloud v3 description uses ADF structure; try to flatten a bit
    if isinstance(value, str):
        return value
    try:
        def walk(node: Any) -> str:
            if isinstance(node, dict):
                t = node.get("type")
                if t == "text":
                    return node.get("text", "")
                parts = []
                for c in node.get("content", []) or []:
                    parts.append(walk(c))
                sep = "\n" if t in {"paragraph", "heading"} else ""
                return sep.join([p for p in parts if p])
            if isinstance(node, list):
                return "\n".join(filter(None, (walk(n) for n in node)))
            return ""

        return walk(value).strip()
    except Exception:
        try:
            return json.dumps(value)
        except Exception:
            return ""


def _extract_sprint_info(fields: Dict[str, Any]) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    # Try common locations for sprint info
    sprint_data = fields.get("sprint")
    if not sprint_data:
        sprint_data = fields.get("customfield_10020")  # common default id for Sprint
    if isinstance(sprint_data, list) and sprint_data:
        sprint_data = sprint_data[-1]

    number: Optional[int] = None
    start: Optional[str] = None
    end: Optional[str] = None
    if isinstance(sprint_data, dict):
        name = sprint_data.get("name") or ""
        m = re.search(r"(\d+)", str(name))
        if m:
            try:
                number = int(m.group(1))
            except Exception:
                number = None
        start = sprint_data.get("startDate") or None
        end = sprint_data.get("endDate") or None
        # Normalise ISO date-only if timestamps
        def only_date(dt: Optional[str]) -> Optional[str]:
            if not dt:
                return None
            try:
                return datetime.fromisoformat(dt.replace("Z", "+00:00")).date().isoformat()
            except Exception:
                return None

        start = only_date(start) or start
        end = only_date(end) or end
    return number, start, end


def _extract_team(fields: Dict[str, Any]) -> str:
    # Prefer a dedicated Team field when present
    team = fields.get("team")
    if isinstance(team, dict) and team.get("name"):
        return str(team.get("name"))
    # Fall back to first component name if any
    comps = fields.get("components")
    if isinstance(comps, list) and comps:
        name = comps[0].get("name") if isinstance(comps[0], dict) else None
        if name:
            return str(name)
    return ""


def _extract_story_points(fields: Dict[str, Any]) -> Optional[float]:
    # Common custom field ids for Story Points
    for key in ("customfield_10016", "customfield_10126", "customfield_10024", "customfield_10014"):
        value = fields.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        # Some JIRA instances use numeric strings
        if isinstance(value, str):
            try:
                return float(value)
            except Exception:
                pass
    return None


def _to_date_str(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
    except Exception:
        return value  # best effort


def _map_issue(issue: Dict[str, Any]) -> Dict[str, Any]:
    fields: Dict[str, Any] = issue.get("fields", {}) or {}
    sprint_no, sprint_start, sprint_end = _extract_sprint_info(fields)

    assignee = fields.get("assignee") or {}
    status = fields.get("status") or {}
    priority = fields.get("priority") or {}

    start_date = (
        fields.get("startDate")
        or fields.get("customfield_10015")  # sometimes used for start date
        or sprint_start
        or fields.get("created")
    )
    end_date = fields.get("duedate") or sprint_end or fields.get("resolutiondate")

    description = fields.get("description")

    return {
        "Sprint Number": sprint_no,
        "Task Number": issue.get("key"),
        "Assignee": assignee.get("displayName") if isinstance(assignee, dict) else None,
        "Team": _extract_team(fields),
        "Task Title": fields.get("summary"),
        "Task Description": _adf_text(description),
        "Task start date": _to_date_str(start_date),
        "Task end date": _to_date_str(end_date),
        "Status": status.get("name") if isinstance(status, dict) else None,
        "Priority": priority.get("name") if isinstance(priority, dict) else None,
        "Story Point": _extract_story_points(fields),
    }


@router.post(
    "/jira/export-issues",
    summary="Fetch JIRA issues and export to Excel",
    response_description=(
        "Downloads issues via JIRA API and writes data_sources/jira-issues.xlsx."
    ),
)
def export_jira_issues(project: str = Query("M1", description="JIRA project key")) -> Dict[str, Any]:
    """Fetch issues from JIRA and write them into jira-issues.xlsx.

    Uses ``JIRA_API_URL``, ``JIRA_USERNAME`` and ``JIRA_API_TOKEN`` from env or config.py.
    """
    try:
        import requests  # type: ignore
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="The 'requests' package is required. Add it to requirements.txt and install.",
        )

    base_url, headers = _jira_auth_header()
    jql = f"project={project}"

    issues: List[Dict[str, Any]] = []

    # Prefer the user-provided path; gracefully fall back to /search if needed.
    # Token pagination only; when nextPageToken is absent, stop.
    def fetch_page(
        next_token: Optional[str] = None,
        use_alt: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        path = "/search/jql" if not use_alt else "/search"
        params = [f"jql={jql}", "maxResults=5000", "fields=*all"]
        if next_token:
            params.append(f"nextPageToken={next_token}")
        url = f"{base_url}{path}?" + "&".join(params)
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 404 and not use_alt:
            return fetch_page(next_token=next_token, use_alt=True)
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        data = resp.json()
        batch = data.get("issues", []) or []
        next_token_out = data.get("nextPageToken") or None
        return batch, next_token_out

    next_token: Optional[str] = None
    while True:
        batch, token = fetch_page(next_token=next_token)
        issues.extend(batch)
        next_token = token
        # break when no nextPageToken is provided
        if next_token is None:
            break

    # Map to the desired schema
    rows = [_map_issue(it) for it in issues]

    # Add Jira browse URL column using JIRA_ENDPOINT_URL when available
    browse_base = _cfg_lookup("JIRA_ENDPOINT_URL")
    if browse_base:
        browse_base = str(browse_base).rstrip("/")
    for r in rows:
        try:
            key = r.get("Task Number")
            r["Jira Url"] = (
                f"{browse_base}/browse/{key}" if browse_base and isinstance(key, str) and key else None
            )
        except Exception:
            r["Jira Url"] = None

    ordered_columns = [
        "Sprint Number",
        "Task Number",
        "Jira Url",
        "Assignee",
        "Team",
        "Task Title",
        "Task Description",
        "Task start date",
        "Task end date",
        "Status",
        "Priority",
        "Story Point",
    ]

    df = pd.DataFrame(rows, columns=ordered_columns)
    out_path = os.path.join(SOURCE_DIR, "jira-issues.xlsx")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        df.to_excel(out_path, index=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write Excel: {exc}")

    return {"ok": True, "exported": len(rows), "path": out_path}


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


class CountByAssigneeResponse(BaseModel):
    """Exact count of tasks for a given assignee."""

    assignee: str = Field(..., description="Assignee name used for matching.")
    count: int = Field(..., description="Number of matching tasks.")


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


class SprintStatusBucketsResponse(BaseModel):
    """Sprint tasks grouped into board columns for quick status lookups."""

    sprint: Optional[int] = Field(
        None, description="Sprint number that was requested or automatically selected."
    )
    statuses: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Mapping of status names to formatted task listings.",
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


class OverdueTaskItem(TaskItem):
    """Task item for overdue listings with Jira link when available."""

    jira_url: Optional[str] = Field(
        None,
        alias="Jira Url",
        description="Direct link to the task in JIRA web UI.",
    )


class OverdueTaskListResponse(BaseModel):
    """Overdue tasks response including optional Jira Url column."""

    count: int = Field(..., description="Number of tasks included in the response.")
    tasks: List[OverdueTaskItem] = Field(
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

        if end_date and end_date < today and status not in {"done", "closed"}:
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


def _tasks_for_sprint(sprint_number: Optional[int]) -> List[Dict[str, object]]:
    tasks: List[Dict[str, object]] = []
    for task in _tasks_cache:
        task_sprint = task.get("Sprint Number")
        if sprint_number is not None:
            try:
                if int(task_sprint) != sprint_number:
                    continue
            except (TypeError, ValueError):
                continue
        tasks.append(task)
    return tasks


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

    sprint_tasks = _tasks_for_sprint(sprint_number)
    tasks = [
        _clean_record({field: task.get(field) for field in fields})
        for task in sprint_tasks
    ]

    return SprintTasksResponse(sprint=sprint_number, count=len(tasks), tasks=tasks)


@router.get(
    "/current-sprint-status",
    response_model=SprintStatusBucketsResponse,
    summary="Get the current sprint's tasks grouped by status",
    response_description="Mapping of statuses to formatted task entries for the latest sprint.",
)
def current_sprint_status() -> SprintStatusBucketsResponse:
    """Return the latest sprint's tasks grouped into board status buckets."""
    _ensure_cache()

    sprint_number = _select_sprint_number(None)
    sprint_tasks = _tasks_for_sprint(sprint_number)

    statuses: Dict[str, List[str]] = {label: [] for label in _STATUS_DISPLAY_ORDER}

    for task in sprint_tasks:
        label = _status_display_label(task.get("Status"))
        entry = _format_task_entry(task.get("Task Number"), task.get("Task Title"))
        if entry is None:
            continue
        statuses.setdefault(label, []).append(entry)

    trimmed_statuses: Dict[str, List[str]] = {}
    for label in _STATUS_DISPLAY_ORDER:
        if statuses.get(label):
            trimmed_statuses[label] = statuses[label]

    for label, entries in statuses.items():
        if label in trimmed_statuses:
            continue
        if entries:
            trimmed_statuses[label] = entries

    return SprintStatusBucketsResponse(sprint=sprint_number, statuses=trimmed_statuses)


def _normalise_status(value: object) -> str:
    if isinstance(value, str):
        return value.strip().casefold()
    return ""


def _clean_value(value: object) -> object:
    """Convert pandas-style missing values (NaN/NaT) to None for safe validation."""
    try:
        if pd.isna(value):  # type: ignore[attr-defined]
            return None
    except Exception:
        pass
    return value


def _clean_record(record: Dict[str, object]) -> Dict[str, object]:
    """Return a shallow copy with NaN/NaT values replaced by None."""
    return {k: _clean_value(v) for k, v in record.items()}


_STATUS_LABELS = {
    "to do": "TO DO",
    "todo": "TO DO",
    "in progress": "IN PROGRESS",
    "in-progress": "IN PROGRESS",
    "doing": "IN PROGRESS",
    "done": "DONE",
    "completed": "DONE",
}

_STATUS_DISPLAY_ORDER = ("TO DO", "IN PROGRESS", "DONE")


def _status_display_label(value: object) -> str:
    status = _normalise_status(value)
    if not status:
        return "UNKNOWN"
    if status in _STATUS_LABELS:
        return _STATUS_LABELS[status]
    cleaned = status.replace("_", " ").replace("-", " ")
    if cleaned:
        return cleaned.upper()
    return "UNKNOWN"


def _format_task_entry(task_number: object, task_title: object) -> Optional[str]:
    parts: List[str] = []
    if isinstance(task_number, str):
        number = task_number.strip()
        if number:
            parts.append(number)
    elif task_number is not None:
        parts.append(str(task_number))

    if isinstance(task_title, str):
        title = task_title.strip()
        if title:
            parts.append(title)
    elif task_title is not None:
        parts.append(str(task_title))

    if not parts:
        return None
    return " ".join(parts)


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
    return status not in {"done", "closed"}


# (Removed JIRA browse URL helper; not needed now)


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
        TaskItem(**_clean_record(task))
        for task in _tasks_cache
        if _is_blocked_status(task.get("Status"))
    ]

    # Sort by end date ascending, with missing dates last for readability
    blocked.sort(key=lambda t: (t.task_end_date is None, t.task_end_date))

    return TaskListResponse(count=len(blocked), tasks=blocked)


@router.get(
    "/overdue-item",
    response_model=OverdueTaskListResponse,
    summary="List tasks that are overdue",
    response_description=(
        "Tasks past their end date that are not yet marked as complete."
    ),
)
def overdue_tasks() -> OverdueTaskListResponse:
    """Return tasks that are overdue and not completed."""
    _ensure_cache()
    today = _today()

    overdue_tasks = []
    for raw in _tasks_cache:
        if not _is_overdue(raw, today):
            continue
        record = _clean_record(raw)
        overdue_tasks.append(OverdueTaskItem(**record))

    # Sort by end date ascending to show the most overdue first
    overdue_tasks.sort(key=lambda t: (t.task_end_date is None, t.task_end_date))

    return OverdueTaskListResponse(count=len(overdue_tasks), tasks=overdue_tasks)


@router.get(
    "/count-by-assignee",
    response_model=CountByAssigneeResponse,
    summary="Count tasks by assignee (exact, case-insensitive)",
    response_description="Returns the exact count from the cached Tasks sheet.",
)
def count_by_assignee(name: str) -> CountByAssigneeResponse:
    """Return the exact number of tasks for a given assignee.

    Matching is case-insensitive equality against the "Assignee" column.
    """
    _ensure_cache()

    target = name.strip().casefold()
    if not target:
        raise HTTPException(status_code=400, detail="Parameter 'name' is required.")

    def _norm(value: object) -> str:
        try:
            if pd.isna(value):  # treat NaN/NaT as empty
                return ""
        except Exception:
            pass
        if isinstance(value, str):
            return value.strip().casefold()
        if value is None:
            return ""
        return str(value).strip().casefold()

    count = 0
    for task in _tasks_cache:
        if _norm(task.get("Assignee")) == target:
            count += 1

    return CountByAssigneeResponse(assignee=name, count=count)
