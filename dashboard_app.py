"""Streamlit dashboard for the RAG project management backend."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st

try:  # pragma: no cover - optional configuration module
    import config  # type: ignore
except Exception:  # pragma: no cover - configuration may not exist
    config = None  # type: ignore


def _resolve_api_server() -> str:
    """Read the API server base URL from the environment or config module."""

    env_value = os.getenv("API_SERVER")
    if env_value:
        return env_value.rstrip("/")

    if config is not None and hasattr(config, "API_SERVER"):
        value = getattr(config, "API_SERVER")
        if isinstance(value, str) and value:
            return value.rstrip("/")

    return "http://localhost:8000"


API_SERVER = _resolve_api_server()
DEFAULT_SPRINT = 20


def _request_json(
    api_server: str,
    method: str,
    path: str,
    *,
    params: Dict[str, Any] | None = None,
    json: Dict[str, Any] | None = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Call the backend API and return the parsed JSON payload."""

    url = f"{api_server.rstrip('/')}/{path.lstrip('/')}"
    try:
        response = requests.request(
            method,
            url,
            params=params,
            json=json,
            timeout=timeout,
        )
    except requests.RequestException as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"Failed to reach {url}: {exc}") from exc

    if response.status_code >= 400:
        try:
            payload = response.json()
            detail = payload.get("detail") or payload
        except ValueError:
            detail = response.text or "Unknown error"
        raise RuntimeError(f"{response.status_code} error from {url}: {detail}")

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(f"Invalid JSON response from {url}") from exc


@st.cache_data(show_spinner=False)
def _cached_task_summary(api_server: str) -> Dict[str, Any]:
    return _request_json(api_server, "GET", "project-dashboard/task-summary")


@st.cache_data(show_spinner=False)
def _cached_sprint_tasks(api_server: str, sprint: int) -> Dict[str, Any]:
    return _request_json(
        api_server,
        "GET",
        "project-dashboard/tasks-of-sprint",
        params={"sprint": sprint},
    )


def _normalise_tasks_frame(tasks: List[Dict[str, Any]]) -> pd.DataFrame:
    """Return a DataFrame with consistent column names for downstream charts."""

    frame = pd.DataFrame(tasks)
    if frame.empty:
        return frame

    rename_map = {
        "assignee": "Assignee",
        "task_title": "Task Title",
        "task_description": "Task Description",
        "team": "Team",
        "status": "Status",
        "priority": "Priority",
        "story_point": "Story Point",
    }
    frame = frame.rename(columns={k: v for k, v in rename_map.items() if k in frame.columns})

    for column in [
        "Task Title",
        "Assignee",
        "Team",
        "Status",
        "Priority",
        "Story Point",
    ]:
        if column not in frame.columns:
            frame[column] = pd.NA

    return frame


def _display_summary_panel(summary: Dict[str, Any]) -> None:
    st.subheader("Summary")
    metrics = st.columns(4)
    metrics[0].metric("Completed today", summary.get("completed_today", 0))
    metrics[1].metric("Updated today", summary.get("updated_today", 0))
    metrics[2].metric("Created today", summary.get("created_today", 0))
    metrics[3].metric("Overdue", summary.get("overdue", 0))


def _display_sprint_overview(tasks_frame: pd.DataFrame, sprint: int) -> None:
    st.subheader(f"Sprint {sprint} overview")

    if tasks_frame.empty:
        st.info("No tasks available for the selected sprint.")
        return

    display_columns = [
        "Task Title",
        "Assignee",
        "Team",
        "Status",
        "Priority",
        "Story Point",
    ]
    st.dataframe(
        tasks_frame[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    # Status distribution
    status_counts = (
        tasks_frame["Status"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    )
    status_chart = status_counts.value_counts().sort_values(ascending=False)

    # Priority distribution
    priority_counts = (
        tasks_frame["Priority"].fillna("Unspecified").astype(str).str.strip().replace("", "Unspecified")
    )
    priority_chart = priority_counts.value_counts().sort_values(ascending=False)

    chart_columns = st.columns(2)
    chart_columns[0].write("**Task status distribution**")
    chart_columns[0].bar_chart(status_chart)
    chart_columns[1].write("**Task priority overview**")
    chart_columns[1].bar_chart(priority_chart)

    # Team progress information
    team_frame = tasks_frame.copy()
    team_frame["Team"] = (
        team_frame["Team"].fillna("Unassigned").astype(str).str.strip().replace("", "Unassigned")
    )
    team_frame["is_done"] = team_frame["Status"].astype(str).str.lower().eq("done")

    progress = (
        team_frame.groupby("Team")["is_done"]
        .agg(completed="sum", total="count")
        .sort_values(by=["completed", "total"], ascending=False)
    )
    progress["progress"] = progress["completed"] / progress["total"].replace(0, 1)

    st.write("**Team progress**")
    if progress.empty:
        st.info("No team data available for this sprint.")
        return

    for team, row in progress.iterrows():
        st.write(f"**{team}**")
        st.progress(float(row["progress"]))
        st.caption(f"{int(row['completed'])} of {int(row['total'])} tasks completed")


def main() -> None:
    st.set_page_config(page_title="Project Dashboard", layout="wide")
    st.title("Project Dashboard")
    st.caption(f"Connected to API server: {API_SERVER}")

    refresh_column, _ = st.columns([1, 3])
    if refresh_column.button(
        "🔄 Refresh data",
        help="Trigger the backend build process to refresh cached data.",
    ):
        with st.spinner("Refreshing project data..."):
            try:
                build_response = _request_json(
                    API_SERVER,
                    "POST",
                    "build",
                    json={"rebuild": False},
                )
            except RuntimeError as exc:
                st.error(f"Failed to refresh data: {exc}")
            else:
                _cached_task_summary.clear()
                _cached_sprint_tasks.clear()
                st.success("Data refreshed successfully.")
                if build_response.get("project_dashboard", {}).get("count") is not None:
                    count = build_response["project_dashboard"].get("count")
                    st.caption(f"Cached {count} project tasks after rebuild.")

    # Task summary panel
    try:
        summary_payload = _cached_task_summary(API_SERVER)
    except RuntimeError as exc:
        st.warning(f"Unable to load task summary: {exc}")
    else:
        _display_summary_panel(summary_payload)

    # Sprint overview visuals
    try:
        sprint_payload = _cached_sprint_tasks(API_SERVER, DEFAULT_SPRINT)
    except RuntimeError as exc:
        st.warning(f"Unable to load sprint data: {exc}")
        return

    tasks = sprint_payload.get("tasks", [])
    tasks_frame = _normalise_tasks_frame(tasks)
    _display_sprint_overview(tasks_frame, sprint_payload.get("sprint", DEFAULT_SPRINT))


if __name__ == "__main__":  # pragma: no cover - entry point for `streamlit run`
    main()
