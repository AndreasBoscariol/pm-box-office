from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape

from pm_box_office.db.connection import connect_database
from pm_box_office.orchestration import repository, runner
from pm_box_office.web.db_init import ensure_initialized


WEB_ROOT = Path(__file__).resolve().parents[1]
router = APIRouter()
HIDDEN_SOURCE_KEYS = {"amc_worker"}
templates = Jinja2Templates(
    env=Environment(
        loader=FileSystemLoader(str(WEB_ROOT / "templates")),
        autoescape=select_autoescape(("html", "xml")),
        cache_size=0,
    )
)
templates.env.filters["time_ago"] = lambda value: time_ago(value)


def visible_ingest_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if item.get("source_key") not in HIDDEN_SOURCE_KEYS]


def time_ago(value: object) -> str:
    timestamp = coerce_datetime(value)
    if timestamp is None:
        return "Never"

    now = dt.datetime.now(timestamp.tzinfo or dt.UTC)
    if timestamp.tzinfo is None:
        now = now.replace(tzinfo=None)

    seconds = max(0, int((now - timestamp).total_seconds()))
    if seconds < 60:
        return "Just now"

    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"

    hours = minutes // 60
    if hours < 48:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"

    days = hours // 24
    return f"{days} day{'s' if days != 1 else ''} ago"


def coerce_datetime(value: object) -> dt.datetime | None:
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


@router.get("/sources")
def sources_dashboard(request: Request) -> object:
    conn = connect_database()
    try:
        ensure_initialized(conn)
        repository.refresh_all_source_freshness(conn)
        sources = visible_ingest_items(repository.list_source_summaries(conn))
        recent_runs = visible_ingest_items(repository.list_recent_runs(conn, limit=50))[:12]
        log_tails = {str(run["run_id"]): repository.list_log_tail(conn, run["run_id"], limit=40) for run in recent_runs[:4]}
        conn.commit()
    finally:
        conn.close()
    return templates.TemplateResponse(
        name="sources.html",
        context={
            "request": request,
            "sources": sources,
            "recent_runs": recent_runs,
            "log_tails": log_tails,
            "message": request.query_params.get("message"),
            "error": request.query_params.get("error"),
        },
        request=request,
    )


@router.post("/sources/{source_key}/run")
def run_source(source_key: str) -> object:
    try:
        run_id = runner.start_source_run(source_key, trigger="manual")
    except repository.OrchestrationError as exc:
        return RedirectResponse(url=f"/sources?error={quote(str(exc))}", status_code=303)
    return RedirectResponse(url=f"/sources?message={quote(f'Started {source_key} run {run_id}')}", status_code=303)


@router.post("/sources/{source_key}/retry")
def retry_source(source_key: str) -> object:
    try:
        run_id = runner.start_source_run(source_key, trigger="retry")
    except repository.OrchestrationError as exc:
        return RedirectResponse(url=f"/sources?error={quote(str(exc))}", status_code=303)
    return RedirectResponse(url=f"/sources?message={quote(f'Retried {source_key} as run {run_id}')}", status_code=303)


@router.post("/sources/runs/{run_id}/cancel")
def cancel_source_run(run_id: str) -> object:
    try:
        uuid.UUID(run_id)
        runner.cancel_run(run_id)
    except ValueError:
        return RedirectResponse(url="/sources?error=Invalid%20run%20id", status_code=303)
    except repository.OrchestrationError as exc:
        return RedirectResponse(url=f"/sources?error={quote(str(exc))}", status_code=303)
    return RedirectResponse(url="/sources?message=Cancellation%20requested", status_code=303)


@router.get("/sources/runs/{run_id}/logs", response_class=HTMLResponse)
def source_run_logs(request: Request, run_id: str) -> object:
    conn = connect_database()
    try:
        ensure_initialized(conn)
        logs = repository.list_log_tail(conn, run_id, limit=80)
        conn.rollback()
    finally:
        conn.close()
    return templates.TemplateResponse(
        name="source_run_logs.html",
        context={"request": request, "run_id": run_id, "logs": logs},
        request=request,
    )
