from __future__ import annotations

import uuid
from pathlib import Path
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
templates = Jinja2Templates(
    env=Environment(
        loader=FileSystemLoader(str(WEB_ROOT / "templates")),
        autoescape=select_autoescape(("html", "xml")),
        cache_size=0,
    )
)


@router.get("/sources")
def sources_dashboard(request: Request) -> object:
    conn = connect_database()
    try:
        ensure_initialized(conn)
        repository.refresh_all_source_freshness(conn)
        sources = repository.list_source_summaries(conn)
        recent_runs = repository.list_recent_runs(conn, limit=12)
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

