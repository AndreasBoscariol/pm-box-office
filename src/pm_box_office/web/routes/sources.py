from __future__ import annotations

import uuid
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from pm_box_office.db.connection import connect_database
from pm_box_office.orchestration import repository, runner
from pm_box_office.web.db_init import ensure_initialized
from pm_box_office.web.templating import templates
from pm_box_office.web.time_format import duration_until, time_ago


router = APIRouter()
HIDDEN_SOURCE_KEYS = {"amc_worker", "social_x", "nitter"}
HIDDEN_SOURCE_NAMES = {"social x", "social x/nitter poc", "nitter", "nitter poc"}


def visible_ingest_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if not is_hidden_ingest_item(item)]


def is_hidden_ingest_item(item: dict[str, Any]) -> bool:
    source_key = str(item.get("source_key") or "").strip().lower()
    display_name = str(item.get("display_name") or "").strip().lower()
    return source_key in HIDDEN_SOURCE_KEYS or display_name in HIDDEN_SOURCE_NAMES


@router.get("/sources")
def sources_dashboard(request: Request) -> object:
    conn = connect_database()
    try:
        ensure_initialized(conn)
        repository.refresh_all_source_freshness(conn)
        sources = visible_ingest_items(repository.list_source_summaries(conn))
        autorun_state = repository.get_autorun_state(conn)
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
            "autorun_state": autorun_state,
            "recent_runs": recent_runs,
            "log_tails": log_tails,
            "message": request.query_params.get("message"),
            "error": request.query_params.get("error"),
        },
        request=request,
    )


@router.post("/sources/run-all")
def run_all_sources() -> object:
    result = runner.start_run_all(trigger="manual_run_all")
    if result["errors"]:
        return RedirectResponse(url=f"/sources?error={quote('; '.join(result['errors']))}", status_code=303)
    started_count = len(result["started"])
    skipped_count = len(result["skipped"])
    message = f"Started {started_count} ingest runs"
    if skipped_count:
        message = f"{message}; skipped {skipped_count}"
    return RedirectResponse(url=f"/sources?message={quote(message)}", status_code=303)


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
