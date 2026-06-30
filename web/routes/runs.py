from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape

from scripts.db import connect_database
from scripts.ingest.amc import db
from scripts.ingest.amc.services import progress_service
from web.db_init import ensure_initialized


router = APIRouter()
templates = Jinja2Templates(
    env=Environment(
        loader=FileSystemLoader("web/templates"),
        autoescape=select_autoescape(("html", "xml")),
        cache_size=0,
    )
)


@router.get("/runs/{run_id}/progress", response_class=HTMLResponse)
def run_progress(request: Request, run_id: str) -> object:
    conn = connect_database()
    try:
        ensure_initialized(conn)
        progress = progress_service.run_progress(conn, run_id)
        conn.rollback()
    finally:
        conn.close()
    if progress["status"] == "cancelled":
        return HTMLResponse("")
    return templates.TemplateResponse(
        name="progress.html",
        context={"request": request, "progress": progress},
        request=request,
    )


@router.post("/runs/{run_id}/cancel")
def cancel_run(request: Request, run_id: str) -> object:
    conn = connect_database()
    try:
        ensure_initialized(conn)
        db.cancel_run(conn, db.as_uuid(run_id))
        conn.commit()
    finally:
        conn.close()
    target = request.headers.get("referer") or "/"
    return RedirectResponse(url=target, status_code=303)
