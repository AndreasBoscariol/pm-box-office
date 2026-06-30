from __future__ import annotations

import datetime as dt
import errno
import os
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape

from scripts.db import connect_database
from scripts.ingest.amc import db
from scripts.ingest.amc.client import HtmlFetcher
from scripts.ingest.amc.services import movie_service, showtime_service, theatre_service
from web.db_init import ensure_initialized


router = APIRouter()
templates = Jinja2Templates(
    env=Environment(
        loader=FileSystemLoader("web/templates"),
        autoescape=select_autoescape(("html", "xml")),
        cache_size=0,
    )
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_PID_PATH = REPO_ROOT / "data" / "run" / "amc_worker.pid"
WORKER_HEARTBEAT_PATH = REPO_ROOT / "data" / "run" / "amc_worker.heartbeat"
WORKER_LOG_PATH = REPO_ROOT / "data" / "logs" / "amc_worker.log"


@router.get("/")
def index(request: Request) -> object:
    today = dt.date.today()
    return RedirectResponse(url=f"/campaigns/{today.isoformat()}", status_code=303)


@router.get("/campaigns/{date_value}")
def campaign(request: Request, date_value: str) -> object:
    exhibition_date = dt.date.fromisoformat(date_value)
    conn = connect_database()
    try:
        ensure_initialized(conn)
        campaign_id = db.ensure_campaign(conn, exhibition_date)
        movies = movie_service.list_movies_for_date(conn, exhibition_date=exhibition_date)
        theatre_row = conn.execute(
            """
            SELECT COUNT(*) FILTER (WHERE active)::integer, MAX(last_seen_at)
            FROM amc_theatres
            """
        ).fetchone()
        recent_runs = conn.execute(
            """
            SELECT run_id, run_type, status, tasks_total, tasks_succeeded, tasks_failed
            FROM collection_runs
            WHERE campaign_id = %s
              AND status <> 'cancelled'
            ORDER BY started_at DESC NULLS LAST
            LIMIT 5
            """,
            (campaign_id,),
        ).fetchall()
        conn.commit()
    finally:
        conn.close()
    return templates.TemplateResponse(
        name="dashboard.html",
        context={
            "request": request,
            "date_value": exhibition_date.isoformat(),
            "movies": movies,
            "active_theatres": int(theatre_row[0] or 0),
            "last_theatre_sync": theatre_row[1],
            "recent_runs": recent_runs,
            "worker_running": is_local_worker_running(),
        },
        request=request,
    )


@router.post("/campaigns/{date_value}/sync-theatres")
def sync_theatres(date_value: str) -> object:
    conn = connect_database()
    try:
        ensure_initialized(conn)
        theatre_service.sync_theatres(conn, HtmlFetcher())
        conn.commit()
    finally:
        conn.close()
    return RedirectResponse(url=f"/campaigns/{date_value}", status_code=303)


@router.post("/campaigns/{date_value}/collect-showtimes")
def collect_showtimes(date_value: str) -> object:
    exhibition_date = dt.date.fromisoformat(date_value)
    conn = connect_database()
    try:
        ensure_initialized(conn)
        showtime_service.create_inventory_run(conn, exhibition_date=exhibition_date)
        conn.commit()
    finally:
        conn.close()
    ensure_local_worker_started()
    return RedirectResponse(url=f"/campaigns/{date_value}", status_code=303)


@router.post("/campaigns/{date_value}/movies/{amc_movie_id}")
async def toggle_movie(date_value: str, amc_movie_id: str, request: Request) -> object:
    exhibition_date = dt.date.fromisoformat(date_value)
    form = await request.form()
    conn = connect_database()
    try:
        ensure_initialized(conn)
        movie_service.set_movie_selected(
            conn,
            exhibition_date=exhibition_date,
            amc_movie_id=amc_movie_id,
            selected=form.get("selected") == "on",
        )
        conn.commit()
    finally:
        conn.close()
    return RedirectResponse(url=f"/campaigns/{date_value}", status_code=303)


@router.post("/campaigns/{date_value}/start-seat-collection")
def start_seat_collection(date_value: str) -> object:
    exhibition_date = dt.date.fromisoformat(date_value)
    conn = connect_database()
    try:
        ensure_initialized(conn)
        movie_service.create_seat_collection_run(conn, exhibition_date=exhibition_date)
        conn.commit()
    finally:
        conn.close()
    ensure_local_worker_started()
    return RedirectResponse(url=f"/campaigns/{date_value}", status_code=303)


@router.post("/campaigns/{date_value}/cancel")
def cancel_campaign(date_value: str) -> object:
    exhibition_date = dt.date.fromisoformat(date_value)
    conn = connect_database()
    try:
        ensure_initialized(conn)
        campaign_id = db.ensure_campaign(conn, exhibition_date)
        db.cancel_campaign_runs(conn, campaign_id)
        conn.commit()
    finally:
        conn.close()
    return RedirectResponse(url=f"/campaigns/{date_value}", status_code=303)


@router.post("/workers/start")
def start_worker(request: Request) -> object:
    ensure_local_worker_started()
    target = request.headers.get("referer") or "/"
    return RedirectResponse(url=target, status_code=303)


@router.get("/workers/log", response_class=HTMLResponse)
def worker_log() -> HTMLResponse:
    return HTMLResponse(render_worker_log_tail())


def ensure_local_worker_started() -> int | None:
    if worker_heartbeat_is_fresh():
        return local_worker_pid()
    pid = local_worker_pid()
    if pid is not None and pid_is_running(pid) and worker_heartbeat_is_fresh():
        return pid
    WORKER_PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    WORKER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_file = WORKER_LOG_PATH.open("ab")
    process = subprocess.Popen(
        [sys.executable, "-m", "scripts.ingest.amc.jobs.worker", "--limit", "1", "--verbose"],
        cwd=REPO_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    WORKER_PID_PATH.write_text(str(process.pid), encoding="utf-8")
    return process.pid


def is_local_worker_running() -> bool:
    return worker_heartbeat_is_fresh()


def local_worker_pid() -> int | None:
    if not WORKER_PID_PATH.exists():
        return None
    try:
        return int(WORKER_PID_PATH.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError as exc:
        if exc.errno == errno.EPERM:
            return True
        return False
    return True


def worker_heartbeat_is_fresh(max_age_seconds: int = 120) -> bool:
    if not WORKER_HEARTBEAT_PATH.exists():
        return False
    try:
        heartbeat = float(WORKER_HEARTBEAT_PATH.read_text(encoding="utf-8").strip())
    except ValueError:
        return False
    return (dt.datetime.now().timestamp() - heartbeat) <= max_age_seconds


def render_worker_log_tail(line_count: int = 40) -> str:
    if not WORKER_LOG_PATH.exists():
        return '<pre class="log-tail">No worker log yet.</pre>'
    lines = WORKER_LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()[-line_count:]
    escaped = "\n".join(html_escape(line) for line in lines)
    return f'<pre class="log-tail">{escaped}</pre>'


def html_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
