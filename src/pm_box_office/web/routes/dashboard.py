from __future__ import annotations

import datetime as dt
import errno
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape

from pm_box_office.config import REPO_ROOT
from pm_box_office.db.connection import connect_database
from pm_box_office.sources.amc import db
from pm_box_office.sources.amc.client import HtmlFetcher
from pm_box_office.sources.amc.services import movie_service, showtime_service, theatre_service
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

WORKER_PID_PATH = REPO_ROOT / "data" / "run" / "amc_worker.pid"
WORKER_HEARTBEAT_PATH = REPO_ROOT / "data" / "run" / "amc_worker.heartbeat"
WORKER_LOG_PATH = REPO_ROOT / "data" / "logs" / "amc_worker.log"
DEFAULT_LOCAL_WORKER_COUNT = 1
DEFAULT_WORKER_BATCH_LIMIT = 1
DEFAULT_WORKER_DELAY_SECONDS = 1.0


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
        queue_health = db.campaign_queue_health(conn, campaign_id)
        conn.commit()
    finally:
        conn.close()
    worker_status = local_worker_status()
    return templates.TemplateResponse(
        name="dashboard.html",
        context={
            "request": request,
            "date_value": exhibition_date.isoformat(),
            "movies": movies,
            "active_theatres": int(theatre_row[0] or 0),
            "last_theatre_sync": theatre_row[1],
            "recent_runs": recent_runs,
            "worker_running": worker_status["running_count"] > 0,
            "worker_status": worker_status,
            "queue_health": queue_health,
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
    form = parse_qs((await request.body()).decode("utf-8"))
    conn = connect_database()
    try:
        ensure_initialized(conn)
        movie_service.set_movie_selected(
            conn,
            exhibition_date=exhibition_date,
            amc_movie_id=amc_movie_id,
            selected=form.get("selected") == ["on"],
        )
        conn.commit()
    finally:
        conn.close()
    return RedirectResponse(url=f"/campaigns/{date_value}", status_code=303)


@router.post("/campaigns/{date_value}/movies")
async def bulk_movies(date_value: str, request: Request) -> object:
    exhibition_date = dt.date.fromisoformat(date_value)
    form = parse_qs((await request.body()).decode("utf-8"))
    action = (form.get("action") or [""])[0]
    conn = connect_database()
    try:
        ensure_initialized(conn)
        movies = movie_service.list_movies_for_date(conn, exhibition_date=exhibition_date)
        if action == "select_all":
            movie_ids = [movie.amc_movie_id for movie in movies]
            movie_service.set_movies_selected(
                conn,
                exhibition_date=exhibition_date,
                amc_movie_ids=movie_ids,
                selected=True,
            )
        elif action == "select_top":
            raw_limit = (form.get("limit") or ["10"])[0]
            try:
                limit = max(0, int(raw_limit))
            except ValueError:
                limit = 10
            movie_ids = [movie.amc_movie_id for movie in movies[:limit]]
            movie_service.set_movies_selected(
                conn,
                exhibition_date=exhibition_date,
                amc_movie_ids=movie_ids,
                selected=True,
            )
        elif action == "clear":
            movie_ids = [movie.amc_movie_id for movie in movies if movie.selected]
            movie_service.set_movies_selected(
                conn,
                exhibition_date=exhibition_date,
                amc_movie_ids=movie_ids,
                selected=False,
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
    ensure_local_workers_started()
    target = request.headers.get("referer") or "/"
    return RedirectResponse(url=target, status_code=303)


@router.get("/workers/log", response_class=HTMLResponse)
def worker_log() -> HTMLResponse:
    return HTMLResponse(render_worker_log_tail())


def ensure_local_worker_started() -> int | None:
    pids = ensure_local_workers_started()
    return pids[0] if pids else None


def ensure_local_workers_started() -> list[int]:
    desired_count = configured_int("AMC_LOCAL_WORKER_COUNT", DEFAULT_LOCAL_WORKER_COUNT, minimum=1)
    pids: list[int] = []
    for index in range(desired_count):
        pid = ensure_local_worker_slot_started(index)
        if pid is not None:
            pids.append(pid)
    return pids


def ensure_local_worker_slot_started(index: int) -> int | None:
    if worker_heartbeat_is_fresh(index):
        return local_worker_pid(index)
    pid = local_worker_pid(index)
    if pid is not None and pid_is_running(pid) and worker_heartbeat_is_fresh(index):
        return pid
    worker_pid_path(index).parent.mkdir(parents=True, exist_ok=True)
    WORKER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    batch_limit = configured_int("AMC_WORKER_BATCH_LIMIT", DEFAULT_WORKER_BATCH_LIMIT, minimum=1)
    delay_seconds = configured_float("AMC_WORKER_DELAY_SECONDS", DEFAULT_WORKER_DELAY_SECONDS, minimum=0.0)
    with WORKER_LOG_PATH.open("ab") as log_file:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "pm_box_office.sources.amc.jobs.worker",
                "--worker-id",
                f"local-{index}",
                "--limit",
                str(batch_limit),
                "--delay-seconds",
                str(delay_seconds),
                "--heartbeat-path",
                str(worker_heartbeat_path(index)),
                "--verbose",
            ],
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    worker_pid_path(index).write_text(str(process.pid), encoding="utf-8")
    return process.pid


def is_local_worker_running() -> bool:
    return local_worker_status()["running_count"] > 0


def local_worker_pid(index: int = 0) -> int | None:
    path = worker_pid_path(index)
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
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


def worker_heartbeat_is_fresh(index: int = 0, max_age_seconds: int = 120) -> bool:
    path = worker_heartbeat_path(index)
    if not path.exists():
        return False
    try:
        heartbeat = float(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return False
    return (dt.datetime.now().timestamp() - heartbeat) <= max_age_seconds


def worker_pid_path(index: int) -> Path:
    if index == 0:
        return WORKER_PID_PATH
    return WORKER_PID_PATH.with_name(f"amc_worker_{index}.pid")


def worker_heartbeat_path(index: int) -> Path:
    if index == 0:
        return WORKER_HEARTBEAT_PATH
    return WORKER_HEARTBEAT_PATH.with_name(f"amc_worker_{index}.heartbeat")


def local_worker_status() -> dict[str, int | float]:
    desired_count = configured_int("AMC_LOCAL_WORKER_COUNT", DEFAULT_LOCAL_WORKER_COUNT, minimum=1)
    running_count = sum(1 for index in range(desired_count) if worker_heartbeat_is_fresh(index))
    return {
        "desired_count": desired_count,
        "running_count": running_count,
        "batch_limit": configured_int("AMC_WORKER_BATCH_LIMIT", DEFAULT_WORKER_BATCH_LIMIT, minimum=1),
        "delay_seconds": configured_float("AMC_WORKER_DELAY_SECONDS", DEFAULT_WORKER_DELAY_SECONDS, minimum=0.0),
    }


def configured_int(name: str, default: int, *, minimum: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return max(minimum, int(raw_value))
    except ValueError:
        return default


def configured_float(name: str, default: float, *, minimum: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return max(minimum, float(raw_value))
    except ValueError:
        return default


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
