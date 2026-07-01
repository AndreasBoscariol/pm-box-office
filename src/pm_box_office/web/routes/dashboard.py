from __future__ import annotations

import datetime as dt
import errno
import json
import math
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
from pm_box_office.sources.amc.diagnostics import BACKOFF_LOG_PATH
from pm_box_office.sources.amc.services import movie_service, sample_service, showtime_service, theatre_service
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
DASHBOARD_DISMISSED_RUNS_PATH = REPO_ROOT / "data" / "run" / "amc_dashboard_dismissed_runs.json"
DEFAULT_LOCAL_WORKER_COUNT = 1
DEFAULT_LOCAL_WORKER_MAX = 2
DEFAULT_AUTOSCALE_DUE_PER_WORKER = 80
DEFAULT_AUTOSCALE_LATE_PER_WORKER = 40
DEFAULT_AUTOSCALE_PEAK_PER_WORKER = 220
DEFAULT_AUTOSCALE_BACKOFF_CAP = 1
DEFAULT_WORKER_BATCH_LIMIT = 1
DEFAULT_WORKER_DELAY_SECONDS = 3.0
THROTTLE_STATUS_CODES = {429, 500, 502, 503, 504}
SEAT_BACKOFF_EVENTS = {"rsc_missing_seats", "seat_task_failed"}
HTTP_BACKOFF_EVENTS = {"http_retry", "http_failed", "empty_body"}


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
        theatre_row = conn.execute(
            """
            SELECT COUNT(*) FILTER (WHERE active)::integer, MAX(last_seen_at)
            FROM amc_theatres
            """
        ).fetchone()
        sample_set = None
        sample_coverage: dict[str, object] = {}
        if int(theatre_row[0] or 0) > 0:
            sample_set = sample_service.ensure_default_theatre_sample(conn)
            sample_coverage = sample_service.sample_coverage(
                conn,
                sample_set=sample_set,
                exhibition_date=exhibition_date,
            )
            sample_overlap = db.theatre_sample_showtime_overlap(
                conn,
                sample_set_id=sample_set.sample_set_id,
                exhibition_date=exhibition_date,
            )
        else:
            sample_overlap = {}
        movies = movie_service.list_movies_for_date(
            conn,
            exhibition_date=exhibition_date,
            sample_set_id=sample_set.sample_set_id if sample_set is not None else None,
        )
        selected_movies = [movie for movie in movies if movie.selected]
        inventory_showtimes = int(sample_coverage.get("full_showtimes") or 0)
        sampled_showtimes = int(sample_coverage.get("sampled_showtimes") or 0)
        selected_sampled_showtimes = sum(int(movie.sampled_showtime_count or 0) for movie in selected_movies)
        dismissed_run_ids = dashboard_dismissed_run_ids()
        recent_runs = conn.execute(
            """
            SELECT run_id, run_type, status, tasks_total, tasks_succeeded, tasks_failed
            FROM collection_runs
            WHERE campaign_id = %s
              AND status <> 'cancelled'
            ORDER BY started_at DESC NULLS LAST
            LIMIT 20
            """,
            (campaign_id,),
        ).fetchall()
        recent_runs = [run for run in recent_runs if str(run[0]) not in dismissed_run_ids][:5]
        queue_health = db.campaign_queue_health(conn, campaign_id)
        conn.commit()
    finally:
        conn.close()
    backoff_summary = recent_backoff_summary()
    autoscale_target = autoscaled_worker_target(queue_health, backoff_summary=backoff_summary)
    ensure_local_workers_started(target_count=autoscale_target)
    worker_status = local_worker_status(target_count=autoscale_target)
    return templates.TemplateResponse(
        name="dashboard.html",
        context={
            "request": request,
            "date_value": exhibition_date.isoformat(),
            "movies": movies,
            "selected_movies": selected_movies,
            "selected_movie_count": len(selected_movies),
            "selected_sampled_showtimes": selected_sampled_showtimes,
            "inventory_movie_count": len(movies),
            "inventory_showtimes": inventory_showtimes,
            "sampled_showtimes": sampled_showtimes,
            "active_theatres": int(theatre_row[0] or 0),
            "last_theatre_sync": theatre_row[1],
            "recent_runs": recent_runs,
            "worker_running": worker_status["running_count"] > 0,
            "worker_status": worker_status,
            "queue_health": queue_health,
            "sample_set": sample_set,
            "sample_coverage": sample_coverage,
            "sample_overlap": sample_overlap,
            "backoff_summary": backoff_summary,
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
        if action == "select_the_numbers_active":
            raw_lookback = (form.get("lookback_days") or ["7"])[0]
            try:
                lookback_days = max(1, int(raw_lookback))
            except ValueError:
                lookback_days = 7
            movie_service.select_the_numbers_active_movies(
                conn,
                exhibition_date=exhibition_date,
                lookback_days=lookback_days,
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
async def start_seat_collection(date_value: str, request: Request) -> object:
    exhibition_date = dt.date.fromisoformat(date_value)
    form = parse_qs((await request.body()).decode("utf-8"))
    sample_key = (form.get("sample_key") or [sample_service.DEFAULT_SAMPLE_KEY])[0]
    queue_health: dict[str, object] | None = None
    conn = connect_database()
    try:
        ensure_initialized(conn)
        movie_service.create_seat_collection_run(
            conn,
            exhibition_date=exhibition_date,
            sample_key=sample_key,
        )
        campaign_id = db.ensure_campaign(conn, exhibition_date)
        queue_health = db.campaign_queue_health(conn, campaign_id)
        conn.commit()
    finally:
        conn.close()
    backoff_summary = recent_backoff_summary()
    ensure_local_workers_started(target_count=autoscaled_worker_target(queue_health, backoff_summary=backoff_summary))
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


@router.post("/runs/{run_id}/dismiss")
def dismiss_run(request: Request, run_id: str) -> object:
    db.as_uuid(run_id)
    remember_dashboard_dismissed_run(run_id)
    target = request.headers.get("referer") or "/"
    return RedirectResponse(url=target, status_code=303)


@router.post("/workers/start")
def start_worker(request: Request) -> object:
    ensure_local_workers_started()
    target = request.headers.get("referer") or "/"
    return RedirectResponse(url=target, status_code=303)


@router.get("/workers/log", response_class=HTMLResponse)
def worker_log() -> HTMLResponse:
    return HTMLResponse(render_worker_log_tail())


@router.post("/workers/log/clear")
def clear_worker_log(request: Request) -> object:
    clear_log_file(WORKER_LOG_PATH)
    target = request.headers.get("referer") or "/"
    return RedirectResponse(url=target, status_code=303)


@router.get("/workers/backoff-log", response_class=HTMLResponse)
def worker_backoff_log() -> HTMLResponse:
    return HTMLResponse(render_backoff_log_tail())


@router.post("/workers/backoff-log/clear")
def clear_worker_backoff_log(request: Request) -> object:
    clear_log_file(BACKOFF_LOG_PATH)
    target = request.headers.get("referer") or "/"
    return RedirectResponse(url=target, status_code=303)


def ensure_local_worker_started() -> int | None:
    pids = ensure_local_workers_started()
    return pids[0] if pids else None


def ensure_local_workers_started(*, target_count: int | None = None) -> list[int]:
    desired_count = normalized_worker_target(target_count)
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
    heartbeat_age = worker_heartbeat_age_seconds(index)
    return heartbeat_age is not None and heartbeat_age <= max_age_seconds


def worker_heartbeat_age_seconds(index: int = 0) -> float | None:
    path = worker_heartbeat_path(index)
    if not path.exists():
        return None
    try:
        heartbeat = float(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None
    return max(0.0, dt.datetime.now().timestamp() - heartbeat)


def worker_pid_path(index: int) -> Path:
    if index == 0:
        return WORKER_PID_PATH
    return WORKER_PID_PATH.with_name(f"amc_worker_{index}.pid")


def worker_heartbeat_path(index: int) -> Path:
    if index == 0:
        return WORKER_HEARTBEAT_PATH
    return WORKER_HEARTBEAT_PATH.with_name(f"amc_worker_{index}.heartbeat")


def local_worker_status(*, target_count: int | None = None) -> dict[str, object]:
    baseline_count = baseline_worker_count()
    max_count = max_worker_count()
    desired_count = normalized_worker_target(target_count)
    running_slots = max(desired_count, max_count)
    slots = [local_worker_slot_status(index) for index in range(running_slots)]
    running_count = sum(1 for slot in slots if slot["fresh"])
    return {
        "baseline_count": baseline_count,
        "desired_count": desired_count,
        "max_count": max_count,
        "running_count": running_count,
        "slots": slots,
        "autoscale_enabled": autoscale_enabled(),
        "batch_limit": configured_int("AMC_WORKER_BATCH_LIMIT", DEFAULT_WORKER_BATCH_LIMIT, minimum=1),
        "delay_seconds": configured_float("AMC_WORKER_DELAY_SECONDS", DEFAULT_WORKER_DELAY_SECONDS, minimum=0.0),
        "due_per_worker": configured_int(
            "AMC_AUTOSCALE_DUE_PER_WORKER",
            DEFAULT_AUTOSCALE_DUE_PER_WORKER,
            minimum=1,
        ),
        "late_per_worker": configured_int(
            "AMC_AUTOSCALE_LATE_PER_WORKER",
            DEFAULT_AUTOSCALE_LATE_PER_WORKER,
            minimum=1,
        ),
        "peak_per_worker": configured_int(
            "AMC_AUTOSCALE_PEAK_PER_WORKER",
            DEFAULT_AUTOSCALE_PEAK_PER_WORKER,
            minimum=1,
        ),
        "backoff_cap": configured_int(
            "AMC_AUTOSCALE_BACKOFF_CAP",
            DEFAULT_AUTOSCALE_BACKOFF_CAP,
            minimum=1,
        ),
    }


def local_worker_slot_status(index: int) -> dict[str, int | float | str | bool | None]:
    pid = local_worker_pid(index)
    heartbeat_age_seconds = worker_heartbeat_age_seconds(index)
    fresh = heartbeat_age_seconds is not None and heartbeat_age_seconds <= 120
    process_running = pid_is_running(pid) if pid is not None else False
    if fresh:
        status = "running"
    elif pid is not None and process_running:
        status = "stale"
    elif pid is not None:
        status = "stopped"
    else:
        status = "idle"
    return {
        "index": index,
        "worker_id": f"local-{index}",
        "pid": pid,
        "status": status,
        "fresh": fresh,
        "process_running": process_running,
        "heartbeat_age_seconds": heartbeat_age_seconds,
    }


def autoscaled_worker_target(
    queue_health: dict[str, object] | None,
    *,
    backoff_summary: dict[str, object] | None = None,
) -> int:
    baseline_count = baseline_worker_count()
    max_count = max_worker_count()
    if not autoscale_enabled() or not queue_health:
        return min(baseline_count, max_count)
    due_per_worker = configured_int(
        "AMC_AUTOSCALE_DUE_PER_WORKER",
        DEFAULT_AUTOSCALE_DUE_PER_WORKER,
        minimum=1,
    )
    late_per_worker = configured_int(
        "AMC_AUTOSCALE_LATE_PER_WORKER",
        DEFAULT_AUTOSCALE_LATE_PER_WORKER,
        minimum=1,
    )
    peak_per_worker = configured_int(
        "AMC_AUTOSCALE_PEAK_PER_WORKER",
        DEFAULT_AUTOSCALE_PEAK_PER_WORKER,
        minimum=1,
    )
    due_now = int(queue_health.get("due_now") or 0)
    late = int(queue_health.get("late") or 0)
    peak_tasks_per_minute = int(queue_health.get("peak_tasks_per_minute") or 0)
    target = max(
        baseline_count,
        math.ceil(due_now / due_per_worker),
        math.ceil(late / late_per_worker),
        math.ceil(peak_tasks_per_minute / peak_per_worker),
    )
    target = min(target, max_count)
    if int((backoff_summary or {}).get("backoff_pressure_events") or 0) > 0:
        backoff_cap = configured_int(
            "AMC_AUTOSCALE_BACKOFF_CAP",
            DEFAULT_AUTOSCALE_BACKOFF_CAP,
            minimum=1,
        )
        target = min(target, max(baseline_count, backoff_cap))
    return target


def baseline_worker_count() -> int:
    return configured_int("AMC_LOCAL_WORKER_COUNT", DEFAULT_LOCAL_WORKER_COUNT, minimum=1)


def max_worker_count() -> int:
    baseline_count = baseline_worker_count()
    return configured_int("AMC_LOCAL_WORKER_MAX", DEFAULT_LOCAL_WORKER_MAX, minimum=baseline_count)


def autoscale_enabled() -> bool:
    return configured_bool("AMC_AUTOSCALE_ENABLED", default=True)


def normalized_worker_target(target_count: int | None) -> int:
    requested_count = baseline_worker_count() if target_count is None else max(1, target_count)
    return min(requested_count, max_worker_count())


def configured_int(name: str, default: int, *, minimum: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return max(minimum, default)
    try:
        return max(minimum, int(raw_value))
    except ValueError:
        return max(minimum, default)


def configured_bool(name: str, *, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off"}


def configured_float(name: str, default: float, *, minimum: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return max(minimum, float(raw_value))
    except ValueError:
        return default


def recent_backoff_summary(
    *,
    path: Path | None = None,
    lookback: dt.timedelta = dt.timedelta(hours=1),
) -> dict[str, object]:
    log_path = path or BACKOFF_LOG_PATH
    now = dt.datetime.now(dt.timezone.utc)
    cutoff = now - lookback
    summary: dict[str, object] = {
        "events": 0,
        "throttling_events": 0,
        "seat_backoff_events": 0,
        "http_backoff_events": 0,
        "successful_fallbacks": 0,
        "backoff_pressure_events": 0,
        "backoff_pressure_rate": 0.0,
        "http_retries": 0,
        "http_failures": 0,
        "rsc_missing_seats": 0,
        "rsc_fallback_succeeded": 0,
        "seat_task_failed": 0,
        "latest_event_at": None,
    }
    if not log_path.exists():
        return summary
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_at = parse_diagnostic_timestamp(row.get("timestamp_utc"))
        if event_at is None or event_at < cutoff:
            continue
        event_type = str(row.get("event_type") or "")
        status_code = diagnostic_status_code(row.get("status_code"))
        summary["events"] = int(summary["events"]) + 1
        summary["latest_event_at"] = event_at
        if event_type == "http_retry":
            summary["http_retries"] = int(summary["http_retries"]) + 1
        elif event_type == "http_failed":
            summary["http_failures"] = int(summary["http_failures"]) + 1
        elif event_type == "rsc_missing_seats":
            summary["rsc_missing_seats"] = int(summary["rsc_missing_seats"]) + 1
        elif event_type == "rsc_fallback_succeeded":
            summary["rsc_fallback_succeeded"] = int(summary["rsc_fallback_succeeded"]) + 1
            summary["successful_fallbacks"] = int(summary["successful_fallbacks"]) + 1
        elif event_type == "seat_task_failed":
            summary["seat_task_failed"] = int(summary["seat_task_failed"]) + 1
        if event_type in SEAT_BACKOFF_EVENTS:
            summary["seat_backoff_events"] = int(summary["seat_backoff_events"]) + 1
            summary["backoff_pressure_events"] = int(summary["backoff_pressure_events"]) + 1
        if event_type in HTTP_BACKOFF_EVENTS or status_code in THROTTLE_STATUS_CODES:
            summary["http_backoff_events"] = int(summary["http_backoff_events"]) + 1
            summary["backoff_pressure_events"] = int(summary["backoff_pressure_events"]) + 1
        if status_code in THROTTLE_STATUS_CODES:
            summary["throttling_events"] = int(summary["throttling_events"]) + 1
    summary["backoff_pressure_rate"] = int(summary["backoff_pressure_events"]) / max(int(summary["events"]), 1)
    return summary


def parse_diagnostic_timestamp(value: object) -> dt.datetime | None:
    if value is None:
        return None
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def diagnostic_status_code(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def render_worker_log_tail(line_count: int = 40) -> str:
    if not WORKER_LOG_PATH.exists():
        return '<pre class="log-tail">No worker log yet.</pre>'
    lines = WORKER_LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()[-line_count:]
    escaped = "\n".join(html_escape(line) for line in lines)
    return f'<pre class="log-tail">{escaped}</pre>'


def render_backoff_log_tail(line_count: int = 40) -> str:
    if not BACKOFF_LOG_PATH.exists():
        return '<pre class="log-tail">No backoff diagnostics yet.</pre>'
    lines = BACKOFF_LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()[-line_count:]
    escaped = "\n".join(html_escape(line) for line in lines)
    return f'<pre class="log-tail">{escaped}</pre>'


def dashboard_dismissed_run_ids(*, path: Path | None = None) -> set[str]:
    dismissed_path = path or DASHBOARD_DISMISSED_RUNS_PATH
    if not dismissed_path.exists():
        return set()
    try:
        rows = json.loads(dismissed_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    if not isinstance(rows, list):
        return set()
    return {str(row) for row in rows}


def remember_dashboard_dismissed_run(run_id: str, *, path: Path | None = None) -> None:
    dismissed_path = path or DASHBOARD_DISMISSED_RUNS_PATH
    dismissed_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(dashboard_dismissed_run_ids(path=dismissed_path) | {str(db.as_uuid(run_id))})
    dismissed_path.write_text(json.dumps(rows[-200:], indent=2), encoding="utf-8")


def clear_log_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def html_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
