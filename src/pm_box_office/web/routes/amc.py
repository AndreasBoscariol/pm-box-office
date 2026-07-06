from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from urllib.parse import parse_qs

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from pm_box_office.config import REPO_ROOT
from pm_box_office.db.connection import connect_database
from pm_box_office.sources.amc import db
from pm_box_office.sources.amc.client import HtmlFetcher
from pm_box_office.sources.amc.services import movie_service, sample_service, showtime_service, theatre_service
from pm_box_office.web.db_init import ensure_initialized
from pm_box_office.web.services import amc_workers
from pm_box_office.web.templating import templates


router = APIRouter()
AMC_DISMISSED_RUNS_PATH = REPO_ROOT / "data" / "run" / "amc_dashboard_dismissed_runs.json"


@router.get("/amc")
def amc_home() -> object:
    today = dt.date.today()
    return RedirectResponse(url=f"/amc/campaigns/{today.isoformat()}", status_code=303)


@router.get("/campaigns/{date_value}")
def legacy_campaign(date_value: str) -> object:
    return RedirectResponse(url=f"/amc/campaigns/{date_value}", status_code=303)


@router.get("/amc/campaigns/{date_value}")
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
        showtime_reporting_span = campaign_showtime_reporting_span(conn, exhibition_date)
        dismissed_run_ids = amc_dismissed_run_ids()
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
        collection_diagnostics = db.campaign_collection_diagnostics(conn, campaign_id)
        conn.commit()
    finally:
        conn.close()
    backoff_summary = amc_workers.recent_backoff_summary()
    autoscale_target = amc_workers.autoscaled_worker_target(queue_health, backoff_summary=backoff_summary)
    amc_workers.ensure_local_workers_started(target_count=autoscale_target)
    worker_status = amc_workers.local_worker_status(target_count=autoscale_target)
    return templates.TemplateResponse(
        name="amc_campaign.html",
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
            "showtime_reporting_span": showtime_reporting_span,
            "active_theatres": int(theatre_row[0] or 0),
            "last_theatre_sync": theatre_row[1],
            "recent_runs": recent_runs,
            "worker_running": worker_status["running_count"] > 0,
            "worker_status": worker_status,
            "queue_health": queue_health,
            "collection_diagnostics": collection_diagnostics,
            "sample_set": sample_set,
            "sample_coverage": sample_coverage,
            "sample_overlap": sample_overlap,
            "backoff_summary": backoff_summary,
        },
        request=request,
    )


@router.post("/amc/campaigns/{date_value}/sync-theatres")
def sync_theatres(date_value: str) -> object:
    conn = connect_database()
    try:
        ensure_initialized(conn)
        theatre_service.sync_theatres(conn, HtmlFetcher())
        conn.commit()
    finally:
        conn.close()
    return RedirectResponse(url=f"/amc/campaigns/{date_value}", status_code=303)


@router.post("/amc/campaigns/{date_value}/collect-showtimes")
def collect_showtimes(date_value: str) -> object:
    exhibition_date = dt.date.fromisoformat(date_value)
    conn = connect_database()
    try:
        ensure_initialized(conn)
        showtime_service.create_inventory_run(conn, exhibition_date=exhibition_date, force_refresh=True)
        conn.commit()
    finally:
        conn.close()
    amc_workers.ensure_local_worker_started()
    return RedirectResponse(url=f"/amc/campaigns/{date_value}", status_code=303)


def campaign_showtime_reporting_span(conn: object, exhibition_date: dt.date) -> dict[str, object]:
    row = conn.execute(
        """
        SELECT
            MIN(local_calendar_start_at),
            MAX(local_calendar_start_at),
            COUNT(*)::integer
        FROM amc_showtimes
        WHERE exhibition_date = %s
        """,
        (exhibition_date,),
    ).fetchone()
    count = int(row[2] or 0) if row else 0
    return {
        "count": count,
        "local_start_min": row[0] if row and count else None,
        "local_start_max": row[1] if row and count else None,
        "label": local_calendar_span_label(row[0], row[1]) if row and count else "-",
    }


def local_calendar_span_label(start: object, end: object) -> str:
    start_dt = local_calendar_datetime(start)
    end_dt = local_calendar_datetime(end)
    if start_dt is None or end_dt is None:
        return "-"
    start_label = local_calendar_datetime_label(start_dt, include_date=True)
    if start_dt.date() == end_dt.date():
        end_label = local_calendar_datetime_label(end_dt, include_date=False)
    else:
        end_label = local_calendar_datetime_label(end_dt, include_date=True)
    return f"{start_label} - {end_label}"


def local_calendar_datetime_label(value: dt.datetime, *, include_date: bool) -> str:
    clock = value.strftime("%I:%M %p").lstrip("0").lower()
    if not include_date:
        return clock
    return f"{value.strftime('%b')} {value.day} {clock}"


def local_calendar_datetime(value: object) -> dt.datetime | None:
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return dt.datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


@router.post("/amc/campaigns/{date_value}/movies/{amc_movie_id}")
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
    return RedirectResponse(url=f"/amc/campaigns/{date_value}", status_code=303)


@router.post("/amc/campaigns/{date_value}/movies")
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
    return RedirectResponse(url=f"/amc/campaigns/{date_value}", status_code=303)


@router.post("/amc/campaigns/{date_value}/start-seat-collection")
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
    backoff_summary = amc_workers.recent_backoff_summary()
    autoscale_target = amc_workers.autoscaled_worker_target(queue_health, backoff_summary=backoff_summary)
    amc_workers.ensure_local_workers_started(target_count=autoscale_target)
    return RedirectResponse(url=f"/amc/campaigns/{date_value}", status_code=303)


@router.post("/amc/campaigns/{date_value}/cancel")
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
    return RedirectResponse(url=f"/amc/campaigns/{date_value}", status_code=303)


@router.post("/runs/{run_id}/dismiss")
def dismiss_run(request: Request, run_id: str) -> object:
    db.as_uuid(run_id)
    remember_amc_dismissed_run(run_id)
    target = request.headers.get("referer") or "/"
    return RedirectResponse(url=target, status_code=303)


@router.post("/amc/workers/start")
def start_worker(request: Request) -> object:
    amc_workers.ensure_local_workers_started()
    target = request.headers.get("referer") or "/"
    return RedirectResponse(url=target, status_code=303)


@router.post("/amc/workers/restart")
def restart_worker(request: Request) -> object:
    amc_workers.restart_local_workers()
    target = request.headers.get("referer") or "/"
    return RedirectResponse(url=target, status_code=303)


@router.get("/amc/workers/log", response_class=HTMLResponse)
def worker_log() -> HTMLResponse:
    return HTMLResponse(amc_workers.render_worker_log_tail())


@router.post("/amc/workers/log/clear")
def clear_worker_log(request: Request) -> object:
    amc_workers.clear_log_file(amc_workers.WORKER_LOG_PATH)
    target = request.headers.get("referer") or "/"
    return RedirectResponse(url=target, status_code=303)


@router.get("/amc/workers/backoff-log", response_class=HTMLResponse)
def worker_backoff_log() -> HTMLResponse:
    return HTMLResponse(amc_workers.render_backoff_log_tail())


@router.post("/amc/workers/backoff-log/clear")
def clear_worker_backoff_log(request: Request) -> object:
    amc_workers.clear_log_file(amc_workers.BACKOFF_LOG_PATH)
    target = request.headers.get("referer") or "/"
    return RedirectResponse(url=target, status_code=303)


def amc_dismissed_run_ids(*, path: Path | None = None) -> set[str]:
    dismissed_path = path or AMC_DISMISSED_RUNS_PATH
    if not dismissed_path.exists():
        return set()
    try:
        rows = json.loads(dismissed_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    if not isinstance(rows, list):
        return set()
    return {str(row) for row in rows}


def remember_amc_dismissed_run(run_id: str, *, path: Path | None = None) -> None:
    dismissed_path = path or AMC_DISMISSED_RUNS_PATH
    dismissed_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(amc_dismissed_run_ids(path=dismissed_path) | {str(db.as_uuid(run_id))})
    dismissed_path.write_text(json.dumps(rows[-200:], indent=2), encoding="utf-8")
