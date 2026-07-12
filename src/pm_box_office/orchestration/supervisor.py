"""Supervisor process for a single local ingest run."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
import threading
import traceback
from typing import Any

from pm_box_office.config import REPO_ROOT
from pm_box_office.db.connection import connect_database, database_url_from_env
from pm_box_office.domain import movies as movie_identity
from pm_box_office.orchestration import repository
from pm_box_office.orchestration.forecast_routing import refresh_rule_for


def enqueue_reconciled_forecast_refreshes(conn: Any, movie_ids: list[int]) -> int:
    if not movie_ids:
        return 0
    table = conn.execute("SELECT to_regclass(%s)", ("analytics.eda_movie_openings",)).fetchone()
    if not table or table[0] is None:
        return 0
    from models.boxoffice.refresh_queue import enqueue_refresh

    rows = conn.execute(
        """
        SELECT release_run_id, movie_id
        FROM analytics.eda_movie_openings
        WHERE movie_id = ANY(%s)
        """,
        (movie_ids,),
    ).fetchall()
    for release_run_id, movie_id in rows:
        enqueue_refresh(
            conn,
            release_run_id=int(release_run_id),
            movie_id=int(movie_id),
            reason="prediction_identity_reconciled",
            priority=10,
        )
    return len(rows)


def enqueue_upcoming_candidate_refreshes(conn: Any) -> int:
    """Queue estimate-backed openings even before a canonical release run exists."""
    from models.boxoffice.future_candidates import build_future_candidate_artifacts
    from models.boxoffice.refresh_queue import enqueue_refresh

    now = dt.datetime.now(dt.timezone.utc)
    try:
        candidates = build_future_candidate_artifacts(conn, as_of_utc=now)
    except Exception:
        # A source run can be valid before the optional forecast-source tables
        # have been migrated.  Do not turn that ingest into a failed run.
        return 0
    estimate_backed_movie_ids = (
        {int(movie_id) for movie_id in candidates.daily_baseline["movie_id"].tolist() if movie_id is not None}
        if "movie_id" in candidates.daily_baseline.columns
        else set()
    )
    queued = 0
    for movie in candidates.movies:
        # The queue is a debounced upsert, so repeated source runs coalesce into
        # one refresh rather than creating a polling loop.
        if movie.opening_weekend_start < now.date() - dt.timedelta(days=1):
            continue
        if movie.movie_id not in estimate_backed_movie_ids:
            continue
        enqueue_refresh(
            conn,
            release_run_id=movie.release_run_id,
            movie_id=movie.movie_id,
            reason="upcoming_opening_estimate",
            priority=10,
        )
        queued += 1
    return queued


def enqueue_polymarket_metadata_refreshes(conn: Any) -> int:
    """Queue forecast refreshes for movies with newly synced Polymarket buckets."""
    relation = conn.execute("SELECT to_regclass(%s)", ("prediction_market_backtest.movie_matches",)).fetchone()
    if not relation or relation[0] is None:
        return 0
    openings = conn.execute("SELECT to_regclass(%s)", ("analytics.eda_movie_openings",)).fetchone()
    if not openings or openings[0] is None:
        return 0
    from models.boxoffice.refresh_queue import enqueue_refresh

    rows = conn.execute(
        """
        SELECT DISTINCT opening.release_run_id, opening.movie_id
        FROM prediction_market_backtest.movie_matches match
        JOIN prediction_market_backtest.event_bucket_validations validation
          ON validation.event_id = match.event_id
        JOIN analytics.eda_movie_openings opening
          ON opening.movie_id = match.movie_id
        WHERE match.movie_id IS NOT NULL
          AND LOWER(COALESCE(validation.validation_status, '')) = 'valid'
          AND COALESCE(jsonb_array_length(validation.validation_errors), 0) = 0
        """
    ).fetchall()
    queued = 0
    for release_run_id, movie_id in rows:
        enqueue_refresh(
            conn,
            release_run_id=int(release_run_id),
            movie_id=int(movie_id),
            reason="polymarket_metadata_synced",
            priority=10,
        )
        queued += 1
    return queued


def ensure_forecast_refresh_worker_started() -> int | None:
    """Start the forecast refresh worker so queued forecast jobs are consumed."""
    from pm_box_office.web.services import forecast_workers

    return forecast_workers.ensure_worker_started()


def activate_today_opening_amc_campaign(conn: Any) -> int:
    """Start the AMC seat workflow automatically for today's estimate-backed openings."""
    from models.boxoffice.future_candidates import build_future_candidate_artifacts
    from pm_box_office.sources.amc import db as amc_db
    from pm_box_office.sources.amc.services import movie_service

    now = dt.datetime.now(dt.timezone.utc)
    try:
        candidates = build_future_candidate_artifacts(conn, as_of_utc=now)
    except Exception:
        return 0
    if "movie_id" not in candidates.daily_baseline.columns:
        return 0
    estimate_backed = {int(movie_id) for movie_id in candidates.daily_baseline["movie_id"].tolist() if movie_id is not None}
    if not any(movie.opening_weekend_start == now.date() and movie.movie_id in estimate_backed for movie in candidates.movies):
        return 0
    campaign_id = amc_db.ensure_campaign(conn, now.date())
    status = conn.execute("SELECT status FROM collection_campaigns WHERE campaign_id = %s", (campaign_id,)).fetchone()
    if status and str(status[0]) == "active":
        return 0
    movie_service.create_seat_collection_run(conn, exhibition_date=now.date())
    return 1


def supervise_run(run_id: str, *, database_url: str | None = None) -> int:
    conn = connect_database(database_url)
    stop_heartbeat = threading.Event()
    heartbeat_thread: threading.Thread | None = None
    process: subprocess.Popen[str] | None = None
    try:
        repository.initialize_orchestration_database(conn)
        command, args = repository.load_run_command(conn, run_id)
        source_row = conn.execute("SELECT source_key FROM ingest_runs WHERE run_id = %s", (str(run_id),)).fetchone()
        source_key = str(source_row[0]) if source_row else command
        child_env = os.environ.copy()
        resolved_url = database_url or database_url_from_env()
        if resolved_url:
            child_env["DATABASE_URL"] = resolved_url
        child_env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            [sys.executable, "-m", command, *args],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=child_env,
            start_new_session=True,
        )
        repository.mark_run_running(conn, run_id=run_id, pid=process.pid)
        repository.append_log(conn, run_id=run_id, stream="system", line=f"Started {command} pid={process.pid}")
        conn.commit()
        heartbeat_thread = start_heartbeat(run_id, stop_heartbeat, database_url=database_url)

        assert process.stdout is not None
        for line in process.stdout:
            repository.append_log(conn, run_id=run_id, stream="stdout", line=line)
            conn.commit()
        exit_code = process.wait()
        status = "succeeded" if exit_code == 0 else "failed"
        repository.complete_run(
            conn,
            run_id=run_id,
            status=status,
            exit_code=exit_code,
            error_summary=None if exit_code == 0 else f"Process exited with code {exit_code}",
        )
        if exit_code == 0:
            queued_forecasts = 0
            rule = refresh_rule_for(source_key)
            reconciliation = {"repointed_predictions": 0, "affected_movie_ids": []}
            if rule.reconcile_identities:
                reconciliation = movie_identity.reconcile_prediction_movie_identities(conn)
            if reconciliation["repointed_predictions"]:
                repository.append_log(conn, run_id=run_id, stream="system", line=f"Reconciled prediction identities: {reconciliation}")
                queued = enqueue_reconciled_forecast_refreshes(conn, reconciliation["affected_movie_ids"])
                repository.append_log(conn, run_id=run_id, stream="system", line=f"Queued {queued} canonical forecast refreshes")
                queued_forecasts += queued
            queued_upcoming = enqueue_upcoming_candidate_refreshes(conn) if rule.refresh_estimate_candidates else 0
            if queued_upcoming:
                repository.append_log(conn, run_id=run_id, stream="system", line=f"Queued {queued_upcoming} upcoming estimate-backed forecast refreshes")
                queued_forecasts += queued_upcoming
            if rule.refresh_polymarket_matches:
                queued_polymarket = enqueue_polymarket_metadata_refreshes(conn)
                repository.append_log(conn, run_id=run_id, stream="system", line=f"Queued {queued_polymarket} Polymarket metadata forecast refreshes")
                queued_forecasts += queued_polymarket
            try:
                from models.boxoffice.input_snapshots import record_ingest_event

                record_ingest_event(
                    conn,
                    source_key=source_key,
                    source_run_id=str(run_id),
                    payload={
                        "command": command,
                        "queued_forecasts": queued_forecasts,
                        "reconciled_predictions": int(reconciliation["repointed_predictions"]),
                    },
                )
            except Exception as exc:  # Provenance must never turn a successful collector into a failed run.
                repository.append_log(conn, run_id=run_id, stream="system", line=f"Pipeline event recording skipped: {exc}")
            if queued_forecasts:
                try:
                    worker_pid = ensure_forecast_refresh_worker_started()
                    repository.append_log(conn, run_id=run_id, stream="system", line=f"Forecast refresh worker running pid={worker_pid}")
                except Exception as exc:  # noqa: BLE001 - keep source ingest success independent of worker startup.
                    repository.append_log(conn, run_id=run_id, stream="system", line=f"Forecast refresh worker was not started automatically: {exc}")
            activated_campaigns = activate_today_opening_amc_campaign(conn) if rule.activate_today_amc_campaign else 0
            if activated_campaigns:
                repository.append_log(conn, run_id=run_id, stream="system", line="Activated AMC campaign for today's estimate-backed opening releases")
            repository.refresh_all_source_freshness(conn)
        conn.commit()
        return int(exit_code)
    except Exception as exc:  # noqa: BLE001 - persist supervisor failures.
        try:
            repository.append_log(conn, run_id=run_id, stream="system", line=traceback.format_exc())
            repository.complete_run(conn, run_id=run_id, status="failed", exit_code=None, error_summary=str(exc))
            conn.commit()
        except Exception:
            conn.rollback()
        return 1
    finally:
        stop_heartbeat.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=2)
        conn.close()


def start_heartbeat(run_id: str, stop_event: threading.Event, *, database_url: str | None) -> threading.Thread:
    thread = threading.Thread(
        target=heartbeat_loop,
        args=(run_id, stop_event, database_url),
        name=f"ingest-heartbeat-{run_id}",
        daemon=True,
    )
    thread.start()
    return thread


def heartbeat_loop(run_id: str, stop_event: threading.Event, database_url: str | None) -> None:
    conn = connect_database(database_url)
    try:
        while not stop_event.wait(5):
            repository.heartbeat_run(conn, run_id)
            conn.commit()
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--database-url")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return supervise_run(args.run_id, database_url=args.database_url)


if __name__ == "__main__":
    raise SystemExit(main())
