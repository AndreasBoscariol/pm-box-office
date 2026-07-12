"""Launch and cancel local ingest runs."""

from __future__ import annotations

import datetime as dt
import os
import signal
import subprocess
import sys
import uuid

from pm_box_office.config import REPO_ROOT
from pm_box_office.db.connection import connect_database, database_url_from_env
from pm_box_office.orchestration import repository
from pm_box_office.orchestration.registry import BOX_OFFICE_PREDICTION_SOURCE_KEYS, RUN_ALL_SOURCE_KEYS


ESTIMATE_RUN_ALL_LOOKBACK_DAYS = 7
THE_NUMBERS_AUTORUN_LOOKBACK_DAYS = 7
THE_NUMBERS_AUTORUN_PUBLISH_LAG_DAYS = 1
THE_NUMBERS_AUTORUN_REFRESH_DAYS = 2
THE_NUMBERS_SOURCE_POLL_LOOKBACK_DAYS = 2


def start_source_run(
    source_key: str,
    *,
    trigger: str = "manual",
    database_url: str | None = None,
    extra_args: list[str] | None = None,
) -> uuid.UUID:
    conn = connect_database(database_url)
    try:
        repository.initialize_orchestration_database(conn)
        repository.seed_sources(conn)
        repository.fail_stale_queued_runs(conn, source_key=source_key)
        resolved_extra_args = extra_args if extra_args is not None else autorun_extra_args(source_key, trigger=trigger)
        run_id = repository.create_run(conn, source_key=source_key, trigger=trigger, extra_args=resolved_extra_args)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    env = os.environ.copy()
    resolved_url = database_url or database_url_from_env()
    if resolved_url:
        env["DATABASE_URL"] = resolved_url
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "pm_box_office.orchestration.supervisor", "--run-id", str(run_id)],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
    except Exception as exc:
        conn = connect_database(database_url)
        try:
            repository.complete_run(
                conn,
                run_id=run_id,
                status="failed",
                exit_code=None,
                error_summary=f"Could not start supervisor: {exc}",
            )
            conn.commit()
        finally:
            conn.close()
        raise

    conn = connect_database(database_url)
    try:
        repository.mark_run_spawned(conn, run_id=run_id, pid=process.pid)
        conn.commit()
    finally:
        conn.close()
    return run_id


def start_run_all(
    *,
    trigger: str = "manual_run_all",
    database_url: str | None = None,
    source_keys: tuple[str, ...] = RUN_ALL_SOURCE_KEYS,
) -> dict[str, list[str]]:
    started: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []
    for source_key in source_keys:
        try:
            run_id = start_source_run(
                source_key,
                trigger=trigger,
                database_url=database_url,
                extra_args=autorun_extra_args(source_key, trigger=trigger),
            )
        except (
            repository.SourceAlreadyRunningError,
            repository.SourceDependencyError,
            repository.SourceDisabledError,
        ) as exc:
            skipped.append(f"{source_key}: {exc}")
        except repository.OrchestrationError as exc:
            errors.append(f"{source_key}: {exc}")
        else:
            started.append(f"{source_key}: {run_id}")
    if started:
        conn = connect_database(database_url)
        try:
            repository.record_autorun_trigger(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    return {"started": started, "skipped": skipped, "errors": errors}


def autorun_extra_args(source_key: str, *, trigger: str, today: dt.date | None = None) -> list[str]:
    if source_key in BOX_OFFICE_PREDICTION_SOURCE_KEYS:
        return rolling_week_args(today=today)
    if source_key == "the_numbers_predictions" and trigger == "auto_source_poll":
        return ["--refresh"]
    if trigger not in {"auto_daily", "auto_source_poll"} or source_key != "the_numbers":
        return []
    run_date = today or dt.date.today()
    end_date = run_date - dt.timedelta(days=THE_NUMBERS_AUTORUN_PUBLISH_LAG_DAYS)
    lookback_days = (
        THE_NUMBERS_SOURCE_POLL_LOOKBACK_DAYS
        if trigger == "auto_source_poll"
        else THE_NUMBERS_AUTORUN_LOOKBACK_DAYS
    )
    start_date = end_date - dt.timedelta(days=lookback_days - 1)
    return [
        "--start-date",
        start_date.isoformat(),
        "--end-date",
        end_date.isoformat(),
        "--refresh-recent-days",
        str(min(THE_NUMBERS_AUTORUN_REFRESH_DAYS, lookback_days)),
    ]


def rolling_week_args(*, today: dt.date | None = None) -> list[str]:
    run_date = today or dt.date.today()
    start_date = run_date - dt.timedelta(days=ESTIMATE_RUN_ALL_LOOKBACK_DAYS - 1)
    return ["--start-date", start_date.isoformat(), "--end-date", run_date.isoformat(), "--refresh"]


def cancel_run(run_id: str | uuid.UUID, *, database_url: str | None = None) -> None:
    conn = connect_database(database_url)
    try:
        repository.initialize_orchestration_database(conn)
        pid = repository.request_cancel(conn, run_id)
        if pid is not None:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError:
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass
        repository.append_log(conn, run_id=run_id, stream="system", line="Cancellation requested.")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
