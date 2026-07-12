#!/usr/bin/env python3
"""Run debounced live forecast refresh jobs."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import socket
import time
import uuid
from dataclasses import dataclass
from dataclasses import replace
from datetime import time as clock_time
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd

from pm_box_office.db.connection import connect_database

from .artifacts import DEFAULT_ARTIFACT_ROOT, load_model_artifacts
from .amc_shadow import build_shadow_rows, build_thursday_preview_shadow_rows, write_shadow_rows
from .future_candidates import build_future_candidate_artifacts
from .live_amc_plugin import build_live_amc_plugin_nowcasts
from .input_snapshots import capture_input_snapshot
from .persistence import ensure_forecast_tables, rows_from_results, write_control_emissions, write_forecast_rows, write_run
from .thursday_amc_preview import build_thursday_amc_preview_nowcasts
from .produce_movie_forecast_timeline import (
    augment_artifacts_with_future_candidates,
    build_results_for_movie,
    resolve_movies,
)
from .origins import build_forecast_origins
from .pre_release import forecast_pre_release_opening_weekend
from .reported_actuals import merge_reported_actuals
from .schema import MovieOpening
from .refresh_queue import (
    ForecastRefresh,
    claim_due_refreshes,
    ensure_refresh_queue,
    mark_refresh_failed,
    mark_refresh_succeeded,
    reset_stale_running_refreshes,
)


LOGGER = logging.getLogger("boxoffice.forecast_refresh")


@dataclass(frozen=True)
class ForecastRefreshResult:
    run_id: str | None
    resolved_model_version: str | None
    row_count: int
    component_row_count: int


RefreshRunner = Callable[[Any, ForecastRefresh, Path], ForecastRefreshResult]


def p_minus_one_result_if_due(
    conn: Any,
    *,
    movie: MovieOpening,
    artifacts: Any,
    run_id: str,
    as_of_utc: dt.datetime,
) -> list[Any]:
    """Backfill the leakage-safe P_-1 row when a refresh first arrives on opening day.

    A release-day refresh normally only builds live origins.  Reconstructing the
    last pre-release origin from the prior local day prevents a late ingest or
    identity reconciliation from leaving a movie absent from the lookup.
    """
    eastern = ZoneInfo("America/New_York")
    if movie.opening_weekend_start > as_of_utc.astimezone(eastern).date():
        return []
    prior_as_of = dt.datetime.combine(
        movie.opening_weekend_start - dt.timedelta(days=1),
        clock_time(23, 59),
        tzinfo=eastern,
    ).astimezone(dt.timezone.utc)
    candidates = build_future_candidate_artifacts(
        conn,
        as_of_utc=prior_as_of,
        release_start=movie.opening_weekend_start,
        release_end=movie.opening_weekend_start,
        movie_id=movie.movie_id,
        release_run_id=movie.release_run_id,
    )
    if not candidates.movies:
        return []
    candidate_movie = next(
        (item for item in candidates.movies if item.release_run_id == movie.release_run_id),
        None,
    )
    if candidate_movie is None:
        return []
    pre_release_artifacts = augment_artifacts_with_future_candidates(artifacts, candidates)
    origin = next(
        (
            item
            for item in build_forecast_origins(
                candidate_movie,
                mode="live",
                as_of_utc=prior_as_of,
                include_live=False,
            )
            if item.origin_key == "P_-1"
        ),
        None,
    )
    if origin is None:
        return []
    try:
        return forecast_pre_release_opening_weekend(
            movie=candidate_movie,
            origin=origin,
            artifacts=pre_release_artifacts,
            run_id=run_id,
            is_backtest=False,
        )
    except (KeyError, ValueError):
        return []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url")
    parser.add_argument("--worker-id", default=f"{socket.gethostname()}-{time.time_ns()}")
    parser.add_argument("--once", action="store_true", help="Claim and process one batch, then exit.")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--idle-seconds", type=float, default=2.0)
    parser.add_argument("--stale-running-minutes", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=int, default=60)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--verbose", action="store_true")
    return parser


def run_forecast_refresh(
    conn: Any,
    refresh: ForecastRefresh,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> ForecastRefreshResult:
    as_of_utc = dt.datetime.now(dt.timezone.utc)
    run_id = f"forecast_refresh_{uuid.uuid4().hex[:12]}"
    artifacts = load_model_artifacts(refresh.model_version, artifact_root=artifact_root)
    future_candidates = build_future_candidate_artifacts(
        conn,
        as_of_utc=as_of_utc,
        movie_id=refresh.movie_id,
        release_run_id=refresh.release_run_id,
    )
    artifacts = augment_artifacts_with_future_candidates(artifacts, future_candidates)
    movies = resolve_movies(
        conn,
        movie_id=refresh.movie_id,
        release_run_id=refresh.release_run_id,
        release_start=None,
        release_end=None,
        future_candidates=future_candidates,
    )
    if not movies:
        return ForecastRefreshResult(
            run_id=None,
            resolved_model_version=artifacts.model_version,
            row_count=0,
            component_row_count=0,
        )

    # Record the source state before model inference. This is intentionally
    # append-only provenance: collectors retain raw observations while the
    # snapshot tells operators which estimate, actual, AMC, and Polymarket
    # inputs were available to this refresh.
    for movie in movies:
        capture_input_snapshot(
            conn,
            release_run_id=movie.release_run_id,
            movie_id=movie.movie_id,
            model_version=artifacts.model_version,
            as_of_utc=as_of_utc,
        )

    artifacts = merge_reported_actuals(
        conn,
        artifacts=artifacts,
        movies=movies,
        as_of_utc=as_of_utc,
    )

    live_plugin = build_live_amc_plugin_nowcasts(conn, movies=movies, as_of_utc=as_of_utc)
    thursday_preview_nowcasts = build_thursday_amc_preview_nowcasts(
        conn,
        movies=movies,
        as_of_utc=as_of_utc,
        artifacts=artifacts,
    )
    if not live_plugin.empty:
        artifacts = replace(
            artifacts,
            live_plugin_nowcasts=pd.concat(
                [artifacts.live_plugin_nowcasts, live_plugin],
                ignore_index=True,
                sort=False,
            ).drop_duplicates(["movie_id", "regime", "forecast_origin", "target_day"], keep="last"),
        )

    results = []
    control_results = []
    shadow_rows = []
    skip_counter = [0]
    for movie in movies:
        results.extend(
            p_minus_one_result_if_due(
                conn,
                movie=movie,
                artifacts=artifacts,
                run_id=run_id,
                as_of_utc=as_of_utc,
            )
        )
        movie_results = build_results_for_movie(
            movie=movie,
            mode="live",
            as_of_utc=as_of_utc,
            artifacts=artifacts,
            run_id=run_id,
            continue_on_missing=True,
            skip_log_limit=0,
            skip_counter=skip_counter,
        )
        results.extend(movie_results)
        # Persist a matched no-AMC OW control whenever the production result
        # actually used an eligible AMC component.  Both paths share the same
        # as-of actuals, baseline artifacts, origin, and listing grid.
        amc_origins = {
            item.origin.origin_key
            for item in movie_results
            if item.target == "opening_weekend" and any(component.component_type == "AMC_nowcast" for component in item.components)
        }
        if amc_origins:
            no_amc_artifacts = replace(artifacts, live_plugin_nowcasts=pd.DataFrame())
            no_amc_results = build_results_for_movie(
                movie=movie, mode="live", as_of_utc=as_of_utc, artifacts=no_amc_artifacts,
                run_id=run_id, continue_on_missing=True, skip_log_limit=0, skip_counter=skip_counter,
            )
            for item in no_amc_results:
                if item.target == "opening_weekend" and item.origin.origin_key in amc_origins:
                    control_results.append(item)
        shadow_rows.extend(
            build_thursday_preview_shadow_rows(
                movie=movie,
                thursday_preview_nowcasts=thursday_preview_nowcasts,
                artifacts=artifacts,
                run_id=run_id,
            )
        )
        shadow_rows.extend(
            build_shadow_rows(
                movie=movie,
                results=movie_results,
                live_plugin=live_plugin,
                artifacts=artifacts,
                thursday_preview_nowcasts=thursday_preview_nowcasts,
            )
        )
    forecast_rows, component_rows = rows_from_results(results)
    ensure_forecast_tables(conn)
    write_run(
        conn,
        run_id=run_id,
        model_version=artifacts.model_version,
        manifest_path=str(artifacts.artifact_dir / "manifest.json"),
        mode="live_refresh",
        as_of_utc=as_of_utc,
    )
    write_forecast_rows(conn, forecast_rows, component_rows, skip_conflicting_emissions=True)
    control_rows, control_components = rows_from_results(control_results)
    try:
        write_control_emissions(conn, control_rows, control_components)
    except RuntimeError as exc:
        # Forward-validation controls are valuable, but they must never block
        # the primary production forecast after an immutable historical control
        # emission has already been recorded.
        if "forecast emission idempotency conflict" not in str(exc):
            raise
        LOGGER.warning("skipped immutable AMC control-emission conflict: %s", exc)
    write_shadow_rows(conn, shadow_rows)
    return ForecastRefreshResult(
        run_id=run_id,
        resolved_model_version=artifacts.model_version,
        row_count=len(forecast_rows),
        component_row_count=len(component_rows),
    )


def process_refresh(
    conn: Any,
    refresh: ForecastRefresh,
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    max_attempts: int = 3,
    retry_delay_seconds: int = 60,
    runner: RefreshRunner = run_forecast_refresh,
) -> ForecastRefreshResult:
    try:
        result = runner(conn, refresh, artifact_root)
        mark_refresh_succeeded(
            conn,
            refresh_id=refresh.refresh_id,
            run_id=result.run_id,
            resolved_model_version=result.resolved_model_version,
            row_count=result.row_count,
        )
        conn.commit()
        LOGGER.info(
            "forecast refresh succeeded refresh_id=%s release_run_id=%s rows=%s components=%s",
            refresh.refresh_id,
            refresh.release_run_id,
            result.row_count,
            result.component_row_count,
        )
        return result
    except Exception as exc:
        LOGGER.exception(
            "forecast refresh failed refresh_id=%s release_run_id=%s",
            refresh.refresh_id,
            refresh.release_run_id,
        )
        conn.rollback()
        mark_refresh_failed(
            conn,
            refresh_id=refresh.refresh_id,
            exc=exc,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
        )
        conn.commit()
        raise


def run_worker(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    conn = connect_database(args.database_url)
    try:
        ensure_refresh_queue(conn)
        conn.commit()
        LOGGER.info("forecast refresh worker started worker_id=%s limit=%s", args.worker_id, args.limit)
        while True:
            reset_count = reset_stale_running_refreshes(
                conn,
                stale_after=dt.timedelta(minutes=args.stale_running_minutes),
            )
            if reset_count:
                LOGGER.warning("reset %s stale forecast refresh jobs", reset_count)
                conn.commit()
            refreshes = claim_due_refreshes(conn, worker_id=args.worker_id, limit=args.limit)
            conn.commit()
            if not refreshes:
                if args.once:
                    LOGGER.info("no due forecast refresh jobs; exiting because --once was set")
                    return 0
                time.sleep(args.idle_seconds)
                continue
            for refresh in refreshes:
                LOGGER.info(
                    "forecast refresh start refresh_id=%s release_run_id=%s model=%s attempt=%s",
                    refresh.refresh_id,
                    refresh.release_run_id,
                    refresh.model_version,
                    refresh.attempt_count,
                )
                try:
                    process_refresh(
                        conn,
                        refresh,
                        artifact_root=args.artifact_root,
                        max_attempts=args.max_attempts,
                        retry_delay_seconds=args.retry_delay_seconds,
                    )
                except Exception:
                    continue
            if args.once:
                LOGGER.info("processed one forecast refresh batch; exiting because --once was set")
                return 0
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
