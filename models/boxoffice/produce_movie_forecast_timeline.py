#!/usr/bin/env python3
"""Generate canonical forecast rows for movie opening-weekend timelines."""

from __future__ import annotations

import argparse
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pm_box_office.db.connection import connect_database

from .artifacts import DEFAULT_ARTIFACT_ROOT, load_model_artifacts
from .constants import PRE_RELEASE_REGIME
from .market_buckets import generate_rounding_variants
from .future_candidates import FutureCandidateArtifacts, build_future_candidate_artifacts
from .live_composition import compose_live_weekend_forecast
from .origins import build_forecast_origins
from .persistence import (
    ensure_forecast_tables,
    remove_obsolete_future_forecasts,
    rows_from_results,
    write_forecast_rows,
    write_run,
)
from .pre_release import forecast_pre_release_opening_weekend
from .reported_actuals import merge_reported_actuals
from .schema import MovieOpening


def parse_as_of(raw: str | None) -> datetime:
    if not raw or raw == "now":
        return datetime.now(timezone.utc)
    value = pd.Timestamp(raw)
    if value.tzinfo is None:
        value = value.tz_localize("UTC")
    return value.tz_convert("UTC").to_pydatetime()


def fetch_frame(conn: Any, sql: str, params: tuple[Any, ...]) -> pd.DataFrame:
    cursor = conn.execute(sql, params)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def resolve_movies(
    conn: Any,
    *,
    movie_id: int | None,
    release_run_id: int | None,
    release_start: str | None,
    release_end: str | None,
    future_candidates: FutureCandidateArtifacts | None = None,
) -> list[MovieOpening]:
    if release_run_id is not None and release_run_id < 0:
        movies = []
    else:
        predicates = ["opening_weekend_start IS NOT NULL"]
        params: list[Any] = []
        if movie_id is not None:
            predicates.append("movie_id = %s")
            params.append(movie_id)
        if release_run_id is not None:
            predicates.append("release_run_id = %s")
            params.append(release_run_id)
        if release_start is not None:
            predicates.append("opening_weekend_start >= %s")
            params.append(release_start)
        if release_end is not None:
            predicates.append("opening_weekend_start <= %s")
            params.append(release_end)
        sql = f"""
            SELECT release_run_id, movie_id, title, opening_weekend_start
            FROM analytics.eda_movie_openings
            WHERE {' AND '.join(predicates)}
            ORDER BY opening_weekend_start, movie_id
        """
        frame = fetch_frame(conn, sql, tuple(params))
        movies = [
            MovieOpening(
                movie_id=int(row.movie_id),
                release_run_id=int(row.release_run_id),
                title=str(row.title),
                opening_weekend_start=pd.Timestamp(row.opening_weekend_start).date(),
            )
            for row in frame.itertuples(index=False)
        ]

    if future_candidates:
        seen = {movie.release_run_id for movie in movies}
        for movie in future_candidates.movies:
            if movie.release_run_id not in seen:
                movies.append(movie)
                seen.add(movie.release_run_id)
    return sorted(movies, key=lambda movie: (movie.opening_weekend_start, movie.movie_id, movie.release_run_id))


def _date_arg(value: str | None) -> Any:
    return pd.Timestamp(value).date() if value else None


def augment_artifacts_with_future_candidates(
    artifacts: Any,
    candidates: FutureCandidateArtifacts,
) -> Any:
    if not candidates.movies:
        return artifacts
    pre_release_panel = pd.concat(
        [artifacts.pre_release_panel, candidates.pre_release_panel],
        ignore_index=True,
        sort=False,
    ) if not candidates.pre_release_panel.empty else artifacts.pre_release_panel
    daily_baseline = pd.concat(
        [artifacts.daily_baseline, candidates.daily_baseline],
        ignore_index=True,
        sort=False,
    ) if not candidates.daily_baseline.empty else artifacts.daily_baseline
    return replace(artifacts, pre_release_panel=pre_release_panel, daily_baseline=daily_baseline)


def build_results_for_movie(
    *,
    movie: MovieOpening,
    mode: str,
    as_of_utc: datetime,
    artifacts: Any,
    run_id: str,
    continue_on_missing: bool,
    skip_log_limit: int,
    skip_counter: list[int],
) -> list[Any]:
    results = []
    market_grids: dict[str, tuple[int, int, int, int]] = {}
    for origin in build_forecast_origins(movie, mode=mode, as_of_utc=as_of_utc):
        try:
            if origin.regime == PRE_RELEASE_REGIME:
                pre_results = forecast_pre_release_opening_weekend(
                    movie=movie,
                    origin=origin,
                    artifacts=artifacts,
                    run_id=run_id,
                    is_backtest=mode == "historical",
                )
                results.extend(pre_results)
                if origin.origin_key == "P_-10":
                    anchor = next((result for result in pre_results if result.target == "opening_weekend"), None)
                    if anchor is not None:
                        try:
                            variants = generate_rounding_variants(
                                release_run_id=movie.release_run_id,
                                effective_listing_origin=origin.origin_key,
                                anchor_forecast_usd=anchor.point_usd,
                                anchor_forecast_emission_hash=anchor.forecast_id,
                                point_policy_version=anchor.point_model,
                            )
                            market_grids = {variant.grid_id: variant.boundaries_usd for variant in variants}
                        except ValueError:
                            market_grids = {}
            else:
                results.extend(
                    compose_live_weekend_forecast(
                        movie=movie,
                        origin=origin,
                        artifacts=artifacts,
                        run_id=run_id,
                        is_live=mode == "live",
                        is_backtest=mode == "historical",
                        market_grids=market_grids or None,
                    )
                )
        except (KeyError, ValueError) as exc:
            if not continue_on_missing:
                raise
            if skip_counter[0] < skip_log_limit:
                print(f"Skipped {movie.release_run_id} {origin.origin_key}: {exc}")
            skip_counter[0] += 1
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url")
    parser.add_argument("--movie-id", type=int)
    parser.add_argument("--release-run-id", type=int)
    parser.add_argument("--release-start")
    parser.add_argument("--release-end")
    parser.add_argument("--mode", choices=["historical", "live"], default="live")
    parser.add_argument("--as-of", default="now")
    parser.add_argument("--model-version", default="latest")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--write-db", action="store_true")
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--components-output-csv", type=Path)
    parser.add_argument("--continue-on-missing", action="store_true")
    parser.add_argument("--skip-log-limit", type=int, default=200)
    parser.add_argument(
        "--include-future-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include unreleased estimate/AMC candidates that do not yet have release_runs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    as_of_utc = parse_as_of(args.as_of)
    run_id = args.run_id or f"forecast_{uuid.uuid4().hex[:12]}"
    artifacts = load_model_artifacts(args.model_version, artifact_root=args.artifact_root)
    conn = connect_database(args.database_url)
    try:
        future_candidates = (
            build_future_candidate_artifacts(
                conn,
                as_of_utc=as_of_utc,
                release_start=_date_arg(args.release_start),
                release_end=_date_arg(args.release_end),
                movie_id=args.movie_id,
                release_run_id=args.release_run_id,
            )
            if args.mode == "live" and args.include_future_candidates
            else FutureCandidateArtifacts([], pd.DataFrame(), pd.DataFrame())
        )
        artifacts = augment_artifacts_with_future_candidates(artifacts, future_candidates)
        movies = resolve_movies(
            conn,
            movie_id=args.movie_id,
            release_run_id=args.release_run_id,
            release_start=args.release_start,
            release_end=args.release_end,
            future_candidates=future_candidates,
        )
        if args.mode == "live":
            artifacts = merge_reported_actuals(
                conn,
                artifacts=artifacts,
                movies=movies,
                as_of_utc=as_of_utc,
            )
        all_results = []
        skip_counter = [0]
        for movie in movies:
            all_results.extend(
                build_results_for_movie(
                    movie=movie,
                    mode=args.mode,
                    as_of_utc=as_of_utc,
                    artifacts=artifacts,
                    run_id=run_id,
                    continue_on_missing=args.continue_on_missing,
                    skip_log_limit=max(0, args.skip_log_limit),
                    skip_counter=skip_counter,
                )
            )
        if skip_counter[0] > max(0, args.skip_log_limit):
            print(f"Skipped {skip_counter[0]:,} missing movie/origin inputs; shown first {max(0, args.skip_log_limit):,}")
        forecast_rows, component_rows = rows_from_results(all_results)
        if args.output_csv:
            args.output_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(forecast_rows).to_csv(args.output_csv, index=False)
        if args.components_output_csv:
            args.components_output_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(component_rows).to_csv(args.components_output_csv, index=False)
        if args.write_db:
            ensure_forecast_tables(conn)
            write_run(
                conn,
                run_id=run_id,
                model_version=artifacts.model_version,
                manifest_path=str(artifacts.artifact_dir / "manifest.json"),
                mode=args.mode,
                as_of_utc=as_of_utc,
            )
            # A historical immutable emission can legitimately differ after a
            # source correction. Preserve that emission and still refresh the
            # other eligible origins/releases in this full live backfill.
            write_forecast_rows(
                conn,
                forecast_rows,
                component_rows,
                skip_conflicting_emissions=args.mode == "live",
            )
            # Only an unfiltered live generation owns the entire future window.
            # Narrow refreshes must not remove unrelated candidates.
            if (
                args.mode == "live"
                and args.include_future_candidates
                and args.movie_id is None
                and args.release_run_id is None
                and args.release_start is None
                and args.release_end is None
            ):
                removed = remove_obsolete_future_forecasts(
                    conn,
                    model_version=artifacts.model_version,
                    opening_weekend_start=as_of_utc.date() - timedelta(days=14),
                    opening_weekend_end=as_of_utc.date() + timedelta(days=370),
                    active_release_run_ids=[movie.release_run_id for movie in future_candidates.movies],
                )
                if removed:
                    print(f"Removed {removed:,} obsolete future-candidate forecast rows")
            conn.commit()
        print(f"Generated {len(forecast_rows):,} forecast rows and {len(component_rows):,} component rows")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
