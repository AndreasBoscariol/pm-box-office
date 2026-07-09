#!/usr/bin/env python3
"""Generate canonical forecast rows for movie opening-weekend timelines."""

from __future__ import annotations

import argparse
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pm_box_office.db.connection import connect_database

from .artifacts import DEFAULT_ARTIFACT_ROOT, load_model_artifacts
from .constants import PRE_RELEASE_REGIME
from .live_composition import compose_live_weekend_forecast
from .origins import build_forecast_origins
from .persistence import ensure_forecast_tables, rows_from_results, write_forecast_rows, write_run
from .pre_release import forecast_pre_release_opening_weekend
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
) -> list[MovieOpening]:
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
    return [
        MovieOpening(
            movie_id=int(row.movie_id),
            release_run_id=int(row.release_run_id),
            title=str(row.title),
            opening_weekend_start=pd.Timestamp(row.opening_weekend_start).date(),
        )
        for row in frame.itertuples(index=False)
    ]


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
    for origin in build_forecast_origins(movie, mode=mode, as_of_utc=as_of_utc):
        try:
            if origin.regime == PRE_RELEASE_REGIME:
                results.extend(
                    forecast_pre_release_opening_weekend(
                        movie=movie,
                        origin=origin,
                        artifacts=artifacts,
                        run_id=run_id,
                        is_backtest=mode == "historical",
                    )
                )
            else:
                results.extend(
                    compose_live_weekend_forecast(
                        movie=movie,
                        origin=origin,
                        artifacts=artifacts,
                        run_id=run_id,
                        is_live=mode == "live",
                        is_backtest=mode == "historical",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    as_of_utc = parse_as_of(args.as_of)
    run_id = args.run_id or f"forecast_{uuid.uuid4().hex[:12]}"
    artifacts = load_model_artifacts(args.model_version, artifact_root=args.artifact_root)
    conn = connect_database(args.database_url)
    try:
        movies = resolve_movies(
            conn,
            movie_id=args.movie_id,
            release_run_id=args.release_run_id,
            release_start=args.release_start,
            release_end=args.release_end,
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
                manifest_path=str(artifacts.artifact_dir / "manifest.yml"),
                mode=args.mode,
                as_of_utc=as_of_utc,
            )
            write_forecast_rows(conn, forecast_rows, component_rows)
            conn.commit()
        print(f"Generated {len(forecast_rows):,} forecast rows and {len(component_rows):,} component rows")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
