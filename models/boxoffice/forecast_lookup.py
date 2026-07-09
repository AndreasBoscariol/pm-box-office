#!/usr/bin/env python3
"""Lookup stored forecast timelines without recomputing forecasts."""

from __future__ import annotations

import argparse
from pathlib import Path

from pm_box_office.db.connection import connect_database

from .lookup import get_movie_forecast_timeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url")
    parser.add_argument("--movie-id", type=int)
    parser.add_argument("--release-run-id", type=int)
    parser.add_argument("--model-version")
    parser.add_argument("--output-csv", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    conn = connect_database(args.database_url)
    try:
        frame = get_movie_forecast_timeline(
            conn,
            movie_id=args.movie_id,
            release_run_id=args.release_run_id,
            model_version=args.model_version,
        )
        if args.output_csv:
            args.output_csv.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(args.output_csv, index=False)
        else:
            print(frame.to_string(index=False))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

