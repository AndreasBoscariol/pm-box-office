"""Small argparse helpers for source ingests."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from pm_box_office.db.connection import database_url_from_env


def parse_date_arg(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO date: {value}") from exc


def add_database_arg(parser: argparse.ArgumentParser, *, default: str | None = None) -> None:
    parser.add_argument(
        "--database-url",
        default=database_url_from_env() if default is None else default,
        help="PostgreSQL connection URL. Defaults to DATABASE_URL, POSTGRES_DSN, or .env.",
    )


def add_cache_args(
    parser: argparse.ArgumentParser,
    *,
    default_cache_dir: Path,
    cache_help: str = "Raw response cache directory.",
    include_dry_run: bool = True,
) -> None:
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir, help=cache_help)
    parser.add_argument("--refresh", action="store_true", help="Refetch even when cache exists.")
    parser.add_argument("--offline", action="store_true", help="Require all responses to exist in cache.")
    if include_dry_run:
        parser.add_argument("--dry-run", action="store_true", help="Report planned work without committing writes.")

