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


def add_database_arg(
    parser: argparse.ArgumentParser,
    *,
    default: str | None = None,
    suppress_default: bool = False,
    help_text: str = "PostgreSQL connection URL. Defaults to DATABASE_URL, POSTGRES_DSN, or .env.",
) -> None:
    kwargs: dict[str, object] = {"help": help_text}
    if suppress_default:
        kwargs["default"] = argparse.SUPPRESS
    else:
        kwargs["default"] = database_url_from_env() if default is None else default
    parser.add_argument("--database-url", **kwargs)


def add_cache_args(
    parser: argparse.ArgumentParser,
    *,
    default_cache_dir: Path,
    cache_help: str = "Raw response cache directory.",
    include_dry_run: bool = True,
    suppress_default: bool = False,
) -> None:
    default_kwargs: dict[str, object] = {"default": argparse.SUPPRESS} if suppress_default else {}
    cache_dir_kwargs: dict[str, object] = (
        {"default": argparse.SUPPRESS} if suppress_default else {"default": default_cache_dir}
    )
    parser.add_argument("--cache-dir", type=Path, help=cache_help, **cache_dir_kwargs)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refetch even when cache exists.",
        **default_kwargs,
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require all responses to exist in cache.",
        **default_kwargs,
    )
    if include_dry_run:
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Report planned work without committing writes.",
            **default_kwargs,
        )
