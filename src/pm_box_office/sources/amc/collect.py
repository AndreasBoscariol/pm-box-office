#!/usr/bin/env python3
"""AMC campaign and collection queue CLI."""

from __future__ import annotations

import argparse
from typing import Any

from pm_box_office.db.connection import connect_database
from pm_box_office.sources.amc import db
from pm_box_office.sources.amc.client import DEFAULT_CACHE_DIR, DEFAULT_USER_AGENT, HtmlFetcher
from pm_box_office.sources.amc.services import (
    movie_service,
    progress_service,
    sample_service,
    seat_service,
    showtime_service,
    theatre_service,
)
from pm_box_office.sources.common.cli import add_cache_args, add_database_arg, parse_date_arg


def parse_offsets(value: str) -> tuple[int, ...]:
    try:
        offsets = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integer minutes") from exc
    if not offsets:
        raise argparse.ArgumentTypeError("At least one offset minute is required")
    return offsets


def build_fetcher(args: Any) -> HtmlFetcher:
    return HtmlFetcher(
        args.cache_dir,
        refresh=args.refresh,
        offline=args.offline,
        delay_seconds=args.delay_seconds,
        user_agent=args.user_agent,
    )


def add_fetch_args(parser: argparse.ArgumentParser, *, suppress_default: bool = False) -> None:
    add_cache_args(
        parser,
        default_cache_dir=DEFAULT_CACHE_DIR,
        cache_help=f"AMC cache directory. Default: {DEFAULT_CACHE_DIR}",
        include_dry_run=False,
        suppress_default=suppress_default,
    )
    delay_kwargs: dict[str, object] = {"default": argparse.SUPPRESS} if suppress_default else {"default": 1.0}
    user_agent_kwargs: dict[str, object] = (
        {"default": argparse.SUPPRESS} if suppress_default else {"default": DEFAULT_USER_AGENT}
    )
    parser.add_argument("--delay-seconds", type=float, **delay_kwargs)
    parser.add_argument("--user-agent", **user_agent_kwargs)


def add_command_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *args: Any,
    **kwargs: Any,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(*args, **kwargs)
    add_database_arg(parser, suppress_default=True)
    return parser


def connect_cli_database(args: Any) -> Any:
    return connect_database(getattr(args, "database_url", None))


def cmd_init_db(args: Any) -> int:
    conn = connect_cli_database(args)
    try:
        db.initialize_amc_database(conn)
        conn.commit()
    finally:
        conn.close()
    print("Initialized AMC database tables.")
    return 0


def cmd_ingest_theatres(args: Any) -> int:
    fetcher = build_fetcher(args)
    conn = connect_cli_database(args)
    try:
        db.initialize_amc_database(conn)
        count = theatre_service.sync_theatres(conn, fetcher)
        conn.commit()
    finally:
        conn.close()
    print(f"Upserted {count} AMC theatres.")
    return 0


def cmd_create_inventory_run(args: Any) -> int:
    conn = connect_cli_database(args)
    try:
        db.initialize_amc_database(conn)
        run_id, task_count = showtime_service.create_inventory_run(conn, exhibition_date=args.target_date)
        conn.commit()
    finally:
        conn.close()
    print(f"Created showtime inventory run {run_id} with {task_count} theatre tasks.")
    return 0


def cmd_list_movies(args: Any) -> int:
    conn = connect_cli_database(args)
    try:
        db.initialize_amc_database(conn)
        rows = movie_service.list_movies_for_date(conn, exhibition_date=args.target_date)
        conn.rollback()
    finally:
        conn.close()
    if not rows:
        print(f"No AMC movies found for {args.target_date.isoformat()}.")
        return 0
    for row in rows:
        marker = "[x]" if row.selected else "[ ]"
        print(
            f"{marker} {row.amc_movie_name} "
            f"(AMC movie id: {row.amc_movie_id}, theatres: {row.theatre_count}, "
            f"showtimes: {row.showtime_count})"
        )
    return 0


def cmd_toggle_movie(args: Any) -> int:
    conn = connect_cli_database(args)
    try:
        db.initialize_amc_database(conn)
        movie_service.set_movie_selected(
            conn,
            exhibition_date=args.target_date,
            amc_movie_id=args.amc_movie_id,
            selected=args.selected,
        )
        conn.commit()
    finally:
        conn.close()
    state = "selected" if args.selected else "deselected"
    print(f"{state.capitalize()} AMC movie {args.amc_movie_id} for {args.target_date.isoformat()}.")
    return 0


def cmd_select_the_numbers_active(args: Any) -> int:
    conn = connect_cli_database(args)
    try:
        db.initialize_amc_database(conn)
        matches = movie_service.select_the_numbers_active_movies(
            conn,
            exhibition_date=args.target_date,
            lookback_days=args.lookback_days,
        )
        conn.commit()
    finally:
        conn.close()
    print(
        f"Selected {len(matches)} AMC movies for {args.target_date.isoformat()} "
        f"with The Numbers daily chart activity in the last {args.lookback_days} days."
    )
    for match in matches:
        print(
            f"- {match.amc_movie_name} (AMC {match.amc_movie_id}; "
            f"The Numbers: {match.the_numbers_title}; latest {match.latest_box_office_date.isoformat()}; "
            f"{match.recent_days_reported} days)"
        )
    return 0


def cmd_create_seat_run(args: Any) -> int:
    conn = connect_cli_database(args)
    try:
        db.initialize_amc_database(conn)
        run_id, task_count = movie_service.create_seat_collection_run(
            conn,
            exhibition_date=args.target_date,
            target_offsets_minutes=args.target_offsets_minutes,
            sample_key=args.sample_key,
        )
        conn.commit()
    finally:
        conn.close()
    print(f"Created sampled seat collection run {run_id} with {task_count} tasks.")
    return 0


def cmd_create_theatre_sample(args: Any) -> int:
    conn = connect_cli_database(args)
    try:
        db.initialize_amc_database(conn)
        sample_set = sample_service.ensure_default_theatre_sample(
            conn,
            sample_key=args.sample_key,
            sample_size=args.sample_size,
            certainty_count=args.certainty_count,
            seed=args.seed,
        )
        members = sample_service.sample_members(conn, sample_set)
        conn.commit()
    finally:
        conn.close()
    print(
        f"AMC theatre sample {sample_set.sample_key}: "
        f"{len(members)}/{sample_set.frame_theatre_count} theatres, "
        f"{sample_set.certainty_count} certainty, seed={sample_set.seed!r}."
    )
    return 0


def cmd_sample_coverage(args: Any) -> int:
    conn = connect_cli_database(args)
    try:
        db.initialize_amc_database(conn)
        sample_set = sample_service.ensure_default_theatre_sample(conn, sample_key=args.sample_key)
        coverage = sample_service.sample_coverage(conn, sample_set=sample_set, exhibition_date=args.target_date)
        conn.rollback()
    finally:
        conn.close()
    showtime_share = coverage.get("sampled_showtime_share")
    screen_share = coverage.get("sampled_screen_share")
    showtime_share_text = f"{showtime_share:.1%}" if isinstance(showtime_share, float) else "-"
    screen_share_text = f"{screen_share:.1%}" if isinstance(screen_share, float) else "-"
    print(
        f"AMC sample {sample_set.sample_key} for {args.target_date.isoformat()}: "
        f"{coverage.get('active_sample_theatres', 0)}/{coverage.get('active_theatres', 0)} theatres, "
        f"{coverage.get('sampled_showtimes', 0)}/{coverage.get('full_showtimes', 0)} showtimes "
        f"({showtime_share_text} share)"
    )
    print(
        f"screens={coverage.get('sample_screens', 0)}/{coverage.get('active_screens', 0)} "
        f"({screen_share_text} share), "
        f"states={coverage.get('sample_states', 0)}, timezones={coverage.get('sample_timezones', 0)}, "
        f"snapshots={coverage.get('snapshots', 0)}, "
        f"missing_snapshots={coverage.get('missing_snapshots', 0)}, "
        f"late_snapshots={coverage.get('late_snapshots', 0)}"
    )
    return 0


def cmd_run_progress(args: Any) -> int:
    conn = connect_cli_database(args)
    try:
        db.initialize_amc_database(conn)
        progress = progress_service.run_progress(conn, args.run_id)
        conn.rollback()
    finally:
        conn.close()
    print(
        f"{progress['run_type']} {progress['run_id']}: {progress['status']} "
        f"{progress['succeeded']}/{progress['total']} succeeded "
        f"({progress['percent']:.1f}%), queued={progress['queued']}, "
        f"running={progress['running']}, failed={progress['failed']}, "
        f"due_now={progress['due_now']}, late={progress['late']}"
    )
    return 0


def cmd_cancel_run(args: Any) -> int:
    conn = connect_cli_database(args)
    try:
        db.initialize_amc_database(conn)
        count = db.cancel_run(conn, db.as_uuid(args.run_id))
        conn.commit()
    finally:
        conn.close()
    print(f"Cancelled {count} queued/running tasks for run {args.run_id}.")
    return 0


def cmd_cancel_campaign(args: Any) -> int:
    conn = connect_cli_database(args)
    try:
        db.initialize_amc_database(conn)
        campaign_id = db.ensure_campaign(conn, args.target_date)
        count = db.cancel_campaign_runs(conn, campaign_id)
        conn.commit()
    finally:
        conn.close()
    print(f"Cancelled {count} queued/running tasks for {args.target_date.isoformat()}.")
    return 0


def cmd_reset_collection_state(args: Any) -> int:
    if not args.confirm_reset:
        raise SystemExit("Refusing to reset collection state without --confirm-reset.")
    conn = connect_cli_database(args)
    try:
        db.initialize_amc_database(conn)
        counts = db.reset_collection_state(
            conn,
            exhibition_date=args.target_date,
            clear_seat_snapshots=not args.keep_seat_snapshots,
        )
        conn.commit()
    finally:
        conn.close()
    print(
        f"Reset AMC collection state for {args.target_date.isoformat()}: "
        f"tasks={counts['collection_tasks']}, runs={counts['collection_runs']}, "
        f"campaign_movies={counts['campaign_movies']}, campaigns={counts['collection_campaigns']}, "
        f"seat_snapshots={counts['amc_seat_snapshots']}."
    )
    print("Preserved AMC theatres, theatre sample membership, movies, and showtime inventory.")
    return 0


def cmd_cleanup_seat_cache_duplicates(args: Any) -> int:
    result = seat_service.cleanup_top_level_seat_cache_duplicates(
        args.cache_dir,
        dry_run=args.dry_run,
    )
    action = "Would delete" if args.dry_run else "Deleted"
    print(
        f"{action} {result.duplicate_files_deleted if not args.dry_run else result.duplicate_paths_found} "
        f"top-level seat cache duplicate files from {args.cache_dir} "
        f"({result.bytes_deleted:,} bytes; scanned {result.archived_files_scanned} archived seat files)."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_database_arg(parser)
    add_fetch_args(parser)

    subparsers = parser.add_subparsers(dest="command", required=True)
    init_db = add_command_parser(subparsers, "init-db", help="Create AMC database tables.")
    init_db.set_defaults(func=cmd_init_db)

    ingest = add_command_parser(subparsers, "ingest-theatres", help="Ingest AMC theatres from sitemap.")
    add_fetch_args(ingest, suppress_default=True)
    ingest.set_defaults(func=cmd_ingest_theatres)

    inventory = add_command_parser(
        subparsers,
        "create-inventory-run",
        help="Create a durable full-network showtime inventory run for a date.",
    )
    inventory.add_argument("target_date", type=parse_date_arg)
    inventory.set_defaults(func=cmd_create_inventory_run)

    movies = add_command_parser(
        subparsers,
        "list-movies",
        help="List database-backed AMC movie inventory for a date.",
    )
    movies.add_argument("target_date", type=parse_date_arg)
    movies.set_defaults(func=cmd_list_movies)

    toggle = add_command_parser(
        subparsers,
        "toggle-movie",
        help="Select or deselect an AMC movie for a campaign date.",
    )
    toggle.add_argument("target_date", type=parse_date_arg)
    toggle.add_argument("amc_movie_id")
    toggle_group = toggle.add_mutually_exclusive_group(required=True)
    toggle_group.add_argument("--selected", dest="selected", action="store_true")
    toggle_group.add_argument("--deselected", dest="selected", action="store_false")
    toggle.set_defaults(func=cmd_toggle_movie)

    select_active = add_command_parser(
        subparsers,
        "select-the-numbers-active",
        help="Select AMC movies still reporting recent The Numbers daily grosses.",
    )
    select_active.add_argument("target_date", type=parse_date_arg)
    select_active.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        help="Recent The Numbers daily box-office window ending on target date. Default: 7.",
    )
    select_active.set_defaults(func=cmd_select_the_numbers_active)

    seat_run = add_command_parser(
        subparsers,
        "create-seat-run",
        help="Create durable seat-scan tasks for selected movies on a date.",
    )
    seat_run.add_argument("target_date", type=parse_date_arg)
    seat_run.add_argument(
        "--target-offset-minutes",
        dest="target_offsets_minutes",
        type=parse_offsets,
        default=movie_service.DEFAULT_SEAT_TARGET_OFFSETS_MINUTES,
        help="Comma-separated smoothing window minutes before showtime. Default: 20,19,...,1.",
    )
    seat_run.add_argument(
        "--sample-key",
        default=sample_service.DEFAULT_SAMPLE_KEY,
        help=f"Saved theatre sample key. Default: {sample_service.DEFAULT_SAMPLE_KEY}.",
    )
    seat_run.set_defaults(func=cmd_create_seat_run)

    theatre_sample = add_command_parser(
        subparsers,
        "create-theatre-sample",
        help="Create or reuse the persistent fixed AMC theatre sample.",
    )
    theatre_sample.add_argument("--sample-key", default=sample_service.DEFAULT_SAMPLE_KEY)
    theatre_sample.add_argument("--sample-size", type=int, default=sample_service.DEFAULT_SAMPLE_SIZE)
    theatre_sample.add_argument("--certainty-count", type=int, default=sample_service.DEFAULT_CERTAINTY_COUNT)
    theatre_sample.add_argument("--seed", default=sample_service.DEFAULT_SAMPLE_SEED)
    theatre_sample.set_defaults(func=cmd_create_theatre_sample)

    sample_coverage = add_command_parser(
        subparsers,
        "sample-coverage",
        help="Report fixed theatre sample coverage for a date.",
    )
    sample_coverage.add_argument("target_date", type=parse_date_arg)
    sample_coverage.add_argument("--sample-key", default=sample_service.DEFAULT_SAMPLE_KEY)
    sample_coverage.set_defaults(func=cmd_sample_coverage)

    progress = add_command_parser(subparsers, "run-progress", help="Print progress for a durable run.")
    progress.add_argument("run_id")
    progress.set_defaults(func=cmd_run_progress)

    cancel_run = add_command_parser(subparsers, "cancel-run", help="Cancel queued/running tasks for a durable run.")
    cancel_run.add_argument("run_id")
    cancel_run.set_defaults(func=cmd_cancel_run)

    cancel_campaign = add_command_parser(
        subparsers,
        "cancel-campaign",
        help="Cancel queued/running durable tasks for a campaign date.",
    )
    cancel_campaign.add_argument("target_date", type=parse_date_arg)
    cancel_campaign.set_defaults(func=cmd_cancel_campaign)

    reset = add_command_parser(
        subparsers,
        "reset-collection-state",
        help="Clear campaign queues and date-scoped seat snapshots while preserving theatres/sample/showtimes.",
    )
    reset.add_argument("--date", dest="target_date", required=True, type=parse_date_arg)
    reset.add_argument("--confirm-reset", action="store_true", help="Required guard for deleting collection state.")
    reset.add_argument(
        "--keep-seat-snapshots",
        action="store_true",
        help="Only clear campaign queue state; preserve date-scoped seat snapshots.",
    )
    reset.set_defaults(func=cmd_reset_collection_state)

    cleanup_seat_cache = add_command_parser(
        subparsers,
        "cleanup-seat-cache-duplicates",
        help="Delete top-level AMC seat cache files that duplicate archived seat snapshots.",
    )
    add_fetch_args(cleanup_seat_cache, suppress_default=True)
    cleanup_seat_cache.add_argument(
        "--dry-run",
        action="store_true",
        help="Report duplicate top-level seat cache files without deleting them.",
    )
    cleanup_seat_cache.set_defaults(func=cmd_cleanup_seat_cache_duplicates)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
