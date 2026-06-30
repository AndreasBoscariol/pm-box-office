#!/usr/bin/env python3
"""AMC one-film box-office signal collection CLI."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from scripts.db import connect_database
except ModuleNotFoundError:  # Allow `python3 scripts/ingest/amc/collect.py`.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.db import connect_database

from scripts.ingest.amc import db
from scripts.ingest.amc.client import DEFAULT_CACHE_DIR, DEFAULT_USER_AGENT, HtmlFetcher
from scripts.ingest.amc.parsers import (
    extract_rendered_showtimes,
    extract_showtimes,
    fetch_seat_fill,
    maybe_parse_apollo_data,
    showtimes_url,
)
from scripts.ingest.amc.sampling import stratified_sample
from scripts.ingest.amc.scheduler import (
    DEFAULT_OFFSETS_MINUTES,
    due_snapshots,
    theatres_in_local_morning,
)
from scripts.ingest.amc.services import movie_service, progress_service, showtime_service, theatre_service
from scripts.ingest.amc.sitemap import fetch_theatre_sitemap, parse_theatre_sitemap
from scripts.ingest.amc.timezones import UTC


@dataclass(frozen=True)
class MovieOption:
    amc_movie_id: str
    amc_movie_name: str
    showtime_count: int
    first_showtime: str
    last_showtime: str
    attribute_names: str


def parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected YYYY-MM-DD date, got {value!r}") from exc


def parse_utc_datetime(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


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


def add_fetch_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"AMC cache directory. Default: {DEFAULT_CACHE_DIR}",
    )
    parser.add_argument("--refresh", action="store_true", help="Refresh cached AMC pages.")
    parser.add_argument("--offline", action="store_true", help="Use only cached AMC pages.")
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)


def target_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-amc-movie-id", help="AMC movie ID to follow up.")
    parser.add_argument("--target-amc-movie-name", help="Exact AMC movie name to follow up.")


def require_target(args: Any) -> None:
    if not args.target_amc_movie_id and not args.target_amc_movie_name:
        raise SystemExit("Provide --target-amc-movie-id or --target-amc-movie-name")


def target_matches(row: Any, *, target_id: str | None, target_name: str | None) -> bool:
    if target_id and row.movie_id == target_id:
        return True
    if target_name and row.movie_name.lower() == target_name.lower():
        return True
    return False


def fetch_showtime_rows(fetcher: HtmlFetcher, *, theatre_slug: str, target_date: dt.date) -> tuple[list[Any], str]:
    url = showtimes_url(target_date, theatre_slug)
    html_text, cache_path, _fetched = fetcher.get(url)
    apollo = maybe_parse_apollo_data(html_text, source_url=url)
    rows = (
        extract_showtimes(apollo, theatre_slug=theatre_slug, date=target_date)
        if apollo is not None
        else extract_rendered_showtimes(html_text, theatre_slug=theatre_slug, date=target_date)
    )
    return rows, str(cache_path)


def movie_options_from_showtimes(rows: list[Any]) -> list[MovieOption]:
    grouped: dict[tuple[str, str], list[Any]] = {}
    for row in rows:
        key = (row.movie_id, row.movie_name)
        grouped.setdefault(key, []).append(row)

    options: list[MovieOption] = []
    for (movie_id, movie_name), movie_rows in grouped.items():
        sorted_rows = sorted(movie_rows, key=lambda row: row.when)
        attributes = sorted(
            {
                attribute
                for row in movie_rows
                for attribute in row.attribute_names.split("|")
                if attribute
            }
        )
        options.append(
            MovieOption(
                amc_movie_id=movie_id,
                amc_movie_name=movie_name,
                showtime_count=len(movie_rows),
                first_showtime=sorted_rows[0].when,
                last_showtime=sorted_rows[-1].when,
                attribute_names=", ".join(attributes),
            )
        )
    return sorted(options, key=lambda option: (-option.showtime_count, option.amc_movie_name.lower()))


def format_movie_options(options: list[MovieOption]) -> str:
    lines = ["AMC movies found:"]
    for index, option in enumerate(options, start=1):
        attributes = f" | {option.attribute_names}" if option.attribute_names else ""
        lines.append(
            f"{index}. {option.amc_movie_name} "
            f"(AMC movie id: {option.amc_movie_id}, showtimes: {option.showtime_count}, "
            f"first: {option.first_showtime}, last: {option.last_showtime}{attributes})"
        )
    return "\n".join(lines)


def choose_movie_option(options: list[MovieOption], *, selection: int | None = None) -> MovieOption:
    if not options:
        raise SystemExit("No AMC movie options found for that theatre/date.")
    if selection is None:
        raw_value = input("Select movie number: ").strip()
        try:
            selection = int(raw_value)
        except ValueError as exc:
            raise SystemExit(f"Expected a movie number, got {raw_value!r}") from exc
    if selection < 1 or selection > len(options):
        raise SystemExit(f"Selection must be between 1 and {len(options)}")
    return options[selection - 1]


def ensure_theatres_available(conn: Any, fetcher: HtmlFetcher) -> int:
    if db.select_largest_theatre(conn) is not None:
        return 0
    xml_text, _cache_path = fetch_theatre_sitemap(fetcher)
    theatres = parse_theatre_sitemap(xml_text)
    return db.upsert_theatres(conn, theatres)


def describe_local_window_block(
    *,
    active_count: int,
    now_utc: dt.datetime,
    start_hour: int,
    end_hour: int,
) -> str:
    return (
        f"No theatres are currently in the local morning window "
        f"{start_hour:02d}:00-{end_hour:02d}:00. "
        f"Active theatres before filtering: {active_count}. "
        f"Current UTC time used: {now_utc.isoformat()}. "
        "For an ad-hoc/backfill run, add --ignore-local-window."
    )


def cmd_init_db(args: Any) -> int:
    conn = connect_database()
    try:
        db.initialize_amc_database(conn)
        conn.commit()
    finally:
        conn.close()
    print("Initialized AMC database tables.")
    return 0


def cmd_ingest_theatres(args: Any) -> int:
    fetcher = build_fetcher(args)
    conn = connect_database()
    try:
        db.initialize_amc_database(conn)
        count = theatre_service.sync_theatres(conn, fetcher)
        run_id = db.create_collection_run(
            conn,
            run_type="sitemap",
            status="completed",
        )
        db.complete_collection_run(conn, run_id, status="completed")
        conn.commit()
    finally:
        conn.close()
    print(f"Upserted {count} AMC theatres.")
    return 0


def cmd_create_inventory_run(args: Any) -> int:
    conn = connect_database()
    try:
        db.initialize_amc_database(conn)
        run_id, task_count = showtime_service.create_inventory_run(conn, exhibition_date=args.target_date)
        conn.commit()
    finally:
        conn.close()
    print(f"Created showtime inventory run {run_id} with {task_count} theatre tasks.")
    return 0


def cmd_list_movies(args: Any) -> int:
    conn = connect_database()
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
    conn = connect_database()
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


def cmd_create_seat_run(args: Any) -> int:
    conn = connect_database()
    try:
        db.initialize_amc_database(conn)
        run_id, task_count = movie_service.create_seat_collection_run(
            conn,
            exhibition_date=args.target_date,
            target_offset_minutes=args.target_offset_minutes,
        )
        conn.commit()
    finally:
        conn.close()
    print(f"Created seat collection run {run_id} with {task_count} tasks.")
    return 0


def cmd_run_progress(args: Any) -> int:
    conn = connect_database()
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
    conn = connect_database()
    try:
        db.initialize_amc_database(conn)
        count = db.cancel_run(conn, db.as_uuid(args.run_id))
        conn.commit()
    finally:
        conn.close()
    print(f"Cancelled {count} queued/running tasks for run {args.run_id}.")
    return 0


def cmd_cancel_campaign(args: Any) -> int:
    conn = connect_database()
    try:
        db.initialize_amc_database(conn)
        campaign_id = db.ensure_campaign(conn, args.target_date)
        count = db.cancel_campaign_runs(conn, campaign_id)
        conn.commit()
    finally:
        conn.close()
    print(f"Cancelled {count} queued/running tasks for {args.target_date.isoformat()}.")
    return 0


def cmd_morning_showtimes(args: Any) -> int:
    require_target(args)
    fetcher = build_fetcher(args)
    target_date = args.target_date
    now_utc = parse_utc_datetime(args.now_utc) if args.now_utc else dt.datetime.now(UTC)
    conn = connect_database()
    run_id: int | None = None
    try:
        db.initialize_amc_database(conn)
        seeded_theatres = ensure_theatres_available(conn, fetcher)
        if seeded_theatres:
            conn.commit()
        all_theatres = db.select_active_theatres(conn)
        theatres = all_theatres
        if not args.ignore_local_window:
            theatres = theatres_in_local_morning(
                all_theatres,
                now_utc=now_utc,
                start_hour=args.morning_start_hour,
                end_hour=args.morning_end_hour,
            )
            if not theatres:
                raise SystemExit(
                    describe_local_window_block(
                        active_count=len(all_theatres),
                        now_utc=now_utc,
                        start_hour=args.morning_start_hour,
                        end_hour=args.morning_end_hour,
                    )
                )
        sample = stratified_sample(theatres, sample_size=args.sample_size, seed=args.seed)
        selected_theatres = [item.theatre for item in sample]
        run_id = db.create_collection_run(
            conn,
            run_type="morning_showtimes",
            target_date=target_date.isoformat(),
            target_amc_movie_id=args.target_amc_movie_id,
            target_amc_movie_name=args.target_amc_movie_name,
            selected_theatre_ids=[theatre.amc_theatre_id for theatre in selected_theatres],
        )

        showtime_count = 0
        target_showtime_count = 0
        for theatre in selected_theatres:
            rows, cache_path = fetch_showtime_rows(fetcher, theatre_slug=theatre.slug, target_date=target_date)
            showtime_count += len(rows)
            target_showtime_count += sum(
                1
                for row in rows
                if target_matches(
                    row,
                    target_id=args.target_amc_movie_id,
                    target_name=args.target_amc_movie_name,
                )
            )
            if not args.dry_run:
                db.upsert_showtimes(
                    conn,
                    theatre=theatre,
                    showtimes=rows,
                    raw_cache_path=cache_path,
                    fetched_at=now_utc,
                )

        db.complete_collection_run(conn, run_id, status="completed")
        if args.dry_run:
            conn.rollback()
        else:
            conn.commit()
    except Exception as exc:
        if run_id is not None:
            db.complete_collection_run(conn, run_id, status="failed", error=str(exc))
            conn.commit()
        raise
    finally:
        conn.close()

    print(
        "Collected "
        f"{showtime_count} showtimes across {len(selected_theatres)} theatres; "
        f"{target_showtime_count} matched the target."
    )
    return 0


def cmd_select_movie(args: Any) -> int:
    fetcher = build_fetcher(args)
    conn = connect_database()
    try:
        db.initialize_amc_database(conn)
        seeded_theatres = ensure_theatres_available(conn, fetcher)
        if seeded_theatres:
            conn.commit()
        if args.theatre_slug:
            theatre = db.select_theatre_by_slug(conn, args.theatre_slug)
            if theatre is None:
                raise SystemExit(f"No active AMC theatre found with slug {args.theatre_slug!r}")
        else:
            theatre = db.select_largest_theatre(conn)
            if theatre is None:
                raise SystemExit("No active AMC theatres found. Run ingest-theatres first.")

        rows, cache_path = fetch_showtime_rows(
            fetcher,
            theatre_slug=theatre.slug,
            target_date=args.target_date,
        )
        options = movie_options_from_showtimes(rows)
        print(
            f"Scanned {theatre.name} ({theatre.slug}) for {args.target_date.isoformat()} "
            f"from {cache_path}."
        )
        print(format_movie_options(options))
        selected = choose_movie_option(options, selection=args.selection)
        print()
        print("Selected:")
        print(f"--target-amc-movie-id {selected.amc_movie_id!r}")
        print(f"--target-amc-movie-name {selected.amc_movie_name!r}")

        if args.save_to_db:
            db.upsert_movie_target(
                conn,
                target_date=args.target_date.isoformat(),
                amc_movie_id=selected.amc_movie_id,
                amc_movie_name=selected.amc_movie_name,
                source_theatre_id=theatre.amc_theatre_id,
                source_theatre_slug=theatre.slug,
                source_theatre_name=theatre.name,
                notes=args.notes,
            )
            conn.commit()
            print("Saved selected movie to amc_movie_targets.")
        else:
            conn.rollback()
    finally:
        conn.close()
    return 0


def cmd_seat_snapshots(args: Any) -> int:
    require_target(args)
    fetcher = build_fetcher(args)
    now_utc = parse_utc_datetime(args.now_utc) if args.now_utc else dt.datetime.now(UTC)
    offsets = parse_offsets(args.offset_minutes)
    conn = connect_database()
    run_id: int | None = None
    try:
        db.initialize_amc_database(conn)
        showtimes = db.select_showtimes_for_target(
            conn,
            target_date=args.target_date.isoformat(),
            target_amc_movie_id=args.target_amc_movie_id,
            target_amc_movie_name=args.target_amc_movie_name,
        )
        due = due_snapshots(
            showtimes,
            now_utc=now_utc,
            offsets_minutes=offsets,
            grace_minutes=args.grace_minutes,
        )
        run_id = db.create_collection_run(
            conn,
            run_type="seat_snapshots",
            target_date=args.target_date.isoformat(),
            target_amc_movie_id=args.target_amc_movie_id,
            target_amc_movie_name=args.target_amc_movie_name,
            selected_theatre_ids=[showtime.amc_theatre_id for showtime, _snapshot in due],
        )

        collected = 0
        skipped_existing = 0
        for showtime, snapshot in due:
            if db.snapshot_exists(
                conn,
                showtime_id=showtime.showtime_id,
                minutes_before_showtime=snapshot.minutes_before_showtime,
            ):
                skipped_existing += 1
                continue
            if args.dry_run:
                collected += 1
                continue
            fill = fetch_seat_fill(
                fetcher,
                theatre_slug=showtime.theatre_slug,
                date=dt.date.fromisoformat(showtime.local_show_date),
                showtime_id=showtime.showtime_id,
            )
            db.upsert_seat_snapshot(
                conn,
                showtime=showtime,
                seat_fill=fill,
                snapshot_utc_at=now_utc,
                minutes_before_showtime=snapshot.minutes_before_showtime,
                fetched_at=now_utc,
            )
            collected += 1

        db.complete_collection_run(conn, run_id, status="completed")
        if args.dry_run:
            conn.rollback()
        else:
            conn.commit()
    except Exception as exc:
        if run_id is not None:
            db.complete_collection_run(conn, run_id, status="failed", error=str(exc))
            conn.commit()
        raise
    finally:
        conn.close()

    print(f"Collected {collected} due seat snapshots; skipped {skipped_existing} existing snapshots.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_fetch_args(parser)

    subparsers = parser.add_subparsers(dest="command", required=True)
    init_db = subparsers.add_parser("init-db", help="Create AMC database tables.")
    init_db.set_defaults(func=cmd_init_db)

    ingest = subparsers.add_parser("ingest-theatres", help="Ingest AMC theatres from sitemap.")
    add_fetch_args(ingest)
    ingest.set_defaults(func=cmd_ingest_theatres)

    inventory = subparsers.add_parser(
        "create-inventory-run",
        help="Create a durable full-network showtime inventory run for a date.",
    )
    inventory.add_argument("target_date", type=parse_date)
    inventory.set_defaults(func=cmd_create_inventory_run)

    movies = subparsers.add_parser(
        "list-movies",
        help="List database-backed AMC movie inventory for a date.",
    )
    movies.add_argument("target_date", type=parse_date)
    movies.set_defaults(func=cmd_list_movies)

    toggle = subparsers.add_parser(
        "toggle-movie",
        help="Select or deselect an AMC movie for a campaign date.",
    )
    toggle.add_argument("target_date", type=parse_date)
    toggle.add_argument("amc_movie_id")
    toggle_group = toggle.add_mutually_exclusive_group(required=True)
    toggle_group.add_argument("--selected", dest="selected", action="store_true")
    toggle_group.add_argument("--deselected", dest="selected", action="store_false")
    toggle.set_defaults(func=cmd_toggle_movie)

    seat_run = subparsers.add_parser(
        "create-seat-run",
        help="Create durable seat-scan tasks for selected movies on a date.",
    )
    seat_run.add_argument("target_date", type=parse_date)
    seat_run.add_argument("--target-offset-minutes", type=int, default=5)
    seat_run.set_defaults(func=cmd_create_seat_run)

    progress = subparsers.add_parser("run-progress", help="Print progress for a durable run.")
    progress.add_argument("run_id")
    progress.set_defaults(func=cmd_run_progress)

    cancel_run = subparsers.add_parser("cancel-run", help="Cancel queued/running tasks for a durable run.")
    cancel_run.add_argument("run_id")
    cancel_run.set_defaults(func=cmd_cancel_run)

    cancel_campaign = subparsers.add_parser(
        "cancel-campaign",
        help="Cancel queued/running durable tasks for a campaign date.",
    )
    cancel_campaign.add_argument("target_date", type=parse_date)
    cancel_campaign.set_defaults(func=cmd_cancel_campaign)

    selector = subparsers.add_parser("select-movie", help="Scan one large theatre and choose a target movie.")
    add_fetch_args(selector)
    selector.add_argument("target_date", type=parse_date)
    selector.add_argument(
        "--theatre-slug",
        help="AMC theatre slug to scan. Defaults to the largest active theatre in the DB.",
    )
    selector.add_argument(
        "--selection",
        type=int,
        help="Choose a numbered option non-interactively.",
    )
    selector.add_argument(
        "--save-to-db",
        action="store_true",
        help="Save the selected target to amc_movie_targets. Default is print-only.",
    )
    selector.add_argument("--notes", help="Optional notes when using --save-to-db.")
    selector.set_defaults(func=cmd_select_movie)

    morning = subparsers.add_parser("morning-showtimes", help="Collect morning showtimes for sampled theatres.")
    add_fetch_args(morning)
    morning.add_argument("target_date", type=parse_date)
    target_args(morning)
    morning.add_argument("--sample-size", type=int, default=60)
    morning.add_argument("--seed", default="")
    morning.add_argument("--now-utc", help="Override current UTC time, ISO format. Useful for tests/dry runs.")
    morning.add_argument("--morning-start-hour", type=int, default=6)
    morning.add_argument("--morning-end-hour", type=int, default=10)
    morning.add_argument("--ignore-local-window", action="store_true")
    morning.add_argument("--dry-run", action="store_true")
    morning.set_defaults(func=cmd_morning_showtimes)

    snapshots = subparsers.add_parser("seat-snapshots", help="Collect due seat snapshots for target showtimes.")
    add_fetch_args(snapshots)
    snapshots.add_argument("target_date", type=parse_date)
    target_args(snapshots)
    snapshots.add_argument(
        "--offset-minutes",
        default=",".join(str(value) for value in DEFAULT_OFFSETS_MINUTES),
        help="Comma-separated minutes before showtime to collect.",
    )
    snapshots.add_argument("--grace-minutes", type=int, default=20)
    snapshots.add_argument("--now-utc", help="Override current UTC time, ISO format. Useful for tests/dry runs.")
    snapshots.add_argument("--dry-run", action="store_true")
    snapshots.set_defaults(func=cmd_seat_snapshots)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
