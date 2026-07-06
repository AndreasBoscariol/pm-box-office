#!/usr/bin/env python3
"""Probe how long AMC seat maps remain collectable around showtime start."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pm_box_office.config import REPO_ROOT
from pm_box_office.db.connection import connect_database
from pm_box_office.sources.amc import db
from pm_box_office.sources.amc.client import DEFAULT_CACHE_DIR, DEFAULT_USER_AGENT, HtmlFetcher
from pm_box_office.sources.amc.parsers import fetch_seat_fill


DEFAULT_OFFSETS_MINUTES = (-1, 0, 1, 2, 5, 10, 15, 20, 30, 45, 60)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "diagnostics"


@dataclass(frozen=True)
class ProbeTarget:
    showtime_id: str
    amc_theatre_id: int
    theatre_slug: str
    local_show_date: str
    utc_start_at: dt.datetime
    timezone: str
    amc_movie_id: str
    amc_movie_name: str


def parse_offsets(value: str) -> tuple[int, ...]:
    try:
        offsets = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integer minute offsets") from exc
    if not offsets:
        raise argparse.ArgumentTypeError("At least one offset is required")
    return offsets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", help="PostgreSQL URL. Defaults to DATABASE_URL/POSTGRES_DSN/.env.")
    parser.add_argument("--showtime-id", action="append", default=[], help="Specific AMC showtime ID to probe.")
    parser.add_argument("--max-showtimes", type=int, default=1, help="Maximum DB-selected showtimes to probe.")
    parser.add_argument("--candidate-window-minutes", type=int, default=180)
    parser.add_argument(
        "--min-start-lead-minutes",
        type=int,
        default=None,
        help="Only select DB candidates starting at least this many minutes from now.",
    )
    parser.add_argument("--movie-id", help="Optional AMC movie ID filter for DB-selected candidates.")
    parser.add_argument("--theatre-id", type=int, help="Optional AMC theatre ID filter for DB-selected candidates.")
    parser.add_argument(
        "--offsets-minutes",
        type=parse_offsets,
        default=DEFAULT_OFFSETS_MINUTES,
        help="Comma-separated offsets relative to showtime start; -1 means one minute before. "
        f"Default: {','.join(str(offset) for offset in DEFAULT_OFFSETS_MINUTES)}",
    )
    parser.add_argument("--list-candidates", action="store_true", help="Print candidate showtimes and exit.")
    parser.add_argument("--skip-wait", action="store_true", help="Probe due/past offsets only; skip future offsets.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--output", type=Path, help="JSONL output path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    conn = connect_database(args.database_url)
    try:
        db.initialize_amc_database(conn)
        targets = load_targets(conn, args)
        conn.commit()
    finally:
        conn.close()

    if not targets:
        print("No AMC showtime candidates found.")
        return 1
    if args.list_candidates:
        for target in targets:
            print(format_target(target))
        return 0

    output_path = args.output or default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fetcher = HtmlFetcher(
        args.cache_dir,
        refresh=False,
        offline=False,
        delay_seconds=args.delay_seconds,
        user_agent=args.user_agent,
        global_requests_per_minute=20,
    )
    print(f"Writing probe rows to {output_path}")
    for target in targets:
        run_probe(fetcher, target, offsets_minutes=args.offsets_minutes, output_path=output_path, skip_wait=args.skip_wait)
    summarize_output(output_path)
    return 0


def load_targets(conn: Any, args: argparse.Namespace) -> list[ProbeTarget]:
    if args.showtime_id:
        rows = conn.execute(
            """
            SELECT showtime_id, amc_theatre_id, theatre_slug, exhibition_date,
                   starts_at_utc, timezone, amc_movie_id, amc_movie_name
            FROM amc_showtimes
            WHERE showtime_id = ANY(%s)
            ORDER BY starts_at_utc, showtime_id
            """,
            (args.showtime_id,),
        ).fetchall()
        return [target_from_row(row) for row in rows]

    now = db.utc_now()
    clauses = [
        "starts_at_utc BETWEEN %s AND %s",
        "attribute_names ILIKE %s",
    ]
    min_start_at = (
        now + dt.timedelta(minutes=args.min_start_lead_minutes)
        if args.min_start_lead_minutes is not None
        else now - dt.timedelta(minutes=max(args.offsets_minutes, default=0) + 5)
    )
    params: list[Any] = [min_start_at, now + dt.timedelta(minutes=args.candidate_window_minutes), "%Reserved Seating%"]
    if args.movie_id:
        clauses.append("amc_movie_id = %s")
        params.append(args.movie_id)
    if args.theatre_id is not None:
        clauses.append("amc_theatre_id = %s")
        params.append(args.theatre_id)
    params.append(max(1, args.max_showtimes))
    rows = conn.execute(
        f"""
        SELECT showtime_id, amc_theatre_id, theatre_slug, exhibition_date,
               starts_at_utc, timezone, amc_movie_id, amc_movie_name
        FROM amc_showtimes
        WHERE {' AND '.join(clauses)}
        ORDER BY ABS(EXTRACT(EPOCH FROM (starts_at_utc - CURRENT_TIMESTAMP))), starts_at_utc, showtime_id
        LIMIT %s
        """,
        params,
    ).fetchall()
    return [target_from_row(row) for row in rows]


def target_from_row(row: Any) -> ProbeTarget:
    return ProbeTarget(
        showtime_id=str(row[0]),
        amc_theatre_id=int(row[1]),
        theatre_slug=str(row[2]),
        local_show_date=db.row_date_text(row[3]),
        utc_start_at=db.ensure_utc(db.row_datetime(row[4])),
        timezone=str(row[5]),
        amc_movie_id=str(row[6]),
        amc_movie_name=str(row[7]),
    )


def run_probe(
    fetcher: HtmlFetcher,
    target: ProbeTarget,
    *,
    offsets_minutes: tuple[int, ...],
    output_path: Path,
    skip_wait: bool,
) -> None:
    print(format_target(target))
    for offset in offsets_minutes:
        planned_at = target.utc_start_at + dt.timedelta(minutes=offset)
        now = db.utc_now()
        if planned_at > now and skip_wait:
            write_row(output_path, target, offset, planned_at, "skipped_future", None, None)
            continue
        wait_seconds = max(0.0, (planned_at - now).total_seconds())
        if wait_seconds:
            print(f"waiting {wait_seconds:.0f}s for offset {offset:+d}m")
            time.sleep(wait_seconds)
        observed_at = db.utc_now()
        try:
            fill = fetch_seat_fill(
                fetcher,
                theatre_slug=target.theatre_slug,
                date=dt.date.fromisoformat(target.local_show_date),
                showtime_id=target.showtime_id,
                prefer_rsc=True,
            )
        except Exception as exc:
            actual_seconds_after_start = int((observed_at - target.utc_start_at).total_seconds())
            write_row(output_path, target, offset, planned_at, "error", actual_seconds_after_start, exc)
            print(f"{offset:+d}m error {type(exc).__name__}: {str(exc)[:160]}")
            continue
        actual_seconds_after_start = int((observed_at - target.utc_start_at).total_seconds())
        write_row(
            output_path,
            target,
            offset,
            planned_at,
            "success",
            actual_seconds_after_start,
            None,
            {
                "total_seats": fill.total_seats,
                "available_seats": fill.available_seats,
                "filled_or_unavailable_seats": fill.filled_or_unavailable_seats,
                "fill_rate": fill.fill_rate,
                "parse_method": fill.parse_method,
                "raw_cache_path": fill.raw_cache_path,
            },
        )
        print(
            f"{offset:+d}m success seats={fill.total_seats} "
            f"available={fill.available_seats} method={fill.parse_method}"
        )


def write_row(
    output_path: Path,
    target: ProbeTarget,
    offset: int,
    planned_at: dt.datetime,
    status: str,
    actual_seconds_after_start: int | None,
    exc: Exception | None,
    extra: dict[str, Any] | None = None,
) -> None:
    row = {
        "observed_at_utc": db.utc_now().isoformat(),
        "planned_at_utc": planned_at.isoformat(),
        "target_offset_minutes": offset,
        "actual_seconds_after_start": actual_seconds_after_start,
        "status": status,
        "showtime_id": target.showtime_id,
        "amc_theatre_id": target.amc_theatre_id,
        "theatre_slug": target.theatre_slug,
        "local_show_date": target.local_show_date,
        "showtime_start_utc": target.utc_start_at.isoformat(),
        "timezone": target.timezone,
        "amc_movie_id": target.amc_movie_id,
        "amc_movie_name": target.amc_movie_name,
    }
    if exc is not None:
        row.update({"error_type": type(exc).__name__, "error_message": str(exc)[:1000]})
    row.update(extra or {})
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def summarize_output(output_path: Path) -> None:
    rows = []
    for line in output_path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    successes = [row for row in rows if row.get("status") == "success"]
    errors = [row for row in rows if row.get("status") == "error"]
    if successes:
        latest_success = max(successes, key=lambda row: int(row.get("actual_seconds_after_start") or -999999))
        print(
            "Latest successful seat-map probe: "
            f"{int(latest_success['actual_seconds_after_start'])}s after start "
            f"(offset {int(latest_success['target_offset_minutes']):+d}m)."
        )
    else:
        print("No successful seat-map probes recorded.")
    if errors:
        first_error = min(errors, key=lambda row: int(row.get("actual_seconds_after_start") or 999999))
        print(
            "First failed seat-map probe: "
            f"{int(first_error['actual_seconds_after_start'])}s after start "
            f"{first_error.get('error_type')}: {first_error.get('error_message')}"
        )


def format_target(target: ProbeTarget) -> str:
    now = db.utc_now()
    seconds_until_start = int((target.utc_start_at - now).total_seconds())
    return (
        f"{target.showtime_id} {target.amc_movie_name} theatre={target.amc_theatre_id} "
        f"start_utc={target.utc_start_at.isoformat()} seconds_until_start={seconds_until_start}"
    )


def default_output_path() -> Path:
    stamp = db.utc_now().strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_DIR / f"amc_seat_tolerance_{stamp}.jsonl"


if __name__ == "__main__":
    raise SystemExit(main())
