#!/usr/bin/env python3
"""Sync Polymarket box-office market metadata into PostgreSQL.

This source is intentionally metadata-only: it discovers movie/opening-weekend
events, parses bucket boundaries, validates complete bucket sets, matches them
to internal movies, and lets the forecast UI use those real market buckets.
Live price/order-book collection belongs to a separate worker.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from pm_box_office.db.connection import connect_database, database_url_from_env
from pm_box_office.domain.movies import normalize_title
from pm_box_office.sources.common.schema import acquire_schema_init_lock
from pm_box_office.sources.polymarket.accounts import (
    GAMMA_BASE_URL,
    PolymarketClient,
    discover_movie_tag_id,
    market_matches_focus,
)
from prediction_market_backtest.matching import MovieCandidate, match_movie
from prediction_market_backtest.semantics import Bucket, parse_bucket, validate_bucket_set, with_override


DEFAULT_CACHE_DIR = Path("data/raw/polymarket/metadata")
DEFAULT_LIMIT = 200
PARSER_VERSION = "polymarket_box_office_metadata_v1"


@dataclass(frozen=True)
class SyncSummary:
    events_seen: int = 0
    events_upserted: int = 0
    markets_upserted: int = 0
    semantics_upserted: int = 0
    valid_events: int = 0
    movie_matches_upserted: int = 0
    matched_movie_ids: tuple[int, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "events_seen": self.events_seen,
            "events_upserted": self.events_upserted,
            "markets_upserted": self.markets_upserted,
            "semantics_upserted": self.semantics_upserted,
            "valid_events": self.valid_events,
            "movie_matches_upserted": self.movie_matches_upserted,
            "matched_movie_ids": list(self.matched_movie_ids),
        }


def ensure_schema(conn: Any) -> None:
    acquire_schema_init_lock(conn)
    conn.executescript(
        """
        CREATE SCHEMA IF NOT EXISTS prediction_market_backtest;

        CREATE TABLE IF NOT EXISTS prediction_market_backtest.polymarket_events (
          event_id TEXT PRIMARY KEY,
          event_slug TEXT,
          title TEXT NOT NULL,
          description TEXT,
          category TEXT,
          tags JSONB NOT NULL DEFAULT '[]',
          active BOOLEAN,
          closed BOOLEAN,
          start_time TIMESTAMPTZ,
          end_time TIMESTAMPTZ,
          resolution_source TEXT,
          resolution_rules_raw TEXT,
          resolution_rules_hash TEXT,
          created_at_exchange TIMESTAMPTZ,
          updated_at_exchange TIMESTAMPTZ,
          synced_at_utc TIMESTAMPTZ NOT NULL,
          raw_gamma_json JSONB NOT NULL
        );

        CREATE TABLE IF NOT EXISTS prediction_market_backtest.polymarket_markets (
          market_id TEXT PRIMARY KEY,
          event_id TEXT NOT NULL REFERENCES prediction_market_backtest.polymarket_events(event_id),
          condition_id TEXT,
          question TEXT NOT NULL,
          slug TEXT,
          active BOOLEAN,
          closed BOOLEAN,
          accepting_orders BOOLEAN,
          neg_risk BOOLEAN,
          tick_size NUMERIC(20,10),
          minimum_order_size NUMERIC(30,10),
          fee_rate NUMERIC(20,10),
          fee_schedule_raw JSONB,
          yes_token_id TEXT,
          no_token_id TEXT,
          outcome_labels JSONB,
          resolution_status TEXT,
          winning_outcome TEXT,
          raw_gamma_json JSONB NOT NULL,
          raw_clob_json JSONB
        );

        CREATE TABLE IF NOT EXISTS prediction_market_backtest.contract_semantics (
          market_id TEXT PRIMARY KEY REFERENCES prediction_market_backtest.polymarket_markets(market_id),
          target_metric TEXT,
          geographic_scope TEXT,
          currency TEXT,
          weekend_duration INTEGER,
          opening_weekend_start_date DATE,
          opening_weekend_end_date DATE,
          bucket_lower NUMERIC(24,2),
          bucket_upper NUMERIC(24,2),
          include_lower BOOLEAN NOT NULL DEFAULT TRUE,
          include_upper BOOLEAN NOT NULL DEFAULT FALSE,
          lower_unbounded BOOLEAN NOT NULL,
          upper_unbounded BOOLEAN NOT NULL,
          gross_unit TEXT NOT NULL DEFAULT 'dollars',
          resolution_source TEXT,
          parser_version TEXT NOT NULL,
          parse_confidence NUMERIC(6,5),
          parse_status TEXT NOT NULL,
          reviewer_override JSONB,
          review_notes TEXT
        );

        CREATE TABLE IF NOT EXISTS prediction_market_backtest.event_bucket_validations (
          event_id TEXT PRIMARY KEY REFERENCES prediction_market_backtest.polymarket_events(event_id),
          validation_status TEXT NOT NULL,
          validation_version TEXT NOT NULL,
          validation_errors JSONB NOT NULL DEFAULT '[]',
          validated_bucket_count INTEGER NOT NULL,
          validated_at TIMESTAMPTZ NOT NULL,
          reviewer_override JSONB
        );

        CREATE TABLE IF NOT EXISTS prediction_market_backtest.movie_matches (
          event_id TEXT PRIMARY KEY REFERENCES prediction_market_backtest.polymarket_events(event_id),
          movie_id BIGINT REFERENCES movies(movie_id),
          normalized_market_title TEXT,
          normalized_internal_title TEXT,
          release_date_distance INTEGER,
          title_similarity NUMERIC(7,6),
          semantic_compatibility JSONB,
          match_score NUMERIC(7,6),
          match_method TEXT,
          match_status TEXT NOT NULL,
          reviewer TEXT,
          reviewer_timestamp TIMESTAMPTZ,
          manual_override BOOLEAN NOT NULL DEFAULT FALSE,
          rejection_reason TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_prediction_market_polymarket_markets_event
          ON prediction_market_backtest.polymarket_markets(event_id);
        CREATE INDEX IF NOT EXISTS idx_prediction_market_movie_matches_movie
          ON prediction_market_backtest.movie_matches(movie_id);
        """
    )


def discover_events(
    client: PolymarketClient,
    *,
    slug: str | None = None,
    limit: int = DEFAULT_LIMIT,
    focus: str = "box_office",
) -> list[dict[str, Any]]:
    if slug:
        event = client.get_json(GAMMA_BASE_URL, f"/events/slug/{slug}")
        if isinstance(event, dict) and event.get("id"):
            return [event]
        fallback = client.get_json(GAMMA_BASE_URL, "/events", {"slug": slug, "limit": 5})
        rows = fallback if isinstance(fallback, list) else fallback.get("data", []) if isinstance(fallback, dict) else []
        return [row for row in rows if isinstance(row, dict)]

    tag_id = discover_movie_tag_id(client)
    by_event: dict[str, dict[str, Any]] = {}
    for closed in (False, True):
        cursor = None
        while len(by_event) < limit:
            payload = client.get_json(
                GAMMA_BASE_URL,
                "/markets/keyset",
                {
                    "tag_id": tag_id,
                    "include_tag": "true",
                    "limit": 100,
                    "closed": str(closed).lower(),
                    "after_cursor": cursor,
                },
            )
            markets = payload.get("markets", []) if isinstance(payload, dict) else []
            for market in markets:
                if not isinstance(market, dict) or not market_matches_focus(market, focus):
                    continue
                for event in market.get("events") or []:
                    if not isinstance(event, dict):
                        continue
                    event_id = str(event.get("id") or event.get("event_id") or "")
                    if not event_id:
                        continue
                    event.setdefault("markets", [])
                    event["markets"].append(market)
                    by_event[event_id] = event
                    if len(by_event) >= limit:
                        break
            cursor = payload.get("next_cursor") if isinstance(payload, dict) else None
            if not cursor or len(by_event) >= limit:
                break
    return [hydrate_event(client, event) for event in by_event.values()]


def hydrate_event(client: PolymarketClient, event: dict[str, Any]) -> dict[str, Any]:
    slug = str(event.get("slug") or event.get("event_slug") or "")
    if not slug:
        return event
    detail = client.get_json(GAMMA_BASE_URL, f"/events/slug/{slug}")
    if isinstance(detail, dict) and detail.get("id"):
        return detail
    return event


def sync_metadata(conn: Any, events: Iterable[dict[str, Any]], *, dry_run: bool = False) -> SyncSummary:
    ensure_schema(conn)
    inventory = movie_inventory(conn)
    stats = {
        "events_seen": 0,
        "events_upserted": 0,
        "markets_upserted": 0,
        "semantics_upserted": 0,
        "valid_events": 0,
        "movie_matches_upserted": 0,
    }
    matched_movie_ids: set[int] = set()

    for event in events:
        event_id = event_id_for(event)
        markets = [market for market in event.get("markets") or [] if isinstance(market, dict)]
        if not event_id or not markets:
            continue
        stats["events_seen"] += 1
        buckets, errors = parse_event_buckets(event)
        validation = validate_bucket_set(buckets) if buckets else None
        validation_errors = list(errors)
        if validation is not None:
            validation_errors.extend(validation.errors)
        valid = bool(validation and validation.valid and len(buckets) == 5 and not validation_errors)
        match = match_event_movie(event, inventory)
        if match.movie_id is not None:
            matched_movie_ids.add(int(match.movie_id))

        if dry_run:
            continue

        upsert_event(conn, event_id, event)
        stats["events_upserted"] += 1
        for market in markets:
            market_id = market_id_for(market)
            if not market_id:
                continue
            upsert_market(conn, event_id, market)
            stats["markets_upserted"] += 1
        for bucket in buckets:
            upsert_contract_semantics(conn, bucket)
            stats["semantics_upserted"] += 1
        upsert_validation(conn, event_id, valid=valid, errors=validation_errors, bucket_count=len(buckets))
        if valid:
            stats["valid_events"] += 1
        upsert_movie_match(conn, event_id, event, match)
        stats["movie_matches_upserted"] += 1

    return SyncSummary(**stats, matched_movie_ids=tuple(sorted(matched_movie_ids)))


def parse_event_buckets(event: dict[str, Any]) -> tuple[list[Bucket], list[str]]:
    rules = event_rules(event)
    scope = "domestic_us_canada" if "domestic" in rules.lower() else "ambiguous"
    source = "the_numbers" if "the-numbers.com" in rules.lower() or "the numbers" in rules.lower() else (
        "box_office_mojo" if "boxofficemojo.com" in rules.lower() or "box office mojo" in rules.lower() else None
    )
    buckets: list[Bucket] = []
    errors: list[str] = []
    for market in event.get("markets") or []:
        if not isinstance(market, dict):
            continue
        market_id = market_id_for(market)
        question = str(market.get("question") or market.get("title") or "")
        if not market_id or not question:
            continue
        try:
            buckets.append(parse_bucket(market_id, question, geographic_scope=scope, resolution_source=source))
        except ValueError as exc:
            errors.append(f"{market_id}: {exc}")
    if "higher range bracket" in rules.lower() or "higher bracket" in rules.lower():
        ordered = sorted(buckets, key=lambda bucket: (bucket.lower is not None, bucket.lower or Decimal(0)))
        buckets = [
            with_override(
                bucket,
                include_lower=True if bucket.lower is not None else bucket.include_lower,
                include_upper=False if bucket.upper is not None else bucket.include_upper,
            )
            for bucket in ordered
        ]
    return buckets, errors


def movie_inventory(conn: Any) -> list[MovieCandidate]:
    relation = conn.execute("SELECT to_regclass(%s)", ("movies",)).fetchone()
    if not relation or relation[0] is None:
        return []
    # Multiple source imports can create otherwise identical movie rows.  AMC
    # is the live operational identity for a currently tracked theatrical
    # release, so put that identity first. ``match_movie`` keeps input order
    # for exact score ties, which makes this a deterministic canonical choice
    # rather than an accidental database-row-order choice.
    amc_relation = conn.execute("SELECT to_regclass(%s)", ("amc_movies",)).fetchone()
    if amc_relation and amc_relation[0] is not None:
        rows = conn.execute(
            """
            SELECT m.movie_id, m.title, m.release_date
            FROM movies m
            LEFT JOIN amc_movies amc
              ON amc.movie_id = m.movie_id AND COALESCE(amc.active, TRUE)
            WHERE m.title IS NOT NULL
            ORDER BY (amc.movie_id IS NOT NULL) DESC, m.movie_id
            """
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT movie_id, title, release_date
            FROM movies
            WHERE title IS NOT NULL
            ORDER BY movie_id
            """
        ).fetchall()
    return [MovieCandidate(int(row[0]), str(row[1]), row[2]) for row in rows]


def match_event_movie(event: dict[str, Any], inventory: list[MovieCandidate]) -> Any:
    title = event_movie_title(event)
    return match_movie(title, event_opening_date(event), inventory)


def event_movie_title(event: dict[str, Any]) -> str:
    title = str(event.get("title") or event.get("question") or "")
    title = re.sub(r"\s+Opening Weekend.*$", "", title, flags=re.I)
    title = re.sub(r"\s+Box Office.*$", "", title, flags=re.I)
    return title.strip(" '\"")


def event_opening_date(event: dict[str, Any]) -> dt.date | None:
    text = " ".join(str(event.get(key) or "") for key in ("title", "description", "resolutionSource", "resolution_rules"))
    match = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", text)
    if match:
        return dt.date.fromisoformat(match.group(0))
    for key in ("startDate", "start_time", "startTime"):
        value = event.get(key)
        if value:
            try:
                return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
            except ValueError:
                pass
    return None


def upsert_event(conn: Any, event_id: str, event: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO prediction_market_backtest.polymarket_events (
            event_id, event_slug, title, description, category, tags, active, closed,
            start_time, end_time, resolution_source, resolution_rules_raw,
            created_at_exchange, updated_at_exchange, synced_at_utc, raw_gamma_json
        )
        VALUES (
            %s, %s, %s, %s, %s, %s::jsonb, %s, %s,
            %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, %s::jsonb
        )
        ON CONFLICT(event_id) DO UPDATE SET
            event_slug = excluded.event_slug,
            title = excluded.title,
            description = excluded.description,
            category = excluded.category,
            tags = excluded.tags,
            active = excluded.active,
            closed = excluded.closed,
            start_time = excluded.start_time,
            end_time = excluded.end_time,
            resolution_source = excluded.resolution_source,
            resolution_rules_raw = excluded.resolution_rules_raw,
            updated_at_exchange = excluded.updated_at_exchange,
            synced_at_utc = excluded.synced_at_utc,
            raw_gamma_json = excluded.raw_gamma_json
        """,
        (
            event_id,
            event.get("slug") or event.get("event_slug"),
            str(event.get("title") or event.get("question") or event_id),
            event.get("description"),
            event.get("category"),
            json.dumps(event.get("tags") or []),
            event.get("active"),
            event.get("closed"),
            timestamp_value(event.get("startDate") or event.get("startTime") or event.get("start_time")),
            timestamp_value(event.get("endDate") or event.get("endTime") or event.get("end_time")),
            event.get("resolutionSource") or event.get("resolution_source"),
            event_rules(event),
            timestamp_value(event.get("createdAt") or event.get("created_at")),
            timestamp_value(event.get("updatedAt") or event.get("updated_at")),
            json.dumps(event, default=str),
        ),
    )


def upsert_market(conn: Any, event_id: str, market: dict[str, Any]) -> None:
    yes_token, no_token, outcomes = token_ids_and_outcomes(market)
    conn.execute(
        """
        INSERT INTO prediction_market_backtest.polymarket_markets (
            market_id, event_id, condition_id, question, slug, active, closed,
            accepting_orders, neg_risk, tick_size, minimum_order_size, fee_rate,
            fee_schedule_raw, yes_token_id, no_token_id, outcome_labels,
            resolution_status, winning_outcome, raw_gamma_json
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s::jsonb, %s, %s, %s::jsonb,
            %s, %s, %s::jsonb
        )
        ON CONFLICT(market_id) DO UPDATE SET
            event_id = excluded.event_id,
            condition_id = excluded.condition_id,
            question = excluded.question,
            slug = excluded.slug,
            active = excluded.active,
            closed = excluded.closed,
            accepting_orders = excluded.accepting_orders,
            neg_risk = excluded.neg_risk,
            tick_size = excluded.tick_size,
            minimum_order_size = excluded.minimum_order_size,
            fee_rate = excluded.fee_rate,
            fee_schedule_raw = excluded.fee_schedule_raw,
            yes_token_id = excluded.yes_token_id,
            no_token_id = excluded.no_token_id,
            outcome_labels = excluded.outcome_labels,
            resolution_status = excluded.resolution_status,
            winning_outcome = excluded.winning_outcome,
            raw_gamma_json = excluded.raw_gamma_json
        """,
        (
            market_id_for(market),
            event_id,
            market.get("conditionId") or market.get("condition_id"),
            str(market.get("question") or market.get("title") or ""),
            market.get("slug"),
            market.get("active"),
            market.get("closed"),
            market.get("acceptingOrders") or market.get("accepting_orders"),
            market.get("negRisk") or market.get("neg_risk"),
            numeric_value(market.get("tickSize") or market.get("tick_size")),
            numeric_value(market.get("minimumOrderSize") or market.get("minimum_order_size")),
            numeric_value(market.get("feeRate") or market.get("fee_rate")),
            json.dumps(market.get("feeSchedule") or market.get("fee_schedule") or {}),
            yes_token,
            no_token,
            json.dumps(outcomes),
            market.get("resolutionStatus") or market.get("resolution_status"),
            market.get("winningOutcome") or market.get("winning_outcome"),
            json.dumps(market, default=str),
        ),
    )


def upsert_contract_semantics(conn: Any, bucket: Bucket) -> None:
    conn.execute(
        """
        INSERT INTO prediction_market_backtest.contract_semantics (
            market_id, target_metric, geographic_scope, currency, weekend_duration,
            bucket_lower, bucket_upper, include_lower, include_upper,
            lower_unbounded, upper_unbounded, gross_unit, resolution_source,
            parser_version, parse_confidence, parse_status
        )
        VALUES (%s, 'opening_weekend_gross', %s, %s, %s, %s, %s, %s, %s, %s, %s, 'dollars', %s, %s, %s, %s)
        ON CONFLICT(market_id) DO UPDATE SET
            target_metric = excluded.target_metric,
            geographic_scope = excluded.geographic_scope,
            currency = excluded.currency,
            weekend_duration = excluded.weekend_duration,
            bucket_lower = excluded.bucket_lower,
            bucket_upper = excluded.bucket_upper,
            include_lower = excluded.include_lower,
            include_upper = excluded.include_upper,
            lower_unbounded = excluded.lower_unbounded,
            upper_unbounded = excluded.upper_unbounded,
            gross_unit = excluded.gross_unit,
            resolution_source = excluded.resolution_source,
            parser_version = excluded.parser_version,
            parse_confidence = excluded.parse_confidence,
            parse_status = excluded.parse_status
        """,
        (
            bucket.market_id,
            bucket.geographic_scope,
            bucket.currency,
            bucket.weekend_duration,
            bucket.lower,
            bucket.upper,
            bucket.include_lower,
            bucket.include_upper,
            bucket.lower is None,
            bucket.upper is None,
            bucket.resolution_source,
            PARSER_VERSION,
            Decimal("0.90000"),
            "parsed",
        ),
    )


def upsert_validation(conn: Any, event_id: str, *, valid: bool, errors: list[str], bucket_count: int) -> None:
    conn.execute(
        """
        INSERT INTO prediction_market_backtest.event_bucket_validations (
            event_id, validation_status, validation_version, validation_errors,
            validated_bucket_count, validated_at
        )
        VALUES (%s, %s, %s, %s::jsonb, %s, CURRENT_TIMESTAMP)
        ON CONFLICT(event_id) DO UPDATE SET
            validation_status = excluded.validation_status,
            validation_version = excluded.validation_version,
            validation_errors = excluded.validation_errors,
            validated_bucket_count = excluded.validated_bucket_count,
            validated_at = excluded.validated_at
        """,
        (event_id, "valid" if valid else "invalid", PARSER_VERSION, json.dumps(errors), bucket_count),
    )


def upsert_movie_match(conn: Any, event_id: str, event: dict[str, Any], match: Any) -> None:
    normalized_market_title = normalize_title(event_movie_title(event))
    conn.execute(
        """
        INSERT INTO prediction_market_backtest.movie_matches (
            event_id, movie_id, normalized_market_title, normalized_internal_title,
            release_date_distance, title_similarity, semantic_compatibility,
            match_score, match_method, match_status, rejection_reason
        )
        VALUES (%s, %s, %s, NULL, %s, %s, %s::jsonb, %s, %s, %s, %s)
        ON CONFLICT(event_id) DO UPDATE SET
            movie_id = CASE
                WHEN prediction_market_backtest.movie_matches.manual_override
                THEN prediction_market_backtest.movie_matches.movie_id
                ELSE excluded.movie_id
            END,
            normalized_market_title = excluded.normalized_market_title,
            release_date_distance = excluded.release_date_distance,
            title_similarity = excluded.title_similarity,
            semantic_compatibility = excluded.semantic_compatibility,
            match_score = excluded.match_score,
            match_method = excluded.match_method,
            match_status = CASE
                WHEN prediction_market_backtest.movie_matches.manual_override
                THEN prediction_market_backtest.movie_matches.match_status
                ELSE excluded.match_status
            END,
            rejection_reason = excluded.rejection_reason
        """,
        (
            event_id,
            match.movie_id,
            normalized_market_title,
            match.release_date_distance,
            match.score,
            json.dumps({"parser_version": PARSER_VERSION}),
            match.score,
            "gamma_title_release_match",
            match.status,
            "no_internal_movie_match" if match.movie_id is None else None,
        ),
    )


def token_ids_and_outcomes(market: dict[str, Any]) -> tuple[str | None, str | None, list[Any]]:
    raw_tokens = market.get("clobTokenIds") or market.get("clob_token_ids") or market.get("tokens") or []
    raw_outcomes = market.get("outcomes") or []
    tokens = parse_jsonish_list(raw_tokens)
    outcomes = parse_jsonish_list(raw_outcomes)
    yes = None
    no = None
    for token in tokens:
        if isinstance(token, dict):
            outcome = str(token.get("outcome") or "").lower()
            token_id = str(token.get("token_id") or token.get("id") or "") or None
            if outcome == "yes":
                yes = token_id
            elif outcome == "no":
                no = token_id
        elif yes is None:
            yes = str(token)
        elif no is None:
            no = str(token)
    if outcomes and len(tokens) >= len(outcomes):
        for outcome, token in zip(outcomes, tokens, strict=False):
            token_id = str(token.get("token_id") or token.get("id") or "") if isinstance(token, dict) else str(token)
            if str(outcome).lower() == "yes":
                yes = token_id
            if str(outcome).lower() == "no":
                no = token_id
    return yes, no, outcomes


def parse_jsonish_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            return [value] if value else []
    return []


def event_id_for(event: dict[str, Any]) -> str:
    return str(event.get("id") or event.get("event_id") or event.get("slug") or "")


def market_id_for(market: dict[str, Any]) -> str:
    return str(market.get("id") or market.get("market_id") or market.get("conditionId") or "")


def event_rules(event: dict[str, Any]) -> str:
    parts = [
        event.get("description"),
        event.get("resolutionSource"),
        event.get("resolutionRules"),
        event.get("resolution_rules"),
    ]
    parts.extend(market.get("description") for market in event.get("markets") or [] if isinstance(market, dict))
    return " ".join(str(part or "") for part in parts)


def timestamp_value(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


def numeric_value(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=database_url_from_env())
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--slug", help="Sync one exact Gamma event slug")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--focus", choices=("box_office", "movies"), default="box_office")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = PolymarketClient(args.cache_dir, refresh=args.refresh)
    events = discover_events(client, slug=args.slug, limit=args.limit, focus=args.focus)
    conn = connect_database(args.database_url)
    try:
        summary = sync_metadata(conn, events, dry_run=args.dry_run)
        if args.dry_run:
            conn.rollback()
        else:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
