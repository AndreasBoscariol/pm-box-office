#!/usr/bin/env python3
"""Find Polymarket accounts that mostly trade movies or box-office markets.

The script uses public Polymarket APIs:

* Gamma API for tags, market discovery, and public profile metadata.
* Data API for market trades and user activity.

It writes a CSV and HTML visual ranked by realized profit. By default, an
account qualifies when at least 90% of historical closed positions are in
box-office markets within Polymarket's ``movies`` tag.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
from html import escape
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
DATA_BASE_URL = "https://data-api.polymarket.com"
MOVIES_TAG_SLUG = "movies"

MARKETS_LIMIT = 100
TRADES_LIMIT = 10_000
ACTIVITY_LIMIT = 500
MAX_DATA_API_OFFSET = 10_000
CLOSED_POSITIONS_LIMIT = 50
MAX_CLOSED_POSITIONS_OFFSET = 100_000
SCORE_SCHEMA_VERSION = 2

BOX_OFFICE_TERMS = (
    "box office",
    "opening weekend",
    "domestic gross",
    "worldwide gross",
    "first weekend",
)


JsonObject = dict[str, Any]


class RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self.lock = threading.Lock()
        self.next_allowed_at = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_allowed_at - now)
            self.next_allowed_at = max(now, self.next_allowed_at) + self.min_interval
        if delay:
            time.sleep(delay)


class HttpRequestError(RuntimeError):
    def __init__(self, url: str, status: int | None, body: str, original: Exception) -> None:
        self.url = url
        self.status = status
        self.body = body
        self.original = original
        super().__init__(f"GET {url} failed: status={status} body={body!r}")


@dataclass(frozen=True)
class MovieMarket:
    condition_id: str
    market_id: str
    question: str
    slug: str
    closed: bool
    description: str


@dataclass
class CandidateSeed:
    movie_trade_count: int = 0
    movie_volume: float = 0.0


@dataclass
class AccountScore:
    wallet: str
    has_x_profile: bool
    x_username: str
    name: str
    pseudonym: str
    bio: str
    verified_badge: bool
    profile_image: str
    focus_trade_volume: float
    total_trade_volume: float
    focus_trade_volume_share: float
    focus_trade_count: int
    total_trade_count: int
    focus_position_count: int
    total_position_count: int
    focus_position_share: float
    focus_position_volume: float
    total_position_volume: float
    focus_profit: float
    total_profit: float
    focus_position_sharpe: float
    total_position_sharpe: float
    focus_avg_roi: float
    total_avg_roi: float
    focus_roi_stddev: float
    total_roi_stddev: float
    first_trade_ts: int | None
    last_trade_ts: int | None
    top_focus_markets: str
    activity_truncated: bool
    closed_positions_truncated: bool


@dataclass
class AccountSearchResult:
    qualifying_scores: list[AccountScore]
    all_scores: list[AccountScore]
    warnings: list[str]
    status_counts: Counter[str]


class PolymarketClient:
    def __init__(
        self,
        cache_dir: Path,
        *,
        refresh: bool = False,
        sleep_seconds: float = 0.1,
        timeout_seconds: float = 30.0,
        retries: int = 4,
        rates: dict[str, float] | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.refresh = refresh
        self.sleep_seconds = sleep_seconds
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.rate_limiters = {
            name: RateLimiter(rate)
            for name, rate in (rates or {}).items()
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_json(self, base_url: str, path: str, params: dict[str, Any] | None = None) -> Any:
        query = urllib.parse.urlencode(
            {
                key: value
                for key, value in (params or {}).items()
                if value is not None
            },
            doseq=True,
        )
        url = f"{base_url}{path}"
        if query:
            url = f"{url}?{query}"

        cache_path = self._cache_path(url)
        if cache_path.exists() and not self.refresh:
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cache_path.unlink(missing_ok=True)

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            if self.sleep_seconds:
                time.sleep(self.sleep_seconds)
            try:
                self._wait_for_rate_limit(base_url, path)
                request = urllib.request.Request(
                    url,
                    headers={
                        "Accept": "application/json",
                        "User-Agent": "pm-box-office-polymarket-movie-accounts/1.0",
                    },
                )
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = response.read().decode("utf-8")
                data = json.loads(payload)
                tmp_path = cache_path.with_name(
                    f"{cache_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
                )
                tmp_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                tmp_path.replace(cache_path)
                return data
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    return None
                body = exc.read().decode("utf-8", errors="replace")
                last_error = exc
                if exc.code not in (429, 500, 502, 503, 504) or attempt == self.retries:
                    raise HttpRequestError(url, exc.code, body, exc)
                retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else 2**attempt
                time.sleep(delay)
            except (TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt == self.retries:
                    break
                time.sleep(2**attempt)

        raise RuntimeError(f"GET {url} failed after retries: {last_error}")

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _wait_for_rate_limit(self, base_url: str, path: str) -> None:
        bucket = self._rate_bucket(base_url, path)
        limiter = self.rate_limiters.get(bucket)
        if limiter:
            limiter.wait()

    @staticmethod
    def _rate_bucket(base_url: str, path: str) -> str:
        if base_url == GAMMA_BASE_URL:
            if path.startswith("/markets"):
                return "gamma_markets"
            if path.startswith("/tags"):
                return "gamma_tags"
            if path.startswith("/public-profile"):
                return "gamma_profile"
            return "gamma_general"
        if base_url == DATA_BASE_URL:
            if path.startswith("/trades"):
                return "data_trades"
            if path.startswith("/closed-positions"):
                return "data_closed_positions"
            if path.startswith("/positions"):
                return "data_positions"
            if path.startswith("/activity"):
                return "data_activity"
            return "data_general"
        return "general"


def discover_movie_tag_id(client: PolymarketClient) -> int:
    tag = client.get_json(GAMMA_BASE_URL, f"/tags/slug/{MOVIES_TAG_SLUG}")
    if not isinstance(tag, dict) or tag.get("slug") != MOVIES_TAG_SLUG:
        raise RuntimeError("Could not confirm Polymarket movies tag")
    return int(tag["id"])


def market_has_movies_tag(market: JsonObject) -> bool:
    return any(tag.get("slug") == MOVIES_TAG_SLUG for tag in market.get("tags") or [])


def market_text(market: JsonObject) -> str:
    fields = [
        market.get("question"),
        market.get("slug"),
        market.get("description"),
        market.get("groupItemTitle"),
    ]
    for event in market.get("events") or []:
        fields.extend([event.get("title"), event.get("slug"), event.get("description")])
    return " ".join(str(field or "") for field in fields).lower()


def market_matches_focus(market: JsonObject, focus: str) -> bool:
    if not market_has_movies_tag(market):
        return False
    if focus == "movies":
        return True

    text = market_text(market)
    if any(term in text for term in BOX_OFFICE_TERMS):
        return True
    if "gross" in text and any(term in text for term in ("domestic", "worldwide", "movie", "film")):
        return True
    if "$" in text and any(term in text for term in ("earn", "make", "revenue", "gross")):
        return True
    return False


def discover_movie_markets(
    client: PolymarketClient,
    *,
    tag_id: int,
    focus: str,
    max_markets: int | None = None,
) -> list[MovieMarket]:
    by_condition_id: dict[str, MovieMarket] = {}
    for closed in (False, True):
        cursor = None
        while True:
            payload = client.get_json(
                GAMMA_BASE_URL,
                "/markets/keyset",
                {
                    "tag_id": tag_id,
                    "include_tag": "true",
                    "limit": MARKETS_LIMIT,
                    "closed": str(closed).lower(),
                    "after_cursor": cursor,
                },
            )
            markets = payload.get("markets", []) if isinstance(payload, dict) else []
            for market in markets:
                condition_id = market.get("conditionId")
                if not condition_id or not market_matches_focus(market, focus):
                    continue
                by_condition_id[condition_id] = MovieMarket(
                    condition_id=condition_id,
                    market_id=str(market.get("id") or ""),
                    question=str(market.get("question") or ""),
                    slug=str(market.get("slug") or ""),
                    closed=bool(market.get("closed")),
                    description=str(market.get("description") or ""),
                )
                if max_markets and len(by_condition_id) >= max_markets:
                    return list(by_condition_id.values())
            cursor = payload.get("next_cursor") if isinstance(payload, dict) else None
            if not cursor:
                break
    return list(by_condition_id.values())


def notional(row: JsonObject) -> float:
    for key in ("usdcSize", "cashSize"):
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    return float(row.get("size") or 0.0) * float(row.get("price") or 0.0)


def fetch_market_trades(
    client: PolymarketClient,
    market: MovieMarket,
) -> tuple[str, list[JsonObject], str | None]:
    trades = client.get_json(
        DATA_BASE_URL,
        "/trades",
        {
            "market": market.condition_id,
            "takerOnly": "false",
            "limit": TRADES_LIMIT,
            "offset": 0,
        },
    )
    if not isinstance(trades, list):
        return market.condition_id, [], f"{market.condition_id}: unexpected trades response"
    warning = None
    if len(trades) >= TRADES_LIMIT:
        warning = f"{market.condition_id}: trade seed may be truncated at {TRADES_LIMIT} rows"
    return market.condition_id, trades, warning


def collect_candidate_wallets(
    client: PolymarketClient,
    markets: list[MovieMarket],
    *,
    workers: int,
) -> tuple[dict[str, CandidateSeed], list[str]]:
    candidates: dict[str, CandidateSeed] = defaultdict(CandidateSeed)
    warnings: list[str] = []

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_to_market = {
            executor.submit(fetch_market_trades, client, market): market
            for market in markets
        }
        completed = 0
        for future in as_completed(future_to_market):
            market = future_to_market[future]
            completed += 1
            print(
                f"[{completed}/{len(markets)}] Fetched trades for {market.slug or market.condition_id}",
                file=sys.stderr,
            )
            _condition_id, trades, warning = future.result()
            if warning:
                warnings.append(warning)
            for trade in trades:
                wallet = normalize_address(trade.get("proxyWallet"))
                if not wallet:
                    continue
                candidates[wallet].movie_trade_count += 1
                candidates[wallet].movie_volume += notional(trade)

    return dict(candidates), warnings


def score_to_record(score: AccountScore) -> JsonObject:
    return asdict(score)


def score_from_record(record: JsonObject) -> AccountScore:
    return AccountScore(
        wallet=str(record.get("wallet") or ""),
        has_x_profile=bool(record.get("has_x_profile", record.get("x_username"))),
        x_username=str(record.get("x_username") or ""),
        name=str(record.get("name") or ""),
        pseudonym=str(record.get("pseudonym") or ""),
        bio=str(record.get("bio") or ""),
        verified_badge=bool(record.get("verified_badge")),
        profile_image=str(record.get("profile_image") or ""),
        focus_trade_volume=float(record.get("focus_trade_volume") or 0.0),
        total_trade_volume=float(record.get("total_trade_volume") or 0.0),
        focus_trade_volume_share=float(record.get("focus_trade_volume_share") or 0.0),
        focus_trade_count=int(record.get("focus_trade_count") or 0),
        total_trade_count=int(record.get("total_trade_count") or 0),
        focus_position_count=int(record.get("focus_position_count") or 0),
        total_position_count=int(record.get("total_position_count") or 0),
        focus_position_share=float(record.get("focus_position_share") or 0.0),
        focus_position_volume=float(record.get("focus_position_volume") or 0.0),
        total_position_volume=float(record.get("total_position_volume") or 0.0),
        focus_profit=float(record.get("focus_profit") or 0.0),
        total_profit=float(record.get("total_profit") or 0.0),
        focus_position_sharpe=float(record.get("focus_position_sharpe") or 0.0),
        total_position_sharpe=float(record.get("total_position_sharpe") or 0.0),
        focus_avg_roi=float(record.get("focus_avg_roi") or 0.0),
        total_avg_roi=float(record.get("total_avg_roi") or 0.0),
        focus_roi_stddev=float(record.get("focus_roi_stddev") or 0.0),
        total_roi_stddev=float(record.get("total_roi_stddev") or 0.0),
        first_trade_ts=record.get("first_trade_ts"),
        last_trade_ts=record.get("last_trade_ts"),
        top_focus_markets=str(record.get("top_focus_markets") or ""),
        activity_truncated=bool(record.get("activity_truncated")),
        closed_positions_truncated=bool(record.get("closed_positions_truncated")),
    )


def qualifies(score: AccountScore, args: argparse.Namespace) -> bool:
    return not qualification_reasons(
        score,
        min_focus_position_share=args.min_focus_position_share,
        min_focus_positions=args.min_focus_positions,
    )


def qualification_reasons(
    score: AccountScore,
    *,
    min_focus_position_share: float,
    min_focus_positions: int,
) -> list[str]:
    reasons: list[str] = []
    if score.total_position_count <= 0:
        reasons.append("no closed positions returned")
    if score.focus_position_share < min_focus_position_share:
        reasons.append(
            f"focus_position_share actual {pct(score.focus_position_share)}, required {pct(min_focus_position_share)}"
        )
    if score.focus_position_count < min_focus_positions:
        reasons.append(
            f"focus_position_count {score.focus_position_count} below required {min_focus_positions}"
        )
    return reasons


def qualification_notes(score: AccountScore) -> list[str]:
    notes: list[str] = []
    if score.closed_positions_truncated:
        notes.append("closed positions truncated by API/page cap")
    if score.activity_truncated:
        notes.append("activity truncated by API/page cap")
    if score.total_trade_count > 0 and score.focus_trade_count == 0:
        notes.append("no focus trades found in sampled activity")
    if score.top_focus_markets == "":
        notes.append("no focus market titles found")
    return notes


def sort_key(score: AccountScore, sort_by: str) -> tuple[float, float, float]:
    keys = {
        "total-profit": score.total_profit,
        "focus-profit": score.focus_profit,
        "focus-position-share": score.focus_position_share,
        "focus-position-count": float(score.focus_position_count),
        "focus-position-volume": score.focus_position_volume,
        "focus-position-sharpe": score.focus_position_sharpe,
        "total-position-sharpe": score.total_position_sharpe,
        "focus-trade-volume": score.focus_trade_volume,
        "focus-trade-volume-share": score.focus_trade_volume_share,
        "total-position-count": float(score.total_position_count),
        "total-trade-volume": score.total_trade_volume,
    }
    primary = keys.get(sort_by, score.total_profit)
    return (primary, score.focus_position_share, score.focus_position_count)


def sorted_scores(rows: list[AccountScore], sort_by: str, *, reverse: bool = True) -> list[AccountScore]:
    return sorted(rows, key=lambda row: sort_key(row, sort_by), reverse=reverse)


def checkpoint_path(args: argparse.Namespace) -> Path:
    return args.checkpoint or args.output.with_suffix(args.output.suffix + ".checkpoint.jsonl")


def checkpoint_run_key_for(
    focus_condition_ids: set[str],
    args: argparse.Namespace,
    *,
    linked_x_only: bool,
) -> str:
    payload = {
        "focus": args.market_focus,
        "conditions": sorted(focus_condition_ids),
        "max_position_pages": args.max_position_pages,
        "linked_x_only": linked_x_only,
        "score_schema_version": SCORE_SCHEMA_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def checkpoint_run_key(focus_condition_ids: set[str], args: argparse.Namespace) -> str:
    # The legacy CLI flag now only controls the initial HTML filter, not which
    # wallets are scored, so the main checkpoint key always represents all
    # candidate wallets.
    return checkpoint_run_key_for(focus_condition_ids, args, linked_x_only=False)


def legacy_scored_checkpoint_keys(focus_condition_ids: set[str], args: argparse.Namespace) -> set[str]:
    return {
        checkpoint_run_key_for(focus_condition_ids, args, linked_x_only=True),
    }


def load_checkpoint(
    path: Path,
    run_key: str,
    *,
    scored_only_run_keys: set[str] | None = None,
) -> tuple[dict[str, str], dict[str, AccountScore]]:
    processed: dict[str, str] = {}
    scored: dict[str, AccountScore] = {}
    if not path.exists():
        return processed, scored

    scored_only_run_keys = scored_only_run_keys or set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid checkpoint line {line_number}: {path}", file=sys.stderr)
                continue
            record_run_key = record.get("run_key")
            if record_run_key != run_key and record_run_key not in scored_only_run_keys:
                continue
            wallet = normalize_address(record.get("wallet"))
            if not wallet:
                continue
            status = str(record.get("status") or "processed")
            if record_run_key in scored_only_run_keys and status != "scored":
                continue
            processed[wallet] = status
            if status == "scored" and isinstance(record.get("score"), dict):
                scored[wallet] = score_from_record(record["score"])
    return processed, scored


def append_checkpoint(
    path: Path,
    lock: threading.Lock,
    *,
    run_key: str,
    wallet: str,
    status: str,
    score: AccountScore | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record: JsonObject = {
        "run_key": run_key,
        "wallet": wallet,
        "status": status,
        "timestamp": int(time.time()),
    }
    if score is not None:
        record["score"] = score_to_record(score)
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def normalize_address(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip().lower()
    if len(value) == 42 and value.startswith("0x"):
        return value
    return ""


def get_public_profile(client: PolymarketClient, wallet: str) -> JsonObject:
    profile = client.get_json(GAMMA_BASE_URL, "/public-profile", {"address": wallet})
    if not isinstance(profile, dict):
        profile = {}
    profile.setdefault("proxyWallet", wallet)
    return profile


def fetch_user_activity(client: PolymarketClient, wallet: str) -> tuple[list[JsonObject], bool]:
    rows: list[JsonObject] = []
    offset = 0
    truncated = False

    while offset <= MAX_DATA_API_OFFSET:
        try:
            page = client.get_json(
                DATA_BASE_URL,
                "/activity",
                {
                    "user": wallet,
                    "type": "TRADE",
                    "start": 1,
                    "limit": ACTIVITY_LIMIT,
                    "offset": offset,
                    "sortBy": "TIMESTAMP",
                    "sortDirection": "DESC",
                },
            )
        except HttpRequestError as exc:
            if exc.status == 400 and rows:
                return rows, True
            raise
        if not isinstance(page, list):
            raise RuntimeError(f"Unexpected activity response for {wallet}")
        rows.extend(page)
        if len(page) < ACTIVITY_LIMIT:
            return rows, truncated
        offset += ACTIVITY_LIMIT

    truncated = True
    return rows, truncated


def fetch_closed_positions(
    client: PolymarketClient,
    wallet: str,
    *,
    max_pages: int | None = None,
) -> tuple[list[JsonObject], bool]:
    rows: list[JsonObject] = []
    offset = 0
    pages = 0

    while offset <= MAX_CLOSED_POSITIONS_OFFSET:
        try:
            page = client.get_json(
                DATA_BASE_URL,
                "/closed-positions",
                {
                    "user": wallet,
                    "limit": CLOSED_POSITIONS_LIMIT,
                    "offset": offset,
                    "sortBy": "REALIZEDPNL",
                    "sortDirection": "DESC",
                },
            )
        except HttpRequestError as exc:
            if exc.status == 400 and rows:
                return rows, True
            raise
        if not isinstance(page, list):
            raise RuntimeError(f"Unexpected closed positions response for {wallet}")
        rows.extend(page)
        pages += 1
        if len(page) < CLOSED_POSITIONS_LIMIT:
            return rows, False
        if max_pages and pages >= max_pages:
            return rows, True
        offset += CLOSED_POSITIONS_LIMIT

    return rows, True


def position_volume(row: JsonObject) -> float:
    value = row.get("totalBought")
    if value not in (None, ""):
        return float(value)
    return 0.0


def realized_profit(row: JsonObject) -> float:
    value = row.get("realizedPnl")
    if value not in (None, ""):
        return float(value)
    return 0.0


def roi_stats(returns: list[float]) -> tuple[float, float, float]:
    if not returns:
        return 0.0, 0.0, 0.0
    avg = sum(returns) / len(returns)
    if len(returns) < 2:
        return avg, 0.0, 0.0
    variance = sum((value - avg) ** 2 for value in returns) / (len(returns) - 1)
    stddev = math.sqrt(variance)
    if stddev <= 0.0:
        return avg, stddev, 0.0
    return avg, stddev, (avg / stddev) * math.sqrt(len(returns))


def score_account(
    activity: list[JsonObject],
    closed_positions: list[JsonObject],
    profile: JsonObject,
    focus_condition_ids: set[str],
    *,
    wallet_hint: str = "",
) -> AccountScore:
    wallet = (
        normalize_address(profile.get("proxyWallet"))
        or normalize_address(activity[0].get("proxyWallet") if activity else "")
        or normalize_address(wallet_hint)
    )
    x_username = str(profile.get("xUsername") or "").strip()
    total_trade_volume = 0.0
    focus_trade_volume = 0.0
    total_trade_count = 0
    focus_trade_count = 0
    timestamps: list[int] = []
    focus_trade_market_volumes: Counter[str] = Counter()

    for row in activity:
        if row.get("type") != "TRADE":
            continue
        volume = notional(row)
        total_trade_volume += volume
        total_trade_count += 1
        timestamp = row.get("timestamp")
        if isinstance(timestamp, int):
            timestamps.append(timestamp)
        if row.get("conditionId") in focus_condition_ids:
            focus_trade_volume += volume
            focus_trade_count += 1
            title = str(row.get("title") or row.get("slug") or row.get("conditionId") or "")
            focus_trade_market_volumes[title] += volume

    focus_position_count = 0
    total_position_count = 0
    focus_position_volume = 0.0
    total_position_volume = 0.0
    focus_profit = 0.0
    total_profit = 0.0
    focus_position_returns: list[float] = []
    total_position_returns: list[float] = []
    focus_position_market_profits: Counter[str] = Counter()

    for row in closed_positions:
        total_position_count += 1
        volume = position_volume(row)
        profit = realized_profit(row)
        total_position_volume += volume
        total_profit += profit
        if volume > 0.0:
            total_position_returns.append(profit / volume)
        if row.get("conditionId") in focus_condition_ids:
            focus_position_count += 1
            focus_position_volume += volume
            focus_profit += profit
            if volume > 0.0:
                focus_position_returns.append(profit / volume)
            title = str(row.get("title") or row.get("slug") or row.get("conditionId") or "")
            focus_position_market_profits[title] += profit

    top_focus_markets = "; ".join(
        f"{title} ({profit:.2f})" for title, profit in focus_position_market_profits.most_common(3)
    )
    if not top_focus_markets:
        top_focus_markets = "; ".join(
            f"{title} ({volume:.2f})" for title, volume in focus_trade_market_volumes.most_common(3)
        )

    focus_avg_roi, focus_roi_stddev, focus_position_sharpe = roi_stats(focus_position_returns)
    total_avg_roi, total_roi_stddev, total_position_sharpe = roi_stats(total_position_returns)

    return AccountScore(
        wallet=wallet,
        has_x_profile=bool(x_username),
        x_username=x_username,
        name=str(profile.get("name") or ""),
        pseudonym=str(profile.get("pseudonym") or ""),
        bio=str(profile.get("bio") or ""),
        verified_badge=bool(profile.get("verifiedBadge")),
        profile_image=str(profile.get("profileImage") or ""),
        focus_trade_volume=focus_trade_volume,
        total_trade_volume=total_trade_volume,
        focus_trade_volume_share=(focus_trade_volume / total_trade_volume if total_trade_volume else 0.0),
        focus_trade_count=focus_trade_count,
        total_trade_count=total_trade_count,
        focus_position_count=focus_position_count,
        total_position_count=total_position_count,
        focus_position_share=(
            focus_position_count / total_position_count if total_position_count else 0.0
        ),
        focus_position_volume=focus_position_volume,
        total_position_volume=total_position_volume,
        focus_profit=focus_profit,
        total_profit=total_profit,
        focus_position_sharpe=focus_position_sharpe,
        total_position_sharpe=total_position_sharpe,
        focus_avg_roi=focus_avg_roi,
        total_avg_roi=total_avg_roi,
        focus_roi_stddev=focus_roi_stddev,
        total_roi_stddev=total_roi_stddev,
        first_trade_ts=min(timestamps) if timestamps else None,
        last_trade_ts=max(timestamps) if timestamps else None,
        top_focus_markets=top_focus_markets,
        activity_truncated=False,
        closed_positions_truncated=False,
    )


def build_rate_limits(args: argparse.Namespace) -> dict[str, float]:
    return {
        "gamma_markets": args.gamma_markets_rps,
        "gamma_tags": args.gamma_tags_rps,
        "gamma_profile": args.gamma_profile_rps,
        "gamma_general": args.gamma_general_rps,
        "data_trades": args.data_trades_rps,
        "data_activity": args.data_activity_rps,
        "data_closed_positions": args.data_closed_positions_rps,
        "data_positions": args.data_positions_rps,
        "data_general": args.data_general_rps,
    }


def process_wallet(
    client: PolymarketClient,
    wallet: str,
    focus_condition_ids: set[str],
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    checkpoint_lock: threading.Lock,
    run_key: str,
) -> tuple[str, AccountScore | None, str, str | None]:
    try:
        profile = get_public_profile(client, wallet)
        activity, truncated = fetch_user_activity(client, wallet)
        closed_positions, closed_truncated = fetch_closed_positions(
            client,
            wallet,
            max_pages=args.max_position_pages,
        )
        score = score_account(activity, closed_positions, profile, focus_condition_ids, wallet_hint=wallet)
        score.wallet = wallet
        score.activity_truncated = truncated
        score.closed_positions_truncated = closed_truncated
        append_checkpoint(
            checkpoint,
            checkpoint_lock,
            run_key=run_key,
            wallet=wallet,
            status="scored",
            score=score,
        )
        return wallet, score, "scored", None
    except Exception as exc:  # noqa: BLE001 - keep other wallets moving.
        return wallet, None, "error", f"{wallet}: {exc}"


def find_accounts(args: argparse.Namespace) -> AccountSearchResult:
    client = PolymarketClient(
        args.cache_dir,
        refresh=args.refresh,
        sleep_seconds=args.sleep,
        timeout_seconds=args.timeout,
        retries=args.retries,
        rates=build_rate_limits(args),
    )

    tag_id = discover_movie_tag_id(client)
    print(f"Confirmed movies tag id={tag_id}", file=sys.stderr)

    markets = discover_movie_markets(
        client,
        tag_id=tag_id,
        focus=args.market_focus,
        max_markets=args.max_markets,
    )
    if not markets:
        raise RuntimeError(f"No {args.market_focus} markets found")
    print(f"Discovered {len(markets)} {args.market_focus} markets", file=sys.stderr)

    candidates, warnings = collect_candidate_wallets(
        client,
        markets,
        workers=args.market_workers,
    )
    sorted_wallets = sorted(
        candidates,
        key=lambda wallet: candidates[wallet].movie_volume,
        reverse=True,
    )
    if args.max_candidates:
        sorted_wallets = sorted_wallets[: args.max_candidates]
    focus_condition_ids = {market.condition_id for market in markets}
    checkpoint = checkpoint_path(args)
    run_key = checkpoint_run_key(focus_condition_ids, args)
    if args.reset_checkpoint and checkpoint.exists():
        checkpoint.unlink()
    processed, checkpoint_scores = load_checkpoint(
        checkpoint,
        run_key,
        scored_only_run_keys=legacy_scored_checkpoint_keys(focus_condition_ids, args),
    )
    status_counts = Counter(processed.values())
    all_scores = list(checkpoint_scores.values())
    wallets_to_score = [wallet for wallet in sorted_wallets if wallet not in processed]

    print(
        f"Scoring {len(wallets_to_score)} candidate wallets "
        f"({len(processed)} resumed from checkpoint)",
        file=sys.stderr,
    )
    print(f"Checkpoint: {checkpoint}", file=sys.stderr)

    checkpoint_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max(1, args.account_workers)) as executor:
        future_to_wallet = {
            executor.submit(
                process_wallet,
                client,
                wallet,
                focus_condition_ids,
                args,
                checkpoint=checkpoint,
                checkpoint_lock=checkpoint_lock,
                run_key=run_key,
            ): wallet
            for wallet in wallets_to_score
        }
        completed = 0
        for future in as_completed(future_to_wallet):
            completed += 1
            wallet, score, status, warning = future.result()
            status_counts[status] += 1
            if warning:
                warnings.append(warning)
            if score is not None:
                all_scores.append(score)
            print(
                f"[{completed}/{len(wallets_to_score)}] Processed {wallet}",
                file=sys.stderr,
            )

    qualifying_scores = [score for score in all_scores if qualifies(score, args)]
    qualifying_scores = sorted_scores(qualifying_scores, args.sort_by)
    if args.top_n:
        qualifying_scores = qualifying_scores[: args.top_n]
    all_scores = sorted_scores(all_scores, args.sort_by)
    return AccountSearchResult(
        qualifying_scores=qualifying_scores,
        all_scores=all_scores,
        warnings=warnings,
        status_counts=status_counts,
    )


def write_scores_csv(path: Path, rows: list[AccountScore]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "wallet",
        "has_x_profile",
        "xUsername",
        "name",
        "pseudonym",
        "bio",
        "verifiedBadge",
        "profileImage",
        "focus_trade_volume",
        "total_trade_volume",
        "focus_trade_volume_share",
        "focus_trade_count",
        "total_trade_count",
        "focus_position_count",
        "total_position_count",
        "focus_position_share",
        "focus_position_volume",
        "total_position_volume",
        "focus_profit",
        "total_profit",
        "focus_position_sharpe",
        "total_position_sharpe",
        "focus_avg_roi",
        "total_avg_roi",
        "focus_roi_stddev",
        "total_roi_stddev",
        "first_trade_ts",
        "last_trade_ts",
        "top_focus_markets",
        "activity_truncated",
        "closed_positions_truncated",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "wallet": row.wallet,
                    "has_x_profile": row.has_x_profile,
                    "xUsername": row.x_username,
                    "name": row.name,
                    "pseudonym": row.pseudonym,
                    "bio": row.bio,
                    "verifiedBadge": row.verified_badge,
                    "profileImage": row.profile_image,
                    "focus_trade_volume": f"{row.focus_trade_volume:.6f}",
                    "total_trade_volume": f"{row.total_trade_volume:.6f}",
                    "focus_trade_volume_share": f"{row.focus_trade_volume_share:.6f}",
                    "focus_trade_count": row.focus_trade_count,
                    "total_trade_count": row.total_trade_count,
                    "focus_position_count": row.focus_position_count,
                    "total_position_count": row.total_position_count,
                    "focus_position_share": f"{row.focus_position_share:.6f}",
                    "focus_position_volume": f"{row.focus_position_volume:.6f}",
                    "total_position_volume": f"{row.total_position_volume:.6f}",
                    "focus_profit": f"{row.focus_profit:.6f}",
                    "total_profit": f"{row.total_profit:.6f}",
                    "focus_position_sharpe": f"{row.focus_position_sharpe:.6f}",
                    "total_position_sharpe": f"{row.total_position_sharpe:.6f}",
                    "focus_avg_roi": f"{row.focus_avg_roi:.6f}",
                    "total_avg_roi": f"{row.total_avg_roi:.6f}",
                    "focus_roi_stddev": f"{row.focus_roi_stddev:.6f}",
                    "total_roi_stddev": f"{row.total_roi_stddev:.6f}",
                    "first_trade_ts": row.first_trade_ts or "",
                    "last_trade_ts": row.last_trade_ts or "",
                    "top_focus_markets": row.top_focus_markets,
                    "activity_truncated": row.activity_truncated,
                    "closed_positions_truncated": row.closed_positions_truncated,
                }
            )


def default_diagnostics_output(output: Path) -> Path:
    suffix = output.suffix or ".csv"
    return output.with_name(f"{output.stem}_diagnostics{suffix}")


def default_all_output(output: Path) -> Path:
    suffix = output.suffix or ".csv"
    return output.with_name(f"{output.stem}_all_scored{suffix}")


def write_diagnostics_csv(path: Path, rows: list[AccountScore], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "qualified_at_run_threshold",
        "disqualification_reasons",
        "notes",
        "wallet",
        "has_x_profile",
        "xUsername",
        "name",
        "pseudonym",
        "focus_position_share",
        "required_focus_position_share",
        "focus_position_share_gap",
        "focus_position_count",
        "required_focus_positions",
        "focus_position_count_gap",
        "total_position_count",
        "focus_position_volume",
        "total_position_volume",
        "focus_profit",
        "total_profit",
        "focus_position_sharpe",
        "total_position_sharpe",
        "focus_avg_roi",
        "total_avg_roi",
        "focus_roi_stddev",
        "total_roi_stddev",
        "focus_trade_volume_share",
        "focus_trade_count",
        "total_trade_count",
        "top_focus_markets",
        "activity_truncated",
        "closed_positions_truncated",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            reasons = qualification_reasons(
                row,
                min_focus_position_share=args.min_focus_position_share,
                min_focus_positions=args.min_focus_positions,
            )
            writer.writerow(
                {
                    "qualified_at_run_threshold": not reasons,
                    "disqualification_reasons": "; ".join(reasons),
                    "notes": "; ".join(qualification_notes(row)),
                    "wallet": row.wallet,
                    "has_x_profile": row.has_x_profile,
                    "xUsername": row.x_username,
                    "name": row.name,
                    "pseudonym": row.pseudonym,
                    "focus_position_share": f"{row.focus_position_share:.6f}",
                    "required_focus_position_share": f"{args.min_focus_position_share:.6f}",
                    "focus_position_share_gap": f"{max(0.0, args.min_focus_position_share - row.focus_position_share):.6f}",
                    "focus_position_count": row.focus_position_count,
                    "required_focus_positions": args.min_focus_positions,
                    "focus_position_count_gap": max(0, args.min_focus_positions - row.focus_position_count),
                    "total_position_count": row.total_position_count,
                    "focus_position_volume": f"{row.focus_position_volume:.6f}",
                    "total_position_volume": f"{row.total_position_volume:.6f}",
                    "focus_profit": f"{row.focus_profit:.6f}",
                    "total_profit": f"{row.total_profit:.6f}",
                    "focus_position_sharpe": f"{row.focus_position_sharpe:.6f}",
                    "total_position_sharpe": f"{row.total_position_sharpe:.6f}",
                    "focus_avg_roi": f"{row.focus_avg_roi:.6f}",
                    "total_avg_roi": f"{row.total_avg_roi:.6f}",
                    "focus_roi_stddev": f"{row.focus_roi_stddev:.6f}",
                    "total_roi_stddev": f"{row.total_roi_stddev:.6f}",
                    "focus_trade_volume_share": f"{row.focus_trade_volume_share:.6f}",
                    "focus_trade_count": row.focus_trade_count,
                    "total_trade_count": row.total_trade_count,
                    "top_focus_markets": row.top_focus_markets,
                    "activity_truncated": row.activity_truncated,
                    "closed_positions_truncated": row.closed_positions_truncated,
                }
            )


def money(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_visual_html(
    path: Path,
    rows: list[AccountScore],
    *,
    focus: str,
    args: argparse.Namespace,
    status_counts: Counter[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data_rows = []
    for row in rows:
        handle = row.x_username.lstrip("@")
        x_url = f"https://x.com/{urllib.parse.quote(handle)}" if handle else ""
        display_name = row.name or row.pseudonym or handle or row.wallet[:10]
        data_rows.append(
            {
                "wallet": row.wallet,
                "hasXProfile": row.has_x_profile,
                "handle": handle,
                "xUrl": x_url,
                "displayName": display_name,
                "bio": row.bio,
                "verified": row.verified_badge,
                "focusPositionShare": row.focus_position_share,
                "focusPositionCount": row.focus_position_count,
                "totalPositionCount": row.total_position_count,
                "focusPositionVolume": row.focus_position_volume,
                "totalPositionVolume": row.total_position_volume,
                "focusProfit": row.focus_profit,
                "totalProfit": row.total_profit,
                "focusPositionSharpe": row.focus_position_sharpe,
                "totalPositionSharpe": row.total_position_sharpe,
                "focusAvgRoi": row.focus_avg_roi,
                "totalAvgRoi": row.total_avg_roi,
                "focusRoiStddev": row.focus_roi_stddev,
                "totalRoiStddev": row.total_roi_stddev,
                "focusTradeVolumeShare": row.focus_trade_volume_share,
                "focusTradeCount": row.focus_trade_count,
                "totalTradeCount": row.total_trade_count,
                "focusTradeVolume": row.focus_trade_volume,
                "totalTradeVolume": row.total_trade_volume,
                "topFocusMarkets": row.top_focus_markets,
                "activityTruncated": row.activity_truncated,
                "closedPositionsTruncated": row.closed_positions_truncated,
                "notes": qualification_notes(row),
            }
        )
    encoded_rows = json.dumps(data_rows, ensure_ascii=False).replace("<", "\\u003c")
    encoded_status = json.dumps(dict(status_counts), ensure_ascii=False).replace("<", "\\u003c")
    initial_x_profile_filter = "linked" if args.linked_x_only else "all"

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Polymarket {escape(focus)} account layer</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17201b;
      --muted: #68736d;
      --line: #dce3df;
      --paper: #f8faf8;
      --panel: #ffffff;
      --green: #16875d;
      --red: #b54545;
      --blue: #2d5f9a;
      --amber: #a76612;
    }}
    body {{
      margin: 0;
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--paper);
    }}
    header {{
      padding: 28px 32px 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 24px;
      letter-spacing: 0;
    }}
    .sub {{
      color: var(--muted);
      max-width: 900px;
    }}
    main {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 22px 24px 40px;
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: repeat(6, minmax(130px, 1fr));
      gap: 12px 14px;
      padding: 14px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-bottom: 14px;
    }}
    label {{
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }}
    input, select {{
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 4px 8px;
      color: var(--ink);
      background: #fff;
      font: inherit;
    }}
    input[type="range"] {{
      padding: 0;
    }}
    .check {{
      align-content: end;
      grid-template-columns: 20px 1fr;
      gap: 8px;
    }}
    .check input {{
      min-height: auto;
      width: 16px;
      height: 16px;
      align-self: center;
    }}
    .name {{
      color: var(--blue);
      font-weight: 700;
      text-decoration: none;
    }}
    .meta, .reason, .notes {{
      color: var(--muted);
      font-size: 12px;
    }}
    .wallet {{
      overflow-wrap: anywhere;
      word-break: break-word;
      font-family: ui-monospace, SFMono-Regular, "SF Mono", Consolas, "Liberation Mono", monospace;
    }}
    .reason.fail {{
      color: var(--red);
    }}
    .reason.pass {{
      color: var(--green);
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(5, minmax(130px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }}
    .summary div {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
    }}
    .summary strong {{
      display: block;
      font-size: 20px;
    }}
    .summary span {{
      color: var(--muted);
      font-size: 12px;
    }}
    .tableWrap {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      padding: 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      font-variant-numeric: tabular-nums;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      background: #f0f4f1;
    }}
    td a {{
      color: var(--blue);
      text-decoration: none;
      font-weight: 650;
    }}
    .num {{
      text-align: right;
      white-space: nowrap;
    }}
    .status {{
      display: inline-block;
      min-width: 70px;
      padding: 2px 6px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 700;
      text-align: center;
    }}
    .status.pass {{
      color: var(--green);
      background: #e7f5ee;
    }}
    .status.fail {{
      color: var(--red);
      background: #faecec;
    }}
    .empty {{
      padding: 22px;
      color: var(--muted);
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    @media (max-width: 760px) {{
      header {{ padding: 22px 18px 14px; }}
      main {{ padding: 16px 12px 28px; }}
      .toolbar, .summary {{ grid-template-columns: 1fr; }}
      .tableWrap {{ overflow-x: auto; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Polymarket {escape(focus)} account review</h1>
    <div class="sub">Every scored account is shown here, including wallets without linked X/Twitter profiles. Adjust thresholds and the X-profile filter below without rerunning the API scan.</div>
  </header>
  <main>
    <section class="summary" id="summary"></section>
    <section class="toolbar" aria-label="Account filters">
      <label>Min focus position share <input id="minShare" type="range" min="0" max="1" step="0.01" value="{args.min_focus_position_share:.2f}"><span id="minShareValue"></span></label>
      <label>Min focus positions <input id="minPositions" type="number" min="0" step="1" value="{args.min_focus_positions}"></label>
      <label>Min focus trade share <input id="minTradeShare" type="range" min="0" max="1" step="0.01" value="0"><span id="minTradeShareValue"></span></label>
      <label>Min focus trades <input id="minTrades" type="number" min="0" step="1" value="0"></label>
      <label>Search <input id="search" type="search" placeholder="handle, name, wallet, market"></label>
      <label>X profile
        <select id="xProfileFilter">
          <option value="all">All accounts</option>
          <option value="linked">Linked X only</option>
          <option value="unlinked">No linked X only</option>
        </select>
      </label>
      <label>Sort by
        <select id="sortBy">
          <option value="totalProfit">Total profit</option>
          <option value="focusProfit">Focus profit</option>
          <option value="focusPositionShare">Focus position share</option>
          <option value="focusPositionSharpe">Focus ROI Sharpe</option>
          <option value="totalPositionSharpe">Total ROI Sharpe</option>
          <option value="focusPositionCount">Focus position count</option>
          <option value="focusPositionVolume">Focus position volume</option>
          <option value="focusTradeVolumeShare">Focus trade share</option>
          <option value="focusTradeCount">Focus trade count</option>
          <option value="totalPositionCount">Total positions</option>
          <option value="totalTradeVolume">Total trade volume</option>
        </select>
      </label>
      <label>Direction
        <select id="sortDir">
          <option value="desc">Descending</option>
          <option value="asc">Ascending</option>
        </select>
      </label>
      <label class="check"><input id="showOnlyQualified" type="checkbox"> Only qualified</label>
      <label class="check"><input id="hideTruncated" type="checkbox"> Hide truncated</label>
    </section>
    <div class="tableWrap">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Status</th>
            <th>X / wallet</th>
            <th class="num">Total profit</th>
            <th class="num">Focus profit</th>
            <th class="num">Focus Sharpe</th>
            <th class="num">Focus positions</th>
            <th class="num">Focus trades</th>
            <th>Why</th>
            <th>Top focus markets</th>
          </tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
    <div id="empty" class="empty" hidden>No accounts match the current filters.</div>
  </main>
  <script>
    const accounts = {encoded_rows};
    const statusCounts = {encoded_status};
    const controls = {{
      minShare: document.getElementById("minShare"),
      minPositions: document.getElementById("minPositions"),
      minTradeShare: document.getElementById("minTradeShare"),
      minTrades: document.getElementById("minTrades"),
      search: document.getElementById("search"),
      xProfileFilter: document.getElementById("xProfileFilter"),
      sortBy: document.getElementById("sortBy"),
      sortDir: document.getElementById("sortDir"),
      showOnlyQualified: document.getElementById("showOnlyQualified"),
      hideTruncated: document.getElementById("hideTruncated"),
    }};
    const sortMap = {{
      "total-profit": "totalProfit",
      "focus-profit": "focusProfit",
      "focus-position-share": "focusPositionShare",
      "focus-position-sharpe": "focusPositionSharpe",
      "total-position-sharpe": "totalPositionSharpe",
      "focus-position-count": "focusPositionCount",
      "focus-position-volume": "focusPositionVolume",
      "focus-trade-volume": "focusTradeVolume",
      "focus-trade-volume-share": "focusTradeVolumeShare",
      "total-position-count": "totalPositionCount",
      "total-trade-volume": "totalTradeVolume",
    }};
    controls.sortBy.value = sortMap[{json.dumps(args.sort_by)}] || "totalProfit";
    controls.xProfileFilter.value = {json.dumps(initial_x_profile_filter)};

    function money(value) {{
      const abs = Math.abs(value || 0);
      return `${{value < 0 ? "-" : ""}}${{abs.toLocaleString(undefined, {{style: "currency", currency: "USD"}})}}`;
    }}
    function pct(value) {{
      return `${{((value || 0) * 100).toFixed(1)}}%`;
    }}
    function decimal(value) {{
      return Number(value || 0).toFixed(2);
    }}
    function text(value) {{
      return String(value ?? "").replace(/[&<>"']/g, char => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[char]));
    }}
    function reasons(account, filters) {{
      const output = [];
      if (account.totalPositionCount <= 0) output.push("no closed positions returned");
      if (account.focusPositionShare < filters.minShare) output.push(`focus position share actual ${{pct(account.focusPositionShare)}}, required ${{pct(filters.minShare)}}`);
      if (account.focusPositionCount < filters.minPositions) output.push(`focus position count ${{account.focusPositionCount}} below ${{filters.minPositions}}`);
      if (account.focusTradeVolumeShare < filters.minTradeShare) output.push(`focus trade share ${{pct(account.focusTradeVolumeShare)}} below ${{pct(filters.minTradeShare)}}`);
      if (account.focusTradeCount < filters.minTrades) output.push(`focus trade count ${{account.focusTradeCount}} below ${{filters.minTrades}}`);
      if (filters.hideTruncated && (account.closedPositionsTruncated || account.activityTruncated)) output.push("hidden by truncated-data filter");
      return output;
    }}
    function currentFilters() {{
      return {{
        minShare: Number(controls.minShare.value || 0),
        minPositions: Number(controls.minPositions.value || 0),
        minTradeShare: Number(controls.minTradeShare.value || 0),
        minTrades: Number(controls.minTrades.value || 0),
        query: controls.search.value.trim().toLowerCase(),
        xProfileFilter: controls.xProfileFilter.value,
        sortBy: controls.sortBy.value,
        sortDir: controls.sortDir.value,
        showOnlyQualified: controls.showOnlyQualified.checked,
        hideTruncated: controls.hideTruncated.checked,
      }};
    }}
    function matchesSearch(account, query) {{
      if (!query) return true;
      return [
        account.handle,
        account.displayName,
        account.wallet,
        account.bio,
        account.topFocusMarkets,
      ].join(" ").toLowerCase().includes(query);
    }}
    function renderSummary(visible, qualified) {{
      const scored = accounts.length;
      const linkedX = accounts.filter(account => account.hasXProfile).length;
      const unlinkedX = accounts.length - linkedX;
      document.getElementById("summary").innerHTML = `
        <div><strong>${{qualified.length}}</strong><span>qualified at current thresholds</span></div>
        <div><strong>${{visible.length}}</strong><span>visible after filters</span></div>
        <div><strong>${{scored}}</strong><span>scored accounts</span></div>
        <div><strong>${{linkedX}}</strong><span>with linked X</span></div>
        <div><strong>${{unlinkedX}}</strong><span>without linked X</span></div>
      `;
    }}
    function render() {{
      const filters = currentFilters();
      document.getElementById("minShareValue").textContent = pct(filters.minShare);
      document.getElementById("minTradeShareValue").textContent = pct(filters.minTradeShare);
      const annotated = accounts.map(account => {{
        const why = reasons(account, filters);
        const qualified = why.length === 0;
        return {{...account, why, qualified}};
      }});
      const qualified = annotated.filter(row => row.qualified);
      let visible = annotated.filter(row => matchesSearch(row, filters.query));
      if (filters.xProfileFilter === "linked") visible = visible.filter(row => row.hasXProfile);
      if (filters.xProfileFilter === "unlinked") visible = visible.filter(row => !row.hasXProfile);
      if (filters.showOnlyQualified) visible = visible.filter(row => row.qualified);
      if (filters.hideTruncated) visible = visible.filter(row => !(row.closedPositionsTruncated || row.activityTruncated));
      visible.sort((a, b) => {{
        const av = Number(a[filters.sortBy] || 0);
        const bv = Number(b[filters.sortBy] || 0);
        const direction = filters.sortDir === "asc" ? 1 : -1;
        return (av === bv ? (b.focusPositionShare - a.focusPositionShare) : (av - bv)) * direction;
      }});
      renderSummary(visible, qualified);
      document.getElementById("empty").hidden = visible.length !== 0;
      document.getElementById("rows").innerHTML = visible.map((account, index) => {{
        const status = account.qualified ? "pass" : "fail";
        const reason = account.qualified ? "meets current thresholds" : account.why.join("; ");
        const notes = account.notes.length ? `<div class="notes">${{text(account.notes.join("; "))}}</div>` : "";
        return `
          <tr>
            <td>${{index + 1}}</td>
            <td><span class="status ${{status}}">${{account.qualified ? "Pass" : "Fail"}}</span></td>
            <td>${{account.hasXProfile ? `<a href="${{text(account.xUrl)}}">@${{text(account.handle)}}</a>` : `<span class="meta">No linked X</span>`}}<div class="meta">${{text(account.displayName)}}</div><div class="meta wallet">${{text(account.wallet)}}</div></td>
            <td class="num">${{money(account.totalProfit)}}</td>
            <td class="num">${{money(account.focusProfit)}}</td>
            <td class="num">${{decimal(account.focusPositionSharpe)}}<div class="meta">${{pct(account.focusAvgRoi)}} avg / ${{pct(account.focusRoiStddev)}} vol</div></td>
            <td class="num">${{pct(account.focusPositionShare)}}<div class="meta">${{account.focusPositionCount}} / ${{account.totalPositionCount}}</div></td>
            <td class="num">${{pct(account.focusTradeVolumeShare)}}<div class="meta">${{account.focusTradeCount}} / ${{account.totalTradeCount}}</div></td>
            <td><div class="reason ${{status}}">${{text(reason)}}</div>${{notes}}</td>
            <td>${{text(account.topFocusMarkets || "No focus markets in sampled closed positions")}}</td>
          </tr>
        `;
      }}).join("");
    }}
    Object.values(controls).forEach(control => control.addEventListener("input", render));
    Object.values(controls).forEach(control => control.addEventListener("change", render));
    render();
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find Polymarket accounts that mainly trade movie or box-office markets, with optional linked X/Twitter filtering.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/accounts/polymarket_box_office_accounts.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=Path("results/accounts/polymarket_box_office_accounts.html"),
        help="HTML visual output path.",
    )
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        default=None,
        help="Compatibility alias for --all-output.",
    )
    parser.add_argument(
        "--all-output",
        type=Path,
        default=None,
        help="CSV path for every scored account, including non-qualifiers, wallets without linked X, and pass/fail reasons. Defaults to OUTPUT stem + _all_scored.csv.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/polymarket/api_cache"),
        help="Directory for cached API responses.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cached API responses and fetch fresh data.",
    )
    parser.add_argument(
        "--market-focus",
        choices=("box-office", "movies"),
        default="box-office",
        help="Use the broad movies tag or a stricter box-office subset.",
    )
    parser.add_argument(
        "--linked-x-only",
        action="store_true",
        help="Deprecated compatibility flag. All candidate wallets are still scored; this only opens the HTML with the linked-X filter selected.",
    )
    parser.add_argument(
        "--min-focus-position-share",
        type=float,
        default=0.9,
        help="Minimum share of historical closed positions in focus markets.",
    )
    parser.add_argument(
        "--min-focus-positions",
        type=int,
        default=3,
        help="Minimum historical closed-position count in focus markets.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Maximum qualifying accounts to include after sorting by total profit.",
    )
    parser.add_argument(
        "--sort-by",
        choices=(
            "total-profit",
            "focus-profit",
            "focus-position-share",
            "focus-position-sharpe",
            "total-position-sharpe",
            "focus-position-count",
            "focus-position-volume",
            "focus-trade-volume",
            "focus-trade-volume-share",
            "total-position-count",
            "total-trade-volume",
        ),
        default="total-profit",
        help="Default sort for the qualifying CSV and initial HTML view.",
    )
    parser.add_argument(
        "--max-markets",
        type=int,
        default=None,
        help="Optional cap for smoke tests or partial runs.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional cap on candidate wallets, ordered by seeded focus-market volume.",
    )
    parser.add_argument(
        "--max-position-pages",
        type=int,
        default=None,
        help="Optional cap on closed-position pages per user for smoke tests.",
    )
    parser.add_argument(
        "--market-workers",
        type=int,
        default=12,
        help="Concurrent workers for focus-market trade seeding.",
    )
    parser.add_argument(
        "--account-workers",
        type=int,
        default=10,
        help="Concurrent workers for per-account profile/activity/position scoring.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="JSONL checkpoint path. Defaults to OUTPUT.checkpoint.jsonl.",
    )
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Discard the matching checkpoint before this run.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Extra seconds to sleep before each uncached API request.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=4,
        help="Retry count for rate limits and transient server/network errors.",
    )
    parser.add_argument(
        "--gamma-markets-rps",
        type=float,
        default=25.0,
        help="Gamma /markets requests per second; docs allow 300 req / 10s.",
    )
    parser.add_argument(
        "--gamma-tags-rps",
        type=float,
        default=10.0,
        help="Gamma /tags requests per second; docs allow 200 req / 10s.",
    )
    parser.add_argument(
        "--gamma-profile-rps",
        type=float,
        default=50.0,
        help="Gamma /public-profile requests per second; under Gamma general 4000 req / 10s.",
    )
    parser.add_argument(
        "--gamma-general-rps",
        type=float,
        default=50.0,
        help="Fallback Gamma requests per second; docs allow 4000 req / 10s general.",
    )
    parser.add_argument(
        "--data-trades-rps",
        type=float,
        default=18.0,
        help="Data /trades requests per second; docs allow 200 req / 10s.",
    )
    parser.add_argument(
        "--data-activity-rps",
        type=float,
        default=40.0,
        help="Data /activity requests per second; kept below Data general 1000 req / 10s.",
    )
    parser.add_argument(
        "--data-closed-positions-rps",
        type=float,
        default=12.0,
        help="Data /closed-positions requests per second; docs allow 150 req / 10s.",
    )
    parser.add_argument(
        "--data-positions-rps",
        type=float,
        default=12.0,
        help="Data /positions requests per second; docs allow 150 req / 10s.",
    )
    parser.add_argument(
        "--data-general-rps",
        type=float,
        default=40.0,
        help="Fallback Data API requests per second; docs allow 1000 req / 10s general.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = find_accounts(args)
    all_output = args.all_output or args.diagnostics_output or default_all_output(args.output)
    write_scores_csv(args.output, result.qualifying_scores)
    write_diagnostics_csv(all_output, result.all_scores, args)
    if args.diagnostics_output and args.all_output and args.diagnostics_output != args.all_output:
        write_diagnostics_csv(args.diagnostics_output, result.all_scores, args)
    write_visual_html(
        args.html_output,
        result.all_scores,
        focus=args.market_focus,
        args=args,
        status_counts=result.status_counts,
    )

    print(f"Wrote {len(result.qualifying_scores)} qualifying accounts to {args.output}", file=sys.stderr)
    print(f"Wrote all {len(result.all_scores)} scored accounts to {all_output}", file=sys.stderr)
    if args.diagnostics_output and args.all_output and args.diagnostics_output != args.all_output:
        print(f"Wrote compatibility diagnostics copy to {args.diagnostics_output}", file=sys.stderr)
    print(f"Wrote visual layer to {args.html_output}", file=sys.stderr)
    for status, count in sorted(result.status_counts.items()):
        print(f"Status {status}: {count}", file=sys.stderr)
    for warning in result.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
