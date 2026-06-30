#!/usr/bin/env python3
"""Collect AMC showtimes and public seat-fill counts.

AMC does not expose a documented public API for these fields. This collector
uses the same webpage method as NameFILIP/amc-good-seats: fetch the AMC page,
parse the embedded Apollo cache from the ``apollo-data`` element, and resolve
Movie, Showtime, Attribute, and Seat objects from that cache.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
from html.parser import HTMLParser
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable


AMC_BASE_URL = "https://www.amctheatres.com"
DEFAULT_CACHE_DIR = Path("data/raw/amc")
DEFAULT_USER_AGENT = "pm-box-office-amc-showtimes/0.1"
DEFAULT_DELAY_SECONDS = 1.0
TRANSIENT_STATUSES = {429, 500, 502, 503, 504}
INVALID_SEAT_TYPES = {"NotASeat", "Companion", "Wheelchair"}
SEAT_NETWORK_KEYWORDS = (
    "graph.amctheatres.com",
    "graphql",
    "seat",
    "showtime",
    "ticket",
    "order",
    "availability",
)


JsonObject = dict[str, Any]


@dataclass(frozen=True)
class ShowtimeRecord:
    theatre_slug: str
    date: str
    showtime_id: str
    when: str
    movie_name: str
    movie_id: str
    showtime_url: str
    attribute_names: str
    total_seats: int | None = None
    available_seats: int | None = None
    filled_or_unavailable_seats: int | None = None
    fill_rate: float | None = None


@dataclass(frozen=True)
class SeatFill:
    theatre_slug: str
    date: str
    showtime_id: str
    showtime_url: str
    total_seats: int
    available_seats: int
    filled_or_unavailable_seats: int
    fill_rate: float | None


@dataclass(frozen=True)
class SeatNetworkDebugResult:
    showtime_id: str
    showtime_url: str
    seat_fill: SeatFill | None
    events: list[JsonObject]
    output_dir: str | None


class ApolloDataParser(HTMLParser):
    """Extract the embedded Apollo cache payload from an AMC HTML page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.apollo_data = ""
        self._capture_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if self._capture_depth:
            self._capture_depth += 1
        elif attrs_dict.get("id") == "apollo-data":
            self._capture_depth = 1
            self._parts = []

    def handle_endtag(self, tag: str) -> None:
        if not self._capture_depth:
            return
        self._capture_depth -= 1
        if self._capture_depth == 0:
            self.apollo_data = "".join(self._parts).strip()

    def handle_data(self, data: str) -> None:
        if self._capture_depth:
            self._parts.append(data)


class RenderedShowtimesParser(HTMLParser):
    """Extract showtimes from AMC's rendered Next.js showtimes HTML."""

    def __init__(self, *, theatre_slug: str, date: dt.date) -> None:
        super().__init__(convert_charrefs=True)
        self.theatre_slug = theatre_slug
        self.date = date
        self.rows: list[ShowtimeRecord] = []
        self._current_movie_name = ""
        self._current_movie_id = ""
        self._in_showtimes_section = False
        self._attribute_depth = 0
        self._attribute_id = ""
        self._attribute_parts: list[str] = []
        self._attribute_names_by_id: dict[str, list[str]] = {}
        self._pending_showtime: dict[str, str] | None = None
        self._time_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key: value or "" for key, value in attrs}

        if tag == "section":
            label = attrs_dict.get("aria-label", "")
            if label.startswith("Showtimes for "):
                self._in_showtimes_section = True
                self._current_movie_name = label.removeprefix("Showtimes for ").strip()
                self._current_movie_id = attrs_dict.get("id", "")

        if not self._in_showtimes_section:
            return

        if tag == "ul" and attrs_dict.get("id", "").endswith("-attributes"):
            self._attribute_depth = 1
            self._attribute_id = attrs_dict["id"]
            self._attribute_parts = []
            return

        if self._attribute_depth:
            self._attribute_depth += 1

        href = attrs_dict.get("href", "")
        showtime_id = attrs_dict.get("id", "")
        if tag == "a" and href.startswith("/showtimes/") and showtime_id.isdigit():
            attribute_names = self._described_attribute_names(attrs_dict.get("aria-describedby", ""))
            self._pending_showtime = {
                "showtime_id": showtime_id,
                "showtime_url": urllib.parse.urljoin(AMC_BASE_URL, href),
                "attribute_names": "|".join(attribute_names),
                "when": "",
            }

        if self._pending_showtime is not None and tag == "time":
            self._time_depth = 1
            self._pending_showtime["when"] = attrs_dict.get("datetime", "")
            return

        if self._time_depth:
            self._time_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self._time_depth:
            self._time_depth -= 1

        if self._pending_showtime is not None and tag == "a":
            self.rows.append(
                ShowtimeRecord(
                    theatre_slug=self.theatre_slug,
                    date=self.date.isoformat(),
                    showtime_id=self._pending_showtime["showtime_id"],
                    when=self._pending_showtime["when"],
                    movie_name=self._current_movie_name,
                    movie_id=self._current_movie_id,
                    showtime_url=self._pending_showtime["showtime_url"],
                    attribute_names=self._pending_showtime["attribute_names"],
                )
            )
            self._pending_showtime = None

        if self._attribute_depth:
            self._attribute_depth -= 1
            if self._attribute_depth == 0:
                self._attribute_names_by_id[self._attribute_id] = [
                    part.strip()
                    for part in self._attribute_parts
                    if part.strip()
                ]
                self._attribute_id = ""
                self._attribute_parts = []

        if self._in_showtimes_section and tag == "section":
            self._in_showtimes_section = False
            self._current_movie_name = ""
            self._current_movie_id = ""

    def handle_data(self, data: str) -> None:
        text = html.unescape(data).strip()
        if self._attribute_depth and text:
            self._attribute_parts.append(text)

    def _described_attribute_names(self, describedby: str) -> list[str]:
        names: list[str] = []
        for element_id in describedby.split():
            names.extend(self._attribute_names_by_id.get(element_id, []))
        return names


class RenderedSeatInputsParser(HTMLParser):
    """Extract rendered seat inputs from AMC's hydrated seats page HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.inputs: list[JsonObject] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "input":
            return
        attrs_dict = {key: value or "" for key, value in attrs}
        name = attrs_dict.get("name", "")
        aria_label = attrs_dict.get("aria-label", "")
        if not name or not aria_label:
            return
        self.inputs.append(
            {
                "name": name,
                "aria_label": aria_label,
                "disabled": "disabled" in attrs_dict or attrs_dict.get("aria-disabled") == "true",
            }
        )


class HtmlFetcher:
    def __init__(
        self,
        cache_dir: Path,
        *,
        refresh: bool = False,
        offline: bool = False,
        delay_seconds: float = DEFAULT_DELAY_SECONDS,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout_seconds: float = 45.0,
        retries: int = 3,
    ) -> None:
        self.cache_dir = cache_dir
        self.refresh = refresh
        self.offline = offline
        self.delay_seconds = delay_seconds
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self._last_request_at = 0.0
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.html"

    def get(self, url: str) -> tuple[str, Path, bool]:
        cache_path = self.cache_path(url)
        if cache_path.exists() and not self.refresh:
            return cache_path.read_text(encoding="utf-8"), cache_path, False
        if self.offline:
            raise FileNotFoundError(f"Cache miss in offline mode: {url}")

        last_error: Exception | None = None
        for attempt in range(self.retries):
            self._wait()
            headers = {
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Encoding": "identity",
                "User-Agent": self.user_agent,
            }
            if "_rsc=" in url:
                headers.update(
                    {
                        "Accept": "text/x-component",
                        "RSC": "1",
                    }
                )
            request = urllib.request.Request(
                url,
                headers=headers,
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    body = response.read().decode("utf-8", errors="replace")
                if not body.strip():
                    raise RuntimeError(f"GET {url} returned an empty response body")
                cache_path.write_text(body, encoding="utf-8")
                self._last_request_at = time.monotonic()
                return body, cache_path, True
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in TRANSIENT_STATUSES or attempt == self.retries - 1:
                    raise
                retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else self.delay_seconds * (attempt + 1)
                time.sleep(delay)
            except (TimeoutError, urllib.error.URLError) as exc:
                last_error = exc
                if attempt == self.retries - 1:
                    break
                time.sleep(self.delay_seconds * (attempt + 1))
        raise RuntimeError(f"GET {url} failed after retry: {last_error}")

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        delay = max(0.0, self.delay_seconds - elapsed)
        if delay:
            time.sleep(delay)


def showtimes_url(day: dt.date, theatre_slug: str) -> str:
    return f"{AMC_BASE_URL}/showtimes/all/{day:%Y-%m-%d}/{theatre_slug}/all"


def showtime_url(day: dt.date, theatre_slug: str, showtime_id: str) -> str:
    return f"{showtimes_url(day, theatre_slug)}/{showtime_id}"


def current_showtime_url(showtime_id: str) -> str:
    return f"{AMC_BASE_URL}/showtimes/{showtime_id}"


def current_showtime_seats_url(showtime_id: str) -> str:
    return f"{current_showtime_url(showtime_id)}/seats"


def current_showtime_seats_rsc_url(showtime_id: str, token: str = "1") -> str:
    query = urllib.parse.urlencode({"_rsc": token})
    return f"{current_showtime_seats_url(showtime_id)}?{query}"


def parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected YYYY-MM-DD date, got {value!r}") from exc


def date_range(start_date: dt.date, end_date: dt.date) -> list[dt.date]:
    if end_date < start_date:
        raise ValueError("end date must be on or after start date")
    days = (end_date - start_date).days
    return [start_date + dt.timedelta(days=offset) for offset in range(days + 1)]


def parse_apollo_data(html: str, *, source_url: str = "") -> JsonObject:
    parser = ApolloDataParser()
    parser.feed(html)
    if not parser.apollo_data:
        location = f" in {source_url}" if source_url else ""
        raise ValueError(f"Could not find apollo-data element{location}")
    try:
        payload = json.loads(parser.apollo_data)
    except json.JSONDecodeError as exc:
        location = f" in {source_url}" if source_url else ""
        raise ValueError(f"Could not decode apollo-data JSON{location}: {exc}") from exc
    if not isinstance(payload, dict):
        location = f" in {source_url}" if source_url else ""
        raise ValueError(f"Expected apollo-data object{location}")
    return payload


def maybe_parse_apollo_data(html: str, *, source_url: str = "") -> JsonObject | None:
    try:
        return parse_apollo_data(html, source_url=source_url)
    except ValueError:
        return None


def values_by_typename(apollo_data: JsonObject, typename: str) -> list[JsonObject]:
    return [
        value
        for value in apollo_data.values()
        if isinstance(value, dict) and value.get("__typename") == typename
    ]


def resolve_ref(apollo_data: JsonObject, ref: str | None) -> JsonObject | None:
    if not ref:
        return None
    value = apollo_data.get(ref)
    return value if isinstance(value, dict) else None


def object_ref(value: Any) -> str | None:
    return value.get("__ref") if isinstance(value, dict) else None


def object_id(obj: JsonObject | None, fallback_ref: str | None = None) -> str:
    if not obj:
        return fallback_ref or ""
    for key in ("id", "movieId", "showtimeId"):
        value = obj.get(key)
        if value not in (None, ""):
            return str(value)
    return fallback_ref or ""


def extract_attribute_names(apollo_data: JsonObject, showtime: JsonObject) -> list[str]:
    attributes = resolve_ref(apollo_data, object_ref(showtime.get("attributes")))
    if not attributes:
        return []
    names: list[str] = []
    for edge_ref in attributes.get("edges") or []:
        edge = resolve_ref(apollo_data, object_ref(edge_ref))
        node = resolve_ref(apollo_data, object_ref(edge.get("node") if edge else None))
        name = node.get("name") if node else None
        if name:
            names.append(str(name))
    return names


def extract_showtimes(
    apollo_data: JsonObject,
    *,
    theatre_slug: str,
    date: dt.date,
) -> list[ShowtimeRecord]:
    rows: list[ShowtimeRecord] = []
    for showtime in values_by_typename(apollo_data, "Showtime"):
        showtime_id = str(showtime.get("showtimeId") or showtime.get("id") or "")
        if not showtime_id:
            continue
        movie_ref = object_ref(showtime.get("movie"))
        movie = resolve_ref(apollo_data, movie_ref)
        attribute_names = extract_attribute_names(apollo_data, showtime)
        rows.append(
            ShowtimeRecord(
                theatre_slug=theatre_slug,
                date=date.isoformat(),
                showtime_id=showtime_id,
                when=str(showtime.get("when") or ""),
                movie_name=str((movie or {}).get("name") or ""),
                movie_id=object_id(movie, movie_ref),
                showtime_url=showtime_url(date, theatre_slug, showtime_id),
                attribute_names="|".join(attribute_names),
            )
        )
    return sorted(rows, key=lambda row: (row.when, row.movie_name, row.showtime_id))


def extract_rendered_showtimes(
    html_text: str,
    *,
    theatre_slug: str,
    date: dt.date,
) -> list[ShowtimeRecord]:
    parser = RenderedShowtimesParser(theatre_slug=theatre_slug, date=date)
    parser.feed(html_text)
    deduped = {row.showtime_id: row for row in parser.rows}
    return sorted(deduped.values(), key=lambda row: (row.when, row.movie_name, row.showtime_id))


def extract_seat_fill(
    apollo_data: JsonObject,
    *,
    theatre_slug: str,
    date: dt.date,
    showtime_id: str,
) -> SeatFill:
    return seat_fill_from_seats(
        values_by_typename(apollo_data, "Seat"),
        theatre_slug=theatre_slug,
        date=date,
        showtime_id=showtime_id,
    )


def seat_fill_from_seats(
    seats: Iterable[JsonObject],
    *,
    theatre_slug: str,
    date: dt.date,
    showtime_id: str,
) -> SeatFill:
    displayed_seats = [
        seat
        for seat in seats
        if seat.get("type") not in INVALID_SEAT_TYPES and bool(seat.get("shouldDisplay"))
    ]
    total_seats = len(displayed_seats)
    available_seats = sum(1 for seat in displayed_seats if bool(seat.get("available")))
    filled_or_unavailable = total_seats - available_seats
    fill_rate = filled_or_unavailable / total_seats if total_seats else None
    return SeatFill(
        theatre_slug=theatre_slug,
        date=date.isoformat(),
        showtime_id=str(showtime_id),
        showtime_url=showtime_url(date, theatre_slug, str(showtime_id)),
        total_seats=total_seats,
        available_seats=available_seats,
        filled_or_unavailable_seats=filled_or_unavailable,
        fill_rate=fill_rate,
    )


def extract_rsc_seat_fill(
    rsc_text: str,
    *,
    theatre_slug: str,
    date: dt.date,
    showtime_id: str,
) -> SeatFill:
    showtime = extract_showtime_from_rsc(rsc_text)
    seating_layout = showtime.get("seatingLayout")
    seats = seating_layout.get("seats") if isinstance(seating_layout, dict) else None
    if not isinstance(seats, list):
        raise ValueError("Could not find showtime.seatingLayout.seats in AMC RSC payload")
    return seat_fill_from_seats(
        [seat for seat in seats if isinstance(seat, dict)],
        theatre_slug=theatre_slug,
        date=date,
        showtime_id=showtime_id,
    )


def extract_showtime_from_rsc(rsc_text: str) -> JsonObject:
    marker = '"showtime":'
    marker_index = 0
    last_error = "Could not find showtime object in AMC RSC payload"
    while True:
        marker_index = rsc_text.find(marker, marker_index)
        if marker_index == -1:
            raise ValueError(last_error)
        value_index = marker_index + len(marker)
        while value_index < len(rsc_text) and rsc_text[value_index].isspace():
            value_index += 1
        if value_index >= len(rsc_text):
            raise ValueError("Could not find showtime value in AMC RSC payload")
        if rsc_text[value_index] != "{":
            marker_index = value_index + 1
            continue
        object_end = find_json_object_end(rsc_text, value_index)
        try:
            showtime = json.loads(rsc_text[value_index : object_end + 1])
        except json.JSONDecodeError as exc:
            last_error = f"Could not decode showtime object in AMC RSC payload: {exc}"
            marker_index = object_end + 1
            continue
        if isinstance(showtime, dict) and "seatingLayout" in showtime:
            return showtime
        last_error = "Could not find showtime object with seatingLayout in AMC RSC payload"
        marker_index = object_end + 1


def find_json_object_end(text: str, object_start: int) -> int:
    depth = 0
    in_string = False
    escaped = False
    for index in range(object_start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("Could not find showtime object end in AMC RSC payload")


def extract_rendered_seat_fill(
    html_text: str,
    *,
    theatre_slug: str,
    date: dt.date,
    showtime_id: str,
) -> SeatFill:
    parser = RenderedSeatInputsParser()
    parser.feed(html_text)
    invalid_labels = {seat_type.lower() for seat_type in INVALID_SEAT_TYPES}
    valid_inputs = [
        seat
        for seat in parser.inputs
        if not any(invalid in str(seat["aria_label"]).lower() for invalid in invalid_labels)
    ]
    total_seats = len(valid_inputs)
    available_seats = sum(1 for seat in valid_inputs if not seat["disabled"])
    filled_or_unavailable = total_seats - available_seats
    fill_rate = filled_or_unavailable / total_seats if total_seats else None
    return SeatFill(
        theatre_slug=theatre_slug,
        date=date.isoformat(),
        showtime_id=str(showtime_id),
        showtime_url=showtime_url(date, theatre_slug, str(showtime_id)),
        total_seats=total_seats,
        available_seats=available_seats,
        filled_or_unavailable_seats=filled_or_unavailable,
        fill_rate=fill_rate,
    )


def is_interesting_seat_network_url(url: str) -> bool:
    lower_url = url.lower()
    return any(keyword in lower_url for keyword in SEAT_NETWORK_KEYWORDS)


def truncate_text(value: str, *, limit: int = 2_000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"... <truncated {len(value) - limit} chars>"


def debug_seat_network(
    *,
    theatre_slug: str,
    date: dt.date,
    showtime_id: str,
    output_dir: Path | None = None,
    headless: bool = True,
    timeout_ms: int = 45_000,
) -> SeatNetworkDebugResult:
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is required for --debug-seat-network. Install it with "
            "`python3 -m pip install playwright` and then run "
            "`python3 -m playwright install chromium`."
        ) from exc

    url = current_showtime_seats_url(showtime_id)
    output_dir_str: str | None = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "seat_network.jsonl").write_text("", encoding="utf-8")
        output_dir_str = str(output_dir)

    events: list[JsonObject] = []

    def append_event(event: JsonObject) -> None:
        events.append(event)
        if output_dir is not None:
            with (output_dir / "seat_network.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
            ),
            viewport={"width": 1440, "height": 1000},
        )
        page = context.new_page()

        def on_request(request: Any) -> None:
            if request.resource_type not in {"fetch", "xhr"} and not is_interesting_seat_network_url(request.url):
                return
            append_event(
                {
                    "event": "request",
                    "resource_type": request.resource_type,
                    "method": request.method,
                    "url": request.url,
                    "post_data": truncate_text(request.post_data or ""),
                }
            )

        def on_response(response: Any) -> None:
            request = response.request
            if request.resource_type not in {"fetch", "xhr"} and not is_interesting_seat_network_url(response.url):
                return
            headers = response.headers
            content_type = headers.get("content-type", "")
            body_path = ""
            body_preview = ""
            body_error = ""
            if "json" in content_type or "text" in content_type or "graphql" in response.url.lower():
                try:
                    body = response.text()
                    body_preview = truncate_text(body)
                    if output_dir is not None and body:
                        body_name = hashlib.sha256(response.url.encode("utf-8")).hexdigest()[:16]
                        suffix = ".json" if "json" in content_type else ".txt"
                        body_file = output_dir / f"{body_name}{suffix}"
                        body_file.write_text(body, encoding="utf-8")
                        body_path = str(body_file)
                except Exception as exc:  # Playwright may not expose every response body.
                    body_error = str(exc)
            append_event(
                {
                    "event": "response",
                    "resource_type": request.resource_type,
                    "method": request.method,
                    "url": response.url,
                    "status": response.status,
                    "content_type": content_type,
                    "body_path": body_path,
                    "body_preview": body_preview,
                    "body_error": body_error,
                }
            )

        page.on("request", on_request)
        page.on("response", on_response)
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

        seat_fill: SeatFill | None = None
        try:
            page.wait_for_selector('input[name][aria-label]', timeout=timeout_ms)
            seat_fill = extract_rendered_seat_fill(
                page.content(),
                theatre_slug=theatre_slug,
                date=date,
                showtime_id=showtime_id,
            )
        except PlaywrightTimeoutError:
            pass

        if output_dir is not None:
            (output_dir / "seat_page.html").write_text(page.content(), encoding="utf-8")

        context.close()
        browser.close()

    return SeatNetworkDebugResult(
        showtime_id=showtime_id,
        showtime_url=url,
        seat_fill=seat_fill,
        events=events,
        output_dir=output_dir_str,
    )


def fetch_showtimes_for_date(
    fetcher: HtmlFetcher,
    *,
    theatre_slug: str,
    date: dt.date,
) -> list[ShowtimeRecord]:
    url = showtimes_url(date, theatre_slug)
    html, _cache_path, _fetched = fetcher.get(url)
    apollo_data = maybe_parse_apollo_data(html, source_url=url)
    if apollo_data is not None:
        return extract_showtimes(apollo_data, theatre_slug=theatre_slug, date=date)
    rows = extract_rendered_showtimes(html, theatre_slug=theatre_slug, date=date)
    if rows:
        return rows
    raise ValueError(
        "Could not find AMC showtimes in embedded Apollo data or rendered HTML "
        f"for {url}. The cached/live page may be an interstitial, empty response, "
        "or a new AMC frontend shape."
    )


def fetch_seat_fill(
    fetcher: HtmlFetcher,
    *,
    theatre_slug: str,
    date: dt.date,
    showtime_id: str,
) -> SeatFill:
    url = current_showtime_seats_url(showtime_id)
    html, _cache_path, _fetched = fetcher.get(url)
    apollo_data = maybe_parse_apollo_data(html, source_url=url)
    if apollo_data is not None:
        return extract_seat_fill(
            apollo_data,
            theatre_slug=theatre_slug,
            date=date,
            showtime_id=showtime_id,
        )
    rendered_fill = extract_rendered_seat_fill(
        html,
        theatre_slug=theatre_slug,
        date=date,
        showtime_id=showtime_id,
    )
    if rendered_fill.total_seats:
        return rendered_fill
    rsc_url = current_showtime_seats_rsc_url(showtime_id)
    rsc_text, _rsc_cache_path, _rsc_fetched = fetcher.get(rsc_url)
    try:
        return extract_rsc_seat_fill(
            rsc_text,
            theatre_slug=theatre_slug,
            date=date,
            showtime_id=showtime_id,
        )
    except ValueError as exc:
        rsc_error = str(exc)
    raise ValueError(
        "Could not find embedded or rendered seat data for "
        f"{url}. AMC's current seats page appears to load the seat map "
        "client-side after the initial HTML, and the RSC payload did not expose "
        f"showtime.seatingLayout.seats ({rsc_error}). Run --debug-seat-network "
        "with --seat-showtime-id to capture the client API requests from a browser."
    )


def collect_showtimes(
    fetcher: HtmlFetcher,
    *,
    theatre_slug: str,
    start_date: dt.date,
    end_date: dt.date,
    with_seat_fill: bool = False,
) -> list[ShowtimeRecord]:
    rows: list[ShowtimeRecord] = []
    for day in date_range(start_date, end_date):
        day_rows = fetch_showtimes_for_date(fetcher, theatre_slug=theatre_slug, date=day)
        if with_seat_fill:
            filled_rows: list[ShowtimeRecord] = []
            for row in day_rows:
                fill = fetch_seat_fill(
                    fetcher,
                    theatre_slug=theatre_slug,
                    date=day,
                    showtime_id=row.showtime_id,
                )
                filled_rows.append(
                    replace(
                        row,
                        total_seats=fill.total_seats,
                        available_seats=fill.available_seats,
                        filled_or_unavailable_seats=fill.filled_or_unavailable_seats,
                        fill_rate=fill.fill_rate,
                    )
                )
            day_rows = filled_rows
        rows.extend(day_rows)
    return rows


def records_to_dicts(records: Iterable[ShowtimeRecord | SeatFill]) -> list[JsonObject]:
    return [asdict(record) for record in records]


def write_records(
    records: Iterable[ShowtimeRecord | SeatFill],
    *,
    output: Path | None,
    output_format: str,
) -> None:
    rows = records_to_dicts(records)
    handle: Any
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        handle = output.open("w", encoding="utf-8", newline="")
    else:
        handle = sys.stdout
    try:
        if output_format == "table":
            write_table(rows, handle)
        elif output_format == "json":
            json.dump(rows, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        elif output_format == "jsonl":
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        elif output_format == "csv":
            fieldnames = list(rows[0].keys()) if rows else list(ShowtimeRecord.__dataclass_fields__)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    finally:
        if output:
            handle.close()


def write_seat_network_debug(result: SeatNetworkDebugResult) -> None:
    print(f"Opened: {result.showtime_url}")
    if result.output_dir:
        print(f"Saved debug artifacts: {result.output_dir}")
    if result.seat_fill:
        print(
            "Rendered seat inputs: "
            f"total={result.seat_fill.total_seats} "
            f"available={result.seat_fill.available_seats} "
            f"filled_or_unavailable={result.seat_fill.filled_or_unavailable_seats} "
            f"fill_rate={format_table_value(result.seat_fill.fill_rate)}"
        )
    else:
        print("Rendered seat inputs: none found before timeout")
    print(f"Captured network events: {len(result.events)}")
    if result.events:
        print()
        write_table(result.events, sys.stdout)


def write_table(rows: list[JsonObject], handle: Any) -> None:
    if not rows:
        handle.write("No records found.\n")
        return
    if "movie_name" in rows[0]:
        columns = [
            "date",
            "when",
            "showtime_id",
            "movie_name",
            "attribute_names",
            "total_seats",
            "available_seats",
            "filled_or_unavailable_seats",
            "fill_rate",
        ]
    elif "event" in rows[0]:
        columns = [
            "event",
            "resource_type",
            "method",
            "status",
            "content_type",
            "url",
            "post_data",
            "body_path",
            "body_preview",
            "body_error",
        ]
    else:
        columns = [
            "date",
            "showtime_id",
            "total_seats",
            "available_seats",
            "filled_or_unavailable_seats",
            "fill_rate",
        ]
    present_columns = [
        column
        for column in columns
        if column in rows[0] and any(row.get(column) not in (None, "") for row in rows)
    ]
    widths = {
        column: max(len(column), *(len(format_table_value(row.get(column))) for row in rows))
        for column in present_columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in present_columns)
    divider = "  ".join("-" * widths[column] for column in present_columns)
    handle.write(header + "\n")
    handle.write(divider + "\n")
    for row in rows:
        handle.write(
            "  ".join(format_table_value(row.get(column)).ljust(widths[column]) for column in present_columns)
            + "\n"
        )


def format_table_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect AMC showtimes and public seat-fill counts from embedded Apollo data.",
    )
    parser.add_argument("theatre_slug", help="AMC theatre slug, e.g. amc-empire-25")
    parser.add_argument("start_date", type=parse_date, help="Start date, YYYY-MM-DD")
    parser.add_argument("end_date", type=parse_date, help="End date, YYYY-MM-DD")
    parser.add_argument(
        "--with-seat-fill",
        action="store_true",
        help="Fetch each showtime detail page and include seat counts.",
    )
    parser.add_argument(
        "--seat-showtime-id",
        help="Fetch only the seat-fill counts for this showtime ID on start_date.",
    )
    parser.add_argument(
        "--seat-html",
        type=Path,
        help=(
            "Parse a browser-rendered AMC seats HTML file for --seat-showtime-id. "
            "Useful with `document.documentElement.outerHTML` captures."
        ),
    )
    parser.add_argument(
        "--debug-seat-network",
        action="store_true",
        help=(
            "Use Playwright to open --seat-showtime-id's seats page and print "
            "the XHR/fetch calls that may contain the client API payload."
        ),
    )
    parser.add_argument(
        "--debug-output-dir",
        type=Path,
        help="Optional directory for --debug-seat-network artifacts.",
    )
    parser.add_argument(
        "--debug-headful",
        action="store_true",
        help="Run --debug-seat-network with a visible browser window.",
    )
    parser.add_argument(
        "--debug-timeout-ms",
        type=int,
        default=45_000,
        help="Timeout for --debug-seat-network page load and seat selector waits.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"HTML cache directory. Default: {DEFAULT_CACHE_DIR}",
    )
    parser.add_argument("--refresh", action="store_true", help="Refresh cached AMC pages.")
    parser.add_argument("--offline", action="store_true", help="Use only cached AMC pages.")
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        help="Minimum delay between live AMC requests.",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="User-Agent header for live AMC requests.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "csv", "json", "jsonl"),
        default="table",
        help="Output format. Default: table",
    )
    parser.add_argument("--output", type=Path, help="Write output to this path instead of stdout.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.end_date < args.start_date:
        raise SystemExit("end_date must be on or after start_date")
    if args.seat_showtime_id and args.end_date != args.start_date:
        raise SystemExit("--seat-showtime-id requires start_date and end_date to be the same")
    if args.seat_html and not args.seat_showtime_id:
        raise SystemExit("--seat-html requires --seat-showtime-id")
    if args.debug_seat_network and not args.seat_showtime_id:
        raise SystemExit("--debug-seat-network requires --seat-showtime-id")

    fetcher = HtmlFetcher(
        args.cache_dir,
        refresh=args.refresh,
        offline=args.offline,
        delay_seconds=args.delay_seconds,
        user_agent=args.user_agent,
    )
    if args.debug_seat_network:
        result = debug_seat_network(
            theatre_slug=args.theatre_slug,
            date=args.start_date,
            showtime_id=args.seat_showtime_id,
            output_dir=args.debug_output_dir,
            headless=not args.debug_headful,
            timeout_ms=args.debug_timeout_ms,
        )
        write_seat_network_debug(result)
        return 0
    if args.seat_html:
        html_text = args.seat_html.read_text(encoding="utf-8")
        records = [
            extract_rendered_seat_fill(
                html_text,
                theatre_slug=args.theatre_slug,
                date=args.start_date,
                showtime_id=args.seat_showtime_id,
            )
        ]
    elif args.seat_showtime_id:
        records: list[ShowtimeRecord | SeatFill] = [
            fetch_seat_fill(
                fetcher,
                theatre_slug=args.theatre_slug,
                date=args.start_date,
                showtime_id=args.seat_showtime_id,
            )
        ]
    else:
        records = collect_showtimes(
            fetcher,
            theatre_slug=args.theatre_slug,
            start_date=args.start_date,
            end_date=args.end_date,
            with_seat_fill=args.with_seat_fill,
        )
    write_records(records, output=args.output, output_format=args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
