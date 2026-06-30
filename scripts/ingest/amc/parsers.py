"""AMC showtime and seat-map parsers."""

from __future__ import annotations

import datetime as dt
import html
import json
import urllib.parse
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Iterable


AMC_BASE_URL = "https://www.amctheatres.com"
INVALID_SEAT_TYPES = {"NotASeat", "Companion", "Wheelchair"}
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


class ApolloDataParser(HTMLParser):
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
                    part.strip() for part in self._attribute_parts if part.strip()
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


def showtimes_url(day: dt.date, theatre_slug: str) -> str:
    return f"{AMC_BASE_URL}/showtimes/all/{day:%Y-%m-%d}/{theatre_slug}/all"


def showtime_url(day: dt.date, theatre_slug: str, showtime_id: str) -> str:
    return f"{showtimes_url(day, theatre_slug)}/{showtime_id}"


def current_showtime_url(showtime_id: str) -> str:
    return f"{AMC_BASE_URL}/showtimes/{showtime_id}"


def current_showtime_seats_url(showtime_id: str) -> str:
    return f"{current_showtime_url(showtime_id)}/seats"


def current_showtime_seats_rsc_url(showtime_id: str, token: str = "1") -> str:
    return f"{current_showtime_seats_url(showtime_id)}?{urllib.parse.urlencode({'_rsc': token})}"


def parse_apollo_data(html_text: str, *, source_url: str = "") -> JsonObject:
    parser = ApolloDataParser()
    parser.feed(html_text)
    if not parser.apollo_data:
        location = f" in {source_url}" if source_url else ""
        raise ValueError(f"Could not find apollo-data element{location}")
    try:
        payload = json.loads(parser.apollo_data)
    except json.JSONDecodeError as exc:
        location = f" in {source_url}" if source_url else ""
        raise ValueError(f"Could not decode apollo-data JSON{location}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Expected apollo-data object")
    return payload


def maybe_parse_apollo_data(html_text: str, *, source_url: str = "") -> JsonObject | None:
    try:
        return parse_apollo_data(html_text, source_url=source_url)
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
        if value_index >= len(rsc_text) or rsc_text[value_index] != "{":
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


def fetch_showtimes_for_date(fetcher: Any, *, theatre_slug: str, date: dt.date) -> list[ShowtimeRecord]:
    url = showtimes_url(date, theatre_slug)
    html_text, _cache_path, _fetched = fetcher.get(url)
    apollo_data = maybe_parse_apollo_data(html_text, source_url=url)
    if apollo_data is not None:
        return extract_showtimes(apollo_data, theatre_slug=theatre_slug, date=date)
    rows = extract_rendered_showtimes(html_text, theatre_slug=theatre_slug, date=date)
    if rows:
        return rows
    raise ValueError(f"Could not find AMC showtimes in embedded Apollo data or rendered HTML for {url}")


def fetch_seat_fill(fetcher: Any, *, theatre_slug: str, date: dt.date, showtime_id: str) -> SeatFill:
    url = current_showtime_seats_url(showtime_id)
    html_text, _cache_path, _fetched = fetcher.get(url)
    apollo_data = maybe_parse_apollo_data(html_text, source_url=url)
    if apollo_data is not None:
        return extract_seat_fill(
            apollo_data,
            theatre_slug=theatre_slug,
            date=date,
            showtime_id=showtime_id,
        )
    rendered_fill = extract_rendered_seat_fill(
        html_text,
        theatre_slug=theatre_slug,
        date=date,
        showtime_id=showtime_id,
    )
    if rendered_fill.total_seats:
        return rendered_fill
    rsc_url = current_showtime_seats_rsc_url(showtime_id)
    rsc_text, _rsc_cache_path, _rsc_fetched = fetcher.get(rsc_url)
    return extract_rsc_seat_fill(
        rsc_text,
        theatre_slug=theatre_slug,
        date=date,
        showtime_id=showtime_id,
    )
