#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import unittest

from scripts.ingest import collect_amc_showtimes as amc


def apollo_html(payload: dict[str, object]) -> str:
    return (
        "<html><body>"
        f"<script id=\"apollo-data\" type=\"application/json\">{json.dumps(payload)}</script>"
        "</body></html>"
    )


THEATRE_DATE_PAYLOAD = {
    "Movie:movie-1": {
        "__typename": "Movie",
        "id": "movie-1",
        "name": "Sample One",
    },
    "Movie:movie-2": {
        "__typename": "Movie",
        "id": "movie-2",
        "name": "Sample Two",
    },
    "Showtime:100": {
        "__typename": "Showtime",
        "showtimeId": "100",
        "when": "2026-07-01T19:00:00-04:00",
        "movie": {"__ref": "Movie:movie-1"},
        "attributes": {"__ref": "Showtime:100:attributes"},
    },
    "Showtime:101": {
        "__typename": "Showtime",
        "showtimeId": "101",
        "when": "2026-07-01T13:00:00-04:00",
        "movie": {"__ref": "Movie:movie-2"},
        "attributes": {"__ref": "Showtime:101:attributes"},
    },
    "Showtime:100:attributes": {
        "__typename": "ShowtimeAttributeConnection",
        "edges": [{"__ref": "Showtime:100:attributes:edge-1"}],
    },
    "Showtime:101:attributes": {
        "__typename": "ShowtimeAttributeConnection",
        "edges": [{"__ref": "Showtime:101:attributes:edge-1"}],
    },
    "Showtime:100:attributes:edge-1": {
        "__typename": "ShowtimeAttributeEdge",
        "node": {"__ref": "Attribute:imax"},
    },
    "Showtime:101:attributes:edge-1": {
        "__typename": "ShowtimeAttributeEdge",
        "node": {"__ref": "Attribute:dolby"},
    },
    "Attribute:imax": {
        "__typename": "Attribute",
        "name": "IMAX with Laser at AMC",
    },
    "Attribute:dolby": {
        "__typename": "Attribute",
        "name": "Dolby Cinema at AMC",
    },
}


SEAT_PAYLOAD = {
    "Seat:A1": {
        "__typename": "Seat",
        "name": "A1",
        "type": "Standard",
        "available": True,
        "shouldDisplay": True,
    },
    "Seat:A2": {
        "__typename": "Seat",
        "name": "A2",
        "type": "Standard",
        "available": False,
        "shouldDisplay": True,
    },
    "Seat:A3": {
        "__typename": "Seat",
        "name": "A3",
        "type": "Wheelchair",
        "available": False,
        "shouldDisplay": True,
    },
    "Seat:A4": {
        "__typename": "Seat",
        "name": "A4",
        "type": "NotASeat",
        "available": False,
        "shouldDisplay": True,
    },
    "Seat:A5": {
        "__typename": "Seat",
        "name": "A5",
        "type": "Standard",
        "available": False,
        "shouldDisplay": False,
    },
}


RENDERED_SHOWTIMES_HTML = """
<html><body>
  <section id="sample-one-12345" aria-label="Showtimes for Sample One">
    <header>
      <img alt="" src="poster.jpg">
      <h1><a href="/movies/sample-one-12345">Sample One</a></h1>
    </header>
    <ul aria-label="AMC Sample 10, Sample One Showtimes by Features and Accesibility">
      <li role="listitem" aria-label="Laser Showtimes">
        <ul id="sample-one-12345-amc-sample-10-laser-0-attributes">
          <li>Laser at AMC</li>
          <li>Reserved Seating</li>
        </ul>
        <ul aria-label="Showtime Group Results">
          <li>
            <a
              id="200"
              href="/showtimes/200"
              aria-describedby="sample-one-12345 sample-one-12345-amc-sample-10-laser-0-attributes"
            >
              <time dateTime="2026-07-01T17:00:00.000Z">1:00pm</time>
            </a>
          </li>
        </ul>
      </li>
    </ul>
  </section>
  <section id="sample-two-67890" aria-label="Showtimes for Sample Two">
    <ul>
      <li role="listitem" aria-label="IMAX Showtimes">
        <ul id="sample-two-67890-amc-sample-10-imax-0-attributes">
          <li>IMAX with Laser at AMC</li>
        </ul>
        <a
          id="201"
          href="/showtimes/201"
          aria-describedby="sample-two-67890 sample-two-67890-amc-sample-10-imax-0-attributes"
        >
          <time dateTime="2026-07-01T20:30:00.000Z">4:30pm</time>
        </a>
      </li>
    </ul>
  </section>
</body></html>
"""


RENDERED_SEATS_HTML = """
<html><body>
  <input name="A1" aria-label="Seat A1" />
  <input name="A2" aria-label="Seat A2" disabled />
  <input name="A3" aria-label="Wheelchair A3" disabled />
  <input name="A4" aria-label="Companion A4" />
  <input name="gap" aria-label="NotASeat gap" disabled />
</body></html>
"""


RSC_SEATS_PAYLOAD = """
2:["$","$Ld",null,{"showtime":{"seatingLayout":{"seats":[
  {"available":true,"column":1,"row":1,"name":"A1","type":"CanReserve","shouldDisplay":true},
  {"available":false,"column":2,"row":1,"name":"A2","type":"LoveSeatLeft","shouldDisplay":true},
  {"available":true,"column":3,"row":1,"name":"A3","type":"Wheelchair","shouldDisplay":true},
  {"available":false,"column":4,"row":1,"name":"","type":"NotASeat","shouldDisplay":false}
]}}}}]
"""


class FakeFetcher:
    def __init__(self, pages: dict[str, str]) -> None:
        self.pages = pages
        self.urls: list[str] = []

    def get(self, url: str) -> tuple[str, None, bool]:
        self.urls.append(url)
        return self.pages[url], None, False


class CollectAmcShowtimesTests(unittest.TestCase):
    def test_extract_showtimes_resolves_movies_and_attributes(self) -> None:
        day = dt.date(2026, 7, 1)
        data = amc.parse_apollo_data(apollo_html(THEATRE_DATE_PAYLOAD))
        rows = amc.extract_showtimes(data, theatre_slug="amc-sample-10", date=day)

        self.assertEqual(["101", "100"], [row.showtime_id for row in rows])
        self.assertEqual("Sample Two", rows[0].movie_name)
        self.assertEqual("movie-2", rows[0].movie_id)
        self.assertEqual("Dolby Cinema at AMC", rows[0].attribute_names)
        self.assertEqual(
            "https://www.amctheatres.com/showtimes/all/2026-07-01/amc-sample-10/all/100",
            rows[1].showtime_url,
        )

    def test_extract_showtimes_handles_empty_payload(self) -> None:
        rows = amc.extract_showtimes({}, theatre_slug="amc-sample-10", date=dt.date(2026, 7, 1))

        self.assertEqual([], rows)

    def test_extract_rendered_showtimes_handles_current_nextjs_html(self) -> None:
        rows = amc.extract_rendered_showtimes(
            RENDERED_SHOWTIMES_HTML,
            theatre_slug="amc-sample-10",
            date=dt.date(2026, 7, 1),
        )

        self.assertEqual(["200", "201"], [row.showtime_id for row in rows])
        self.assertEqual("Sample One", rows[0].movie_name)
        self.assertEqual("sample-one-12345", rows[0].movie_id)
        self.assertEqual("Laser at AMC|Reserved Seating", rows[0].attribute_names)
        self.assertEqual("https://www.amctheatres.com/showtimes/200", rows[0].showtime_url)

    def test_parse_apollo_data_requires_embedded_json(self) -> None:
        with self.assertRaisesRegex(ValueError, "apollo-data"):
            amc.parse_apollo_data("<html><body>No embedded cache</body></html>")

    def test_extract_seat_fill_counts_displayed_valid_unavailable_seats(self) -> None:
        fill = amc.extract_seat_fill(
            SEAT_PAYLOAD,
            theatre_slug="amc-sample-10",
            date=dt.date(2026, 7, 1),
            showtime_id="100",
        )

        self.assertEqual(2, fill.total_seats)
        self.assertEqual(1, fill.available_seats)
        self.assertEqual(1, fill.filled_or_unavailable_seats)
        self.assertEqual(0.5, fill.fill_rate)

    def test_extract_seat_fill_handles_sold_out_and_zero_seat_maps(self) -> None:
        sold_out = {
            "Seat:A1": {
                "__typename": "Seat",
                "type": "Standard",
                "available": False,
                "shouldDisplay": True,
            },
            "Seat:A2": {
                "__typename": "Seat",
                "type": "Standard",
                "available": False,
                "shouldDisplay": True,
            },
        }
        fill = amc.extract_seat_fill(
            sold_out,
            theatre_slug="amc-sample-10",
            date=dt.date(2026, 7, 1),
            showtime_id="100",
        )
        empty = amc.extract_seat_fill(
            {},
            theatre_slug="amc-sample-10",
            date=dt.date(2026, 7, 1),
            showtime_id="100",
        )

        self.assertEqual(0, fill.available_seats)
        self.assertEqual(2, fill.filled_or_unavailable_seats)
        self.assertEqual(1.0, fill.fill_rate)
        self.assertIsNone(empty.fill_rate)

    def test_extract_rendered_seat_fill_counts_disabled_valid_seat_inputs(self) -> None:
        fill = amc.extract_rendered_seat_fill(
            RENDERED_SEATS_HTML,
            theatre_slug="amc-sample-10",
            date=dt.date(2026, 7, 1),
            showtime_id="100",
        )

        self.assertEqual(2, fill.total_seats)
        self.assertEqual(1, fill.available_seats)
        self.assertEqual(1, fill.filled_or_unavailable_seats)
        self.assertEqual(0.5, fill.fill_rate)

    def test_fetch_seat_fill_can_use_rendered_seat_inputs(self) -> None:
        fetcher = FakeFetcher({amc.current_showtime_seats_url("100"): RENDERED_SEATS_HTML})

        fill = amc.fetch_seat_fill(
            fetcher,  # type: ignore[arg-type]
            theatre_slug="amc-sample-10",
            date=dt.date(2026, 7, 1),
            showtime_id="100",
        )

        self.assertEqual(2, fill.total_seats)
        self.assertEqual(1, fill.filled_or_unavailable_seats)

    def test_extract_rsc_seat_fill_counts_showtime_seating_layout(self) -> None:
        fill = amc.extract_rsc_seat_fill(
            RSC_SEATS_PAYLOAD,
            theatre_slug="amc-sample-10",
            date=dt.date(2026, 7, 1),
            showtime_id="100",
        )

        self.assertEqual(2, fill.total_seats)
        self.assertEqual(1, fill.available_seats)
        self.assertEqual(1, fill.filled_or_unavailable_seats)

    def test_fetch_seat_fill_falls_back_to_rsc_payload(self) -> None:
        fetcher = FakeFetcher(
            {
                amc.current_showtime_seats_url("100"): (
                    "<html><body><div role=\"status\">Loading</div></body></html>"
                ),
                amc.current_showtime_seats_rsc_url("100"): RSC_SEATS_PAYLOAD,
            }
        )

        fill = amc.fetch_seat_fill(
            fetcher,  # type: ignore[arg-type]
            theatre_slug="amc-sample-10",
            date=dt.date(2026, 7, 1),
            showtime_id="100",
        )

        self.assertEqual(2, fill.total_seats)
        self.assertEqual(1, fill.filled_or_unavailable_seats)

    def test_collect_showtimes_can_attach_seat_fill_with_fake_fetcher(self) -> None:
        day = dt.date(2026, 7, 1)
        theatre_slug = "amc-sample-10"
        pages = {
            amc.showtimes_url(day, theatre_slug): apollo_html(THEATRE_DATE_PAYLOAD),
            amc.current_showtime_seats_url("100"): apollo_html(SEAT_PAYLOAD),
            amc.current_showtime_seats_url("101"): apollo_html(SEAT_PAYLOAD),
        }
        fetcher = FakeFetcher(pages)

        rows = amc.collect_showtimes(
            fetcher,  # type: ignore[arg-type]
            theatre_slug=theatre_slug,
            start_date=day,
            end_date=day,
            with_seat_fill=True,
        )

        self.assertEqual(2, len(rows))
        self.assertEqual([1, 1], [row.filled_or_unavailable_seats for row in rows])
        self.assertEqual(3, len(fetcher.urls))

    def test_fetch_seat_fill_reports_current_non_apollo_seat_pages(self) -> None:
        fetcher = FakeFetcher(
            {
                amc.current_showtime_seats_url("100"): (
                    "<html><body><div role=\"status\">Loading</div></body></html>"
                ),
                amc.current_showtime_seats_rsc_url("100"): "no seats here",
            }
        )

        with self.assertRaisesRegex(ValueError, "RSC payload"):
            amc.fetch_seat_fill(
                fetcher,  # type: ignore[arg-type]
                theatre_slug="amc-sample-10",
                date=dt.date(2026, 7, 1),
                showtime_id="100",
            )


if __name__ == "__main__":
    unittest.main()
