#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pm_box_office.sources.boxofficetheory import ingest
from tests.postgres_test_utils import drop_isolated_postgres_schema, make_isolated_postgres_schema


TRACKING_TABLE_HTML = """
<table>
<tr>
  <th>Release Date</th>
  <th>Title</th>
  <th>Distributor</th>
  <th>3-Day (FSS) LOW-END Opening</th>
  <th>3-Day (FSS) HIGH-END Opening</th>
  <th>3-Day (FSS) PINPOINT Opening Forecast</th>
  <th>Domestic Total LOW-END</th>
  <th>Domestic Total HIGH-END</th>
  <th>Domestic Total PINPOINT Forecast</th>
</tr>
<tr>
  <td>6/12/2026</td>
  <td>Disclosure Day</td>
  <td>Universal Pictures</td>
  <td>$40,000,000</td>
  <td>$55,000,000</td>
  <td>$50,500,000</td>
  <td>$125,000,000</td>
  <td>$206,000,000</td>
  <td>$159,000,000</td>
</tr>
<tr>
  <td>6/12/2026</td>
  <td>Stop! That! Train!</td>
  <td>Bleecker Street</td>
  <td>$3,000,000</td>
  <td>$8,000,000</td>
  <td>$4,500,000</td>
  <td>$5,500,000</td>
  <td>$19,000,000</td>
  <td>$9,000,000</td>
</tr>
</table>
"""


WEEKEND_TABLE_HTML = """
<table>
<tr>
  <th>Film</th>
  <th>Distributor</th>
  <th>3-Day (Fri-Sun) Weekend Forecast</th>
  <th>3-Day Change from Last Weekend</th>
  <th>4-Day (Fri-Mon) Weekend Forecast</th>
  <th>Projected Domestic Total through Monday, May 25</th>
  <th>Estimated Location Count (as of Tue)</th>
</tr>
<tr>
  <td>Star Wars: The Mandalorian &amp; Grogu</td>
  <td>Disney (Lucasfilm)</td>
  <td>$85,100,000</td>
  <td>NEW</td>
  <td>$103,800,000</td>
  <td>$103,800,000</td>
  <td>4,300</td>
</tr>
<tr>
  <td>Michael</td>
  <td>Lionsgate</td>
  <td>$17,000,000</td>
  <td>-35%</td>
  <td>$22,000,000</td>
  <td>$314,400,000</td>
  <td>2,700</td>
</tr>
</table>
"""


PROSE_HTML = """
<p><b><i>Minions &amp; Monsters </i></b>(Universal &amp; Illumination)<br />
BOT Domestic Opening Weekend Forecast Range: <strong>$68 — 87 million (5-Day)</strong><br />
Studio Tracking: $95 million (5-day)</p>
<p><b><i>Young Washington </i></b>(Angel Studios)<br />
BOT Domestic Opening Weekend Forecast Range: <strong>$23 million+</strong></p>
"""


def post_record(content_html: str, *, post_id: int = 5000, date_gmt: str = "2026-06-05T19:16:41") -> ingest.PostRecord:
    return ingest.PostRecord(
        source_post_id=post_id,
        post_url=f"https://boxofficetheory.com/sample-{post_id}/",
        title="Box Office Tracking & Forecasts: Sample",
        author="Shawn Robbins",
        published_at=ingest.parse_wp_gmt_datetime(date_gmt),
        published_date=ingest.parse_wp_gmt_datetime(date_gmt)[:10],
        excerpt="Sample excerpt",
        content_html=content_html,
        source_url="https://boxofficetheory.com/wp-json/wp/v2/posts?page=1",
    )


class BoxOfficeTheoryParserTests(unittest.TestCase):
    def test_parse_post_records_from_wordpress_api(self) -> None:
        payload = [
            {
                "id": 5000,
                "date_gmt": "2026-06-05T19:16:41",
                "link": "https://boxofficetheory.com/sample/?utm_source=x",
                "title": {"rendered": "Box Office Tracking &amp; Forecasts: Sample"},
                "content": {"rendered": TRACKING_TABLE_HTML},
                "excerpt": {"rendered": "<p>Sample excerpt.</p>"},
                "categories": [6],
            }
        ]

        posts = ingest.parse_post_records(json.dumps(payload), source_url="api")

        self.assertEqual(1, len(posts))
        self.assertEqual(5000, posts[0].source_post_id)
        self.assertEqual("https://boxofficetheory.com/sample/", posts[0].post_url)
        self.assertEqual("Box Office Tracking & Forecasts: Sample", posts[0].title)
        self.assertEqual("2026-06-05T19:16:41+00:00", posts[0].published_at)
        self.assertEqual("Sample excerpt.", posts[0].excerpt)

    def test_tracking_table_extracts_opening_and_total_predictions(self) -> None:
        predictions = ingest.parse_predictions(post_record(TRACKING_TABLE_HTML))

        self.assertEqual(2, len(predictions))
        first = predictions[0]
        self.assertEqual("Disclosure Day", first.source_movie_title)
        self.assertEqual("Universal Pictures", first.distributor)
        self.assertEqual("2026-06-12", first.release_date)
        self.assertEqual("pre_release_tracking", first.prediction_scope)
        self.assertEqual("domestic_opening_and_total", first.forecast_metric)
        self.assertEqual(40_000_000, first.opening_weekend_low_usd)
        self.assertEqual(55_000_000, first.opening_weekend_high_usd)
        self.assertEqual(50_500_000, first.opening_weekend_pinpoint_usd)
        self.assertEqual(3, first.opening_weekend_day_count)
        self.assertEqual(125_000_000, first.domestic_total_low_usd)
        self.assertEqual(206_000_000, first.domestic_total_high_usd)
        self.assertEqual(159_000_000, first.domestic_total_pinpoint_usd)
        self.assertEqual("tracking_forecast_table", first.source_context)

    def test_weekend_table_extracts_weekly_forecast_predictions(self) -> None:
        predictions = ingest.parse_predictions(post_record(WEEKEND_TABLE_HTML, post_id=4945, date_gmt="2026-05-20T17:04:11"))

        self.assertEqual(2, len(predictions))
        star_wars = predictions[0]
        self.assertEqual("Star Wars: The Mandalorian & Grogu", star_wars.source_movie_title)
        self.assertEqual("Disney (Lucasfilm)", star_wars.distributor)
        self.assertEqual("weekend_forecast", star_wars.prediction_scope)
        self.assertEqual("domestic_weekend", star_wars.forecast_metric)
        self.assertEqual(85_100_000, star_wars.weekend_forecast_usd)
        self.assertEqual(3, star_wars.weekend_day_count)
        self.assertEqual(103_800_000, star_wars.alternate_opening_weekend_usd)
        self.assertEqual(4, star_wars.alternate_opening_weekend_day_count)
        self.assertEqual(103_800_000, star_wars.projected_domestic_total_usd)
        self.assertEqual("NEW", star_wars.change_label)
        self.assertEqual(4_300, star_wars.location_count)

        michael = predictions[1]
        self.assertEqual(-35.0, michael.percent_change)
        self.assertEqual(314_400_000, michael.projected_domestic_total_usd)

    def test_prose_ranges_extract_public_bot_predictions(self) -> None:
        predictions = ingest.parse_predictions(post_record(PROSE_HTML, post_id=5067, date_gmt="2026-07-01T17:11:43"))

        self.assertEqual(2, len(predictions))
        minions = predictions[0]
        self.assertEqual("Minions & Monsters", minions.source_movie_title)
        self.assertEqual("Universal & Illumination", minions.distributor)
        self.assertEqual(68_000_000, minions.opening_weekend_low_usd)
        self.assertEqual(87_000_000, minions.opening_weekend_high_usd)
        self.assertEqual(5, minions.opening_weekend_day_count)
        self.assertEqual("prose_opening_range", minions.source_context)

        washington = predictions[1]
        self.assertEqual("Young Washington", washington.source_movie_title)
        self.assertEqual(23_000_000, washington.opening_weekend_low_usd)
        self.assertIsNone(washington.opening_weekend_high_usd)

    def test_fetcher_caches_api_response_with_headers(self) -> None:
        class Response:
            headers = {"X-WP-TotalPages": "3"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self) -> bytes:
                return b"[]"

        with tempfile.TemporaryDirectory() as tmp, mock.patch("urllib.request.urlopen", return_value=Response()):
            fetcher = ingest.CachedJsonFetcher(
                Path(tmp),
                refresh=False,
                offline=False,
                delay_seconds=0,
                user_agent=ingest.DEFAULT_USER_AGENT,
            )
            body, cache_path, fetched, headers = fetcher.get("https://boxofficetheory.com/wp-json/wp/v2/posts")

            self.assertTrue(fetched)
            self.assertEqual("[]", body)
            self.assertTrue(cache_path.exists())
            self.assertEqual("3", headers["x-wp-totalpages"])


class BoxOfficeTheoryDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn, self.schema = make_isolated_postgres_schema()
        ingest.initialize_database(self.conn)
        self.conn.commit()

    def tearDown(self) -> None:
        drop_isolated_postgres_schema(self.conn, self.schema)

    def test_import_parse_result_persists_predictions(self) -> None:
        post = post_record(TRACKING_TABLE_HTML)
        predictions = ingest.parse_predictions(post)
        result = ingest.PageParseResult(
            post=post,
            predictions=predictions,
            fetched_at="2026-07-06T00:00:00+00:00",
            raw_cache_path=Path("cache.json"),
            raw_json="[]",
        )

        post_count, prediction_count = ingest.import_parse_result(
            self.conn,
            result,
            issue_source="test_boxofficetheory",
        )

        self.assertEqual(1, post_count)
        self.assertEqual(2, prediction_count)
        row = self.conn.execute(
            """
            SELECT source_movie_title, opening_weekend_pinpoint_usd, domestic_total_pinpoint_usd
            FROM boxofficetheory_predictions
            WHERE source_movie_title = 'Disclosure Day'
            """
        ).fetchone()
        self.assertEqual("Disclosure Day", row[0])
        self.assertEqual(50_500_000, row[1])
        self.assertEqual(159_000_000, row[2])


if __name__ == "__main__":
    unittest.main()
