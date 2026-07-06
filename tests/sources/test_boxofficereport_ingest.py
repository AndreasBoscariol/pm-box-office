#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from unittest import mock
from pathlib import Path

from pm_box_office.sources.boxofficereport import ingest
from tests.postgres_test_utils import drop_isolated_postgres_schema, make_isolated_postgres_schema


ARCHIVE_HTML = """
<html>
<body>
<h2>Weekend Predictions Archive</h2>
<table>
<tr>
<td>Weekend</td>
<td>#1 Film</td>
<td>Prediction</td>
<td>Actual</td>
</tr>
<tr>
<td><a href="http://www.boxofficereport.com/predictions/predictions20260702.html">Jul. 3, 2026 - Jul. 5, 2026</a></td>
<td>Minions &amp; Monsters</td>
<td>$51.0 M</td>
<td>N/A</td>
</tr>
<tr>
<td><a href="http://www.boxofficereport.com/predictions/predictions20161228.html">Dec. 30, 2016 - Jan. 2, 2017</a></td>
<td>Rogue One: A Star Wars Story</td>
<td>$76.0 M</td>
<td>$65.5 M</td>
</tr>
</table>
</body>
</html>
"""


ARTICLE_HTML = """
<html>
<head>
<title>Box Office Report - Weekend Box Office Predictions: July 3 - July 5, 2026</title>
</head>
<body>
<h2>Weekend Box Office Predictions<br>July 3 - July 5, 2026</h2>
<table>
<tr>
<td><h5>Published on July 1, 2026 at 8:45PM Pacific<br>
<a href="https://x.com/DanielJGarris">By Daniel Garris</a></h5></td>
</tr>
</table>
<h4>Weekend predictions for the top 8 films at the domestic box office.</h4>
<table>
<tr>
<td>Rank</td>
<td>Film (Distributor)</td>
<td>Weekend<br>Gross</td>
<td>Total<br>Gross</td>
<td>%<br>Change</td>
<td>Week<br>#</td>
</tr>
<tr>
<td>1</td>
<td>Minions &amp; Monsters<br>(Universal)</td>
<td>$51.0 M</td>
<td>$78.0 M</td>
<td>NEW</td>
<td>1</td>
</tr>
<tr>
<td>2</td>
<td>Toy Story 5<br>(Disney / Pixar)</td>
<td>$36.0 M</td>
<td>$371.5 M</td>
<td>-49%</td>
<td>3</td>
</tr>
</table>
</body>
</html>
"""


class BoxOfficeReportParserTests(unittest.TestCase):
    def test_archive_parser_discovers_prediction_pages(self) -> None:
        pages = ingest.parse_archive(ARCHIVE_HTML)

        self.assertEqual(2, len(pages))
        self.assertEqual(
            "http://www.boxofficereport.com/predictions/predictions20260702.html",
            pages[0].article_url,
        )
        self.assertEqual("2026-07-03", pages[0].target_start_date)
        self.assertEqual("2026-07-05", pages[0].target_end_date)
        self.assertEqual("Minions & Monsters", pages[0].archive_top_film)
        self.assertEqual(51_000_000, pages[0].archive_top_prediction_usd)
        self.assertIsNone(pages[0].archive_top_actual_usd)
        self.assertEqual("2016-12-30", pages[1].target_start_date)
        self.assertEqual("2017-01-02", pages[1].target_end_date)
        self.assertEqual(65_500_000, pages[1].archive_top_actual_usd)

    def test_article_parser_extracts_publication_time_and_prediction_rows(self) -> None:
        fallback = ingest.parse_archive(ARCHIVE_HTML)[0]
        article, predictions = ingest.parse_article(
            ARTICLE_HTML,
            article_url=fallback.article_url,
            fallback=fallback,
        )

        self.assertEqual("Weekend Box Office Predictions July 3 - July 5, 2026", article.title)
        self.assertEqual("Daniel Garris", article.author)
        self.assertEqual("2026-07-01", article.prediction_made_date)
        self.assertEqual("2026-07-01T20:45:00-07:00", article.prediction_made_at)
        self.assertEqual("2026-07-03", article.target_start_date)
        self.assertEqual("2026-07-05", article.target_end_date)

        self.assertEqual(2, len(predictions))
        minions = predictions[0]
        self.assertEqual("Minions & Monsters", minions.source_movie_title)
        self.assertEqual("Universal", minions.distributor)
        self.assertEqual(1, minions.source_rank)
        self.assertEqual("domestic_opening_weekend", minions.forecast_metric)
        self.assertEqual(51_000_000, minions.weekend_gross_prediction_usd)
        self.assertEqual(78_000_000, minions.total_gross_prediction_usd)
        self.assertEqual("NEW", minions.change_label)
        self.assertIsNone(minions.percent_change)
        self.assertEqual(1, minions.week_number)
        self.assertEqual("2026-07-01T20:45:00-07:00", minions.prediction_made_at)

        toy_story = predictions[1]
        self.assertEqual("Toy Story 5", toy_story.source_movie_title)
        self.assertEqual("Disney / Pixar", toy_story.distributor)
        self.assertEqual("domestic_weekend", toy_story.forecast_metric)
        self.assertEqual(-49.0, toy_story.percent_change)
        self.assertEqual(3, toy_story.week_number)

    def test_full_refresh_args_cover_entire_archive(self) -> None:
        args = ingest.build_arg_parser().parse_args(["--full-refresh"])

        ingest.configure_full_refresh_args(args)

        self.assertTrue(args.refresh)
        self.assertEqual(ingest.FULL_REFRESH_START_DATE, args.start_date)
        self.assertEqual(ingest.FULL_REFRESH_END_DATE, args.end_date)

    def test_fetcher_caches_http_response(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self) -> bytes:
                return b"<html></html>"

        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "urllib.request.urlopen",
            return_value=Response(),
        ):
            fetcher = ingest.HtmlFetcher(
                Path(tmp),
                refresh=False,
                offline=False,
                delay_seconds=0,
                user_agent=ingest.DEFAULT_USER_AGENT,
            )

            body, cache_path, fetched = fetcher.get("http://www.boxofficereport.com/predictions/sample.html")

            self.assertTrue(fetched)
            self.assertEqual("<html></html>", body)
            self.assertTrue(cache_path.exists())


class BoxOfficeReportDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn, self.schema = make_isolated_postgres_schema()
        ingest.initialize_database(self.conn)
        self.conn.commit()

    def tearDown(self) -> None:
        drop_isolated_postgres_schema(self.conn, self.schema)

    def test_import_parse_result_persists_prediction_made_timestamp(self) -> None:
        fallback = ingest.parse_archive(ARCHIVE_HTML)[0]
        article, predictions = ingest.parse_article(ARTICLE_HTML, article_url=fallback.article_url, fallback=fallback)
        result = ingest.ParseResult(
            archive_page=fallback,
            article=article,
            predictions=predictions,
            fetched_at="2026-07-06T00:00:00+00:00",
            raw_cache_path=Path("cache.html"),
            html=ARTICLE_HTML,
        )

        article_count, prediction_count = ingest.import_parse_result(
            self.conn,
            result,
            issue_source="test_boxofficereport",
        )

        self.assertEqual(1, article_count)
        self.assertEqual(2, prediction_count)
        row = self.conn.execute(
            """
            SELECT a.prediction_made_at::text, p.source_movie_title, p.weekend_gross_prediction_usd
            FROM boxofficereport_weekend_predictions p
            JOIN boxofficereport_articles a USING(article_id)
            WHERE p.source_rank = 1
            """
        ).fetchone()
        self.assertIn("2026-07-01", row[0])
        self.assertEqual("Minions & Monsters", row[1])
        self.assertEqual(51_000_000, row[2])


if __name__ == "__main__":
    unittest.main()
