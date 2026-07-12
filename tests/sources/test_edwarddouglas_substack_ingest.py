#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pm_box_office.sources.edwarddouglas_substack import ingest
from tests.postgres_test_utils import drop_isolated_postgres_schema, make_isolated_postgres_schema


SUBSTACK_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss xmlns:dc="http://purl.org/dc/elements/1.1/"
     xmlns:content="http://purl.org/rss/1.0/modules/content/"
     version="2.0">
  <channel>
    <title>The Weekend Warrior Newsletter</title>
    <item>
      <title><![CDATA[THE WEEKEND WARRIOR SHOW for May 29, 2026]]></title>
      <description><![CDATA[BACKROOMS, THE BREADWINNER, PRESSURE]]></description>
      <link>https://edwarddouglas.substack.com/p/the-weekend-warrior-show-for-may-25c?utm_source=feed</link>
      <guid isPermaLink="false">https://edwarddouglas.substack.com/p/the-weekend-warrior-show-for-may-25c</guid>
      <dc:creator><![CDATA[Edward Douglas]]></dc:creator>
      <pubDate>Tue, 26 May 2026 21:05:44 GMT</pubDate>
      <content:encoded><![CDATA[
        <p><strong>UPDATED BOX OFFICE PREDICTIONS FOR MAY 29, 2026<br></strong>
        Box Office Data Provided by <a href="http://the-numbers.com">The-Numbers.com</a></p>
        <p>1. <strong>Backrooms </strong>(A24) - $46.2 million N/A (UP $5.7 million)<br>
        2. <strong>Star Wars: The Mandalorian and Grogu </strong>(LucasFilm/Disney) - $35 million -57%<br>
        9. <strong>Pressure </strong>(Focus Features) - $2.6 million N/A</p>
      ]]></content:encoded>
    </item>
  </channel>
</rss>
"""


SUBSTACK_POST_JSON = """
{
  "id": 199377007,
  "slug": "the-weekend-warrior-show-for-may-25c",
  "canonical_url": "https://edwarddouglas.substack.com/p/the-weekend-warrior-show-for-may-25c",
  "title": "THE WEEKEND WARRIOR SHOW for May 29, 2026",
  "subtitle": "BACKROOMS, THE BREADWINNER, PRESSURE",
  "description": "BACKROOMS, THE BREADWINNER, PRESSURE",
  "post_date": "2026-05-26T21:05:44.000Z",
  "body_html": "<p><strong>Box Office Predictions for 6/5/2026<br></strong>Box Office Data Provided by <a href=\\"http://the-numbers.com\\">The-Numbers.com</a></p><p>1. <strong>Scary Movie </strong>(Paramount) - $44.2  million N/A<br>2. <strong>Backrooms </strong>(A24) - $32.6 million -60%<br>&#8211; <strong>Peddi  </strong>(Prathyangira Cinemas) - $2.7 million N/A</p>",
  "publishedBylines": [{"name": "Edward Douglas"}]
}
"""


class EdwardDouglasSubstackParserTests(unittest.TestCase):
    def test_parse_rss_posts_extracts_substack_item_content(self) -> None:
        posts = ingest.parse_rss_posts(SUBSTACK_RSS, source_url=ingest.FEED_URL)

        self.assertEqual(1, len(posts))
        post = posts[0]
        self.assertEqual("the-weekend-warrior-show-for-may-25c", post.source_post_id)
        self.assertEqual("https://edwarddouglas.substack.com/p/the-weekend-warrior-show-for-may-25c", post.post_url)
        self.assertEqual("THE WEEKEND WARRIOR SHOW for May 29, 2026", post.title)
        self.assertEqual("Edward Douglas", post.author)
        self.assertEqual("2026-05-26T21:05:44+00:00", post.published_at)
        self.assertIn("UPDATED BOX OFFICE PREDICTIONS", post.content_html)

    def test_weekend_warrior_prediction_rows_parse_as_weekend_forecasts(self) -> None:
        post = ingest.parse_rss_posts(SUBSTACK_RSS, source_url=ingest.FEED_URL)[0]

        predictions = ingest.parse_predictions(post)

        self.assertEqual(3, len(predictions))
        self.assertEqual("Backrooms", predictions[0].source_movie_title)
        self.assertEqual("A24", predictions[0].distributor)
        self.assertEqual("2026-05-29", predictions[0].release_date)
        self.assertEqual(46_200_000, predictions[0].weekend_forecast_usd)
        self.assertEqual(3, predictions[0].weekend_day_count)
        self.assertEqual("N/A (UP $5.7 million)", predictions[0].change_label)
        self.assertEqual(ingest.PARSER_VERSION, predictions[0].parser_version)
        self.assertEqual("edwarddouglas_substack:US_CA:backrooms", predictions[0].source_movie_id)
        self.assertEqual("Star Wars: The Mandalorian and Grogu", predictions[1].source_movie_title)
        self.assertEqual(-57.0, predictions[1].percent_change)

    def test_archive_json_rows_support_numeric_dates_and_dash_entries(self) -> None:
        post = ingest.parse_post_record_json(
            SUBSTACK_POST_JSON,
            source_url="https://edwarddouglas.substack.com/api/v1/posts/the-weekend-warrior-show-for-may-25c",
        )

        predictions = ingest.parse_predictions(post)

        self.assertEqual(["Scary Movie", "Backrooms", "Peddi"], [prediction.source_movie_title for prediction in predictions])
        self.assertEqual("2026-06-05", predictions[0].release_date)
        self.assertEqual(44_200_000, predictions[0].weekend_forecast_usd)
        self.assertEqual("Prathyangira Cinemas", predictions[2].distributor)
        self.assertEqual(2_700_000, predictions[2].weekend_forecast_usd)

    def test_heading_date_without_for_sets_weekend_forecast_date(self) -> None:
        post = ingest.theory.PostRecord(
            source_post_id="november-14",
            post_url="https://edwarddouglas.substack.com/p/november-14",
            title="THE WEEKEND WARRIOR November 14, 2025",
            author="Edward Douglas",
            published_at="2025-11-12T12:00:00+00:00",
            published_date="2025-11-12",
            excerpt=None,
            content_html=(
                "<p><strong>Box Office Predictions November 14, 2025</strong><br>"
                "1. <strong>The Running Man </strong>(Paramount) - $24.2 million N/A</p>"
            ),
            source_url="cache",
        )

        predictions = ingest.parse_predictions(post)

        self.assertEqual(1, len(predictions))
        self.assertEqual("2025-11-14", predictions[0].release_date)
        self.assertEqual("edwarddouglas_substack:US_CA:the-running-man", predictions[0].source_movie_id)

    def test_stale_heading_year_is_coerced_near_published_date(self) -> None:
        post = ingest.theory.PostRecord(
            source_post_id="feb-7",
            post_url="https://edwarddouglas.substack.com/p/feb-7",
            title="THE WEEKEND WARRIOR Feb 7, 2024 Reviews",
            author="Edward Douglas",
            published_at="2025-02-05T12:00:00+00:00",
            published_date="2025-02-05",
            excerpt=None,
            content_html=(
                "<p><strong>Box Office Predictions Feb 7, 2024</strong><br>"
                "1. <strong>Dog Man </strong>(Universal) - $19.5 million -46%</p>"
            ),
            source_url="cache",
        )

        predictions = ingest.parse_predictions(post)

        self.assertEqual(1, len(predictions))
        self.assertEqual("2025-02-07", predictions[0].release_date)

    def test_fetcher_caches_rss_response(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self) -> bytes:
                return b"<rss />"

        with tempfile.TemporaryDirectory() as tmp, mock.patch("urllib.request.urlopen", return_value=Response()):
            fetcher = ingest.CachedFeedFetcher(
                Path(tmp),
                refresh=False,
                offline=False,
                delay_seconds=0,
                user_agent=ingest.DEFAULT_USER_AGENT,
            )
            body, cache_path, fetched = fetcher.get(ingest.FEED_URL)

            self.assertTrue(fetched)
            self.assertEqual("<rss />", body)
            self.assertTrue(cache_path.exists())

    def test_parse_post_record_json_extracts_body_html_for_archive_backfill(self) -> None:
        post = ingest.parse_post_record_json(
            SUBSTACK_POST_JSON,
            source_url="https://edwarddouglas.substack.com/api/v1/posts/the-weekend-warrior-show-for-may-25c",
        )

        self.assertEqual("the-weekend-warrior-show-for-may-25c", post.source_post_id)
        self.assertEqual("https://edwarddouglas.substack.com/p/the-weekend-warrior-show-for-may-25c", post.post_url)
        self.assertEqual("2026-05-26T21:05:44+00:00", post.published_at)
        self.assertIn("Box Office Predictions for 6/5/2026", post.content_html)

    def test_full_refresh_forces_archive_discovery(self) -> None:
        args = ingest.build_arg_parser().parse_args(["--full-refresh", "--discovery", "rss", "--max-pages", "1"])

        ingest.configure_full_refresh_args(args)

        self.assertEqual("archive", args.discovery)
        self.assertIsNone(args.max_pages)
        self.assertTrue(args.refresh)
        self.assertEqual(ingest.FULL_REFRESH_START_DATE, args.start_date)


class EdwardDouglasSubstackDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn, self.schema = make_isolated_postgres_schema()
        ingest.initialize_database(self.conn)
        self.conn.commit()

    def tearDown(self) -> None:
        drop_isolated_postgres_schema(self.conn, self.schema)

    def test_import_parse_result_persists_substack_predictions(self) -> None:
        self.conn.execute(
            """
            INSERT INTO movies (title, release_date, movie_url, release_year)
            VALUES ('Backrooms', '2026-05-29', 'https://www.the-numbers.com/movie/Backrooms#tab=summary', 2026)
            """
        )
        post = ingest.parse_rss_posts(SUBSTACK_RSS, source_url=ingest.FEED_URL)[0]
        predictions = ingest.parse_predictions(post)
        result = ingest.theory.PageParseResult(
            post=post,
            predictions=predictions,
            fetched_at="2026-07-06T00:00:00+00:00",
            raw_cache_path=Path("feed.xml"),
            raw_json=SUBSTACK_RSS,
        )

        post_count, prediction_count = ingest.import_parse_result(
            self.conn,
            result,
            issue_source="test_edwarddouglas_substack",
        )

        self.assertEqual(1, post_count)
        self.assertEqual(3, prediction_count)
        row = self.conn.execute(
            """
            SELECT source_movie_title, weekend_forecast_usd, release_date, parser_version
            FROM edwarddouglas_substack_predictions
            WHERE source_movie_title = 'Backrooms'
            """
        ).fetchone()
        self.assertEqual("Backrooms", row[0])
        self.assertEqual(46_200_000, row[1])
        self.assertEqual("2026-05-29", row[2].isoformat())
        self.assertEqual(ingest.PARSER_VERSION, row[3])
        source_row = self.conn.execute(
            """
            SELECT source, source_movie_id, source_title
            FROM movie_source_ids
            WHERE source_title = 'Backrooms'
            """
        ).fetchone()
        self.assertEqual("edwarddouglas_substack", source_row[0])
        self.assertTrue(source_row[1].startswith("edwarddouglas_substack:US_CA:backrooms:"))
        self.assertEqual("Backrooms", source_row[2])

    def test_weekly_forecast_matches_active_holdover_release(self) -> None:
        movie_id = self.conn.execute(
            """
            INSERT INTO movies (title, release_date, movie_url, release_year)
            VALUES ('Zootopia 2', '2025-11-26', 'https://www.the-numbers.com/movie/Zootopia-2-(2025)', 2025)
            RETURNING movie_id
            """
        ).fetchone()[0]
        release_run_id = self.conn.execute(
            """
            INSERT INTO release_runs (movie_id, market, release_type, opening_date, source)
            VALUES (%s, 'US_CA', 'wide', '2025-11-26', 'test')
            RETURNING release_run_id
            """,
            (movie_id,),
        ).fetchone()[0]
        self.conn.execute(
            """
            INSERT INTO daily_box_office (
                release_run_id, box_office_date, market, day_number, gross_usd,
                source, fetched_at, raw_cache_path
            ) VALUES (%s, '2026-01-23', 'US_CA', 59, 1000000, 'test', '2026-01-23', 'test')
            """,
            (release_run_id,),
        )
        post = ingest.theory.PostRecord(
            source_post_id="jan-23",
            post_url="https://edwarddouglas.substack.com/p/jan-23",
            title="THE WEEKEND WARRIOR Jan. 23, 2026",
            author="Edward Douglas",
            published_at="2026-01-21T12:00:00+00:00",
            published_date="2026-01-21",
            excerpt=None,
            content_html=(
                "<p><strong>Box Office Predictions for January 23, 2026</strong><br>"
                "1. <strong>Zootopia 2 </strong>(Disney) - $10 million -40%</p>"
            ),
            source_url="cache",
        )
        result = ingest.theory.PageParseResult(
            post=post,
            predictions=ingest.parse_predictions(post),
            fetched_at="2026-01-21T12:00:00+00:00",
            raw_cache_path=Path("feed.json"),
            raw_json="{}",
        )

        ingest.import_parse_result(self.conn, result, issue_source="test_edwarddouglas_substack")

        row = self.conn.execute(
            """
            SELECT movie_id, match_status, match_method, source_movie_id
            FROM edwarddouglas_substack_predictions
            WHERE source_movie_title = 'Zootopia 2'
            """
        ).fetchone()
        self.assertEqual(movie_id, row[0])
        self.assertEqual("matched", row[1])
        self.assertEqual("weekly_forecast_title_activity_window", row[2])
        self.assertEqual("edwarddouglas_substack:US_CA:zootopia-2", row[3])


if __name__ == "__main__":
    unittest.main()
