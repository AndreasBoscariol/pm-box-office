#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pm_box_office.sources.toddmthatcher import ingest
from tests.postgres_test_utils import drop_isolated_postgres_schema, make_isolated_postgres_schema


ARCHIVE_HTML = """
<html><body>
  <article class="post">
    <h2 class="entry-title">
      <a href="https://toddmthatcher.com/2026/07/09/the-odyssey-box-office-prediction/?utm_source=x">
        The Odyssey Box Office Prediction
      </a>
    </h2>
    <time datetime="2026-07-09T12:00:00-04:00">July 9, 2026</time>
    <div class="entry-summary">The Odyssey opening weekend prediction: $106.2 million</div>
  </article>
  <article class="post">
    <h2 class="entry-title">
      <a href="https://toddmthatcher.com/2026/07/07/july-10-12-box-office-predictions/">July 10-12 Box Office Predictions</a>
    </h2>
    <time datetime="2026-07-07T12:00:00-04:00">July 7, 2026</time>
  </article>
  <article class="post">
    <h2 class="entry-title"><a href="https://toddmthatcher.com/oscar-post/">Oscar Predictions</a></h2>
  </article>
</body></html>
"""


SINGLE_POST_HTML = """
<html>
<head>
  <meta property="article:published_time" content="2026-07-09T12:00:00-04:00" />
</head>
<body>
<article>
  <h1 class="entry-title">The Odyssey Box Office Prediction</h1>
  <div class="entry-content">
    <p>Adapted from Homer, The Odyssey opens July 17th.</p>
    <p>The Odyssey opening weekend prediction: $106.2 million</p>
  </div>
</article>
</body>
</html>
"""


WEEKLY_POST_HTML = """
<html>
<head>
  <meta property="article:published_time" content="2026-07-07T12:00:00-04:00" />
</head>
<body>
<article>
  <h1 class="entry-title">July 10-12 Box Office Predictions</h1>
  <div class="entry-content">
    <p>Here’s how I have it shaking out:</p>
    <p>1. Moana</p>
    <p>Predicted Gross: $54.3 million</p>
    <p>2. Evil Dead Burn</p>
    <p>Predicted Gross: $21.9 million</p>
    <p>Box Office Results (July 3-5)</p>
    <p>Actuals should not be parsed as forecasts.</p>
  </div>
</article>
</body>
</html>
"""


RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel>
    <item>
      <title>The Odyssey Box Office Prediction</title>
      <link>https://toddmthatcher.com/2026/07/09/the-odyssey-box-office-prediction/?utm_source=rss</link>
      <pubDate>Thu, 09 Jul 2026 16:00:00 +0000</pubDate>
      <dc:creator>toddmthatcher</dc:creator>
      <description>The Odyssey opening weekend prediction: $106.2 million</description>
    </item>
    <item>
      <title>Oscar Predictions: Moana</title>
      <link>https://toddmthatcher.com/oscar/</link>
      <pubDate>Wed, 08 Jul 2026 16:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>
"""


class ToddMThatcherParserTests(unittest.TestCase):
    def test_archive_parser_discovers_supported_prediction_posts(self) -> None:
        articles = ingest.parse_archive(ARCHIVE_HTML, source_url=ingest.CATEGORY_URL)

        self.assertEqual(2, len(articles))
        self.assertEqual("https://toddmthatcher.com/2026/07/09/the-odyssey-box-office-prediction/", articles[0].article_url)
        self.assertEqual("2026-07-09", articles[0].published_date)
        self.assertEqual("July 10-12 Box Office Predictions", articles[1].title)

    def test_rss_parser_filters_to_box_office_prediction_posts(self) -> None:
        articles = ingest.parse_rss(RSS_XML, source_url=ingest.CATEGORY_RSS_URL)

        self.assertEqual(1, len(articles))
        self.assertEqual("The Odyssey Box Office Prediction", articles[0].title)
        self.assertEqual("2026-07-09", articles[0].published_date)

    def test_single_movie_post_extracts_opening_prediction_and_release_date(self) -> None:
        article, predictions = ingest.parse_article(
            SINGLE_POST_HTML,
            article_url="https://toddmthatcher.com/2026/07/09/the-odyssey-box-office-prediction/",
        )

        self.assertEqual("The Odyssey Box Office Prediction", article.title)
        self.assertEqual(1, len(predictions))
        self.assertEqual("The Odyssey", predictions[0].source_movie_title)
        self.assertEqual("domestic_opening_weekend", predictions[0].forecast_metric)
        self.assertEqual(106_200_000, predictions[0].weekend_gross_prediction_usd)
        self.assertEqual("2026-07-17", predictions[0].target_start_date)
        self.assertEqual("2026-07-19", predictions[0].target_end_date)

    def test_single_movie_post_infers_next_weekend_release_date(self) -> None:
        html = """
        <html><head>
          <meta property="article:published_time" content="2017-11-07T12:00:00-04:00" />
        </head><body><article>
          <h1 class="entry-title">Justice League Box Office Prediction</h1>
          <div class="entry-content">
            <p>Justice League debuts next weekend.</p>
            <p>Justice League opening weekend prediction: $128.4 million</p>
          </div>
        </article></body></html>
        """

        _article, predictions = ingest.parse_article(
            html,
            article_url="https://toddmthatcher.com/2017/11/07/justice-league-box-office-prediction/",
        )

        self.assertEqual(1, len(predictions))
        self.assertEqual("2017-11-17", predictions[0].target_start_date)
        self.assertEqual("2017-11-19", predictions[0].target_end_date)

    def test_weekly_post_extracts_named_holiday_weekend_dates(self) -> None:
        html = """
        <html><head>
          <meta property="article:published_time" content="2015-11-23T12:00:00-04:00" />
        </head><body><article>
          <h1 class="entry-title">Thanksgiving 2015 Box Office Predictions</h1>
          <div class="entry-content">
            <p>1. Creed</p>
            <p>Predicted Gross: $28 million</p>
          </div>
        </article></body></html>
        """

        _article, predictions = ingest.parse_article(
            html,
            article_url="https://toddmthatcher.com/2015/11/23/thanksgiving-2015-box-office-predictions/",
        )

        self.assertEqual(1, len(predictions))
        self.assertEqual("2015-11-27", predictions[0].target_start_date)
        self.assertEqual("2015-11-29", predictions[0].target_end_date)

    def test_weekly_post_extracts_ranked_predicted_gross_rows_before_results(self) -> None:
        _article, predictions = ingest.parse_article(
            WEEKLY_POST_HTML,
            article_url="https://toddmthatcher.com/2026/07/07/july-10-12-box-office-predictions/",
        )

        self.assertEqual(2, len(predictions))
        self.assertEqual("Moana", predictions[0].source_movie_title)
        self.assertEqual(1, predictions[0].source_rank)
        self.assertEqual(54_300_000, predictions[0].weekend_gross_prediction_usd)
        self.assertEqual("2026-07-10", predictions[0].target_start_date)
        self.assertEqual("2026-07-12", predictions[0].target_end_date)
        self.assertEqual("toddmthatcher:US_CA:moana", predictions[0].source_movie_id)
        self.assertEqual("Evil Dead Burn", predictions[1].source_movie_title)

    def test_full_refresh_args_cover_archive(self) -> None:
        args = ingest.build_arg_parser().parse_args(["--full-refresh"])

        ingest.configure_full_refresh_args(args)

        self.assertTrue(args.refresh)
        self.assertEqual("archive", args.discovery)
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

        with tempfile.TemporaryDirectory() as tmp, mock.patch("urllib.request.urlopen", return_value=Response()):
            fetcher = ingest.HtmlFetcher(
                Path(tmp),
                refresh=False,
                offline=False,
                delay_seconds=0,
                user_agent=ingest.DEFAULT_USER_AGENT,
            )

            body, cache_path, fetched = fetcher.get("https://toddmthatcher.com/sample/")

            self.assertTrue(fetched)
            self.assertEqual("<html></html>", body)
            self.assertTrue(cache_path.exists())


class ToddMThatcherDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn, self.schema = make_isolated_postgres_schema()
        ingest.initialize_database(self.conn)
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS release_runs (
                release_run_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                movie_id BIGINT NOT NULL REFERENCES movies(movie_id),
                market TEXT NOT NULL,
                release_type TEXT,
                opening_date DATE,
                source TEXT
            );

            CREATE TABLE IF NOT EXISTS daily_box_office (
                daily_box_office_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                release_run_id BIGINT NOT NULL REFERENCES release_runs(release_run_id),
                box_office_date DATE NOT NULL,
                market TEXT NOT NULL,
                day_number INTEGER,
                gross_usd BIGINT,
                source TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                raw_cache_path TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def tearDown(self) -> None:
        self.conn.rollback()
        drop_isolated_postgres_schema(self.conn, self.schema)

    def test_prediction_upsert_keeps_article_source_row_key_unique(self) -> None:
        article, predictions = ingest.parse_article(
            WEEKLY_POST_HTML,
            article_url="https://toddmthatcher.com/2026/07/07/july-10-12-box-office-predictions/",
        )
        article_id = ingest.upsert_article(
            self.conn,
            article,
            status="parsed",
            fetched_at="2026-07-07T16:00:00+00:00",
            raw_cache_path=Path("cache.html"),
            html=WEEKLY_POST_HTML,
        )

        ingest.insert_predictions(
            self.conn,
            article_id,
            predictions,
            fetched_at="2026-07-07T16:00:00+00:00",
            raw_cache_path=Path("cache.html"),
        )
        ingest.insert_predictions(
            self.conn,
            article_id,
            predictions,
            fetched_at="2026-07-07T17:00:00+00:00",
            raw_cache_path=Path("cache.html"),
        )
        self.conn.commit()

        row = self.conn.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT source_row_key)
            FROM toddmthatcher_weekend_predictions
            WHERE article_id = %s
            """,
            (article_id,),
        ).fetchone()
        self.assertEqual((2, 2), tuple(row))

    def test_weekly_forecast_matches_active_holdover_release(self) -> None:
        movie_id = self.conn.execute(
            """
            INSERT INTO movies (title, release_date, movie_url, release_year)
            VALUES ('Thor: Ragnarok', '2017-11-03', 'https://www.the-numbers.com/movie/Thor-Ragnarok-(2017)', 2017)
            RETURNING movie_id
            """
        ).fetchone()[0]
        release_run_id = self.conn.execute(
            """
            INSERT INTO release_runs (movie_id, market, release_type, opening_date, source)
            VALUES (%s, 'US_CA', 'wide', '2017-11-03', 'test')
            RETURNING release_run_id
            """,
            (movie_id,),
        ).fetchone()[0]
        self.conn.execute(
            """
            INSERT INTO daily_box_office (
                release_run_id, box_office_date, market, day_number, gross_usd,
                source, fetched_at, raw_cache_path
            ) VALUES (%s, '2017-11-10', 'US_CA', 8, 10000000, 'test', '2017-11-10', 'test')
            """,
            (release_run_id,),
        )
        _article, predictions = ingest.parse_article(
            """
            <html><head>
              <meta property="article:published_time" content="2017-11-06T12:00:00-04:00" />
            </head><body><article>
              <h1 class="entry-title">Box Office Predictions: November 10-12</h1>
              <div class="entry-content">
                <p>1. Thor: Ragnarok</p>
                <p>Predicted Gross: $53.8 million</p>
              </div>
            </article></body></html>
            """,
            article_url="https://toddmthatcher.com/2017/11/06/box-office-predictions-november-10-12/",
        )

        matched = ingest.match_predictions(self.conn, predictions)[0]

        self.assertEqual(movie_id, matched.movie_id)
        self.assertEqual("matched", matched.match_status)
        self.assertEqual("weekly_forecast_title_activity_window", matched.match_method)

    def test_weekly_forecast_matches_holdover_by_release_window_without_daily_rows(self) -> None:
        movie_id = self.conn.execute(
            """
            INSERT INTO movies (title, release_date, movie_url, release_year)
            VALUES ('Coco', '2017-11-22', 'https://www.the-numbers.com/movie/Coco-(2017)', 2017)
            RETURNING movie_id
            """
        ).fetchone()[0]
        _article, predictions = ingest.parse_article(
            """
            <html><head>
              <meta property="article:published_time" content="2017-12-04T12:00:00-04:00" />
            </head><body><article>
              <h1 class="entry-title">Box Office Predictions: December 8-10</h1>
              <div class="entry-content">
                <p>1. Coco</p>
                <p>Predicted Gross: $18.3 million</p>
              </div>
            </article></body></html>
            """,
            article_url="https://toddmthatcher.com/2017/12/04/box-office-predictions-december-8-10/",
        )

        matched = ingest.match_predictions(self.conn, predictions)[0]

        self.assertEqual(movie_id, matched.movie_id)
        self.assertEqual("matched", matched.match_status)
        self.assertEqual("weekly_forecast_title_release_window", matched.match_method)

    def test_single_movie_prediction_without_parsed_date_matches_future_release(self) -> None:
        movie_id = self.conn.execute(
            """
            INSERT INTO movies (title, release_date, movie_url, release_year)
            VALUES ('Coco', '2017-11-22', 'https://www.the-numbers.com/movie/Coco-(2017)', 2017)
            RETURNING movie_id
            """
        ).fetchone()[0]
        _article, predictions = ingest.parse_article(
            """
            <html><head>
              <meta property="article:published_time" content="2017-11-14T12:00:00-04:00" />
            </head><body><article>
              <h1 class="entry-title">Coco Box Office Prediction</h1>
              <div class="entry-content">
                <p>Coco opening weekend prediction: $54.1 million (Friday to Sunday), $74.6 million (Wednesday to Sunday)</p>
              </div>
            </article></body></html>
            """,
            article_url="https://toddmthatcher.com/2017/11/14/coco-box-office-prediction/",
        )

        matched = ingest.match_predictions(self.conn, predictions)[0]

        self.assertIsNone(predictions[0].target_start_date)
        self.assertEqual(movie_id, matched.movie_id)
        self.assertEqual("matched", matched.match_status)
        self.assertEqual("single_movie_title_future_release", matched.match_method)

    def test_single_movie_prediction_prefers_canonical_duplicate_release(self) -> None:
        self.conn.execute(
            """
            INSERT INTO movies (title, release_date, release_year)
            VALUES ('Pressure', '2026-05-29', 2026)
            """
        )
        movie_id = self.conn.execute(
            """
            INSERT INTO movies (title, release_date, movie_url, release_year)
            VALUES ('Pressure', '2026-05-29', 'https://www.the-numbers.com/movie/Pressure-(2026)', 2026)
            RETURNING movie_id
            """
        ).fetchone()[0]
        _article, predictions = ingest.parse_article(
            """
            <html><head>
              <meta property="article:published_time" content="2026-05-24T12:00:00-04:00" />
            </head><body><article>
              <h1 class="entry-title">Pressure Box Office Prediction</h1>
              <div class="entry-content">
                <p>Pressure opening weekend prediction: $4.9 million</p>
              </div>
            </article></body></html>
            """,
            article_url="https://toddmthatcher.com/2026/05/24/pressure-box-office-prediction/",
        )

        matched = ingest.match_predictions(self.conn, predictions)[0]

        self.assertEqual(movie_id, matched.movie_id)
        self.assertEqual("matched", matched.match_status)
        self.assertEqual("single_movie_title_future_release", matched.match_method)

    def test_title_alias_matches_ampersand_catalog_title(self) -> None:
        movie_id = self.conn.execute(
            """
            INSERT INTO movies (title, release_date, movie_url, release_year)
            VALUES ('Pain & Gain (2013)', '2013-04-26', 'https://www.the-numbers.com/movie/Pain-and-Gain', 2013)
            RETURNING movie_id
            """
        ).fetchone()[0]
        _article, predictions = ingest.parse_article(
            """
            <html><head>
              <meta property="article:published_time" content="2013-04-22T12:00:00-04:00" />
            </head><body><article>
              <h1 class="entry-title">Pain and Gain Box Office Prediction</h1>
              <div class="entry-content">
                <p>Pain and Gain opening weekend prediction: $18.5 million</p>
              </div>
            </article></body></html>
            """,
            article_url="https://toddmthatcher.com/2013/04/22/pain-and-gain-box-office-prediction/",
        )

        matched = ingest.match_predictions(self.conn, predictions)[0]

        self.assertEqual(movie_id, matched.movie_id)
        self.assertEqual("matched", matched.match_status)

    def test_rerelease_title_is_ignored(self) -> None:
        _article, predictions = ingest.parse_article(
            """
            <html><head>
              <meta property="article:published_time" content="2025-08-23T12:00:00-04:00" />
            </head><body><article>
              <h1 class="entry-title">Jaws 50th Anniversary Box Office Prediction</h1>
              <div class="entry-content">
                <p>Jaws 50th Anniversary opening weekend prediction: $5.6 million</p>
              </div>
            </article></body></html>
            """,
            article_url="https://toddmthatcher.com/2025/08/23/jaws-50th-anniversary-box-office-prediction/",
        )

        matched = ingest.match_predictions(self.conn, predictions)[0]

        self.assertIsNone(matched.movie_id)
        self.assertEqual("ignored_rerelease", matched.match_status)
        self.assertEqual("rerelease_title_filter", matched.match_method)


if __name__ == "__main__":
    unittest.main()
