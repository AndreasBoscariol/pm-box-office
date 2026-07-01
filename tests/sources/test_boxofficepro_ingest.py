#!/usr/bin/env python3
from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from pm_box_office.sources.boxofficepro import ingest
from tests.postgres_test_utils import drop_isolated_postgres_schema, make_isolated_postgres_schema


ARCHIVE_HTML = """
<html>
  <body>
    <article>
      <article class="single-item">
        <a href="https://www.boxofficepro.com/category/forecasts-tracking/" class="small single-item__cat">Forecasts &amp; Tracking</a>
        <a href="https://www.boxofficepro.com/weekend-preview-sample/" class="single-item__link">
          <h2 class="single-item__heading">Weekend Preview: July 3 - 5, 2026</h2>
        </a>
        <div class="single-item__excerpt"><p>The Boxoffice Podium | July 3 - 5, 2026</p></div>
      </article>
      <article class="single-item">
        <a href="https://www.boxofficepro.com/uk-ireland-forecast/" class="single-item__link">
          <h2 class="single-item__heading">U.K. &amp; Ireland Forecast</h2>
        </a>
      </article>
    </article>
  </body>
</html>
"""


WEEKEND_ARTICLE_HTML = """
<html>
  <head>
    <meta property="article:published_time" content="2026-07-01T10:00:00-04:00" />
  </head>
  <body>
    <h1>Weekend Preview: July 3 - 5, 2026</h1>
    <h2>The Boxoffice Podium</h2>
    <h3 class="wp-block-heading">
      1. Minions &amp; Monsters<br>
      Universal/Illumination | NEW<br>
      Opening Weekend Range: $65M - $75M<br>
      Showtime Market Share: 31%
    </h3>
    <h3 class="wp-block-heading">
      2. Toy Story 5<br>
      Disney/Pixar | Week 3<br>
      Weekend Range: $33M - $38M<br>
      Showtime Market Share: 17%
    </h3>
    <h3 class="wp-block-heading">
      Young Washington<br>
      Angel Studios | NEW<br>
      Opening Weekend Range: $12M - $17M<br>
      Showtime Market Share: 8%
    </h3>
    <h3 class="wp-block-heading">
      Supergirl<br>
      Warner Bros./DC Studios | Week 2<br>
      Weekend Range: $12M - $17M<br>
      Showtime Market Share: 11%
    </h3>
    <h3 class="wp-block-heading">
      Partial Block<br>
      Some Studio | NEW<br>
      Weekend Range: TBD
    </h3>
    <h2>Other Tracking Notes</h2>
    <h3 class="wp-block-heading">
      Garbage Movie<br>
      Noise Studio | NEW<br>
      Weekend Range: $1M - $2M
    </h3>
  </body>
</html>
"""


class BoxofficeProParserTests(unittest.TestCase):
    def test_archive_parser_discovers_weekend_cards(self) -> None:
        articles = ingest.parse_archive(
            ARCHIVE_HTML,
            source_url="https://www.boxofficepro.com/category/forecasts-tracking/",
        )

        self.assertEqual(2, len(articles))
        self.assertEqual("https://www.boxofficepro.com/weekend-preview-sample/", articles[0].article_url)
        self.assertEqual("2026-07-03", articles[0].published_date)
        self.assertEqual("weekend_preview", articles[0].article_type)
        self.assertTrue(ingest.is_weekend_preview_article(articles[0]))
        self.assertFalse(ingest.is_domestic_article(articles[1]))
        self.assertFalse(ingest.is_weekend_preview_article(articles[1]))

    def test_archive_discovery_stops_after_later_block_when_articles_exist(self) -> None:
        class FixtureFetcher:
            def get(self, url: str):
                if url.endswith("/page/2/"):
                    raise ingest.FetchBlocked("blocked fixture")
                return ARCHIVE_HTML, Path("archive.html"), False

        args = type(
            "Args",
            (),
            {
                "max_pages": 2,
                "start_date": ingest.dt.date(2026, 7, 1),
                "end_date": ingest.dt.date(2026, 7, 31),
                "max_articles": None,
            },
        )()

        with redirect_stderr(io.StringIO()):
            articles = ingest.discover_articles(FixtureFetcher(), args)

        self.assertEqual(1, len(articles))
        self.assertEqual("Weekend Preview: July 3 - 5, 2026", articles[0].title)

    def test_weekend_preview_parser_extracts_podium_blocks(self) -> None:
        article, predictions, rejected = ingest.parse_article(
            WEEKEND_ARTICLE_HTML,
            article_url="https://www.boxofficepro.com/weekend-preview-sample/",
        )

        self.assertEqual("weekend_preview", article.article_type)
        self.assertEqual("2026-07-01", article.published_date)
        self.assertEqual(4, len(predictions))
        self.assertEqual(1, len(rejected))
        self.assertEqual("weekend_block_missing_required_fields", rejected[0].reason)

        minions = predictions[0]
        self.assertEqual("Minions & Monsters", minions.source_movie_title)
        self.assertEqual("minions and monsters", minions.normalized_movie_title)
        self.assertEqual("Universal/Illumination", minions.distributor)
        self.assertEqual("NEW", minions.release_status)
        self.assertEqual(1, minions.source_rank)
        self.assertEqual("domestic_opening_weekend", minions.forecast_metric)
        self.assertEqual(65_000_000, minions.range_low_usd)
        self.assertEqual(75_000_000, minions.range_high_usd)
        self.assertEqual(31.0, minions.showtime_market_share_pct)
        self.assertEqual("2026-07-03", minions.target_start_date)
        self.assertEqual("2026-07-05", minions.target_end_date)
        self.assertEqual("weekend_podium", minions.source_context)
        self.assertEqual(ingest.PARSER_VERSION, minions.parser_version)
        self.assertIn("Opening Weekend Range: $65M - $75M", minions.raw_forecast_text)

        toy_story = predictions[1]
        self.assertEqual("Toy Story 5", toy_story.source_movie_title)
        self.assertEqual("domestic_weekend", toy_story.forecast_metric)
        self.assertEqual("Week 3", toy_story.release_status)
        self.assertEqual(17.0, toy_story.showtime_market_share_pct)

        young_washington = predictions[2]
        self.assertIsNone(young_washington.source_rank)
        self.assertEqual("Young Washington", young_washington.source_movie_title)
        self.assertEqual("Angel Studios", young_washington.distributor)

        self.assertNotIn("Garbage Movie", [prediction.source_movie_title for prediction in predictions])

    def test_title_normalization_for_the_numbers_matching(self) -> None:
        self.assertEqual(
            ingest.normalize_movie_title("Disney's  Sample Movie (2026)"),
            ingest.normalize_movie_title("Sample   Movie"),
        )
        self.assertEqual(
            ingest.normalize_movie_title("Mission: Impossible - Final Reckoning"),
            ingest.normalize_movie_title("Mission Impossible Final Reckoning"),
        )


class BoxofficeProPostgresTests(unittest.TestCase):
    def test_clean_tables_are_idempotent_and_match_the_numbers_movies(self) -> None:
        conn, schema = make_isolated_postgres_schema()
        try:
            conn.executescript(
                """
                CREATE TABLE movies (
                    movie_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                    movie_url TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    release_year INTEGER
                );
                CREATE TABLE boxofficepro_forecast_articles (article_url TEXT PRIMARY KEY);
                CREATE TABLE boxofficepro_predictions (source_movie_title TEXT);
                CREATE TABLE boxofficepro_import_issues (details TEXT);
                INSERT INTO movies (movie_url, title, release_year)
                VALUES
                    ('https://www.the-numbers.com/movie/Minions-and-Monsters-(2026)', 'Minions & Monsters (2026)', 2026),
                    ('https://www.the-numbers.com/movie/Toy-Story-5-(2026)', 'Toy Story 5 (2026)', 2026),
                    ('https://www.the-numbers.com/movie/Supergirl-(2026)', 'Supergirl (2026)', 2026),
                    ('https://www.the-numbers.com/movie/Duplicate-Title-(2025)', 'Duplicate Title (2025)', 2025),
                    ('https://www.the-numbers.com/movie/Duplicate-Title-(2027)', 'Duplicate Title (2027)', 2027);
                """
            )
            ingest.initialize_database(conn)
            ingest.initialize_database(conn)
            conn.commit()

            old_tables = conn.execute(
                """
                SELECT to_regclass('boxofficepro_forecast_articles'),
                       to_regclass('boxofficepro_predictions'),
                       to_regclass('boxofficepro_import_issues')
                """
            ).fetchone()
            self.assertEqual((None, None, None), tuple(old_tables))

            article, predictions, rejected = ingest.parse_article(
                WEEKEND_ARTICLE_HTML,
                article_url="https://www.boxofficepro.com/weekend-preview-sample/",
            )
            self.assertEqual(4, len(predictions))
            self.assertEqual(1, len(rejected))

            predictions = ingest.match_predictions(conn, predictions)

            with tempfile.TemporaryDirectory() as tmp:
                cache_path = Path(tmp) / "fixture.html"
                cache_path.write_text(WEEKEND_ARTICLE_HTML, encoding="utf-8")
                article_id = ingest.upsert_article(
                    conn,
                    article,
                    status="parsed",
                    fetched_at="2026-07-01T00:00:00+00:00",
                    raw_cache_path=cache_path,
                    html=WEEKEND_ARTICLE_HTML,
                )
                for _ in range(2):
                    ingest.insert_predictions(
                        conn,
                        article_id,
                        predictions,
                        fetched_at="2026-07-01T00:00:00+00:00",
                        raw_cache_path=cache_path,
                    )
                    conn.commit()

            self.assertTrue(ingest.article_already_parsed(conn, article.article_url))

            rows = conn.execute(
                """
                SELECT
                    p.source_movie_title,
                    p.match_status,
                    m.title,
                    p.distributor,
                    p.release_status,
                    p.forecast_metric,
                    p.showtime_market_share_pct
                FROM boxofficepro_weekend_predictions p
                LEFT JOIN movies m ON m.movie_id = p.matched_movie_id
                ORDER BY p.row_ordinal
                """
            ).fetchall()

            self.assertEqual(4, len(rows))
            self.assertEqual(
                (
                    "Minions & Monsters",
                    "matched",
                    "Minions & Monsters (2026)",
                    "Universal/Illumination",
                    "NEW",
                    "domestic_opening_weekend",
                    31.0,
                ),
                tuple(rows[0]),
            )
            self.assertEqual(("Toy Story 5", "matched", "Toy Story 5 (2026)"), tuple(rows[1][:3]))
            self.assertEqual(("Young Washington", "unmatched", None), tuple(rows[2][:3]))
            self.assertEqual(("Supergirl", "matched", "Supergirl (2026)"), tuple(rows[3][:3]))
        finally:
            drop_isolated_postgres_schema(conn, schema)


if __name__ == "__main__":
    unittest.main()
