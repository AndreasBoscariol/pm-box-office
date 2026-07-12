from __future__ import annotations

from pathlib import Path
import unittest

from pm_box_office.sources.joblo import ingest
from tests.postgres_test_utils import drop_isolated_postgres_schema, make_isolated_postgres_schema


ARCHIVE_HTML = """
<html>
  <body>
    <article>
      <h3><a href="https://www.joblo.com/toy-story-5-will-easily-beat-supergirl/">Box Office Predictions: Toy Story 5 will easily beat Supergirl</a></h3>
      <p class="excerpt">In its second frame, Pixar's Toy Story 5 won't have any trouble topping the box office.</p>
    </article>
    <article>
      <h3><a href="https://www.joblo.com/toy-story-5-has-pixars-2nd-biggest-opening/">Weekend Box Office: Toy Story 5 has Pixar's 2nd biggest opening ever</a></h3>
    </article>
  </body>
</html>
"""


RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel>
    <item>
      <title>Box Office Predictions: Toy Story 5 will easily beat Supergirl</title>
      <link>https://www.joblo.com/toy-story-5-will-easily-beat-supergirl/?utm_source=rss</link>
      <pubDate>Thu, 25 Jun 2026 18:33:47 +0000</pubDate>
      <dc:creator><![CDATA[Chris Bumbray]]></dc:creator>
      <description><![CDATA[<p>In its second frame, Pixar's Toy Story 5 won't have any trouble topping the box office.</p>]]></description>
    </item>
    <item>
      <title>Weekend Box Office: Toy Story 5 has Pixar's 2nd biggest opening ever</title>
      <link>https://www.joblo.com/toy-story-5-has-pixars-2nd-biggest-opening/</link>
      <pubDate>Sun, 28 Jun 2026 18:33:47 +0000</pubDate>
    </item>
  </channel>
</rss>
"""


ARTICLE_HTML = """
<html>
  <head>
    <meta property="article:published_time" content="2026-06-25T10:00:00-04:00" />
    <meta name="author" content="Chris Bumbray" />
  </head>
  <body>
    <h1>Box Office Predictions: Toy Story 5 will easily beat Supergirl</h1>
    <p>As such, expect Toy Story 5 to easily top the box office.</p>
    <h3>Here are our predictions:</h3>
    <ol>
      <li>Toy Story 5: $80 million</li>
      <li>Supergirl: $45 million</li>
      <li>Jackass: Best and Last: $15 million</li>
      <li>Obsession: $10 million</li>
      <li>Backrooms: $5 million</li>
    </ol>
  </body>
</html>
"""


class JobloIngestTests(unittest.TestCase):
    def test_parse_archive_filters_to_prediction_articles(self) -> None:
        articles = ingest.parse_archive(ARCHIVE_HTML, source_url=ingest.ARCHIVE_URL)

        self.assertEqual(1, len(articles))
        self.assertEqual("box_office_predictions", articles[0].article_type)
        self.assertEqual(
            "https://www.joblo.com/toy-story-5-will-easily-beat-supergirl/",
            articles[0].article_url,
        )

    def test_parse_rss_canonicalizes_prediction_links(self) -> None:
        articles = ingest.parse_rss(RSS_XML, source_url=ingest.RSS_URL)

        self.assertEqual(1, len(articles))
        self.assertEqual("Chris Bumbray", articles[0].author)
        self.assertEqual("2026-06-25", articles[0].published_date)
        self.assertEqual(
            "https://www.joblo.com/toy-story-5-will-easily-beat-supergirl/",
            articles[0].article_url,
        )

    def test_parse_article_prediction_list(self) -> None:
        fallback = ingest.ArchiveArticle(
            article_url="https://www.joblo.com/toy-story-5-will-easily-beat-supergirl/",
            title="Box Office Predictions: Toy Story 5 will easily beat Supergirl",
            author=None,
            published_date=None,
            article_type="box_office_predictions",
            source_url=ingest.RSS_URL,
        )

        article, predictions = ingest.parse_article(
            ARTICLE_HTML,
            article_url=fallback.article_url,
            fallback=fallback,
        )

        self.assertEqual("Chris Bumbray", article.author)
        self.assertEqual(5, len(predictions))
        self.assertEqual("Toy Story 5", predictions[0].source_movie_title)
        self.assertEqual(80_000_000, predictions[0].range_low_usd)
        self.assertEqual(80_000_000, predictions[0].range_high_usd)
        self.assertEqual("2026-06-26", predictions[0].target_start_date)
        self.assertEqual("2026-06-28", predictions[0].target_end_date)
        self.assertEqual("domestic_weekend", predictions[0].forecast_metric)

    def test_parse_article_preserves_numeric_movie_title(self) -> None:
        article, predictions = ingest.parse_article(
            """
            <html><head>
              <meta property="article:published_time" content="2025-06-19T10:00:00-04:00" />
            </head><body>
              <h1>Box Office Predictions: Zombies take on Dragons</h1>
              <h3>Here are our predictions:</h3>
              <p>28 Years Later: $35 million</p>
            </body></html>
            """,
            article_url="https://www.joblo.com/zombies-take-on-dragons/",
        )

        self.assertEqual("28 Years Later", predictions[0].source_movie_title)

    def test_prose_parser_rejects_possessive_review_context(self) -> None:
        _article, predictions = ingest.parse_article(
            """
            <html><head>
              <meta property="article:published_time" content="2022-04-07T10:00:00-04:00" />
            </head><body>
              <h1>Box Office Predictions: Sonic The Hedgehog 2 set to race ahead of Ambulance</h1>
              <p>Sonic The Hedgehog 2’s 67% fresh score on Rotten Tomatoes might entice
              those on the fence. I think word of mouth could allow Ambulance to gross
              beyond tracking and that’s why I’m predicting $15 million for the weekend.</p>
            </body></html>
            """,
            article_url="https://www.joblo.com/box-office-predictions-sonic-the-hedgehog-2/",
        )

        self.assertEqual([], predictions)

    def test_list_parser_rejects_long_prose_title_fragment(self) -> None:
        _article, predictions = ingest.parse_article(
            """
            <html><head>
              <meta property="article:published_time" content="2023-07-20T10:00:00-04:00" />
            </head><body>
              <h1>Box Office Predictions: Barbenheimer arrives</h1>
              <p>So now on to the second of our Barbenheimer films, Christopher Nolan's
              R rated 3-hour opus about J. Robert Oppenheimer, a man many deem the most
              important figure in the history of the world: $40 million.</p>
            </body></html>
            """,
            article_url="https://www.joblo.com/box-office-predictions-barbenheimer/",
        )

        self.assertEqual([], predictions)

    def test_prediction_upsert_keeps_article_source_row_key_unique(self) -> None:
        conn, schema_name = make_isolated_postgres_schema()
        try:
            ingest.initialize_database(conn)
            fallback = ingest.ArchiveArticle(
                article_url="https://www.joblo.com/toy-story-5-will-easily-beat-supergirl/",
                title="Box Office Predictions: Toy Story 5 will easily beat Supergirl",
                author=None,
                published_date="2026-06-25",
                article_type="box_office_predictions",
                source_url=ingest.RSS_URL,
            )
            article, predictions = ingest.parse_article(
                ARTICLE_HTML,
                article_url=fallback.article_url,
                fallback=fallback,
            )
            article_id = ingest.upsert_article(
                conn,
                article,
                status="parsed",
                fetched_at="2026-06-25T15:00:00+00:00",
                raw_cache_path=Path("cache.html"),
                html=ARTICLE_HTML,
            )

            ingest.insert_predictions(
                conn,
                article_id,
                predictions,
                fetched_at="2026-06-25T15:00:00+00:00",
                raw_cache_path=Path("cache.html"),
            )
            ingest.insert_predictions(
                conn,
                article_id,
                predictions,
                fetched_at="2026-06-25T16:00:00+00:00",
                raw_cache_path=Path("cache.html"),
            )

            count = conn.execute("SELECT COUNT(*) FROM joblo_weekend_predictions").fetchone()[0]
            fetched_at = conn.execute("SELECT fetched_at FROM joblo_weekend_predictions LIMIT 1").fetchone()[0]
            self.assertEqual(5, count)
            self.assertEqual("2026-06-25T16:00:00+00:00", fetched_at)
        finally:
            drop_isolated_postgres_schema(conn, schema_name)


if __name__ == "__main__":
    unittest.main()
