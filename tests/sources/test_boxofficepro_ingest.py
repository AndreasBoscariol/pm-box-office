#!/usr/bin/env python3
from __future__ import annotations

import io
import tempfile
import unittest
from unittest import mock
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
    <h2>The Battle for Second</h2>
    <h3 class="wp-block-heading">
      2. Toy Story 5<br>
      Disney/Pixar | Week 3<br>
      Weekend Range: $33M - $38M<br>
      Showtime Market Share: 17%
    </h3>
    <h2>Battle for Third</h2>
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


LEGACY_WEEKEND_TABLE_HTML = """
<html>
  <head>
    <meta property="article:published_time" content="2016-02-10T10:00:00-04:00" />
  </head>
  <body>
    <h1>Weekend Forecast: 'Deadpool', 'Kung Fu Panda 3', 'Zoolander 2' &amp; 'How to Be Single'</h1>
    <p><strong>Check out our complete four-day forecast in the table below.</strong></p>
    <table>
      <thead>
        <tr>
          <th>Title</th>
          <th>Release Date</th>
          <th>Distributor</th>
          <th>Weekend</th>
          <th>Domestic Total Through Monday, Feb 15</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Deadpool</td>
          <td>Feb 12, 2016</td>
          <td>Fox</td>
          <td>$93,000,000</td>
          <td>$93,000,000</td>
        </tr>
        <tr>
          <td>Kung Fu Panda 3</td>
          <td>Jan 29, 2016</td>
          <td>Fox / DreamWorks Animation</td>
          <td>$23,000,000</td>
          <td>$97,200,000</td>
        </tr>
      </tbody>
    </table>
  </body>
</html>
"""


LEGACY_WEEKEND_STATUS_TABLE_HTML = """
<html>
  <head>
    <meta property="article:published_time" content="2023-09-20T12:00:00-04:00" />
  </head>
  <body>
    <h1>Weekend Box Office Forecast: EXPEND4BLES Aims to Lead a Sluggish Late September Frame</h1>
    <table>
      <tr>
        <th>Film</th>
        <th>Studio</th>
        <th>3-Day Weekend Forecast</th>
        <th>Projected Domestic Total through Sunday, September 24</th>
        <th>Fri Location Count Projection (as of Wed)</th>
        <th>3-Day % Change from Last Wknd</th>
      </tr>
      <tr>
        <td>Expend4bles</td>
        <td>Lionsgate</td>
        <td>$11,700,000</td>
        <td>$11,700,000</td>
        <td>~3,400</td>
        <td>NEW</td>
      </tr>
      <tr>
        <td>The Nun II</td>
        <td>Warner Bros.</td>
        <td>$8,200,000</td>
        <td>$69,100,000</td>
        <td>~3,536</td>
        <td>-42%</td>
      </tr>
    </table>
  </body>
</html>
"""


LEGACY_STANDALONE_HEADING_HTML = """
<html>
  <head>
    <meta property="article:published_time" content="2023-12-21T12:00:00-04:00" />
  </head>
  <body>
    <h1>4-Day Christmas Weekend Forecast: AQUAMAN AND THE LOST KINGDOM, MIGRATION, and THE COLOR PURPLE</h1>
    <h2>Aquaman and the Lost Kingdom</h2>
    <h3>Warner Bros.</h3>
    <h3>December 22, 2023 (WIDE)</h3>
    <h3>4-Day Opening Weekend Range: $29M-$40M</h3>
    <h2>Migration</h2>
    <h3>Universal</h3>
    <h3>December 22, 2023 (WIDE)</h3>
    <h3>4-Day Opening Weekend Range: $15M-$20M</h3>
  </body>
</html>
"""


LEGACY_PODIUM_AGGREGATE_HTML = """
<html>
  <head>
    <meta property="article:published_time" content="2024-06-26T12:00:00-04:00" />
  </head>
  <body>
    <h1>Weekend Preview: Sample</h1>
    <h2>The Boxoffice Podium</h2>
    <h3>Forecasting the Top 3 Movies at the Domestic Box Office | June 21 – 23, 2024</h3>
    <h3>Week 26 | June 28 – 30, 2024</h3>
    <h3>Top 10 Range | Weekend 26, 2024: $135M — $175M</h3>
    <h3>
      1. Actual Movie<br>
      Actual Studio | NEW<br>
      Opening Weekend Range: $50M - $55M
    </h3>
  </body>
</html>
"""


LONG_RANGE_FORECAST_HTML = """
<html>
  <head>
    <meta property="article:published_time" content="2026-06-19T12:00:00-04:00" />
  </head>
  <body>
    <h1>Long Range Forecast: MINIONS &amp; MONSTERS Set to Dominate July Fourth Long Weekend</h1>
    <div class="entry-content">
      <p>Long Range Forecast - July 1, 2026</p>
      <p>
        <strong><em>Minions &amp; Monsters</em></strong> | Universal<br>
        Domestic Opening Weekend Range: $75M - $85M (3-Day); $90M - $110M (5-Day)<br>
        Domestic Total Range: $240M - $310M
      </p>
      <p>
        <strong><em>Counterprogramming Movie</em></strong><br>
        Opening Weekend Range: $8 - 12 million<br>
        Domestic Total Range: $20 - $35 million
      </p>
    </div>
  </body>
</html>
"""


RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"
  xmlns:dc="http://purl.org/dc/elements/1.1/"
  xmlns:content="http://purl.org/rss/1.0/modules/content/">
  <channel>
    <title>Forecasts &amp; Tracking Archives - Boxoffice Pro</title>
    <item>
      <title>Weekend Preview: MINIONS &amp; MONSTERS Set to Lead July 4 Weekend</title>
      <link>https://www.boxofficepro.com/weekend-preview-minions-monsters-set-to-lead-july-4-weekend/?utm_source=rss&amp;utm_medium=rss</link>
      <pubDate>Tue, 30 Jun 2026 18:33:47 +0000</pubDate>
      <dc:creator><![CDATA[Boxoffice Staff]]></dc:creator>
      <description><![CDATA[<p>The Boxoffice Podium | July 3 - 5, 2026</p>]]></description>
    </item>
    <item>
      <title>Long Range Forecast: Summer Rolls On</title>
      <link>https://www.boxofficepro.com/long-range-forecast-summer-rolls-on/</link>
      <pubDate>Fri, 26 Jun 2026 12:00:00 +0000</pubDate>
      <dc:creator><![CDATA[Boxoffice Staff]]></dc:creator>
      <description><![CDATA[<p>Long range notes.</p>]]></description>
    </item>
    <item>
      <title>U.K. and Ireland Forecast: SUPERGIRL</title>
      <link>https://www.boxofficepro.com/uk-ireland-supergirl/</link>
      <pubDate>Wed, 25 Jun 2026 12:00:00 +0000</pubDate>
      <dc:creator><![CDATA[Boxoffice Staff]]></dc:creator>
      <description><![CDATA[<p>International notes.</p>]]></description>
    </item>
    <item>
      <title>Weekend Preview: SCARY MOVIE Ramps Up the R-Rated Comedy Revival at the Box Office</title>
      <link>https://www.boxofficepro.com/weekend-preview-scary-movie-to-scare-up-big-laughs-and-big-grosses/</link>
      <pubDate>Wed, 03 Jun 2026 16:50:28 +0000</pubDate>
      <dc:creator><![CDATA[Boxoffice Staff]]></dc:creator>
      <description><![CDATA[<p>The Boxoffice Podium | June 5 - 7, 2026</p>]]></description>
    </item>
  </channel>
</rss>
"""


ARCHIVE_HTML_WITH_RSS_DUPLICATE = """
<html>
  <body>
    <article class="single-item">
      <a href="https://www.boxofficepro.com/weekend-preview-minions-monsters-set-to-lead-july-4-weekend/" class="single-item__link">
        <h2 class="single-item__heading">Weekend Preview: MINIONS &amp; MONSTERS Set to Lead July 4 Weekend</h2>
      </a>
      <time class="single-item__date" datetime="2026-06-30T18:33:47+00:00">Jun 30th</time>
      <div class="single-item__excerpt"><p>The Boxoffice Podium | July 3 - 5, 2026</p></div>
    </article>
    <article class="single-item">
      <a href="https://www.boxofficepro.com/weekend-preview-backrooms-poised-to-become-biggest-box-office-surprise-of-2026/" class="single-item__link">
        <h2 class="single-item__heading">Weekend Preview: BACKROOMS Poised to Become Biggest Box Office Surprise of 2026</h2>
      </a>
      <time class="single-item__date" datetime="2026-05-27T12:00:00+00:00">May 27th</time>
      <div class="single-item__excerpt"><p>The Boxoffice Podium | May 29 - 31, 2026</p></div>
    </article>
  </body>
</html>
"""


class BoxofficeProParserTests(unittest.TestCase):
    def test_rss_parser_discovers_weekend_items_with_metadata(self) -> None:
        articles = ingest.parse_rss(RSS_XML, source_url=ingest.FORECAST_RSS_URL)

        self.assertEqual(4, len(articles))
        minions = articles[0]
        self.assertEqual(
            "https://www.boxofficepro.com/weekend-preview-minions-monsters-set-to-lead-july-4-weekend/",
            minions.article_url,
        )
        self.assertEqual("Weekend Preview: MINIONS & MONSTERS Set to Lead July 4 Weekend", minions.title)
        self.assertEqual("Boxoffice Staff", minions.author)
        self.assertEqual("2026-06-30", minions.published_date)
        self.assertEqual("weekend_preview", minions.article_type)
        self.assertEqual("The Boxoffice Podium | July 3 - 5, 2026", minions.excerpt)

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

    def test_auto_discovery_uses_rss_only_when_feed_covers_start_date(self) -> None:
        class FixtureFetcher:
            def __init__(self) -> None:
                self.urls: list[str] = []

            def get(self, url: str):
                self.urls.append(url)
                return RSS_XML, Path("feed.xml"), False

        args = type(
            "Args",
            (),
            {
                "discovery": "auto",
                "max_pages": 3,
                "start_date": ingest.dt.date(2026, 6, 10),
                "end_date": ingest.dt.date(2026, 7, 10),
                "max_articles": None,
            },
        )()
        fetcher = FixtureFetcher()

        with redirect_stderr(io.StringIO()):
            articles = ingest.discover_articles(fetcher, args)

        self.assertEqual([ingest.FORECAST_RSS_URL], fetcher.urls)
        self.assertEqual(2, len(articles))
        self.assertEqual(
            [
                "Long Range Forecast: Summer Rolls On",
                "Weekend Preview: MINIONS & MONSTERS Set to Lead July 4 Weekend",
            ],
            [article.title for article in articles],
        )

    def test_auto_discovery_adds_archive_when_start_date_predates_rss_window_and_dedupes(self) -> None:
        class FixtureFetcher:
            def __init__(self) -> None:
                self.urls: list[str] = []

            def get(self, url: str):
                self.urls.append(url)
                if url == ingest.FORECAST_RSS_URL:
                    return RSS_XML, Path("feed.xml"), False
                return ARCHIVE_HTML_WITH_RSS_DUPLICATE, Path("archive.html"), False

        args = type(
            "Args",
            (),
            {
                "discovery": "auto",
                "max_pages": 1,
                "start_date": ingest.dt.date(2026, 5, 1),
                "end_date": ingest.dt.date(2026, 7, 10),
                "max_articles": None,
            },
        )()
        fetcher = FixtureFetcher()

        with redirect_stderr(io.StringIO()):
            articles = ingest.discover_articles(fetcher, args)

        self.assertEqual([ingest.FORECAST_RSS_URL, ingest.archive_url(1)], fetcher.urls)
        self.assertEqual(
            [
                "https://www.boxofficepro.com/weekend-preview-backrooms-poised-to-become-biggest-box-office-surprise-of-2026/",
                "https://www.boxofficepro.com/weekend-preview-scary-movie-to-scare-up-big-laughs-and-big-grosses/",
                "https://www.boxofficepro.com/long-range-forecast-summer-rolls-on/",
                "https://www.boxofficepro.com/weekend-preview-minions-monsters-set-to-lead-july-4-weekend/",
            ],
            [article.article_url for article in articles],
        )

    def test_full_refresh_args_force_all_archive_and_rss_settings(self) -> None:
        parser = ingest.build_arg_parser()
        args = parser.parse_args(["--full-refresh"])

        ingest.configure_full_refresh_args(args)

        self.assertTrue(args.refresh)
        self.assertEqual("auto", args.discovery)
        self.assertEqual(ingest.FULL_REFRESH_START_DATE, args.start_date)
        self.assertEqual(ingest.FULL_REFRESH_END_DATE, args.end_date)
        self.assertEqual(ingest.FULL_REFRESH_MAX_PAGES, args.max_pages)

    def test_full_refresh_discovery_uses_rss_and_archive_even_when_rss_covers_dates(self) -> None:
        class FixtureFetcher:
            def __init__(self) -> None:
                self.urls: list[str] = []

            def get(self, url: str):
                self.urls.append(url)
                if url == ingest.FORECAST_RSS_URL:
                    return RSS_XML, Path("feed.xml"), False
                if url == ingest.archive_url(1):
                    return ARCHIVE_HTML_WITH_RSS_DUPLICATE, Path("archive.html"), False
                return "<html><body></body></html>", Path("empty.html"), False

        args = type(
            "Args",
            (),
            {
                "full_refresh": True,
                "refresh": False,
                "discovery": "rss",
                "max_pages": 2,
                "start_date": ingest.dt.date(2026, 6, 10),
                "end_date": ingest.dt.date(2026, 7, 10),
                "max_articles": None,
            },
        )()
        ingest.configure_full_refresh_args(args)
        fetcher = FixtureFetcher()

        with redirect_stderr(io.StringIO()):
            articles = ingest.discover_articles(fetcher, args)

        self.assertEqual([ingest.FORECAST_RSS_URL, ingest.archive_url(1), ingest.archive_url(2)], fetcher.urls)
        self.assertIn(
            "https://www.boxofficepro.com/weekend-preview-backrooms-poised-to-become-biggest-box-office-surprise-of-2026/",
            [article.article_url for article in articles],
        )
        self.assertIn(
            "https://www.boxofficepro.com/long-range-forecast-summer-rolls-on/",
            [article.article_url for article in articles],
        )

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
        self.assertEqual(3, young_washington.source_rank)
        self.assertEqual("Young Washington", young_washington.source_movie_title)
        self.assertEqual("Angel Studios", young_washington.distributor)

        supergirl = predictions[3]
        self.assertEqual(3, supergirl.source_rank)

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

    def test_holdover_with_opening_weekend_label_is_domestic_weekend(self) -> None:
        prediction = ingest.parse_weekend_movie_block(
            """
            3. Backrooms
            A24 | Week 2
            Opening Weekend Range: $30M - $35M
            Showtime Market Share: 13%
            """,
            article_url="https://www.boxofficepro.com/weekend-preview-sample/",
            target_start_date="2026-06-05",
            target_end_date="2026-06-07",
            row_ordinal=1,
        )

        self.assertIsNotNone(prediction)
        assert prediction is not None
        self.assertEqual("domestic_weekend", prediction.forecast_metric)

    def test_wildcard_prefix_is_removed_from_movie_title(self) -> None:
        prediction = ingest.parse_weekend_movie_block(
            """
            Wildcard: Longlegs
            Neon | NEW
            Opening Weekend Range: $7M - $10M
            Showtime Market Share: 10%
            """,
            article_url="https://www.boxofficepro.com/weekend-preview-sample/",
            target_start_date="2024-07-12",
            target_end_date="2024-07-14",
            row_ordinal=1,
            fallback_rank=4,
        )

        self.assertIsNotNone(prediction)
        assert prediction is not None
        self.assertEqual("Longlegs", prediction.source_movie_title)
        self.assertEqual("longlegs", prediction.normalized_movie_title)
        self.assertEqual(4, prediction.source_rank)
        self.assertEqual("Longlegs", ingest.clean_legacy_movie_title("Wild Card: Longlegs"))

    def test_weekend_three_day_range_label_is_parsed(self) -> None:
        prediction = ingest.parse_weekend_movie_block(
            """
            1. Moana 2
            Walt Disney Pictures | NEW
            Weekend 3-Day Range: $140M - $170M
            Weekend 5-Day Range: $170M - $200M
            """,
            article_url="https://www.boxofficepro.com/weekend-preview-moana-2/",
            target_start_date="2024-11-27",
            target_end_date="2024-12-01",
            row_ordinal=1,
        )

        self.assertIsNotNone(prediction)
        assert prediction is not None
        self.assertEqual(140_000_000, prediction.range_low_usd)
        self.assertEqual(170_000_000, prediction.range_high_usd)

    def test_boxoffice_barometer_and_h2_movie_blocks_are_parsed(self) -> None:
        blocks = [
            ingest.HeadingBlock(level=2, text="Boxoffice Barometer"),
            ingest.HeadingBlock(
                level=3,
                text="Forecasting the Top 3 Movies at the Domestic Box Office\nApril 19-21, 2024",
            ),
            ingest.HeadingBlock(
                level=2,
                text=(
                    "1. Abigail\n"
                    "Universal Pictures | NEW\n"
                    "Opening Weekend Range: $20M - $25M\n"
                    "Showtime Marketshare (US): 17%"
                ),
            ),
            ingest.HeadingBlock(level=2, text="Long Range Forecast: Related Story"),
        ]

        predictions, rejected = ingest.parse_weekend_podium_blocks(
            blocks,
            article_url="https://www.boxofficepro.com/weekend-preview-abigail/",
            article_title="Weekend Preview: ABIGAIL Pacing to Outgun Henry Cavill for #1",
            published_date="2024-04-17",
        )

        self.assertEqual([], rejected)
        self.assertEqual(1, len(predictions))
        self.assertEqual("Abigail", predictions[0].source_movie_title)
        self.assertEqual("2024-04-19", predictions[0].target_start_date)

    def test_legacy_weekend_forecast_table_extracts_exact_rows(self) -> None:
        article, predictions, rejected = ingest.parse_article(
            LEGACY_WEEKEND_TABLE_HTML,
            article_url="https://www.boxofficepro.com/weekend-forecast-deadpool/",
        )

        self.assertEqual("weekend_preview", article.article_type)
        self.assertEqual("2016-02-10", article.published_date)
        self.assertEqual([], rejected)
        self.assertEqual(2, len(predictions))

        deadpool = predictions[0]
        self.assertEqual("Deadpool", deadpool.source_movie_title)
        self.assertEqual("Fox", deadpool.distributor)
        self.assertEqual("NEW", deadpool.release_status)
        self.assertEqual("domestic_opening_weekend", deadpool.forecast_metric)
        self.assertEqual(93_000_000, deadpool.range_low_usd)
        self.assertEqual(93_000_000, deadpool.range_high_usd)
        self.assertEqual("2016-02-12", deadpool.target_start_date)
        self.assertEqual("2016-02-15", deadpool.target_end_date)
        self.assertEqual("legacy_forecast_table", deadpool.source_context)
        self.assertEqual(ingest.LEGACY_TABLE_PARSER_VERSION, deadpool.parser_version)

        panda = predictions[1]
        self.assertEqual("Kung Fu Panda 3", panda.source_movie_title)
        self.assertEqual("Week 3", panda.release_status)
        self.assertEqual("domestic_weekend", panda.forecast_metric)

    def test_legacy_status_table_without_release_dates_extracts_rows(self) -> None:
        _article, predictions, rejected = ingest.parse_article(
            LEGACY_WEEKEND_STATUS_TABLE_HTML,
            article_url="https://www.boxofficepro.com/weekend-box-office-forecast-expend4bles/",
        )

        self.assertEqual([], rejected)
        self.assertEqual(2, len(predictions))
        self.assertEqual("Expend4bles", predictions[0].source_movie_title)
        self.assertEqual("NEW", predictions[0].release_status)
        self.assertEqual("domestic_opening_weekend", predictions[0].forecast_metric)
        self.assertEqual("The Nun II", predictions[1].source_movie_title)
        self.assertEqual("Change From Last Weekend: -42%", predictions[1].release_status)
        self.assertEqual("domestic_weekend", predictions[1].forecast_metric)

    def test_legacy_standalone_heading_forecasts_extract_rows(self) -> None:
        _article, predictions, rejected = ingest.parse_article(
            LEGACY_STANDALONE_HEADING_HTML,
            article_url="https://www.boxofficepro.com/4-day-christmas-weekend-forecast/",
        )

        self.assertEqual([], rejected)
        self.assertEqual(2, len(predictions))
        self.assertEqual("Aquaman and the Lost Kingdom", predictions[0].source_movie_title)
        self.assertEqual("Warner Bros.", predictions[0].distributor)
        self.assertEqual(29_000_000, predictions[0].range_low_usd)
        self.assertEqual(40_000_000, predictions[0].range_high_usd)
        self.assertEqual("2023-12-22", predictions[0].target_start_date)
        self.assertEqual("2023-12-25", predictions[0].target_end_date)
        self.assertEqual("legacy_forecast_heading", predictions[0].source_context)

    def test_podium_aggregate_heading_is_not_parsed_as_a_movie(self) -> None:
        _article, predictions, rejected = ingest.parse_article(
            LEGACY_PODIUM_AGGREGATE_HTML,
            article_url="https://www.boxofficepro.com/weekend-preview-podium-aggregate/",
        )

        self.assertEqual([], rejected)
        self.assertEqual(["Actual Movie"], [prediction.source_movie_title for prediction in predictions])
        self.assertEqual(["weekend_podium"], [prediction.source_context for prediction in predictions])

    def test_long_range_forecast_text_extracts_opening_and_total_ranges(self) -> None:
        article, predictions, rejected = ingest.parse_article(
            LONG_RANGE_FORECAST_HTML,
            article_url="https://www.boxofficepro.com/long-range-forecast-minions-monsters/",
        )

        self.assertEqual("long_range_forecast", article.article_type)
        self.assertEqual([], rejected)
        self.assertEqual(5, len(predictions))

        minions_opening = predictions[0]
        self.assertEqual("Minions & Monsters", minions_opening.source_movie_title)
        self.assertEqual("Universal", minions_opening.distributor)
        self.assertEqual("domestic_opening_weekend", minions_opening.forecast_metric)
        self.assertEqual(75_000_000, minions_opening.range_low_usd)
        self.assertEqual(85_000_000, minions_opening.range_high_usd)
        self.assertEqual("2026-07-01", minions_opening.target_start_date)
        self.assertEqual("2026-07-03", minions_opening.target_end_date)
        self.assertEqual("generic_forecast_range", minions_opening.source_context)
        self.assertEqual(ingest.GENERIC_FORECAST_PARSER_VERSION, minions_opening.parser_version)

        minions_five_day = predictions[1]
        self.assertEqual("domestic_opening_weekend", minions_five_day.forecast_metric)
        self.assertEqual(90_000_000, minions_five_day.range_low_usd)
        self.assertEqual(110_000_000, minions_five_day.range_high_usd)
        self.assertEqual("2026-07-01", minions_five_day.target_start_date)
        self.assertEqual("2026-07-05", minions_five_day.target_end_date)

        minions_total = predictions[2]
        self.assertEqual("domestic_total", minions_total.forecast_metric)
        self.assertEqual(240_000_000, minions_total.range_low_usd)
        self.assertEqual(310_000_000, minions_total.range_high_usd)
        self.assertIsNone(minions_total.target_start_date)

        counterprogramming = predictions[3]
        self.assertEqual("Counterprogramming Movie", counterprogramming.source_movie_title)
        self.assertEqual("Unknown", counterprogramming.distributor)
        self.assertEqual("domestic_opening_weekend", counterprogramming.forecast_metric)
        self.assertEqual(8_000_000, counterprogramming.range_low_usd)
        self.assertEqual(12_000_000, counterprogramming.range_high_usd)

    def test_parse_workers_above_one_requires_offline_mode(self) -> None:
        args = ingest.build_arg_parser().parse_args(["--parse-workers", "2", "--dry-run"])

        with self.assertRaisesRegex(SystemExit, "--parse-workers greater than 1 requires --offline"):
            ingest.validate_args(args)

    def test_cached_articles_parse_concurrently_in_input_order(self) -> None:
        articles = [
            ingest.ArchiveArticle(
                article_url="https://www.boxofficepro.com/weekend-preview-sample/",
                title="Weekend Preview: July 3 - 5, 2026",
                author=None,
                published_date="2026-07-01",
                article_type="weekend_preview",
                source_url=ingest.FORECAST_ARCHIVE_URL,
            ),
            ingest.ArchiveArticle(
                article_url="https://www.boxofficepro.com/long-range-forecast-minions-monsters/",
                title="Long Range Forecast: MINIONS & MONSTERS",
                author=None,
                published_date="2026-06-19",
                article_type="long_range_forecast",
                source_url=ingest.FORECAST_ARCHIVE_URL,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            ingest.cache_path_for_url(cache_dir, articles[0].article_url).write_text(
                WEEKEND_ARTICLE_HTML,
                encoding="utf-8",
            )
            ingest.cache_path_for_url(cache_dir, articles[1].article_url).write_text(
                LONG_RANGE_FORECAST_HTML,
                encoding="utf-8",
            )

            results = ingest.parse_cached_articles_concurrently(articles, cache_dir=cache_dir, workers=2)

        self.assertEqual([article.article_url for article in articles], [result.archive_article.article_url for result in results])
        self.assertEqual("weekend_preview", results[0].article.article_type)
        self.assertEqual(4, len(results[0].predictions))
        self.assertEqual("long_range_forecast", results[1].article.article_type)
        self.assertEqual(5, len(results[1].predictions))

    def test_fetcher_caches_successful_http_response(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self) -> bytes:
                return b"<rss></rss>"

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
                fetch_mode="http",
            )
            body, cache_path, fetched = fetcher.get(ingest.FORECAST_RSS_URL)
            cached_body = cache_path.read_text(encoding="utf-8")

        self.assertEqual("<rss></rss>", body)
        self.assertTrue(fetched)
        self.assertEqual(".xml", cache_path.suffix)
        self.assertEqual("<rss></rss>", cached_body)

    def test_fetcher_falls_back_to_browser_after_http_block(self) -> None:
        class FixtureFetcher(ingest.HtmlFetcher):
            def _get_http(self, url: str) -> str:
                raise ingest.FetchBlocked("HTTP 403 fixture")

            def _get_browser(self, url: str) -> str:
                return "<rss></rss>"

        with tempfile.TemporaryDirectory() as tmp:
            fetcher = FixtureFetcher(
                Path(tmp),
                refresh=False,
                offline=False,
                delay_seconds=0,
                user_agent=ingest.DEFAULT_USER_AGENT,
                fetch_mode="auto",
            )
            body, _cache_path, fetched = fetcher.get(ingest.FORECAST_RSS_URL)

        self.assertEqual("<rss></rss>", body)
        self.assertTrue(fetched)

    def test_browser_fetcher_reports_missing_playwright(self) -> None:
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("playwright"):
                raise ImportError("missing playwright fixture")
            return real_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp, mock.patch("builtins.__import__", side_effect=fake_import):
            fetcher = ingest.HtmlFetcher(
                Path(tmp),
                refresh=False,
                offline=False,
                delay_seconds=0,
                user_agent=ingest.DEFAULT_USER_AGENT,
                fetch_mode="browser",
            )
            with self.assertRaisesRegex(ingest.FetchBlocked, "Playwright is required"):
                fetcher.get(ingest.FORECAST_RSS_URL)


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
                LEFT JOIN movies m ON m.movie_id = p.movie_id
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
            self.assertEqual(("Young Washington", "provisional", "Young Washington"), tuple(rows[2][:3]))
            self.assertEqual(("Supergirl", "matched", "Supergirl (2026)"), tuple(rows[3][:3]))
        finally:
            drop_isolated_postgres_schema(conn, schema)

    def test_unmatched_opening_forecast_gets_provisional_identity_and_repoints_to_canonical(self) -> None:
        conn, schema = make_isolated_postgres_schema()
        try:
            ingest.initialize_database(conn)
            article, predictions, _rejected = ingest.parse_article(
                WEEKEND_ARTICLE_HTML,
                article_url="https://www.boxofficepro.com/weekend-preview-sample/",
            )
            young = [prediction for prediction in predictions if prediction.source_movie_title == "Young Washington"]
            self.assertEqual(1, len(young))

            first_match = ingest.match_predictions(conn, young)
            second_match = ingest.match_predictions(conn, young)
            conn.commit()

            self.assertEqual("provisional", first_match[0].match_status)
            self.assertEqual("provisional", second_match[0].match_status)
            movie_count = conn.execute(
                "SELECT COUNT(*) FROM movies WHERE title = 'Young Washington'"
            ).fetchone()[0]
            source_row = conn.execute(
                """
                SELECT movie_id, source_movie_id, source_title, match_status, match_method
                FROM movie_source_ids
                WHERE source = 'boxofficepro'
                  AND source_movie_id = %s
                """,
                (young[0].source_movie_id,),
            ).fetchone()
            self.assertEqual(1, movie_count)
            self.assertEqual(young[0].source_movie_id, source_row[1])
            self.assertEqual("Young Washington", source_row[2])
            self.assertEqual("provisional", source_row[3])

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
                ingest.insert_predictions(
                    conn,
                    article_id,
                    first_match,
                    fetched_at="2026-07-01T00:00:00+00:00",
                    raw_cache_path=cache_path,
                )

            canonical_movie_id = conn.execute(
                """
                INSERT INTO movies (movie_url, title, release_year, release_date, updated_at)
                VALUES (
                    'https://www.the-numbers.com/movie/Young-Washington-(2026)',
                    'Young Washington (2026)',
                    2026,
                    '2026-07-03',
                    CURRENT_TIMESTAMP
                )
                RETURNING movie_id
                """
            ).fetchone()[0]

            repointed = ingest.match_predictions(conn, young)
            conn.commit()
            self.assertEqual("matched", repointed[0].match_status)
            self.assertEqual(int(canonical_movie_id), repointed[0].movie_id)

            source_row = conn.execute(
                """
                SELECT movie_id, match_status, match_method
                FROM movie_source_ids
                WHERE source = 'boxofficepro'
                  AND source_movie_id = %s
                """,
                (young[0].source_movie_id,),
            ).fetchone()
            prediction_row = conn.execute(
                """
                SELECT movie_id, match_status, match_method
                FROM boxofficepro_weekend_predictions
                WHERE source_movie_id = %s
                """,
                (young[0].source_movie_id,),
            ).fetchone()
            self.assertEqual((int(canonical_movie_id), "matched", "normalized_exact_release_date"), tuple(source_row))
            self.assertEqual((int(canonical_movie_id), "matched", "normalized_exact_release_date"), tuple(prediction_row))
        finally:
            drop_isolated_postgres_schema(conn, schema)


if __name__ == "__main__":
    unittest.main()
