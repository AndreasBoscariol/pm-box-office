#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pm_box_office.sources.boxofficetheory_substack import ingest
from tests.postgres_test_utils import drop_isolated_postgres_schema, make_isolated_postgres_schema


SUBSTACK_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss xmlns:dc="http://purl.org/dc/elements/1.1/"
     xmlns:content="http://purl.org/rss/1.0/modules/content/"
     version="2.0">
  <channel>
    <title>Shawn Robbins @ Box Office Theory</title>
    <item>
      <title><![CDATA[Box Office Weekend Forecast: Sample]]></title>
      <description><![CDATA[Sample description.]]></description>
      <link>https://boxofficetheory.substack.com/p/sample-post?utm_source=feed</link>
      <guid isPermaLink="false">https://boxofficetheory.substack.com/p/sample-post</guid>
      <dc:creator><![CDATA[Shawn Robbins]]></dc:creator>
      <pubDate>Wed, 01 Jul 2026 17:11:10 GMT</pubDate>
      <content:encoded><![CDATA[
        <p><em><strong>Minions &amp; Monsters<br></strong></em>
        BOT Domestic Opening Weekend Forecast Range: <strong>$68 &#8212; 87 million (5-Day)</strong><br />
        Studio Tracking: $95 million (5-day)</p>
        <p><em><strong>Young Washington<br></strong></em>(Angel Studios)<br />
        BOT Domestic Opening Weekend Forecast Range: <strong>$23 million+</strong></p>
      ]]></content:encoded>
    </item>
  </channel>
</rss>
"""


SUBSTACK_POST_JSON = """
{
  "id": 204461747,
  "slug": "sample-post",
  "canonical_url": "https://boxofficetheory.substack.com/p/sample-post",
  "title": "Box Office Weekend Forecast: Sample",
  "subtitle": "Sample subtitle.",
  "description": "Sample description.",
  "post_date": "2026-07-01T17:11:10.226Z",
  "body_html": "<p><em><strong>Minions &amp; Monsters </strong></em>(Universal &amp; Illumination)<br>BOT Domestic Opening Weekend Forecast Range: <strong>$68 — 87 million (5-Day)</strong></p>",
  "publishedBylines": [{"name": "Shawn Robbins"}]
}
"""


SUBSTACK_IMAGE_POST_JSON = """
{
  "id": 204726745,
  "slug": "image-chart-post",
  "canonical_url": "https://boxofficetheory.substack.com/p/image-chart-post",
  "title": "Box Office Tracking & Forecasts: Sample Chart",
  "subtitle": "Sample subtitle.",
  "description": "Sample description.",
  "post_date": "2026-07-02T22:08:53.000Z",
  "body_html": "<p><strong>5-Week Box Office Tracking &amp; Forecasts</strong></p><img width=\\"900\\" height=\\"400\\" data-attrs=\\"{&quot;src&quot;:&quot;https://substack-post-media.s3.amazonaws.com/public/images/hero_900x400.png&quot;,&quot;height&quot;:400,&quot;width&quot;:900,&quot;topImage&quot;:true,&quot;alt&quot;:null,&quot;title&quot;:null}\\" /><img width=\\"1456\\" height=\\"417\\" data-attrs=\\"{&quot;src&quot;:&quot;https://substack-post-media.s3.amazonaws.com/public/images/chart_1788x512.png&quot;,&quot;height&quot;:417,&quot;width&quot;:1456,&quot;topImage&quot;:false,&quot;alt&quot;:null,&quot;title&quot;:null}\\" />",
  "publishedBylines": [{"name": "Shawn Robbins"}]
}
"""


def ocr_token(text: str, left: int, top: int, *, width: int = 55, confidence: float = 90.0) -> ingest.OcrToken:
    return ingest.OcrToken(text=text, confidence=confidence, left=left, top=top, width=width, height=18)


class BoxOfficeTheorySubstackParserTests(unittest.TestCase):
    def test_parse_rss_posts_extracts_substack_item_content(self) -> None:
        posts = ingest.parse_rss_posts(SUBSTACK_RSS, source_url=ingest.FEED_URL)

        self.assertEqual(1, len(posts))
        post = posts[0]
        self.assertEqual("sample-post", post.source_post_id)
        self.assertEqual("https://boxofficetheory.substack.com/p/sample-post", post.post_url)
        self.assertEqual("Box Office Weekend Forecast: Sample", post.title)
        self.assertEqual("Shawn Robbins", post.author)
        self.assertEqual("2026-07-01T17:11:10+00:00", post.published_at)
        self.assertIn("BOT Domestic Opening Weekend Forecast Range", post.content_html)

    def test_substack_prose_ranges_reuse_theory_prediction_parser(self) -> None:
        post = ingest.parse_rss_posts(SUBSTACK_RSS, source_url=ingest.FEED_URL)[0]

        predictions = ingest.parse_predictions(post)

        self.assertEqual(2, len(predictions))
        self.assertEqual("Minions & Monsters", predictions[0].source_movie_title)
        self.assertEqual(68_000_000, predictions[0].opening_weekend_low_usd)
        self.assertEqual(87_000_000, predictions[0].opening_weekend_high_usd)
        self.assertEqual(5, predictions[0].opening_weekend_day_count)
        self.assertEqual(ingest.PARSER_VERSION, predictions[0].parser_version)
        self.assertEqual("Young Washington", predictions[1].source_movie_title)
        self.assertEqual("Angel Studios", predictions[1].distributor)
        self.assertEqual(23_000_000, predictions[1].opening_weekend_low_usd)

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
            source_url="https://boxofficetheory.substack.com/api/v1/posts/sample-post",
        )

        self.assertEqual("sample-post", post.source_post_id)
        self.assertEqual("https://boxofficetheory.substack.com/p/sample-post", post.post_url)
        self.assertEqual("2026-07-01T17:11:10.226000+00:00", post.published_at)
        self.assertIn("BOT Domestic Opening Weekend Forecast Range", post.content_html)

    def test_full_refresh_forces_archive_discovery(self) -> None:
        args = ingest.build_arg_parser().parse_args(["--full-refresh", "--discovery", "rss", "--max-pages", "1"])

        ingest.configure_full_refresh_args(args)

        self.assertEqual("archive", args.discovery)
        self.assertIsNone(args.max_pages)
        self.assertTrue(args.refresh)
        self.assertTrue(args.ocr_image_charts)
        self.assertEqual(ingest.FULL_REFRESH_START_DATE, args.start_date)

    def test_discovers_substack_forecast_chart_images(self) -> None:
        post = ingest.parse_post_record_json(
            SUBSTACK_IMAGE_POST_JSON,
            source_url="https://boxofficetheory.substack.com/api/v1/posts/image-chart-post",
        )

        images = ingest.discover_chart_images(post)

        self.assertEqual(1, len(images))
        self.assertEqual(
            "https://substack-post-media.s3.amazonaws.com/public/images/chart_1788x512.png",
            images[0].image_url,
        )
        self.assertEqual(1456, images[0].width)

    def test_parse_tracking_chart_ocr_rows_builds_prediction_values(self) -> None:
        post = ingest.parse_post_record_json(
            SUBSTACK_IMAGE_POST_JSON,
            source_url="https://boxofficetheory.substack.com/api/v1/posts/image-chart-post",
        )
        image = ingest.SubstackChartImage(
            post_url=post.post_url,
            post_title=post.title,
            published_date=post.published_date,
            image_url="https://substack-post-media.s3.amazonaws.com/public/images/chart_1788x512.png",
            width=1456,
            height=417,
            alt_text="",
            title="",
            context=post.title,
            ordinal=1,
        )
        ocr = ingest.OcrResult(
            text="",
            mean_confidence=82.0,
            tokens=[
                ocr_token("7/10/2026", 18, 178, width=92),
                ocr_token("Evil", 150, 178, width=35),
                ocr_token("Dead", 196, 178, width=42),
                ocr_token("Burn", 248, 178, width=40),
                ocr_token("Warner", 430, 145, width=68),
                ocr_token("Bros.", 508, 145, width=45),
                ocr_token("Pictures", 455, 178, width=72),
                ocr_token("$25,000,000", 620, 178, width=90),
                ocr_token("$32,000,000", 760, 178, width=90),
                ocr_token("$28,000,000", 900, 178, width=90),
                ocr_token("8%", 1030, 178, width=28),
                ocr_token("$50,000,000", 1130, 178, width=90),
                ocr_token("$78,000,000", 1270, 178, width=90),
                ocr_token("$69,000,000", 1410, 178, width=90),
                ocr_token("2.46", 1660, 178, width=42),
            ],
        )

        rows = ingest.parse_tracking_chart_ocr_rows(post, image=image, ocr=ocr, first_row_ordinal=1)

        self.assertEqual(1, len(rows))
        row = rows[0]
        self.assertEqual("Evil Dead Burn", row.title)
        self.assertEqual("Warner Bros. Pictures", row.distributor)
        self.assertEqual("2026-07-10", row.release_date)
        self.assertEqual(25_000_000, row.opening_weekend_low_usd)
        self.assertEqual(32_000_000, row.opening_weekend_high_usd)
        self.assertEqual(28_000_000, row.opening_weekend_pinpoint_usd)
        self.assertEqual(69_000_000, row.domestic_total_pinpoint_usd)
        self.assertEqual(8.0, row.percent_change)
        self.assertEqual(2.46, row.domestic_multiplier_pinpoint)

    def test_reconcile_total_pinpoint_uses_multiplier_when_ocr_value_is_outside_range(self) -> None:
        corrected = ingest.reconcile_total_pinpoint(
            opening_pinpoint=10_000_000,
            total_low=25_000_000,
            total_high=72_000_000,
            total_pinpoint=89_000_000,
            multiplier=3.9,
        )

        self.assertEqual(39_000_000, corrected)


class BoxOfficeTheorySubstackDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn, self.schema = make_isolated_postgres_schema()
        ingest.initialize_database(self.conn)
        self.conn.commit()

    def tearDown(self) -> None:
        drop_isolated_postgres_schema(self.conn, self.schema)

    def test_import_parse_result_persists_substack_predictions(self) -> None:
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
            issue_source="test_boxofficetheory_substack",
        )

        self.assertEqual(1, post_count)
        self.assertEqual(2, prediction_count)
        row = self.conn.execute(
            """
            SELECT source_movie_title, opening_weekend_low_usd, opening_weekend_high_usd, parser_version
            FROM boxofficetheory_substack_predictions
            WHERE source_movie_title = 'Minions & Monsters'
            """
        ).fetchone()
        self.assertEqual("Minions & Monsters", row[0])
        self.assertEqual(68_000_000, row[1])
        self.assertEqual(87_000_000, row[2])
        self.assertEqual(ingest.PARSER_VERSION, row[3])


if __name__ == "__main__":
    unittest.main()
