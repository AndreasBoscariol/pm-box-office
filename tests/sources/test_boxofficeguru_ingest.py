#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pm_box_office.sources.boxofficeguru import ingest


ARCHIVE_HTML = """
<html>
<body>
<center><p><b><font size=+1>2026</font></b></p></center>
<table>
<tr>
<td><a href="010526.htm">Jan. 2 - 4</a></td>
<td><a href="020226.htm">Jan. 30 - Feb. 1</a></td>
<td><a href="062926.htm">Jun. 26 - 28</a></td>
</tr>
</table>
<center><p><b><font size=+1>1997</font></b></p></center>
<table>
<tr>
<td><a href="http://www.boxofficeguru.com/122297.htm">Dec. 19 - 21</a></td>
</tr>
</table>
</body>
</html>
"""


RECAP_HTML = """
<html>
<head><title>Weekend Box Office</title></head>
<body>
<center><p><b><font>Weekend Box Office (December 19 - 21, 1997)</font></b></p></center>
<p>For the weekend, my projections were too conservative as <b>Titanic</b>
and <b>Tomorrow Never Dies</b> went well beyond my forecasts of $14M and
$16M respectively. Interest in both films was greater than I had expected.</p>
<p><b>Mouse Hunt</b> opened close to my $7M projection while <b>Scream 2</b>
dropped harder than my 35-40% prediction.</p>
<p><i>Last Updated : December 22, 1997 at 7:55PM EST</i></p>
</body>
</html>
"""


FORECAST_TABLE_HTML = """
<html>
<head><title>Weekend Box Office</title></head>
<body>
<center><p><b><font>Weekend Box Office (July 3 - 5, 2026)</font></b></p></center>
<p><i>Last Updated : July 2, 2026 at 12:00PM ET</i></p>
<table>
<tr>
<td>#</td>
<td>Title</td>
<td>Jul 3 - 5</td>
<td>Jun 26 - 28</td>
<td>% Chg.</td>
<td>Theaters</td>
<td>Weeks</td>
<td>AVG</td>
<td>Cumulative</td>
<td>Distributor</td>
</tr>
<tr>
<td>1</td>
<td>Minions &amp; Monsters</td>
<td>$ 52,000,000</td>
<td></td>
<td></td>
<td>4,243</td>
<td>1</td>
<td>$ 12,255</td>
<td>$ 78,000,000</td>
<td>Universal</td>
</tr>
</table>
</body>
</html>
"""


class BoxOfficeGuruParserTests(unittest.TestCase):
    def test_archive_parser_discovers_weekend_pages_with_section_years(self) -> None:
        pages = ingest.parse_archive(ARCHIVE_HTML)

        self.assertEqual(4, len(pages))
        self.assertEqual("http://www.boxofficeguru.com/010526.htm", pages[0].article_url)
        self.assertEqual("2026-01-02", pages[0].target_start_date)
        self.assertEqual("2026-01-04", pages[0].target_end_date)
        self.assertEqual("2026-01-30", pages[1].target_start_date)
        self.assertEqual("2026-02-01", pages[1].target_end_date)
        self.assertEqual("1997-12-19", pages[3].target_start_date)
        self.assertEqual("1997-12-21", pages[3].target_end_date)

    def test_recap_parser_extracts_forecast_mentions_with_inferred_prediction_date(self) -> None:
        fallback = ingest.ArchiveWeekendPage(
            article_url="http://www.boxofficeguru.com/122297.htm",
            title="Weekend Box Office (Dec. 19 - 21, 1997)",
            target_start_date="1997-12-19",
            target_end_date="1997-12-21",
            source_url=ingest.ARCHIVE_URL,
        )

        article, predictions = ingest.parse_article(RECAP_HTML, article_url=fallback.article_url, fallback=fallback)

        self.assertEqual("recap_with_forecast_mentions", article.status)
        self.assertEqual("1997-12-18", article.prediction_made_date)
        self.assertEqual("inferred_thursday_preview_from_recap", article.prediction_made_inference)
        self.assertEqual(3, len(predictions))
        self.assertEqual("Titanic", predictions[0].source_movie_title)
        self.assertEqual(14_000_000, predictions[0].weekend_gross_prediction_usd)
        self.assertEqual("Tomorrow Never Dies", predictions[1].source_movie_title)
        self.assertEqual(16_000_000, predictions[1].weekend_gross_prediction_usd)
        self.assertEqual("Mouse Hunt", predictions[2].source_movie_title)
        self.assertEqual(7_000_000, predictions[2].weekend_gross_prediction_usd)

    def test_pre_weekend_table_snapshot_extracts_prediction_rows(self) -> None:
        fallback = ingest.ArchiveWeekendPage(
            article_url="http://www.boxofficeguru.com/070626.htm",
            title="Weekend Box Office (Jul. 3 - 5, 2026)",
            target_start_date="2026-07-03",
            target_end_date="2026-07-05",
            source_url=ingest.ARCHIVE_URL,
        )

        article, predictions = ingest.parse_article(
            FORECAST_TABLE_HTML,
            article_url=fallback.article_url,
            fallback=fallback,
            snapshot_timestamp="20260702170000",
            snapshot_url="https://web.archive.org/web/20260702170000id_/http://www.boxofficeguru.com/070626.htm",
        )

        self.assertEqual("forecast_snapshot", article.status)
        self.assertEqual("2026-07-02", article.prediction_made_date)
        self.assertEqual(1, len(predictions))
        self.assertEqual("Minions & Monsters", predictions[0].source_movie_title)
        self.assertEqual(1, predictions[0].source_rank)
        self.assertEqual(52_000_000, predictions[0].weekend_gross_prediction_usd)
        self.assertEqual(78_000_000, predictions[0].total_gross_prediction_usd)
        self.assertEqual("Universal", predictions[0].distributor)

    def test_full_refresh_args_cover_entire_archive(self) -> None:
        args = ingest.build_arg_parser().parse_args(["--full-refresh"])

        ingest.configure_full_refresh_args(args)

        self.assertTrue(args.refresh)
        self.assertEqual(ingest.FULL_REFRESH_START_DATE, args.start_date)
        self.assertEqual(ingest.FULL_REFRESH_END_DATE, args.end_date)

    def test_wayback_discovery_failure_uses_live_fallback_when_requested(self) -> None:
        class BlockedFetcher:
            def get(self, _url: str):
                raise ingest.FetchBlocked("CDX timeout fixture")

        page = ingest.ArchiveWeekendPage(
            article_url="http://www.boxofficeguru.com/070626.htm",
            title="Weekend Box Office (Jul. 3 - 5, 2026)",
            target_start_date="2026-07-03",
            target_end_date="2026-07-05",
            source_url=ingest.ARCHIVE_URL,
        )
        args = ingest.build_arg_parser().parse_args(["--use-wayback", "--include-live-fallback"])

        snapshot_urls = ingest.discover_snapshot_urls(BlockedFetcher(), page, args)

        self.assertEqual([(page.article_url, None)], snapshot_urls)

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

            body, cache_path, fetched = fetcher.get("http://www.boxofficeguru.com/sample.htm")

            self.assertTrue(fetched)
            self.assertEqual("<html></html>", body)
            self.assertTrue(cache_path.exists())


if __name__ == "__main__":
    unittest.main()
