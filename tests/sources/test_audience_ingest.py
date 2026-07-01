#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import gzip
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path

from pm_box_office.db.connection import table_names
from pm_box_office.sources.audience import ingest
from tests.postgres_test_utils import drop_isolated_postgres_schema, make_isolated_postgres_schema


RELEASE_SCHEDULE_HTML = """
<html>
  <body>
    <table>
      <tr>
        <th>Release Date</th><th>Movie</th><th>Release Pattern</th>
        <th>Distributor</th><th>Domestic Box Office</th>
      </tr>
      <tr>
        <td>Jul 3, 2026</td>
        <td><a href="/movie/Wide-Movie-(2026)">Wide Movie</a></td>
        <td>Wide</td><td>Example Studios</td><td>$1,234</td>
      </tr>
      <tr>
        <td>Jul 10, 2026</td>
        <td><a href="/movie/Limited-Movie-(2026)">Limited Movie</a></td>
        <td>Limited</td><td>Indie Co.</td><td></td>
      </tr>
      <tr>
        <td>Jul 10, 2026</td>
        <td><a href="/movie/Untitled-Event-Film-(2026)">Untitled Event Film</a></td>
        <td>Wide</td><td>Placeholder Co.</td><td></td>
      </tr>
      <tr>
        <td>Jul 17, 2026</td>
        <td><a href="/movie/Imax-Movie-(2026)">IMAX Movie</a></td>
        <td>IMAX</td><td>Big Co.</td><td>$2,000</td>
      </tr>
      <tr>
        <td>Jul 5, 2026</td>
        <td><a href="/movie/Rerelease-Movie-(2026)">Rerelease Movie</a></td>
        <td>Re-release</td><td>Archive Co.</td><td>$3,000</td>
      </tr>
      <tr>
        <td>TBD</td>
        <td><a href="/movie/Tbd-Movie-(2026)">TBD Movie</a></td>
        <td>Wide</td><td>Later Co.</td><td></td>
      </tr>
    </table>
  </body>
</html>
"""


LETTERBOXD_FILM_HTML = """
<html>
  <head>
    <title>Sample Movie (2026) - Letterboxd</title>
    <meta property="og:title" content="Sample Movie (2026)" />
    <meta name="twitter:data2" content="3.8 out of 5" />
    <script type="application/ld+json">
      {"@type": "Movie", "name": "Sample Movie", "aggregateRating": {"ratingValue": "3.8", "ratingCount": "25000", "reviewCount": "1234"}}
    </script>
  </head>
  <body>
    <a href="https://www.imdb.com/title/tt1234567/">IMDb</a>
    <a href="https://www.themoviedb.org/movie/98765">TMDb</a>
    <p>100K watched</p>
    <p>25K ratings</p>
    <p>1,234 reviews</p>
    <p>5,678 logs</p>
    <p>99 fans</p>
  </body>
</html>
"""


MICHAEL_LETTERBOXD_HTML = """
<html>
  <head>
    <title>Michael (2026) - Letterboxd</title>
    <meta property="og:title" content="Michael (2026)" />
    <script type="application/ld+json">
      {"@type": "Movie", "name": "Michael", "aggregateRating": {"ratingValue": "3.6", "ratingCount": "500"}}
    </script>
  </head>
  <body>
    <p>10K watched</p>
    <p>500 ratings</p>
    <p>50 reviews</p>
    <p>75 logs</p>
  </body>
</html>
"""


def gzipped_tsv(text: str) -> bytes:
    return gzip.compress(text.encode("utf-8"))


class AudienceParserTests(unittest.TestCase):
    def test_the_numbers_release_schedule_parser_handles_release_types_and_tbd(self) -> None:
        rows = ingest.parse_the_numbers_release_schedule(RELEASE_SCHEDULE_HTML, default_year=2026)

        self.assertEqual(6, len(rows))
        self.assertEqual("https://www.the-numbers.com/movie/Wide-Movie-(2026)", rows[0].movie_url)
        self.assertEqual("2026-07-03", rows[0].release_date)
        self.assertEqual("Wide", rows[0].release_pattern)
        self.assertEqual(1234, rows[0].domestic_box_office_to_date_usd)
        self.assertEqual("Limited", rows[1].release_pattern)
        by_title = {row.title: row for row in rows}
        self.assertEqual("IMAX", by_title["IMAX Movie"].release_pattern)
        self.assertEqual("Re-release", by_title["Rerelease Movie"].release_pattern)
        self.assertIsNone(by_title["TBD Movie"].release_date)

    def test_imdb_tsv_parsing_and_matching(self) -> None:
        titles = ingest.parse_imdb_titles(
            gzipped_tsv(
                "tconst\ttitleType\tprimaryTitle\toriginalTitle\tisAdult\tstartYear\tendYear\truntimeMinutes\tgenres\n"
                "tt1111111\tmovie\tSample Movie\tSample Movie\t0\t2026\t\\N\t100\tDrama\n"
                "tt2222222\tmovie\tAlias Original\tAlias Original\t0\t2026\t\\N\t90\tComedy\n"
                "tt3333333\tmovie\tDuplicate Movie\tDuplicate Movie\t0\t2026\t\\N\t90\tComedy\n"
                "tt4444444\tmovie\tDuplicate Movie\tDuplicate Movie\t0\t2026\t\\N\t90\tComedy\n"
                "tt5555555\tmovie\tThe Breadwinner\tThe Breadwinner\t0\t2026\t\\N\t90\tDrama\n"
            )
        )
        akas = ingest.parse_imdb_akas(
            gzipped_tsv(
                "titleId\tordering\ttitle\tregion\tlanguage\ttypes\tattributes\tisOriginalTitle\n"
                "tt2222222\t1\tAlias Movie\tUS\t\\N\timdbDisplay\t\\N\t0\n"
            )
        )

        sample = ingest.CandidateMovie(1, None, "Sample Movie", 2026, None, "sample movie")
        alias = ingest.CandidateMovie(2, None, "Alias Movie", 2026, None, "alias movie")
        duplicate = ingest.CandidateMovie(3, None, "Duplicate Movie", 2026, None, "duplicate movie")
        missing = ingest.CandidateMovie(4, None, "Missing Movie", 2026, None, "missing movie")
        url_reordered = ingest.CandidateMovie(
            5,
            "https://www.the-numbers.com/movie/Breadwinner-The-(2026)",
            "Breadwinner, The (2026)",
            2026,
            None,
            "breadwinner the",
        )

        self.assertEqual("tt1111111", ingest.match_imdb_title(sample, titles, akas).tconst)
        self.assertEqual("tt2222222", ingest.match_imdb_title(alias, titles, akas).tconst)
        self.assertEqual("ambiguous", ingest.match_imdb_title(duplicate, titles, akas).match_status)
        self.assertEqual("not_found", ingest.match_imdb_title(missing, titles, akas).match_status)
        self.assertEqual("tt5555555", ingest.match_imdb_title(url_reordered, titles, akas).tconst)

    def test_imdb_ratings_parser(self) -> None:
        ratings = ingest.parse_imdb_ratings(
            gzipped_tsv("tconst\taverageRating\tnumVotes\ntt1111111\t7.4\t12345\n")
        )

        self.assertEqual("tt1111111", ratings[0].tconst)
        self.assertEqual(7.4, ratings[0].average_rating)
        self.assertEqual(12345, ratings[0].num_votes)

    def test_imdb_existing_match_snapshots_from_ratings_without_basics_scan(self) -> None:
        class FakeCursor:
            def __init__(self, row=None) -> None:
                self.row = row

            def fetchone(self):
                return self.row

        class FakeConn:
            def __init__(self) -> None:
                self.snapshots = 0
                self.commits = 0

            def execute(self, sql, params=None):
                if "SELECT tconst" in sql and "FROM movie_imdb_titles" in sql:
                    return FakeCursor(("tt11378946",))
                if "INSERT INTO imdb_title_snapshots" in sql:
                    self.snapshots += 1
                return FakeCursor()

            def executemany(self, *_args, **_kwargs):
                raise AssertionError("Existing IMDb matches should not scan/upsert title basics")

            def commit(self) -> None:
                self.commits += 1

            def rollback(self) -> None:
                pass

        class FakeFetcher:
            def __init__(self) -> None:
                self.urls: list[str] = []

            def get_bytes(self, url: str, *, suffix: str):
                self.urls.append(url)
                if "title.ratings" not in url:
                    raise AssertionError(f"Unexpected IMDb dataset fetch: {url}")
                return (
                    gzipped_tsv("tconst\taverageRating\tnumVotes\ntt11378946\t7.5\t139295\n"),
                    Path("ratings.tsv.gz"),
                    False,
                )

        args = types.SimpleNamespace(snapshot_date=dt.date(2026, 6, 30), quiet=True)
        conn = FakeConn()
        fetcher = FakeFetcher()
        candidate = ingest.CandidateMovie(1, None, "Michael", 2026, None, "michael")

        inserted = ingest.ingest_imdb(args, conn, [candidate], fetcher)

        self.assertEqual(1, inserted)
        self.assertEqual(1, conn.snapshots)
        self.assertEqual(["https://datasets.imdbws.com/title.ratings.tsv.gz"], fetcher.urls)

    def test_letterboxd_film_page_parser_extracts_aggregate_counts(self) -> None:
        page = ingest.parse_letterboxd_film_page(
            LETTERBOXD_FILM_HTML,
            film_url="https://letterboxd.com/film/sample-movie/",
        )

        self.assertEqual("sample-movie", page.letterboxd_slug)
        self.assertEqual("Sample Movie", page.source_title)
        self.assertEqual(2026, page.source_year)
        self.assertEqual("tt1234567", page.imdb_tconst)
        self.assertEqual("98765", page.tmdb_id)
        self.assertEqual(100_000, page.watched_count)
        self.assertEqual(25_000, page.rating_count)
        self.assertEqual(1234, page.review_count)
        self.assertEqual(5678, page.log_count)
        self.assertEqual(99, page.fan_count)
        self.assertEqual("parsed", page.parse_status)

    def test_cached_fetcher_offline_cache_miss_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fetcher = ingest.CachedFetcher(
                Path(tmp),
                refresh=False,
                offline=True,
                delay_seconds=20,
                user_agent="pm-box-office-audience-ingest-bot/0.1",
            )
            with self.assertRaises(FileNotFoundError):
                fetcher.get_text("https://example.com/missing")

    def test_random_user_agent_uses_generator_and_keeps_project_identifier(self) -> None:
        original = sys.modules.get("fake_useragent")
        fake_module = types.ModuleType("fake_useragent")

        class UserAgent:
            @property
            def random(self) -> str:
                return "Mozilla/5.0 Fixture Browser"

        fake_module.UserAgent = UserAgent
        sys.modules["fake_useragent"] = fake_module
        try:
            user_agent = ingest.resolve_user_agent("random")
        finally:
            if original is None:
                sys.modules.pop("fake_useragent", None)
            else:
                sys.modules["fake_useragent"] = original

        self.assertIn("Mozilla/5.0 Fixture Browser", user_agent)
        self.assertIn("pm-box-office-audience-ingest/0.1", user_agent)

    def test_random_user_agent_falls_back_without_library(self) -> None:
        original = sys.modules.get("fake_useragent")
        sys.modules.pop("fake_useragent", None)
        try:
            user_agent = ingest.resolve_user_agent("random")
        finally:
            if original is not None:
                sys.modules["fake_useragent"] = original

        self.assertIn("pm-box-office-audience-ingest/0.1", user_agent)

    def test_imdb_and_letterboxd_source_stages_run_concurrently(self) -> None:
        original = ingest.run_audience_source_stage
        started: set[str] = set()
        lock = threading.Lock()
        both_started = threading.Event()

        def fake_stage(args, *, source, candidates):
            with lock:
                started.add(source)
                if started == {"imdb", "letterboxd"}:
                    both_started.set()
            self.assertTrue(both_started.wait(timeout=1.0))
            return 11 if source == "imdb" else 22

        ingest.run_audience_source_stage = fake_stage
        try:
            args = types.SimpleNamespace(skip_imdb=False, skip_letterboxd=False, quiet=True)
            imdb_rows, letterboxd_rows = ingest.run_audience_source_stages(args, candidates=[])
        finally:
            ingest.run_audience_source_stage = original

        self.assertEqual(11, imdb_rows)
        self.assertEqual(22, letterboxd_rows)

    def test_title_variants_include_movie_url_and_reordered_article(self) -> None:
        variants = ingest.title_variants(
            "Breadwinner, The (2026)",
            "https://www.the-numbers.com/movie/Breadwinner-The-(2026)",
        )

        normalized = {ingest.normalize_title(value) for value in variants}
        self.assertIn("the breadwinner", normalized)
        self.assertIn("breadwinner the", normalized)

    def test_letterboxd_candidate_slugs_strip_punctuation_and_add_year(self) -> None:
        bleach = ingest.CandidateMovie(
            1,
            None,
            "Bleach: Thousand-Year Blood War - The Calamity",
            2026,
            None,
            "bleach thousand year blood war the calamity",
        )
        michael = ingest.CandidateMovie(
            2,
            None,
            "Michael (Wide)",
            2026,
            None,
            "michael wide",
        )

        self.assertEqual(
            [
                "bleach-thousand-year-blood-war-the-calamity-2026",
                "bleach-thousand-year-blood-war-the-calamity",
            ],
            ingest.letterboxd_candidate_slugs(bleach),
        )
        self.assertEqual(["michael-2026", "michael"], ingest.letterboxd_candidate_slugs(michael))

    def test_wikidata_matching_prefers_exact_year_and_ids(self) -> None:
        candidates = [
            ingest.CandidateMovie(
                1,
                "https://www.the-numbers.com/movie/Scary-Movie-(2026)",
                "Scary Movie",
                2026,
                "2026-06-12",
                "scary movie",
            ),
            ingest.CandidateMovie(
                2,
                "https://www.the-numbers.com/movie/Obsession-(2026)",
                "Obsession",
                2026,
                "2026-07-03",
                "obsession",
            ),
        ]
        records = ingest.parse_wikidata_bindings(
            {
                "results": {
                    "bindings": [
                        {
                            "item": {"value": "http://www.wikidata.org/entity/Q1"},
                            "itemLabel": {"value": "Scary Movie"},
                            "imdb": {"value": "tt9999999"},
                            "tmdb": {"value": "123"},
                            "letterboxd": {"value": "scary-movie-2026"},
                            "pubdate": {"value": "2026-06-12T00:00:00Z"},
                        },
                        {
                            "item": {"value": "http://www.wikidata.org/entity/Q2"},
                            "itemLabel": {"value": "Scary Movie"},
                            "imdb": {"value": "tt0175142"},
                            "pubdate": {"value": "2000-07-07T00:00:00Z"},
                        },
                        {
                            "item": {"value": "http://www.wikidata.org/entity/Q3"},
                            "itemLabel": {"value": "Obsession"},
                            "imdb": {"value": "tt8888888"},
                            "letterboxd": {"value": "obsession-2026"},
                            "pubdate": {"value": "2026-07-03T00:00:00Z"},
                        },
                    ]
                }
            }
        )

        matches = {match.movie_id: match for match in ingest.match_wikidata_movies(candidates, records)}

        self.assertEqual("matched", matches[1].status)
        self.assertEqual("tt9999999", matches[1].imdb_tconst)
        self.assertEqual("scary-movie-2026", matches[1].letterboxd_slug)
        self.assertEqual("matched", matches[2].status)
        self.assertEqual("obsession-2026", matches[2].letterboxd_slug)

    def test_letterboxd_movie_failure_does_not_stop_next_movie(self) -> None:
        class FakeCursor:
            def fetchone(self):
                return None

        class FakeConn:
            def __init__(self) -> None:
                self.commits = 0
                self.rollbacks = 0

            def execute(self, *_args, **_kwargs):
                return FakeCursor()

            def commit(self) -> None:
                self.commits += 1

            def rollback(self) -> None:
                self.rollbacks += 1

        class FakeFetcher:
            def __init__(self) -> None:
                self.urls: list[str] = []

            def get_text(self, url: str):
                self.urls.append(url)
                if "blocked-movie" in url or "Blocked%20Movie" in url:
                    raise RuntimeError("HTTP Error 403: Forbidden")
                return "", Path("cache.html"), False

        conn = FakeConn()
        fetcher = FakeFetcher()
        args = types.SimpleNamespace(snapshot_date=dt.date(2026, 6, 30), quiet=True)
        candidates = [
            ingest.CandidateMovie(1, None, "Blocked Movie", 2026, None, "blocked movie"),
            ingest.CandidateMovie(2, None, "Next Movie", 2026, None, "next movie"),
        ]

        inserted = ingest.ingest_letterboxd(args, conn, candidates, fetcher)

        self.assertEqual(0, inserted)
        self.assertTrue(any("Next%20Movie" in url for url in fetcher.urls))
        self.assertGreaterEqual(conn.rollbacks, 1)
        self.assertGreaterEqual(conn.commits, 2)

    def test_letterboxd_predictable_slug_is_tried_before_search(self) -> None:
        class FakeCursor:
            def fetchone(self):
                return None

        class FakeConn:
            def __init__(self) -> None:
                self.commits = 0

            def execute(self, *_args, **_kwargs):
                return FakeCursor()

            def commit(self) -> None:
                self.commits += 1

            def rollback(self) -> None:
                pass

        class FakeFetcher:
            def __init__(self) -> None:
                self.urls: list[str] = []

            def get_text(self, url: str):
                self.urls.append(url)
                if url.endswith("/film/michael-2026/"):
                    return MICHAEL_LETTERBOXD_HTML, Path("michael.html"), False
                if "/search/" in url:
                    raise AssertionError("Search should not run after a predictable slug match")
                return "", Path("missing.html"), False

        conn = FakeConn()
        fetcher = FakeFetcher()
        args = types.SimpleNamespace(snapshot_date=dt.date(2026, 6, 30), quiet=True)
        candidate = ingest.CandidateMovie(1, None, "Michael", 2026, None, "michael")

        inserted = ingest.ingest_letterboxd(args, conn, [candidate], fetcher)

        self.assertEqual(1, inserted)
        self.assertEqual(
            [
                "https://letterboxd.com/film/michael-2026/",
            ],
            fetcher.urls,
        )


class AudiencePostgresTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn, self.schema = make_isolated_postgres_schema()
        ingest.initialize_database(self.conn)
        self.conn.commit()

    def tearDown(self) -> None:
        drop_isolated_postgres_schema(self.conn, self.schema)

    def test_schema_initialization_is_idempotent(self) -> None:
        ingest.initialize_database(self.conn)
        names = set(table_names(self.conn))

        self.assertIn("the_numbers_release_schedule", names)
        self.assertIn("imdb_titles", names)
        self.assertIn("letterboxd_film_snapshots", names)
        self.assertTrue(
            self.conn.execute("SELECT to_regclass('analytics.movie_audience_daily_features_v1')").fetchone()[0]
        )
        snapshot_columns = {
            row[0]
            for row in self.conn.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = 'letterboxd_film_snapshots'
                """
            ).fetchall()
        }
        self.assertFalse({"watched_count", "rating_count", "review_count", "log_count"} & snapshot_columns)
        imdb_snapshot_columns = {
            row[0]
            for row in self.conn.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = 'imdb_title_snapshots'
                """
            ).fetchall()
        }
        self.assertFalse({"user_review_count", "source_kind"} & imdb_snapshot_columns)

    def test_candidate_selection_includes_upcoming_and_current_movies_with_limit(self) -> None:
        rows = ingest.parse_the_numbers_release_schedule(RELEASE_SCHEDULE_HTML, default_year=2026)
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "fixture.html"
            cache_path.write_text("fixture", encoding="utf-8")
            ingest.upsert_release_schedule_rows(
                self.conn,
                rows,
                fetched_at="2026-06-30T00:00:00+00:00",
                raw_cache_path=cache_path,
            )
            ingest.upsert_movies_from_release_schedule(self.conn, rows)
        self.conn.execute(
            """
            INSERT INTO movies (movie_id, movie_url, title, release_year, release_date)
            VALUES (100, 'https://www.the-numbers.com/movie/Current-Movie-(2026)', 'Current Movie (2026)', 2026, '2026-06-20')
            """
        )
        self.conn.execute(
            """
            INSERT INTO release_runs (release_run_id, movie_id, market, release_type, source, source_release_key)
            VALUES (1000, 100, 'US_CA', 'movie_page_full_run', 'the_numbers', 'current')
            """
        )
        self.conn.execute(
            """
            INSERT INTO daily_box_office (
                release_run_id, box_office_date, day_number, gross_usd, theaters,
                cumulative_gross_usd, is_preview, source, source_url, fetched_at, raw_cache_path
            ) VALUES (1000, '2026-06-29', 10, 500, 2000, 10000, 0, 'the_numbers', 'movie', 'now', 'cache')
            """
        )
        self.conn.execute(
            """
            CREATE TABLE daily_chart_pages (
                chart_date TEXT NOT NULL,
                movie_url TEXT NOT NULL,
                title TEXT NOT NULL,
                gross_usd INTEGER,
                theaters INTEGER,
                days_in_release INTEGER
            )
            """
        )
        self.conn.executemany(
            """
            INSERT INTO daily_chart_pages (chart_date, movie_url, title, gross_usd, theaters, days_in_release)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    "2026-06-29",
                    "https://www.the-numbers.com/movie/Chart-Only-Movie-(2026)",
                    "Chart Only Movie",
                    None,
                    None,
                    8,
                ),
                (
                    "2026-06-29",
                    "https://www.the-numbers.com/movie/Old-Chart-Movie-(2025)",
                    "Old Chart Movie",
                    300,
                    1000,
                    400,
                ),
                (
                    "2026-06-29",
                    "https://www.the-numbers.com/movie/Citizen-Kane-(1941)",
                    "Citizen Kane (Special Engagement, re-release)",
                    None,
                    None,
                    2,
                ),
            ],
        )
        upserted = ingest.upsert_movies_from_current_chart_pages(
            self.conn,
            snapshot_date=dt.date(2026, 6, 30),
            active_days=7,
        )
        self.conn.commit()
        self.assertEqual(1, upserted)

        candidates = ingest.select_candidate_movies(
            self.conn,
            snapshot_date=dt.date(2026, 6, 30),
            lookahead_days=14,
            active_days=7,
            movie_limit=None,
        )
        titles = {candidate.title for candidate in candidates}
        self.assertIn("Wide Movie", titles)
        self.assertIn("Limited Movie", titles)
        self.assertNotIn("Untitled Event Film", titles)
        self.assertNotIn("IMAX Movie", titles)
        self.assertNotIn("Rerelease Movie", titles)
        self.assertNotIn("Current Movie (2026)", titles)
        self.assertIn("Chart Only Movie", titles)
        self.assertNotIn("Old Chart Movie", titles)
        self.assertNotIn("Rerelease Movie", titles)
        self.assertNotIn("Citizen Kane (Special Engagement, re-release)", titles)
        chart_movie = next(candidate for candidate in candidates if candidate.title == "Chart Only Movie")
        self.assertEqual(2026, chart_movie.release_year)

        limited = ingest.select_candidate_movies(
            self.conn,
            snapshot_date=dt.date(2026, 6, 30),
            lookahead_days=14,
            active_days=7,
            movie_limit=1,
        )
        self.assertEqual(1, len(limited))

    def test_snapshots_are_idempotent_and_feed_analysis_views(self) -> None:
        self.conn.execute(
            """
            INSERT INTO movies (movie_id, movie_url, title, release_year, release_date)
            VALUES
                (1, 'https://www.the-numbers.com/movie/Sample-Movie-(2026)', 'Sample Movie (2026)', 2026, '2026-07-03'),
                (2, 'https://www.the-numbers.com/movie/Ambiguous-Movie-(2026)', 'Ambiguous Movie (2026)', 2026, '2026-07-03')
            """
        )
        self.conn.execute(
            """
            INSERT INTO release_runs (release_run_id, movie_id, market, release_type, source, source_release_key)
            VALUES (10, 1, 'US_CA', 'movie_page_full_run', 'the_numbers', 'sample')
            """
        )
        self.conn.executemany(
            """
            INSERT INTO daily_box_office (
                release_run_id, box_office_date, day_number, gross_usd, theaters,
                cumulative_gross_usd, is_preview, source, source_url, fetched_at, raw_cache_path
            ) VALUES (10, %s, %s, %s, 3000, %s, 0, 'the_numbers', 'movie', 'now', 'cache')
            """,
            [
                ("2026-07-03", 1, 1000, 1000),
                ("2026-07-04", 2, 2000, 3000),
            ],
        )
        imdb_title = ingest.ImdbTitle("tt1234567", "Sample Movie", "Sample Movie", 2026, "movie", 0, "Drama")
        ingest.upsert_imdb_titles(self.conn, [imdb_title], last_seen_at="2026-06-30T00:00:00+00:00")
        ingest.upsert_imdb_match(
            self.conn,
            ingest.ImdbMatch(1, "tt1234567", "matched", "fixture", 1.0),
        )
        ingest.upsert_imdb_match(
            self.conn,
            ingest.ImdbMatch(2, None, "ambiguous", "fixture", None, "tt1, tt2"),
        )

        page = ingest.parse_letterboxd_film_page(
            LETTERBOXD_FILM_HTML,
            film_url="https://letterboxd.com/film/sample-movie/",
        )
        ingest.upsert_letterboxd_film(self.conn, page, last_seen_at="2026-06-30T00:00:00+00:00")
        ingest.upsert_letterboxd_match(
            self.conn,
            ingest.LetterboxdMatch(1, "sample-movie", "matched", "fixture", 1.0),
        )

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "fixture"
            cache.write_text("fixture", encoding="utf-8")
            for _ in range(2):
                ingest.insert_imdb_snapshot(
                    self.conn,
                    rating=ingest.ImdbRating("tt1234567", 7.4, 12345),
                    snapshot_date=dt.date(2026, 7, 2),
                    fetched_at="2026-07-02T00:00:00+00:00",
                    raw_cache_path=cache,
                )
                ingest.insert_letterboxd_snapshot(
                    self.conn,
                    page=page,
                    snapshot_date=dt.date(2026, 7, 2),
                    fetched_at="2026-07-02T00:00:00+00:00",
                    raw_cache_path=cache,
                )
            self.conn.commit()

        self.assertEqual(1, self.conn.execute("SELECT COUNT(*) FROM imdb_title_snapshots").fetchone()[0])
        self.assertEqual(1, self.conn.execute("SELECT COUNT(*) FROM letterboxd_film_snapshots").fetchone()[0])
        feature_rows = self.conn.execute(
            "SELECT movie_id, imdb_num_votes, letterboxd_fan_count FROM analytics.movie_audience_daily_features_v1"
        ).fetchall()
        self.assertEqual([(1, 12345, 99)], feature_rows)
        panel_row = self.conn.execute(
            """
            SELECT movie_id, box_office_date, audience_snapshot_date, imdb_num_votes, letterboxd_fan_count
            FROM analytics.box_office_audience_panel_v1
            WHERE box_office_date = '2026-07-03'
            """
        ).fetchone()
        self.assertEqual((1, dt.date(2026, 7, 3), dt.date(2026, 7, 2), 12345, 99), panel_row)

    def test_failed_state_can_be_reset(self) -> None:
        self.conn.execute(
            """
            INSERT INTO movies (movie_id, movie_url, title)
            VALUES (1, 'https://www.the-numbers.com/movie/Sample-Movie-(2026)', 'Sample Movie (2026)')
            """
        )
        ingest.upsert_state(
            self.conn,
            movie_id=1,
            source="letterboxd",
            stage="error",
            status="failed",
            last_error="boom",
        )
        ingest.reset_failed_states(self.conn)
        self.conn.commit()

        self.assertEqual(0, self.conn.execute("SELECT COUNT(*) FROM audience_ingest_state").fetchone()[0])


if __name__ == "__main__":
    unittest.main()
