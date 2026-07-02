#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import io
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from pm_box_office.db.connection import table_names
from pm_box_office.sources.rotten_tomatoes import ingest
from tests.postgres_test_utils import drop_isolated_postgres_schema, make_isolated_postgres_schema


MEDIA_HTML = """
<!doctype html>
<html>
  <head>
    <script>
      window.mpscall = {"cag[score]":"88","cag[certified_fresh]":"1","cag[fresh_rotten]":"certified-fresh","cag[release]":"2023","cag[movieshow]":"Barbie","field[rtid]":"317d7155-533b-396f-8c1c-34a22e2e8ef9","title":"Barbie"};
    </script>
  </head>
  <body>
    <script>
      (function(root) {
        root.RottenTomatoes || (root.RottenTomatoes = {});
        root.RottenTomatoes.context || (root.RottenTomatoes.context = {});
        root.RottenTomatoes.context.review = {"mediaType":"movie","title":"Barbie","emsId":"317d7155-533b-396f-8c1c-34a22e2e8ef9","type":"all-critics","sort":undefined};
      }(this));
    </script>
  </body>
</html>
"""


REVIEWS_PAGE_1 = {
    "reviews": [
        {
            "reviewId": "102949294",
            "scoreSentiment": "POSITIVE",
            "originalScore": "5/5",
            "isTopReview": True,
            "publicationReviewUrl": "https://inews.co.uk/barbie",
            "createDate": "2024-09-18T00:45:39.000Z",
            "reviewQuote": "A bold future.",
            "critic": {
                "displayName": "Christina Newland",
                "rottenTomatoesUrl": "https://www.rottentomatoes.com/critic/christina-newland",
                "isTopCritic": True,
                "tomatometerApproved": True,
                "vanity": "christina-newland",
                "encryptedCriticId": "9Zkb10yG",
            },
            "publication": {
                "name": "iNews.co.uk",
                "editorialUrl": "https://inews.co.uk",
                "tomatometerApproved": True,
            },
        }
    ],
    "pageInfo": {"hasNextPage": True, "endCursor": "MQ=="},
}


REVIEWS_PAGE_2 = {
    "reviews": [
        {
            "reviewId": "103040076",
            "scoreSentiment": "NEGATIVE",
            "originalScore": "2.5/5",
            "publicationReviewUrl": "https://screenrealm.com/barbie",
            "createDate": "2025-07-29T12:20:13.000Z",
            "reviewQuote": "The passion is clear.",
            "critic": {
                "displayName": "Guillermo Troncoso",
                "rottenTomatoesUrl": "https://www.rottentomatoes.com/critic/guillermo-troncoso",
                "isTopCritic": False,
                "tomatometerApproved": True,
                "encryptedCriticId": "ZG0nv2kr",
            },
            "publication": {"name": "Screen Realm"},
        }
    ],
    "pageInfo": {"hasNextPage": False, "endCursor": "Mg=="},
}


class FakeFetcher:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def get_json(self, url: str):
        self.urls.append(url)
        if "after=MQ%3D%3D" in url:
            return REVIEWS_PAGE_2, Path("page2.json"), False
        return REVIEWS_PAGE_1, Path("page1.json"), False


class RottenTomatoesParserTests(unittest.TestCase):
    def test_media_page_parser_extracts_context_and_score_metadata(self) -> None:
        media = ingest.parse_media_page(MEDIA_HTML, source_url="https://www.rottentomatoes.com/m/barbie")

        self.assertIsNotNone(media)
        assert media is not None
        self.assertEqual("317d7155-533b-396f-8c1c-34a22e2e8ef9", media.ems_id)
        self.assertEqual("barbie", media.vanity_slug)
        self.assertEqual("Barbie", media.title)
        self.assertEqual(2023, media.release_year)
        self.assertEqual(88, media.tomatometer_score)
        self.assertTrue(media.certified_fresh)

    def test_reviews_parser_extracts_critic_fields(self) -> None:
        reviews, cursor = ingest.parse_reviews_payload(
            REVIEWS_PAGE_1,
            ems_id="317d7155-533b-396f-8c1c-34a22e2e8ef9",
            movie_id=1,
        )

        self.assertEqual("MQ==", cursor)
        self.assertEqual(1, len(reviews))
        review = reviews[0]
        self.assertEqual("317d7155-533b-396f-8c1c-34a22e2e8ef9:102949294", review.review_key)
        self.assertEqual("Christina Newland", review.critic_name)
        self.assertEqual("9Zkb10yG", review.critic_id)
        self.assertTrue(review.is_top_critic)
        self.assertTrue(review.is_tomatometer_approved)
        self.assertEqual("POSITIVE", review.sentiment)
        self.assertEqual("fresh", review.fresh_rotten)
        self.assertEqual(dt.date(2024, 9, 18), review.review_date)

    def test_cursor_pagination_requests_until_no_next_page(self) -> None:
        fetcher = FakeFetcher()

        reviews, cache_path = ingest.fetch_all_reviews(
            fetcher,
            ems_id="317d7155-533b-396f-8c1c-34a22e2e8ef9",
            movie_id=1,
        )

        self.assertEqual(2, len(reviews))
        self.assertEqual(Path("page2.json"), cache_path)
        self.assertEqual(2, len(fetcher.urls))
        self.assertIn("type=critic", fetcher.urls[0])
        self.assertIn("after=MQ%3D%3D", fetcher.urls[1])

    def test_conservative_match_scoring_accepts_exact_year_and_rejects_mismatch(self) -> None:
        movie = ingest.CandidateMovie(1, "Barbie", 2023, dt.date(2023, 7, 21), None)
        matched = ingest.parse_media_page(MEDIA_HTML, source_url="https://www.rottentomatoes.com/m/barbie")
        assert matched is not None

        self.assertGreaterEqual(ingest.score_media_match(movie, matched), 0.95)
        mismatched = ingest.RottenTomatoesMedia(
            **{**matched.__dict__, "release_year": 1997}
        )
        self.assertEqual(0.0, ingest.score_media_match(movie, mismatched))

    def test_offline_fetcher_fails_clearly_on_missing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = ingest.TextFetcher(Path(tmpdir), offline=True)
            with self.assertRaises(FileNotFoundError) as context:
                fetcher.get_text("https://www.rottentomatoes.com/m/missing", suffix="html")
            self.assertIn("Missing cached Rotten Tomatoes response", str(context.exception))


class RottenTomatoesDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn, self.schema = make_isolated_postgres_schema()
        ingest.initialize_database(self.conn)
        self.conn.commit()

    def tearDown(self) -> None:
        drop_isolated_postgres_schema(self.conn, self.schema)

    def test_schema_and_upserts_are_idempotent(self) -> None:
        media = ingest.parse_media_page(MEDIA_HTML, source_url="https://www.rottentomatoes.com/m/barbie")
        assert media is not None
        self.conn.execute(
            "INSERT INTO movies (movie_id, title, release_year, release_date) VALUES (1, 'Barbie', 2023, '2023-07-21')"
        )
        match = ingest.MovieMatch(1, media.ems_id, "matched", "unit_test", 1.0, None)
        reviews, _cursor = ingest.parse_reviews_payload(REVIEWS_PAGE_1, ems_id=media.ems_id, movie_id=1)

        for _ in range(2):
            ingest.upsert_media(self.conn, media, fetched_at=ingest.utc_now(), raw_cache_path=Path("media.html"))
            ingest.upsert_movie_match(self.conn, match)
            ingest.insert_reviews(self.conn, reviews, fetched_at=ingest.utc_now(), raw_cache_path=Path("reviews.json"))
            ingest.upsert_state(self.conn, movie_id=1, stage="done", status="completed")
            ingest.insert_issue(
                self.conn,
                issue_source="test",
                issue_type="empty_reviews",
                movie_id=1,
                ems_id=media.ems_id,
                source_url=media.source_url,
                details="sample issue",
            )
        self.conn.commit()

        self.assertIn("rotten_tomatoes_media", table_names(self.conn))
        self.assertEqual(1, self.conn.execute("SELECT COUNT(*) FROM rotten_tomatoes_reviews").fetchone()[0])
        self.assertEqual(1, self.conn.execute("SELECT COUNT(*) FROM rotten_tomatoes_ingest_issues").fetchone()[0])
        source_row = self.conn.execute(
            "SELECT source_movie_id FROM movie_source_ids WHERE source = 'rottentomatoes'"
        ).fetchone()
        self.assertEqual(media.ems_id, source_row[0])
        features = self.conn.execute(
            "SELECT rt_critic_review_count, rt_top_critic_review_count, rt_fresh_review_count FROM analytics.rotten_tomatoes_movie_review_features_v1 WHERE movie_id = 1"
        ).fetchone()
        self.assertEqual((1, 1, 1), tuple(features))

    def test_dry_run_lists_candidates_without_fetching(self) -> None:
        self.conn.execute(
            "INSERT INTO movies (movie_id, title, release_year, release_date) VALUES (1, 'Barbie', 2023, '2023-07-21')"
        )
        self.conn.commit()
        class NoCloseConn:
            def __init__(self, raw):
                self.raw = raw

            def __getattr__(self, name):
                return getattr(self.raw, name)

            def close(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as tmpdir:
            database_url = None
            args = types.SimpleNamespace(
                database_url=database_url,
                cache_dir=Path(tmpdir),
                delay_seconds=5.0,
                movie_limit=1,
                release_year=2023,
                refresh=False,
                offline=True,
                reset_failed=False,
                dry_run=True,
                fail_fast=False,
                issue_source="test",
                user_agent=ingest.DEFAULT_USER_AGENT,
            )
            original_connect = ingest.connect_database
            ingest.connect_database = lambda _url=None: NoCloseConn(self.conn)
            try:
                stdout = io.StringIO()
                stderr = io.StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    self.assertEqual(0, ingest.run(args))
                self.assertIn("Barbie", stdout.getvalue())
                self.assertIn("Candidate movies: 1", stderr.getvalue())
            finally:
                ingest.connect_database = original_connect


if __name__ == "__main__":
    unittest.main()
