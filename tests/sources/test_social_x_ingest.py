from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path

from pm_box_office.sources.social_x import ingest


class SocialXQueryGenerationTests(unittest.TestCase):
    def test_low_ambiguity_title_gets_exact_context_and_hashtag_queries(self) -> None:
        movie = ingest.CandidateMovie(
            movie_id=1,
            title="Dune: Part Two",
            release_date="2024-03-01",
            release_year=2024,
        )

        queries = ingest.build_query_variants(movie)
        query_texts = {query.query_text for query in queries}

        self.assertIn('"Dune: Part Two"', query_texts)
        self.assertIn('"Dune: Part Two" trailer', query_texts)
        self.assertIn('"Dune: Part Two" box office', query_texts)
        self.assertIn("#DunePartTwo", query_texts)
        self.assertTrue(all(query.ambiguity_level == "low" for query in queries))

    def test_high_ambiguity_title_never_uses_raw_title_alone(self) -> None:
        movie = ingest.CandidateMovie(
            movie_id=2,
            title="Civil War",
            release_date="2024-04-12",
            release_year=2024,
            distributor="A24",
        )

        queries = ingest.build_query_variants(movie)
        query_texts = {query.query_text for query in queries}

        self.assertNotIn('"Civil War"', query_texts)
        self.assertIn('"Civil War" movie', query_texts)
        self.assertIn('"Civil War" A24', query_texts)
        self.assertIn("#CivilWar movie", query_texts)
        self.assertTrue(all(query.ambiguity_level == "high" for query in queries))


class SocialXPostNormalizationTests(unittest.TestCase):
    def test_normalize_post_accepts_ntscraper_like_shapes(self) -> None:
        raw = {
            "content": "Dune Part Two is selling fast",
            "link": "https://nitter.example/someone/status/1234567890",
            "user": {"username": "someone"},
            "date": "2024-03-01T12:00:00Z",
        }

        post = ingest.normalize_post(
            raw,
            movie_id=1,
            query_text='"Dune: Part Two"',
            query_kind="exact_title",
            source_instance="https://nitter.example",
            target_date="2024-03-01",
            collected_at="2024-03-02T00:00:00+00:00",
        )

        self.assertEqual("1234567890", post.post_key)
        self.assertEqual("1234567890", post.source_post_id)
        self.assertEqual("someone", post.author_handle)
        self.assertEqual("Dune Part Two is selling fast", post.text)

    def test_normalize_post_falls_back_to_stable_hash_without_url(self) -> None:
        kwargs = {
            "movie_id": 1,
            "query_text": '"Dune: Part Two"',
            "query_kind": "exact_title",
            "source_instance": None,
            "target_date": "2024-03-01",
            "collected_at": "2024-03-02T00:00:00+00:00",
        }
        first = ingest.normalize_post({"text": "same", "username": "u", "time": "noon"}, **kwargs)
        second = ingest.normalize_post({"tweet": "same", "username": "u", "timestamp": "noon"}, **kwargs)

        self.assertEqual(first.post_key, second.post_key)
        self.assertIsNone(first.source_post_id)

    def test_dedupe_posts_collapses_same_post_across_queries(self) -> None:
        collected_at = "2024-03-02T00:00:00+00:00"
        exact = ingest.normalize_post(
            {"text": "same", "url": "https://x.test/u/status/42", "username": "u"},
            movie_id=1,
            query_text='"Dune: Part Two"',
            query_kind="exact_title",
            source_instance=None,
            target_date="2024-03-01",
            collected_at=collected_at,
        )
        hashtag = ingest.normalize_post(
            {"text": "same", "url": "https://x.test/u/status/42", "username": "u"},
            movie_id=1,
            query_text="#DunePartTwo",
            query_kind="hashtag",
            source_instance=None,
            target_date="2024-03-01",
            collected_at=collected_at,
        )

        deduped = ingest.dedupe_posts([exact, hashtag])

        self.assertEqual(1, len(deduped))
        self.assertEqual("42", deduped[0].post_key)


class SocialXAggregateTests(unittest.TestCase):
    def test_client_uses_ntscraper_terms_argument(self) -> None:
        class FakeScraper:
            def __init__(self) -> None:
                self.calls = []

            def get_tweets(self, **kwargs):
                self.calls.append(kwargs)
                return {"tweets": []}

        client = ingest.NitterSearchClient(instance="https://nitter.example", delay_seconds=0)
        fake = FakeScraper()
        client._scraper = fake

        result = client.search(
            query_text='"Dune: Part Two"',
            target_date=dt.date(2024, 3, 1),
            max_posts=10,
            language="en",
        )

        self.assertEqual({"tweets": []}, result)
        self.assertEqual('"Dune: Part Two"', fake.calls[0]["terms"])
        self.assertNotIn("term", fake.calls[0])
        self.assertEqual("2024-03-01", fake.calls[0]["since"])
        self.assertEqual("2024-03-02", fake.calls[0]["until"])

    def test_daily_aggregate_marks_cap_hit(self) -> None:
        query = ingest.SocialQuery(1, '"Dune: Part Two"', "exact_title", "low", "en")
        post = ingest.normalize_post(
            {"text": "same", "url": "https://x.test/u/status/42"},
            movie_id=1,
            query_text=query.query_text,
            query_kind=query.query_kind,
            source_instance=None,
            target_date="2024-03-01",
            collected_at="2024-03-02T00:00:00+00:00",
        )
        result = ingest.QueryCollectionResult(
            query=query,
            target_date="2024-03-01",
            posts=[post],
            raw_hit_count=500,
            collection_status="max_posts_cap_hit",
            max_posts_cap_hit=True,
            cache_path=Path("cache.json"),
            fetched=True,
        )

        aggregate = ingest.aggregate_daily_results(
            movie_id=1,
            target_date="2024-03-01",
            source_instance=None,
            collected_at="2024-03-02T00:00:00+00:00",
            query_results=[result],
        )

        self.assertEqual("max_posts_cap_hit", aggregate.collection_status)
        self.assertTrue(aggregate.max_posts_cap_hit)
        self.assertEqual(1, aggregate.observed_count)

    def test_cache_roundtrip_without_live_client(self) -> None:
        query = ingest.SocialQuery(1, '"Dune: Part Two"', "exact_title", "low", "en")
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ingest.NitterSearchCache(Path(tmpdir), refresh=False, offline=False)
            path = cache.path_for(
                source_instance=None,
                query_text=query.query_text,
                target_date="2024-03-01",
                language="en",
                max_posts=10,
            )
            cache.write(path, {"tweets": [{"text": "cached", "url": "https://x.test/u/status/7"}]})
            offline_cache = ingest.NitterSearchCache(Path(tmpdir), refresh=False, offline=True)

            result = ingest.collect_query_day(
                client=ingest.NitterSearchClient(delay_seconds=0),
                cache=offline_cache,
                query=query,
                target_date=dt.date(2024, 3, 1),
                max_posts=10,
                source_instance=None,
                collected_at="2024-03-02T00:00:00+00:00",
            )

        self.assertEqual("complete_enough", result.collection_status)
        self.assertFalse(result.fetched)
        self.assertEqual(1, len(result.posts))
        self.assertEqual("7", result.posts[0].post_key)


class SocialXCliTests(unittest.TestCase):
    def test_cli_help_includes_database_url_and_poc_controls(self) -> None:
        help_text = ingest.build_parser().format_help()

        self.assertIn("--database-url", help_text)
        self.assertIn("--max-posts-per-query", help_text)
        self.assertIn("--instance", help_text)
        self.assertIn("--offline", help_text)


if __name__ == "__main__":
    unittest.main()
