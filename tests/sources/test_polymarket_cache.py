from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pm_box_office.sources.polymarket import accounts


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class PolymarketCacheTests(unittest.TestCase):
    def test_fetch_writes_compressed_sqlite_cache_without_loose_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            payload = {"id": 1, "slug": "movies"}
            client = accounts.PolymarketClient(cache_dir, sleep_seconds=0, retries=0)

            with patch.object(accounts.urllib.request, "urlopen", return_value=FakeResponse(payload)):
                result = client.get_json(accounts.GAMMA_BASE_URL, "/tags/slug/movies")

            self.assertEqual(payload, result)
            self.assertTrue((cache_dir / accounts.CompressedSqliteCache.filename).exists())
            self.assertEqual([], list(cache_dir.glob("*.json")))

            with patch.object(accounts.urllib.request, "urlopen") as urlopen:
                cached = client.get_json(accounts.GAMMA_BASE_URL, "/tags/slug/movies")

            self.assertEqual(payload, cached)
            urlopen.assert_not_called()

    def test_legacy_loose_json_is_read_and_migrated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            payload = {"legacy": True}
            url = f"{accounts.GAMMA_BASE_URL}/tags/slug/movies"
            legacy_path = cache_dir / f"{accounts.PolymarketClient._cache_key(url)}.json"
            legacy_path.write_text(json.dumps(payload), encoding="utf-8")
            client = accounts.PolymarketClient(cache_dir, sleep_seconds=0, retries=0)

            with patch.object(accounts.urllib.request, "urlopen") as urlopen:
                result = client.get_json(accounts.GAMMA_BASE_URL, "/tags/slug/movies")

            self.assertEqual(payload, result)
            urlopen.assert_not_called()
            sqlite_cache = accounts.CompressedSqliteCache(cache_dir)
            self.assertEqual(payload, sqlite_cache.read(accounts.PolymarketClient._cache_key(url)))

    def test_compact_legacy_cache_can_delete_imported_json_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            urls = [
                f"{accounts.GAMMA_BASE_URL}/markets?limit={index}"
                for index in range(3)
            ]
            for index, url in enumerate(urls):
                path = cache_dir / f"{accounts.PolymarketClient._cache_key(url)}.json"
                path.write_text(json.dumps({"index": index}), encoding="utf-8")

            result = accounts.compact_legacy_api_cache(cache_dir, delete_legacy_files=True)

            self.assertEqual(3, result.imported)
            self.assertEqual(3, result.deleted)
            self.assertEqual([], list(cache_dir.glob("*.json")))
            sqlite_cache = accounts.CompressedSqliteCache(cache_dir)
            for index, url in enumerate(urls):
                self.assertEqual({"index": index}, sqlite_cache.read(accounts.PolymarketClient._cache_key(url)))

    def test_compact_cache_tree_imports_immediate_child_cache_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root_dir = Path(tmp)
            cache_a = root_dir / "api_cache"
            cache_b = root_dir / "api_cache_legacy"
            cache_a.mkdir()
            cache_b.mkdir()
            urls = [
                f"{accounts.DATA_BASE_URL}/trades?market={index}"
                for index in range(2)
            ]
            for index, (cache_dir, url) in enumerate(zip((cache_a, cache_b), urls, strict=True)):
                path = cache_dir / f"{accounts.PolymarketClient._cache_key(url)}.json"
                path.write_text(json.dumps({"index": index}), encoding="utf-8")

            results = accounts.compact_legacy_api_cache_tree(root_dir, delete_legacy_files=True)

            self.assertEqual({cache_a, cache_b}, set(results))
            self.assertEqual(1, results[cache_a].imported)
            self.assertEqual(1, results[cache_b].imported)
            self.assertEqual([], list(cache_a.glob("*.json")))
            self.assertEqual([], list(cache_b.glob("*.json")))
            for index, (cache_dir, url) in enumerate(zip((cache_a, cache_b), urls, strict=True)):
                sqlite_cache = accounts.CompressedSqliteCache(cache_dir)
                self.assertEqual({"index": index}, sqlite_cache.read(accounts.PolymarketClient._cache_key(url)))


if __name__ == "__main__":
    unittest.main()
