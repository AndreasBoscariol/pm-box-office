"""Theatre synchronization service."""

from __future__ import annotations

from typing import Any

from scripts.ingest.amc import db
from scripts.ingest.amc.client import HtmlFetcher
from scripts.ingest.amc.sitemap import fetch_theatre_sitemap, parse_theatre_sitemap


def sync_theatres(conn: Any, fetcher: HtmlFetcher) -> int:
    xml_text, _cache_path = fetch_theatre_sitemap(fetcher)
    theatres = parse_theatre_sitemap(xml_text)
    return db.upsert_theatres(conn, theatres)
