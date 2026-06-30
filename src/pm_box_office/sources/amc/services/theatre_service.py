"""Theatre synchronization service."""

from __future__ import annotations

from typing import Any

from pm_box_office.sources.amc import db
from pm_box_office.sources.amc.client import HtmlFetcher
from pm_box_office.sources.amc.sitemap import fetch_theatre_sitemap, parse_theatre_sitemap


def sync_theatres(conn: Any, fetcher: HtmlFetcher) -> int:
    xml_text, _cache_path = fetch_theatre_sitemap(fetcher)
    theatres = parse_theatre_sitemap(xml_text)
    return db.upsert_theatres(conn, theatres)
