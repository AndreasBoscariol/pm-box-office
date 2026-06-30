"""Parse AMC theatre sitemap records."""

from __future__ import annotations

import re
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from scripts.ingest.amc.client import HtmlFetcher
from scripts.ingest.amc.timezones import infer_us_timezone


SITEMAP_THEATRES_URL = "https://www.amctheatres.com/sitemaps/sitemap-theatres.xml"
SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
PAGEMAP_NS = "{http://www.google.com/schemas/sitemap-pagemap/1.0}"


@dataclass(frozen=True)
class AmcTheatre:
    amc_theatre_id: int
    slug: str
    theatre_url: str
    name: str
    address_line1: str
    city: str
    state: str
    postal_code: str
    latitude: float | None
    longitude: float | None
    timezone: str
    inferred_screen_count: int | None


def fetch_theatre_sitemap(fetcher: HtmlFetcher) -> tuple[str, str]:
    xml_text, cache_path, _fetched = fetcher.get(SITEMAP_THEATRES_URL)
    return xml_text, str(cache_path)


def parse_theatre_sitemap(xml_text: str) -> list[AmcTheatre]:
    root = ET.fromstring(xml_text)
    theatres: list[AmcTheatre] = []
    for url_node in root.findall(f"{SITEMAP_NS}url"):
        loc = text_or_empty(url_node.find(f"{SITEMAP_NS}loc"))
        if not loc:
            continue
        attributes = pagemap_attributes(url_node)
        theatre_id = attributes.get("theatreId")
        title = attributes.get("title", "")
        state = attributes.get("state", "")
        latitude = parse_float(attributes.get("latitude"))
        longitude = parse_float(attributes.get("longitude"))
        if not theatre_id or not title:
            continue
        theatres.append(
            AmcTheatre(
                amc_theatre_id=int(theatre_id),
                slug=slug_from_url(loc),
                theatre_url=loc,
                name=title,
                address_line1=attributes.get("addressLine1", ""),
                city=attributes.get("city", ""),
                state=state,
                postal_code=attributes.get("postalCode", ""),
                latitude=latitude,
                longitude=longitude,
                timezone=infer_us_timezone(latitude, longitude, state),
                inferred_screen_count=infer_screen_count(title),
            )
        )
    return sorted(theatres, key=lambda theatre: theatre.amc_theatre_id)


def pagemap_attributes(url_node: ET.Element) -> dict[str, str]:
    attributes: dict[str, str] = {}
    for attr in url_node.findall(f".//{PAGEMAP_NS}Attribute"):
        name = attr.attrib.get("name")
        if name:
            attributes[name] = text_or_empty(attr)
    return attributes


def slug_from_url(url: str) -> str:
    path = urllib.parse.urlparse(url).path.rstrip("/")
    return path.rsplit("/", 1)[-1]


def infer_screen_count(theatre_name: str) -> int | None:
    match = re.search(r"(\d{1,2})\s*$", theatre_name.strip())
    if not match:
        return None
    count = int(match.group(1))
    return count if 1 <= count <= 40 else None


def parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def text_or_empty(node: ET.Element | None) -> str:
    if node is None or node.text is None:
        return ""
    return node.text.strip()
