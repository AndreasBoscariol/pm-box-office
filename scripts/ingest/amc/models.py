"""Shared AMC ingestion model aliases.

The concrete dataclasses currently live beside the code that persists or
parses them. This module gives the web app and worker a stable import location
as the package grows.
"""

from __future__ import annotations

from scripts.ingest.amc.db import (
    CollectionTask,
    MovieInventoryRow,
    StoredShowtime,
    StoredTheatre,
)
from scripts.ingest.amc.parsers import SeatFill, ShowtimeRecord
from scripts.ingest.amc.sitemap import AmcTheatre

__all__ = [
    "AmcTheatre",
    "CollectionTask",
    "MovieInventoryRow",
    "SeatFill",
    "ShowtimeRecord",
    "StoredShowtime",
    "StoredTheatre",
]
