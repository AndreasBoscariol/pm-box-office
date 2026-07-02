"""Compatibility exports for Rotten Tomatoes persistence helpers."""

from pm_box_office.sources.rotten_tomatoes.ingest import (  # noqa: F401
    initialize_database,
    insert_issue,
    insert_reviews,
    upsert_media,
    upsert_movie_match,
)

