"""Compatibility exports for Boxoffice Pro persistence helpers."""

from pm_box_office.sources.boxofficepro.ingest import (  # noqa: F401
    initialize_database,
    insert_predictions,
    match_predictions,
    upsert_article,
)
