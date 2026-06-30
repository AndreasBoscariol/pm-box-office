"""Repository exports for Wikipedia persistence."""

from pm_box_office.sources.wikipedia.ingest import (  # noqa: F401
    initialize_wikipedia_database,
    insert_pageviews,
    insert_revisions,
    select_candidate_movies,
    upsert_wiki_match,
)

load_candidate_movies = select_candidate_movies
upsert_pageviews = insert_pageviews
upsert_revisions = insert_revisions
