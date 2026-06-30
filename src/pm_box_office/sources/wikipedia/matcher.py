"""Wikipedia movie matching exports."""

from pm_box_office.sources.wikipedia.ingest import (  # noqa: F401
    CandidateMovie,
    WikiMatch,
    match_wikipedia_page,
)

find_wiki_match = match_wikipedia_page
