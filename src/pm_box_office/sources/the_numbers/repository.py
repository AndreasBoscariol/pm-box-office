"""Repository exports for The Numbers persistence."""

from pm_box_office.sources.the_numbers.ingest import (  # noqa: F401
    initialize_database,
    insert_daily_chart_rows,
    insert_movie_daily_rows,
    movie_page_imported,
    source_page_recorded,
)

chart_page_imported = source_page_recorded
insert_chart_rows = insert_daily_chart_rows
