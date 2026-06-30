"""Parser exports for The Numbers pages."""

from pm_box_office.sources.the_numbers.ingest import (  # noqa: F401
    DailyChartRow,
    MovieDailyRow,
    ParsedTable,
    TableParser,
    parse_daily_chart,
    parse_movie_page,
)

parse_chart_page = parse_daily_chart
