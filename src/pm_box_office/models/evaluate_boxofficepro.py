#!/usr/bin/env python3
"""Evaluate Boxoffice Pro weekend forecasts against The Numbers actuals."""

from __future__ import annotations

import argparse
import datetime as dt
import math
from dataclasses import dataclass
from typing import Any, Iterable

from pm_box_office.db.connection import connect_database


@dataclass(frozen=True)
class ForecastActualRow:
    prediction_id: int
    article_id: int
    article_url: str
    article_title: str
    movie_id: int
    movie_title: str
    source_movie_title: str
    forecast_metric: str
    source_context: str
    published_date: dt.date | None
    target_start_date: dt.date
    target_end_date: dt.date
    lead_days: int | None
    lead_bucket: str
    range_low_usd: float
    range_high_usd: float
    midpoint_usd: float
    actual_usd: float
    signed_error_usd: float
    absolute_error_usd: float
    percentage_error: float | None
    absolute_percentage_error: float | None
    interval_hit: bool


@dataclass(frozen=True)
class SummaryMetrics:
    label: str
    row_count: int
    movie_count: int
    article_count: int
    mean_actual_usd: float
    mean_midpoint_usd: float
    pearson_correlation: float | None
    r2_gross: float | None
    mse_usd: float
    mae_usd: float
    rmse_usd: float
    mape: float | None
    accuracy_pct: float | None
    mean_signed_error_usd: float
    interval_coverage: float


def fetch_forecast_actual_rows(
    conn: Any,
    *,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    forecast_metric: str | None = None,
    source_context: str | None = None,
    min_match_score: float | None = None,
    include_unmatched: bool = False,
) -> list[ForecastActualRow]:
    filters = [
        "p.target_start_date IS NOT NULL",
        "p.target_end_date IS NOT NULL",
        "p.target_end_date >= p.target_start_date",
    ]
    params: list[Any] = []
    if not include_unmatched:
        filters.append("p.matched_movie_id IS NOT NULL")
    if start_date is not None:
        filters.append("p.target_start_date >= %s")
        params.append(start_date)
    if end_date is not None:
        filters.append("p.target_end_date <= %s")
        params.append(end_date)
    if forecast_metric:
        filters.append("p.forecast_metric = %s")
        params.append(forecast_metric)
    if source_context:
        filters.append("p.source_context = %s")
        params.append(source_context)
    if min_match_score is not None:
        filters.append("COALESCE(p.match_score, 0) >= %s")
        params.append(min_match_score)

    where_sql = " AND ".join(filters)
    rows = conn.execute(
        f"""
        SELECT
            p.prediction_id,
            p.article_id,
            a.article_url,
            a.title AS article_title,
            m.movie_id,
            m.title AS movie_title,
            p.source_movie_title,
            p.forecast_metric,
            p.source_context,
            a.discovered_date::date AS published_date,
            p.target_start_date::date,
            p.target_end_date::date,
            (p.target_start_date::date - a.discovered_date::date) AS lead_days,
            p.range_low_usd,
            p.range_high_usd,
            SUM(dbo.gross_usd)::double precision AS actual_usd,
            COUNT(DISTINCT dbo.box_office_date::date) AS actual_day_count,
            (p.target_end_date::date - p.target_start_date::date + 1) AS expected_day_count
        FROM boxofficepro_weekend_predictions p
        JOIN boxofficepro_articles a ON a.article_id = p.article_id
        JOIN movies m ON m.movie_id = p.matched_movie_id
        JOIN release_runs rr
          ON rr.movie_id = m.movie_id
         AND rr.market = p.market
         AND rr.source = 'the_numbers'
         AND rr.release_type = 'movie_page_full_run'
        JOIN daily_box_office dbo
          ON dbo.release_run_id = rr.release_run_id
         AND dbo.market = p.market
         AND dbo.source = 'the_numbers'
         AND dbo.is_preview = 0
         AND dbo.gross_usd IS NOT NULL
         AND dbo.box_office_date::date >= p.target_start_date::date
         AND dbo.box_office_date::date <= p.target_end_date::date
        WHERE {where_sql}
        GROUP BY
            p.prediction_id,
            p.article_id,
            a.article_url,
            a.title,
            m.movie_id,
            m.title,
            p.source_movie_title,
            p.forecast_metric,
            p.source_context,
            a.discovered_date,
            p.target_start_date,
            p.target_end_date,
            p.range_low_usd,
            p.range_high_usd
        HAVING COUNT(DISTINCT dbo.box_office_date::date) = (p.target_end_date::date - p.target_start_date::date + 1)
        ORDER BY p.target_start_date, p.source_rank NULLS LAST, p.source_movie_title
        """,
        params,
    ).fetchall()
    return [build_forecast_actual_row(row) for row in rows]


def build_forecast_actual_row(row: Any) -> ForecastActualRow:
    published_date = coerce_optional_date(row[9])
    target_start_date = coerce_date(row[10])
    lead_days = int(row[12]) if row[12] is not None else None
    low = float(row[13])
    high = float(row[14])
    midpoint = (low + high) / 2.0
    actual = float(row[15])
    signed_error = midpoint - actual
    absolute_error = abs(signed_error)
    percentage_error = signed_error / actual if actual else None
    absolute_percentage_error = absolute_error / actual if actual else None
    return ForecastActualRow(
        prediction_id=int(row[0]),
        article_id=int(row[1]),
        article_url=str(row[2]),
        article_title=str(row[3]),
        movie_id=int(row[4]),
        movie_title=str(row[5]),
        source_movie_title=str(row[6]),
        forecast_metric=str(row[7]),
        source_context=str(row[8]),
        published_date=published_date,
        target_start_date=target_start_date,
        target_end_date=coerce_date(row[11]),
        lead_days=lead_days,
        lead_bucket=lead_bucket_for_days(lead_days),
        range_low_usd=low,
        range_high_usd=high,
        midpoint_usd=midpoint,
        actual_usd=actual,
        signed_error_usd=signed_error,
        absolute_error_usd=absolute_error,
        percentage_error=percentage_error,
        absolute_percentage_error=absolute_percentage_error,
        interval_hit=low <= actual <= high,
    )


def summarize_rows(label: str, rows: Iterable[ForecastActualRow]) -> SummaryMetrics | None:
    row_list = list(rows)
    if not row_list:
        return None
    actuals = [row.actual_usd for row in row_list]
    forecasts = [row.midpoint_usd for row in row_list]
    absolute_errors = [row.absolute_error_usd for row in row_list]
    signed_errors = [row.signed_error_usd for row in row_list]
    squared_errors = [error**2 for error in signed_errors]
    absolute_percentage_errors = [
        row.absolute_percentage_error for row in row_list if row.absolute_percentage_error is not None
    ]
    mape = mean(absolute_percentage_errors) if absolute_percentage_errors else None
    return SummaryMetrics(
        label=label,
        row_count=len(row_list),
        movie_count=len({row.movie_id for row in row_list}),
        article_count=len({row.article_id for row in row_list}),
        mean_actual_usd=mean(actuals),
        mean_midpoint_usd=mean(forecasts),
        pearson_correlation=pearson_correlation(forecasts, actuals),
        r2_gross=r2_score(actuals, forecasts),
        mse_usd=mean(squared_errors),
        mae_usd=mean(absolute_errors),
        rmse_usd=math.sqrt(mean(squared_errors)),
        mape=mape,
        accuracy_pct=1.0 - mape if mape is not None else None,
        mean_signed_error_usd=mean(signed_errors),
        interval_coverage=mean([1.0 if row.interval_hit else 0.0 for row in row_list]),
    )


def grouped_summaries(rows: list[ForecastActualRow]) -> list[SummaryMetrics]:
    summaries: list[SummaryMetrics] = []
    for field_name, label in (
        ("forecast_metric", "forecast_metric"),
        ("source_context", "source_context"),
        ("lead_bucket", "lead_bucket"),
    ):
        values = sorted({getattr(row, field_name) for row in rows})
        for value in values:
            summary = summarize_rows(f"{label}={value}", [row for row in rows if getattr(row, field_name) == value])
            if summary is not None:
                summaries.append(summary)
    return summaries


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys):
        raise ValueError("correlation inputs must have equal length")
    if len(xs) < 2:
        return None
    mean_x = mean(xs)
    mean_y = mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denominator_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    denominator = denominator_x * denominator_y
    if denominator == 0:
        return None
    return numerator / denominator


def r2_score(actuals: list[float], predictions: list[float]) -> float | None:
    if len(actuals) != len(predictions):
        raise ValueError("r2 inputs must have equal length")
    if len(actuals) < 2:
        return None
    actual_mean = mean(actuals)
    total = sum((actual - actual_mean) ** 2 for actual in actuals)
    if total == 0:
        return None
    residual = sum((actual - prediction) ** 2 for actual, prediction in zip(actuals, predictions))
    return 1.0 - residual / total


def lead_bucket_for_days(lead_days: int | None) -> str:
    if lead_days is None:
        return "missing_lead"
    if lead_days < 0:
        return "after_open"
    if lead_days <= 2:
        return "0_2d"
    if lead_days <= 7:
        return "3_7d"
    if lead_days <= 14:
        return "8_14d"
    return "15d_plus"


def coerce_date(value: Any) -> dt.date:
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value))


def coerce_optional_date(value: Any) -> dt.date | None:
    if value is None:
        return None
    return coerce_date(value)


def format_usd(value: float) -> str:
    return f"${value:,.0f}"


def format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def print_summary(summary: SummaryMetrics) -> None:
    print(summary.label)
    print(f"  rows: {summary.row_count}")
    print(f"  movies: {summary.movie_count}")
    print(f"  articles: {summary.article_count}")
    print(f"  mean actual: {format_usd(summary.mean_actual_usd)}")
    print(f"  mean midpoint forecast: {format_usd(summary.mean_midpoint_usd)}")
    print(f"  Pearson r: {format_float(summary.pearson_correlation)}")
    print(f"  R2 gross: {format_float(summary.r2_gross)}")
    print(f"  MSE: {format_usd(summary.mse_usd)}")
    print(f"  MAE: {format_usd(summary.mae_usd)}")
    print(f"  RMSE: {format_usd(summary.rmse_usd)}")
    print(f"  MAPE: {format_percent(summary.mape)}")
    print(f"  accuracy: {format_percent(summary.accuracy_pct)}")
    print(f"  mean signed error: {format_usd(summary.mean_signed_error_usd)}")
    print(f"  interval coverage: {format_percent(summary.interval_coverage)}")


def parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected YYYY-MM-DD date, got {value!r}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Boxoffice Pro forecast ranges against The Numbers actuals."
    )
    parser.add_argument("--database-url", help="PostgreSQL URL. Defaults to DATABASE_URL/POSTGRES_DSN/.env.")
    parser.add_argument("--start-date", type=parse_date, help="Minimum forecast target_start_date, YYYY-MM-DD.")
    parser.add_argument("--end-date", type=parse_date, help="Maximum forecast target_end_date, YYYY-MM-DD.")
    parser.add_argument("--forecast-metric", help="Restrict to one boxofficepro_weekend_predictions.forecast_metric.")
    parser.add_argument("--source-context", help="Restrict to one parser source_context.")
    parser.add_argument("--min-match-score", type=float, help="Minimum Boxoffice Pro to The Numbers match score.")
    parser.add_argument(
        "--include-unmatched",
        action="store_true",
        help="Do not pre-filter unmatched predictions. Rows still require matched actuals to be scored.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.start_date and args.end_date and args.end_date < args.start_date:
        parser.error("--end-date must be on or after --start-date")
    if args.min_match_score is not None and not 0 <= args.min_match_score <= 1:
        parser.error("--min-match-score must be between 0 and 1")

    conn = connect_database(args.database_url)
    try:
        rows = fetch_forecast_actual_rows(
            conn,
            start_date=args.start_date,
            end_date=args.end_date,
            forecast_metric=args.forecast_metric,
            source_context=args.source_context,
            min_match_score=args.min_match_score,
            include_unmatched=args.include_unmatched,
        )
    finally:
        conn.close()

    overall = summarize_rows("overall", rows)
    if overall is None:
        print("No eligible Boxoffice Pro forecast rows found with complete The Numbers actuals.")
        return 0

    print_summary(overall)
    group_metrics = grouped_summaries(rows)
    if group_metrics:
        print()
        print("Grouped summaries")
        for summary in group_metrics:
            print()
            print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
