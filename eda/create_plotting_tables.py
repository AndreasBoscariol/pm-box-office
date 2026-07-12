#!/usr/bin/env python3
"""Create compact plotting tables for ``eda/plots.ipynb``.

The minimum useful plotting layer has four physical tables:

1. ``analytics.eda_movie_openings``: one row per release run/opening.
   Use for outcome distributions, movie-level scatterplots, grouped boxplots,
   opening-weekend profiles, competition variables, and missingness.
2. ``analytics.eda_movie_days``: one row per movie/release-run/calendar day.
   Use for movie-time trajectories, Friday/Saturday/Sunday profile reshaping,
   Wiki activity lines, and later-performance/decay plots.
3. ``analytics.eda_news_estimates``: one row per source estimate.
   Use for estimate calibration, source bias, range coverage, and source
   agreement plots. This is separated because estimate source/date is a real
   grain, and packing it into the movie table would slow source-level plots.
4. ``analytics.eda_calendar_weekends``: one row per Friday-Sunday weekend.
   Use for true calendar-time series, seasonal plots, ACFs, and heatmaps.

The script rebuilds the tables with set-based ``CREATE TABLE AS`` statements.
These are derived EDA tables, so the fastest path is to drop and recreate them
from the normalized ingest tables whenever upstream data changes.
"""

from __future__ import annotations

import argparse
from typing import Any

from pm_box_office.db.connection import connect_database
from pm_box_office.sources.common.cli import add_database_arg


WIDE_RELEASE_THRESHOLD = 600
LARGE_RELEASE_THRESHOLD = 3000


def relation_exists(conn: Any, relation_name: str) -> bool:
    row = conn.execute("SELECT to_regclass(%s)", (relation_name,)).fetchone()
    return bool(row and row[0])


def table_kind(*, logged: bool) -> str:
    return "TABLE" if logged else "UNLOGGED TABLE"


def empty_estimate_select() -> str:
    return """
        SELECT
            NULL::text AS estimate_source,
            NULL::bigint AS source_prediction_id,
            NULL::bigint AS movie_id,
            NULL::date AS estimate_date,
            NULL::date AS target_start_date,
            NULL::date AS target_end_date,
            NULL::integer AS target_day_count,
            NULL::text AS forecast_metric,
            NULL::numeric AS estimate_low_usd,
            NULL::numeric AS estimate_high_usd,
            NULL::numeric AS estimate_mid_usd,
            NULL::numeric AS estimate_width_usd,
            NULL::text AS source_movie_title,
            NULL::text AS distributor,
            NULL::text AS raw_forecast_text
        WHERE FALSE
    """


def estimate_union_sql(conn: Any) -> str:
    selects: list[str] = []
    if relation_exists(conn, "boxofficepro_weekend_predictions"):
        selects.append(
            """
            SELECT
                'boxofficepro'::text AS estimate_source,
                p.prediction_id::bigint AS source_prediction_id,
                p.movie_id,
                COALESCE(a.discovered_date, p.target_start_date)::date AS estimate_date,
                p.target_start_date,
                p.target_end_date,
                (p.target_end_date - p.target_start_date + 1)::integer AS target_day_count,
                p.forecast_metric,
                p.range_low_usd::numeric AS estimate_low_usd,
                p.range_high_usd::numeric AS estimate_high_usd,
                ((p.range_low_usd::numeric + p.range_high_usd::numeric) / 2.0) AS estimate_mid_usd,
                (p.range_high_usd::numeric - p.range_low_usd::numeric) AS estimate_width_usd,
                p.source_movie_title,
                NULLIF(p.distributor, '') AS distributor,
                p.raw_forecast_text
            FROM boxofficepro_weekend_predictions p
            LEFT JOIN boxofficepro_articles a ON a.article_id = p.article_id
            WHERE p.range_low_usd IS NOT NULL
              AND p.range_high_usd IS NOT NULL
            """
        )
    if relation_exists(conn, "boxofficereport_weekend_predictions"):
        selects.append(
            """
            SELECT
                'boxofficereport'::text AS estimate_source,
                p.prediction_id::bigint AS source_prediction_id,
                p.movie_id,
                COALESCE(p.prediction_made_at::date, a.prediction_made_date, p.target_start_date)::date AS estimate_date,
                p.target_start_date,
                p.target_end_date,
                (p.target_end_date - p.target_start_date + 1)::integer AS target_day_count,
                p.forecast_metric,
                p.weekend_gross_prediction_usd::numeric AS estimate_low_usd,
                p.weekend_gross_prediction_usd::numeric AS estimate_high_usd,
                p.weekend_gross_prediction_usd::numeric AS estimate_mid_usd,
                0::numeric AS estimate_width_usd,
                p.source_movie_title,
                NULLIF(p.distributor, '') AS distributor,
                p.raw_forecast_text
            FROM boxofficereport_weekend_predictions p
            LEFT JOIN boxofficereport_articles a ON a.article_id = p.article_id
            WHERE p.weekend_gross_prediction_usd IS NOT NULL
            """
        )
    if relation_exists(conn, "boxofficeguru_predictions"):
        selects.append(
            """
            SELECT
                'boxofficeguru'::text AS estimate_source,
                p.prediction_id::bigint AS source_prediction_id,
                p.movie_id,
                COALESCE(p.prediction_made_at::date, a.prediction_made_date, p.target_start_date)::date AS estimate_date,
                p.target_start_date,
                p.target_end_date,
                (p.target_end_date - p.target_start_date + 1)::integer AS target_day_count,
                p.forecast_metric,
                p.weekend_gross_prediction_usd::numeric AS estimate_low_usd,
                p.weekend_gross_prediction_usd::numeric AS estimate_high_usd,
                p.weekend_gross_prediction_usd::numeric AS estimate_mid_usd,
                0::numeric AS estimate_width_usd,
                p.source_movie_title,
                NULLIF(p.distributor, '') AS distributor,
                p.raw_forecast_text
            FROM boxofficeguru_predictions p
            LEFT JOIN boxofficeguru_articles a ON a.article_id = p.article_id
            WHERE p.weekend_gross_prediction_usd IS NOT NULL
            """
        )
    if relation_exists(conn, "toddmthatcher_weekend_predictions"):
        selects.append(
            """
            SELECT
                'toddmthatcher'::text AS estimate_source,
                p.prediction_id::bigint AS source_prediction_id,
                p.movie_id,
                COALESCE(p.prediction_made_date, a.discovered_date, p.target_start_date)::date AS estimate_date,
                p.target_start_date,
                p.target_end_date,
                (p.target_end_date - p.target_start_date + 1)::integer AS target_day_count,
                p.forecast_metric,
                p.weekend_gross_prediction_usd::numeric AS estimate_low_usd,
                p.weekend_gross_prediction_usd::numeric AS estimate_high_usd,
                p.weekend_gross_prediction_usd::numeric AS estimate_mid_usd,
                0::numeric AS estimate_width_usd,
                p.source_movie_title,
                NULL::text AS distributor,
                p.raw_forecast_text
            FROM toddmthatcher_weekend_predictions p
            LEFT JOIN toddmthatcher_articles a ON a.article_id = p.article_id
            WHERE p.weekend_gross_prediction_usd IS NOT NULL
            """
        )
    if relation_exists(conn, "boxofficetheory_predictions"):
        selects.append(
            """
            SELECT
                'boxofficetheory'::text AS estimate_source,
                p.prediction_id::bigint AS source_prediction_id,
                p.movie_id,
                COALESCE(p.prediction_made_date, post.published_date, p.release_date)::date AS estimate_date,
                (
                    COALESCE(p.release_date, post.published_date)::date
                    + ((5 - EXTRACT(DOW FROM COALESCE(p.release_date, post.published_date)::date)::integer + 7) % 7)
                )::date AS target_start_date,
                NULL::date AS target_end_date,
                CASE
                    WHEN COALESCE(
                        p.opening_weekend_low_usd,
                        p.opening_weekend_high_usd,
                        p.opening_weekend_pinpoint_usd
                    ) IS NOT NULL
                    THEN p.opening_weekend_day_count
                    WHEN p.weekend_forecast_usd IS NOT NULL
                    THEN p.weekend_day_count
                    ELSE NULL
                END AS target_day_count,
                p.forecast_metric,
                COALESCE(
                    p.opening_weekend_low_usd,
                    p.opening_weekend_pinpoint_usd,
                    p.weekend_forecast_usd
                )::numeric AS estimate_low_usd,
                COALESCE(
                    p.opening_weekend_high_usd,
                    p.opening_weekend_pinpoint_usd,
                    p.weekend_forecast_usd
                )::numeric AS estimate_high_usd,
                COALESCE(
                    p.opening_weekend_pinpoint_usd::numeric,
                    p.weekend_forecast_usd::numeric,
                    (
                        p.opening_weekend_low_usd::numeric
                        + p.opening_weekend_high_usd::numeric
                    ) / 2.0
                ) AS estimate_mid_usd,
                CASE
                    WHEN p.opening_weekend_low_usd IS NOT NULL
                     AND p.opening_weekend_high_usd IS NOT NULL
                    THEN p.opening_weekend_high_usd::numeric - p.opening_weekend_low_usd::numeric
                    ELSE 0::numeric
                END AS estimate_width_usd,
                p.source_movie_title,
                NULLIF(p.distributor, '') AS distributor,
                p.raw_forecast_text
            FROM boxofficetheory_predictions p
            LEFT JOIN boxofficetheory_posts post ON post.post_id = p.post_id
            WHERE COALESCE(
                    p.opening_weekend_low_usd,
                    p.opening_weekend_high_usd,
                    p.opening_weekend_pinpoint_usd,
                    p.weekend_forecast_usd
                  ) IS NOT NULL
            """
        )
    if relation_exists(conn, "boxofficetheory_substack_predictions"):
        selects.append(
            """
            SELECT
                'boxofficetheory_substack'::text AS estimate_source,
                p.prediction_id::bigint AS source_prediction_id,
                p.movie_id,
                COALESCE(p.prediction_made_date, post.published_date, p.release_date)::date AS estimate_date,
                (
                    COALESCE(p.release_date, post.published_date)::date
                    + ((5 - EXTRACT(DOW FROM COALESCE(p.release_date, post.published_date)::date)::integer + 7) % 7)
                )::date AS target_start_date,
                NULL::date AS target_end_date,
                CASE
                    WHEN COALESCE(
                        p.opening_weekend_low_usd,
                        p.opening_weekend_high_usd,
                        p.opening_weekend_pinpoint_usd
                    ) IS NOT NULL
                    THEN p.opening_weekend_day_count
                    WHEN p.weekend_forecast_usd IS NOT NULL
                    THEN p.weekend_day_count
                    ELSE NULL
                END AS target_day_count,
                p.forecast_metric,
                COALESCE(
                    p.opening_weekend_low_usd,
                    p.opening_weekend_pinpoint_usd,
                    p.weekend_forecast_usd
                )::numeric AS estimate_low_usd,
                COALESCE(
                    p.opening_weekend_high_usd,
                    p.opening_weekend_pinpoint_usd,
                    p.weekend_forecast_usd
                )::numeric AS estimate_high_usd,
                COALESCE(
                    p.opening_weekend_pinpoint_usd::numeric,
                    p.weekend_forecast_usd::numeric,
                    (
                        p.opening_weekend_low_usd::numeric
                        + p.opening_weekend_high_usd::numeric
                    ) / 2.0
                ) AS estimate_mid_usd,
                CASE
                    WHEN p.opening_weekend_low_usd IS NOT NULL
                     AND p.opening_weekend_high_usd IS NOT NULL
                    THEN p.opening_weekend_high_usd::numeric - p.opening_weekend_low_usd::numeric
                    ELSE 0::numeric
                END AS estimate_width_usd,
                p.source_movie_title,
                NULLIF(p.distributor, '') AS distributor,
                p.raw_forecast_text
            FROM boxofficetheory_substack_predictions p
            LEFT JOIN boxofficetheory_substack_posts post ON post.post_id = p.post_id
            WHERE COALESCE(
                    p.opening_weekend_low_usd,
                    p.opening_weekend_high_usd,
                    p.opening_weekend_pinpoint_usd,
                    p.weekend_forecast_usd
                  ) IS NOT NULL
            """
        )
    if relation_exists(conn, "edwarddouglas_substack_predictions"):
        selects.append(
            """
            SELECT
                'edwarddouglas_substack'::text AS estimate_source,
                p.prediction_id::bigint AS source_prediction_id,
                p.movie_id,
                COALESCE(p.prediction_made_date, post.published_date, p.release_date)::date AS estimate_date,
                COALESCE(p.release_date, post.published_date)::date AS target_start_date,
                NULL::date AS target_end_date,
                CASE
                    WHEN COALESCE(
                        p.opening_weekend_low_usd,
                        p.opening_weekend_high_usd,
                        p.opening_weekend_pinpoint_usd
                    ) IS NOT NULL
                    THEN p.opening_weekend_day_count
                    WHEN p.weekend_forecast_usd IS NOT NULL
                    THEN p.weekend_day_count
                    ELSE NULL
                END AS target_day_count,
                p.forecast_metric,
                COALESCE(
                    p.opening_weekend_low_usd,
                    p.opening_weekend_pinpoint_usd,
                    p.weekend_forecast_usd
                )::numeric AS estimate_low_usd,
                COALESCE(
                    p.opening_weekend_high_usd,
                    p.opening_weekend_pinpoint_usd,
                    p.weekend_forecast_usd
                )::numeric AS estimate_high_usd,
                COALESCE(
                    p.opening_weekend_pinpoint_usd::numeric,
                    p.weekend_forecast_usd::numeric,
                    (
                        p.opening_weekend_low_usd::numeric
                        + p.opening_weekend_high_usd::numeric
                    ) / 2.0
                ) AS estimate_mid_usd,
                CASE
                    WHEN p.opening_weekend_low_usd IS NOT NULL
                     AND p.opening_weekend_high_usd IS NOT NULL
                    THEN p.opening_weekend_high_usd::numeric - p.opening_weekend_low_usd::numeric
                    ELSE 0::numeric
                END AS estimate_width_usd,
                p.source_movie_title,
                NULLIF(p.distributor, '') AS distributor,
                p.raw_forecast_text
            FROM edwarddouglas_substack_predictions p
            LEFT JOIN edwarddouglas_substack_posts post ON post.post_id = p.post_id
            WHERE COALESCE(
                    p.opening_weekend_low_usd,
                    p.opening_weekend_high_usd,
                    p.opening_weekend_pinpoint_usd,
                    p.weekend_forecast_usd
                  ) IS NOT NULL
            """
        )
    if relation_exists(conn, "joblo_weekend_predictions"):
        selects.append(
            """
            SELECT
                'joblo'::text AS estimate_source,
                p.prediction_id::bigint AS source_prediction_id,
                p.movie_id,
                COALESCE(a.discovered_date, p.target_start_date)::date AS estimate_date,
                p.target_start_date,
                p.target_end_date,
                CASE
                    WHEN p.target_start_date IS NOT NULL AND p.target_end_date IS NOT NULL
                    THEN (p.target_end_date - p.target_start_date + 1)::integer
                    ELSE 3::integer
                END AS target_day_count,
                p.forecast_metric,
                p.range_low_usd::numeric AS estimate_low_usd,
                p.range_high_usd::numeric AS estimate_high_usd,
                ((p.range_low_usd::numeric + p.range_high_usd::numeric) / 2.0) AS estimate_mid_usd,
                (p.range_high_usd::numeric - p.range_low_usd::numeric) AS estimate_width_usd,
                p.source_movie_title,
                NULLIF(p.distributor, '') AS distributor,
                p.raw_forecast_text
            FROM joblo_weekend_predictions p
            LEFT JOIN joblo_articles a ON a.article_id = p.article_id
            """
        )
    if (
        relation_exists(conn, "the_numbers_prediction_rows")
        and relation_exists(conn, "the_numbers_prediction_images")
        and relation_exists(conn, "the_numbers_prediction_articles")
    ):
        selects.append(
            """
            SELECT
                'the_numbers_predictions'::text AS estimate_source,
                p.prediction_id::bigint AS source_prediction_id,
                p.movie_id,
                COALESCE(a.article_date, p.fetched_at::date, p.release_date)::date AS estimate_date,
                p.release_date AS target_start_date,
                NULL::date AS target_end_date,
                3::integer AS target_day_count,
                CONCAT_WS(':', p.table_kind, p.metric, NULLIF(p.row_label, '')) AS forecast_metric,
                p.predicted_usd::numeric AS estimate_low_usd,
                p.predicted_usd::numeric AS estimate_high_usd,
                p.predicted_usd::numeric AS estimate_mid_usd,
                0::numeric AS estimate_width_usd,
                p.source_movie_title,
                NULLIF(p.distributor, '') AS distributor,
                p.raw_text AS raw_forecast_text
            FROM the_numbers_prediction_rows p
            JOIN the_numbers_prediction_images i ON i.image_id = p.image_id
            JOIN the_numbers_prediction_articles a ON a.article_id = i.article_id
            WHERE p.predicted_usd IS NOT NULL
              AND p.metric IN ('weekend_gross', 'predicted_weekend')
            """
        )
    return "\nUNION ALL\n".join(selects) if selects else empty_estimate_select()


def release_schedule_sql(conn: Any) -> str:
    if not relation_exists(conn, "the_numbers_release_schedule"):
        return """
            SELECT
                NULL::bigint AS movie_id,
                NULL::date AS release_date,
                NULL::text AS release_pattern,
                NULL::text AS distributor
            WHERE FALSE
        """
    return """
        SELECT DISTINCT ON (movie_id)
            movie_id,
            release_date,
            release_pattern,
            distributor
        FROM the_numbers_release_schedule
        WHERE movie_id IS NOT NULL
        ORDER BY movie_id, release_date NULLS LAST, fetched_at DESC NULLS LAST
    """


def the_numbers_metadata_sql(conn: Any) -> str:
    if not relation_exists(conn, "the_numbers_movie_metadata"):
        return """
            SELECT
                NULL::bigint AS movie_id,
                NULL::text AS mpa_rating,
                NULL::text AS genre,
                NULL::text AS franchise,
                NULL::numeric AS production_budget_usd
            WHERE FALSE
        """
    return """
        SELECT
            movie_id,
            mpa_rating,
            genre,
            franchise,
            production_budget_usd::numeric AS production_budget_usd
        FROM the_numbers_movie_metadata
    """


def rotten_tomatoes_features_sql(conn: Any) -> str:
    if not relation_exists(conn, "analytics.rotten_tomatoes_movie_review_features_v1"):
        return """
            SELECT
                NULL::bigint AS movie_id,
                NULL::integer AS rt_critic_review_count,
                NULL::integer AS rt_top_critic_review_count,
                NULL::integer AS rt_fresh_review_count,
                NULL::integer AS rt_rotten_review_count,
                NULL::integer AS rt_top_fresh_review_count,
                NULL::integer AS rt_top_rotten_review_count,
                NULL::double precision AS rt_fresh_share,
                NULL::date AS rt_first_review_date,
                NULL::date AS rt_last_review_date,
                NULL::integer AS rt_prerelease_review_count,
                NULL::integer AS rt_top_prerelease_review_count
            WHERE FALSE
        """
    return "SELECT * FROM analytics.rotten_tomatoes_movie_review_features_v1"


def audience_features_sql(conn: Any) -> str:
    if not relation_exists(conn, "analytics.movie_audience_daily_features_v1"):
        return """
            SELECT
                NULL::bigint AS movie_id,
                NULL::date AS snapshot_date,
                NULL::double precision AS imdb_average_rating,
                NULL::integer AS imdb_num_votes,
                NULL::integer AS letterboxd_fan_count,
                NULL::double precision AS letterboxd_average_rating,
                NULL::integer AS imdb_num_votes_delta_1d,
                NULL::integer AS imdb_num_votes_delta_7d
            WHERE FALSE
        """
    return "SELECT * FROM analytics.movie_audience_daily_features_v1"


def wiki_daily_sql(conn: Any) -> str:
    if not (
        relation_exists(conn, "movie_source_ids")
        and relation_exists(conn, "wiki_pageviews_daily")
        and relation_exists(conn, "wiki_revisions")
    ):
        return """
            SELECT
                NULL::bigint AS movie_id,
                NULL::date AS activity_date,
                0::bigint AS wiki_views,
                0::bigint AS wiki_human_revisions,
                0::bigint AS wiki_unique_editors
            WHERE FALSE
        """
    return """
        WITH wiki_matches AS (
            SELECT
                movie_id,
                split_part(source_movie_id, ':', 1) AS language,
                split_part(source_movie_id, ':', 2)::integer AS wiki_page_id
            FROM movie_source_ids
            WHERE source = 'wikipedia'
              AND match_status IN ('matched', 'manual_override')
              AND source_movie_id ~ '^[a-z-]+:[0-9]+$'
        ),
        pageviews AS (
            SELECT
                wm.movie_id,
                pv.view_date::date AS activity_date,
                SUM(pv.views)::bigint AS wiki_views
            FROM wiki_matches wm
            JOIN wiki_pageviews_daily pv
              ON pv.language = wm.language
             AND pv.wiki_page_id = wm.wiki_page_id
            GROUP BY wm.movie_id, pv.view_date::date
        ),
        revisions AS (
            SELECT
                wm.movie_id,
                wr.rev_date::date AS activity_date,
                COUNT(*)::bigint AS wiki_human_revisions,
                COUNT(DISTINCT wr.user_key)::bigint AS wiki_unique_editors
            FROM wiki_matches wm
            JOIN wiki_revisions wr
              ON wr.language = wm.language
             AND wr.wiki_page_id = wm.wiki_page_id
            WHERE wr.is_bot = 0
            GROUP BY wm.movie_id, wr.rev_date::date
        )
        SELECT
            COALESCE(p.movie_id, r.movie_id) AS movie_id,
            COALESCE(p.activity_date, r.activity_date) AS activity_date,
            COALESCE(p.wiki_views, 0)::bigint AS wiki_views,
            COALESCE(r.wiki_human_revisions, 0)::bigint AS wiki_human_revisions,
            COALESCE(r.wiki_unique_editors, 0)::bigint AS wiki_unique_editors
        FROM pageviews p
        FULL OUTER JOIN revisions r
          ON r.movie_id = p.movie_id
         AND r.activity_date = p.activity_date
    """


def create_movie_openings_sql(
    conn: Any,
    *,
    logged: bool,
    wide_threshold: int,
    large_threshold: int,
) -> str:
    return f"""
    DROP TABLE IF EXISTS analytics.eda_movie_openings CASCADE;
    CREATE {table_kind(logged=logged)} analytics.eda_movie_openings AS
    WITH release_schedule AS (
        {release_schedule_sql(conn)}
    ),
    movie_metadata AS (
        {the_numbers_metadata_sql(conn)}
    ),
    rt_features AS (
        {rotten_tomatoes_features_sql(conn)}
    ),
    audience_features AS (
        {audience_features_sql(conn)}
    ),
    wiki_daily AS (
        {wiki_daily_sql(conn)}
    ),
    estimate_rows AS (
        {estimate_union_sql(conn)}
    ),
    opening AS (
        SELECT
            rr.release_run_id,
            rr.movie_id,
            rr.market,
            rr.release_type,
            MIN(dbo.box_office_date::date) AS opening_date
        FROM release_runs rr
        JOIN daily_box_office dbo ON dbo.release_run_id = rr.release_run_id
        WHERE dbo.is_preview = 0
          AND dbo.gross_usd IS NOT NULL
        GROUP BY rr.release_run_id, rr.movie_id, rr.market, rr.release_type
    ),
    opening_window AS (
        SELECT
            opening.*,
            CASE
                WHEN EXTRACT(DOW FROM opening.opening_date)::integer IN (5, 6, 0)
                THEN (
                    opening.opening_date
                    - (((EXTRACT(DOW FROM opening.opening_date)::integer - 5 + 7) % 7) * INTERVAL '1 day')
                )::date
                ELSE (
                    opening.opening_date
                    + (((5 - EXTRACT(DOW FROM opening.opening_date)::integer + 7) % 7) * INTERVAL '1 day')
                )::date
            END AS opening_weekend_start
        FROM opening
    ),
    daily AS (
        SELECT
            rr.release_run_id,
            rr.movie_id,
            dbo.box_office_date::date AS box_office_date,
            dbo.gross_usd::numeric AS gross_usd,
            dbo.theaters,
            dbo.per_theater_usd,
            dbo.cumulative_gross_usd::numeric AS cumulative_gross_usd,
            dbo.is_preview,
            dbo.rank
        FROM release_runs rr
        JOIN daily_box_office dbo ON dbo.release_run_id = rr.release_run_id
    ),
    weekend_actuals AS (
        SELECT
            ow.release_run_id,
            SUM(d.gross_usd) FILTER (
                WHERE d.box_office_date BETWEEN ow.opening_weekend_start AND ow.opening_weekend_start + INTERVAL '2 days'
                  AND d.is_preview = 0
            ) AS opening_weekend_gross_usd,
            SUM(d.gross_usd) FILTER (WHERE d.box_office_date = ow.opening_weekend_start) AS friday_gross_usd,
            SUM(d.gross_usd) FILTER (WHERE d.box_office_date = ow.opening_weekend_start + INTERVAL '1 day') AS saturday_gross_usd,
            SUM(d.gross_usd) FILTER (WHERE d.box_office_date = ow.opening_weekend_start + INTERVAL '2 days') AS sunday_gross_usd,
            MAX(d.theaters) FILTER (
                WHERE d.box_office_date BETWEEN ow.opening_weekend_start AND ow.opening_weekend_start + INTERVAL '2 days'
            ) AS opening_weekend_theaters,
            MAX(d.per_theater_usd) FILTER (
                WHERE d.box_office_date BETWEEN ow.opening_weekend_start AND ow.opening_weekend_start + INTERVAL '2 days'
            ) AS reported_opening_weekend_pta_usd,
            COUNT(*) FILTER (
                WHERE d.box_office_date BETWEEN ow.opening_weekend_start AND ow.opening_weekend_start + INTERVAL '2 days'
                  AND d.gross_usd IS NOT NULL
            ) AS opening_weekend_reported_days
        FROM opening_window ow
        JOIN daily d ON d.release_run_id = ow.release_run_id
        GROUP BY ow.release_run_id
    ),
    opening_day AS (
        SELECT DISTINCT ON (ow.release_run_id)
            ow.release_run_id,
            d.gross_usd AS opening_day_gross_usd,
            d.theaters AS opening_day_theaters
        FROM opening_window ow
        JOIN daily d
          ON d.release_run_id = ow.release_run_id
         AND d.box_office_date = ow.opening_date
        ORDER BY ow.release_run_id, d.box_office_date
    ),
    domestic_totals AS (
        SELECT DISTINCT ON (release_run_id)
            release_run_id,
            cumulative_gross_usd AS domestic_total_gross_usd,
            box_office_date AS latest_gross_date
        FROM daily
        WHERE cumulative_gross_usd IS NOT NULL
        ORDER BY release_run_id, box_office_date DESC
    ),
    later_weekends AS (
        SELECT
            ow.release_run_id,
            SUM(d.gross_usd) FILTER (
                WHERE d.box_office_date BETWEEN ow.opening_weekend_start + INTERVAL '7 days'
                                      AND ow.opening_weekend_start + INTERVAL '9 days'
            ) AS second_weekend_gross_usd
        FROM opening_window ow
        JOIN daily d ON d.release_run_id = ow.release_run_id
        GROUP BY ow.release_run_id
    ),
    estimate_summary AS (
        SELECT DISTINCT ON (ow.release_run_id)
            ow.release_run_id,
            e.estimate_source AS latest_estimate_source,
            e.estimate_date AS latest_estimate_date,
            e.estimate_low_usd AS latest_estimate_low_usd,
            e.estimate_high_usd AS latest_estimate_high_usd,
            e.estimate_mid_usd AS latest_estimate_mid_usd,
            e.estimate_width_usd AS latest_estimate_width_usd
        FROM opening_window ow
        JOIN estimate_rows e ON e.movie_id = ow.movie_id
        WHERE e.estimate_mid_usd IS NOT NULL
          AND e.target_day_count = 3
          AND (e.estimate_date IS NULL OR e.estimate_date <= ow.opening_weekend_start)
          AND (
                e.target_start_date IS NULL
                OR e.target_start_date = ow.opening_weekend_start
              )
        ORDER BY ow.release_run_id, e.estimate_date DESC NULLS LAST, e.estimate_source
    ),
    estimate_agreement AS (
        SELECT
            ow.release_run_id,
            COUNT(*)::integer AS estimate_count,
            COUNT(DISTINCT e.estimate_source)::integer AS estimate_source_count,
            AVG(e.estimate_mid_usd) AS mean_estimate_mid_usd,
            STDDEV_SAMP(e.estimate_mid_usd) AS estimate_mid_stddev_usd,
            MIN(e.estimate_mid_usd) AS min_estimate_mid_usd,
            MAX(e.estimate_mid_usd) AS max_estimate_mid_usd
        FROM opening_window ow
        JOIN estimate_rows e ON e.movie_id = ow.movie_id
        WHERE e.estimate_mid_usd IS NOT NULL
          AND e.target_day_count = 3
          AND (e.estimate_date IS NULL OR e.estimate_date <= ow.opening_weekend_start)
          AND (
                e.target_start_date IS NULL
                OR e.target_start_date = ow.opening_weekend_start
              )
        GROUP BY ow.release_run_id
    ),
    wiki_summary AS (
        SELECT
            ow.release_run_id,
            SUM(w.wiki_views) FILTER (
                WHERE w.activity_date BETWEEN ow.opening_date - INTERVAL '30 days'
                                          AND ow.opening_date - INTERVAL '30 days'
            ) AS wiki_views_at_m30,
            SUM(w.wiki_views) FILTER (
                WHERE w.activity_date BETWEEN ow.opening_date - INTERVAL '30 days'
                                          AND ow.opening_date - INTERVAL '21 days'
            ) AS wiki_views_cume_to_m21,
            SUM(w.wiki_views) FILTER (
                WHERE w.activity_date BETWEEN ow.opening_date - INTERVAL '30 days'
                                          AND ow.opening_date - INTERVAL '14 days'
            ) AS wiki_views_cume_to_m14,
            SUM(w.wiki_views) FILTER (
                WHERE w.activity_date BETWEEN ow.opening_date - INTERVAL '30 days'
                                          AND ow.opening_date - INTERVAL '7 days'
            ) AS wiki_views_cume_to_m7,
            SUM(w.wiki_views) FILTER (
                WHERE w.activity_date BETWEEN ow.opening_date - INTERVAL '30 days'
                                          AND ow.opening_date - INTERVAL '3 days'
            ) AS wiki_views_cume_to_m3,
            SUM(w.wiki_views) FILTER (
                WHERE w.activity_date BETWEEN ow.opening_date - INTERVAL '30 days'
                                          AND ow.opening_date - INTERVAL '1 day'
            ) AS wiki_views_cume_to_m1,
            SUM(w.wiki_views) FILTER (
                WHERE w.activity_date BETWEEN ow.opening_date - INTERVAL '30 days'
                                          AND ow.opening_weekend_start + INTERVAL '2 days'
            ) AS wiki_views_cume_to_opening_sunday,
            SUM(w.wiki_human_revisions) FILTER (
                WHERE w.activity_date BETWEEN ow.opening_date - INTERVAL '30 days'
                                          AND ow.opening_date - INTERVAL '1 day'
            ) AS wiki_revisions_cume_to_m1
        FROM opening_window ow
        LEFT JOIN wiki_daily w ON w.movie_id = ow.movie_id
        GROUP BY ow.release_run_id
    ),
    audience_at_opening AS (
        SELECT DISTINCT ON (ow.release_run_id)
            ow.release_run_id,
            af.snapshot_date AS audience_snapshot_date,
            af.imdb_average_rating,
            af.imdb_num_votes,
            af.letterboxd_fan_count,
            af.letterboxd_average_rating
        FROM opening_window ow
        JOIN audience_features af ON af.movie_id = ow.movie_id
        WHERE af.snapshot_date <= ow.opening_weekend_start + INTERVAL '2 days'
        ORDER BY ow.release_run_id, af.snapshot_date DESC
    ),
    base AS (
        SELECT
            ow.release_run_id,
            ow.movie_id,
            movie.title,
            COALESCE(movie.release_date, ow.opening_date) AS release_date,
            ow.opening_date,
            ow.opening_weekend_start,
            (ow.opening_weekend_start + INTERVAL '2 days')::date AS opening_weekend_end,
            ow.market,
            COALESCE(ow.release_type, rs.release_pattern) AS release_type,
            rs.distributor,
            meta.mpa_rating,
            meta.genre,
            meta.franchise,
            (NULLIF(meta.franchise, '') IS NOT NULL) AS is_franchise,
            meta.production_budget_usd::numeric AS production_budget_usd,
            EXTRACT(YEAR FROM ow.opening_date)::integer AS release_year,
            EXTRACT(MONTH FROM ow.opening_date)::integer AS release_month,
            EXTRACT(WEEK FROM ow.opening_date)::integer AS release_week_of_year,
            EXTRACT(ISODOW FROM ow.opening_date)::integer AS opening_isodow,
            CASE
                WHEN EXTRACT(MONTH FROM ow.opening_date)::integer IN (5, 6, 7, 8) THEN 'summer'
                WHEN EXTRACT(MONTH FROM ow.opening_date)::integer IN (11, 12) THEN 'holiday'
                WHEN EXTRACT(MONTH FROM ow.opening_date)::integer IN (1, 2, 3, 4) THEN 'winter_spring'
                ELSE 'fall'
            END AS season_bucket,
            od.opening_day_gross_usd,
            od.opening_day_theaters,
            wa.opening_weekend_gross_usd,
            wa.friday_gross_usd,
            wa.saturday_gross_usd,
            wa.sunday_gross_usd,
            wa.opening_weekend_theaters,
            wa.reported_opening_weekend_pta_usd,
            CASE
                WHEN wa.opening_weekend_theaters > 0
                THEN wa.opening_weekend_gross_usd / wa.opening_weekend_theaters
                ELSE NULL
            END AS computed_opening_weekend_pta_usd,
            wa.opening_weekend_reported_days,
            dt.domestic_total_gross_usd,
            dt.latest_gross_date,
            lw.second_weekend_gross_usd,
            CASE
                WHEN wa.opening_weekend_gross_usd > 0
                THEN (wa.opening_weekend_gross_usd - lw.second_weekend_gross_usd)
                     / wa.opening_weekend_gross_usd
                ELSE NULL
            END AS second_weekend_drop,
            es.latest_estimate_source,
            es.latest_estimate_date,
            es.latest_estimate_low_usd,
            es.latest_estimate_high_usd,
            es.latest_estimate_mid_usd,
            es.latest_estimate_width_usd,
            ea.estimate_count,
            ea.estimate_source_count,
            ea.mean_estimate_mid_usd,
            ea.estimate_mid_stddev_usd,
            ea.min_estimate_mid_usd,
            ea.max_estimate_mid_usd,
            ws.wiki_views_at_m30,
            ws.wiki_views_cume_to_m21,
            ws.wiki_views_cume_to_m14,
            ws.wiki_views_cume_to_m7,
            ws.wiki_views_cume_to_m3,
            ws.wiki_views_cume_to_m1,
            ws.wiki_views_cume_to_opening_sunday,
            ws.wiki_revisions_cume_to_m1,
            rt.rt_critic_review_count,
            rt.rt_top_critic_review_count,
            rt.rt_fresh_review_count,
            rt.rt_rotten_review_count,
            rt.rt_top_fresh_review_count,
            rt.rt_top_rotten_review_count,
            rt.rt_fresh_share,
            rt.rt_first_review_date,
            rt.rt_last_review_date,
            rt.rt_prerelease_review_count,
            rt.rt_top_prerelease_review_count,
            aud.audience_snapshot_date,
            aud.imdb_average_rating,
            aud.imdb_num_votes,
            aud.letterboxd_fan_count,
            aud.letterboxd_average_rating
        FROM opening_window ow
        JOIN movies movie ON movie.movie_id = ow.movie_id
        LEFT JOIN release_schedule rs ON rs.movie_id = ow.movie_id
        LEFT JOIN movie_metadata meta ON meta.movie_id = ow.movie_id
        LEFT JOIN opening_day od ON od.release_run_id = ow.release_run_id
        LEFT JOIN weekend_actuals wa ON wa.release_run_id = ow.release_run_id
        LEFT JOIN domestic_totals dt ON dt.release_run_id = ow.release_run_id
        LEFT JOIN later_weekends lw ON lw.release_run_id = ow.release_run_id
        LEFT JOIN estimate_summary es ON es.release_run_id = ow.release_run_id
        LEFT JOIN estimate_agreement ea ON ea.release_run_id = ow.release_run_id
        LEFT JOIN wiki_summary ws ON ws.release_run_id = ow.release_run_id
        LEFT JOIN rt_features rt ON rt.movie_id = ow.movie_id
        LEFT JOIN audience_at_opening aud ON aud.release_run_id = ow.release_run_id
    ),
    competition AS (
        SELECT
            b.release_run_id,
            COUNT(other.release_run_id) FILTER (
                WHERE COALESCE(other.opening_weekend_theaters, 0) >= {wide_threshold}
            )::integer AS other_wide_openers,
            COALESCE(SUM(other.opening_weekend_theaters) FILTER (
                WHERE COALESCE(other.opening_weekend_theaters, 0) >= {wide_threshold}
            ), 0)::integer AS other_wide_opener_theaters,
            MAX(other.opening_weekend_gross_usd) AS largest_competing_opener_gross_usd,
            COUNT(other.release_run_id) FILTER (
                WHERE other.genre IS NOT DISTINCT FROM b.genre
            )::integer AS same_genre_competing_openers
        FROM base b
        LEFT JOIN base other
          ON other.opening_weekend_start = b.opening_weekend_start
         AND other.release_run_id <> b.release_run_id
        GROUP BY b.release_run_id
    )
    SELECT
        b.*,
        CASE
            WHEN b.opening_weekend_theaters IS NULL THEN 'unknown'
            WHEN b.opening_weekend_theaters <= 50 THEN 'platform'
            WHEN b.opening_weekend_theaters < {wide_threshold} THEN 'limited'
            WHEN b.opening_weekend_theaters < {large_threshold} THEN 'wide'
            ELSE 'large_wide'
        END AS release_width_bucket,
        COALESCE(b.opening_weekend_theaters, 0) >= {wide_threshold} AS is_wide_release,
        COALESCE(b.opening_weekend_theaters, 0) >= {large_threshold} AS is_large_release,
        CASE WHEN b.opening_weekend_gross_usd > 0 THEN LN(b.opening_weekend_gross_usd) END AS log_opening_weekend_gross,
        CASE WHEN b.domestic_total_gross_usd > 0 THEN LN(b.domestic_total_gross_usd) END AS log_domestic_total_gross,
        CASE WHEN b.opening_weekend_theaters > 0 THEN LN(b.opening_weekend_theaters) END AS log_opening_weekend_theaters,
        CASE WHEN b.computed_opening_weekend_pta_usd > 0 THEN LN(b.computed_opening_weekend_pta_usd) END AS log_opening_weekend_pta,
        CASE WHEN b.production_budget_usd > 0 THEN LN(b.production_budget_usd) END AS log_production_budget,
        CASE
            WHEN b.opening_weekend_gross_usd > 0
            THEN b.domestic_total_gross_usd / b.opening_weekend_gross_usd
            ELSE NULL
        END AS opening_multiple,
        CASE
            WHEN b.production_budget_usd > 0
            THEN b.opening_weekend_gross_usd / b.production_budget_usd
            ELSE NULL
        END AS opening_weekend_to_budget_ratio,
        CASE
            WHEN b.opening_weekend_gross_usd > 0
            THEN b.friday_gross_usd / b.opening_weekend_gross_usd
            ELSE NULL
        END AS friday_share,
        CASE
            WHEN b.opening_weekend_gross_usd > 0
            THEN b.saturday_gross_usd / b.opening_weekend_gross_usd
            ELSE NULL
        END AS saturday_share,
        CASE
            WHEN b.opening_weekend_gross_usd > 0
            THEN b.sunday_gross_usd / b.opening_weekend_gross_usd
            ELSE NULL
        END AS sunday_share,
        CASE
            WHEN b.latest_estimate_mid_usd > 0
            THEN b.opening_weekend_gross_usd / b.latest_estimate_mid_usd
            ELSE NULL
        END AS actual_to_latest_estimate_ratio,
        b.opening_weekend_gross_usd - b.latest_estimate_mid_usd AS latest_estimate_error_usd,
        CASE
            WHEN b.latest_estimate_mid_usd > 0
            THEN (b.opening_weekend_gross_usd - b.latest_estimate_mid_usd) / b.latest_estimate_mid_usd
            ELSE NULL
        END AS latest_estimate_pct_error,
        CASE
            WHEN b.latest_estimate_low_usd IS NULL OR b.latest_estimate_high_usd IS NULL THEN NULL
            ELSE b.opening_weekend_gross_usd BETWEEN b.latest_estimate_low_usd AND b.latest_estimate_high_usd
        END AS actual_inside_latest_estimate_range,
        c.other_wide_openers,
        c.other_wide_opener_theaters,
        c.largest_competing_opener_gross_usd,
        c.same_genre_competing_openers,
        jsonb_build_object(
            'production_budget_usd', b.production_budget_usd IS NULL,
            'latest_estimate_mid_usd', b.latest_estimate_mid_usd IS NULL,
            'wiki_views_cume_to_m1', b.wiki_views_cume_to_m1 IS NULL,
            'rt_fresh_share', b.rt_fresh_share IS NULL,
            'imdb_num_votes', b.imdb_num_votes IS NULL,
            'opening_weekend_theaters', b.opening_weekend_theaters IS NULL,
            'domestic_total_gross_usd', b.domestic_total_gross_usd IS NULL
        ) AS missingness_flags
    FROM base b
    LEFT JOIN competition c ON c.release_run_id = b.release_run_id;

    ALTER TABLE analytics.eda_movie_openings
        ADD PRIMARY KEY (release_run_id);
    CREATE INDEX idx_eda_movie_openings_movie
        ON analytics.eda_movie_openings(movie_id);
    CREATE INDEX idx_eda_movie_openings_opening_weekend
        ON analytics.eda_movie_openings(opening_weekend_start);
    CREATE INDEX idx_eda_movie_openings_width_genre
        ON analytics.eda_movie_openings(release_width_bucket, genre);
    """


def create_movie_days_sql(conn: Any, *, logged: bool) -> str:
    return f"""
    DROP TABLE IF EXISTS analytics.eda_movie_days CASCADE;
    CREATE {table_kind(logged=logged)} analytics.eda_movie_days AS
    WITH audience_features AS (
        {audience_features_sql(conn)}
    ),
    wiki_daily AS (
        {wiki_daily_sql(conn)}
    ),
    daily AS (
        SELECT
            o.release_run_id,
            o.movie_id,
            o.title,
            o.opening_date,
            o.opening_weekend_start,
            o.release_width_bucket,
            o.genre,
            o.distributor,
            o.franchise,
            o.is_franchise,
            dbo.box_office_date::date AS calendar_date,
            dbo.box_office_date::date - o.opening_date AS movie_time_day,
            dbo.gross_usd::numeric AS gross_usd,
            dbo.theaters,
            dbo.per_theater_usd,
            dbo.cumulative_gross_usd::numeric AS cumulative_gross_usd,
            dbo.is_preview = 1 AS is_preview,
            dbo.is_estimate = 1 AS is_estimate,
            dbo.rank,
            EXTRACT(ISODOW FROM dbo.box_office_date::date)::integer AS isodow,
            (
                dbo.box_office_date::date
                - (((EXTRACT(DOW FROM dbo.box_office_date::date)::integer - 5 + 7) % 7) * INTERVAL '1 day')
            )::date AS calendar_weekend_start
        FROM analytics.eda_movie_openings o
        JOIN daily_box_office dbo ON dbo.release_run_id = o.release_run_id
    ),
    daily_with_activity AS (
        SELECT
            d.*,
            w.wiki_views,
            w.wiki_human_revisions,
            w.wiki_unique_editors,
            SUM(COALESCE(w.wiki_views, 0)) OVER (
                PARTITION BY d.release_run_id
                ORDER BY d.calendar_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS wiki_views_cumulative_to_date,
            af.snapshot_date AS audience_snapshot_date,
            af.imdb_average_rating,
            af.imdb_num_votes,
            af.letterboxd_fan_count,
            af.letterboxd_average_rating
        FROM daily d
        LEFT JOIN wiki_daily w
          ON w.movie_id = d.movie_id
         AND w.activity_date = d.calendar_date
        LEFT JOIN LATERAL (
            SELECT *
            FROM audience_features af
            WHERE af.movie_id = d.movie_id
              AND af.snapshot_date <= d.calendar_date
            ORDER BY af.snapshot_date DESC
            LIMIT 1
        ) af ON TRUE
    )
    SELECT
        *,
        ((calendar_weekend_start - opening_weekend_start) / 7)::integer AS weekend_number,
        isodow IN (5, 6, 7) AS is_friday_to_sunday,
        CASE isodow
            WHEN 1 THEN 'Mon'
            WHEN 2 THEN 'Tue'
            WHEN 3 THEN 'Wed'
            WHEN 4 THEN 'Thu'
            WHEN 5 THEN 'Fri'
            WHEN 6 THEN 'Sat'
            WHEN 7 THEN 'Sun'
        END AS day_name,
        CASE WHEN gross_usd > 0 THEN LN(gross_usd) END AS log_gross_usd,
        CASE WHEN theaters > 0 THEN LN(theaters) END AS log_theaters,
        CASE WHEN per_theater_usd > 0 THEN LN(per_theater_usd) END AS log_per_theater_usd
    FROM daily_with_activity;

    CREATE INDEX idx_eda_movie_days_release_date
        ON analytics.eda_movie_days(release_run_id, calendar_date);
    CREATE INDEX idx_eda_movie_days_movie_time
        ON analytics.eda_movie_days(movie_id, movie_time_day);
    CREATE INDEX idx_eda_movie_days_weekend
        ON analytics.eda_movie_days(calendar_weekend_start);
    """


def create_news_estimates_sql(conn: Any, *, logged: bool) -> str:
    return f"""
    DROP TABLE IF EXISTS analytics.eda_news_estimates CASCADE;
    CREATE {table_kind(logged=logged)} analytics.eda_news_estimates AS
    WITH estimate_rows AS (
        {estimate_union_sql(conn)}
    )
    SELECT
        ROW_NUMBER() OVER (
            ORDER BY e.estimate_source, e.source_prediction_id, o.release_run_id
        )::bigint AS eda_estimate_id,
        e.estimate_source,
        e.source_prediction_id,
        o.release_run_id,
        e.movie_id,
        COALESCE(o.title, e.source_movie_title) AS title,
        o.opening_date,
        o.opening_weekend_start,
        o.opening_weekend_gross_usd AS actual_opening_weekend_gross_usd,
        o.release_width_bucket,
        o.genre,
        COALESCE(e.distributor, o.distributor) AS distributor,
        o.franchise,
        o.is_franchise,
        e.estimate_date,
        e.target_start_date,
        e.target_end_date,
        e.target_day_count,
        e.forecast_metric,
        e.estimate_low_usd,
        e.estimate_high_usd,
        e.estimate_mid_usd,
        e.estimate_width_usd,
        e.source_movie_title,
        e.raw_forecast_text,
        (o.release_run_id IS NOT NULL) AS has_opening_match,
        (o.opening_weekend_start - e.estimate_date)::integer AS days_before_opening_weekend,
        o.opening_weekend_gross_usd - e.estimate_mid_usd AS estimate_error_usd,
        ABS(o.opening_weekend_gross_usd - e.estimate_mid_usd) AS absolute_estimate_error_usd,
        CASE
            WHEN e.estimate_mid_usd > 0
            THEN (o.opening_weekend_gross_usd - e.estimate_mid_usd) / e.estimate_mid_usd
            ELSE NULL
        END AS estimate_pct_error,
        CASE
            WHEN e.estimate_mid_usd > 0
            THEN o.opening_weekend_gross_usd / e.estimate_mid_usd
            ELSE NULL
        END AS actual_to_estimate_ratio,
        CASE
            WHEN e.estimate_low_usd IS NULL OR e.estimate_high_usd IS NULL THEN NULL
            ELSE o.opening_weekend_gross_usd BETWEEN e.estimate_low_usd AND e.estimate_high_usd
        END AS actual_inside_range
    FROM estimate_rows e
    LEFT JOIN analytics.eda_movie_openings o
      ON o.movie_id = e.movie_id
     AND e.target_day_count = 3
     AND (
            e.target_start_date IS NULL
            OR e.target_start_date = o.opening_weekend_start
         )
    WHERE e.estimate_mid_usd IS NOT NULL;

    ALTER TABLE analytics.eda_news_estimates
        ADD PRIMARY KEY (eda_estimate_id);
    CREATE INDEX idx_eda_news_estimates_source_date
        ON analytics.eda_news_estimates(estimate_source, estimate_date);
    CREATE INDEX idx_eda_news_estimates_movie
        ON analytics.eda_news_estimates(movie_id, release_run_id);
    CREATE INDEX idx_eda_news_estimates_opening
        ON analytics.eda_news_estimates(opening_weekend_start);
    """


def create_calendar_weekends_sql(*, logged: bool, wide_threshold: int, large_threshold: int) -> str:
    return f"""
    DROP TABLE IF EXISTS analytics.eda_calendar_weekends CASCADE;
    CREATE {table_kind(logged=logged)} analytics.eda_calendar_weekends AS
    WITH daily_market AS (
        SELECT
            (
                dbo.box_office_date::date
                - (((EXTRACT(DOW FROM dbo.box_office_date::date)::integer - 5 + 7) % 7) * INTERVAL '1 day')
            )::date AS weekend_start,
            SUM(dbo.gross_usd::numeric) AS total_weekend_market_gross_usd,
            COUNT(DISTINCT rr.movie_id) AS active_movie_count
        FROM daily_box_office dbo
        JOIN release_runs rr ON rr.release_run_id = dbo.release_run_id
        WHERE dbo.is_preview = 0
          AND dbo.gross_usd IS NOT NULL
          AND EXTRACT(ISODOW FROM dbo.box_office_date::date)::integer IN (5, 6, 7)
        GROUP BY 1
    ),
    opener_agg AS (
        SELECT
            opening_weekend_start AS weekend_start,
            COUNT(*)::integer AS opener_count,
            COUNT(*) FILTER (WHERE opening_weekend_theaters >= {wide_threshold})::integer AS wide_release_count,
            COUNT(*) FILTER (WHERE opening_weekend_theaters >= {large_threshold})::integer AS large_release_count,
            COUNT(*) FILTER (WHERE is_franchise)::integer AS franchise_release_count,
            COUNT(*) FILTER (WHERE genre ILIKE '%horror%')::integer AS horror_release_count,
            SUM(opening_weekend_gross_usd) AS total_opening_weekend_gross_usd,
            SUM(opening_weekend_theaters)::integer AS total_opening_theaters,
            AVG(opening_weekend_gross_usd) AS average_opening_weekend_gross_usd,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY opening_weekend_gross_usd)
                AS median_opening_weekend_gross_usd,
            MAX(opening_weekend_gross_usd) AS top_opener_gross_usd,
            MAX(opening_weekend_theaters) AS max_opener_theaters
        FROM analytics.eda_movie_openings
        GROUP BY opening_weekend_start
    ),
    top_opener AS (
        SELECT DISTINCT ON (opening_weekend_start)
            opening_weekend_start AS weekend_start,
            release_run_id AS top_opener_release_run_id,
            movie_id AS top_opener_movie_id,
            title AS top_opener_title,
            opening_weekend_gross_usd AS top_opener_gross_usd
        FROM analytics.eda_movie_openings
        ORDER BY opening_weekend_start, opening_weekend_gross_usd DESC NULLS LAST
    ),
    holdovers AS (
        SELECT
            d.calendar_weekend_start AS weekend_start,
            d.release_run_id,
            MAX(d.title) AS title,
            SUM(d.gross_usd) AS weekend_gross_usd
        FROM analytics.eda_movie_days d
        JOIN analytics.eda_movie_openings o ON o.release_run_id = d.release_run_id
        WHERE d.is_friday_to_sunday
          AND d.calendar_weekend_start > o.opening_weekend_start
        GROUP BY d.calendar_weekend_start, d.release_run_id
    ),
    top_holdover AS (
        SELECT DISTINCT ON (weekend_start)
            weekend_start,
            release_run_id AS top_holdover_release_run_id,
            title AS top_holdover_title,
            weekend_gross_usd AS top_holdover_gross_usd
        FROM holdovers
        ORDER BY weekend_start, weekend_gross_usd DESC NULLS LAST
    )
    SELECT
        dm.weekend_start,
        (dm.weekend_start + INTERVAL '2 days')::date AS weekend_end,
        EXTRACT(YEAR FROM dm.weekend_start)::integer AS calendar_year,
        EXTRACT(WEEK FROM dm.weekend_start)::integer AS week_of_year,
        EXTRACT(MONTH FROM dm.weekend_start)::integer AS calendar_month,
        CASE
            WHEN EXTRACT(MONTH FROM dm.weekend_start)::integer IN (5, 6, 7, 8) THEN 'summer'
            WHEN EXTRACT(MONTH FROM dm.weekend_start)::integer IN (11, 12) THEN 'holiday'
            WHEN EXTRACT(MONTH FROM dm.weekend_start)::integer IN (1, 2, 3, 4) THEN 'winter_spring'
            ELSE 'fall'
        END AS season_bucket,
        dm.total_weekend_market_gross_usd,
        dm.active_movie_count,
        COALESCE(oa.total_opening_weekend_gross_usd, 0) AS total_opening_weekend_gross_usd,
        COALESCE(oa.opener_count, 0) AS opener_count,
        COALESCE(oa.wide_release_count, 0) AS wide_release_count,
        COALESCE(oa.large_release_count, 0) AS large_release_count,
        COALESCE(oa.franchise_release_count, 0) AS franchise_release_count,
        COALESCE(oa.horror_release_count, 0) AS horror_release_count,
        COALESCE(oa.total_opening_theaters, 0) AS total_opening_theaters,
        oa.average_opening_weekend_gross_usd,
        oa.median_opening_weekend_gross_usd,
        top.top_opener_release_run_id,
        top.top_opener_movie_id,
        top.top_opener_title,
        top.top_opener_gross_usd,
        hold.top_holdover_release_run_id,
        hold.top_holdover_title,
        hold.top_holdover_gross_usd,
        CASE
            WHEN dm.total_weekend_market_gross_usd > 0
            THEN top.top_opener_gross_usd / dm.total_weekend_market_gross_usd
            ELSE NULL
        END AS top_opener_market_share,
        CASE
            WHEN dm.total_weekend_market_gross_usd > 0
            THEN COALESCE(oa.total_opening_weekend_gross_usd, 0) / dm.total_weekend_market_gross_usd
            ELSE NULL
        END AS opener_market_share
    FROM daily_market dm
    LEFT JOIN opener_agg oa ON oa.weekend_start = dm.weekend_start
    LEFT JOIN top_opener top ON top.weekend_start = dm.weekend_start
    LEFT JOIN top_holdover hold ON hold.weekend_start = dm.weekend_start;

    ALTER TABLE analytics.eda_calendar_weekends
        ADD PRIMARY KEY (weekend_start);
    CREATE INDEX idx_eda_calendar_weekends_year_week
        ON analytics.eda_calendar_weekends(calendar_year, week_of_year);
    """


def refresh_plotting_tables(
    conn: Any,
    *,
    logged: bool,
    wide_threshold: int,
    large_threshold: int,
) -> None:
    conn.execute("CREATE SCHEMA IF NOT EXISTS analytics")
    conn.executescript(
        create_movie_openings_sql(
            conn,
            logged=logged,
            wide_threshold=wide_threshold,
            large_threshold=large_threshold,
        )
    )
    conn.executescript(create_movie_days_sql(conn, logged=logged))
    conn.executescript(create_news_estimates_sql(conn, logged=logged))
    conn.executescript(
        create_calendar_weekends_sql(
            logged=logged,
            wide_threshold=wide_threshold,
            large_threshold=large_threshold,
        )
    )
    conn.commit()


def count_rows(conn: Any, table_name: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    return int(row[0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild analytics plotting tables for eda/plots.ipynb."
    )
    add_database_arg(parser)
    parser.add_argument(
        "--logged",
        action="store_true",
        help="Create logged tables. Defaults to UNLOGGED because the tables are rebuildable.",
    )
    parser.add_argument(
        "--wide-threshold",
        type=int,
        default=WIDE_RELEASE_THRESHOLD,
        help="Opening theater count threshold for wide-release flags.",
    )
    parser.add_argument(
        "--large-threshold",
        type=int,
        default=LARGE_RELEASE_THRESHOLD,
        help="Opening theater count threshold for large-release flags.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    conn = connect_database(args.database_url)
    try:
        refresh_plotting_tables(
            conn,
            logged=args.logged,
            wide_threshold=args.wide_threshold,
            large_threshold=args.large_threshold,
        )
        for table_name in (
            "analytics.eda_movie_openings",
            "analytics.eda_movie_days",
            "analytics.eda_news_estimates",
            "analytics.eda_calendar_weekends",
        ):
            print(f"{table_name}: {count_rows(conn, table_name):,} rows")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
