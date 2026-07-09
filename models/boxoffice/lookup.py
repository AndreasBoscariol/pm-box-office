"""Read-only forecast timeline lookup."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .constants import COMPONENT_TABLE, FORECAST_TABLE


def get_movie_forecast_timeline(
    conn: Any,
    *,
    movie_id: int | None = None,
    release_run_id: int | None = None,
    model_version: str | None = None,
) -> pd.DataFrame:
    if movie_id is None and release_run_id is None:
        raise ValueError("movie_id or release_run_id is required")

    params: list[Any] = []
    predicates = []
    if movie_id is not None:
        predicates.append("f.movie_id = %s")
        params.append(movie_id)
    if release_run_id is not None:
        predicates.append("f.release_run_id = %s")
        params.append(release_run_id)
    if model_version is not None:
        predicates.append("f.model_version = %s")
        params.append(model_version)
    else:
        predicates.append(
            f"""
            f.model_version = (
                SELECT latest.model_version
                FROM {FORECAST_TABLE} latest
                WHERE latest.movie_id = f.movie_id
                  AND latest.release_run_id = f.release_run_id
                ORDER BY latest.created_at DESC, latest.model_version DESC
                LIMIT 1
            )
            """
        )

    sql = f"""
        SELECT
            f.*,
            jsonb_agg(
                jsonb_build_object(
                    'component_day', c.component_day,
                    'component_type', c.component_type,
                    'component_point_usd', c.component_point_usd,
                    'component_sigma_log', c.component_sigma_log,
                    'component_source', c.component_source
                )
                ORDER BY CASE c.component_day
                    WHEN 'Friday' THEN 1
                    WHEN 'Saturday' THEN 2
                    WHEN 'Sunday' THEN 3
                    ELSE 9
                END
            ) FILTER (WHERE c.forecast_id IS NOT NULL) AS components
        FROM {FORECAST_TABLE} f
        LEFT JOIN {COMPONENT_TABLE} c USING (forecast_id)
        WHERE {' AND '.join(predicates)}
        GROUP BY f.forecast_id
        ORDER BY
            CASE f.regime
                WHEN 'pre_release' THEN 1
                WHEN 'live_friday' THEN 2
                WHEN 'live_saturday' THEN 3
                WHEN 'live_sunday' THEN 4
                ELSE 9
            END,
            COALESCE(f.origin_day, 99),
            CASE f.origin_key
                WHEN 'FRI_10:00' THEN 10
                WHEN 'FRI_12:00' THEN 12
                WHEN 'FRI_14:00' THEN 14
                WHEN 'FRI_16:00' THEN 16
                WHEN 'FRI_18:00' THEN 18
                WHEN 'FRI_20:00' THEN 20
                WHEN 'FRI_EOD' THEN 24
                WHEN 'SAT_10:00' THEN 10
                WHEN 'SAT_12:00' THEN 12
                WHEN 'SAT_14:00' THEN 14
                WHEN 'SAT_16:00' THEN 16
                WHEN 'SAT_18:00' THEN 18
                WHEN 'SAT_20:00' THEN 20
                WHEN 'SAT_EOD' THEN 24
                WHEN 'SUN_10:00' THEN 10
                WHEN 'SUN_12:00' THEN 12
                WHEN 'SUN_14:00' THEN 14
                WHEN 'SUN_16:00' THEN 16
                WHEN 'SUN_18:00' THEN 18
                WHEN 'SUN_20:00' THEN 20
                WHEN 'SUN_EOD' THEN 24
                ELSE 0
            END,
            CASE f.target
                WHEN 'opening_weekend' THEN 1
                WHEN 'friday' THEN 2
                WHEN 'saturday' THEN 3
                WHEN 'sunday' THEN 4
                ELSE 9
            END
    """
    cursor = conn.execute(sql, params)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    return pd.DataFrame(rows, columns=columns)
