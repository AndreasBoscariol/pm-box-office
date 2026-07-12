"""Leakage-safe AMC data access and feature construction for production forecasts."""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
from datetime import date
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


EOD_TIME = "23:59"


def fetch_frame(conn: Any, sql: str, params: Iterable[Any] | None = None) -> pd.DataFrame:
    cursor = conn.execute(sql, params)
    return pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])


def fetch_actuals(conn: Any) -> pd.DataFrame:
    return fetch_frame(
        conn,
        """
        SELECT rr.release_run_id, rr.movie_id, m.title, m.release_date,
               dbo.box_office_date::date AS exhibition_date, dbo.day_number,
               dbo.gross_usd::double precision AS actual_gross_usd,
               dbo.theaters::double precision AS actual_theaters,
               COALESCE(dbo.is_preview, 0)::integer AS is_preview,
               dbo.source,
               dbo.fetched_at
        FROM daily_box_office dbo
        JOIN release_runs rr ON rr.release_run_id = dbo.release_run_id
        JOIN movies m ON m.movie_id = rr.movie_id
        WHERE dbo.source = 'the_numbers' AND dbo.gross_usd > 0
        """,
    )


def fetch_preview_actuals(conn: Any) -> pd.DataFrame:
    """Fetch classified preview targets; legacy daily rows are never primary."""
    try:
        exists = bool(conn.execute("SELECT to_regclass('reported_preview_actuals') IS NOT NULL").fetchone()[0])
    except Exception:
        exists = False
    if not exists:
        return pd.DataFrame()
    return fetch_frame(
        conn,
        """
            SELECT r.release_run_id, rr.movie_id, m.title,
                   r.preview_business_date AS exhibition_date,
                   r.preview_gross_usd::double precision AS actual_gross_usd,
                   r.preview_gross_usd::double precision AS preview_gross_usd,
                   r.preview_target_type, r.includes_wednesday_early_access,
                   r.preview_days_included, r.preview_source,
                   r.preview_published_at, r.received_at,
                   r.is_primary_training_target, r.is_wide_release,
                   1::integer AS is_preview
            FROM reported_preview_actuals r
            JOIN release_runs rr ON rr.release_run_id = r.release_run_id
            JOIN movies m ON m.movie_id = rr.movie_id
            LEFT JOIN reported_preview_actuals newer ON newer.supersedes_id = r.reported_preview_actual_id
            WHERE newer.reported_preview_actual_id IS NULL
        """,
    )


def fetch_schedule(conn: Any) -> pd.DataFrame:
    return fetch_frame(
        conn,
        """
        WITH active_sample AS (
            SELECT sample_set_id FROM amc_theatre_sample_sets
            WHERE status = 'active'
            ORDER BY (sample_key = 'top_hybrid_30') DESC, sample_key LIMIT 1
        ), sample_weights AS (
            SELECT member.amc_theatre_id, member.analysis_weight
            FROM active_sample JOIN amc_theatre_sample_members member
              ON member.sample_set_id = active_sample.sample_set_id
        ), showtime_capacity AS (
            SELECT showtime_id, MAX(total_seats)::double precision AS known_total_seats
            FROM amc_seat_snapshots GROUP BY showtime_id
        )
        SELECT s.movie_id, s.amc_movie_id, MAX(s.amc_movie_name) AS amc_movie_name,
               s.exhibition_date, COUNT(*)::integer AS n_scheduled_showtimes,
               COUNT(DISTINCT s.amc_theatre_id)::integer AS n_scheduled_theatres,
               COUNT(*) FILTER (WHERE s.is_premium_format)::integer AS n_premium_showtimes,
               AVG(CASE WHEN s.is_premium_format THEN 1.0 ELSE 0.0 END)::double precision AS premium_format_share,
               COUNT(DISTINCT s.timezone)::integer AS n_timezones_scheduled,
               SUM(COALESCE(sample_weights.analysis_weight, 1.0))::double precision AS weighted_scheduled_showtimes,
               SUM(COALESCE(showtime_capacity.known_total_seats, 0) * COALESCE(sample_weights.analysis_weight, 1.0))::double precision AS c_scheduled_known,
               COUNT(showtime_capacity.known_total_seats)::integer AS n_showtimes_with_known_capacity
        FROM amc_showtimes s
        LEFT JOIN sample_weights ON sample_weights.amc_theatre_id = s.amc_theatre_id
        LEFT JOIN showtime_capacity ON showtime_capacity.showtime_id = s.showtime_id
        WHERE s.exhibition_date IS NOT NULL AND s.status = 'active'
        GROUP BY s.movie_id, s.amc_movie_id, s.exhibition_date
        """,
    )


def fetch_snapshots(conn: Any) -> pd.DataFrame:
    return fetch_frame(
        conn,
        """
        WITH active_sample AS (
            SELECT sample_set_id FROM amc_theatre_sample_sets
            WHERE status = 'active'
            ORDER BY (sample_key = 'top_hybrid_30') DESC, sample_key LIMIT 1
        ), sample_weights AS (
            SELECT member.amc_theatre_id, member.analysis_weight
            FROM active_sample JOIN amc_theatre_sample_members member
              ON member.sample_set_id = active_sample.sample_set_id
        ), rescue_showtimes AS (
            SELECT DISTINCT showtime_id FROM amc_collection_diagnostic_events
            WHERE event_type = 'seat_late_rescue_scheduled' AND showtime_id IS NOT NULL
        )
        SELECT s.movie_id, s.amc_movie_id, s.exhibition_date, s.showtime_id,
               s.amc_theatre_id, s.timezone, s.local_calendar_start_at, s.starts_at_utc,
               s.business_minute, s.showtime_block, ss.seat_snapshot_id,
               ss.target_offset_minutes, ss.scheduled_for, ss.observed_at,
               ss.lateness_seconds, ss.total_seats::double precision AS total_seats,
               ss.filled_or_unavailable_seats::double precision AS filled_or_unavailable_seats,
               ss.fill_rate::double precision AS fill_rate, ss.parse_method,
               COALESCE(sample_weights.analysis_weight, 1.0)::double precision AS analysis_weight,
               (rescue_showtimes.showtime_id IS NOT NULL) AS is_late_rescue
        FROM amc_showtimes s
        JOIN amc_seat_snapshots ss ON ss.showtime_id = s.showtime_id
        LEFT JOIN sample_weights ON sample_weights.amc_theatre_id = s.amc_theatre_id
        LEFT JOIN rescue_showtimes ON rescue_showtimes.showtime_id = s.showtime_id
        WHERE s.exhibition_date IS NOT NULL
        """,
    )


def origin_timestamp_utc(exhibition_date: object, origin: str, timezone_name: str) -> pd.Timestamp:
    value = EOD_TIME if origin.upper() == "EOD" else origin
    hour, minute = value.split(":", 1)
    local_dt = dt.datetime.combine(
        pd.Timestamp(exhibition_date).date(),
        dt.time(int(hour), int(minute)),
        tzinfo=ZoneInfo(timezone_name),
    )
    return pd.Timestamp(local_dt.astimezone(dt.timezone.utc))


def build_origin_grid(schedule: pd.DataFrame, origins: Iterable[str], timezone_name: str) -> pd.DataFrame:
    base = schedule[["movie_id", "amc_movie_id", "exhibition_date"]].drop_duplicates()
    rows = [
        {
            "movie_id": row.movie_id,
            "amc_movie_id": row.amc_movie_id,
            "exhibition_date": row.exhibition_date,
            "forecast_origin": origin,
            "forecast_origin_utc": origin_timestamp_utc(row.exhibition_date, origin, timezone_name),
            "forecast_origin_tz": timezone_name,
        }
        for row in base.itertuples(index=False)
        for origin in origins
    ]
    return pd.DataFrame(rows)


def latest_snapshots_as_of(snapshots: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty or grid.empty:
        return pd.DataFrame()
    snap = snapshots.copy()
    snap["observed_at"] = pd.to_datetime(snap["observed_at"], utc=True, errors="coerce")
    snap["exhibition_date"] = pd.to_datetime(snap["exhibition_date"], errors="coerce").dt.date
    keys = ["movie_id", "amc_movie_id", "exhibition_date", "forecast_origin", "forecast_origin_utc"]
    merged = snap.merge(grid[keys], on=["movie_id", "amc_movie_id", "exhibition_date"], how="inner")
    merged = merged.loc[merged["observed_at"].le(merged["forecast_origin_utc"])].copy()
    if merged.empty:
        return merged
    merged = merged.sort_values(["forecast_origin_utc", "showtime_id", "observed_at", "seat_snapshot_id"])
    return merged.drop_duplicates(
        ["movie_id", "amc_movie_id", "exhibition_date", "forecast_origin", "showtime_id"], keep="last"
    )


def _parse_method_mix(values: pd.Series) -> str:
    return "|".join(f"{key}:{value:.3f}" for key, value in values.fillna("unknown").astype(str).value_counts(normalize=True).items())


def aggregate_as_of_features(asof_snapshots: pd.DataFrame) -> pd.DataFrame:
    keys = ["movie_id", "amc_movie_id", "exhibition_date", "forecast_origin", "forecast_origin_utc"]
    if asof_snapshots.empty:
        return pd.DataFrame(columns=keys)
    frame = asof_snapshots.copy()
    frame["weighted_filled"] = frame["filled_or_unavailable_seats"] * frame["analysis_weight"]
    frame["weighted_capacity"] = frame["total_seats"] * frame["analysis_weight"]
    frame["is_rsc_success"] = frame["parse_method"].fillna("").str.lower().str.contains("rsc")
    frame["collection_delay_minutes"] = (pd.to_datetime(frame["observed_at"], utc=True, errors="coerce") - pd.to_datetime(frame["scheduled_for"], utc=True, errors="coerce")).dt.total_seconds() / 60.0
    frame["staleness_minutes"] = (pd.to_datetime(frame["forecast_origin_utc"], utc=True, errors="coerce") - pd.to_datetime(frame["observed_at"], utc=True, errors="coerce")).dt.total_seconds() / 60.0
    frame["effective_minutes_after_show"] = (pd.to_datetime(frame["observed_at"], utc=True, errors="coerce") - pd.to_datetime(frame["starts_at_utc"], utc=True, errors="coerce")).dt.total_seconds() / 60.0
    frame["show_started"] = pd.to_datetime(frame["starts_at_utc"], utc=True, errors="coerce").le(
        pd.to_datetime(frame["forecast_origin_utc"], utc=True, errors="coerce")
    )
    frame["started_weighted_seats"] = np.where(frame["show_started"], frame["weighted_filled"], 0.0)
    frame["future_advance_weighted_seats"] = np.where(~frame["show_started"], frame["weighted_filled"], 0.0)
    for minutes in (15, 30, 60):
        frame[f"late_gt_{minutes}m"] = frame["collection_delay_minutes"].gt(minutes)
    frame["late_capacity_gt_30m"] = np.where(frame["late_gt_30m"], frame["weighted_capacity"], 0.0)
    percentile = lambda values, q: float(np.nanpercentile(values, q)) if len(values.dropna()) else np.nan
    grouped = frame.groupby(keys, dropna=False)
    out = grouped.agg(
        s_obs=("weighted_filled", "sum"), c_obs=("weighted_capacity", "sum"),
        n_snapshots=("showtime_id", "count"), n_theatres_observed=("amc_theatre_id", "nunique"),
        lateness_p50=("lateness_seconds", lambda x: percentile(x, 50)), lateness_p90=("lateness_seconds", lambda x: percentile(x, 90)),
        rsc_success_rate=("is_rsc_success", "mean"), rescue_share=("is_late_rescue", "mean"),
        mean_target_offset_minutes=("target_offset_minutes", "mean"), latest_observed_at=("observed_at", "max"),
        delay_p50_minutes=("collection_delay_minutes", lambda x: percentile(x, 50)), delay_p90_minutes=("collection_delay_minutes", lambda x: percentile(x, 90)),
        staleness_p50_minutes=("staleness_minutes", lambda x: percentile(x, 50)), staleness_p90_minutes=("staleness_minutes", lambda x: percentile(x, 90)),
        effective_minutes_after_show_p50=("effective_minutes_after_show", lambda x: percentile(x, 50)),
        effective_minutes_after_show_p90=("effective_minutes_after_show", lambda x: percentile(x, 90)),
        started_show_seats=("started_weighted_seats", "sum"),
        future_show_advance_seats=("future_advance_weighted_seats", "sum"),
        n_started_shows=("show_started", "sum"),
        first_preview_showtime_utc=("starts_at_utc", "min"),
        share_snapshots_late_gt_15m=("late_gt_15m", "mean"), share_snapshots_late_gt_30m=("late_gt_30m", "mean"),
        share_snapshots_late_gt_60m=("late_gt_60m", "mean"), late_capacity_gt_30m=("late_capacity_gt_30m", "sum"),
    ).reset_index()
    out["parse_method_mix"] = grouped["parse_method"].apply(_parse_method_mix).to_numpy()
    out["f_obs"] = out["s_obs"] / out["c_obs"].replace(0, np.nan)
    out["share_capacity_observed_late_gt_30m"] = out["late_capacity_gt_30m"] / out["c_obs"].replace(0, np.nan)
    out["fraction_scheduled_showtimes_started"] = out["n_started_shows"] / out["n_snapshots"].replace(0, np.nan)
    out["hours_relative_to_first_preview"] = (
        pd.to_datetime(out["forecast_origin_utc"], utc=True, errors="coerce")
        - pd.to_datetime(out["first_preview_showtime_utc"], utc=True, errors="coerce")
    ).dt.total_seconds() / 3600.0
    return out


def latest_snapshots_final(snapshots: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame()
    frame = snapshots.copy()
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True, errors="coerce")
    frame["exhibition_date"] = pd.to_datetime(frame["exhibition_date"], errors="coerce").dt.date
    frame = frame.sort_values(["movie_id", "amc_movie_id", "exhibition_date", "showtime_id", "observed_at", "target_offset_minutes"])
    return frame.drop_duplicates(["movie_id", "amc_movie_id", "exhibition_date", "showtime_id"], keep="last")


def aggregate_final_eod_features(final_snapshots: pd.DataFrame) -> pd.DataFrame:
    keys = ["movie_id", "amc_movie_id", "exhibition_date"]
    if final_snapshots.empty:
        return pd.DataFrame(columns=keys)
    frame = final_snapshots.copy()
    frame["weighted_filled"] = frame["filled_or_unavailable_seats"] * frame["analysis_weight"]
    frame["weighted_capacity"] = frame["total_seats"] * frame["analysis_weight"]
    out = frame.groupby(keys, dropna=False).agg(
        s_final_eod=("weighted_filled", "sum"), c_final_eod=("weighted_capacity", "sum"),
        n_final_snapshots=("showtime_id", "count"), n_final_theatres=("amc_theatre_id", "nunique"),
        latest_final_observed_at=("observed_at", "max"),
    ).reset_index()
    out["f_final_eod"] = out["s_final_eod"] / out["c_final_eod"].replace(0, np.nan)
    return out


@dataclass(frozen=True)
class AMCFeatureBuilder:
    conn: Any
    origin_timezone: str = "America/New_York"

    def build_asof_features(
        self,
        *,
        movie_id: int,
        exhibition_date: date,
        forecast_origin: str,
        as_of_utc: pd.Timestamp,
    ) -> pd.DataFrame:
        """Build one movie-day-origin AMC feature row using snapshots observed by ``as_of_utc``."""

        schedule = fetch_schedule(self.conn)
        snapshots = fetch_snapshots(self.conn)
        if schedule.empty:
            return pd.DataFrame()
        schedule["exhibition_date"] = pd.to_datetime(schedule["exhibition_date"], errors="coerce").dt.date
        snapshots["exhibition_date"] = pd.to_datetime(snapshots["exhibition_date"], errors="coerce").dt.date

        schedule = schedule.loc[
            schedule["movie_id"].eq(movie_id) & schedule["exhibition_date"].eq(exhibition_date)
        ].copy()
        snapshots = snapshots.loc[
            snapshots["movie_id"].eq(movie_id)
            & snapshots["exhibition_date"].eq(exhibition_date)
            & (pd.to_datetime(snapshots["observed_at"], utc=True, errors="coerce") <= pd.Timestamp(as_of_utc).tz_convert("UTC"))
        ].copy()
        if schedule.empty:
            return pd.DataFrame()

        grid = build_origin_grid(schedule, [forecast_origin], self.origin_timezone)
        grid["forecast_origin_utc"] = pd.Timestamp(as_of_utc).tz_convert("UTC")
        asof = latest_snapshots_as_of(snapshots, grid)
        return aggregate_as_of_features(asof)
