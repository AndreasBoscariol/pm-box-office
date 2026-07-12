#!/usr/bin/env python3
"""Leakage-safe same-day AMC seat nowcast EDA.

This script treats AMC seat fills as external same-day predictors for the
national actual daily gross.  It builds one row per movie-day-forecast-origin,
using only seat snapshots whose ``observed_at`` timestamp was available by that
origin, then compares:

M0: no-seat baseline
M1: baseline + pre/as-scheduled capacity features
M2: baseline + schedule features + as-of seat fill

It also keeps a two-stage AMC layer:

P2: train-only pace-adjusted nowcast of final same-day AMC EOD seats
B3: bridge from predicted AMC EOD seats to national actual daily gross, using
    rolling same-movie residuals when prior actual days are available

Outputs are written under ``data/diagnostics`` and ``data/plots``.
"""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from pm_box_office.db.connection import connect_database
from pm_box_office.sources.common.cli import add_database_arg


REPO_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTICS_DIR = REPO_ROOT / "data" / "diagnostics"
PLOTS_DIR = REPO_ROOT / "data" / "plots"

DEFAULT_ORIGINS = ("10:00", "12:00", "14:00", "16:00", "18:00", "20:00", "EOD")
DEFAULT_ORIGIN_TZ = "America/New_York"
EOD_TIME = "23:59"
MIN_ROLLING_TRAIN_N = 20
RIDGE_LAMBDA = 1e-6


@dataclass(frozen=True)
class RollingConfig:
    min_train_n: int = MIN_ROLLING_TRAIN_N
    ridge_lambda: float = RIDGE_LAMBDA


def relation_exists(conn: Any, relation_name: str) -> bool:
    row = conn.execute("SELECT to_regclass(%s)", (relation_name,)).fetchone()
    return bool(row and row[0])


def fetch_frame(conn: Any, sql: str, params: Iterable[Any] | None = None) -> pd.DataFrame:
    cursor = conn.execute(sql, params)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def parse_origin_time(origin: str) -> dt.time:
    value = EOD_TIME if origin.upper() == "EOD" else origin
    hour, minute = value.split(":", 1)
    return dt.time(int(hour), int(minute))


def origin_timestamp_utc(exhibition_date: pd.Timestamp, origin: str, timezone_name: str) -> pd.Timestamp:
    local_zone = ZoneInfo(timezone_name)
    origin_date = pd.Timestamp(exhibition_date).date()
    local_dt = dt.datetime.combine(origin_date, parse_origin_time(origin), tzinfo=local_zone)
    return pd.Timestamp(local_dt.astimezone(dt.timezone.utc))


def safe_log_ratio(num: pd.Series | np.ndarray, den: pd.Series | np.ndarray) -> pd.Series:
    num_s = pd.to_numeric(pd.Series(num), errors="coerce")
    den_s = pd.to_numeric(pd.Series(den), errors="coerce")
    out = pd.Series(np.nan, index=num_s.index, dtype="float64")
    mask = (num_s > 0) & (den_s > 0)
    out.loc[mask] = np.log(num_s.loc[mask] / den_s.loc[mask])
    return out


def pct_improvement(baseline: float, candidate: float) -> float:
    if not np.isfinite(baseline) or baseline == 0 or not np.isfinite(candidate):
        return np.nan
    return float((baseline - candidate) / baseline)


def nonempty_median(values: pd.Series, default: float = np.nan) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.median()) if len(clean) else default


def interval_score_log(error: pd.Series, lower_q: float, upper_q: float, alpha: float) -> pd.Series:
    err = pd.to_numeric(error, errors="coerce")
    width = upper_q - lower_q
    below = (lower_q - err).clip(lower=0)
    above = (err - upper_q).clip(lower=0)
    return width + (2.0 / alpha) * below + (2.0 / alpha) * above


def read_baseline_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rename_map = {
        "actual_daily_gross": "actual_gross_usd",
        "actual_gross": "actual_gross_usd",
        "baseline_daily_gross": "baseline_gross_usd",
        "baseline_forecast_usd": "baseline_gross_usd",
        "no_seat_baseline_usd": "baseline_gross_usd",
    }
    frame = frame.rename(columns={k: v for k, v in rename_map.items() if k in frame.columns})
    required = {"exhibition_date", "baseline_gross_usd"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required baseline columns: {sorted(missing)}")
    frame["exhibition_date"] = pd.to_datetime(frame["exhibition_date"], errors="coerce").dt.date
    frame["baseline_gross_usd"] = pd.to_numeric(frame["baseline_gross_usd"], errors="coerce")
    return frame


def fetch_baseline_estimates(conn: Any) -> pd.DataFrame:
    if not relation_exists(conn, "movie_day_estimates"):
        return pd.DataFrame(columns=["movie_id", "exhibition_date", "baseline_gross_usd", "baseline_source"])
    return fetch_frame(
        conn,
        """
        SELECT DISTINCT ON (movie_id, exhibition_date)
            movie_id,
            exhibition_date,
            estimate_usd::double precision AS baseline_gross_usd,
            source AS baseline_source,
            COALESCE(published_at, recorded_at) AS baseline_known_at
        FROM movie_day_estimates
        WHERE estimate_usd > 0
          AND (
                is_baseline
             OR source IN ('baseline', 'no_seat_baseline', 'daily_no_seat_baseline')
          )
        ORDER BY movie_id, exhibition_date, COALESCE(published_at, recorded_at) DESC
        """,
    )


def fetch_actuals(conn: Any) -> pd.DataFrame:
    return fetch_frame(
        conn,
        """
        SELECT
            rr.release_run_id,
            rr.movie_id,
            m.title,
            m.release_date,
            dbo.box_office_date::date AS exhibition_date,
            dbo.day_number,
            dbo.gross_usd::double precision AS actual_gross_usd,
            dbo.theaters::double precision AS actual_theaters
        FROM daily_box_office dbo
        JOIN release_runs rr ON rr.release_run_id = dbo.release_run_id
        JOIN movies m ON m.movie_id = rr.movie_id
        WHERE dbo.source = 'the_numbers'
          AND dbo.gross_usd > 0
        """,
    )


def fetch_schedule(conn: Any) -> pd.DataFrame:
    return fetch_frame(
        conn,
        """
        WITH active_sample AS (
            SELECT sample_set_id
            FROM amc_theatre_sample_sets
            WHERE status = 'active'
            ORDER BY (sample_key = 'top_hybrid_30') DESC, sample_key
            LIMIT 1
        ),
        sample_weights AS (
            SELECT member.amc_theatre_id, member.analysis_weight
            FROM active_sample
            JOIN amc_theatre_sample_members member
              ON member.sample_set_id = active_sample.sample_set_id
        ),
        showtime_capacity AS (
            SELECT showtime_id, MAX(total_seats)::double precision AS known_total_seats
            FROM amc_seat_snapshots
            GROUP BY showtime_id
        )
        SELECT
            s.movie_id,
            s.amc_movie_id,
            MAX(s.amc_movie_name) AS amc_movie_name,
            s.exhibition_date,
            COUNT(*)::integer AS n_scheduled_showtimes,
            COUNT(DISTINCT s.amc_theatre_id)::integer AS n_scheduled_theatres,
            COUNT(*) FILTER (WHERE s.is_premium_format)::integer AS n_premium_showtimes,
            AVG(CASE WHEN s.is_premium_format THEN 1.0 ELSE 0.0 END)::double precision AS premium_format_share,
            COUNT(DISTINCT s.timezone)::integer AS n_timezones_scheduled,
            SUM(COALESCE(sample_weights.analysis_weight, 1.0))::double precision AS weighted_scheduled_showtimes,
            SUM(COALESCE(showtime_capacity.known_total_seats, 0) * COALESCE(sample_weights.analysis_weight, 1.0))::double precision
                AS c_scheduled_known,
            COUNT(showtime_capacity.known_total_seats)::integer AS n_showtimes_with_known_capacity
        FROM amc_showtimes s
        LEFT JOIN sample_weights ON sample_weights.amc_theatre_id = s.amc_theatre_id
        LEFT JOIN showtime_capacity ON showtime_capacity.showtime_id = s.showtime_id
        WHERE s.exhibition_date IS NOT NULL
          AND s.status = 'active'
        GROUP BY s.movie_id, s.amc_movie_id, s.exhibition_date
        """,
    )


def fetch_snapshots(conn: Any) -> pd.DataFrame:
    return fetch_frame(
        conn,
        """
        WITH active_sample AS (
            SELECT sample_set_id
            FROM amc_theatre_sample_sets
            WHERE status = 'active'
            ORDER BY (sample_key = 'top_hybrid_30') DESC, sample_key
            LIMIT 1
        ),
        sample_weights AS (
            SELECT member.amc_theatre_id, member.analysis_weight
            FROM active_sample
            JOIN amc_theatre_sample_members member
              ON member.sample_set_id = active_sample.sample_set_id
        ),
        rescue_showtimes AS (
            SELECT DISTINCT showtime_id
            FROM amc_collection_diagnostic_events
            WHERE event_type = 'seat_late_rescue_scheduled'
              AND showtime_id IS NOT NULL
        )
        SELECT
            s.movie_id,
            s.amc_movie_id,
            s.exhibition_date,
            s.showtime_id,
            s.amc_theatre_id,
            s.timezone,
            s.local_calendar_start_at,
            s.starts_at_utc,
            s.business_minute,
            s.showtime_block,
            ss.seat_snapshot_id,
            ss.target_offset_minutes,
            ss.scheduled_for,
            ss.observed_at,
            ss.lateness_seconds,
            ss.total_seats::double precision AS total_seats,
            ss.filled_or_unavailable_seats::double precision AS filled_or_unavailable_seats,
            ss.fill_rate::double precision AS fill_rate,
            ss.parse_method,
            COALESCE(sample_weights.analysis_weight, 1.0)::double precision AS analysis_weight,
            (rescue_showtimes.showtime_id IS NOT NULL) AS is_late_rescue
        FROM amc_showtimes s
        JOIN amc_seat_snapshots ss ON ss.showtime_id = s.showtime_id
        LEFT JOIN sample_weights ON sample_weights.amc_theatre_id = s.amc_theatre_id
        LEFT JOIN rescue_showtimes ON rescue_showtimes.showtime_id = s.showtime_id
        WHERE s.exhibition_date IS NOT NULL
        """,
    )


def build_origin_grid(schedule: pd.DataFrame, origins: Iterable[str], timezone_name: str) -> pd.DataFrame:
    base = schedule[["movie_id", "amc_movie_id", "exhibition_date"]].drop_duplicates().copy()
    rows = []
    for _, row in base.iterrows():
        for origin in origins:
            rows.append(
                {
                    "movie_id": row["movie_id"],
                    "amc_movie_id": row["amc_movie_id"],
                    "exhibition_date": row["exhibition_date"],
                    "forecast_origin": origin,
                    "forecast_origin_utc": origin_timestamp_utc(row["exhibition_date"], origin, timezone_name),
                    "forecast_origin_tz": timezone_name,
                }
            )
    return pd.DataFrame(rows)


def latest_snapshots_available(
    snapshots: pd.DataFrame,
    grid: pd.DataFrame,
    *,
    availability_column: str,
) -> pd.DataFrame:
    if snapshots.empty or grid.empty:
        return pd.DataFrame()
    snap = snapshots.copy()
    snap["observed_at"] = pd.to_datetime(snap["observed_at"], utc=True, errors="coerce")
    snap["scheduled_for"] = pd.to_datetime(snap["scheduled_for"], utc=True, errors="coerce")
    snap["exhibition_date"] = pd.to_datetime(snap["exhibition_date"], errors="coerce").dt.date
    origin_keys = ["movie_id", "amc_movie_id", "exhibition_date", "forecast_origin", "forecast_origin_utc"]
    merged = snap.merge(grid[origin_keys], on=["movie_id", "amc_movie_id", "exhibition_date"], how="inner")
    merged = merged.loc[merged[availability_column].le(merged["forecast_origin_utc"])].copy()
    if merged.empty:
        return merged
    merged = merged.sort_values(["forecast_origin_utc", "showtime_id", availability_column, "observed_at", "seat_snapshot_id"])
    return merged.drop_duplicates(["movie_id", "amc_movie_id", "exhibition_date", "forecast_origin", "showtime_id"], keep="last")


def latest_snapshots_as_of(snapshots: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    return latest_snapshots_available(snapshots, grid, availability_column="observed_at")


def parse_method_mix(values: pd.Series) -> str:
    counts = values.fillna("unknown").astype(str).value_counts(normalize=True)
    return "|".join(f"{key}:{value:.3f}" for key, value in counts.items())


def aggregate_as_of_features(asof_snapshots: pd.DataFrame) -> pd.DataFrame:
    keys = ["movie_id", "amc_movie_id", "exhibition_date", "forecast_origin", "forecast_origin_utc"]
    if asof_snapshots.empty:
        return pd.DataFrame(columns=keys)
    frame = asof_snapshots.copy()
    frame["weighted_filled"] = frame["filled_or_unavailable_seats"] * frame["analysis_weight"]
    frame["weighted_capacity"] = frame["total_seats"] * frame["analysis_weight"]
    frame["is_rsc_success"] = frame["parse_method"].fillna("").str.lower().str.contains("rsc")
    frame["collection_delay_minutes"] = (
        pd.to_datetime(frame["observed_at"], utc=True, errors="coerce")
        - pd.to_datetime(frame["scheduled_for"], utc=True, errors="coerce")
    ).dt.total_seconds() / 60.0
    frame["staleness_minutes"] = (
        pd.to_datetime(frame["forecast_origin_utc"], utc=True, errors="coerce")
        - pd.to_datetime(frame["observed_at"], utc=True, errors="coerce")
    ).dt.total_seconds() / 60.0
    frame["effective_minutes_after_show"] = (
        pd.to_datetime(frame["observed_at"], utc=True, errors="coerce")
        - pd.to_datetime(frame["starts_at_utc"], utc=True, errors="coerce")
    ).dt.total_seconds() / 60.0
    frame["late_gt_15m"] = frame["collection_delay_minutes"].gt(15)
    frame["late_gt_30m"] = frame["collection_delay_minutes"].gt(30)
    frame["late_gt_60m"] = frame["collection_delay_minutes"].gt(60)
    frame["late_capacity_gt_30m"] = np.where(frame["late_gt_30m"], frame["weighted_capacity"], 0.0)
    grouped = frame.groupby(keys, dropna=False)
    out = grouped.agg(
        s_obs=("weighted_filled", "sum"),
        c_obs=("weighted_capacity", "sum"),
        n_snapshots=("showtime_id", "count"),
        n_theatres_observed=("amc_theatre_id", "nunique"),
        lateness_p50=("lateness_seconds", lambda x: float(np.nanpercentile(x, 50)) if len(x.dropna()) else np.nan),
        lateness_p90=("lateness_seconds", lambda x: float(np.nanpercentile(x, 90)) if len(x.dropna()) else np.nan),
        rsc_success_rate=("is_rsc_success", "mean"),
        rescue_share=("is_late_rescue", "mean"),
        mean_target_offset_minutes=("target_offset_minutes", "mean"),
        latest_observed_at=("observed_at", "max"),
        delay_p50_minutes=("collection_delay_minutes", lambda x: float(np.nanpercentile(x, 50)) if len(x.dropna()) else np.nan),
        delay_p90_minutes=("collection_delay_minutes", lambda x: float(np.nanpercentile(x, 90)) if len(x.dropna()) else np.nan),
        staleness_p50_minutes=("staleness_minutes", lambda x: float(np.nanpercentile(x, 50)) if len(x.dropna()) else np.nan),
        staleness_p90_minutes=("staleness_minutes", lambda x: float(np.nanpercentile(x, 90)) if len(x.dropna()) else np.nan),
        effective_minutes_after_show_p50=(
            "effective_minutes_after_show",
            lambda x: float(np.nanpercentile(x, 50)) if len(x.dropna()) else np.nan,
        ),
        effective_minutes_after_show_p90=(
            "effective_minutes_after_show",
            lambda x: float(np.nanpercentile(x, 90)) if len(x.dropna()) else np.nan,
        ),
        share_snapshots_late_gt_15m=("late_gt_15m", "mean"),
        share_snapshots_late_gt_30m=("late_gt_30m", "mean"),
        share_snapshots_late_gt_60m=("late_gt_60m", "mean"),
        late_capacity_gt_30m=("late_capacity_gt_30m", "sum"),
    ).reset_index()
    out["parse_method_mix"] = grouped["parse_method"].apply(parse_method_mix).to_numpy()
    out["f_obs"] = out["s_obs"] / out["c_obs"].replace(0, np.nan)
    out["share_capacity_observed_late_gt_30m"] = out["late_capacity_gt_30m"] / out["c_obs"].replace(0, np.nan)
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
    grouped = frame.groupby(keys, dropna=False)
    out = grouped.agg(
        s_final_eod=("weighted_filled", "sum"),
        c_final_eod=("weighted_capacity", "sum"),
        n_final_snapshots=("showtime_id", "count"),
        n_final_theatres=("amc_theatre_id", "nunique"),
        latest_final_observed_at=("observed_at", "max"),
    ).reset_index()
    out["f_final_eod"] = out["s_final_eod"] / out["c_final_eod"].replace(0, np.nan)
    return out


def summarize_snapshot_quality(snapshots: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame()
    frame = snapshots.copy()
    frame["scheduled_for"] = pd.to_datetime(frame["scheduled_for"], utc=True, errors="coerce")
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True, errors="coerce")
    frame["exhibition_date"] = pd.to_datetime(frame["exhibition_date"], errors="coerce").dt.date
    frame["day_of_week"] = pd.to_datetime(frame["exhibition_date"], errors="coerce").dt.day_name()
    frame["scheduled_hour_utc"] = frame["scheduled_for"].dt.hour
    frame["observed_minus_scheduled_minutes"] = (frame["observed_at"] - frame["scheduled_for"]).dt.total_seconds() / 60.0
    frame["is_rsc_success"] = frame["parse_method"].fillna("").str.lower().str.contains("rsc")
    keys = ["exhibition_date", "day_of_week", "timezone", "parse_method", "scheduled_hour_utc"]
    return (
        frame.groupby(keys, dropna=False)
        .agg(
            n_snapshots=("showtime_id", "count"),
            n_movies=("amc_movie_id", "nunique"),
            n_theatres=("amc_theatre_id", "nunique"),
            lateness_minutes_p50=(
                "observed_minus_scheduled_minutes",
                lambda x: float(np.nanpercentile(x, 50)) if len(x.dropna()) else np.nan,
            ),
            lateness_minutes_p90=(
                "observed_minus_scheduled_minutes",
                lambda x: float(np.nanpercentile(x, 90)) if len(x.dropna()) else np.nan,
            ),
            rsc_success_rate=("is_rsc_success", "mean"),
            rescue_share=("is_late_rescue", "mean"),
        )
        .reset_index()
    )


def build_panel(
    conn: Any,
    *,
    origins: Iterable[str],
    origin_timezone: str,
    baseline_csv: Path | None = None,
) -> pd.DataFrame:
    schedule = fetch_schedule(conn)
    if schedule.empty:
        raise RuntimeError("No AMC showtimes found; run AMC collection before this EDA.")
    actuals = fetch_actuals(conn)
    baseline = read_baseline_csv(baseline_csv) if baseline_csv else fetch_baseline_estimates(conn)
    snapshots = fetch_snapshots(conn)

    for frame in [schedule, actuals, baseline, snapshots]:
        if "exhibition_date" in frame.columns:
            frame["exhibition_date"] = pd.to_datetime(frame["exhibition_date"], errors="coerce").dt.date

    grid = build_origin_grid(schedule, origins, origin_timezone)
    asof = latest_snapshots_as_of(snapshots, grid)
    features = aggregate_as_of_features(asof)
    oracle_features = aggregate_as_of_features(
        latest_snapshots_available(snapshots, grid, availability_column="scheduled_for")
    )
    final_features = aggregate_final_eod_features(latest_snapshots_final(snapshots))

    panel = grid.merge(schedule, on=["movie_id", "amc_movie_id", "exhibition_date"], how="left")
    panel = panel.merge(features, on=["movie_id", "amc_movie_id", "exhibition_date", "forecast_origin", "forecast_origin_utc"], how="left")
    panel = panel.merge(final_features, on=["movie_id", "amc_movie_id", "exhibition_date"], how="left")
    panel = panel.merge(actuals, on=["movie_id", "exhibition_date"], how="left")
    panel = merge_baseline(panel, baseline)
    panel = finalize_panel_columns(panel)
    oracle_panel = replace_asof_features(panel, oracle_features)
    oracle_panel = finalize_panel_columns(oracle_panel)
    panel.attrs["snapshot_quality"] = summarize_snapshot_quality(snapshots)
    panel.attrs["oracle_panel"] = oracle_panel
    return panel


def merge_baseline(panel: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    if baseline.empty:
        panel = panel.copy()
        panel["baseline_gross_usd"] = np.nan
        panel["baseline_source"] = pd.NA
        return panel
    baseline = baseline.copy()
    keys: list[str]
    if {"movie_id", "exhibition_date"}.issubset(baseline.columns):
        keys = ["movie_id", "exhibition_date"]
    elif {"amc_movie_id", "exhibition_date"}.issubset(baseline.columns):
        keys = ["amc_movie_id", "exhibition_date"]
    else:
        raise ValueError("Baseline data must include either movie_id or amc_movie_id plus exhibition_date.")
    keep = keys + [c for c in ["baseline_gross_usd", "baseline_source", "baseline_known_at"] if c in baseline.columns]
    return panel.merge(baseline[keep].drop_duplicates(keys), on=keys, how="left")


ASOF_FEATURE_COLUMNS = [
    "s_obs",
    "c_obs",
    "n_snapshots",
    "n_theatres_observed",
    "lateness_p50",
    "lateness_p90",
    "rsc_success_rate",
    "rescue_share",
    "mean_target_offset_minutes",
    "latest_observed_at",
    "delay_p50_minutes",
    "delay_p90_minutes",
    "staleness_p50_minutes",
    "staleness_p90_minutes",
    "effective_minutes_after_show_p50",
    "effective_minutes_after_show_p90",
    "share_snapshots_late_gt_15m",
    "share_snapshots_late_gt_30m",
    "share_snapshots_late_gt_60m",
    "late_capacity_gt_30m",
    "share_capacity_observed_late_gt_30m",
    "parse_method_mix",
    "f_obs",
]


def replace_asof_features(panel: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    keys = ["movie_id", "amc_movie_id", "exhibition_date", "forecast_origin", "forecast_origin_utc"]
    out = panel.drop(columns=[column for column in ASOF_FEATURE_COLUMNS if column in panel.columns]).copy()
    out.attrs = {}
    features = features.copy()
    features.attrs = {}
    return out.merge(features, on=keys, how="left")


def finalize_panel_columns(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel.attrs = {}
    numeric_defaults = {
        "s_obs": 0.0,
        "c_obs": 0.0,
        "n_snapshots": 0,
        "n_theatres_observed": 0,
        "s_final_eod": 0.0,
        "c_final_eod": 0.0,
        "n_final_snapshots": 0,
        "n_final_theatres": 0,
        "rsc_success_rate": 0.0,
        "rescue_share": 0.0,
        "share_snapshots_late_gt_15m": 0.0,
        "share_snapshots_late_gt_30m": 0.0,
        "share_snapshots_late_gt_60m": 0.0,
        "share_capacity_observed_late_gt_30m": 0.0,
    }
    for column, default in numeric_defaults.items():
        if column in panel.columns:
            panel[column] = pd.to_numeric(panel[column], errors="coerce").fillna(default)
    panel["coverage"] = panel["c_obs"] / panel["c_scheduled_known"].replace(0, np.nan)
    actual_gross = pd.to_numeric(panel["actual_gross_usd"], errors="coerce")
    baseline_gross = pd.to_numeric(panel["baseline_gross_usd"], errors="coerce")
    panel["log_actual_gross"] = np.log(actual_gross.where(actual_gross > 0))
    panel["log_baseline_gross"] = np.log(baseline_gross.where(baseline_gross > 0))
    panel["baseline_residual_log"] = panel["log_actual_gross"] - panel["log_baseline_gross"]
    panel["day_of_week"] = pd.to_datetime(panel["exhibition_date"], errors="coerce").dt.day_name()
    panel["is_opening_day"] = pd.to_numeric(panel["day_number"], errors="coerce").eq(1)
    panel["is_weekend"] = panel["day_of_week"].isin(["Friday", "Saturday", "Sunday"])
    panel["run_day"] = pd.to_numeric(panel["day_number"], errors="coerce")
    panel["log1p_s_obs"] = np.log1p(panel["s_obs"])
    panel["log1p_c_obs"] = np.log1p(panel["c_obs"])
    panel["log1p_s_final_eod"] = np.log1p(panel["s_final_eod"])
    panel["log1p_c_final_eod"] = np.log1p(panel["c_final_eod"])
    panel["log1p_c_scheduled"] = np.log1p(panel["c_scheduled_known"])
    panel["log1p_n_scheduled_showtimes"] = np.log1p(panel["n_scheduled_showtimes"])
    coverage = pd.to_numeric(panel["coverage"], errors="coerce")
    delay = pd.to_numeric(panel["delay_p90_minutes"], errors="coerce")
    rsc = pd.to_numeric(panel["rsc_success_rate"], errors="coerce")
    panel["collection_quality_bucket"] = np.select(
        [
            coverage.ge(0.80) & delay.le(15) & rsc.ge(0.80),
            coverage.ge(0.50) & delay.le(45),
        ],
        ["high", "medium"],
        default="low",
    )
    return panel.replace([np.inf, -np.inf], np.nan).sort_values(
        ["exhibition_date", "movie_id", "forecast_origin"]
    ).reset_index(drop=True)


def feature_matrix(frame: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]) -> pd.DataFrame:
    parts = [pd.Series(1.0, index=frame.index, name="intercept")]
    for column in numeric_features:
        values = pd.to_numeric(frame[column], errors="coerce")
        parts.append(values.fillna(values.median()).fillna(0.0).rename(column))
    for column in categorical_features:
        dummies = pd.get_dummies(frame[column].fillna("missing").astype(str), prefix=column, dtype=float)
        if len(dummies.columns) > 1:
            dummies = dummies.iloc[:, 1:]
        parts.append(dummies)
    return pd.concat(parts, axis=1)


def design_matrix(frame: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    x = feature_matrix(frame, numeric_features, categorical_features)
    y = pd.to_numeric(frame["baseline_residual_log"], errors="coerce")
    return x, y


def design_matrix_for_target(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    x = feature_matrix(frame, numeric_features, categorical_features)
    y = pd.to_numeric(frame[target_column], errors="coerce")
    return x, y


def fit_ridge_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    ridge_lambda: float,
) -> tuple[float, dict[str, float]]:
    combined = pd.concat([train, test], axis=0, ignore_index=False)
    x_all, y_all = design_matrix(combined, numeric_features, categorical_features)
    x_train = x_all.loc[train.index].to_numpy(dtype=float)
    y_train = y_all.loc[train.index].to_numpy(dtype=float)
    x_test = x_all.loc[test.index].to_numpy(dtype=float)
    mask = np.isfinite(y_train)
    x_train = x_train[mask]
    y_train = y_train[mask]
    if len(y_train) == 0:
        return 0.0, {}
    penalty = np.eye(x_train.shape[1]) * ridge_lambda
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(x_train.T @ x_train + penalty) @ x_train.T @ y_train
    prediction = float((x_test @ beta)[0])
    coefs = {name: float(value) for name, value in zip(x_all.columns, beta)}
    return prediction, coefs


def fit_ridge_predict_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    target_column: str,
    *,
    ridge_lambda: float,
) -> tuple[float, dict[str, float]]:
    combined = pd.concat([train, test], axis=0, ignore_index=False)
    x_all, y_all = design_matrix_for_target(combined, numeric_features, categorical_features, target_column)
    x_train = x_all.loc[train.index].to_numpy(dtype=float)
    y_train = y_all.loc[train.index].to_numpy(dtype=float)
    x_test = x_all.loc[test.index].to_numpy(dtype=float)
    mask = np.isfinite(y_train)
    x_train = x_train[mask]
    y_train = y_train[mask]
    if len(y_train) == 0:
        return float(np.nan), {}
    penalty = np.eye(x_train.shape[1]) * ridge_lambda
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(x_train.T @ x_train + penalty) @ x_train.T @ y_train
    prediction = float((x_test @ beta)[0])
    coefs = {name: float(value) for name, value in zip(x_all.columns, beta)}
    return prediction, coefs


def rolling_origin_predictions(panel: pd.DataFrame, config: RollingConfig = RollingConfig()) -> pd.DataFrame:
    usable = panel.dropna(subset=["actual_gross_usd", "baseline_gross_usd", "baseline_residual_log"]).copy()
    usable = usable.loc[(usable["actual_gross_usd"] > 0) & (usable["baseline_gross_usd"] > 0)].copy()
    if usable.empty:
        return pd.DataFrame()

    schedule_numeric = [
        "log1p_c_scheduled",
        "log1p_n_scheduled_showtimes",
        "n_scheduled_theatres",
        "premium_format_share",
        "run_day",
        "is_opening_day",
        "is_weekend",
    ]
    seat_numeric = schedule_numeric + [
        "log1p_s_obs",
        "f_obs",
        "coverage",
        "n_snapshots",
        "lateness_p50",
        "rsc_success_rate",
        "rescue_share",
    ]
    categoricals = ["day_of_week"]

    rows = []
    coefficient_rows = []
    for origin, origin_frame in usable.groupby("forecast_origin", sort=False):
        origin_frame = origin_frame.sort_values(["exhibition_date", "movie_id"]).copy()
        for idx, row in origin_frame.iterrows():
            train = origin_frame.loc[origin_frame["exhibition_date"] < row["exhibition_date"]].copy()
            if len(train) < config.min_train_n:
                residual_b = 0.0
                residual_c = 0.0
                coefs_b: dict[str, float] = {}
                coefs_c: dict[str, float] = {}
            else:
                test = origin_frame.loc[[idx]].copy()
                residual_b, coefs_b = fit_ridge_predict(
                    train,
                    test,
                    schedule_numeric,
                    categoricals,
                    ridge_lambda=config.ridge_lambda,
                )
                residual_c, coefs_c = fit_ridge_predict(
                    train,
                    test,
                    seat_numeric,
                    categoricals,
                    ridge_lambda=config.ridge_lambda,
                )
            model_preds = {
                "M0_baseline": row["baseline_gross_usd"],
                "M1_baseline_plus_schedule": row["baseline_gross_usd"] * np.exp(residual_b),
                "M2_baseline_plus_schedule_plus_seats": row["baseline_gross_usd"] * np.exp(residual_c),
            }
            for model, pred in model_preds.items():
                rows.append(
                    {
                        "model": model,
                        "forecast_origin": origin,
                        "movie_id": row["movie_id"],
                        "amc_movie_id": row["amc_movie_id"],
                        "title": row.get("title", row.get("amc_movie_name")),
                        "exhibition_date": row["exhibition_date"],
                        "train_n": int(len(train)),
                        "actual_gross_usd": row["actual_gross_usd"],
                        "baseline_gross_usd": row["baseline_gross_usd"],
                        "pred_gross_usd": pred,
                        "error_usd": row["actual_gross_usd"] - pred,
                        "abs_error_usd": abs(row["actual_gross_usd"] - pred),
                        "log_error": np.log(row["actual_gross_usd"] / pred) if pred > 0 else np.nan,
                        "abs_pct_error": abs(row["actual_gross_usd"] - pred) / row["actual_gross_usd"],
                        "coverage": row["coverage"],
                        "s_obs": row["s_obs"],
                        "f_obs": row["f_obs"],
                        "c_scheduled_known": row["c_scheduled_known"],
                    }
                )
            for model, coefs in [("M1_baseline_plus_schedule", coefs_b), ("M2_baseline_plus_schedule_plus_seats", coefs_c)]:
                coefficient_rows.append(
                    {
                        "forecast_origin": origin,
                        "movie_id": row["movie_id"],
                        "exhibition_date": row["exhibition_date"],
                        "model": model,
                        "train_n": int(len(train)),
                        "coef_log1p_s_obs": coefs.get("log1p_s_obs", np.nan),
                        "coef_f_obs": coefs.get("f_obs", np.nan),
                        "coef_coverage": coefs.get("coverage", np.nan),
                    }
                )
    predictions = pd.DataFrame(rows)
    coefs = pd.DataFrame(coefficient_rows)
    predictions.attrs["coefficients"] = coefs
    return predictions.replace([np.inf, -np.inf], np.nan)


def build_amc_pseudo_nowcast_panel(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel.attrs = {}
    out = panel.loc[~panel["forecast_origin"].eq("EOD")].copy()
    out["s_eod"] = pd.to_numeric(out["s_final_eod"], errors="coerce")
    out["z_eod"] = np.log1p(out["s_eod"])
    out["raw_pace"] = pd.to_numeric(out["s_obs"], errors="coerce") / out["s_eod"].replace(0, np.nan)
    out = out.loc[out["s_eod"].gt(0)].copy()
    return out.replace([np.inf, -np.inf], np.nan)


def rolling_pace_prior(train: pd.DataFrame, row: pd.Series) -> tuple[float, str, int]:
    same_day = train.loc[train["day_of_week"].eq(row["day_of_week"]), "raw_pace"]
    pace = nonempty_median(same_day)
    source = "day_origin"
    n = int(pd.to_numeric(same_day, errors="coerce").dropna().shape[0])
    if not np.isfinite(pace) or pace <= 0:
        pace = nonempty_median(train["raw_pace"])
        source = "origin_pooled"
        n = int(pd.to_numeric(train["raw_pace"], errors="coerce").dropna().shape[0])
    if not np.isfinite(pace) or pace <= 0:
        pace = 1.0
        source = "default"
        n = 0
    return float(np.clip(pace, 0.01, 1.0)), source, n


def add_pace_prior_features(frame: pd.DataFrame, pace: float) -> pd.DataFrame:
    out = frame.copy()
    out["pace_prior"] = float(pace)
    out["z_pace_prior"] = np.log1p(pd.to_numeric(out["s_obs"], errors="coerce")) - np.log(float(pace))
    out["pace_adjusted_s_obs"] = pd.to_numeric(out["s_obs"], errors="coerce") / float(pace)
    out["log1p_pace_adjusted_s_obs"] = np.log1p(out["pace_adjusted_s_obs"])
    return out.replace([np.inf, -np.inf], np.nan)


def rolling_amc_pseudo_nowcast_predictions(
    panel: pd.DataFrame,
    config: RollingConfig = RollingConfig(),
) -> pd.DataFrame:
    usable = build_amc_pseudo_nowcast_panel(panel)
    usable = usable.dropna(subset=["z_eod", "log1p_c_scheduled"]).copy()
    if usable.empty:
        return pd.DataFrame()

    schedule_numeric = ["log1p_c_scheduled"]
    seat_numeric = ["log1p_s_obs", "log1p_c_scheduled", "coverage"]
    pace_numeric = [
        "z_pace_prior",
        "log1p_s_obs",
        "log1p_c_scheduled",
        "coverage",
        "delay_p50_minutes",
        "delay_p90_minutes",
        "staleness_p50_minutes",
        "staleness_p90_minutes",
        "n_snapshots",
        "n_theatres_observed",
        "rsc_success_rate",
        "rescue_share",
        "share_snapshots_late_gt_30m",
        "share_capacity_observed_late_gt_30m",
    ]
    categoricals: list[str] = ["day_of_week", "collection_quality_bucket"]

    rows = []
    coefficient_rows = []
    for origin, origin_frame in usable.groupby("forecast_origin", sort=False):
        origin_frame = origin_frame.sort_values(["exhibition_date", "movie_id", "amc_movie_id"]).copy()
        for idx, row in origin_frame.iterrows():
            train = origin_frame.loc[origin_frame["exhibition_date"] < row["exhibition_date"]].copy()
            if len(train) < config.min_train_n:
                pred_schedule = float(train["z_eod"].mean()) if len(train) else np.nan
                pred_seat = pred_schedule
                pace, pace_source, pace_n = rolling_pace_prior(train, row)
                s_obs = pd.to_numeric(pd.Series([row["s_obs"]]), errors="coerce").iloc[0]
                pred_pace = float(np.log1p(s_obs) - np.log(pace)) if np.isfinite(s_obs) else pred_schedule
                coefs_schedule: dict[str, float] = {}
                coefs_seat: dict[str, float] = {}
                coefs_pace: dict[str, float] = {}
            else:
                test = origin_frame.loc[[idx]].copy()
                pace, pace_source, pace_n = rolling_pace_prior(train, row)
                pred_schedule, coefs_schedule = fit_ridge_predict_target(
                    train,
                    test,
                    schedule_numeric,
                    [],
                    "z_eod",
                    ridge_lambda=config.ridge_lambda,
                )
                pred_seat, coefs_seat = fit_ridge_predict_target(
                    train,
                    test,
                    seat_numeric,
                    [],
                    "z_eod",
                    ridge_lambda=config.ridge_lambda,
                )
                pace_train = add_pace_prior_features(train, pace)
                pace_test = add_pace_prior_features(test, pace)
                pred_pace, coefs_pace = fit_ridge_predict_target(
                    pace_train,
                    pace_test,
                    pace_numeric,
                    categoricals,
                    "z_eod",
                    ridge_lambda=config.ridge_lambda,
                )
            for model, pred in [
                ("P0_schedule_only", pred_schedule),
                ("P1_schedule_plus_asof_seats", pred_seat),
                ("P2_pace_adjusted_eod_seats", pred_pace),
            ]:
                error = row["z_eod"] - pred if np.isfinite(pred) else np.nan
                rows.append(
                    {
                        "model": model,
                        "forecast_origin": origin,
                        "movie_id": row["movie_id"],
                        "amc_movie_id": row["amc_movie_id"],
                        "title": row.get("title", row.get("amc_movie_name")),
                        "amc_movie_name": row.get("amc_movie_name"),
                        "exhibition_date": row["exhibition_date"],
                        "day_of_week": row["day_of_week"],
                        "train_n": int(len(train)),
                        "z_eod": row["z_eod"],
                        "pred_z_eod": pred,
                        "error_log1p_seats": error,
                        "abs_error_log1p_seats": abs(error) if np.isfinite(error) else np.nan,
                        "s_eod": row["s_eod"],
                        "s_obs": row["s_obs"],
                        "c_scheduled_known": row["c_scheduled_known"],
                        "coverage": row["coverage"],
                        "n_snapshots": row["n_snapshots"],
                        "collection_quality_bucket": row.get("collection_quality_bucket"),
                        "pace_prior": pace,
                        "pace_prior_source": pace_source,
                        "pace_prior_n": pace_n,
                    }
                )
            coefficient_rows.append(
                {
                    "forecast_origin": origin,
                    "movie_id": row["movie_id"],
                    "amc_movie_id": row["amc_movie_id"],
                    "exhibition_date": row["exhibition_date"],
                    "train_n": int(len(train)),
                    "coef_schedule_log1p_c_scheduled": coefs_schedule.get("log1p_c_scheduled", np.nan),
                    "coef_seat_log1p_s_obs": coefs_seat.get("log1p_s_obs", np.nan),
                    "coef_seat_log1p_c_scheduled": coefs_seat.get("log1p_c_scheduled", np.nan),
                    "coef_seat_coverage": coefs_seat.get("coverage", np.nan),
                    "coef_pace_z_pace_prior": coefs_pace.get("z_pace_prior", np.nan),
                    "coef_pace_delay_p90_minutes": coefs_pace.get("delay_p90_minutes", np.nan),
                    "coef_pace_staleness_p90_minutes": coefs_pace.get("staleness_p90_minutes", np.nan),
                }
            )
    predictions = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    predictions.attrs["coefficients"] = pd.DataFrame(coefficient_rows)
    return predictions


def summarize_amc_pseudo_nowcast(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    predictions = predictions.copy()
    predictions.attrs = {}
    rows = []
    schedule = predictions.loc[predictions["model"].eq("P0_schedule_only")].copy()
    schedule.attrs = {}
    for (origin, model), group in predictions.groupby(["forecast_origin", "model"], sort=False):
        merge_keys = group[["forecast_origin", "movie_id", "amc_movie_id", "exhibition_date"]].copy()
        merge_keys.attrs = {}
        same_schedule = merge_keys.merge(
            schedule,
            on=["forecast_origin", "movie_id", "amc_movie_id", "exhibition_date"],
            how="left",
            suffixes=("", "_schedule"),
        )
        mae = float(group["abs_error_log1p_seats"].mean())
        rmse = float(np.sqrt(np.nanmean(np.square(group["error_log1p_seats"]))))
        schedule_mae = float(same_schedule["abs_error_log1p_seats"].mean()) if len(same_schedule) else np.nan
        schedule_rmse = (
            float(np.sqrt(np.nanmean(np.square(same_schedule["error_log1p_seats"]))))
            if len(same_schedule)
            else np.nan
        )
        rows.append(
            {
                "forecast_origin": origin,
                "model": model,
                "n": int(len(group)),
                "MAE_log1p_EOD_seats": mae,
                "RMSE_log1p_EOD_seats": rmse,
                "schedule_MAE_same_sample": schedule_mae,
                "schedule_RMSE_same_sample": schedule_rmse,
                "MAE_improvement_pct_vs_schedule": pct_improvement(schedule_mae, mae),
                "RMSE_improvement_pct_vs_schedule": pct_improvement(schedule_rmse, rmse),
                "mean_train_n": float(group["train_n"].mean()),
                "mean_coverage": float(group["coverage"].mean()),
                "mean_s_obs": float(group["s_obs"].mean()),
                "mean_s_eod": float(group["s_eod"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_actual_bridge_panel(panel: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "movie_id",
        "amc_movie_id",
        "amc_movie_name",
        "title",
        "exhibition_date",
        "day_of_week",
        "actual_gross_usd",
        "s_final_eod",
        "c_final_eod",
        "n_final_snapshots",
        "n_final_theatres",
        "c_scheduled_known",
        "n_scheduled_showtimes",
        "n_scheduled_theatres",
    ]
    available = [column for column in cols if column in panel.columns]
    bridge = panel.loc[panel["forecast_origin"].eq("EOD"), available].drop_duplicates(
        ["movie_id", "amc_movie_id", "exhibition_date"]
    )
    bridge = bridge.copy()
    bridge["actual_gross_usd"] = pd.to_numeric(bridge["actual_gross_usd"], errors="coerce")
    bridge["s_final_eod"] = pd.to_numeric(bridge["s_final_eod"], errors="coerce")
    bridge["log_actual_gross"] = np.log(bridge["actual_gross_usd"].where(bridge["actual_gross_usd"] > 0))
    bridge["log1p_s_final_eod"] = np.log1p(bridge["s_final_eod"])
    bridge["bridge_residual"] = bridge["log_actual_gross"] - bridge["log1p_s_final_eod"]
    return bridge.loc[bridge["actual_gross_usd"].gt(0) & bridge["s_final_eod"].gt(0)].replace([np.inf, -np.inf], np.nan)


def summarize_amc_to_actual_bridge(bridge: pd.DataFrame) -> pd.DataFrame:
    if bridge.empty:
        return pd.DataFrame()
    corr = float(bridge["log_actual_gross"].corr(bridge["log1p_s_final_eod"]))
    residual = bridge["bridge_residual"].dropna()
    rows = [
        {
            "scope": "overall",
            "day_of_week": "",
            "n": int(len(bridge)),
            "corr_log_actual_vs_log1p_eod_seats": corr,
            "median_bridge_residual": float(residual.median()) if len(residual) else np.nan,
            "mad_bridge_residual": float((residual - residual.median()).abs().median()) if len(residual) else np.nan,
            "bridge_residual_p25": float(residual.quantile(0.25)) if len(residual) else np.nan,
            "bridge_residual_p75": float(residual.quantile(0.75)) if len(residual) else np.nan,
        }
    ]
    for day, group in bridge.groupby("day_of_week", dropna=False):
        day_residual = group["bridge_residual"].dropna()
        rows.append(
            {
                "scope": "day_of_week",
                "day_of_week": day,
                "n": int(len(group)),
                "corr_log_actual_vs_log1p_eod_seats": float(group["log_actual_gross"].corr(group["log1p_s_final_eod"]))
                if len(group) > 1
                else np.nan,
                "median_bridge_residual": float(day_residual.median()) if len(day_residual) else np.nan,
                "mad_bridge_residual": float((day_residual - day_residual.median()).abs().median()) if len(day_residual) else np.nan,
                "bridge_residual_p25": float(day_residual.quantile(0.25)) if len(day_residual) else np.nan,
                "bridge_residual_p75": float(day_residual.quantile(0.75)) if len(day_residual) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def bridge_residual_prior(train: pd.DataFrame, row: pd.Series) -> float:
    same_movie = train.loc[train["movie_id"].eq(row["movie_id"]), "bridge_residual"].dropna()
    if len(same_movie):
        return float(same_movie.median())
    pooled = train["bridge_residual"].dropna()
    return float(pooled.median()) if len(pooled) else np.nan


def bridge_residual_prior_b3(train: pd.DataFrame, row: pd.Series) -> tuple[float, str, int]:
    same_movie = train.loc[train["movie_id"].eq(row["movie_id"]), "bridge_residual"].dropna()
    if len(same_movie):
        return float(same_movie.median()), "same_movie_prior_days", int(len(same_movie))
    same_day = train.loc[train["day_of_week"].eq(row["day_of_week"]), "bridge_residual"].dropna()
    if len(same_day):
        return float(same_day.median()), "pooled_same_day", int(len(same_day))
    pooled = train["bridge_residual"].dropna()
    if len(pooled):
        return float(pooled.median()), "pooled_all_days", int(len(pooled))
    return np.nan, "missing", 0


def schedule_only_actual_prediction(train: pd.DataFrame, row: pd.Series, ridge_lambda: float) -> float:
    if len(train) < 2:
        return np.nan
    train = train.copy()
    test = pd.DataFrame([row]).copy()
    for frame in [train, test]:
        frame["target_log_actual"] = np.log(pd.to_numeric(frame["actual_gross_usd"], errors="coerce").where(frame["actual_gross_usd"] > 0))
        frame["log1p_c_scheduled"] = np.log1p(pd.to_numeric(frame["c_scheduled_known"], errors="coerce"))
    pred, _ = fit_ridge_predict_target(
        train,
        test,
        ["log1p_c_scheduled"],
        [],
        "target_log_actual",
        ridge_lambda=ridge_lambda,
    )
    return float(np.exp(pred)) if np.isfinite(pred) else np.nan


def same_week_bridge_predictions(
    bridge: pd.DataFrame,
    amc_pseudo_predictions: pd.DataFrame,
    *,
    holdout_days: Iterable[str] = ("Friday", "Saturday", "Sunday"),
    ridge_lambda: float = RIDGE_LAMBDA,
    include_diagnostic_candidates: bool = False,
) -> pd.DataFrame:
    if bridge.empty:
        return pd.DataFrame()
    if amc_pseudo_predictions.empty or "model" not in amc_pseudo_predictions.columns:
        pseudo = pd.DataFrame()
    else:
        candidate_models = ["P1_schedule_plus_asof_seats"]
        if include_diagnostic_candidates:
            candidate_models.append("P2_pace_adjusted_eod_seats")
        pseudo = amc_pseudo_predictions.loc[amc_pseudo_predictions["model"].isin(candidate_models)].copy()
        pseudo = pseudo.rename(columns={"pred_z_eod": "pred_z_final_eod_asof", "model": "seat_nowcast_model"})
    bridge = bridge.sort_values(["exhibition_date", "movie_id", "amc_movie_id"]).copy()
    rows = []
    for _, row in bridge.loc[bridge["day_of_week"].isin(list(holdout_days))].iterrows():
        train = bridge.loc[bridge["exhibition_date"] < row["exhibition_date"]].copy()
        if train.empty:
            continue
        b_hat = bridge_residual_prior(train, row)
        b3_hat, b3_source, b3_n = bridge_residual_prior_b3(train, row)
        pred_b0 = schedule_only_actual_prediction(train, row, ridge_lambda)
        pred_b1 = np.exp(b_hat + row["log1p_s_final_eod"]) if np.isfinite(b_hat) else np.nan
        base = {
            "movie_id": row["movie_id"],
            "amc_movie_id": row["amc_movie_id"],
            "title": row.get("title", row.get("amc_movie_name")),
            "amc_movie_name": row.get("amc_movie_name"),
            "exhibition_date": row["exhibition_date"],
            "day_of_week": row["day_of_week"],
            "actual_gross_usd": row["actual_gross_usd"],
            "train_n": int(len(train)),
            "bridge_residual_prior": b_hat,
            "bridge_residual_prior_b3": b3_hat,
            "bridge_residual_prior_b3_source": b3_source,
            "bridge_residual_prior_b3_n": b3_n,
        }
        for model, pred, origin in [
            ("B0_schedule_only_actual", pred_b0, "final"),
            ("B1_true_final_eod_bridge", pred_b1, "final"),
        ]:
            rows.append(
                {
                    **base,
                    "forecast_origin": origin,
                    "model": model,
                    "pred_gross_usd": pred,
                    "log_error": np.log(row["actual_gross_usd"] / pred) if pred and pred > 0 else np.nan,
                    "abs_error_usd": abs(row["actual_gross_usd"] - pred) if pred and pred > 0 else np.nan,
                    "abs_pct_error": abs(row["actual_gross_usd"] - pred) / row["actual_gross_usd"] if pred and pred > 0 else np.nan,
                }
            )
        if pseudo.empty:
            asof_rows = pd.DataFrame()
        else:
            asof_rows = pseudo.loc[
                pseudo["movie_id"].eq(row["movie_id"])
                & pseudo["amc_movie_id"].eq(row["amc_movie_id"])
                & pseudo["exhibition_date"].eq(row["exhibition_date"])
            ]
        for _, asof in asof_rows.iterrows():
            if asof["seat_nowcast_model"] == "P1_schedule_plus_asof_seats":
                model = "B2_asof_predicted_eod_bridge"
                prior = b_hat
                prior_source = "same_movie_or_pooled"
                prior_n = np.nan
            else:
                model = "B3_pace_eod_same_movie_bridge"
                prior = b3_hat
                prior_source = b3_source
                prior_n = b3_n
            pred = np.exp(prior + asof["pred_z_final_eod_asof"]) if np.isfinite(prior) else np.nan
            rows.append(
                {
                    **base,
                    "forecast_origin": asof["forecast_origin"],
                    "model": model,
                    "seat_nowcast_model": asof["seat_nowcast_model"],
                    "pred_gross_usd": pred,
                    "log_error": np.log(row["actual_gross_usd"] / pred) if pred and pred > 0 else np.nan,
                    "abs_error_usd": abs(row["actual_gross_usd"] - pred) if pred and pred > 0 else np.nan,
                    "abs_pct_error": abs(row["actual_gross_usd"] - pred) / row["actual_gross_usd"] if pred and pred > 0 else np.nan,
                    "pred_z_final_eod_asof": asof["pred_z_final_eod_asof"],
                    "bridge_residual_prior_used": prior,
                    "bridge_residual_prior_source": prior_source,
                    "bridge_residual_prior_n": prior_n,
                    "collection_quality_bucket": asof.get("collection_quality_bucket"),
                }
            )
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def select_bias_residual_pool(
    train: pd.DataFrame,
    row: pd.Series,
    *,
    min_day_origin_n: int = 3,
    min_fallback_n: int = 8,
) -> tuple[pd.Series, str]:
    candidates = [
        (
            "day_origin_b2",
            min_day_origin_n,
            train.loc[
                train["day_of_week"].eq(row["day_of_week"])
                & train["forecast_origin"].eq(row["forecast_origin"])
                & train["model"].eq("B2_asof_predicted_eod_bridge"),
                "log_error",
            ],
        ),
        (
            "origin_b2",
            min_fallback_n,
            train.loc[
                train["forecast_origin"].eq(row["forecast_origin"])
                & train["model"].eq("B2_asof_predicted_eod_bridge"),
                "log_error",
            ],
        ),
        (
            "day_b2",
            min_fallback_n,
            train.loc[
                train["day_of_week"].eq(row["day_of_week"])
                & train["model"].eq("B2_asof_predicted_eod_bridge"),
                "log_error",
            ],
        ),
        (
            "pooled_b2",
            min_fallback_n,
            train.loc[train["model"].eq("B2_asof_predicted_eod_bridge"), "log_error"],
        ),
    ]
    for pool_name, min_n, residuals in candidates:
        clean = pd.to_numeric(residuals, errors="coerce").dropna()
        if len(clean) >= min_n:
            return clean, pool_name
    return pd.Series(dtype="float64"), "insufficient_history"


def add_b2_bias_adjusted_predictions(
    same_week_predictions: pd.DataFrame,
    *,
    method: str = "median",
) -> pd.DataFrame:
    if same_week_predictions.empty:
        return same_week_predictions
    b2 = same_week_predictions.loc[same_week_predictions["model"].eq("B2_asof_predicted_eod_bridge")].copy()
    if b2.empty:
        return same_week_predictions
    b2["exhibition_date"] = pd.to_datetime(b2["exhibition_date"], errors="coerce")
    ordered = b2.sort_values(["exhibition_date", "movie_id", "amc_movie_id", "forecast_origin"]).copy()
    adjusted_rows = []
    for idx, row in ordered.iterrows():
        train = ordered.loc[ordered["exhibition_date"] < row["exhibition_date"]].copy()
        residuals, source = select_bias_residual_pool(train, row)
        if len(residuals):
            mu = float(residuals.mean()) if method == "mean" else float(residuals.median())
        else:
            mu = 0.0
        pred = pd.to_numeric(pd.Series([row["pred_gross_usd"]]), errors="coerce").iloc[0]
        actual = pd.to_numeric(pd.Series([row["actual_gross_usd"]]), errors="coerce").iloc[0]
        pred_adj = float(pred * np.exp(mu)) if np.isfinite(pred) and pred > 0 else np.nan
        actual_value = float(actual) if np.isfinite(actual) else np.nan
        adjusted = row.copy()
        adjusted["exhibition_date"] = pd.Timestamp(row["exhibition_date"]).date()
        adjusted["model"] = f"B2_bias_adjusted_{method}"
        adjusted["pred_gross_usd"] = pred_adj
        adjusted["log_error"] = np.log(actual_value / pred_adj) if np.isfinite(actual_value) and pred_adj > 0 else np.nan
        adjusted["abs_error_usd"] = abs(actual_value - pred_adj) if np.isfinite(actual_value) and pred_adj > 0 else np.nan
        adjusted["abs_pct_error"] = abs(actual_value - pred_adj) / actual_value if np.isfinite(actual_value) and actual_value > 0 else np.nan
        adjusted["bias_adjustment_log"] = mu
        adjusted["bias_adjustment_factor"] = float(np.exp(mu))
        adjusted["bias_adjustment_source"] = source
        adjusted["bias_adjustment_train_n"] = int(len(residuals))
        adjusted_rows.append(adjusted)
    if not adjusted_rows:
        return same_week_predictions
    adjusted_frame = pd.DataFrame(adjusted_rows)
    out = pd.concat([same_week_predictions, adjusted_frame], ignore_index=True, sort=False)
    return out.replace([np.inf, -np.inf], np.nan)


def summarize_same_week_holdout(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows = []
    for (day, origin, model), group in predictions.groupby(["day_of_week", "forecast_origin", "model"], dropna=False):
        rows.append(
            {
                "day_of_week": day,
                "forecast_origin": origin,
                "model": model,
                "n": int(len(group)),
                "MAE_log": float(group["log_error"].abs().mean()),
                "RMSE_log": float(np.sqrt(np.nanmean(np.square(group["log_error"])))),
                "ME_log": float(group["log_error"].mean()),
                "BiasFactor": float(np.exp(group["log_error"].mean())),
                "MdAPE": float(group["abs_pct_error"].median()),
                "MAE_usd": float(group["abs_error_usd"].mean()),
                "mean_train_n": float(group["train_n"].mean()),
            }
        )
    return pd.DataFrame(rows)


def error_decomposition(
    actual_bridge_panel: pd.DataFrame,
    amc_pseudo_predictions: pd.DataFrame,
    same_week_predictions: pd.DataFrame,
) -> pd.DataFrame:
    if actual_bridge_panel.empty or amc_pseudo_predictions.empty or same_week_predictions.empty:
        return pd.DataFrame()
    bridge = actual_bridge_panel[
        ["movie_id", "amc_movie_id", "exhibition_date", "day_of_week", "log_actual_gross", "log1p_s_final_eod", "bridge_residual"]
    ].copy()
    asof_bridge = same_week_predictions.loc[
        same_week_predictions["model"].isin(
            ["B2_asof_predicted_eod_bridge", "B2_bias_adjusted_median", "B3_pace_eod_same_movie_bridge"]
        )
    ].copy()
    if asof_bridge.empty:
        return pd.DataFrame()
    rows = asof_bridge.merge(
        bridge,
        on=["movie_id", "amc_movie_id", "exhibition_date", "day_of_week"],
        how="left",
    )
    rows["eod_seat_nowcast_error"] = rows["log1p_s_final_eod"] - rows["pred_z_final_eod_asof"]
    prior_column = np.where(
        rows["model"].eq("B3_pace_eod_same_movie_bridge"),
        rows["bridge_residual_prior_b3"],
        rows["bridge_residual_prior"],
    )
    bias_adjustment = (
        pd.to_numeric(rows["bias_adjustment_log"], errors="coerce").fillna(0.0)
        if "bias_adjustment_log" in rows.columns
        else pd.Series(0.0, index=rows.index)
    )
    prior_column = np.where(
        rows["model"].eq("B2_bias_adjusted_median"),
        prior_column + bias_adjustment,
        prior_column,
    )
    rows["bridge_error"] = rows["bridge_residual"] - prior_column
    rows["total_log_gross_error"] = rows["log_error"]
    summaries = []
    for (day, origin, model), group in rows.groupby(["day_of_week", "forecast_origin", "model"], dropna=False):
        summaries.append(
            {
                "day_of_week": day,
                "forecast_origin": origin,
                "model": model,
                "n": int(len(group)),
                "seat_error_MAE": float(group["eod_seat_nowcast_error"].abs().mean()),
                "seat_error_RMSE": float(np.sqrt(np.nanmean(np.square(group["eod_seat_nowcast_error"])))),
                "bridge_error_MAE": float(group["bridge_error"].abs().mean()),
                "bridge_error_RMSE": float(np.sqrt(np.nanmean(np.square(group["bridge_error"])))),
                "total_error_MAE": float(group["total_log_gross_error"].abs().mean()),
                "total_error_RMSE": float(np.sqrt(np.nanmean(np.square(group["total_log_gross_error"])))),
                "seat_error_share_abs": float(
                    group["eod_seat_nowcast_error"].abs().mean()
                    / (group["eod_seat_nowcast_error"].abs().mean() + group["bridge_error"].abs().mean())
                )
                if (group["eod_seat_nowcast_error"].abs().mean() + group["bridge_error"].abs().mean()) > 0
                else np.nan,
            }
        )
    return pd.DataFrame(summaries)


def pace_by_origin(panel: pd.DataFrame) -> pd.DataFrame:
    pseudo = build_amc_pseudo_nowcast_panel(panel)
    if pseudo.empty:
        return pd.DataFrame()
    pseudo["pace"] = pd.to_numeric(pseudo["s_obs"], errors="coerce") / pd.to_numeric(pseudo["s_eod"], errors="coerce")
    pseudo["log_pace"] = np.log1p(pd.to_numeric(pseudo["s_obs"], errors="coerce")) - np.log1p(
        pd.to_numeric(pseudo["s_eod"], errors="coerce")
    )
    rows = []
    for (day, origin), group in pseudo.groupby(["day_of_week", "forecast_origin"], dropna=False):
        rows.append(
            {
                "day_of_week": day,
                "forecast_origin": origin,
                "n": int(len(group)),
                "median_pace": float(group["pace"].median()),
                "p25_pace": float(group["pace"].quantile(0.25)),
                "p75_pace": float(group["pace"].quantile(0.75)),
                "median_log_pace": float(group["log_pace"].median()),
                "mean_coverage": float(group["coverage"].mean()),
            }
        )
    return pd.DataFrame(rows)


def within_movie_bridge_summary(bridge: pd.DataFrame) -> pd.DataFrame:
    if bridge.empty:
        return pd.DataFrame()
    frame = bridge.copy()
    frame["demeaned_log_actual"] = frame["log_actual_gross"] - frame.groupby("movie_id")["log_actual_gross"].transform("mean")
    frame["demeaned_log_seats"] = frame["log1p_s_final_eod"] - frame.groupby("movie_id")["log1p_s_final_eod"].transform("mean")
    rows = [
        {
            "scope": "overall_raw",
            "day_of_week": "",
            "n": int(len(frame)),
            "correlation": float(frame["log_actual_gross"].corr(frame["log1p_s_final_eod"])),
        },
        {
            "scope": "overall_movie_demeaned",
            "day_of_week": "",
            "n": int(len(frame)),
            "correlation": float(frame["demeaned_log_actual"].corr(frame["demeaned_log_seats"])),
        },
    ]
    for day, group in frame.groupby("day_of_week", dropna=False):
        rows.append(
            {
                "scope": "movie_demeaned_by_day",
                "day_of_week": day,
                "n": int(len(group)),
                "correlation": float(group["demeaned_log_actual"].corr(group["demeaned_log_seats"])) if len(group) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def schedule_controlled_signal(panel: pd.DataFrame, bridge: pd.DataFrame) -> pd.DataFrame:
    if bridge.empty:
        return pd.DataFrame()
    panel = panel.copy()
    panel.attrs = {}
    base = bridge.copy()
    base.attrs = {}
    base["gross_per_capacity"] = base["log_actual_gross"] - np.log1p(pd.to_numeric(base["c_scheduled_known"], errors="coerce"))
    base["final_seat_intensity"] = base["log1p_s_final_eod"] - np.log1p(pd.to_numeric(base["c_scheduled_known"], errors="coerce"))
    rows = []
    rows.append(
        {
            "signal": "final_eod",
            "forecast_origin": "final",
            "day_of_week": "all",
            "n": int(len(base)),
            "corr_gross_per_capacity_vs_signal": float(base["gross_per_capacity"].corr(base["final_seat_intensity"])),
        }
    )
    for day, group in base.groupby("day_of_week", dropna=False):
        rows.append(
            {
                "signal": "final_eod",
                "forecast_origin": "final",
                "day_of_week": day,
                "n": int(len(group)),
                "corr_gross_per_capacity_vs_signal": float(group["gross_per_capacity"].corr(group["final_seat_intensity"]))
                if len(group) > 1
                else np.nan,
            }
        )
    asof_left = panel.loc[~panel["forecast_origin"].eq("EOD")].copy()
    asof_left.attrs = {}
    asof_right = base[["movie_id", "amc_movie_id", "exhibition_date", "day_of_week", "gross_per_capacity"]].copy()
    asof_right.attrs = {}
    asof = asof_left.merge(
        asof_right,
        on=["movie_id", "amc_movie_id", "exhibition_date", "day_of_week"],
        how="inner",
    )
    asof["asof_seat_intensity"] = np.log1p(pd.to_numeric(asof["s_obs"], errors="coerce")) - np.log1p(
        pd.to_numeric(asof["c_scheduled_known"], errors="coerce")
    )
    for (origin, day), group in asof.groupby(["forecast_origin", "day_of_week"], dropna=False):
        rows.append(
            {
                "signal": "asof",
                "forecast_origin": origin,
                "day_of_week": day,
                "n": int(len(group)),
                "corr_gross_per_capacity_vs_signal": float(group["gross_per_capacity"].corr(group["asof_seat_intensity"]))
                if len(group) > 1
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def forecast_trajectories(same_week_predictions: pd.DataFrame) -> pd.DataFrame:
    if same_week_predictions.empty:
        return pd.DataFrame()
    keep = same_week_predictions[
        [
            "movie_id",
            "amc_movie_id",
            "title",
            "amc_movie_name",
            "exhibition_date",
            "day_of_week",
            "forecast_origin",
            "model",
            "actual_gross_usd",
            "pred_gross_usd",
            "log_error",
        ]
    ].copy()
    order = {origin: idx for idx, origin in enumerate(["10:00", "12:00", "14:00", "16:00", "18:00", "20:00", "final"])}
    keep["origin_order"] = keep["forecast_origin"].map(order)
    return keep.sort_values(["exhibition_date", "movie_id", "model", "origin_order"])


def update_reliability(same_week_predictions: pd.DataFrame) -> pd.DataFrame:
    b2 = same_week_predictions.loc[same_week_predictions["model"].eq("B2_asof_predicted_eod_bridge")].copy()
    if b2.empty:
        return pd.DataFrame()
    order = {origin: idx for idx, origin in enumerate(["10:00", "12:00", "14:00", "16:00", "18:00", "20:00"])}
    b2["origin_order"] = b2["forecast_origin"].map(order)
    rows = []
    for _, group in b2.sort_values("origin_order").groupby(["movie_id", "amc_movie_id", "exhibition_date"], dropna=False):
        prev = None
        for _, row in group.iterrows():
            if prev is not None:
                update = row["pred_gross_usd"] - prev["pred_gross_usd"]
                remaining = prev["actual_gross_usd"] - prev["pred_gross_usd"]
                rows.append(
                    {
                        "day_of_week": row["day_of_week"],
                        "from_origin": prev["forecast_origin"],
                        "to_origin": row["forecast_origin"],
                        "correct_direction": np.sign(update) == np.sign(remaining) if update != 0 and remaining != 0 else np.nan,
                    }
                )
            prev = row
    updates = pd.DataFrame(rows)
    if updates.empty:
        return updates
    return (
        updates.groupby(["day_of_week", "from_origin", "to_origin"], dropna=False)
        .agg(n=("correct_direction", "count"), correct_direction_rate=("correct_direction", "mean"))
        .reset_index()
    )


def interval_coverage(same_week_predictions: pd.DataFrame) -> pd.DataFrame:
    if same_week_predictions.empty:
        return pd.DataFrame()
    scored = rolling_empirical_intervals(same_week_predictions)
    if scored.empty:
        return pd.DataFrame()
    rows = []
    for (day, origin, model, level), group in scored.groupby(
        ["day_of_week", "forecast_origin", "model", "interval_level"], dropna=False
    ):
        valid = group.dropna(subset=["lower_usd", "upper_usd", "actual_gross_usd"]).copy()
        if valid.empty:
            continue
        alpha = 1.0 - float(level)
        cover = valid["covered"]
        width = valid["upper_usd"] - valid["lower_usd"]
        score = interval_score_log(valid["log_error"], valid["lower_log_error_q"], valid["upper_log_error_q"], alpha)
        rows.append(
            {
                "day_of_week": day,
                "forecast_origin": origin,
                "model": model,
                "interval_level": level,
                "n": int(len(valid)),
                "coverage": float(cover.mean()),
                "median_width_usd": float(width.median()),
                "median_width_to_pred": float((width / valid["pred_gross_usd"]).median()),
                "mean_interval_score_log": float(score.mean()),
                "median_interval_score_log": float(score.median()),
                "mean_interval_train_n": float(valid["interval_train_n"].mean()),
                "primary_pool_share": float(valid["interval_pool_used"].eq("day_origin_model").mean()),
            }
        )
    return pd.DataFrame(rows)


def select_hierarchical_residual_pool(
    train: pd.DataFrame,
    row: pd.Series,
    *,
    min_pool_n: int,
) -> tuple[pd.Series, str]:
    candidates = [
        (
            "day_origin_model",
            train.loc[
                train["day_of_week"].eq(row["day_of_week"])
                & train["forecast_origin"].eq(row["forecast_origin"])
                & train["model"].eq(row["model"]),
                "log_error",
            ],
        ),
        (
            "origin_model",
            train.loc[
                train["forecast_origin"].eq(row["forecast_origin"]) & train["model"].eq(row["model"]),
                "log_error",
            ],
        ),
        (
            "day_model",
            train.loc[train["day_of_week"].eq(row["day_of_week"]) & train["model"].eq(row["model"]), "log_error"],
        ),
        ("pooled_same_model", train.loc[train["model"].eq(row["model"]), "log_error"]),
        ("pooled_b2", train.loc[train["model"].eq("B2_asof_predicted_eod_bridge"), "log_error"]),
    ]
    for pool_name, residuals in candidates:
        clean = pd.to_numeric(residuals, errors="coerce").dropna()
        if len(clean) >= min_pool_n:
            return clean, pool_name
    clean = pd.to_numeric(candidates[-1][1], errors="coerce").dropna()
    return clean, "pooled_b2_too_small"


def rolling_empirical_intervals(
    same_week_predictions: pd.DataFrame,
    *,
    min_pool_n: int = 8,
) -> pd.DataFrame:
    if same_week_predictions.empty:
        return pd.DataFrame()
    frame = same_week_predictions.copy()
    frame["exhibition_date"] = pd.to_datetime(frame["exhibition_date"], errors="coerce")
    frame = frame.loc[frame["model"].isin(["B2_asof_predicted_eod_bridge", "B2_bias_adjusted_median"])].copy()
    if frame.empty:
        return pd.DataFrame()
    rows = []
    ordered = frame.sort_values(["exhibition_date", "movie_id", "amc_movie_id", "forecast_origin"]).copy()
    for _, row in ordered.iterrows():
        train = ordered.loc[ordered["exhibition_date"] < row["exhibition_date"]].copy()
        residuals, pool_used = select_hierarchical_residual_pool(train, row, min_pool_n=min_pool_n)
        if len(residuals) < min_pool_n:
            continue
        for level, alpha in [(0.80, 0.20), (0.95, 0.05)]:
            lo_q = float(residuals.quantile(alpha / 2))
            hi_q = float(residuals.quantile(1 - alpha / 2))
            pred_value = pd.to_numeric(pd.Series([row["pred_gross_usd"]]), errors="coerce").iloc[0]
            actual_value = pd.to_numeric(pd.Series([row["actual_gross_usd"]]), errors="coerce").iloc[0]
            pred = float(pred_value) if np.isfinite(pred_value) else np.nan
            lower = pred * np.exp(lo_q) if np.isfinite(pred) and pred > 0 else np.nan
            upper = pred * np.exp(hi_q) if np.isfinite(pred) and pred > 0 else np.nan
            actual = float(actual_value) if np.isfinite(actual_value) else np.nan
            rows.append(
                {
                    "day_of_week": row["day_of_week"],
                    "forecast_origin": row["forecast_origin"],
                    "model": row["model"],
                    "movie_id": row["movie_id"],
                    "amc_movie_id": row["amc_movie_id"],
                    "title": row.get("title", row.get("amc_movie_name")),
                    "exhibition_date": row["exhibition_date"],
                    "interval_level": level,
                    "actual_gross_usd": actual,
                    "pred_gross_usd": pred,
                    "lower_usd": lower,
                    "upper_usd": upper,
                    "covered": bool(actual >= lower and actual <= upper)
                    if np.isfinite(actual) and np.isfinite(lower) and np.isfinite(upper)
                    else np.nan,
                    "log_error": row["log_error"],
                    "lower_log_error_q": lo_q,
                    "upper_log_error_q": hi_q,
                    "interval_train_n": int(len(residuals)),
                    "interval_pool_used": pool_used,
                }
            )
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def rolling_empirical_interval_rows(same_week_predictions: pd.DataFrame) -> pd.DataFrame:
    return rolling_empirical_intervals(same_week_predictions)


def summarize_b2_candidate(
    same_week_metrics: pd.DataFrame,
    intervals: pd.DataFrame,
    decomposition: pd.DataFrame,
) -> pd.DataFrame:
    if same_week_metrics.empty:
        return pd.DataFrame()
    b2 = same_week_metrics.loc[
        same_week_metrics["model"].isin(["B2_asof_predicted_eod_bridge", "B2_bias_adjusted_median"])
    ].copy()
    if b2.empty:
        return pd.DataFrame()
    key_cols = ["day_of_week", "forecast_origin", "model"]
    interval_pieces = []
    if not intervals.empty:
        for level, prefix in [(0.80, "interval_80"), (0.95, "interval_95")]:
            piece = intervals.loc[intervals["interval_level"].eq(level)].copy()
            if piece.empty:
                continue
            piece = piece.rename(
                columns={
                    "coverage": f"{prefix}_coverage",
                    "median_width_to_pred": f"{prefix}_median_width_to_pred",
                    "mean_interval_score_log": f"{prefix}_mean_score_log",
                    "mean_interval_train_n": f"{prefix}_mean_train_n",
                    "primary_pool_share": f"{prefix}_primary_pool_share",
                }
            )
            interval_pieces.append(
                piece[
                    key_cols
                    + [
                        f"{prefix}_coverage",
                        f"{prefix}_median_width_to_pred",
                        f"{prefix}_mean_score_log",
                        f"{prefix}_mean_train_n",
                        f"{prefix}_primary_pool_share",
                    ]
                ]
            )
    out = b2.copy()
    for piece in interval_pieces:
        out = out.merge(piece, on=key_cols, how="left")
    if not decomposition.empty:
        decomp = decomposition.loc[
            decomposition["model"].isin(["B2_asof_predicted_eod_bridge", "B2_bias_adjusted_median"])
        ].copy()
        out = out.merge(
            decomp[
                key_cols
                + [
                    "seat_error_MAE",
                    "seat_error_RMSE",
                    "bridge_error_MAE",
                    "bridge_error_RMSE",
                    "seat_error_share_abs",
                ]
            ],
            on=key_cols,
            how="left",
        )
    return out.sort_values(["day_of_week", "forecast_origin"]).reset_index(drop=True)


def delay_by_origin_day(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    rows = []
    for (day, origin), group in panel.loc[~panel["forecast_origin"].eq("EOD")].groupby(
        ["day_of_week", "forecast_origin"], dropna=False
    ):
        rows.append(
            {
                "day_of_week": day,
                "forecast_origin": origin,
                "n": int(len(group)),
                "median_delay_minutes": float(pd.to_numeric(group["delay_p50_minutes"], errors="coerce").median()),
                "p90_delay_minutes": float(pd.to_numeric(group["delay_p90_minutes"], errors="coerce").median()),
                "median_staleness_minutes": float(pd.to_numeric(group["staleness_p50_minutes"], errors="coerce").median()),
                "p90_staleness_minutes": float(pd.to_numeric(group["staleness_p90_minutes"], errors="coerce").median()),
                "median_effective_minutes_after_show": float(
                    pd.to_numeric(group["effective_minutes_after_show_p50"], errors="coerce").median()
                ),
                "share_snapshots_late_gt_30m": float(
                    pd.to_numeric(group["share_snapshots_late_gt_30m"], errors="coerce").mean()
                ),
                "share_capacity_observed_late_gt_30m": float(
                    pd.to_numeric(group["share_capacity_observed_late_gt_30m"], errors="coerce").mean()
                ),
                "mean_coverage": float(pd.to_numeric(group["coverage"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_amc_pseudo_nowcast_by_day(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or "day_of_week" not in predictions.columns:
        return pd.DataFrame()
    predictions = predictions.copy()
    predictions.attrs = {}
    rows = []
    schedule = predictions.loc[predictions["model"].eq("P0_schedule_only")].copy()
    schedule.attrs = {}
    keys = ["day_of_week", "forecast_origin", "movie_id", "amc_movie_id", "exhibition_date"]
    for (day, origin, model), group in predictions.groupby(["day_of_week", "forecast_origin", "model"], sort=False):
        merge_keys = group[keys].copy()
        merge_keys.attrs = {}
        same_schedule = merge_keys.merge(schedule, on=keys, how="left", suffixes=("", "_schedule"))
        mae = float(group["abs_error_log1p_seats"].mean())
        rmse = float(np.sqrt(np.nanmean(np.square(group["error_log1p_seats"]))))
        schedule_mae = float(same_schedule["abs_error_log1p_seats"].mean()) if len(same_schedule) else np.nan
        schedule_rmse = (
            float(np.sqrt(np.nanmean(np.square(same_schedule["error_log1p_seats"]))))
            if len(same_schedule)
            else np.nan
        )
        rows.append(
            {
                "day_of_week": day,
                "forecast_origin": origin,
                "model": model,
                "n": int(len(group)),
                "MAE_log1p_EOD_seats": mae,
                "RMSE_log1p_EOD_seats": rmse,
                "schedule_MAE_same_sample": schedule_mae,
                "schedule_RMSE_same_sample": schedule_rmse,
                "MAE_improvement_pct_vs_schedule": pct_improvement(schedule_mae, mae),
                "RMSE_improvement_pct_vs_schedule": pct_improvement(schedule_rmse, rmse),
                "mean_train_n": float(group["train_n"].mean()),
                "mean_coverage": float(group["coverage"].mean()),
                "mean_s_obs": float(group["s_obs"].mean()),
                "mean_s_eod": float(group["s_eod"].mean()),
            }
        )
    return pd.DataFrame(rows)


def operational_vs_oracle_metrics(
    operational_panel: pd.DataFrame,
    oracle_panel: pd.DataFrame,
    operational_predictions: pd.DataFrame,
    oracle_predictions: pd.DataFrame,
) -> pd.DataFrame:
    if operational_panel.empty or oracle_panel.empty or operational_predictions.empty or oracle_predictions.empty:
        return pd.DataFrame()
    op_pace = pace_by_origin(operational_panel).rename(
        columns={"median_pace": "operational_median_pace", "mean_coverage": "operational_mean_coverage"}
    )
    or_pace = pace_by_origin(oracle_panel).rename(
        columns={"median_pace": "oracle_median_pace", "mean_coverage": "oracle_mean_coverage"}
    )
    operational_metrics = summarize_amc_pseudo_nowcast_by_day(operational_predictions)
    oracle_metrics = summarize_amc_pseudo_nowcast_by_day(oracle_predictions)
    if operational_metrics.empty or oracle_metrics.empty:
        return pd.DataFrame()
    op_m = operational_metrics.loc[operational_metrics["model"].eq("P1_schedule_plus_asof_seats")].rename(
        columns={
            "RMSE_log1p_EOD_seats": "operational_RMSE_log",
            "MAE_log1p_EOD_seats": "operational_MAE_log",
        }
    )
    or_m = oracle_metrics.loc[oracle_metrics["model"].eq("P1_schedule_plus_asof_seats")].rename(
        columns={
            "RMSE_log1p_EOD_seats": "oracle_scheduled_RMSE_log",
            "MAE_log1p_EOD_seats": "oracle_scheduled_MAE_log",
        }
    )
    merged = op_m[
        ["day_of_week", "forecast_origin", "n", "operational_RMSE_log", "operational_MAE_log"]
    ].merge(
        or_m[["day_of_week", "forecast_origin", "oracle_scheduled_RMSE_log", "oracle_scheduled_MAE_log"]],
        on=["day_of_week", "forecast_origin"],
        how="inner",
    )
    delay = delay_by_origin_day(operational_panel)
    rows = []
    for _, metric in merged.iterrows():
        day = metric["day_of_week"]
        if day not in {"Friday", "Saturday", "Sunday"}:
            continue
        op_day = op_pace.loc[
            op_pace["forecast_origin"].eq(metric["forecast_origin"]) & op_pace["day_of_week"].eq(day)
        ]
        or_day = or_pace.loc[
            or_pace["forecast_origin"].eq(metric["forecast_origin"]) & or_pace["day_of_week"].eq(day)
        ]
        delay_day = delay.loc[
            delay["forecast_origin"].eq(metric["forecast_origin"]) & delay["day_of_week"].eq(day)
        ]
        if op_day.empty or or_day.empty:
            continue
        rows.append(
            {
                "day_of_week": day,
                "forecast_origin": metric["forecast_origin"],
                "model": "P1_schedule_plus_asof_seats",
                "n": int(metric["n"]),
                "operational_RMSE_log": metric["operational_RMSE_log"],
                "oracle_scheduled_RMSE_log": metric["oracle_scheduled_RMSE_log"],
                "oracle_improvement_pct": pct_improvement(
                    metric["operational_RMSE_log"], metric["oracle_scheduled_RMSE_log"]
                ),
                "operational_MAE_log": metric["operational_MAE_log"],
                "oracle_scheduled_MAE_log": metric["oracle_scheduled_MAE_log"],
                "operational_median_pace": float(op_day["operational_median_pace"].iloc[0]),
                "oracle_median_pace": float(or_day["oracle_median_pace"].iloc[0]),
                "pace_loss_from_lag": float(or_day["oracle_median_pace"].iloc[0] - op_day["operational_median_pace"].iloc[0]),
                "median_delay_minutes": float(delay_day["median_delay_minutes"].iloc[0]) if not delay_day.empty else np.nan,
                "p90_delay_minutes": float(delay_day["p90_delay_minutes"].iloc[0]) if not delay_day.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def delay_error_relation(panel: pd.DataFrame, amc_pseudo_predictions: pd.DataFrame) -> pd.DataFrame:
    if panel.empty or amc_pseudo_predictions.empty:
        return pd.DataFrame()
    p1 = amc_pseudo_predictions.loc[amc_pseudo_predictions["model"].eq("P1_schedule_plus_asof_seats")].copy()
    delay_cols = [
        "movie_id",
        "amc_movie_id",
        "exhibition_date",
        "forecast_origin",
        "day_of_week",
        "delay_p50_minutes",
        "delay_p90_minutes",
        "staleness_p50_minutes",
        "share_snapshots_late_gt_30m",
        "share_capacity_observed_late_gt_30m",
        "coverage",
    ]
    joined = p1.merge(panel[delay_cols], on=["movie_id", "amc_movie_id", "exhibition_date", "forecast_origin"], how="left")
    joined["abs_eod_seat_error"] = pd.to_numeric(joined["error_log1p_seats"], errors="coerce").abs()
    day_column = "day_of_week_y" if "day_of_week_y" in joined.columns else "day_of_week"
    rows = []
    for (day, origin), group in joined.groupby([day_column, "forecast_origin"], dropna=False):
        rows.append(
            {
                "day_of_week": day,
                "forecast_origin": origin,
                "n": int(len(group)),
                "corr_delay_p50_vs_abs_error": float(group["delay_p50_minutes"].corr(group["abs_eod_seat_error"]))
                if len(group) > 1
                else np.nan,
                "corr_delay_p90_vs_abs_error": float(group["delay_p90_minutes"].corr(group["abs_eod_seat_error"]))
                if len(group) > 1
                else np.nan,
                "corr_staleness_p50_vs_abs_error": float(group["staleness_p50_minutes"].corr(group["abs_eod_seat_error"]))
                if len(group) > 1
                else np.nan,
                "corr_late_capacity_share_vs_abs_error": float(
                    group["share_capacity_observed_late_gt_30m"].corr(group["abs_eod_seat_error"])
                )
                if len(group) > 1
                else np.nan,
                "median_abs_eod_seat_error": float(group["abs_eod_seat_error"].median()),
                "median_delay_minutes": float(group["delay_p50_minutes"].median()),
                "p90_delay_minutes": float(group["delay_p90_minutes"].median()),
            }
        )
    return pd.DataFrame(rows)


def summarize_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows = []
    baseline = predictions.loc[predictions["model"].eq("M0_baseline")]
    for (origin, model), group in predictions.groupby(["forecast_origin", "model"], sort=False):
        same_base = group[["forecast_origin", "movie_id", "exhibition_date"]].merge(
            baseline,
            on=["forecast_origin", "movie_id", "exhibition_date"],
            how="left",
            suffixes=("", "_baseline"),
        )
        mae = float(group["abs_error_usd"].mean())
        rmse_log = float(np.sqrt(np.nanmean(np.square(group["log_error"]))))
        mdape = float(group["abs_pct_error"].median())
        base_mae = float(same_base["abs_error_usd"].mean()) if len(same_base) else np.nan
        base_rmse_log = float(np.sqrt(np.nanmean(np.square(same_base["log_error"])))) if len(same_base) else np.nan
        rows.append(
            {
                "forecast_origin": origin,
                "model": model,
                "n": int(len(group)),
                "MAE_usd": mae,
                "RMSE_log": rmse_log,
                "ME_log": float(group["log_error"].mean()),
                "MdAPE": mdape,
                "baseline_MAE_usd_same_sample": base_mae,
                "baseline_RMSE_log_same_sample": base_rmse_log,
                "MAE_improvement_pct_vs_baseline": pct_improvement(base_mae, mae),
                "RMSE_log_improvement_pct_vs_baseline": pct_improvement(base_rmse_log, rmse_log),
                "mean_train_n": float(group["train_n"].mean()),
            }
        )
    return pd.DataFrame(rows)


def promise_threshold(metrics: pd.DataFrame, coefficients: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    for origin in metrics["forecast_origin"].dropna().unique():
        origin_metrics = metrics.loc[metrics["forecast_origin"].eq(origin)]
        model_c = origin_metrics.loc[origin_metrics["model"].eq("M2_baseline_plus_schedule_plus_seats")]
        model_b = origin_metrics.loc[origin_metrics["model"].eq("M1_baseline_plus_schedule")]
        if model_c.empty:
            continue
        c_row = model_c.iloc[0]
        b_row = model_b.iloc[0] if not model_b.empty else None
        coef_slice = coefficients.loc[
            coefficients["forecast_origin"].eq(origin)
            & coefficients["model"].eq("M2_baseline_plus_schedule_plus_seats")
            & coefficients["train_n"].ge(MIN_ROLLING_TRAIN_N)
        ]
        seat_coef_positive_share = (
            float((coef_slice["coef_log1p_s_obs"] > 0).mean()) if len(coef_slice) else np.nan
        )
        rows.append(
            {
                "forecast_origin": origin,
                "seat_beats_baseline_on_MAE": bool(c_row["MAE_improvement_pct_vs_baseline"] > 0),
                "seat_beats_baseline_on_RMSE_log": bool(c_row["RMSE_log_improvement_pct_vs_baseline"] > 0),
                "seat_beats_schedule_on_MAE": bool(b_row is not None and c_row["MAE_usd"] < b_row["MAE_usd"]),
                "seat_beats_schedule_on_RMSE_log": bool(b_row is not None and c_row["RMSE_log"] < b_row["RMSE_log"]),
                "passes_5pct_MAE_bar_vs_baseline": bool(c_row["MAE_improvement_pct_vs_baseline"] >= 0.05),
                "passes_5pct_RMSE_log_bar_vs_baseline": bool(c_row["RMSE_log_improvement_pct_vs_baseline"] >= 0.05),
                "seat_log_scale_coef_positive_share": seat_coef_positive_share,
                "rolling_n": int(c_row["n"]),
            }
        )
    return pd.DataFrame(rows)


def write_plots(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    snapshot_quality: pd.DataFrame | None = None,
    amc_pseudo_metrics: pd.DataFrame | None = None,
    actual_bridge_panel: pd.DataFrame | None = None,
    same_week_predictions: pd.DataFrame | None = None,
    pace_table: pd.DataFrame | None = None,
    schedule_controlled: pd.DataFrame | None = None,
    interval_table: pd.DataFrame | None = None,
    oracle_panel: pd.DataFrame | None = None,
    delay_relation: pd.DataFrame | None = None,
) -> None:
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_panel = panel.dropna(subset=["actual_gross_usd"]).copy()
    origins_to_plot = [origin for origin in ["12:00", "14:00", "16:00"] if origin in set(plot_panel["forecast_origin"])]
    for origin in origins_to_plot:
        group = plot_panel.loc[plot_panel["forecast_origin"].eq(origin)].copy()
        if group.empty:
            continue
        group["facet"] = np.where(group["is_opening_day"], "opening", "non-opening") + " " + group["day_of_week"].astype(str)

        fig, ax = plt.subplots(figsize=(9, 6))
        for facet, facet_group in group.groupby("facet"):
            ax.scatter(np.log1p(facet_group["s_obs"]), np.log(facet_group["actual_gross_usd"]), alpha=0.65, label=facet, s=28)
        ax.set_title(f"{origin}: log actual gross vs log1p as-of filled seats")
        ax.set_xlabel("log1p(weighted filled/unavailable seats observed as-of origin)")
        ax.set_ylabel("log(actual daily gross)")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"seat_nowcast_log_actual_vs_seats_{origin.replace(':', '')}.png", dpi=160)
        plt.close(fig)

        residual_group = group.dropna(subset=["baseline_residual_log"]).copy()
        if not residual_group.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].scatter(residual_group["log1p_s_obs"], residual_group["baseline_residual_log"], alpha=0.65, s=28)
            axes[0].set_xlabel("log1p(S_obs)")
            axes[0].set_ylabel("log(actual) - log(no-seat baseline)")
            axes[0].set_title(f"{origin}: residual vs seat scale")
            axes[1].scatter(residual_group["f_obs"], residual_group["baseline_residual_log"], alpha=0.65, s=28)
            axes[1].set_xlabel("F_obs")
            axes[1].set_ylabel("log(actual) - log(no-seat baseline)")
            axes[1].set_title(f"{origin}: residual vs fill intensity")
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / f"seat_nowcast_residual_vs_signal_{origin.replace(':', '')}.png", dpi=160)
            plt.close(fig)

    if not predictions.empty:
        seat_pred = predictions.loc[predictions["model"].eq("M2_baseline_plus_schedule_plus_seats")].copy()
        if not seat_pred.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(seat_pred["coverage"], seat_pred["abs_error_usd"], alpha=0.65, s=28)
            ax.set_xlabel("as-of capacity coverage")
            ax.set_ylabel("|forecast error|, USD")
            ax.set_title("Seat-update forecast error vs scrape coverage")
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / "seat_nowcast_error_vs_coverage.png", dpi=160)
            plt.close(fig)

        metrics = summarize_metrics(predictions)
        if not metrics.empty:
            pivot = metrics.pivot(index="forecast_origin", columns="model", values="RMSE_log").reindex(DEFAULT_ORIGINS)
            fig, ax = plt.subplots(figsize=(9, 5))
            for column in pivot.columns:
                ax.plot(pivot.index, pivot[column], marker="o", label=column)
            ax.set_xlabel("forecast origin")
            ax.set_ylabel("rolling-origin log RMSE")
            ax.set_title("Accuracy curve by forecast origin")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / "seat_nowcast_accuracy_curve_by_origin.png", dpi=160)
            plt.close(fig)

    lateness = panel.dropna(subset=["lateness_p50"]).copy()
    if not lateness.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        for origin, group in lateness.groupby("forecast_origin", sort=False):
            ax.hist(group["lateness_p50"] / 60.0, bins=30, alpha=0.35, label=origin)
        ax.set_xlabel("median snapshot lateness, minutes")
        ax.set_ylabel("movie-day-origin count")
        ax.set_title("As-of snapshot lateness distribution")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "seat_nowcast_lateness_distribution.png", dpi=160)
        plt.close(fig)

    if snapshot_quality is not None and not snapshot_quality.empty:
        quality = snapshot_quality.dropna(subset=["scheduled_hour_utc", "lateness_minutes_p50"]).copy()
        top_parsers = quality.groupby("parse_method")["n_snapshots"].sum().nlargest(4).index
        quality = quality.loc[quality["parse_method"].isin(top_parsers)]
        if not quality.empty:
            hourly_rows = []
            for (hour, parser), group in quality.groupby(["scheduled_hour_utc", "parse_method"], dropna=False):
                hourly_rows.append(
                    {
                        "scheduled_hour_utc": hour,
                        "parse_method": parser,
                        "weighted_lateness_minutes_p50": float(
                            np.average(group["lateness_minutes_p50"], weights=group["n_snapshots"].clip(lower=1))
                        ),
                    }
                )
            hourly = pd.DataFrame(hourly_rows)
            fig, ax = plt.subplots(figsize=(9, 5))
            for parser, group in hourly.groupby("parse_method", sort=False):
                ax.plot(group["scheduled_hour_utc"], group["weighted_lateness_minutes_p50"], marker="o", label=parser)
            ax.set_xlabel("scheduled hour, UTC")
            ax.set_ylabel("weighted p50 observed_at - scheduled_for, minutes")
            ax.set_title("Snapshot lateness by scheduled hour and parser")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / "seat_nowcast_lateness_by_hour_parser.png", dpi=160)
            plt.close(fig)

    if amc_pseudo_metrics is not None and not amc_pseudo_metrics.empty:
        pivot = amc_pseudo_metrics.pivot(index="forecast_origin", columns="model", values="RMSE_log1p_EOD_seats")
        pivot = pivot.reindex([origin for origin in DEFAULT_ORIGINS if origin != "EOD"])
        fig, ax = plt.subplots(figsize=(9, 5))
        for column in pivot.columns:
            ax.plot(pivot.index, pivot[column], marker="o", label=column)
        ax.set_xlabel("forecast origin")
        ax.set_ylabel("rolling-origin RMSE for log1p(EOD weighted seats)")
        ax.set_title("AMC-only pseudo-nowcast accuracy by origin")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "seat_nowcast_amc_pseudo_accuracy_curve.png", dpi=160)
        plt.close(fig)

    if actual_bridge_panel is not None and not actual_bridge_panel.empty:
        bridge = actual_bridge_panel.copy()
        fig, ax = plt.subplots(figsize=(9, 6))
        for day, group in bridge.groupby("day_of_week", dropna=False):
            ax.scatter(group["log1p_s_final_eod"], group["log_actual_gross"], alpha=0.75, s=35, label=day)
            for _, row in group.iterrows():
                label = str(row.get("title") or row.get("amc_movie_name") or "")[:18]
                ax.annotate(label, (row["log1p_s_final_eod"], row["log_actual_gross"]), fontsize=6, alpha=0.65)
        ax.set_xlabel("log1p(final EOD weighted AMC occupied seats)")
        ax.set_ylabel("log(actual national daily gross)")
        ax.set_title("AMC-to-Actual Bridge: EOD Seats vs Actual Gross")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "seat_nowcast_bridge_eod_seats_vs_actual.png", dpi=160)
        plt.close(fig)

        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        fig, ax = plt.subplots(figsize=(9, 5))
        for title, group in bridge.groupby("title", dropna=False):
            ordered = group.copy()
            ordered["day_order"] = ordered["day_of_week"].map({day: idx for idx, day in enumerate(day_order)})
            ordered = ordered.sort_values("day_order")
            ax.plot(ordered["day_of_week"], ordered["bridge_residual"], marker="o", alpha=0.5, linewidth=1)
        ax.axhline(bridge["bridge_residual"].median(), color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("day of week")
        ax.set_ylabel("log(actual gross) - log1p(final EOD AMC seats)")
        ax.set_title("AMC-to-Actual Bridge Residual by Day")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "seat_nowcast_bridge_residual_by_day.png", dpi=160)
        plt.close(fig)

    if same_week_predictions is not None and not same_week_predictions.empty:
        plot_rows = same_week_predictions.loc[
            same_week_predictions["model"].isin(
                [
                    "B0_schedule_only_actual",
                    "B1_true_final_eod_bridge",
                    "B2_asof_predicted_eod_bridge",
                    "B3_pace_eod_same_movie_bridge",
                ]
            )
        ].copy()
        if not plot_rows.empty:
            facets = ["Friday", "Saturday", "Sunday"]
            fig, axes = plt.subplots(1, len(facets), figsize=(15, 5), sharex=False, sharey=False)
            for ax, day in zip(axes, facets):
                group = plot_rows.loc[plot_rows["day_of_week"].eq(day)]
                for model, model_group in group.groupby("model"):
                    ax.scatter(model_group["actual_gross_usd"], model_group["pred_gross_usd"], alpha=0.7, s=32, label=model)
                if not group.empty:
                    lo = min(group["actual_gross_usd"].min(), group["pred_gross_usd"].min())
                    hi = max(group["actual_gross_usd"].max(), group["pred_gross_usd"].max())
                    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
                ax.set_title(day)
                ax.set_xlabel("actual gross")
                ax.set_ylabel("predicted gross")
            axes[-1].legend(fontsize=7)
            fig.tight_layout()
        fig.savefig(PLOTS_DIR / "seat_nowcast_same_week_holdout_actual_vs_pred.png", dpi=160)
        plt.close(fig)

    if actual_bridge_panel is not None and not actual_bridge_panel.empty:
        bridge = actual_bridge_panel.copy()
        bridge["demeaned_log_actual"] = bridge["log_actual_gross"] - bridge.groupby("movie_id")["log_actual_gross"].transform("mean")
        bridge["demeaned_log_seats"] = bridge["log1p_s_final_eod"] - bridge.groupby("movie_id")["log1p_s_final_eod"].transform("mean")
        fig, ax = plt.subplots(figsize=(8, 6))
        for day, group in bridge.groupby("day_of_week", dropna=False):
            ax.scatter(group["demeaned_log_seats"], group["demeaned_log_actual"], alpha=0.75, s=35, label=day)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("movie-demeaned log1p(final EOD AMC seats)")
        ax.set_ylabel("movie-demeaned log(actual gross)")
        ax.set_title("Within-Movie AMC Bridge")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "seat_nowcast_bridge_demeaned_within_movie.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 6))
        fs = bridge.loc[bridge["day_of_week"].isin(["Friday", "Saturday", "Sunday"])].copy()
        wide = fs.pivot_table(index="movie_id", columns="day_of_week", values="bridge_residual", aggfunc="median")
        if {"Friday", "Saturday"}.issubset(wide.columns):
            ax.scatter(wide["Friday"], wide["Saturday"], label="Fri -> Sat", alpha=0.75)
        if {"Saturday", "Sunday"}.issubset(wide.columns):
            ax.scatter(wide["Saturday"], wide["Sunday"], label="Sat -> Sun", alpha=0.75)
        ax.set_xlabel("prior-day bridge residual")
        ax.set_ylabel("next-day bridge residual")
        ax.set_title("Bridge Residual Stability")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "seat_nowcast_bridge_residual_fri_to_sat_sun.png", dpi=160)
        plt.close(fig)

        sc_plot = bridge.copy()
        sc_plot["gross_per_capacity"] = sc_plot["log_actual_gross"] - np.log1p(pd.to_numeric(sc_plot["c_scheduled_known"], errors="coerce"))
        sc_plot["seat_intensity"] = sc_plot["log1p_s_final_eod"] - np.log1p(pd.to_numeric(sc_plot["c_scheduled_known"], errors="coerce"))
        fig, ax = plt.subplots(figsize=(8, 6))
        for day, group in sc_plot.groupby("day_of_week", dropna=False):
            ax.scatter(group["seat_intensity"], group["gross_per_capacity"], alpha=0.75, s=35, label=day)
        ax.set_xlabel("log1p(final EOD seats) - log1p(scheduled capacity)")
        ax.set_ylabel("log(actual gross) - log1p(scheduled capacity)")
        ax.set_title("Seat Intensity vs Gross Per Scheduled Capacity")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "seat_nowcast_seat_intensity_vs_gross_per_capacity.png", dpi=160)
        plt.close(fig)

    if panel is not None and not panel.empty:
        pseudo_panel = build_amc_pseudo_nowcast_panel(panel)
        pseudo_panel = pseudo_panel.loc[pseudo_panel["day_of_week"].isin(["Friday", "Saturday", "Sunday"])].copy()
        if not pseudo_panel.empty:
            pseudo_panel["pace"] = pd.to_numeric(pseudo_panel["s_obs"], errors="coerce") / pd.to_numeric(
                pseudo_panel["s_eod"], errors="coerce"
            )
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
            for ax, day in zip(axes, ["Friday", "Saturday", "Sunday"]):
                day_rows = pseudo_panel.loc[pseudo_panel["day_of_week"].eq(day)]
                for _, group in day_rows.groupby(["movie_id", "amc_movie_id", "exhibition_date"], dropna=False):
                    ax.plot(group["forecast_origin"], group["pace"], marker="o", alpha=0.35, linewidth=1)
                ax.set_title(day)
                ax.set_xlabel("forecast origin")
                ax.set_ylabel("S_obs / S_final_EOD")
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / "seat_nowcast_seat_pace_curves_by_origin_day.png", dpi=160)
            plt.close(fig)

    if same_week_predictions is not None and not same_week_predictions.empty:
        traj = forecast_trajectories(same_week_predictions)
        asof_bridge = traj.loc[traj["model"].isin(["B2_asof_predicted_eod_bridge", "B3_pace_eod_same_movie_bridge"])].copy()
        if not asof_bridge.empty:
            fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            for ax, day in zip(axes, ["Friday", "Saturday", "Sunday"]):
                day_rows = asof_bridge.loc[asof_bridge["day_of_week"].eq(day)]
                for _, group in day_rows.groupby(["movie_id", "amc_movie_id", "exhibition_date", "model"], dropna=False):
                    ax.plot(group["forecast_origin"], group["pred_gross_usd"], marker="o", alpha=0.35, linewidth=1)
                    ax.axhline(group["actual_gross_usd"].iloc[0], color="black", linewidth=0.5, alpha=0.15)
                ax.set_title(day)
                ax.set_ylabel("predicted gross")
            axes[-1].set_xlabel("forecast origin")
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / "seat_nowcast_forecast_trajectory_by_movie_day.png", dpi=160)
            plt.close(fig)

    if interval_table is not None and not interval_table.empty:
        subset = interval_table.loc[
            interval_table["model"].isin(["B2_asof_predicted_eod_bridge", "B3_pace_eod_same_movie_bridge"])
        ].copy()
        if not subset.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            for level, group in subset.groupby("interval_level"):
                med = group.groupby("forecast_origin")["median_width_to_pred"].median().reindex(
                    [origin for origin in DEFAULT_ORIGINS if origin != "EOD"]
                )
                ax.plot(med.index, med.values, marker="o", label=f"{int(level * 100)}%")
            ax.set_xlabel("forecast origin")
            ax.set_ylabel("median interval width / prediction")
            ax.set_title("Empirical Interval Width by Origin")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / "seat_nowcast_interval_width_by_origin.png", dpi=160)
            plt.close(fig)

    if oracle_panel is not None and not oracle_panel.empty and panel is not None and not panel.empty:
        op = pace_by_origin(panel)
        oracle = pace_by_origin(oracle_panel)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        for ax, day in zip(axes, ["Friday", "Saturday", "Sunday"]):
            op_day = op.loc[op["day_of_week"].eq(day)]
            or_day = oracle.loc[oracle["day_of_week"].eq(day)]
            ax.plot(op_day["forecast_origin"], op_day["median_pace"], marker="o", label="operational")
            ax.plot(or_day["forecast_origin"], or_day["median_pace"], marker="o", label="on-time oracle")
            ax.set_title(day)
            ax.set_xlabel("forecast origin")
            ax.set_ylabel("median observed share of final EOD seats")
        axes[-1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "seat_nowcast_seat_pace_operational_vs_oracle.png", dpi=160)
        plt.close(fig)

    if delay_relation is not None and not delay_relation.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        subset = delay_relation.loc[delay_relation["day_of_week"].isin(["Friday", "Saturday", "Sunday"])]
        for day, group in subset.groupby("day_of_week", dropna=False):
            ax.plot(group["forecast_origin"], group["corr_delay_p90_vs_abs_error"], marker="o", label=day)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("forecast origin")
        ax.set_ylabel("corr(p90 collection delay, abs EOD-seat error)")
        ax.set_title("Delay vs EOD Seat Nowcast Error")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "seat_nowcast_delay_minutes_vs_eod_seat_error.png", dpi=160)
        plt.close(fig)

    if panel is not None and not panel.empty:
        delay = delay_by_origin_day(panel)
        subset = delay.loc[delay["day_of_week"].isin(["Friday", "Saturday", "Sunday"])]
        if not subset.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            for day, group in subset.groupby("day_of_week", dropna=False):
                ax.plot(group["forecast_origin"], group["p90_delay_minutes"], marker="o", label=day)
            ax.set_xlabel("forecast origin")
            ax.set_ylabel("median row p90 collection delay, minutes")
            ax.set_title("Collection Delay by Origin")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / "seat_nowcast_delay_by_origin_parser.png", dpi=160)
            plt.close(fig)

    if oracle_panel is not None and not oracle_panel.empty and panel is not None and not panel.empty:
        op = pace_by_origin(panel)
        oracle = pace_by_origin(oracle_panel)
        loss = op.merge(
            oracle,
            on=["day_of_week", "forecast_origin"],
            suffixes=("_operational", "_oracle"),
        )
        loss["pace_loss"] = loss["median_pace_oracle"] - loss["median_pace_operational"]
        subset = loss.loc[loss["day_of_week"].isin(["Friday", "Saturday", "Sunday"])]
        if not subset.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            for day, group in subset.groupby("day_of_week", dropna=False):
                ax.plot(group["forecast_origin"], group["pace_loss"], marker="o", label=day)
            ax.set_xlabel("forecast origin")
            ax.set_ylabel("oracle median pace - operational median pace")
            ax.set_title("Coverage/Pace Loss From Backlog")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / "seat_nowcast_coverage_loss_from_backlog_by_origin.png", dpi=160)
            plt.close(fig)


def write_outputs(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    promise: pd.DataFrame,
    snapshot_quality: pd.DataFrame | None = None,
    amc_pseudo_predictions: pd.DataFrame | None = None,
    amc_pseudo_metrics: pd.DataFrame | None = None,
    actual_bridge_panel: pd.DataFrame | None = None,
    bridge_summary: pd.DataFrame | None = None,
    same_week_predictions: pd.DataFrame | None = None,
    same_week_metrics: pd.DataFrame | None = None,
    decomposition: pd.DataFrame | None = None,
    pace: pd.DataFrame | None = None,
    within_movie: pd.DataFrame | None = None,
    schedule_controlled: pd.DataFrame | None = None,
    trajectories: pd.DataFrame | None = None,
    reliability: pd.DataFrame | None = None,
    intervals: pd.DataFrame | None = None,
    interval_rows: pd.DataFrame | None = None,
    b2_candidate_summary: pd.DataFrame | None = None,
    delay_loss: pd.DataFrame | None = None,
    oracle_metrics: pd.DataFrame | None = None,
    delay_by_origin: pd.DataFrame | None = None,
    delay_relation: pd.DataFrame | None = None,
) -> None:
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(DIAGNOSTICS_DIR / "same_day_seat_nowcast_origin_panel.csv", index=False)
    if snapshot_quality is not None and not snapshot_quality.empty:
        snapshot_quality.to_csv(DIAGNOSTICS_DIR / "same_day_seat_nowcast_lateness_by_hour.csv", index=False)
    if not predictions.empty:
        predictions.to_csv(DIAGNOSTICS_DIR / "same_day_seat_nowcast_rolling_predictions.csv", index=False)
        coefficients = predictions.attrs.get("coefficients", pd.DataFrame())
        if isinstance(coefficients, pd.DataFrame) and not coefficients.empty:
            coefficients.to_csv(DIAGNOSTICS_DIR / "same_day_seat_nowcast_rolling_coefficients.csv", index=False)
    if not metrics.empty:
        metrics.to_csv(DIAGNOSTICS_DIR / "same_day_seat_nowcast_model_metrics.csv", index=False)
    if not promise.empty:
        promise.to_csv(DIAGNOSTICS_DIR / "same_day_seat_nowcast_promise_thresholds.csv", index=False)
    if amc_pseudo_predictions is not None and not amc_pseudo_predictions.empty:
        amc_pseudo_predictions.to_csv(DIAGNOSTICS_DIR / "same_day_seat_nowcast_amc_pseudo_predictions.csv", index=False)
        coefficients = amc_pseudo_predictions.attrs.get("coefficients", pd.DataFrame())
        if isinstance(coefficients, pd.DataFrame) and not coefficients.empty:
            coefficients.to_csv(DIAGNOSTICS_DIR / "same_day_seat_nowcast_amc_pseudo_coefficients.csv", index=False)
    if amc_pseudo_metrics is not None and not amc_pseudo_metrics.empty:
        amc_pseudo_metrics.to_csv(DIAGNOSTICS_DIR / "same_day_seat_nowcast_amc_pseudo_metrics.csv", index=False)
    if actual_bridge_panel is not None and not actual_bridge_panel.empty:
        actual_bridge_panel.to_csv(DIAGNOSTICS_DIR / "same_day_seat_nowcast_actual_bridge_panel.csv", index=False)
    if bridge_summary is not None and not bridge_summary.empty:
        bridge_summary.to_csv(DIAGNOSTICS_DIR / "same_day_seat_nowcast_actual_bridge_summary.csv", index=False)
    if same_week_predictions is not None and not same_week_predictions.empty:
        same_week_predictions.to_csv(DIAGNOSTICS_DIR / "same_day_seat_nowcast_same_week_predictions.csv", index=False)
    if same_week_metrics is not None and not same_week_metrics.empty:
        same_week_metrics.to_csv(DIAGNOSTICS_DIR / "same_day_seat_nowcast_same_week_metrics.csv", index=False)
    if decomposition is not None and not decomposition.empty:
        decomposition.to_csv(DIAGNOSTICS_DIR / "same_day_seat_nowcast_error_decomposition.csv", index=False)
    if pace is not None and not pace.empty:
        pace.to_csv(DIAGNOSTICS_DIR / "same_day_seat_nowcast_pace_by_origin.csv", index=False)
    if within_movie is not None and not within_movie.empty:
        within_movie.to_csv(DIAGNOSTICS_DIR / "same_day_seat_nowcast_within_movie_bridge_summary.csv", index=False)
    if schedule_controlled is not None and not schedule_controlled.empty:
        schedule_controlled.to_csv(DIAGNOSTICS_DIR / "same_day_seat_nowcast_schedule_controlled_signal.csv", index=False)
    if trajectories is not None and not trajectories.empty:
        trajectories.to_csv(DIAGNOSTICS_DIR / "same_day_seat_nowcast_forecast_trajectories.csv", index=False)
    if reliability is not None and not reliability.empty:
        reliability.to_csv(DIAGNOSTICS_DIR / "same_day_seat_nowcast_update_reliability.csv", index=False)
    if intervals is not None and not intervals.empty:
        intervals.to_csv(DIAGNOSTICS_DIR / "same_day_seat_nowcast_interval_coverage.csv", index=False)
    if interval_rows is not None and not interval_rows.empty:
        interval_rows.to_csv(DIAGNOSTICS_DIR / "same_day_seat_nowcast_interval_rows.csv", index=False)
    if b2_candidate_summary is not None and not b2_candidate_summary.empty:
        b2_candidate_summary.to_csv(DIAGNOSTICS_DIR / "same_day_seat_nowcast_b2_candidate_summary.csv", index=False)
    if delay_loss is not None and not delay_loss.empty:
        delay_loss.to_csv(DIAGNOSTICS_DIR / "same_day_seat_nowcast_delay_loss.csv", index=False)
        delay_loss.to_csv(DIAGNOSTICS_DIR / "same_day_seat_nowcast_operational_vs_oracle_metrics.csv", index=False)
    if oracle_metrics is not None and not oracle_metrics.empty:
        oracle_metrics.to_csv(DIAGNOSTICS_DIR / "same_day_seat_nowcast_oracle_pseudo_metrics.csv", index=False)
    if delay_by_origin is not None and not delay_by_origin.empty:
        delay_by_origin.to_csv(DIAGNOSTICS_DIR / "same_day_seat_nowcast_delay_by_origin_day.csv", index=False)
    if delay_relation is not None and not delay_relation.empty:
        delay_relation.to_csv(DIAGNOSTICS_DIR / "same_day_seat_nowcast_delay_error_relation.csv", index=False)


def run_analysis(
    *,
    database_url: str | None = None,
    baseline_csv: Path | None = None,
    origin_timezone: str = DEFAULT_ORIGIN_TZ,
    origins: Iterable[str] = DEFAULT_ORIGINS,
    min_train_n: int = MIN_ROLLING_TRAIN_N,
    write: bool = True,
    plots: bool = True,
) -> dict[str, pd.DataFrame]:
    conn = connect_database(database_url)
    try:
        panel = build_panel(conn, origins=origins, origin_timezone=origin_timezone, baseline_csv=baseline_csv)
    finally:
        conn.close()

    snapshot_quality = panel.attrs.get("snapshot_quality", pd.DataFrame())
    oracle_panel = panel.attrs.get("oracle_panel", pd.DataFrame())
    predictions = rolling_origin_predictions(panel, RollingConfig(min_train_n=min_train_n))
    coefficients = predictions.attrs.get("coefficients", pd.DataFrame()) if not predictions.empty else pd.DataFrame()
    metrics = summarize_metrics(predictions)
    promise = promise_threshold(metrics, coefficients if isinstance(coefficients, pd.DataFrame) else pd.DataFrame())
    amc_pseudo_predictions = rolling_amc_pseudo_nowcast_predictions(panel, RollingConfig(min_train_n=min_train_n))
    amc_pseudo_metrics = summarize_amc_pseudo_nowcast(amc_pseudo_predictions)
    oracle_pseudo_predictions = rolling_amc_pseudo_nowcast_predictions(oracle_panel, RollingConfig(min_train_n=min_train_n))
    oracle_pseudo_metrics = summarize_amc_pseudo_nowcast(oracle_pseudo_predictions)
    actual_bridge_panel = build_actual_bridge_panel(panel)
    bridge_summary = summarize_amc_to_actual_bridge(actual_bridge_panel)
    same_week_predictions = same_week_bridge_predictions(actual_bridge_panel, amc_pseudo_predictions)
    same_week_predictions = add_b2_bias_adjusted_predictions(same_week_predictions)
    same_week_metrics = summarize_same_week_holdout(same_week_predictions)
    decomposition = error_decomposition(actual_bridge_panel, amc_pseudo_predictions, same_week_predictions)
    pace = pace_by_origin(panel)
    within_movie = within_movie_bridge_summary(actual_bridge_panel)
    schedule_controlled = schedule_controlled_signal(panel, actual_bridge_panel)
    trajectories = forecast_trajectories(same_week_predictions)
    reliability = update_reliability(same_week_predictions)
    intervals = interval_coverage(same_week_predictions)
    interval_rows = rolling_empirical_interval_rows(same_week_predictions)
    b2_candidate_summary = summarize_b2_candidate(same_week_metrics, intervals, decomposition)
    delay_loss = operational_vs_oracle_metrics(panel, oracle_panel, amc_pseudo_predictions, oracle_pseudo_predictions)
    delay_origin = delay_by_origin_day(panel)
    delay_relation = delay_error_relation(panel, amc_pseudo_predictions)

    if write:
        write_outputs(
            panel,
            predictions,
            metrics,
            promise,
            snapshot_quality,
            amc_pseudo_predictions,
            amc_pseudo_metrics,
            actual_bridge_panel,
            bridge_summary,
            same_week_predictions,
            same_week_metrics,
            decomposition,
            pace,
            within_movie,
            schedule_controlled,
            trajectories,
            reliability,
            intervals,
            interval_rows,
            b2_candidate_summary,
            delay_loss,
            oracle_pseudo_metrics,
            delay_origin,
            delay_relation,
        )
    if plots:
        write_plots(
            panel,
            predictions,
            snapshot_quality,
            amc_pseudo_metrics,
            actual_bridge_panel,
            same_week_predictions,
            pace,
            schedule_controlled,
            intervals,
            oracle_panel,
            delay_relation,
        )
    return {
        "panel": panel,
        "oracle_panel": oracle_panel if isinstance(oracle_panel, pd.DataFrame) else pd.DataFrame(),
        "predictions": predictions,
        "metrics": metrics,
        "promise": promise,
        "coefficients": coefficients if isinstance(coefficients, pd.DataFrame) else pd.DataFrame(),
        "snapshot_quality": snapshot_quality if isinstance(snapshot_quality, pd.DataFrame) else pd.DataFrame(),
        "amc_pseudo_predictions": amc_pseudo_predictions,
        "amc_pseudo_metrics": amc_pseudo_metrics,
        "oracle_pseudo_predictions": oracle_pseudo_predictions,
        "oracle_pseudo_metrics": oracle_pseudo_metrics,
        "actual_bridge_panel": actual_bridge_panel,
        "bridge_summary": bridge_summary,
        "same_week_predictions": same_week_predictions,
        "same_week_metrics": same_week_metrics,
        "decomposition": decomposition,
        "pace": pace,
        "within_movie": within_movie,
        "schedule_controlled": schedule_controlled,
        "trajectories": trajectories,
        "reliability": reliability,
        "intervals": intervals,
        "interval_rows": interval_rows,
        "b2_candidate_summary": b2_candidate_summary,
        "delay_loss": delay_loss,
        "delay_by_origin": delay_origin,
        "delay_relation": delay_relation,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_database_arg(parser)
    parser.add_argument("--baseline-csv", type=Path, help="Optional CSV containing no-seat daily baseline forecasts.")
    parser.add_argument("--origin-timezone", default=DEFAULT_ORIGIN_TZ, help="Timezone used for the fixed forecast-origin clock.")
    parser.add_argument("--origins", nargs="+", default=list(DEFAULT_ORIGINS), help="Forecast origins, e.g. 10:00 12:00 EOD.")
    parser.add_argument("--min-train-n", type=int, default=MIN_ROLLING_TRAIN_N)
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation.")
    parser.add_argument("--no-write", action="store_true", help="Build outputs in memory only.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    outputs = run_analysis(
        database_url=args.database_url,
        baseline_csv=args.baseline_csv,
        origin_timezone=args.origin_timezone,
        origins=args.origins,
        min_train_n=args.min_train_n,
        write=not args.no_write,
        plots=not args.no_plots,
    )
    panel = outputs["panel"]
    predictions = outputs["predictions"]
    metrics = outputs["metrics"]
    amc_pseudo_predictions = outputs["amc_pseudo_predictions"]
    amc_pseudo_metrics = outputs["amc_pseudo_metrics"]
    actual_bridge_panel = outputs["actual_bridge_panel"]
    bridge_summary = outputs["bridge_summary"]
    same_week_predictions = outputs["same_week_predictions"]
    same_week_metrics = outputs["same_week_metrics"]
    decomposition = outputs["decomposition"]
    intervals = outputs["intervals"]
    interval_rows = outputs["interval_rows"]
    b2_candidate_summary = outputs["b2_candidate_summary"]
    delay_loss = outputs["delay_loss"]
    delay_by_origin = outputs["delay_by_origin"]
    delay_relation = outputs["delay_relation"]
    print("Same-day seat nowcast EDA complete:")
    print(f"  origin panel rows: {len(panel)}")
    print(f"  rows with actuals: {int(panel['actual_gross_usd'].notna().sum())}")
    print(f"  rows with baseline: {int(panel['baseline_gross_usd'].notna().sum())}")
    print(f"  rolling prediction rows: {len(predictions)}")
    print(f"  metric rows: {len(metrics)}")
    print(f"  AMC pseudo-nowcast prediction rows: {len(amc_pseudo_predictions)}")
    print(f"  AMC pseudo-nowcast metric rows: {len(amc_pseudo_metrics)}")
    print(f"  actual bridge rows: {len(actual_bridge_panel)}")
    print(f"  actual bridge summary rows: {len(bridge_summary)}")
    print(f"  same-week holdout prediction rows: {len(same_week_predictions)}")
    print(f"  same-week holdout metric rows: {len(same_week_metrics)}")
    print(f"  error decomposition rows: {len(decomposition)}")
    print(f"  interval coverage rows: {len(intervals)}")
    print(f"  interval detail rows: {len(interval_rows)}")
    print(f"  B2 candidate summary rows: {len(b2_candidate_summary)}")
    print(f"  operational-vs-oracle delay rows: {len(delay_loss)}")
    print(f"  delay by origin/day rows: {len(delay_by_origin)}")
    print(f"  delay/error relation rows: {len(delay_relation)}")
    if panel["baseline_gross_usd"].isna().all():
        print("  note: no no-seat baseline was found; pass --baseline-csv or populate movie_day_estimates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
