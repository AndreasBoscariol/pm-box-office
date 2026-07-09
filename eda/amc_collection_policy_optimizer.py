#!/usr/bin/env python3
"""Optimize AMC theatre subset and collection timing under a request budget.

This EDA answers one operational question:

    Given a 20-requests/minute AMC limit, what theatre count, theatre subset,
    and snapshot timing policy gives the best live same-day forecast accuracy
    for the least collection delay?

The target is fixed across all candidate panels:

    Y_cumulative = full active-sample final EOD weighted occupied seats

When national daily actuals exist, the script also evaluates:

    Y_actual = national daily actual gross

Outputs are written to data/diagnostics/amc_collection_policy_optimizer.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from pm_box_office.db.connection import connect_database
from pm_box_office.sources.common.cli import add_database_arg


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "diagnostics" / "amc_collection_policy_optimizer"

DEFAULT_PANEL_SIZES = ("10", "15", "20", "30", "40", "50", "75", "100", "125", "150", "full_active")
DEFAULT_ORIGINS = ("10:00", "12:00", "14:00", "16:00", "18:00", "20:00")
DEFAULT_TIMING_POLICIES = (
    "pre_only",
    "early_post_only",
    "stable_post_only",
    "latest_available",
    "pre_plus_post",
)
DEFAULT_SELECTION_STRATEGIES = (
    "random_stratified",
    "top_volume",
    "top_low_latency",
    "top_reliability",
    "top_hybrid",
    "greedy_train_selected",
)
DEFAULT_ORIGIN_TZ = "America/New_York"
DEFAULT_REQUEST_LIMIT_PER_MINUTE = 20
DEFAULT_MIN_TRAIN_DAYS = 3
RANDOM_SEED = 20260708

TIMING_BUCKETS = {
    "pre": {"label": "pre", "target_minutes_after_start": -30, "min_after_start": -10_000, "max_after_start": 0},
    "early_post": {"label": "early_post", "target_minutes_after_start": 15, "min_after_start": 0, "max_after_start": 15},
    "stable_post": {"label": "stable_post", "target_minutes_after_start": 30, "min_after_start": 15, "max_after_start": 30},
}
POLICY_BUCKETS = {
    "pre_only": ("pre",),
    "early_post_only": ("early_post",),
    "stable_post_only": ("stable_post",),
    "latest_available": ("pre", "early_post", "stable_post"),
    "pre_plus_post": ("pre", "early_post", "stable_post"),
}


@dataclass(frozen=True)
class RunConfig:
    request_limit_per_minute: int
    panel_sizes: tuple[str, ...]
    origins: tuple[str, ...]
    timing_policies: tuple[str, ...]
    selection_strategies: tuple[str, ...]
    origin_timezone: str
    min_train_days: int
    output_dir: Path


def fetch_frame(conn: Any, sql: str, params: Iterable[Any] | None = None) -> pd.DataFrame:
    cursor = conn.execute(sql, params)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def relation_exists(conn: Any, relation_name: str) -> bool:
    row = conn.execute("SELECT to_regclass(%s)", (relation_name,)).fetchone()
    return bool(row and row[0])


def fetch_actuals(conn: Any) -> pd.DataFrame:
    if not relation_exists(conn, "daily_box_office"):
        return pd.DataFrame(columns=["movie_id", "exhibition_date", "actual_gross_usd"])
    return fetch_frame(
        conn,
        """
        SELECT
            rr.movie_id,
            dbo.box_office_date::date AS exhibition_date,
            dbo.gross_usd::double precision AS actual_gross_usd
        FROM daily_box_office dbo
        JOIN release_runs rr ON rr.release_run_id = dbo.release_run_id
        WHERE dbo.source = 'the_numbers'
          AND dbo.gross_usd > 0
        """,
    )


def fetch_theatres(conn: Any) -> pd.DataFrame:
    return fetch_frame(
        conn,
        """
        WITH active_sample AS (
            SELECT sample_set_id
            FROM amc_theatre_sample_sets
            WHERE status = 'active'
            ORDER BY sample_key
            LIMIT 1
        )
        SELECT
            t.amc_theatre_id,
            t.name AS theatre_name,
            t.state,
            t.timezone,
            t.inferred_screen_count,
            COALESCE(member.analysis_weight, 1.0)::double precision AS analysis_weight,
            (member.amc_theatre_id IS NOT NULL) AS in_active_sample
        FROM amc_theatres t
        LEFT JOIN active_sample ON TRUE
        LEFT JOIN amc_theatre_sample_members member
          ON member.sample_set_id = active_sample.sample_set_id
         AND member.amc_theatre_id = t.amc_theatre_id
        WHERE COALESCE(t.active, TRUE)
        """,
    )


def fetch_showtimes(conn: Any) -> pd.DataFrame:
    return fetch_frame(
        conn,
        """
        WITH capacity AS (
            SELECT showtime_id, MAX(total_seats)::double precision AS known_total_seats
            FROM amc_seat_snapshots
            GROUP BY showtime_id
        )
        SELECT
            s.showtime_id,
            s.movie_id,
            s.amc_movie_id,
            s.amc_movie_name,
            s.amc_theatre_id,
            s.exhibition_date,
            s.starts_at_utc,
            s.local_calendar_start_at,
            s.timezone,
            s.showtime_block,
            s.is_premium_format,
            COALESCE(capacity.known_total_seats, 0)::double precision AS known_total_seats
        FROM amc_showtimes s
        LEFT JOIN capacity ON capacity.showtime_id = s.showtime_id
        WHERE s.exhibition_date IS NOT NULL
          AND s.status = 'active'
        """,
    )


def fetch_snapshots(conn: Any) -> pd.DataFrame:
    return fetch_frame(
        conn,
        """
        SELECT
            ss.seat_snapshot_id,
            ss.showtime_id,
            ss.target_offset_minutes,
            ss.scheduled_for,
            ss.observed_at,
            ss.lateness_seconds,
            ss.total_seats::double precision AS total_seats,
            ss.filled_or_unavailable_seats::double precision AS filled_or_unavailable_seats,
            ss.parse_method
        FROM amc_seat_snapshots ss
        """,
    )


def normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "exhibition_date" in frame.columns:
        frame["exhibition_date"] = pd.to_datetime(frame["exhibition_date"], errors="coerce").dt.date
    for column in ["starts_at_utc", "scheduled_for", "observed_at", "local_calendar_start_at"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return frame


def parse_origin_time(origin: str) -> dt.time:
    hour, minute = origin.split(":", 1)
    return dt.time(int(hour), int(minute))


def origin_timestamp_utc(exhibition_date: dt.date, origin: str, timezone_name: str) -> pd.Timestamp:
    local_zone = ZoneInfo(timezone_name)
    local_dt = dt.datetime.combine(exhibition_date, parse_origin_time(origin), tzinfo=local_zone)
    return pd.Timestamp(local_dt.astimezone(dt.timezone.utc))


def add_snapshot_timing(showtimes: pd.DataFrame, snapshots: pd.DataFrame, theatres: pd.DataFrame) -> pd.DataFrame:
    frame = snapshots.merge(showtimes, on="showtime_id", how="inner", suffixes=("", "_show"))
    missing_theatre_columns = [
        column for column in ["state", "analysis_weight", "in_active_sample"] if column not in frame.columns
    ]
    if missing_theatre_columns:
        frame = frame.merge(
            theatres[["amc_theatre_id", *missing_theatre_columns]],
            on="amc_theatre_id",
            how="left",
        )
    frame["analysis_weight"] = pd.to_numeric(frame["analysis_weight"], errors="coerce").fillna(1.0)
    frame["in_active_sample"] = frame["in_active_sample"].fillna(False).astype(bool)
    frame["delay_minutes"] = (
        pd.to_datetime(frame["observed_at"], utc=True, errors="coerce")
        - pd.to_datetime(frame["scheduled_for"], utc=True, errors="coerce")
    ).dt.total_seconds() / 60.0
    frame["actual_minutes_from_start"] = (
        pd.to_datetime(frame["observed_at"], utc=True, errors="coerce")
        - pd.to_datetime(frame["starts_at_utc"], utc=True, errors="coerce")
    ).dt.total_seconds() / 60.0
    frame["weighted_filled"] = frame["filled_or_unavailable_seats"] * frame["analysis_weight"]
    frame["weighted_capacity"] = frame["total_seats"] * frame["analysis_weight"]
    frame["is_parse_success"] = ~frame["parse_method"].fillna("").str.lower().isin(["", "unknown", "failed"])
    frame["is_on_time"] = pd.to_numeric(frame["delay_minutes"], errors="coerce").le(5)
    return frame.replace([np.inf, -np.inf], np.nan)


def choose_bucket_snapshots(snapshots: pd.DataFrame, bucket: str) -> pd.DataFrame:
    spec = TIMING_BUCKETS[bucket]
    frame = snapshots.loc[
        snapshots["actual_minutes_from_start"].ge(spec["min_after_start"])
        & snapshots["actual_minutes_from_start"].lt(spec["max_after_start"])
    ].copy()
    if frame.empty:
        return frame
    frame["bucket"] = bucket
    target = float(spec["target_minutes_after_start"])
    frame["bucket_distance"] = (frame["actual_minutes_from_start"] - target).abs()
    frame = frame.sort_values(["showtime_id", "bucket_distance", "observed_at", "seat_snapshot_id"])
    return frame.drop_duplicates(["showtime_id"], keep="first")


def build_bucket_snapshot_table(snapshots: pd.DataFrame) -> pd.DataFrame:
    parts = [choose_bucket_snapshots(snapshots, bucket) for bucket in TIMING_BUCKETS]
    parts = [part for part in parts if not part.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def latest_final_snapshots(snapshots: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty:
        return snapshots.copy()
    frame = snapshots.sort_values(["showtime_id", "observed_at", "seat_snapshot_id"]).copy()
    return frame.drop_duplicates(["showtime_id"], keep="last")


def build_targets(showtimes: pd.DataFrame, snapshots: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    final = latest_final_snapshots(snapshots.loc[snapshots["in_active_sample"]].copy())
    if final.empty:
        raise RuntimeError("No final active-sample AMC snapshots found; run AMC seat collection before this EDA.")
    grouped = final.groupby(["movie_id", "amc_movie_id", "amc_movie_name", "exhibition_date"], dropna=False).agg(
        y_cumulative=("weighted_filled", "sum"),
        full_final_capacity=("weighted_capacity", "sum"),
        full_final_showtimes=("showtime_id", "count"),
        full_active_theatres=("amc_theatre_id", "nunique"),
    )
    targets = grouped.reset_index()
    schedule = (
        showtimes.loc[showtimes["in_active_sample"]]
        .groupby(["movie_id", "amc_movie_id", "exhibition_date"], dropna=False)
        .agg(
            scheduled_capacity=("weighted_known_total_seats", "sum"),
            scheduled_showtimes=("showtime_id", "count"),
            scheduled_theatres=("amc_theatre_id", "nunique"),
        )
        .reset_index()
    )
    targets = targets.merge(schedule, on=["movie_id", "amc_movie_id", "exhibition_date"], how="left")
    targets = targets.merge(actuals, on=["movie_id", "exhibition_date"], how="left")
    targets["day_of_week"] = pd.to_datetime(targets["exhibition_date"], errors="coerce").dt.day_name()
    return targets.loc[targets["y_cumulative"].gt(0)].reset_index(drop=True)


def prepare_showtimes(showtimes: pd.DataFrame, theatres: pd.DataFrame) -> pd.DataFrame:
    frame = showtimes.merge(
        theatres[["amc_theatre_id", "state", "analysis_weight", "in_active_sample"]],
        on="amc_theatre_id",
        how="left",
    )
    frame["analysis_weight"] = pd.to_numeric(frame["analysis_weight"], errors="coerce").fillna(1.0)
    frame["in_active_sample"] = frame["in_active_sample"].fillna(False).astype(bool)
    frame["known_total_seats"] = pd.to_numeric(frame["known_total_seats"], errors="coerce").fillna(0.0)
    frame["weighted_known_total_seats"] = frame["known_total_seats"] * frame["analysis_weight"]
    return frame


def rank_series(values: pd.Series, *, ascending: bool) -> pd.Series:
    return values.rank(method="average", ascending=ascending, na_option="bottom")


def theatre_train_stats(
    showtimes: pd.DataFrame,
    snapshots: pd.DataFrame,
    train_dates: set[dt.date],
) -> pd.DataFrame:
    show = showtimes.loc[showtimes["exhibition_date"].isin(train_dates) & showtimes["in_active_sample"]].copy()
    if show.empty:
        base = showtimes.loc[showtimes["in_active_sample"], ["amc_theatre_id", "state"]].drop_duplicates()
    else:
        base = show.groupby(["amc_theatre_id", "state"], dropna=False).agg(
            train_showtimes=("showtime_id", "count"),
            train_capacity=("weighted_known_total_seats", "sum"),
        ).reset_index()
    snap = snapshots.loc[snapshots["exhibition_date"].isin(train_dates) & snapshots["in_active_sample"]].copy()
    if not snap.empty:
        snap_stats = snap.groupby("amc_theatre_id", dropna=False).agg(
            train_filled=("weighted_filled", "sum"),
            parse_success=("is_parse_success", "mean"),
            on_time_rate=("is_on_time", "mean"),
            p90_delay=("delay_minutes", lambda x: float(np.nanpercentile(x.dropna(), 90)) if len(x.dropna()) else np.nan),
            median_delay=("delay_minutes", "median"),
        ).reset_index()
        base = base.merge(snap_stats, on="amc_theatre_id", how="left")
    for column in ["train_showtimes", "train_capacity", "train_filled", "parse_success", "on_time_rate", "p90_delay", "median_delay"]:
        if column not in base.columns:
            base[column] = np.nan
    base["volume"] = pd.to_numeric(base["train_filled"], errors="coerce").fillna(0.0) + pd.to_numeric(
        base["train_capacity"], errors="coerce"
    ).fillna(0.0) * 0.01
    base["parse_success"] = pd.to_numeric(base["parse_success"], errors="coerce").fillna(0.0)
    base["on_time_rate"] = pd.to_numeric(base["on_time_rate"], errors="coerce").fillna(0.0)
    base["p90_delay"] = pd.to_numeric(base["p90_delay"], errors="coerce")
    base["median_delay"] = pd.to_numeric(base["median_delay"], errors="coerce")
    worst_delay = base["p90_delay"].dropna().max()
    base["p90_delay"] = base["p90_delay"].fillna(worst_delay if np.isfinite(worst_delay) else 999.0)
    base["hybrid_score"] = (
        rank_series(base["volume"], ascending=True)
        + rank_series(base["parse_success"], ascending=True)
        + rank_series(base["on_time_rate"], ascending=True)
        - rank_series(base["p90_delay"], ascending=True)
    )
    return base


def greedy_train_selected(stats: pd.DataFrame, n: int) -> list[int]:
    if stats.empty:
        return []
    remaining = stats.sort_values("volume", ascending=False)["amc_theatre_id"].tolist()
    selected: list[int] = []
    # A lightweight greedy proxy: anchor with volume, then prefer theatres that
    # add reliability while diversifying state when possible.
    used_states: set[str] = set()
    state_by_theatre = stats.set_index("amc_theatre_id")["state"].fillna("unknown").astype(str).to_dict()
    score = stats.set_index("amc_theatre_id")["hybrid_score"].fillna(-np.inf).to_dict()
    while remaining and len(selected) < n:
        if not selected:
            choice = remaining[0]
        else:
            choice = max(
                remaining,
                key=lambda theatre_id: (
                    state_by_theatre.get(theatre_id, "unknown") not in used_states,
                    score.get(theatre_id, -np.inf),
                ),
            )
        selected.append(int(choice))
        used_states.add(state_by_theatre.get(choice, "unknown"))
        remaining.remove(choice)
    return selected


def select_theatres(stats: pd.DataFrame, strategy: str, n_label: str, *, seed: int) -> list[int]:
    universe = stats["amc_theatre_id"].dropna().astype(int).drop_duplicates().tolist()
    if n_label == "full_active":
        return universe
    n = min(int(n_label), len(universe))
    if n <= 0:
        return []
    if strategy == "top_volume":
        ordered = stats.sort_values(["volume", "amc_theatre_id"], ascending=[False, True])
        return ordered["amc_theatre_id"].astype(int).head(n).tolist()
    if strategy == "top_low_latency":
        ordered = stats.sort_values(["p90_delay", "median_delay", "volume"], ascending=[True, True, False])
        return ordered["amc_theatre_id"].astype(int).head(n).tolist()
    if strategy == "top_reliability":
        ordered = stats.sort_values(["parse_success", "on_time_rate", "p90_delay"], ascending=[False, False, True])
        return ordered["amc_theatre_id"].astype(int).head(n).tolist()
    if strategy == "top_hybrid":
        ordered = stats.sort_values(["hybrid_score", "volume"], ascending=[False, False])
        return ordered["amc_theatre_id"].astype(int).head(n).tolist()
    if strategy == "greedy_train_selected":
        return greedy_train_selected(stats, n)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    grouped = list(stats.groupby(stats["state"].fillna("unknown"), dropna=False))
    rng.shuffle(grouped)
    quota = max(1, math.ceil(n / max(len(grouped), 1)))
    for _, group in grouped:
        choices = group["amc_theatre_id"].dropna().astype(int).to_numpy()
        if len(choices):
            take = min(quota, len(choices), n - len(selected))
            selected.extend(rng.choice(choices, size=take, replace=False).astype(int).tolist())
        if len(selected) >= n:
            break
    if len(selected) < n:
        remaining = [theatre_id for theatre_id in universe if theatre_id not in set(selected)]
        take = min(n - len(selected), len(remaining))
        if take:
            selected.extend(rng.choice(np.array(remaining), size=take, replace=False).astype(int).tolist())
    return selected[:n]


def planned_tasks_for_policy(showtimes: pd.DataFrame, selected_theatres: list[int], policy: str) -> pd.DataFrame:
    show = showtimes.loc[showtimes["amc_theatre_id"].isin(selected_theatres)].copy()
    if show.empty:
        return pd.DataFrame()
    rows = []
    for bucket in POLICY_BUCKETS[policy]:
        spec = TIMING_BUCKETS[bucket]
        part = show.copy()
        part["target_bucket"] = bucket
        part["target_timing_policy"] = policy
        part["scheduled_for"] = part["starts_at_utc"] + pd.to_timedelta(spec["target_minutes_after_start"], unit="m")
        part["priority"] = {"stable_post": 0, "early_post": 1, "pre": 2}[bucket]
        rows.append(part)
    tasks = pd.concat(rows, ignore_index=True)
    keep = [
        "showtime_id",
        "movie_id",
        "amc_movie_id",
        "amc_theatre_id",
        "exhibition_date",
        "starts_at_utc",
        "known_total_seats",
        "analysis_weight",
        "target_bucket",
        "target_timing_policy",
        "scheduled_for",
        "priority",
    ]
    return tasks[keep]


def simulate_execution(tasks: pd.DataFrame, request_limit_per_minute: int) -> pd.DataFrame:
    if tasks.empty:
        return tasks.copy()
    frame = tasks.sort_values(["scheduled_for", "priority", "showtime_id", "target_bucket"]).reset_index(drop=True).copy()
    frame["queue_position"] = np.arange(len(frame))
    frame["minute_slot"] = frame["queue_position"] // request_limit_per_minute
    first_scheduled = frame["scheduled_for"].iloc[0]
    available_slot = 0
    used_in_slot = 0
    observed_slots: list[int] = []
    for scheduled_for in frame["scheduled_for"]:
        scheduled_slot = max(0, int(math.floor((scheduled_for - first_scheduled).total_seconds() / 60.0)))
        if scheduled_slot > available_slot:
            available_slot = scheduled_slot
            used_in_slot = 0
        observed_slots.append(available_slot)
        used_in_slot += 1
        if used_in_slot >= request_limit_per_minute:
            available_slot += 1
            used_in_slot = 0
    frame["simulated_observed_at"] = first_scheduled + pd.to_timedelta(observed_slots, unit="m")
    frame["simulated_delay_minutes"] = (frame["simulated_observed_at"] - frame["scheduled_for"]).dt.total_seconds() / 60.0
    return frame


def attach_bucket_signals(tasks: pd.DataFrame, bucket_snapshots: pd.DataFrame) -> pd.DataFrame:
    if tasks.empty:
        return tasks.copy()
    signal_cols = [
        "showtime_id",
        "bucket",
        "filled_or_unavailable_seats",
        "total_seats",
        "weighted_filled",
        "weighted_capacity",
        "seat_snapshot_id",
    ]
    available = bucket_snapshots[signal_cols].rename(columns={"bucket": "target_bucket"}).copy()
    out = tasks.merge(available, on=["showtime_id", "target_bucket"], how="left")
    out["has_signal"] = out["weighted_filled"].notna()
    out["signal_weighted_filled"] = pd.to_numeric(out["weighted_filled"], errors="coerce").fillna(0.0)
    out["signal_weighted_capacity"] = pd.to_numeric(out["weighted_capacity"], errors="coerce").fillna(0.0)
    return out


def collapse_latest_per_showtime(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    order = {"pre": 0, "early_post": 1, "stable_post": 2}
    out = frame.copy()
    out["bucket_order"] = out["target_bucket"].map(order).fillna(-1)
    out = out.sort_values(["showtime_id", "bucket_order", "simulated_observed_at"])
    return out.drop_duplicates(["showtime_id"], keep="last")


def aggregate_policy_signal(
    tasks: pd.DataFrame,
    targets: pd.DataFrame,
    mode: str,
    origin: str | None,
    origin_timezone: str,
    request_limit_per_minute: int,
) -> pd.DataFrame:
    if tasks.empty:
        return pd.DataFrame()
    frame = tasks.loc[tasks["has_signal"]].copy()
    if mode == "rolling":
        if origin is None:
            raise ValueError("origin is required for rolling mode")
        origin_by_date = {
            date_value: origin_timestamp_utc(date_value, origin, origin_timezone)
            for date_value in frame["exhibition_date"].dropna().unique()
        }
        frame["forecast_origin"] = origin
        frame["forecast_origin_utc"] = pd.to_datetime(
            frame["exhibition_date"].map(origin_by_date),
            utc=True,
            errors="coerce",
        )
        frame = frame.loc[frame["simulated_observed_at"].le(frame["forecast_origin_utc"])].copy()
    else:
        frame["forecast_origin"] = "EOD"
        frame["forecast_origin_utc"] = pd.NaT
    frame = collapse_latest_per_showtime(frame)
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby(["movie_id", "amc_movie_id", "exhibition_date", "forecast_origin"], dropna=False).agg(
        panel_signal=("signal_weighted_filled", "sum"),
        panel_capacity=("signal_weighted_capacity", "sum"),
        panel_showtimes=("showtime_id", "count"),
        panel_theatres=("amc_theatre_id", "nunique"),
        median_simulated_delay_minutes=("simulated_delay_minutes", "median"),
        p90_simulated_delay_minutes=(
            "simulated_delay_minutes",
            lambda x: float(np.nanpercentile(x.dropna(), 90)) if len(x.dropna()) else np.nan,
        ),
        latest_simulated_observed_at=("simulated_observed_at", "max"),
    ).reset_index()
    request_stats = tasks.groupby(["movie_id", "amc_movie_id", "exhibition_date"], dropna=False).agg(
        requests_required=("showtime_id", "count")
    ).reset_index()
    out = targets.merge(grouped, on=["movie_id", "amc_movie_id", "exhibition_date"], how="left")
    out = out.merge(request_stats, on=["movie_id", "amc_movie_id", "exhibition_date"], how="left")
    out["forecast_origin"] = out["forecast_origin"].fillna(origin if mode == "rolling" else "EOD")
    for column in ["panel_signal", "panel_capacity", "panel_showtimes", "panel_theatres", "requests_required"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    out["minutes_required_at_20rpm"] = out["requests_required"] / request_limit_per_minute
    out["coverage_of_full_target"] = out["panel_signal"] / out["y_cumulative"].replace(0, np.nan)
    out["pace_vs_full_target"] = out["panel_signal"] / out["y_cumulative"].replace(0, np.nan)
    out["share_available_by_origin"] = out["panel_showtimes"] / out["requests_required"].replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def median_ratio(train: pd.DataFrame, numerator: str, denominator: str) -> float:
    num = pd.to_numeric(train[numerator], errors="coerce")
    den = pd.to_numeric(train[denominator], errors="coerce")
    ratio = (num / den.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
    return float(ratio.median()) if len(ratio) else np.nan


def make_predictions(panel: pd.DataFrame, train_panel: pd.DataFrame, target: str) -> pd.DataFrame:
    out = panel.copy()
    if target == "Y_actual":
        target_col = "actual_gross_usd"
    else:
        target_col = "y_cumulative"
    schedule_ratio = median_ratio(train_panel, target_col, "scheduled_capacity")
    panel_ratio = median_ratio(train_panel.loc[train_panel["panel_signal"].gt(0)], "panel_signal", target_col)
    out["target"] = target
    out["target_value"] = pd.to_numeric(out[target_col], errors="coerce")
    out["y_hat_schedule"] = out["scheduled_capacity"] * schedule_ratio if np.isfinite(schedule_ratio) else np.nan
    out["fallback"] = out["panel_signal"].le(0) | ~np.isfinite(panel_ratio)
    out["y_hat_policy"] = np.where(
        out["fallback"],
        out["y_hat_schedule"],
        out["panel_signal"] / panel_ratio,
    )
    out["log_error_policy"] = np.log(out["y_hat_policy"] / out["target_value"].replace(0, np.nan))
    out["log_error_schedule"] = np.log(out["y_hat_schedule"] / out["target_value"].replace(0, np.nan))
    out["abs_log_error_policy"] = out["log_error_policy"].abs()
    out["abs_log_error_schedule"] = out["log_error_schedule"].abs()
    return out.replace([np.inf, -np.inf], np.nan)


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    keys = [
        "target",
        "mode",
        "timing_policy",
        "selection_strategy",
        "n_theatres",
        "forecast_origin",
        "day_of_week",
    ]
    rows = []
    for key, group in predictions.groupby(keys, dropna=False):
        err = group["log_error_policy"].dropna()
        sched = group["log_error_schedule"].dropna()
        rmse = float(np.sqrt(np.nanmean(np.square(err)))) if len(err) else np.nan
        schedule_rmse = float(np.sqrt(np.nanmean(np.square(sched)))) if len(sched) else np.nan
        rows.append(
            {
                **dict(zip(keys, key, strict=False)),
                "RMSE_log_target": rmse,
                "MAE_log_target": float(group["abs_log_error_policy"].mean()),
                "lift_vs_schedule_RMSE": (schedule_rmse - rmse) / schedule_rmse
                if np.isfinite(schedule_rmse) and schedule_rmse != 0 and np.isfinite(rmse)
                else np.nan,
                "fallback_rate": float(group["fallback"].mean()),
                "median_coverage_of_full_target": float(group["coverage_of_full_target"].median()),
                "median_pace_vs_full_target": float(group["pace_vs_full_target"].median()),
                "median_simulated_delay_minutes": float(group["median_simulated_delay_minutes"].median()),
                "p90_simulated_delay_minutes": float(group["p90_simulated_delay_minutes"].quantile(0.90)),
                "requests_required": float(group["requests_required"].median()),
                "minutes_required_at_20rpm": float(group["minutes_required_at_20rpm"].median()),
                "share_available_by_origin": float(group["share_available_by_origin"].median()),
                "n_test_movie_days": int(group[["movie_id", "amc_movie_id", "exhibition_date"]].drop_duplicates().shape[0]),
            }
        )
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def make_decisions(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    keys = ["target", "mode", "timing_policy", "selection_strategy", "forecast_origin", "day_of_week"]
    overall = (
        summary.groupby(keys + ["n_theatres"], dropna=False)
        .agg(
            median_RMSE=("RMSE_log_target", "median"),
            p75_RMSE=("RMSE_log_target", lambda x: float(x.quantile(0.75))),
            fallback_rate=("fallback_rate", "median"),
            p90_simulated_delay_minutes=("p90_simulated_delay_minutes", "median"),
            n_test_movie_days=("n_test_movie_days", "sum"),
        )
        .reset_index()
    )
    rows = []
    for key, group in overall.groupby(keys, dropna=False):
        full = group.loc[group["n_theatres"].astype(str).eq("full_active")]
        if full.empty:
            continue
        full_median = float(full["median_RMSE"].median())
        full_p75 = float(full["p75_RMSE"].median())
        candidates = group.copy()
        candidates["n_sort"] = candidates["n_theatres"].map(lambda x: 10_000 if str(x) == "full_active" else int(x))
        candidates = candidates.sort_values("n_sort")
        eligible = candidates.loc[
            candidates["median_RMSE"].le(full_median * 1.05)
            & candidates["p75_RMSE"].le(full_p75 * 1.10)
            & candidates["fallback_rate"].lt(0.25)
        ].copy()
        choice = eligible.head(1)
        base = dict(zip(keys, key, strict=False))
        if choice.empty:
            rows.append(
                {
                    **base,
                    "recommended_n_theatres": pd.NA,
                    "decision_status": "no_panel_met_rule",
                    "full_active_median_RMSE": full_median,
                    "full_active_p75_RMSE": full_p75,
                    "note": "High fallback or RMSE gap; policy is not identifiable at tested panel sizes.",
                }
            )
        else:
            row = choice.iloc[0]
            rows.append(
                {
                    **base,
                    "recommended_n_theatres": row["n_theatres"],
                    "decision_status": "meets_accuracy_and_fallback_rule",
                    "full_active_median_RMSE": full_median,
                    "full_active_p75_RMSE": full_p75,
                    "recommended_median_RMSE": row["median_RMSE"],
                    "recommended_p75_RMSE": row["p75_RMSE"],
                    "recommended_fallback_rate": row["fallback_rate"],
                    "recommended_p90_simulated_delay_minutes": row["p90_simulated_delay_minutes"],
                    "note": "",
                }
            )
    return pd.DataFrame(rows)


def evaluate(config: RunConfig, database_url: str | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conn = connect_database(database_url)
    try:
        theatres = normalize_dates(fetch_theatres(conn))
        showtimes = normalize_dates(fetch_showtimes(conn))
        raw_snapshots = normalize_dates(fetch_snapshots(conn))
        actuals = normalize_dates(fetch_actuals(conn))
    finally:
        conn.close()

    if theatres.empty or showtimes.empty or raw_snapshots.empty:
        raise RuntimeError("AMC theatres, showtimes, and seat snapshots are required for this optimizer.")

    showtimes = prepare_showtimes(showtimes, theatres)
    snapshots = add_snapshot_timing(showtimes, raw_snapshots, theatres)
    bucket_snapshots = build_bucket_snapshot_table(snapshots)
    targets = build_targets(showtimes, snapshots, actuals)
    test_dates = sorted(targets["exhibition_date"].dropna().unique())
    if len(test_dates) <= config.min_train_days:
        raise RuntimeError(f"Need more than {config.min_train_days} AMC collection dates for rolling evaluation.")

    all_predictions = []
    panel_records = []
    for date_index, test_date in enumerate(test_dates):
        train_dates = {date_value for date_value in test_dates if date_value < test_date}
        if len(train_dates) < config.min_train_days:
            continue
        stats = theatre_train_stats(showtimes, snapshots, train_dates)
        test_targets = targets.loc[targets["exhibition_date"].eq(test_date)].copy()
        train_targets = targets.loc[targets["exhibition_date"].isin(train_dates)].copy()
        for strategy in config.selection_strategies:
            for n_label in config.panel_sizes:
                selected = select_theatres(stats, strategy, n_label, seed=RANDOM_SEED + date_index)
                if not selected:
                    continue
                n_actual = "full_active" if n_label == "full_active" else str(len(selected))
                fold_dates = set(train_dates)
                fold_dates.add(test_date)
                show_window = showtimes.loc[showtimes["exhibition_date"].isin(fold_dates)].copy()
                target_window = targets.loc[targets["exhibition_date"].isin(fold_dates)].copy()
                for policy in config.timing_policies:
                    tasks = planned_tasks_for_policy(show_window, selected, policy)
                    tasks = simulate_execution(tasks, config.request_limit_per_minute)
                    tasks = attach_bucket_signals(tasks, bucket_snapshots)
                    for mode, origins in [("cumulative", ("EOD",)), ("rolling", config.origins)]:
                        for origin in origins:
                            panel = aggregate_policy_signal(
                                tasks,
                                target_window,
                                mode,
                                None if mode == "cumulative" else origin,
                                config.origin_timezone,
                                config.request_limit_per_minute,
                            )
                            if panel.empty:
                                continue
                            panel["mode"] = mode
                            panel["timing_policy"] = policy
                            panel["selection_strategy"] = strategy
                            panel["n_theatres"] = n_actual
                            panel["selected_theatres"] = len(selected)
                            train_panel = panel.loc[panel["exhibition_date"].isin(train_dates)].copy()
                            test_panel = panel.loc[panel["exhibition_date"].eq(test_date)].copy()
                            if train_panel.empty or test_panel.empty:
                                continue
                            targets_to_score = ["Y_cumulative"]
                            if test_panel["actual_gross_usd"].notna().any() and train_panel["actual_gross_usd"].notna().any():
                                targets_to_score.append("Y_actual")
                            for target in targets_to_score:
                                if target == "Y_actual":
                                    train_score = train_panel.loc[train_panel["actual_gross_usd"].gt(0)].copy()
                                    test_score = test_panel.loc[test_panel["actual_gross_usd"].gt(0)].copy()
                                else:
                                    train_score = train_panel
                                    test_score = test_panel
                                if train_score.empty or test_score.empty:
                                    continue
                                pred = make_predictions(test_score, train_score, target)
                                all_predictions.append(pred)
                            panel_records.append(
                                {
                                    "test_date": test_date,
                                    "timing_policy": policy,
                                    "selection_strategy": strategy,
                                    "n_theatres": n_actual,
                                    "selected_theatre_ids": "|".join(map(str, sorted(selected))),
                                }
                            )
    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    summary = summarize_predictions(predictions)
    decisions = make_decisions(summary)
    panels = pd.DataFrame(panel_records).drop_duplicates() if panel_records else pd.DataFrame()
    return predictions, summary, decisions


def write_outputs(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    decisions: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "policy_predictions.csv", index=False)
    summary.to_csv(output_dir / "policy_metrics.csv", index=False)
    decisions.to_csv(output_dir / "policy_decisions.csv", index=False)
    fallback = summary.loc[summary["fallback_rate"].ge(0.25)].copy() if not summary.empty else pd.DataFrame()
    fallback.to_csv(output_dir / "high_fallback_policies.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_database_arg(parser)
    parser.add_argument("--request-limit-per-minute", type=int, default=DEFAULT_REQUEST_LIMIT_PER_MINUTE)
    parser.add_argument("--panel-sizes", nargs="+", default=list(DEFAULT_PANEL_SIZES))
    parser.add_argument("--origins", nargs="+", default=list(DEFAULT_ORIGINS))
    parser.add_argument("--timing-policies", nargs="+", default=list(DEFAULT_TIMING_POLICIES), choices=list(POLICY_BUCKETS))
    parser.add_argument("--selection-strategies", nargs="+", default=list(DEFAULT_SELECTION_STRATEGIES), choices=list(DEFAULT_SELECTION_STRATEGIES))
    parser.add_argument("--origin-timezone", default=DEFAULT_ORIGIN_TZ)
    parser.add_argument("--min-train-days", type=int, default=DEFAULT_MIN_TRAIN_DAYS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfig(
        request_limit_per_minute=args.request_limit_per_minute,
        panel_sizes=tuple(args.panel_sizes),
        origins=tuple(args.origins),
        timing_policies=tuple(args.timing_policies),
        selection_strategies=tuple(args.selection_strategies),
        origin_timezone=args.origin_timezone,
        min_train_days=args.min_train_days,
        output_dir=args.output_dir,
    )
    predictions, summary, decisions = evaluate(config, args.database_url)
    write_outputs(predictions, summary, decisions, config.output_dir)
    print(f"Wrote {len(predictions):,} prediction rows to {config.output_dir / 'policy_predictions.csv'}")
    print(f"Wrote {len(summary):,} metric rows to {config.output_dir / 'policy_metrics.csv'}")
    print(f"Wrote {len(decisions):,} decision rows to {config.output_dir / 'policy_decisions.csv'}")
    if not summary.empty and summary["fallback_rate"].ge(0.25).any():
        print("Some policies have fallback_rate >= 25%; see high_fallback_policies.csv.")


if __name__ == "__main__":
    main()
