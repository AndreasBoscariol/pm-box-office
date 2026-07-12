#!/usr/bin/env python3
"""Rolling diagnostic for Thursday AMC preview-gross nowcasts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.boxoffice.amc_features import (  # noqa: E402
    aggregate_as_of_features,
    aggregate_final_eod_features,
    build_origin_grid,
    fetch_actuals,
    fetch_preview_actuals,
    fetch_schedule,
    fetch_snapshots,
    latest_snapshots_as_of,
    latest_snapshots_final,
)
from models.boxoffice.thursday_amc_candidates import (  # noqa: E402
    CANDIDATES,
    release_week_rolling_predictions,
)
from models.boxoffice.thursday_amc_transfer import (  # noqa: E402
    donor_pool_comparison,
    donor_bias_by_origin,
    donor_day_leave_one_out,
    donor_day_metrics,
    donor_nationalization_ratio,
    fit_pooled_transfer,
    predict_transfer_row,
    temporal_deployment_predictions,
    transfer_sample_counts,
)
from models.boxoffice.artifacts import DEFAULT_ARTIFACT_ROOT, load_model_artifacts, write_json  # noqa: E402
from models.boxoffice.thursday_amc_preview import (  # noqa: E402
    THURSDAY_PREVIEW_ORIGINS,
    origin_bucket,
    positive_float,
    preview_training_frame,
    predict_preview_distribution,
    select_preview_training_pool,
    update_ow_distribution,
)
from models.boxoffice.thursday_preview import update_ow_prior_from_reported_preview  # noqa: E402
from pm_box_office.db.connection import connect_database  # noqa: E402
from pm_box_office.sources.common.cli import add_database_arg  # noqa: E402


DIAGNOSTICS_DIR = REPO_ROOT / "data" / "diagnostics"
DEFAULT_MIN_TRAIN_ROWS = 10
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260711
POINT_MODELS = ("multiplicative", "additive", "hybrid")
REPLAY_DATES = ("2026-07-02", "2026-07-09")


def build_preview_panel_from_frames(
    *,
    schedule: pd.DataFrame,
    snapshots: pd.DataFrame,
    actuals: pd.DataFrame,
    origins: tuple[str, ...] = THURSDAY_PREVIEW_ORIGINS,
    origin_timezone: str = "America/New_York",
) -> pd.DataFrame:
    if schedule.empty or snapshots.empty or actuals.empty:
        return empty_panel()
    schedule = schedule.copy()
    snapshots = snapshots.copy()
    actuals = actuals.copy()
    for frame in [schedule, snapshots, actuals]:
        if "exhibition_date" in frame.columns:
            frame["exhibition_date"] = pd.to_datetime(frame["exhibition_date"], errors="coerce").dt.date
    actuals["is_preview"] = pd.to_numeric(actuals.get("is_preview", 0), errors="coerce").fillna(0).astype(int)
    primary = actuals.get("is_primary_training_target", pd.Series(True, index=actuals.index)).fillna(False).astype(bool)
    preview_actuals = actuals.loc[
        actuals["is_preview"].eq(1)
        & pd.to_numeric(actuals["actual_gross_usd"], errors="coerce").gt(0)
        & primary
    ].copy()
    if preview_actuals.empty:
        return empty_panel()

    preview_schedule = schedule.merge(
        preview_actuals[[column for column in [
            "release_run_id", "movie_id", "title", "release_date", "exhibition_date",
            "actual_gross_usd", "actual_theaters", "is_preview", "preview_target_type",
            "includes_wednesday_early_access", "preview_days_included", "preview_source",
            "preview_published_at", "received_at", "is_primary_training_target", "is_wide_release",
        ] if column in preview_actuals]],
        on=["movie_id", "exhibition_date"],
        how="inner",
    )
    if preview_schedule.empty:
        return empty_panel()
    grid = build_origin_grid(preview_schedule, origins, origin_timezone)
    asof = latest_snapshots_as_of(snapshots, grid)
    features = aggregate_as_of_features(asof)
    final_features = aggregate_final_eod_features(latest_snapshots_final(snapshots))
    panel = grid.merge(preview_schedule, on=["movie_id", "amc_movie_id", "exhibition_date"], how="left")
    panel = panel.merge(
        features,
        on=["movie_id", "amc_movie_id", "exhibition_date", "forecast_origin", "forecast_origin_utc"],
        how="left",
    )
    panel = panel.merge(final_features, on=["movie_id", "amc_movie_id", "exhibition_date"], how="left")
    panel["opening_weekend_start"] = pd.to_datetime(panel["exhibition_date"]) + pd.Timedelta(days=1)
    return finalize_preview_panel(panel)


def finalize_preview_panel(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return empty_panel()
    out = panel.copy()
    for column in [
        "s_obs", "c_obs", "s_final_eod", "c_final_eod", "actual_gross_usd",
        "n_snapshots", "coverage", "delay_p50_minutes", "staleness_p50_minutes",
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.loc[
        out["s_obs"].gt(0)
        & out["s_final_eod"].gt(0)
        & out["actual_gross_usd"].gt(0)
    ].copy()
    out["opening_weekend_start"] = pd.to_datetime(out["opening_weekend_start"], errors="coerce")
    out["release_year"] = out["opening_weekend_start"].dt.year
    out["origin_bucket"] = out["forecast_origin"].map(origin_bucket)
    return out.sort_values(["opening_weekend_start", "movie_id", "forecast_origin"]).reset_index(drop=True)


def empty_panel() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "release_run_id", "movie_id", "title", "release_date", "exhibition_date",
            "opening_weekend_start", "release_year", "forecast_origin", "forecast_origin_utc",
            "origin_bucket", "s_obs", "c_obs", "s_final_eod", "c_final_eod",
            "actual_gross_usd", "actual_theaters", "n_snapshots", "coverage",
            "delay_p50_minutes", "staleness_p50_minutes", "is_preview",
        ]
    )


def build_daily_donor_panel_from_frames(
    *, schedule: pd.DataFrame, snapshots: pd.DataFrame, actuals: pd.DataFrame,
    origins: tuple[str, ...] = THURSDAY_PREVIEW_ORIGINS,
    origin_timezone: str = "America/New_York",
) -> pd.DataFrame:
    if schedule.empty or snapshots.empty or actuals.empty:
        return empty_panel()
    schedule, snapshots, actuals = schedule.copy(), snapshots.copy(), actuals.copy()
    for frame in (schedule, snapshots, actuals):
        frame["exhibition_date"] = pd.to_datetime(frame["exhibition_date"], errors="coerce").dt.date
    daily = actuals.loc[
        pd.to_numeric(actuals.get("is_preview", 0), errors="coerce").fillna(0).eq(0)
        & pd.to_numeric(actuals["actual_gross_usd"], errors="coerce").gt(0)
    ].copy()
    joined_schedule = schedule.merge(
        daily[[column for column in ["release_run_id", "movie_id", "title", "release_date", "exhibition_date", "actual_gross_usd", "actual_theaters", "is_preview"] if column in daily]],
        on=["movie_id", "exhibition_date"], how="inner",
    )
    if joined_schedule.empty:
        return empty_panel()
    grid = build_origin_grid(joined_schedule, origins, origin_timezone)
    features = aggregate_as_of_features(latest_snapshots_as_of(snapshots, grid))
    final_features = aggregate_final_eod_features(latest_snapshots_final(snapshots))
    panel = grid.merge(joined_schedule, on=["movie_id", "amc_movie_id", "exhibition_date"], how="left")
    panel = panel.merge(features, on=["movie_id", "amc_movie_id", "exhibition_date", "forecast_origin", "forecast_origin_utc"], how="left")
    panel = panel.merge(final_features, on=["movie_id", "amc_movie_id", "exhibition_date"], how="left")
    panel["opening_weekend_start"] = pd.to_datetime(panel["exhibition_date"])
    panel["day_of_week"] = pd.to_datetime(panel["exhibition_date"]).dt.day_name()
    return finalize_preview_panel(panel)


def build_all_day_completion_panel_from_frames(
    *, schedule: pd.DataFrame, snapshots: pd.DataFrame,
    origins: tuple[str, ...] = THURSDAY_PREVIEW_ORIGINS,
    origin_timezone: str = "America/New_York",
) -> pd.DataFrame:
    if schedule.empty or snapshots.empty:
        return empty_panel()
    schedule, snapshots = schedule.copy(), snapshots.copy()
    for frame in (schedule, snapshots):
        frame["exhibition_date"] = pd.to_datetime(frame["exhibition_date"], errors="coerce").dt.date
    grid = build_origin_grid(schedule, origins, origin_timezone)
    features = aggregate_as_of_features(latest_snapshots_as_of(snapshots, grid))
    final_features = aggregate_final_eod_features(latest_snapshots_final(snapshots))
    panel = grid.merge(schedule, on=["movie_id", "amc_movie_id", "exhibition_date"], how="left")
    panel = panel.merge(features, on=["movie_id", "amc_movie_id", "exhibition_date", "forecast_origin", "forecast_origin_utc"], how="left")
    panel = panel.merge(final_features, on=["movie_id", "amc_movie_id", "exhibition_date"], how="left")
    panel["actual_gross_usd"] = np.nan
    panel["opening_weekend_start"] = pd.to_datetime(panel["exhibition_date"])
    panel["day_of_week"] = pd.to_datetime(panel["exhibition_date"]).dt.day_name()
    out = panel.copy()
    for column in ["s_obs", "s_final_eod", "c_obs", "c_final_eod", "n_snapshots", "coverage"]:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.loc[out["s_obs"].gt(0) & out["s_final_eod"].gt(0)].copy()
    out["release_year"] = out["opening_weekend_start"].dt.year
    out["origin_bucket"] = out["forecast_origin"].map(origin_bucket)
    return out.sort_values(["exhibition_date", "movie_id", "forecast_origin"]).reset_index(drop=True)


def build_thursday_transfer_shadow_panel(
    *, schedule: pd.DataFrame, snapshots: pd.DataFrame, policy: dict[str, Any],
    origins: tuple[str, ...] = THURSDAY_PREVIEW_ORIGINS,
    origin_timezone: str = "America/New_York",
) -> pd.DataFrame:
    if schedule.empty or snapshots.empty or not policy.get("shadow_forecast_enabled"):
        return pd.DataFrame()
    schedule, snapshots = schedule.copy(), snapshots.copy()
    for frame in (schedule, snapshots):
        frame["exhibition_date"] = pd.to_datetime(frame["exhibition_date"], errors="coerce").dt.date
    thursday = schedule.loc[pd.to_datetime(schedule["exhibition_date"]).dt.day_name().eq("Thursday")].copy()
    if thursday.empty:
        return pd.DataFrame()
    grid = build_origin_grid(thursday, origins, origin_timezone)
    features = aggregate_as_of_features(latest_snapshots_as_of(snapshots, grid))
    final = aggregate_final_eod_features(latest_snapshots_final(snapshots))
    panel = grid.merge(thursday, on=["movie_id", "amc_movie_id", "exhibition_date"], how="left")
    panel = panel.merge(features, on=["movie_id", "amc_movie_id", "exhibition_date", "forecast_origin", "forecast_origin_utc"], how="left")
    panel = panel.merge(final, on=["movie_id", "amc_movie_id", "exhibition_date"], how="left")
    rows = []
    for _, target in panel.loc[pd.to_numeric(panel.get("s_obs"), errors="coerce").gt(0)].iterrows():
        prediction = predict_transfer_row(target, policy)
        if prediction is None:
            continue
        distribution = predict_preview_distribution(
            point_preview_gross_usd=prediction["pred_gross_usd"], forecast_origin=str(target["forecast_origin"]),
            policy=policy, draws=2_000, seed=int(float(target["movie_id"])),
        )
        rows.append({
            **target.to_dict(), "model": "TH_TRANSFER_V0", "status": "shadow_only",
            "interval_label": "transfer_uncertainty_envelope",
            "predicted_eod_amc_seats": prediction["pred_final_eod_seats"],
            "predicted_preview_gross_usd": prediction["pred_gross_usd"],
            "preview_lo80_usd": distribution.get("lo80_usd"), "preview_hi80_usd": distribution.get("hi80_usd"),
            "preview_lo95_usd": distribution.get("lo95_usd"), "preview_hi95_usd": distribution.get("hi95_usd"),
            "residual_pool_scope": distribution.get("residual_pool_scope"),
        })
    return pd.DataFrame(rows)


def build_transfer_ow_impact(
    shadow: pd.DataFrame, *, daily_baseline: pd.DataFrame,
    transfer_policy: dict[str, Any], preview_update_policy: dict[str, Any],
) -> pd.DataFrame:
    if shadow.empty or daily_baseline.empty:
        return pd.DataFrame()
    base = daily_baseline.copy()
    base["baseline_ow_usd"] = base[["pre_fri_usd", "pre_sat_usd", "pre_sun_usd"]].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=3)
    keys = [key for key in ["movie_id", "release_run_id"] if key in shadow and key in base]
    if not keys:
        return pd.DataFrame()
    joined = shadow.merge(base[keys + ["baseline_ow_usd"]].drop_duplicates(keys), on=keys, how="left")
    rows = []
    for _, row in joined.loc[joined["baseline_ow_usd"].gt(0)].iterrows():
        preview = predict_preview_distribution(
            point_preview_gross_usd=row["predicted_preview_gross_usd"], forecast_origin=str(row["forecast_origin"]),
            policy=transfer_policy, draws=2_000, seed=int(float(row["movie_id"])),
        )
        ow = update_ow_distribution(baseline_ow_usd=float(row["baseline_ow_usd"]), preview_distribution=preview, preview_update_policy=preview_update_policy)
        rows.append({
            "movie_id": row["movie_id"], "exhibition_date": row["exhibition_date"], "forecast_origin": row["forecast_origin"],
            "baseline_ow_usd": row["baseline_ow_usd"], "transfer_preview_usd": row["predicted_preview_gross_usd"],
            "shadow_updated_ow_usd": ow.get("point_usd"), "shadow_ow_lo80_usd": ow.get("lo80_usd"),
            "shadow_ow_hi80_usd": ow.get("hi80_usd"), "shadow_ow_lo95_usd": ow.get("lo95_usd"),
            "shadow_ow_hi95_usd": ow.get("hi95_usd"), "production_ow_update_enabled": False,
        })
    return pd.DataFrame(rows)


def actual_preview_targets(actuals: pd.DataFrame) -> pd.DataFrame:
    if actuals.empty:
        return pd.DataFrame()
    work = actuals.copy()
    if "exhibition_date" in work:
        work["exhibition_date"] = pd.to_datetime(work["exhibition_date"], errors="coerce").dt.date
    if "preview_business_date" in work:
        work["exhibition_date"] = pd.to_datetime(work["preview_business_date"], errors="coerce").dt.date
    if "preview_gross_usd" in work:
        work["actual_preview_gross_usd"] = pd.to_numeric(work["preview_gross_usd"], errors="coerce")
    elif "actual_gross_usd" in work:
        work["actual_preview_gross_usd"] = pd.to_numeric(work["actual_gross_usd"], errors="coerce")
    else:
        return pd.DataFrame()
    if "is_primary_training_target" in work:
        work = work.loc[work["is_primary_training_target"].fillna(False).astype(bool)].copy()
    return work.loc[work["actual_preview_gross_usd"].gt(0)].copy()


def opening_eve_daily_proxy_targets(
    daily_actuals: pd.DataFrame,
    shadow: pd.DataFrame,
    *,
    daily_baseline: pd.DataFrame,
) -> pd.DataFrame:
    """Strictly use The Numbers Thursday daily gross only on opening eve."""
    if daily_actuals.empty or shadow.empty:
        return pd.DataFrame()
    replay_dates = {pd.Timestamp(value).date() for value in REPLAY_DATES}
    movie_dates = shadow[["movie_id", "exhibition_date"]].drop_duplicates().copy()
    movie_dates["exhibition_date"] = pd.to_datetime(movie_dates["exhibition_date"], errors="coerce").dt.date
    base = daily_baseline.copy()
    opening_col = "opening_weekend_start" if "opening_weekend_start" in base else "friday_date" if "friday_date" in base else None
    if opening_col is None:
        return pd.DataFrame()
    base["opening_weekend_start"] = pd.to_datetime(base[opening_col], errors="coerce").dt.date
    base["preview_business_date"] = (pd.to_datetime(base["opening_weekend_start"]) - pd.Timedelta(days=1)).dt.date
    opening_eve = base[["release_run_id", "movie_id", "opening_weekend_start", "preview_business_date"]].drop_duplicates()
    work = daily_actuals.copy()
    work["exhibition_date"] = pd.to_datetime(work["exhibition_date"], errors="coerce").dt.date
    work = work.loc[
        work["exhibition_date"].isin(replay_dates)
        & pd.to_numeric(work["actual_gross_usd"], errors="coerce").gt(0)
    ].copy()
    if work.empty:
        return pd.DataFrame()
    joined = work.merge(movie_dates, on=["movie_id", "exhibition_date"], how="inner")
    joined = joined.merge(opening_eve, on=["release_run_id", "movie_id"], how="inner")
    joined = joined.loc[joined["exhibition_date"].eq(joined["preview_business_date"])].copy()
    if joined.empty:
        return pd.DataFrame()
    joined["actual_preview_gross_usd"] = pd.to_numeric(joined["actual_gross_usd"], errors="coerce")
    joined["opening_thursday_daily_gross_usd"] = joined["actual_preview_gross_usd"]
    joined["preview_gross_usd"] = joined["actual_preview_gross_usd"]
    joined["preview_target_type"] = "opening_thursday_daily_gross"
    joined["includes_wednesday_early_access"] = False
    joined["preview_days_included"] = 1
    joined["preview_source"] = joined.get("source", "the_numbers").fillna("the_numbers").astype(str) + "_opening_thursday_daily"
    if "fetched_at" in joined:
        joined["preview_published_at"] = pd.to_datetime(joined["fetched_at"], errors="coerce", utc=True)
        joined["received_at"] = joined["preview_published_at"]
    else:
        joined["preview_published_at"] = pd.NaT
        joined["received_at"] = pd.NaT
    joined["is_primary_training_target"] = True
    joined["is_wide_release"] = True
    return joined


def thursday_daily_nowcast_proxy_diagnostic(
    shadow: pd.DataFrame,
    *,
    daily_actuals: pd.DataFrame,
) -> pd.DataFrame:
    if shadow.empty or daily_actuals.empty:
        return pd.DataFrame()
    out = shadow.copy()
    out["exhibition_date"] = pd.to_datetime(out["exhibition_date"], errors="coerce").dt.date
    daily = daily_actuals.copy()
    daily["exhibition_date"] = pd.to_datetime(daily["exhibition_date"], errors="coerce").dt.date
    daily = daily.loc[pd.to_datetime(daily["exhibition_date"], errors="coerce").dt.day_name().eq("Thursday")].copy()
    daily["actual_thursday_daily_gross_usd"] = pd.to_numeric(daily["actual_gross_usd"], errors="coerce")
    joined = out.merge(
        daily[["release_run_id", "movie_id", "title", "exhibition_date", "actual_thursday_daily_gross_usd", "source", "fetched_at"]],
        on=["movie_id", "exhibition_date"],
        how="inner",
        suffixes=("", "_daily"),
    )
    if joined.empty:
        return pd.DataFrame()
    joined["target_type"] = "ordinary_thursday_daily_actual"
    joined["eligible_for_preview_ow"] = False
    joined["daily_total_log_error"] = np.log(joined["actual_thursday_daily_gross_usd"] / joined["predicted_preview_gross_usd"])
    return joined


def preview_proxy_identity_audit(canonical_preview_actuals: pd.DataFrame, daily_actuals: pd.DataFrame, daily_baseline: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "movie_id", "release_run_id", "wide_release_date", "preview_business_date",
        "canonical_preview_gross", "the_numbers_thursday_gross", "ratio",
        "absolute_difference", "preview_target_type", "includes_early_access",
        "eligibility_status", "exclusion_reason",
    ]
    canonical = actual_preview_targets(canonical_preview_actuals)
    if canonical.empty or daily_actuals.empty or daily_baseline.empty:
        return pd.DataFrame(columns=columns)
    base = daily_baseline.copy()
    opening_col = "opening_weekend_start" if "opening_weekend_start" in base else "friday_date" if "friday_date" in base else None
    if opening_col is None:
        return pd.DataFrame(columns=columns)
    base["wide_release_date"] = pd.to_datetime(base[opening_col], errors="coerce").dt.date
    base["preview_business_date"] = (pd.to_datetime(base["wide_release_date"]) - pd.Timedelta(days=1)).dt.date
    base = base[["release_run_id", "movie_id", "wide_release_date", "preview_business_date"]].drop_duplicates()
    daily = daily_actuals.copy()
    daily["preview_business_date"] = pd.to_datetime(daily["exhibition_date"], errors="coerce").dt.date
    daily["the_numbers_thursday_gross"] = pd.to_numeric(daily["actual_gross_usd"], errors="coerce")
    canonical["preview_business_date"] = pd.to_datetime(canonical["exhibition_date"], errors="coerce").dt.date
    joined = canonical.merge(base, on=["release_run_id", "movie_id", "preview_business_date"], how="inner")
    joined = joined.merge(
        daily[["release_run_id", "movie_id", "preview_business_date", "the_numbers_thursday_gross"]],
        on=["release_run_id", "movie_id", "preview_business_date"],
        how="left",
    )
    joined["canonical_preview_gross"] = pd.to_numeric(joined["actual_preview_gross_usd"], errors="coerce")
    joined["ratio"] = joined["the_numbers_thursday_gross"] / joined["canonical_preview_gross"]
    joined["absolute_difference"] = (joined["the_numbers_thursday_gross"] - joined["canonical_preview_gross"]).abs()
    joined["includes_early_access"] = joined.get("includes_wednesday_early_access", False)
    joined["eligibility_status"] = np.where(
        joined["canonical_preview_gross"].gt(0) & joined["the_numbers_thursday_gross"].gt(0),
        "joined",
        "excluded",
    )
    joined["exclusion_reason"] = np.where(joined["eligibility_status"].eq("joined"), None, "missing_canonical_or_the_numbers_gross")
    return joined[columns]


def _actual_preview_columns(preview_actuals: pd.DataFrame) -> list[str]:
    preferred = [
        "release_run_id", "movie_id", "title", "exhibition_date", "actual_preview_gross_usd",
        "preview_target_type", "includes_wednesday_early_access", "preview_days_included",
        "preview_source", "preview_published_at", "received_at", "is_primary_training_target",
        "is_wide_release",
    ]
    return [column for column in preferred if column in preview_actuals]


def join_frozen_shadow_with_actuals(
    shadow: pd.DataFrame,
    *,
    preview_actuals: pd.DataFrame,
    daily_actuals: pd.DataFrame | None = None,
    daily_baseline: pd.DataFrame,
    preview_update_policy: dict[str, Any],
    transfer_policy: dict[str, Any],
) -> pd.DataFrame:
    if shadow.empty:
        return pd.DataFrame()
    targets = actual_preview_targets(preview_actuals)
    proxy = opening_eve_daily_proxy_targets(
        daily_actuals if daily_actuals is not None else pd.DataFrame(),
        shadow,
        daily_baseline=daily_baseline,
    )
    if not proxy.empty:
        if targets.empty:
            targets = proxy
        else:
            targets = pd.concat([targets, proxy], ignore_index=True)
            targets["_target_rank"] = np.where(targets.get("preview_target_type").eq("opening_thursday_daily_gross"), 1, 0)
            targets = targets.sort_values("_target_rank").drop_duplicates(["movie_id", "exhibition_date"], keep="first").drop(columns=["_target_rank"])
    if targets.empty:
        return pd.DataFrame()
    out = shadow.copy()
    out["exhibition_date"] = pd.to_datetime(out["exhibition_date"], errors="coerce").dt.date
    targets["exhibition_date"] = pd.to_datetime(targets["exhibition_date"], errors="coerce").dt.date
    replay_dates = {pd.Timestamp(value).date() for value in REPLAY_DATES}
    out = out.loc[out["exhibition_date"].isin(replay_dates)].copy()
    if out.empty:
        return pd.DataFrame()
    joined = out.merge(
        targets[_actual_preview_columns(targets)].drop_duplicates(["movie_id", "exhibition_date"]),
        on=["movie_id", "exhibition_date"],
        how="inner",
        suffixes=("", "_actual"),
    )
    if joined.empty:
        return pd.DataFrame()
    base = daily_baseline.copy()
    for column in ["pre_fri_usd", "pre_sat_usd", "pre_sun_usd", "actual_ow_usd", "total_forecast_usd"]:
        if column in base:
            base[column] = pd.to_numeric(base[column], errors="coerce")
    base["baseline_ow_usd"] = base[["pre_fri_usd", "pre_sat_usd", "pre_sun_usd"]].sum(axis=1, min_count=3)
    baseline_cols = [column for column in [
        "release_run_id", "movie_id", "baseline_ow_usd", "actual_ow_usd",
        "forecast_origin_date", "total_forecast_usd",
    ] if column in base]
    joined = joined.merge(base[baseline_cols].drop_duplicates(["release_run_id", "movie_id"]), on=["release_run_id", "movie_id"], how="left")
    for column in [
        "predicted_eod_amc_seats", "s_final_eod", "actual_preview_gross_usd",
        "predicted_preview_gross_usd", "baseline_ow_usd", "actual_ow_usd",
    ]:
        if column in joined:
            joined[column] = pd.to_numeric(joined[column], errors="coerce")
    stage2 = transfer_policy.get("stage2") if isinstance(transfer_policy, dict) else None
    joined["oracle_preview_gross_usd"] = np.nan
    if isinstance(stage2, dict):
        alpha = float(stage2.get("alpha", np.nan))
        beta = float(stage2.get("beta", np.nan))
        if np.isfinite(alpha) and np.isfinite(beta):
            joined["oracle_preview_gross_usd"] = np.exp(alpha + beta * np.log1p(joined["s_final_eod"].clip(lower=0)))
    joined["actual_preview_per_final_amc_seat"] = joined["actual_preview_gross_usd"] / joined["s_final_eod"]
    joined["stage1_log_error"] = np.log(joined["s_final_eod"] / joined["predicted_eod_amc_seats"])
    joined["stage2_oracle_log_error"] = np.log(joined["actual_preview_gross_usd"] / joined["oracle_preview_gross_usd"])
    joined["total_log_error"] = np.log(joined["actual_preview_gross_usd"] / joined["predicted_preview_gross_usd"])
    joined["preview_scale_bucket"] = pd.cut(
        joined["actual_preview_gross_usd"],
        bins=[0, 250_000, 1_000_000, 5_000_000, np.inf],
        labels=["lt_250k", "250k_1m", "1m_5m", "5m_plus"],
    ).astype(str)
    joined["preview_start_group"] = np.where(
        pd.to_numeric(joined.get("hours_relative_to_first_preview"), errors="coerce").fillna(99).le(4),
        "early_preview_window",
        "later_preview_window",
    )
    return _add_replay_ow_updates(joined, preview_update_policy=preview_update_policy, transfer_policy=transfer_policy)


def _add_replay_ow_updates(
    joined: pd.DataFrame,
    *,
    preview_update_policy: dict[str, Any],
    transfer_policy: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for _, row in joined.iterrows():
        baseline = float(row.get("baseline_ow_usd")) if pd.notna(row.get("baseline_ow_usd")) else math.nan
        actual_preview = float(row.get("actual_preview_gross_usd")) if pd.notna(row.get("actual_preview_gross_usd")) else math.nan
        actual_update = update_ow_prior_from_reported_preview(
            baseline_ow_usd=baseline,
            row=pd.Series({
                "thursday_preview_gross_usd": actual_preview,
                "forecast_origin_date": row.get("forecast_origin_date") or "1900-01-01",
                "thursday_preview_cutoff_date": pd.Timestamp(row.get("exhibition_date")).date(),
            }),
            policy=preview_update_policy,
        )
        preview_dist = predict_preview_distribution(
            point_preview_gross_usd=float(row.get("predicted_preview_gross_usd")),
            forecast_origin=str(row.get("forecast_origin")),
            policy=transfer_policy,
            draws=2_000,
            seed=int(float(row.get("movie_id") or 0) * 100 + len(str(row.get("forecast_origin")))),
        )
        amc_update = update_ow_distribution(
            baseline_ow_usd=baseline,
            preview_distribution=preview_dist,
            preview_update_policy=preview_update_policy,
        )
        payload = row.to_dict()
        payload.update({
            "base_ow_forecast_usd": baseline,
            "actual_preview_updated_ow_usd": actual_update.preview_updated_ow_usd,
            "actual_preview_update_applied": actual_update.preview_update_applied,
            "actual_preview_update_fallback_reason": actual_update.fallback_reason,
            "amc_updated_ow_usd": amc_update.get("point_usd"),
            "amc_ow_lo80_usd": amc_update.get("lo80_usd"),
            "amc_ow_hi80_usd": amc_update.get("hi80_usd"),
            "amc_ow_lo95_usd": amc_update.get("lo95_usd"),
            "amc_ow_hi95_usd": amc_update.get("hi95_usd"),
            "amc_ow_update_enabled": bool(amc_update.get("enabled")),
            "amc_ow_fallback_reason": amc_update.get("fallback_reason"),
        })
        rows.append(payload)
    return pd.DataFrame(rows)


def _error_metrics(frame: pd.DataFrame, *, actual_col: str, forecast_col: str, extra: dict[str, Any]) -> dict[str, Any]:
    actual = pd.to_numeric(frame.get(actual_col), errors="coerce")
    forecast = pd.to_numeric(frame.get(forecast_col), errors="coerce")
    valid = actual.gt(0) & forecast.gt(0)
    residual = np.log(actual.loc[valid] / forecast.loc[valid])
    return {
        **extra,
        "n": int(valid.sum()),
        "movie_n": int(frame.loc[valid, "movie_id"].nunique()) if "movie_id" in frame else int(valid.sum()),
        "ME_log": float(residual.mean()) if len(residual) else np.nan,
        "MAE_log": float(residual.abs().mean()) if len(residual) else np.nan,
        "RMSE_log": float(np.sqrt(np.mean(residual ** 2))) if len(residual) else np.nan,
        "dollar_MAE": float((actual.loc[valid] - forecast.loc[valid]).abs().mean()) if len(residual) else np.nan,
    }


def replay_error_tables(joined: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if joined.empty:
        return {
            "stage1": pd.DataFrame(columns=["grouping", "group", "n", "movie_n", "ME_log", "MAE_log", "RMSE_log", "median_APE"]),
            "stage2": pd.DataFrame(columns=["grouping", "group", "n", "movie_n", "ME_log", "MAE_log", "RMSE_log", "dollar_MAE"]),
            "total": pd.DataFrame(columns=["grouping", "group", "n", "movie_n", "ME_log", "MAE_log", "RMSE_log", "dollar_MAE"]),
            "benchmarks": pd.DataFrame(columns=["benchmark", "n", "movie_n", "ME_log", "MAE_log", "RMSE_log", "dollar_MAE"]),
        }
    stage1_rows = []
    for keys, groupby in [
        (["forecast_origin"], joined.groupby("forecast_origin", dropna=False)),
        (["exhibition_date"], joined.groupby("exhibition_date", dropna=False)),
        (["preview_start_group"], joined.groupby("preview_start_group", dropna=False)),
        (["preview_scale_bucket"], joined.groupby("preview_scale_bucket", dropna=False)),
    ]:
        for key, group in groupby:
            residual = group["stage1_log_error"].dropna()
            stage1_rows.append({
                "grouping": ",".join(keys),
                "group": str(key),
                "n": int(len(residual)),
                "movie_n": int(group["movie_id"].nunique()),
                "ME_log": float(residual.mean()) if len(residual) else np.nan,
                "MAE_log": float(residual.abs().mean()) if len(residual) else np.nan,
                "RMSE_log": float(np.sqrt(np.mean(residual ** 2))) if len(residual) else np.nan,
                "median_APE": float(((group["s_final_eod"] - group["predicted_eod_amc_seats"]).abs() / group["s_final_eod"]).median()),
            })
    stage2_rows = []
    movie_level = joined.sort_values(["movie_id", "exhibition_date", "forecast_origin"]).drop_duplicates(["movie_id", "exhibition_date"])
    for key, group in movie_level.groupby("exhibition_date", dropna=False):
        stage2_rows.append(_error_metrics(group, actual_col="actual_preview_gross_usd", forecast_col="oracle_preview_gross_usd", extra={"grouping": "exhibition_date", "group": str(key)}))
    for key, group in movie_level.groupby("preview_scale_bucket", dropna=False):
        stage2_rows.append(_error_metrics(group, actual_col="actual_preview_gross_usd", forecast_col="oracle_preview_gross_usd", extra={"grouping": "preview_scale_bucket", "group": str(key)}))
    total_rows = []
    for key, group in joined.groupby("forecast_origin", dropna=False):
        total_rows.append(_error_metrics(group, actual_col="actual_preview_gross_usd", forecast_col="predicted_preview_gross_usd", extra={"grouping": "forecast_origin", "group": str(key)}))
    return {
        "stage1": pd.DataFrame(stage1_rows),
        "stage2": pd.DataFrame(stage2_rows),
        "total": pd.DataFrame(total_rows),
        "benchmarks": replay_benchmark_comparison(joined),
    }


def replay_benchmark_comparison(joined: pd.DataFrame) -> pd.DataFrame:
    if joined.empty:
        return pd.DataFrame()
    out = joined.copy()
    bridge_ratio = out["predicted_preview_gross_usd"] / np.log1p(out["predicted_eod_amc_seats"]).replace(0, np.nan)
    out["direct_asof_preview_usd"] = bridge_ratio * np.log1p(pd.to_numeric(out["s_obs"], errors="coerce").clip(lower=0))
    fraction = (pd.to_numeric(out["s_obs"], errors="coerce") / pd.to_numeric(out["s_final_eod"], errors="coerce")).replace([np.inf, -np.inf], np.nan)
    origin_fraction = out.assign(completion_fraction=fraction).groupby("forecast_origin")["completion_fraction"].transform("median")
    projected_eod = pd.to_numeric(out["s_obs"], errors="coerce") / origin_fraction.clip(lower=0.01)
    out["completion_ratio_preview_usd"] = bridge_ratio * np.log1p(projected_eod.clip(lower=0))
    specs = {
        "direct_asof": "direct_asof_preview_usd",
        "completion_ratio": "completion_ratio_preview_usd",
        "predicted_eod_bridge": "predicted_preview_gross_usd",
    }
    rows = []
    for model, forecast_col in specs.items():
        rows.append(_error_metrics(out, actual_col="actual_preview_gross_usd", forecast_col=forecast_col, extra={"benchmark": model}))
    return pd.DataFrame(rows)


def frozen_stage1_error_table(frozen: pd.DataFrame) -> pd.DataFrame:
    if frozen.empty:
        return pd.DataFrame(columns=["grouping", "group", "n", "movie_n", "ME_log", "MAE_log", "RMSE_log", "median_APE"])
    work = frozen.copy()
    for column in ["s_final_eod", "predicted_eod_amc_seats", "predicted_preview_gross_usd"]:
        work[column] = pd.to_numeric(work.get(column), errors="coerce")
    work = work.loc[work["s_final_eod"].gt(0) & work["predicted_eod_amc_seats"].gt(0)].copy()
    if work.empty:
        return pd.DataFrame(columns=["grouping", "group", "n", "movie_n", "ME_log", "MAE_log", "RMSE_log", "median_APE"])
    work["stage1_log_error"] = np.log(work["s_final_eod"] / work["predicted_eod_amc_seats"])
    work["preview_scale_bucket"] = pd.cut(
        work["predicted_preview_gross_usd"],
        bins=[0, 1_000, 5_000, 25_000, np.inf],
        labels=["lt_1k_shadow", "1k_5k_shadow", "5k_25k_shadow", "25k_plus_shadow"],
    ).astype(str)
    work["preview_start_group"] = np.where(
        pd.to_numeric(work.get("hours_relative_to_first_preview"), errors="coerce").fillna(99).le(4),
        "early_preview_window",
        "later_preview_window",
    )
    rows = []
    for keys, groupby in [
        (["forecast_origin"], work.groupby("forecast_origin", dropna=False)),
        (["exhibition_date"], work.groupby("exhibition_date", dropna=False)),
        (["preview_start_group"], work.groupby("preview_start_group", dropna=False)),
        (["preview_scale_bucket"], work.groupby("preview_scale_bucket", dropna=False)),
    ]:
        for key, group in groupby:
            residual = group["stage1_log_error"].dropna()
            rows.append({
                "grouping": ",".join(keys),
                "group": str(key),
                "n": int(len(residual)),
                "movie_n": int(group["movie_id"].nunique()),
                "ME_log": float(residual.mean()) if len(residual) else np.nan,
                "MAE_log": float(residual.abs().mean()) if len(residual) else np.nan,
                "RMSE_log": float(np.sqrt(np.mean(residual ** 2))) if len(residual) else np.nan,
                "median_APE": float(((group["s_final_eod"] - group["predicted_eod_amc_seats"]).abs() / group["s_final_eod"]).median()),
            })
    return pd.DataFrame(rows)


def interval_score(actual: pd.Series, lower: pd.Series, upper: pd.Series, alpha: float) -> pd.Series:
    return (upper - lower) + (2 / alpha) * (lower - actual).clip(lower=0) + (2 / alpha) * (actual - upper).clip(lower=0)


def ow_replay_comparison(joined: pd.DataFrame) -> pd.DataFrame:
    if joined.empty:
        return pd.DataFrame()
    specs = {
        "base_pre_preview": ("base_ow_forecast_usd", None, None),
        "reported_preview_actual_update": ("actual_preview_updated_ow_usd", None, None),
        "amc_preview_nowcast_update": ("amc_updated_ow_usd", "amc_ow_lo80_usd", "amc_ow_hi80_usd"),
    }
    rows = []
    for model, (point, lo80, hi80) in specs.items():
        metrics = _error_metrics(joined, actual_col="actual_ow_usd", forecast_col=point, extra={"ow_model": model})
        actual = pd.to_numeric(joined["actual_ow_usd"], errors="coerce")
        forecast = pd.to_numeric(joined[point], errors="coerce")
        valid = actual.gt(0) & forecast.gt(0)
        if lo80 and hi80:
            lo80s, hi80s = pd.to_numeric(joined[lo80], errors="coerce"), pd.to_numeric(joined[hi80], errors="coerce")
            lo95s, hi95s = pd.to_numeric(joined["amc_ow_lo95_usd"], errors="coerce"), pd.to_numeric(joined["amc_ow_hi95_usd"], errors="coerce")
            metrics["coverage80"] = float(actual[valid].between(lo80s[valid], hi80s[valid]).mean()) if valid.any() else np.nan
            metrics["coverage95"] = float(actual[valid].between(lo95s[valid], hi95s[valid]).mean()) if valid.any() else np.nan
            wis80 = interval_score(actual[valid], lo80s[valid], hi80s[valid], 0.20)
            wis95 = interval_score(actual[valid], lo95s[valid], hi95s[valid], 0.05)
            metrics["WIS"] = float(((wis80 + wis95) / 2.0).mean()) if valid.any() else np.nan
        else:
            metrics["coverage80"] = np.nan
            metrics["coverage95"] = np.nan
            metrics["WIS"] = np.nan
        metrics["bucket_log_score"] = float((-np.log(np.clip(1.0 - np.minimum(np.abs(np.log(actual[valid] / forecast[valid])), 5.0) / 5.0, 1e-6, 1.0))).mean()) if valid.any() else np.nan
        metrics["RPS"] = float((np.minimum(np.abs(np.log(actual[valid] / forecast[valid])), 5.0) / 5.0).mean()) if valid.any() else np.nan
        rows.append(metrics)
    return pd.DataFrame(rows)


def reported_preview_promotion_recommendation(
    joined: pd.DataFrame,
    policy: dict[str, Any],
    *,
    proxy_identity: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Production status for the new opening-Thursday daily target.

    The historical policy currently loaded from artifacts may still be the old
    reported-preview updater, so this function deliberately refuses to promote
    unless the artifact advertises the same target definition.
    """
    replay = ow_replay_comparison(joined)
    same_target_policy = str((policy or {}).get("target_definition") or "").lower() == "opening_thursday_daily_gross"
    if replay.empty or "ow_model" not in replay:
        return {
            "opening_thursday_daily_actual_to_ow_recommendation": "hold",
            "reason": "no_opening_thursday_daily_replay_rows_joined" if replay.empty else "invalid_replay_metrics",
            "rolling_amc_preview_to_ow_production_recommendation": "hold_shadow_only",
            "live_replay_passed": False,
            "same_target_historical_policy_present": same_target_policy,
            "policy_version": policy.get("policy_version") if isinstance(policy, dict) else None,
            "safeguards_required": [
                "opening_thursday_daily_target_definition",
                "baseline_fallback_when_thursday_daily_missing",
                "versioned_frozen_coefficients",
                "idempotent_actual_thursday_daily_updates",
                "thursday_daily_to_ow_residual_uncertainty",
            ],
        }
    target_types = set(joined.get("preview_target_type", pd.Series(dtype="object")).dropna().astype(str))
    opening_thursday_rows = bool(target_types & {"opening_thursday_daily_gross"})
    base = replay.loc[replay["ow_model"].eq("base_pre_preview")]
    actual = replay.loc[replay["ow_model"].eq("reported_preview_actual_update")]
    live_replay_passed = bool(
        not base.empty and not actual.empty
        and float(actual.iloc[0]["MAE_log"]) <= float(base.iloc[0]["MAE_log"])
        and float(actual.iloc[0]["RMSE_log"]) <= float(base.iloc[0]["RMSE_log"])
    )
    recommendation = "enable" if live_replay_passed and same_target_policy and opening_thursday_rows else "hold"
    return {
        "opening_thursday_daily_actual_to_ow_recommendation": recommendation,
        "rolling_amc_opening_thursday_daily_to_ow_recommendation": "hold_shadow_only",
        "live_replay_passed": live_replay_passed,
        "same_target_historical_policy_present": same_target_policy,
        "policy_version": policy.get("policy_version") if isinstance(policy, dict) else None,
        "opening_thursday_daily_rows_present": opening_thursday_rows,
        "target_definition": "opening_thursday_daily_gross",
        "safeguards_required": [
            "opening_thursday_daily_target_definition",
            "baseline_fallback_when_thursday_daily_missing",
            "versioned_frozen_coefficients",
            "idempotent_actual_thursday_daily_updates",
            "thursday_daily_to_ow_residual_uncertainty",
        ],
    }


def add_baseline_ow(panel: pd.DataFrame, daily_baseline: pd.DataFrame) -> pd.DataFrame:
    if panel.empty or daily_baseline.empty:
        panel = panel.copy()
        panel["baseline_ow_usd"] = np.nan
        return panel
    base = daily_baseline.copy()
    for column in ["pre_fri_usd", "pre_sat_usd", "pre_sun_usd"]:
        base[column] = pd.to_numeric(base.get(column), errors="coerce")
    base["baseline_ow_usd"] = base[["pre_fri_usd", "pre_sat_usd", "pre_sun_usd"]].sum(axis=1, min_count=3)
    keys = ["release_run_id", "movie_id"]
    return panel.merge(base[keys + ["baseline_ow_usd"]].drop_duplicates(keys), on=keys, how="left")


def forecast_preview_row(panel: pd.DataFrame, target: pd.Series, *, min_train_rows: int) -> dict[str, Any] | None:
    current_date = pd.Timestamp(target["exhibition_date"]).date()
    forecast_origin = str(target["forecast_origin"])
    same_origin = preview_training_frame(panel, current_date=current_date, forecast_origin=forecast_origin)
    all_preview = preview_training_frame(panel, current_date=current_date)
    pool_name, pool = select_preview_training_pool(
        same_origin,
        forecast_origin=forecast_origin,
        min_train_rows=min_train_rows,
        all_preview_train=all_preview,
    )
    if pool.empty:
        return None
    work = pool.copy()
    work["raw_pace"] = pd.to_numeric(work["s_obs"], errors="coerce") / pd.to_numeric(work["s_final_eod"], errors="coerce")
    pace = float(work["raw_pace"].dropna().median())
    if not math.isfinite(pace) or pace <= 0:
        return None
    pace = float(np.clip(pace, 0.01, 1.0))
    s_obs = positive_float(target.get("s_obs"))
    if not math.isfinite(s_obs):
        return None
    pred_final_mult = s_obs / pace
    pred_final_add = additive_projection(work, target)
    if not math.isfinite(pred_final_add):
        pred_final_add = pred_final_mult
    pred_final_hybrid = 0.5 * (pred_final_mult + pred_final_add)
    work["bridge_residual"] = np.log(pd.to_numeric(work["actual_gross_usd"], errors="coerce")) - np.log1p(
        pd.to_numeric(work["s_final_eod"], errors="coerce")
    )
    bridge = float(work["bridge_residual"].dropna().median())
    if not math.isfinite(bridge):
        return None
    predictions = {
        "multiplicative": float(math.exp(bridge + math.log1p(pred_final_mult))),
        "additive": float(math.exp(bridge + math.log1p(pred_final_add))),
        "hybrid": float(math.exp(bridge + math.log1p(pred_final_hybrid))),
    }
    return {
        "training_pool": pool_name,
        "training_n": int(len(work)),
        "training_movie_n": int(work["movie_id"].nunique()),
        "pace_median": pace,
        "bridge_median": bridge,
        "predicted_final_preview_seats_multiplicative": pred_final_mult,
        "predicted_final_preview_seats_additive": pred_final_add,
        "predicted_final_preview_seats_hybrid": pred_final_hybrid,
        **{f"forecast_preview_gross_{name}_usd": value for name, value in predictions.items()},
    }


def additive_projection(train: pd.DataFrame, target: pd.Series) -> float:
    required = {"s_obs", "s_final_eod", "c_obs", "c_scheduled_known"}
    if not required.issubset(train.columns) or not required.issubset(target.index):
        return math.nan
    remaining_capacity = (
        pd.to_numeric(train["c_scheduled_known"], errors="coerce") - pd.to_numeric(train["c_obs"], errors="coerce")
    ).clip(lower=0)
    remaining_sales = (
        pd.to_numeric(train["s_final_eod"], errors="coerce") - pd.to_numeric(train["s_obs"], errors="coerce")
    ).clip(lower=0)
    valid = remaining_capacity.gt(0) & remaining_sales.notna()
    if valid.sum() == 0 or remaining_capacity.loc[valid].sum() <= 0:
        return math.nan
    occupancy = float(remaining_sales.loc[valid].sum() / remaining_capacity.loc[valid].sum())
    target_remaining_capacity = max(float(target.get("c_scheduled_known") or 0) - float(target.get("c_obs") or 0), 0.0)
    prediction = positive_float(target.get("s_obs")) + target_remaining_capacity * float(np.clip(occupancy, 0, 1))
    return prediction if math.isfinite(prediction) and prediction > 0 else math.nan


def build_rolling_predictions(panel: pd.DataFrame, *, min_train_rows: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ordered = panel.sort_values(["opening_weekend_start", "movie_id", "forecast_origin"]).reset_index(drop=True)
    for _, target in ordered.iterrows():
        pred = forecast_preview_row(ordered, target, min_train_rows=min_train_rows)
        base = target.to_dict()
        if pred is None:
            rows.append({**base, "scored": False, "fallback_reason": "insufficient_preview_history"})
            continue
        actual = positive_float(target.get("actual_gross_usd"))
        payload = {**base, **pred, "scored": True, "fallback_reason": None}
        for model in POINT_MODELS:
            forecast = positive_float(payload.get(f"forecast_preview_gross_{model}_usd"))
            payload[f"log_error_{model}"] = math.log(actual / forecast) if math.isfinite(actual) and math.isfinite(forecast) else np.nan
            payload[f"abs_log_error_{model}"] = abs(payload[f"log_error_{model}"]) if math.isfinite(payload[f"log_error_{model}"]) else np.nan
            payload[f"squared_log_error_{model}"] = payload[f"log_error_{model}"] ** 2 if math.isfinite(payload[f"log_error_{model}"]) else np.nan
            payload[f"abs_dollar_error_{model}"] = abs(actual - forecast) if math.isfinite(actual) and math.isfinite(forecast) else np.nan
        rows.append(payload)
    return pd.DataFrame(rows)


def evaluate_models(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or "scored" not in predictions.columns:
        scored = pd.DataFrame()
    else:
        scored = predictions.loc[predictions["scored"].eq(True)].copy()
    rows = []
    for model in POINT_MODELS:
        required = [f"log_error_{model}", f"abs_dollar_error_{model}"]
        clean = scored.dropna(subset=required) if set(required).issubset(scored.columns) else pd.DataFrame()
        rows.append(
            {
                "point_model": model,
                "n": int(len(clean)),
                "movie_n": int(clean["movie_id"].nunique()) if not clean.empty else 0,
                "ME_log": float(clean[f"log_error_{model}"].mean()) if not clean.empty else np.nan,
                "MAE_log": float(clean[f"abs_log_error_{model}"].mean()) if not clean.empty else np.nan,
                "RMSE_log": float(np.sqrt(clean[f"squared_log_error_{model}"].mean())) if not clean.empty else np.nan,
                "dollar_MAE": float(clean[f"abs_dollar_error_{model}"].mean()) if not clean.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def evaluate_by_year(predictions: pd.DataFrame, *, selected_model: str) -> pd.DataFrame:
    if predictions.empty or "scored" not in predictions.columns:
        return pd.DataFrame()
    scored = predictions.loc[predictions["scored"].eq(True)].copy()
    rows = []
    for year, group in scored.groupby("release_year", dropna=False):
        rows.append(
            {
                "release_year": int(year) if pd.notna(year) else None,
                "point_model": selected_model,
                "n": int(len(group)),
                "movie_n": int(group["movie_id"].nunique()),
                "MAE_log": float(group[f"abs_log_error_{selected_model}"].mean()),
                "RMSE_log": float(np.sqrt(group[f"squared_log_error_{selected_model}"].mean())),
                "dollar_MAE": float(group[f"abs_dollar_error_{selected_model}"].mean()),
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap(predictions: pd.DataFrame, *, selected_model: str, baseline_model: str, resamples: int, seed: int) -> pd.DataFrame:
    if predictions.empty or "scored" not in predictions.columns:
        return pd.DataFrame()
    scored = predictions.loc[predictions["scored"].eq(True)].copy()
    if scored.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    for metric, selected_col, baseline_col in [
        ("abs_log_error", f"abs_log_error_{selected_model}", f"abs_log_error_{baseline_model}"),
        ("squared_log_error", f"squared_log_error_{selected_model}", f"squared_log_error_{baseline_model}"),
        ("abs_dollar_error", f"abs_dollar_error_{selected_model}", f"abs_dollar_error_{baseline_model}"),
    ]:
        diff = (scored[selected_col] - scored[baseline_col]).dropna().to_numpy("float64")
        if diff.size == 0:
            continue
        samples = rng.choice(diff, size=(resamples, diff.size), replace=True).mean(axis=1)
        rows.append(
            {
                "loss_metric": metric,
                "selected_model": selected_model,
                "baseline_model": baseline_model,
                "n": int(diff.size),
                "mean_loss_difference": float(diff.mean()),
                "ci95_low": float(np.quantile(samples, 0.025)),
                "ci95_high": float(np.quantile(samples, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def build_policy(predictions: pd.DataFrame, metrics: pd.DataFrame, *, min_train_rows: int) -> dict[str, Any]:
    candidates = metrics.loc[metrics["n"].gt(0)].sort_values(["RMSE_log", "MAE_log", "point_model"]) if not metrics.empty else pd.DataFrame()
    selected = str(candidates.iloc[0]["point_model"]) if not candidates.empty else "hybrid"
    if predictions.empty or "scored" not in predictions.columns:
        scored = pd.DataFrame()
    else:
        scored = predictions.loc[predictions["scored"].eq(True)].copy()
    sigma = float(scored[f"log_error_{selected}"].std(ddof=1)) if len(scored) > 1 else 0.55
    return {
        "enabled": bool(len(scored) >= min_train_rows),
        "policy_version": "thursday_amc_preview_pace_bridge_v1",
        "model": "AMC_thursday_preview_pace_bridge_v1",
        "selected_point_model": selected,
        "min_train_rows": int(min_train_rows),
        "fallback_order": ["origin", "origin_bucket", "all_preview", "no_amc_update"],
        "training_rows_scored": int(len(scored)),
        "training_movies_scored": int(scored["movie_id"].nunique()) if not scored.empty else 0,
        "sigma_log": sigma if math.isfinite(sigma) and sigma > 0 else 0.55,
        "promotion_status": "shadow_only",
    }


def evaluate_candidate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    work = predictions.copy()
    work["actual"] = pd.to_numeric(work["actual_gross_usd"], errors="coerce")
    work["forecast"] = pd.to_numeric(work["forecast_preview_gross_usd"], errors="coerce")
    work = work.loc[work["actual"].gt(0) & work["forecast"].gt(0)].copy()
    work["ape"] = (work["actual"] - work["forecast"]).abs() / work["actual"]
    work["abs_dollar_error"] = (work["actual"] - work["forecast"]).abs()
    for level in (80, 95):
        lo, hi = work[f"lo{level}_usd"], work[f"hi{level}_usd"]
        work[f"covered{level}"] = work["actual"].between(lo, hi)
        alpha = 1.0 - level / 100.0
        work[f"interval_score{level}"] = (hi - lo) + (2 / alpha) * (lo - work["actual"]).clip(lower=0) + (2 / alpha) * (work["actual"] - hi).clip(lower=0)
    rows = []
    for (candidate, origin), group in work.groupby(["candidate", "forecast_origin"], dropna=False):
        residual = group["log_residual"]
        rows.append({
            "candidate": candidate, "forecast_origin": origin, "n": len(group),
            "movie_n": group["movie_id"].nunique(), "ME_log": residual.mean(),
            "MAE_log": residual.abs().mean(), "RMSE_log": float(np.sqrt(np.mean(residual ** 2))),
            "median_APE": group["ape"].median(), "dollar_MAE": group["abs_dollar_error"].mean(),
            "coverage80": group.loc[group["lo80_usd"].notna(), "covered80"].mean(),
            "coverage95": group.loc[group["lo95_usd"].notna(), "covered95"].mean(),
            "WIS": (group["interval_score80"].mean() + group["interval_score95"].mean()) / 2.0,
        })
    return pd.DataFrame(rows)


def compare_collection_panels(panel_frames: dict[str, pd.DataFrame], *, min_train_movies: int) -> pd.DataFrame:
    """Replay identical rolling folds for frozen theatre panels."""
    rows = []
    reference_seats = None
    for sample_key, panel in panel_frames.items():
        predictions = release_week_rolling_predictions(panel, min_train_movies=min_train_movies)
        metrics = evaluate_candidate_predictions(predictions)
        seat_total = float(pd.to_numeric(panel.get("s_final_eod"), errors="coerce").sum())
        if reference_seats is None or "175" in sample_key:
            reference_seats = seat_total
        best = metrics.loc[metrics["candidate"].ne("TH4_prior_assisted")].sort_values(["RMSE_log", "MAE_log"]).head(1)
        payload = best.iloc[0].to_dict() if not best.empty else {}
        rows.append({"sample_key": sample_key, "capture_seats": seat_total, **payload})
    denominator = reference_seats or 0.0
    for row in rows:
        row["capture_ratio"] = row["capture_seats"] / denominator if denominator > 0 else np.nan
    return pd.DataFrame(rows)


def build_residual_pools(predictions: pd.DataFrame, *, selected_candidate: str) -> dict[str, list[float]]:
    chosen = predictions.loc[predictions["candidate"].eq(selected_candidate)].copy()
    pools: dict[str, list[float]] = {}
    for origin, group in chosen.groupby("forecast_origin"):
        pools[f"origin:{origin}"] = group["log_residual"].dropna().astype(float).tolist()
    chosen["origin_bucket"] = chosen["forecast_origin"].map(origin_bucket)
    for bucket, group in chosen.groupby("origin_bucket"):
        pools[f"origin_bucket:{bucket}"] = group["log_residual"].dropna().astype(float).tolist()
    pools["pooled_thursday"] = chosen["log_residual"].dropna().astype(float).tolist()
    return pools


def build_candidate_policy(predictions: pd.DataFrame, metrics: pd.DataFrame, *, min_train_rows: int) -> dict[str, Any]:
    eligible = metrics.loc[metrics["candidate"].ne("TH4_prior_assisted") & metrics["n"].ge(min_train_rows)].copy() if not metrics.empty and {"candidate", "n"}.issubset(metrics) else pd.DataFrame()
    selected = str(eligible.sort_values(["RMSE_log", "MAE_log", "candidate"]).iloc[0]["candidate"]) if not eligible.empty else "TH2_predicted_eod_bridge"
    chosen = predictions.loc[predictions["candidate"].eq(selected)] if not predictions.empty and "candidate" in predictions else pd.DataFrame()
    latest = pd.to_datetime(chosen.get("preview_published_at"), utc=True, errors="coerce").max() if not chosen.empty else pd.NaT
    representative = chosen.iloc[-1] if not chosen.empty else pd.Series(dtype=object)
    return {
        "enabled": bool(chosen["movie_id"].nunique() >= min_train_rows) if not chosen.empty else False,
        "policy_version": "thursday_amc_preview_b2_candidates_v2",
        "promotion_status": "shadow_only",
        "selected_candidate": selected,
        "stage1": representative.get("stage1"), "stage2": representative.get("stage2"),
        "bias_adjustment": float(chosen["bias_adjustment"].iloc[-1]) if not chosen.empty else 0.0,
        "training_cutoff": latest.isoformat() if pd.notna(latest) else None,
        "eligible_target": {"preview_target_type": "thursday_only", "includes_wednesday_early_access": False, "preview_days_included": 1, "is_wide_release": True},
        "residual_pools": build_residual_pools(predictions, selected_candidate=selected) if not predictions.empty else {},
        "min_residual_pool_n": 5,
        "min_train_rows": int(min_train_rows),
        "training_rows_scored": int(len(chosen)),
        "training_movies_scored": int(chosen["movie_id"].nunique()) if not chosen.empty else 0,
        "fallback_order": ["origin", "origin_bucket", "pooled_thursday", "no_amc_update"],
    }


def run(database_url: str | None, *, model_version: str, min_train_rows: int, output_dir: Path) -> dict[str, Path]:
    conn = connect_database(database_url)
    try:
        schedule = fetch_schedule(conn)
        snapshots = fetch_snapshots(conn)
        actuals = fetch_preview_actuals(conn)
        daily_actuals = fetch_actuals(conn)
    finally:
        conn.close()
    artifacts = load_model_artifacts(model_version)
    panel = build_preview_panel_from_frames(schedule=schedule, snapshots=snapshots, actuals=actuals)
    donor_panel = build_daily_donor_panel_from_frames(schedule=schedule, snapshots=snapshots, actuals=daily_actuals)
    all_day_stage1_panel = build_all_day_completion_panel_from_frames(schedule=schedule, snapshots=snapshots)
    panel = add_baseline_ow(panel, artifacts.daily_baseline)
    legacy_predictions = build_rolling_predictions(panel, min_train_rows=min_train_rows)
    legacy_metrics = evaluate_models(legacy_predictions)
    predictions = release_week_rolling_predictions(panel, min_train_movies=min_train_rows)
    metrics = evaluate_candidate_predictions(predictions)
    selected = "hybrid"
    year = evaluate_by_year(legacy_predictions, selected_model=selected)
    boot = paired_bootstrap(
        legacy_predictions,
        selected_model=selected,
        baseline_model="multiplicative" if selected != "multiplicative" else "hybrid",
        resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
        seed=DEFAULT_BOOTSTRAP_SEED,
    )
    thursday_specific_policy = build_candidate_policy(predictions, metrics, min_train_rows=min_train_rows)
    donor_comparison = donor_pool_comparison(donor_panel, stage1_panel=all_day_stage1_panel)
    selected_pool = "HYBRID_PARTIAL_POOL"
    transfer_policy = fit_pooled_transfer(
        donor_panel,
        stage1_panel=all_day_stage1_panel,
        selected_stage2_pool=selected_pool,
        min_movies=min_train_rows,
    )
    policy = {
        **transfer_policy,
        "thursday_specific": thursday_specific_policy,
        "pooled_day_transfer": transfer_policy,
    }
    donor_predictions = donor_day_leave_one_out(donor_panel, stage1_panel=all_day_stage1_panel, donor_pool=selected_pool)
    donor_metrics = donor_day_metrics(donor_predictions)
    temporal_predictions = temporal_deployment_predictions(donor_panel, stage1_panel=all_day_stage1_panel, donor_pool=selected_pool)
    sample_counts = transfer_sample_counts(
        all_day_stage1_panel,
        donor_panel,
        thursday_preview_labels=int(panel[["movie_id", "exhibition_date"]].drop_duplicates().shape[0]) if not panel.empty else 0,
    )
    transfer_shadow = build_thursday_transfer_shadow_panel(schedule=schedule, snapshots=snapshots, policy=transfer_policy)
    transfer_ow_impact = build_transfer_ow_impact(
        transfer_shadow, daily_baseline=artifacts.daily_baseline,
        transfer_policy=transfer_policy, preview_update_policy=artifacts.thursday_preview_policy,
    )
    frozen_replay = transfer_shadow.copy()
    replay_joined = join_frozen_shadow_with_actuals(
        frozen_replay,
        preview_actuals=actuals,
        daily_actuals=daily_actuals,
        daily_baseline=artifacts.daily_baseline,
        preview_update_policy=artifacts.thursday_preview_policy,
        transfer_policy=transfer_policy,
    )
    if not replay_joined.empty and "sample" in sample_counts:
        label_mask = sample_counts["sample"].eq("opening_thursday_daily_gross_labels")
        if label_mask.any():
            opening_dates = pd.to_datetime(replay_joined["exhibition_date"], errors="coerce")
            release_weeks = pd.to_datetime(
                replay_joined.get("opening_weekend_start", replay_joined.get("friday_date", replay_joined["exhibition_date"])),
                errors="coerce",
            ).dt.to_period("W-FRI")
            sample_counts.loc[label_mask, "distinct_calendar_dates"] = int(opening_dates.nunique())
            sample_counts.loc[label_mask, "distinct_release_weeks"] = int(release_weeks.nunique())
            sample_counts.loc[label_mask, "distinct_movie_days"] = int(
                replay_joined[["movie_id", "exhibition_date"]].drop_duplicates().shape[0]
            )
            sample_counts.loc[label_mask, "distinct_movies"] = int(replay_joined["movie_id"].nunique())
            sample_counts.loc[label_mask, "origin_rows"] = int(len(replay_joined))
    replay_errors = replay_error_tables(replay_joined)
    if replay_joined.empty:
        replay_errors["stage1"] = frozen_stage1_error_table(frozen_replay)
    replay_ow_comparison = ow_replay_comparison(replay_joined)
    daily_nowcast_proxy = thursday_daily_nowcast_proxy_diagnostic(
        frozen_replay,
        daily_actuals=daily_actuals,
    )
    proxy_identity = preview_proxy_identity_audit(
        actuals,
        daily_actuals,
        artifacts.daily_baseline,
    )
    promotion_recommendation = reported_preview_promotion_recommendation(
        replay_joined,
        artifacts.thursday_preview_policy,
        proxy_identity=proxy_identity,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "panel": output_dir / "thursday_amc_preview_nowcast_panel.csv",
        "predictions": output_dir / "thursday_amc_preview_nowcast_rolling_predictions.csv",
        "metrics": output_dir / "thursday_amc_preview_nowcast_model_comparison.csv",
        "year": output_dir / "thursday_amc_preview_nowcast_year_stability.csv",
        "bootstrap": output_dir / "thursday_amc_preview_nowcast_paired_bootstrap.csv",
        "policy": output_dir / "thursday_amc_preview_policy.json",
        "daily_gross_policy": output_dir / "amc_daily_gross_policy.json",
        "opening_thursday_daily_policy": output_dir / "opening_thursday_daily_policy.json",
        "legacy_metrics": output_dir / "thursday_amc_preview_legacy_model_comparison.csv",
        "residuals": output_dir / "thursday_amc_preview_residual_pools.json",
        "subgroups": output_dir / "thursday_amc_preview_subgroup_metrics.csv",
        "panels": output_dir / "thursday_amc_preview_panel_comparison.csv",
        "donor_metrics": output_dir / "donor_day_leave_one_out_metrics.csv",
        "donor_bias": output_dir / "donor_day_bias_by_origin.csv",
        "donor_ratio": output_dir / "donor_day_nationalization_ratio.csv",
        "donor_pool_comparison": output_dir / "donor_stage2_pool_comparison.csv",
        "temporal_deployment": output_dir / "donor_temporal_deployment_predictions.csv",
        "transfer_sample_counts": output_dir / "thursday_transfer_sample_counts.csv",
        "transfer_shadow": output_dir / "thursday_transfer_shadow_predictions.csv",
        "stage_decomposition": output_dir / "thursday_transfer_stage_decomposition.csv",
        "ow_impact": output_dir / "thursday_transfer_ow_impact.csv",
        "frozen_replay_forecasts": output_dir / "thursday_transfer_frozen_replay_forecasts.csv",
        "frozen_replay_joined": output_dir / "thursday_transfer_frozen_replay_joined_actuals.csv",
        "opening_thursday_daily_forecasts": output_dir / "opening_thursday_daily_frozen_replay_forecasts.csv",
        "opening_thursday_daily_joined": output_dir / "opening_thursday_daily_frozen_replay_joined_actuals.csv",
        "opening_thursday_daily_stage1": output_dir / "opening_thursday_daily_frozen_replay_stage1_errors.csv",
        "opening_thursday_daily_stage2_oracle": output_dir / "opening_thursday_daily_frozen_replay_stage2_oracle_errors.csv",
        "opening_thursday_daily_total": output_dir / "opening_thursday_daily_frozen_replay_total_errors.csv",
        "opening_thursday_daily_benchmarks": output_dir / "opening_thursday_daily_frozen_replay_benchmark_comparison.csv",
        "frozen_replay_stage1": output_dir / "thursday_transfer_frozen_replay_stage1_errors.csv",
        "frozen_replay_stage2_oracle": output_dir / "thursday_transfer_frozen_replay_stage2_oracle_errors.csv",
        "frozen_replay_total": output_dir / "thursday_transfer_frozen_replay_total_errors.csv",
        "frozen_replay_benchmarks": output_dir / "thursday_transfer_frozen_replay_benchmark_comparison.csv",
        "ow_replay_comparison": output_dir / "thursday_preview_ow_replay_comparison.csv",
        "opening_thursday_daily_ow_replay_comparison": output_dir / "opening_thursday_daily_ow_replay_comparison.csv",
        "promotion_recommendation": output_dir / "thursday_preview_promotion_recommendation.json",
        "opening_thursday_daily_promotion_recommendation": output_dir / "opening_thursday_daily_promotion_recommendation.json",
        "daily_nowcast_proxy": output_dir / "thursday_daily_nowcast_proxy_diagnostic.csv",
        "proxy_identity_audit": output_dir / "preview_proxy_identity_audit.csv",
    }
    panel.to_csv(paths["panel"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    year.to_csv(paths["year"], index=False)
    boot.to_csv(paths["bootstrap"], index=False)
    write_json(paths["policy"], policy)
    write_json(paths["daily_gross_policy"], policy)
    write_json(paths["opening_thursday_daily_policy"], policy)
    legacy_metrics.to_csv(paths["legacy_metrics"], index=False)
    write_json(paths["residuals"], policy.get("residual_pools", {}))
    metrics.to_csv(paths["subgroups"], index=False)
    pd.DataFrame(columns=["sample_key", "capture_ratio", "RMSE_log", "coverage80", "coverage95", "WIS"]).to_csv(paths["panels"], index=False)
    donor_metrics.to_csv(paths["donor_metrics"], index=False)
    donor_bias_by_origin(donor_predictions).to_csv(paths["donor_bias"], index=False)
    donor_nationalization_ratio(donor_panel).to_csv(paths["donor_ratio"], index=False)
    donor_comparison.to_csv(paths["donor_pool_comparison"], index=False)
    temporal_predictions.to_csv(paths["temporal_deployment"], index=False)
    sample_counts.to_csv(paths["transfer_sample_counts"], index=False)
    transfer_shadow.to_csv(paths["transfer_shadow"], index=False)
    donor_predictions.to_csv(paths["stage_decomposition"], index=False)
    transfer_ow_impact.to_csv(paths["ow_impact"], index=False)
    frozen_replay.to_csv(paths["frozen_replay_forecasts"], index=False)
    replay_joined.to_csv(paths["frozen_replay_joined"], index=False)
    frozen_replay.to_csv(paths["opening_thursday_daily_forecasts"], index=False)
    replay_joined.to_csv(paths["opening_thursday_daily_joined"], index=False)
    replay_errors["stage1"].to_csv(paths["frozen_replay_stage1"], index=False)
    replay_errors["stage2"].to_csv(paths["frozen_replay_stage2_oracle"], index=False)
    replay_errors["total"].to_csv(paths["frozen_replay_total"], index=False)
    replay_errors["benchmarks"].to_csv(paths["frozen_replay_benchmarks"], index=False)
    replay_errors["stage1"].to_csv(paths["opening_thursday_daily_stage1"], index=False)
    replay_errors["stage2"].to_csv(paths["opening_thursday_daily_stage2_oracle"], index=False)
    replay_errors["total"].to_csv(paths["opening_thursday_daily_total"], index=False)
    replay_errors["benchmarks"].to_csv(paths["opening_thursday_daily_benchmarks"], index=False)
    replay_ow_comparison.to_csv(paths["ow_replay_comparison"], index=False)
    replay_ow_comparison.to_csv(paths["opening_thursday_daily_ow_replay_comparison"], index=False)
    write_json(paths["promotion_recommendation"], promotion_recommendation)
    write_json(paths["opening_thursday_daily_promotion_recommendation"], promotion_recommendation)
    daily_nowcast_proxy.to_csv(paths["daily_nowcast_proxy"], index=False)
    proxy_identity.to_csv(paths["proxy_identity_audit"], index=False)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_database_arg(parser)
    parser.add_argument("--model-version", default="latest")
    parser.add_argument("--min-train-rows", type=int, default=DEFAULT_MIN_TRAIN_ROWS)
    parser.add_argument("--output-dir", type=Path, default=DIAGNOSTICS_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = run(
        args.database_url,
        model_version=args.model_version,
        min_train_rows=args.min_train_rows,
        output_dir=args.output_dir,
    )
    for label, path in paths.items():
        print(f"Wrote {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
