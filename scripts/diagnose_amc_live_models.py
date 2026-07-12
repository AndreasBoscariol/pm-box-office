#!/usr/bin/env python3
"""Clustered diagnostics for sparse-history AMC live box-office forecasts."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eda.During import same_day_seat_nowcast_eda as amc_eda
from models.boxoffice.constants import LIVE_ORIGINS
from models.boxoffice.live_amc_plugin import LiveAmcPluginConfig, nowcast_from_panel_row
from pm_box_office.db.connection import connect_database

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = REPO_ROOT / "models/boxoffice/boxoffice_local_007_weighted_interval_calibration/daily_regime_baseline_forecasts.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data/diagnostics"
ORIGIN_GROUP = {"10:00": "early", "12:00": "early", "14:00": "early", "16:00": "late", "18:00": "late", "20:00": "late", "EOD": "late"}


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__)
    out.add_argument("--database-url")
    out.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE)
    out.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return out


def baseline_long(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = pd.read_csv(path)
    rows = []
    for day, date_col, pre_col, actual_col in [
        ("Friday", "friday_date", "pre_fri_usd", "actual_fri_usd"),
        ("Saturday", "saturday_date", "pre_sat_usd", "actual_sat_usd"),
        ("Sunday", "sunday_date", "pre_sun_usd", "actual_sun_usd"),
    ]:
        part = wide[["movie_id", "release_run_id", "title", date_col, pre_col, actual_col]].copy()
        part.columns = ["movie_id", "release_run_id", "title", "exhibition_date", "baseline_gross_usd", "baseline_actual_usd"]
        part["day_of_week"] = day
        rows.append(part)
    long = pd.concat(rows, ignore_index=True)
    long["exhibition_date"] = pd.to_datetime(long["exhibition_date"], errors="coerce").dt.date
    return long, wide


def build_panel(conn: Any, baseline_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = amc_eda.build_panel(conn, origins=LIVE_ORIGINS, origin_timezone="America/New_York")
    long, wide = baseline_long(baseline_path)
    panel = panel.drop(columns=["baseline_gross_usd", "baseline_source"], errors="ignore")
    panel = panel.merge(
        long[["movie_id", "exhibition_date", "baseline_gross_usd"]],
        on=["movie_id", "exhibition_date"],
        how="left",
    )
    return panel, wide


def historical_amc_predictions(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    config = LiveAmcPluginConfig(min_train_rows=3)
    for _, target in panel.iterrows():
        actual = number(target.get("actual_gross_usd"))
        baseline = number(target.get("baseline_gross_usd"))
        if not positive(actual):
            continue
        plugin = nowcast_from_panel_row(panel, target, config=config)
        if plugin is None:
            continue
        row = target.to_dict()
        row["amc_pred_usd"] = plugin["pred_daily_gross_usd"]
        row["feature_quality_bucket"] = plugin["feature_quality_bucket"]
        row["origin_group"] = ORIGIN_GROUP.get(str(target["forecast_origin"]), "late")
        row["movie_day_key"] = f"{target['movie_id']}|{target['exhibition_date']}"
        rows.append(row)
    return pd.DataFrame(rows)


def constrained_beta(train: pd.DataFrame) -> float:
    base = np.log(train["baseline_gross_usd"].to_numpy(dtype=float))
    amc = np.log(train["amc_pred_usd"].to_numpy(dtype=float))
    actual = np.log(train["actual_gross_usd"].to_numpy(dtype=float))
    x = amc - base
    denominator = float(x @ x)
    return float(np.clip((x @ (actual - base)) / denominator, 0.0, 1.0)) if denominator > 0 else 0.0


def add_clustered_blends(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["beta_pooled"] = np.nan
    out["beta_origin_group"] = np.nan
    out["blend_pooled_usd"] = np.nan
    out["blend_grouped_usd"] = np.nan
    for _, row in out.iterrows():
        if not positive(row["baseline_gross_usd"]):
            continue
        train = out.loc[
            ~out["movie_day_key"].eq(row["movie_day_key"])
            & pd.to_numeric(out["baseline_gross_usd"], errors="coerce").gt(0)
        ]
        pooled_beta = constrained_beta(train)
        group_train = train.loc[train["origin_group"].eq(row["origin_group"])]
        group_beta = constrained_beta(group_train) if len(group_train) >= 10 else pooled_beta
        delta = math.log(row["amc_pred_usd"] / row["baseline_gross_usd"])
        out.loc[row.name, "beta_pooled"] = pooled_beta
        out.loc[row.name, "beta_origin_group"] = group_beta
        out.loc[row.name, "blend_pooled_usd"] = row["baseline_gross_usd"] * math.exp(pooled_beta * delta)
        out.loc[row.name, "blend_grouped_usd"] = row["baseline_gross_usd"] * math.exp(group_beta * delta)
    return out


def stage_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for idx, row in out.iterrows():
        train = out.loc[out["exhibition_date"].lt(row["exhibition_date"])].copy()
        same_origin = train.loc[train["forecast_origin"].astype(str).eq(str(row["forecast_origin"]))]
        valid = same_origin.loc[(same_origin["s_obs"] > 0) & (same_origin["s_final_eod"] > 0)]
        if len(valid) < 3:
            continue
        pace = np.median(valid["s_obs"] / valid["s_final_eod"])
        multiplicative = row["s_obs"] / np.clip(pace, 0.01, 1.0)
        remaining_capacity = (valid["c_scheduled_known"] - valid["c_obs"]).clip(lower=0)
        remaining_sales = (valid["s_final_eod"] - valid["s_obs"]).clip(lower=0)
        q = remaining_sales.sum() / remaining_capacity.sum() if remaining_capacity.sum() > 0 else 0.0
        additive = row["s_obs"] + max(row["c_scheduled_known"] - row["c_obs"], 0.0) * q
        bridge = train.loc[(train["actual_gross_usd"] > 0) & (train["s_final_eod"] > 0)]
        same_day = bridge.loc[bridge["day_of_week"].eq(row["day_of_week"])]
        if len(same_day) >= 3:
            bridge = same_day
        if len(bridge) < 3:
            continue
        x = np.log1p(bridge["s_final_eod"].to_numpy(dtype=float))
        y = np.log(bridge["actual_gross_usd"].to_numpy(dtype=float))
        alpha_fixed = float(np.median(y - x))
        centered_x = x - x.mean()
        slope = float((centered_x @ (y - y.mean()) + 10.0) / (centered_x @ centered_x + 10.0))
        alpha_slope = float(y.mean() - slope * x.mean())
        for name, seats in {"multiplicative": multiplicative, "additive": additive, "hybrid": 0.5 * (multiplicative + additive)}.items():
            out.loc[idx, f"pred_seats_{name}"] = seats
            out.loc[idx, f"pred_gross_{name}"] = math.exp(alpha_fixed + math.log1p(seats))
        out.loc[idx, "pred_gross_shrunk_slope"] = math.exp(alpha_slope + slope * math.log1p(multiplicative))
        out.loc[idx, "bridge_slope"] = slope
        out.loc[idx, "pace_error_log"] = math.log(row["s_final_eod"] / multiplicative) if positive(row["s_final_eod"]) else np.nan
        oracle_bridge = math.exp(alpha_fixed + math.log1p(row["s_final_eod"]))
        out.loc[idx, "bridge_error_log"] = math.log(row["actual_gross_usd"] / oracle_bridge)
        out.loc[idx, "total_error_log"] = math.log(row["actual_gross_usd"] / out.loc[idx, "pred_gross_multiplicative"])
    return out


def add_rolling_holdover_bias_corrections(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply origin-group bias estimated only from earlier true holdover movie-days."""

    out = frame.copy()
    out["bias_origin_group"] = out["forecast_origin"].map(ORIGIN_GROUP)
    out.loc[out["forecast_origin"].eq("EOD"), "bias_origin_group"] = "EOD"
    out["pred_gross_policy"] = np.where(
        out["forecast_origin"].eq("EOD"), out["pred_gross_additive"], out["pred_gross_hybrid"]
    )
    for candidate in ["pred_gross_hybrid", "pred_gross_additive", "pred_gross_policy"]:
        corrected = f"{candidate}_bias_corrected"
        out[corrected] = np.nan
        out[f"{candidate}_bias_log"] = np.nan
        for idx, row in out.iterrows():
            point = number(row.get(candidate))
            if not positive(point) or number(row.get("run_day")) <= 3:
                continue
            train = out.loc[
                out["exhibition_date"].lt(row["exhibition_date"])
                & out["bias_origin_group"].eq(row["bias_origin_group"])
                & pd.to_numeric(out["run_day"], errors="coerce").gt(3)
                & pd.to_numeric(out[candidate], errors="coerce").gt(0)
                & pd.to_numeric(out["actual_gross_usd"], errors="coerce").gt(0)
            ].drop_duplicates(["movie_day_key", "forecast_origin"], keep="last")
            if train["movie_day_key"].nunique() < 10:
                bias = 0.0
            else:
                residuals = np.log(train["actual_gross_usd"] / train[candidate])
                bias = float(residuals.mean())
            out.loc[idx, corrected] = point * math.exp(bias)
            out.loc[idx, f"{candidate}_bias_log"] = bias
    return out


def fit_friday_update(train: pd.DataFrame, target: str) -> float:
    x = np.log(train["actual_fri_usd"] / train["pre_fri_usd"]).to_numpy(dtype=float)
    y = np.log(train[f"actual_{target}_usd"] / train[f"pre_{target}_usd"]).to_numpy(dtype=float)
    return float((x @ y + 5.0 * 0.5) / (x @ x + 5.0))


def weekend_candidates(frame: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    friday = frame.loc[frame["day_of_week"].eq("Friday")].merge(
        baseline,
        on=["movie_id", "release_run_id"],
        how="inner",
        suffixes=("", "_wide"),
    )
    friday = friday.loc[
        pd.to_datetime(friday["exhibition_date"], errors="coerce").dt.date.eq(
            pd.to_datetime(friday["friday_date"], errors="coerce").dt.date
        )
    ].copy()
    rows = []
    for _, row in friday.iterrows():
        train = baseline.loc[~baseline["release_run_id"].eq(row["release_run_id"])].dropna(
            subset=["actual_fri_usd", "actual_sat_usd", "actual_sun_usd", "pre_fri_usd", "pre_sat_usd", "pre_sun_usd"]
        )
        gamma_sat = fit_friday_update(train, "sat")
        gamma_sun = fit_friday_update(train, "sun")
        actual_ow = row["actual_fri_usd"] + row["actual_sat_usd"] + row["actual_sun_usd"]
        for label, friday_point in {
            "A_baseline_pre": row["pre_fri_usd"],
            "B_amc_pre": row["amc_pred_usd"],
            "C_blend_pre": row["blend_grouped_usd"],
        }.items():
            rows.append(candidate_row(row, label, friday_point, row["pre_sat_usd"], row["pre_sun_usd"], actual_ow))
        for label, friday_point in {"D_amc_soft_af2": row["amc_pred_usd"], "E_blend_soft_af2": row["blend_grouped_usd"]}.items():
            surprise = math.log(friday_point / row["pre_fri_usd"])
            sat = row["pre_sat_usd"] * math.exp(gamma_sat * surprise)
            sun = row["pre_sun_usd"] * math.exp(gamma_sun * surprise)
            item = candidate_row(row, label, friday_point, sat, sun, actual_ow)
            item.update({"gamma_sat": gamma_sat, "gamma_sun": gamma_sun})
            rows.append(item)
    return pd.DataFrame(rows)


def candidate_row(row: pd.Series, model: str, fri: float, sat: float, sun: float, actual: float) -> dict[str, object]:
    return {
        "model": model, "movie_id": row["movie_id"], "release_run_id": row["release_run_id"],
        "title": row["title"], "exhibition_date": row["exhibition_date"], "forecast_origin": row["forecast_origin"],
        "friday_point_usd": fri, "saturday_point_usd": sat, "sunday_point_usd": sun,
        "point_usd": fri + sat + sun, "actual_usd": actual,
    }


def metrics(frame: pd.DataFrame, prediction: str, *, groups: list[str]) -> pd.DataFrame:
    work = frame.loc[(frame[prediction] > 0) & (frame["actual_gross_usd"] > 0)].copy()
    work["error_log"] = np.log(work["actual_gross_usd"] / work[prediction])
    work["abs_error_usd"] = (work["actual_gross_usd"] - work[prediction]).abs()
    return work.groupby(groups, dropna=False).agg(
        n_rows=("error_log", "size"), n_movie_days=("movie_day_key", "nunique"),
        ME_log=("error_log", "mean"), MAE_log=("error_log", lambda x: float(np.mean(np.abs(x)))),
        RMSE_log=("error_log", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        MAE_usd=("abs_error_usd", "mean"),
    ).reset_index()


def metrics_with_total(frame: pd.DataFrame, prediction: str) -> pd.DataFrame:
    by_origin = metrics(frame, prediction, groups=["forecast_origin"])
    total_frame = frame.copy()
    total_frame["forecast_origin"] = "ALL"
    return pd.concat([by_origin, metrics(total_frame, prediction, groups=["forecast_origin"])], ignore_index=True)


def weekend_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["error_log"] = np.log(work["actual_usd"] / work["point_usd"])
    rows = []
    for model, group in work.groupby("model"):
        residuals = group["error_log"].to_numpy(dtype=float)
        lo, hi = np.quantile(residuals, [0.10, 0.90]) if len(residuals) >= 5 else (-1.28155 * 0.45, 1.28155 * 0.45)
        misses_low = np.maximum(lo - residuals, 0)
        misses_high = np.maximum(residuals - hi, 0)
        rows.append({
            "model": model, "n_rows": len(group), "n_release_weekends": group["release_run_id"].nunique(),
            "MAE_log": np.mean(np.abs(residuals)), "RMSE_log": np.sqrt(np.mean(residuals**2)),
            "coverage_80": np.mean((residuals >= lo) & (residuals <= hi)),
            "lower_tail_miss": np.mean(residuals < lo), "upper_tail_miss": np.mean(residuals > hi),
            "interval_score_log_80": (hi - lo) + 10 * np.mean(misses_low + misses_high),
            "interval_width_to_point": math.exp(hi) - math.exp(lo),
        })
    return pd.DataFrame(rows)


def error_decomposition(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.dropna(subset=["pace_error_log", "bridge_error_log", "total_error_log"])
    rows = []
    for origin, group in valid.groupby("forecast_origin"):
        pace = group["pace_error_log"].to_numpy()
        bridge = group["bridge_error_log"].to_numpy()
        total_var = np.var(group["total_error_log"].to_numpy())
        rows.append({
            "forecast_origin": origin, "n_movie_days": group["movie_day_key"].nunique(),
            "pace_median_bias": np.median(pace), "pace_MAE": np.mean(np.abs(pace)), "pace_RMSE": np.sqrt(np.mean(pace**2)),
            "pace_MAD": np.median(np.abs(pace - np.median(pace))),
            "bridge_median_bias": np.median(bridge), "bridge_MAE": np.mean(np.abs(bridge)), "bridge_RMSE": np.sqrt(np.mean(bridge**2)),
            "bridge_MAD": np.median(np.abs(bridge - np.median(bridge))),
            "pace_bridge_correlation": np.corrcoef(pace, bridge)[0, 1] if len(group) > 2 else np.nan,
            "pace_variance_share": np.var(pace) / total_var if total_var > 0 else np.nan,
            "bridge_variance_share": np.var(bridge) / total_var if total_var > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def seat_projection_promotion_table(frame: pd.DataFrame, *, output_dir: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    """Compare seat candidates after the common gross bridge and clustered interval calibration."""

    work = frame.loc[pd.to_numeric(frame["run_day"], errors="coerce").gt(3)].copy()
    work["origin_group"] = work["forecast_origin"].map(ORIGIN_GROUP)
    work.loc[work["forecast_origin"].eq("EOD"), "origin_group"] = "EOD"
    if "pred_gross_policy" not in work.columns:
        work["pred_gross_policy"] = np.where(
            work["forecast_origin"].eq("EOD"), work["pred_gross_additive"], work["pred_gross_hybrid"]
        )
    candidates = {
        "multiplicative": "pred_gross_multiplicative",
        "additive": "pred_gross_additive",
        "hybrid": "pred_gross_hybrid",
        "hybrid_live_additive_eod": "pred_gross_policy",
        "hybrid_bias_corrected": "pred_gross_hybrid_bias_corrected",
        "additive_bias_corrected": "pred_gross_additive_bias_corrected",
        "policy_bias_corrected": "pred_gross_policy_bias_corrected",
    }
    detail = []
    for model, column in candidates.items():
        usable = work.loc[work[column].gt(0) & work["actual_gross_usd"].gt(0)].copy()
        usable["seat_projection"] = model
        usable["gross_prediction_usd"] = usable[column]
        usable["gross_error_log"] = np.log(usable["actual_gross_usd"] / usable[column])
        usable = usable.drop_duplicates(["movie_day_key", "forecast_origin"], keep="last").copy()
        for idx, row in usable.iterrows():
            calibration = usable.loc[
                usable["origin_group"].eq(row["origin_group"])
                & ~usable["release_run_id"].eq(row["release_run_id"])
            ]
            residuals = calibration["gross_error_log"].dropna().to_numpy(dtype=float)
            if residuals.size >= 10:
                lo, hi = np.quantile(residuals, [0.10, 0.90])
            else:
                lo, hi = (-1.28155 * 0.45, 1.28155 * 0.45)
            error = float(row["gross_error_log"])
            usable.loc[idx, "lo80_log"] = lo
            usable.loc[idx, "hi80_log"] = hi
            usable.loc[idx, "covered_80"] = lo <= error <= hi
            usable.loc[idx, "interval_score_80"] = (hi - lo) + 10.0 * max(lo - error, 0.0) + 10.0 * max(error - hi, 0.0)
        detail.append(usable)
    scored = pd.concat(detail, ignore_index=True)
    rows = []
    for (model, origin_group), group in scored.groupby(["seat_projection", "origin_group"]):
        baseline = scored.loc[
            scored["seat_projection"].eq("multiplicative")
            & scored["origin_group"].eq(origin_group),
            ["movie_day_key", "forecast_origin", "gross_error_log"],
        ].rename(columns={"gross_error_log": "multiplicative_error_log"})
        paired = group.merge(baseline, on=["movie_day_key", "forecast_origin"], how="inner")
        candidate_mae = float(np.mean(np.abs(paired["gross_error_log"])))
        baseline_mae = float(np.mean(np.abs(paired["multiplicative_error_log"])))
        candidate_rmse = float(np.sqrt(np.mean(np.square(paired["gross_error_log"]))))
        baseline_rmse = float(np.sqrt(np.mean(np.square(paired["multiplicative_error_log"]))))
        weekend = paired.groupby("release_run_id").agg(
            candidate=("gross_error_log", lambda x: float(np.mean(np.abs(x)))),
            baseline=("multiplicative_error_log", lambda x: float(np.mean(np.abs(x)))),
        )
        rows.append({
            "seat_projection": model,
            "origin_group": origin_group,
            "n_movie_days": paired["movie_day_key"].nunique(),
            "n_release_weekends": paired["release_run_id"].nunique(),
            "gross_ME_log": paired["gross_error_log"].mean(),
            "gross_MAE_log": candidate_mae,
            "gross_RMSE_log": candidate_rmse,
            "delta_MAE_vs_multiplicative": candidate_mae - baseline_mae,
            "delta_RMSE_vs_multiplicative": candidate_rmse - baseline_rmse,
            "coverage_80": group["covered_80"].astype(float).mean(),
            "interval_score_80": group["interval_score_80"].mean(),
            "leave_one_weekend_out_win_rate": float((weekend["candidate"] < weekend["baseline"]).mean()),
        })
    scored.to_csv(output_dir / "amc_seat_projection_gross_detail.csv", index=False)
    return pd.DataFrame(rows).sort_values(["origin_group", "gross_MAE_log", "seat_projection"])


def number(value: object) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(out) if np.isfinite(out) else math.nan


def positive(value: object) -> bool:
    return math.isfinite(number(value)) and number(value) > 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    conn = connect_database(args.database_url)
    try:
        panel, baseline = build_panel(conn, args.baseline_csv)
    finally:
        conn.close()
    predictions = add_clustered_blends(historical_amc_predictions(panel))
    predictions = stage_predictions(predictions)
    predictions = add_rolling_holdover_bias_corrections(predictions)

    comparison_parts = []
    comparison_models = {
        "baseline": "baseline_gross_usd", "AMC_only": "amc_pred_usd",
        "AMC_blend_pooled": "blend_pooled_usd", "AMC_blend_early_late": "blend_grouped_usd",
        "seat_multiplicative": "pred_gross_multiplicative", "seat_additive": "pred_gross_additive",
        "seat_hybrid": "pred_gross_hybrid", "bridge_shrunk_slope": "pred_gross_shrunk_slope",
    }
    for model, column in comparison_models.items():
        evaluation = predictions
        if model == "AMC_only":
            evaluation = predictions
        result = metrics_with_total(evaluation, column)
        result.insert(0, "model", model)
        comparison_parts.append(result)
    overlap = predictions.loc[pd.to_numeric(predictions["baseline_gross_usd"], errors="coerce").gt(0)]
    overlap_result = metrics_with_total(overlap, "amc_pred_usd")
    overlap_result.insert(0, "model", "AMC_only_baseline_overlap")
    comparison_parts.append(overlap_result)
    point_comparison = pd.concat(comparison_parts, ignore_index=True)
    weekends = weekend_candidates(predictions, baseline)
    weekend_summary = weekend_metrics(weekends)
    decomposition = error_decomposition(predictions)
    seat_promotion = seat_projection_promotion_table(predictions, output_dir=args.output_dir)

    predictions.to_csv(args.output_dir / "amc_point_gate_comparison.csv", index=False)
    point_comparison.to_csv(args.output_dir / "amc_point_model_metrics.csv", index=False)
    decomposition.to_csv(args.output_dir / "amc_error_decomposition_by_origin.csv", index=False)
    seat_promotion.to_csv(args.output_dir / "amc_seat_projection_gross_promotion.csv", index=False)
    predictions[[c for c in predictions.columns if c in ["movie_id", "title", "exhibition_date", "forecast_origin", "s_obs", "s_final_eod", "pred_seats_multiplicative", "pred_seats_additive", "pred_seats_hybrid", "pace_error_log"]]].to_csv(args.output_dir / "amc_pace_residual_detail.csv", index=False)
    predictions[[c for c in predictions.columns if c in ["movie_id", "title", "exhibition_date", "forecast_origin", "s_final_eod", "actual_gross_usd", "pred_gross_multiplicative", "pred_gross_shrunk_slope", "bridge_slope", "bridge_error_log"]]].to_csv(args.output_dir / "amc_bridge_residual_detail.csv", index=False)
    weekends.to_csv(args.output_dir / "amc_weekend_model_comparison.csv", index=False)
    weekend_summary.to_csv(args.output_dir / "amc_weekend_model_metrics.csv", index=False)
    friday_audit_columns = [
        "movie_id", "release_run_id", "title", "exhibition_date", "forecast_origin", "is_opening_day",
        "run_day", "premium_format_share", "coverage", "n_scheduled_showtimes", "n_scheduled_theatres",
        "n_theatres_observed", "staleness_p50_minutes", "feature_quality_bucket", "amc_pred_usd",
        "actual_gross_usd",
    ]
    friday_audit = predictions.loc[predictions["day_of_week"].eq("Friday"), friday_audit_columns].copy()
    friday_audit["thursday_amc_seats_included"] = False
    friday_audit["preview_alignment_status"] = np.where(
        friday_audit["is_opening_day"],
        "reported_daily_target_may_include_previews; preview_gross_not_separately_available",
        "not_opening_day",
    )
    friday_audit["log_residual"] = np.log(friday_audit["actual_gross_usd"] / friday_audit["amc_pred_usd"])
    friday_audit.sort_values("log_residual").to_csv(args.output_dir / "amc_friday_target_audit.csv", index=False)
    friday_audit["target_scope"] = np.where(friday_audit["is_opening_day"], "opening_day", "holdover")
    friday_summary = friday_audit.groupby(["forecast_origin", "target_scope"], dropna=False).agg(
        n_rows=("log_residual", "size"), n_movie_days=("movie_id", "nunique"),
        ME_log=("log_residual", "mean"), MAE_log=("log_residual", lambda x: float(np.mean(np.abs(x)))),
        median_premium_format_share=("premium_format_share", "median"), median_coverage=("coverage", "median"),
    ).reset_index()
    friday_summary.to_csv(args.output_dir / "amc_friday_target_summary.csv", index=False)
    bridge_slope_summary = predictions.dropna(subset=["bridge_slope"]).groupby("forecast_origin").agg(
        n_rows=("bridge_slope", "size"), n_movie_days=("movie_day_key", "nunique"),
        median_shrunk_slope=("bridge_slope", "median"), p10_shrunk_slope=("bridge_slope", lambda x: x.quantile(0.10)),
        p90_shrunk_slope=("bridge_slope", lambda x: x.quantile(0.90)),
    ).reset_index()
    bridge_slope_summary.to_csv(args.output_dir / "amc_bridge_slope_summary.csv", index=False)
    print(point_comparison.to_string(index=False))
    print("\nWeekend candidates\n", weekend_summary.to_string(index=False))
    print("\nError decomposition\n", decomposition.to_string(index=False))
    print("\nSeat projection promotion\n", seat_promotion.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
