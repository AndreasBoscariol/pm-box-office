#!/usr/bin/env python3
"""Freeze production box-office forecast artifacts from diagnostic outputs."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import DEFAULT_ARTIFACT_ROOT, utc_now_iso, write_json, write_manifest
from .constants import ACTUAL_COLUMNS, BASELINE_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTICS_DIR = REPO_ROOT / "data" / "diagnostics"
PREDICTIONS_DIR = REPO_ROOT / "data" / "predictions"


def default_model_version() -> str:
    return datetime.now(timezone.utc).strftime("boxoffice_%Y_%m_%d_001")


def copy_if_exists(source: Path, destination: Path) -> str | None:
    if not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination.name


def build_locked_point_policy(point_policy_csv: Path | None) -> dict[str, object]:
    policy = {
        "default": "dollar_median_consensus_usd",
        "by_origin_day": {"-1": "recency_weighted_log_consensus_lambda_1_usd"},
        "excluded_sources": ["the_numbers_predictions", "boxofficeguru"],
        "promotion_rule": "training scripts decide policies; production scripts apply policies",
    }
    if point_policy_csv and point_policy_csv.exists():
        frame = pd.read_csv(point_policy_csv)
        method_col = "selected_forecast_method" if "selected_forecast_method" in frame.columns else "forecast_method"
        if {"origin_day", method_col}.issubset(frame.columns):
            policy["by_origin_day"] = {
                str(int(row.origin_day)): str(getattr(row, method_col))
                for row in frame.itertuples(index=False)
                if pd.notna(row.origin_day) and pd.notna(getattr(row, method_col))
            }
    return policy


def _finite_positive_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce")
    return values[np.isfinite(values) & values.gt(0)]


def _pre_release_upper_caps(origin_day: int) -> tuple[float, float]:
    if origin_day >= -1:
        return 1.25, 1.40
    if origin_day >= -2:
        return 1.25, 1.45
    if origin_day >= -7:
        return 1.30, 1.55
    return 1.30, 1.60


def _residual_quantile_payload(
    residuals: pd.Series,
    *,
    upper_caps: tuple[float, float] | None = None,
) -> dict[str, object]:
    clean = residuals.replace([np.inf, -np.inf], np.nan).dropna()
    quantiles = clean.quantile([0.025, 0.10, 0.50, 0.90, 0.975])
    hi80_log = float(quantiles.loc[0.90])
    hi95_log = float(quantiles.loc[0.975])
    method = "empirical_log_residual_quantile"
    if upper_caps is not None:
        hi80_cap, hi95_cap = upper_caps
        hi80_log = min(hi80_log, float(np.log(hi80_cap)))
        hi95_log = min(hi95_log, float(np.log(hi95_cap)))
        hi95_log = max(hi95_log, hi80_log)
        method = "operational_capped_empirical_log_residual_quantile"
    return {
        "n": int(clean.size),
        "method": method,
        "sigma_log": float(clean.std(ddof=1)) if clean.size > 1 else 0.0,
        "lo95_log": float(quantiles.loc[0.025]),
        "lo80_log": float(quantiles.loc[0.10]),
        "median_log": float(quantiles.loc[0.50]),
        "hi80_log": hi80_log,
        "hi95_log": hi95_log,
        "residual_samples_log": [float(value) for value in clean.to_numpy()],
    }


def build_interval_policy(panel_csv: Path | None) -> dict[str, object]:
    policy: dict[str, object] = {
        "default_sigma_log": 0.55,
        "sigma_log_by_origin_day": {},
        "method": "normal_log_error_interval",
        "levels": [80, 95],
    }
    if not panel_csv or not panel_csv.exists():
        return policy
    frame = pd.read_csv(panel_csv)
    selected_columns = {"primary_point_forecast_usd", "selected_lo_95", "selected_hi_95"}
    if selected_columns.issubset(frame.columns):
        policy["method"] = "selected_interval_panel_with_sigma_fallback"
        if "sigma_global" in frame.columns:
            global_sigmas = _finite_positive_series(frame, "sigma_global")
            if not global_sigmas.empty:
                policy["default_sigma_log"] = float(max(0.05, global_sigmas.median()))

        point = pd.to_numeric(frame["primary_point_forecast_usd"], errors="coerce")
        lo95 = pd.to_numeric(frame["selected_lo_95"], errors="coerce")
        hi95 = pd.to_numeric(frame["selected_hi_95"], errors="coerce")
        valid = point.gt(0) & lo95.gt(0) & hi95.gt(point) & np.isfinite(point) & np.isfinite(lo95) & np.isfinite(hi95)
        if "origin_day" in frame.columns and valid.any():
            work = frame.loc[valid, ["origin_day"]].copy()
            work["implied_sigma"] = np.log(hi95.loc[valid] / point.loc[valid]) / 1.95996
            sigmas = work.groupby("origin_day")["implied_sigma"].median().dropna()
            if not sigmas.empty:
                policy["sigma_log_by_origin_day"] = {str(int(k)): float(max(0.05, v)) for k, v in sigmas.items()}
        actual_col = "actual_opening_weekend_gross_usd"
        if actual_col in frame.columns and "origin_day" in frame.columns:
            actual = pd.to_numeric(frame[actual_col], errors="coerce")
            valid_residual = valid & actual.gt(0) & np.isfinite(actual)
            if valid_residual.any():
                work = frame.loc[valid_residual, ["origin_day"]].copy()
                work["log_residual"] = np.log(actual.loc[valid_residual] / point.loc[valid_residual])
                policy["log_residual_quantiles_by_origin_day"] = {
                    str(int(origin_day)): _residual_quantile_payload(
                        group["log_residual"],
                        upper_caps=_pre_release_upper_caps(int(origin_day)),
                    )
                    for origin_day, group in work.groupby("origin_day")
                    if len(group) >= 5
                }
                policy["operator_interval_note"] = (
                    "Pre-release estimate intervals use empirical lower tails but cap upper tails "
                    "for operator display; uncapped breakout cases are tracked in diagnostics."
                )
        return policy

    forecast_col = "primary_point_forecast_usd" if "primary_point_forecast_usd" in frame.columns else None
    actual_col = "actual_opening_weekend_gross_usd"
    if not forecast_col or actual_col not in frame.columns or "origin_day" not in frame.columns:
        return policy
    actual = pd.to_numeric(frame[actual_col], errors="coerce")
    pred = pd.to_numeric(frame[forecast_col], errors="coerce")
    mask = actual.gt(0) & pred.gt(0)
    work = frame.loc[mask, ["origin_day"]].copy()
    work["log_error"] = np.log(actual.loc[mask] / pred.loc[mask])
    sigmas = work.groupby("origin_day")["log_error"].std(ddof=1).dropna()
    if not sigmas.empty:
        policy["sigma_log_by_origin_day"] = {str(int(k)): float(max(0.05, v)) for k, v in sigmas.items()}
        policy["default_sigma_log"] = float(max(0.05, work["log_error"].std(ddof=1)))
    return policy


def build_daily_interval_policy(daily_baseline_csv: Path | None) -> dict[str, object]:
    policy: dict[str, object] = {
        "method": "empirical_daily_log_residual_quantile",
        "levels": [80, 95],
        "by_regime_day": {},
    }
    if not daily_baseline_csv or not daily_baseline_csv.exists():
        return policy
    frame = pd.read_csv(daily_baseline_csv)
    for (regime, day), (forecast_col, _source) in BASELINE_COLUMNS.items():
        actual_col = ACTUAL_COLUMNS[day]
        if forecast_col not in frame.columns or actual_col not in frame.columns:
            continue
        forecast = pd.to_numeric(frame[forecast_col], errors="coerce")
        actual = pd.to_numeric(frame[actual_col], errors="coerce")
        mask = forecast.gt(0) & actual.gt(0) & np.isfinite(forecast) & np.isfinite(actual)
        if mask.sum() < 5:
            continue
        residuals = np.log(actual.loc[mask] / forecast.loc[mask])
        by_regime = policy["by_regime_day"].setdefault(regime, {})
        by_regime[day] = _residual_quantile_payload(residuals)
    return policy


def maybe_run_diagnostics(database_url: str | None) -> None:
    cmd = ["python", "eda/Prior/rolling_forecast_consensus_benchmark.py"]
    if database_url:
        cmd.extend(["--database-url", database_url])
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-version", default=default_model_version())
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--database-url")
    parser.add_argument("--run-consensus-diagnostics", action="store_true")
    parser.add_argument("--training-data-cutoff-utc", default=utc_now_iso())
    parser.add_argument("--pre-release-panel", type=Path, default=DIAGNOSTICS_DIR / "rolling_forecast_origin_interval_panel.csv")
    parser.add_argument("--point-policy-csv", type=Path, default=DIAGNOSTICS_DIR / "rolling_forecast_origin_point_model_policy.csv")
    parser.add_argument("--daily-baseline-csv", type=Path, default=PREDICTIONS_DIR / "daily_regime_baseline_forecasts.csv")
    parser.add_argument("--live-plugin-nowcasts-csv", type=Path, default=PREDICTIONS_DIR / "live_daily_plugin_nowcasts.csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.run_consensus_diagnostics:
        maybe_run_diagnostics(args.database_url)

    artifact_dir = args.artifact_root / args.model_version
    artifact_dir.mkdir(parents=True, exist_ok=True)

    pre_release_panel = copy_if_exists(args.pre_release_panel, artifact_dir / "pre_release_panel.csv")
    daily_baseline = copy_if_exists(args.daily_baseline_csv, artifact_dir / "daily_regime_baseline_forecasts.csv")
    live_plugin = copy_if_exists(args.live_plugin_nowcasts_csv, artifact_dir / "live_daily_plugin_nowcasts.csv")

    point_policy = build_locked_point_policy(args.point_policy_csv)
    interval_policy = build_interval_policy(args.pre_release_panel)
    daily_interval_policy = build_daily_interval_policy(args.daily_baseline_csv)
    write_json(artifact_dir / "pre_release_point_policy.json", point_policy)
    write_json(artifact_dir / "pre_release_interval_policy.json", interval_policy)
    write_json(artifact_dir / "daily_interval_policy.json", daily_interval_policy)

    manifest = {
        "model_version": args.model_version,
        "created_at_utc": utc_now_iso(),
        "training_data_cutoff_utc": args.training_data_cutoff_utc,
        "pre_release": {
            "point_policy_path": "pre_release_point_policy.json",
            "interval_policy_path": "pre_release_interval_policy.json",
            "panel_path": pre_release_panel,
            "point_policy": point_policy,
            "interval_policy": interval_policy,
        },
        "daily_baseline": {
            "baseline_path": daily_baseline,
        },
        "amc": {
            "live_plugin_nowcasts_path": live_plugin,
            "origins": ["10:00", "12:00", "14:00", "16:00", "18:00", "20:00", "EOD"],
        },
        "composition": {
            "n_sim": 50000,
            "interval_levels": [80, 95],
            "daily_interval_policy_path": "daily_interval_policy.json",
            "use_daily_error_covariance": False,
            "fallback_to_independent_components": True,
            "fallback_daily_sigma_log": 0.45,
        },
    }
    write_manifest(artifact_dir / "manifest.yml", manifest)
    write_json(artifact_dir / "manifest.json", manifest)
    print(f"Wrote artifacts to {artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
