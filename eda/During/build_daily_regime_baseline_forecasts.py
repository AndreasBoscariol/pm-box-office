#!/usr/bin/env python3
"""Build daily regime baselines for live weekend AMC seat forecasts.

This materializes the one CSV expected by ``live_weekend_seat_regime_forecast``.
It bridges the pre-weekend daily shape forecast with the frozen post-Friday
AF2+C0 baseline and the frozen post-Saturday AS0 baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTICS_DIR = REPO_ROOT / "data" / "diagnostics"
PREDICTIONS_DIR = REPO_ROOT / "data" / "predictions"

DEFAULT_PRE_WEEKEND_CSV = DIAGNOSTICS_DIR / "weekend_shape_operational_predictions.csv"
DEFAULT_AFTER_FRIDAY_CSV = DIAGNOSTICS_DIR / "af2_internal_calibration_predictions.csv"
DEFAULT_AFTER_SATURDAY_CSV = DIAGNOSTICS_DIR / "as0_corridor_quantile_interval_predictions.csv"
DEFAULT_WEEKEND_SHAPE_BASE_CSV = DIAGNOSTICS_DIR / "weekend_shape_base.csv"
DEFAULT_OUTPUT_CSV = PREDICTIONS_DIR / "daily_regime_baseline_forecasts.csv"

REQUIRED_OUTPUT_COLUMNS = [
    "movie_id",
    "release_date",
    "friday_date",
    "saturday_date",
    "sunday_date",
    "pre_fri_usd",
    "pre_sat_usd",
    "pre_sun_usd",
    "after_fri_sat_usd",
    "after_fri_sun_usd",
    "after_sat_sun_usd",
    "actual_fri_usd",
    "actual_sat_usd",
    "actual_sun_usd",
    "actual_ow_usd",
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return pd.read_csv(path)


def collapse_pre_weekend(frame: pd.DataFrame, origin_day: int) -> pd.DataFrame:
    pre = frame.loc[frame["forecast_origin_day"].eq(origin_day)].copy()
    if pre.empty:
        raise ValueError(f"No pre-weekend rows found for forecast_origin_day={origin_day}")

    pre["opening_weekend_start"] = pd.to_datetime(pre["opening_weekend_start"], errors="coerce")
    keep = [
        "release_run_id",
        "movie_id",
        "title",
        "opening_weekend_start",
        "forecast_origin_day",
        "forecast_origin_date",
        "total_forecast_usd",
        "total_forecast_source",
        "shape_model",
        "pred_friday_gross",
        "pred_saturday_gross",
        "pred_sunday_gross",
        "friday_gross_usd",
        "saturday_gross_usd",
        "sunday_gross_usd",
        "actual_opening_weekend_gross_usd",
    ]
    present = [column for column in keep if column in pre.columns]
    pre = pre[present].sort_values(["opening_weekend_start", "release_run_id"])
    return pre.drop_duplicates("release_run_id", keep="last")


def select_after_friday(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.loc[frame["origin"].eq("after_friday")].copy()
    if "calibration_model" in out.columns:
        preferred = out.loc[out["calibration_model"].eq("AF2+C0")].copy()
        if not preferred.empty:
            out = preferred
    out["opening_weekend_start"] = pd.to_datetime(out["opening_weekend_start"], errors="coerce")
    out = out.sort_values(["opening_weekend_start", "release_run_id"]).drop_duplicates(
        "release_run_id",
        keep="last",
    )

    sat = pd.to_numeric(out["pred_saturday_gross_usd"], errors="coerce")
    sun = pd.to_numeric(out["pred_sunday_gross_usd"], errors="coerce")
    fri_actual = pd.to_numeric(out["friday_gross_usd"], errors="coerce")
    ow_pred = pd.to_numeric(
        out.get("calibrated_pred_ow", out.get("pred_opening_weekend_gross_usd")),
        errors="coerce",
    )

    raw_remaining = sat + sun
    calibrated_remaining = ow_pred - fri_actual
    scale = calibrated_remaining / raw_remaining
    valid = raw_remaining.gt(0) & calibrated_remaining.gt(0) & scale.replace([np.inf, -np.inf], np.nan).notna()

    out["after_fri_sat_usd"] = sat
    out["after_fri_sun_usd"] = sun
    out.loc[valid, "after_fri_sat_usd"] = sat.loc[valid] * scale.loc[valid]
    out.loc[valid, "after_fri_sun_usd"] = sun.loc[valid] * scale.loc[valid]

    return out[
        [
            "release_run_id",
            "after_fri_sat_usd",
            "after_fri_sun_usd",
            "calibration_model",
        ]
    ].rename(columns={"calibration_model": "after_friday_baseline_model"})


def select_after_saturday(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.loc[frame["origin"].eq("after_saturday")].copy()
    if "calibration_model" in out.columns:
        preferred = out.loc[out["calibration_model"].eq("AS0")].copy()
        if not preferred.empty:
            out = preferred
    out["opening_weekend_start"] = pd.to_datetime(out["opening_weekend_start"], errors="coerce")
    out = out.sort_values(["opening_weekend_start", "release_run_id"]).drop_duplicates(
        "release_run_id",
        keep="last",
    )
    out["after_sat_sun_usd"] = pd.to_numeric(out["pred_sunday_gross_usd"], errors="coerce")
    return out[
        [
            "release_run_id",
            "after_sat_sun_usd",
            "calibration_model",
        ]
    ].rename(columns={"calibration_model": "after_saturday_baseline_model"})


def build_daily_regime_baselines(
    *,
    pre_weekend_csv: Path,
    after_friday_csv: Path,
    after_saturday_csv: Path,
    weekend_shape_base_csv: Path,
    pre_weekend_origin_day: int,
) -> pd.DataFrame:
    pre = collapse_pre_weekend(read_csv(pre_weekend_csv), pre_weekend_origin_day)
    after_friday = select_after_friday(read_csv(after_friday_csv))
    after_saturday = select_after_saturday(read_csv(after_saturday_csv))

    base = read_csv(weekend_shape_base_csv)
    base = base[["release_run_id", "opening_date"]].copy()
    base["opening_date"] = pd.to_datetime(base["opening_date"], errors="coerce")
    base = base.drop_duplicates("release_run_id", keep="last")

    out = (
        pre.merge(after_friday, on="release_run_id", how="inner")
        .merge(after_saturday, on="release_run_id", how="inner")
        .merge(base, on="release_run_id", how="left")
    )

    out["release_date"] = out["opening_date"].fillna(out["opening_weekend_start"])
    out["friday_date"] = out["opening_weekend_start"]
    out["saturday_date"] = out["opening_weekend_start"] + pd.to_timedelta(1, unit="D")
    out["sunday_date"] = out["opening_weekend_start"] + pd.to_timedelta(2, unit="D")

    rename = {
        "pred_friday_gross": "pre_fri_usd",
        "pred_saturday_gross": "pre_sat_usd",
        "pred_sunday_gross": "pre_sun_usd",
        "friday_gross_usd": "actual_fri_usd",
        "saturday_gross_usd": "actual_sat_usd",
        "sunday_gross_usd": "actual_sun_usd",
        "actual_opening_weekend_gross_usd": "actual_ow_usd",
    }
    out = out.rename(columns=rename)

    for column in [c for c in REQUIRED_OUTPUT_COLUMNS if c.endswith("_usd")]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ["release_date", "friday_date", "saturday_date", "sunday_date"]:
        out[column] = pd.to_datetime(out[column], errors="coerce").dt.date

    output_columns = REQUIRED_OUTPUT_COLUMNS + [
        "release_run_id",
        "title",
        "pre_weekend_origin_day",
        "forecast_origin_date",
        "total_forecast_usd",
        "total_forecast_source",
        "shape_model",
        "after_friday_baseline_model",
        "after_saturday_baseline_model",
    ]
    out["pre_weekend_origin_day"] = pre_weekend_origin_day
    out = out[output_columns].replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=REQUIRED_OUTPUT_COLUMNS).sort_values(["friday_date", "movie_id"])
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-weekend-csv", type=Path, default=DEFAULT_PRE_WEEKEND_CSV)
    parser.add_argument("--after-friday-csv", type=Path, default=DEFAULT_AFTER_FRIDAY_CSV)
    parser.add_argument("--after-saturday-csv", type=Path, default=DEFAULT_AFTER_SATURDAY_CSV)
    parser.add_argument("--weekend-shape-base-csv", type=Path, default=DEFAULT_WEEKEND_SHAPE_BASE_CSV)
    parser.add_argument("--pre-weekend-origin-day", type=int, default=-1)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    baselines = build_daily_regime_baselines(
        pre_weekend_csv=args.pre_weekend_csv,
        after_friday_csv=args.after_friday_csv,
        after_saturday_csv=args.after_saturday_csv,
        weekend_shape_base_csv=args.weekend_shape_base_csv,
        pre_weekend_origin_day=args.pre_weekend_origin_day,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    baselines.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv} ({len(baselines):,} rows)")


if __name__ == "__main__":
    main()
