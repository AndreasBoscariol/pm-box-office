#!/usr/bin/env python3
"""Diagnose forecast interval width, units, fallbacks, and calibration segments."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FORECAST_TABLE = "analytics.movie_opening_weekend_forecasts"


def safe_log_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    out = pd.Series(np.nan, index=num.index, dtype="float64")
    mask = num.gt(0) & den.gt(0)
    out.loc[mask] = np.log(num.loc[mask] / den.loc[mask])
    return out


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((column for column in candidates if column in df.columns), None)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def rows_as_frame(cursor: Any) -> pd.DataFrame:
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def load_forecasts(args: argparse.Namespace) -> pd.DataFrame:
    if args.forecasts_csv:
        return read_table(args.forecasts_csv)
    if not args.model_version:
        raise ValueError("Provide --forecasts-csv or --model-version.")

    from pm_box_office.db.connection import connect_database

    conn = connect_database(args.database_url)
    try:
        cursor = conn.execute(
            f"""
            SELECT *
            FROM {FORECAST_TABLE}
            WHERE model_version = %s
            """,
            (args.model_version,),
        )
        return rows_as_frame(cursor)
    finally:
        conn.close()


def add_interval_multipliers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in ["point_usd", "lo80_usd", "hi80_usd", "lo95_usd", "hi95_usd", "actual_usd"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    out["lo80_mult"] = out["lo80_usd"] / out["point_usd"]
    out["hi80_mult"] = out["hi80_usd"] / out["point_usd"]
    out["lo95_mult"] = out["lo95_usd"] / out["point_usd"]
    out["hi95_mult"] = out["hi95_usd"] / out["point_usd"]
    out["lo80_log_implied"] = safe_log_ratio(out["lo80_usd"], out["point_usd"])
    out["hi80_log_implied"] = safe_log_ratio(out["hi80_usd"], out["point_usd"])
    out["lo95_log_implied"] = safe_log_ratio(out["lo95_usd"], out["point_usd"])
    out["hi95_log_implied"] = safe_log_ratio(out["hi95_usd"], out["point_usd"])
    out["width80_to_point"] = (out["hi80_usd"] - out["lo80_usd"]) / out["point_usd"]
    out["width95_to_point"] = (out["hi95_usd"] - out["lo95_usd"]) / out["point_usd"]
    return out.replace([np.inf, -np.inf], np.nan)


def summarize_db_intervals(forecasts: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        column
        for column in ["regime", "origin_key", "origin_day", "forecast_origin", "target", "interval_model"]
        if column in forecasts.columns
    ]
    metrics = {
        "point_usd": ["count", "median"],
        "lo80_mult": ["median", "min"],
        "hi80_mult": ["median", "max"],
        "lo95_mult": ["median", "min"],
        "hi95_mult": ["median", "max"],
        "width80_to_point": ["median"],
        "width95_to_point": ["median"],
        "hi95_log_implied": ["median", "max"],
    }
    summary = forecasts.groupby(group_cols, dropna=False).agg(metrics)
    summary.columns = ["_".join(column).strip("_") for column in summary.columns]
    return summary.reset_index()


def suspicious_rows(forecasts: pd.DataFrame) -> pd.DataFrame:
    mask = (
        forecasts["hi80_mult"].gt(2.0)
        | forecasts["hi95_mult"].gt(3.0)
        | forecasts["lo80_mult"].lt(0.35)
        | forecasts["lo95_mult"].lt(0.20)
        | forecasts["width95_to_point"].gt(3.0)
    )
    cols = [
        "movie_id",
        "release_run_id",
        "title",
        "regime",
        "origin_key",
        "origin_day",
        "forecast_origin",
        "target",
        "point_usd",
        "lo80_usd",
        "hi80_usd",
        "lo95_usd",
        "hi95_usd",
        "lo80_mult",
        "hi80_mult",
        "lo95_mult",
        "hi95_mult",
        "lo95_log_implied",
        "hi95_log_implied",
        "interval_model",
        "point_model",
        "component_source",
        "source_count",
        "estimate_sources",
    ]
    cols = [column for column in cols if column in forecasts.columns]
    return forecasts.loc[mask, cols].sort_values(["hi95_mult"], ascending=False)


def score_realized_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    if "actual_usd" not in forecasts.columns:
        return pd.DataFrame()
    out = forecasts.copy()
    valid = out["actual_usd"].gt(0) & out["point_usd"].gt(0)
    out = out.loc[valid].copy()
    if out.empty:
        return pd.DataFrame()

    out["realized_log_error"] = np.log(out["actual_usd"] / out["point_usd"])
    out["covered80"] = out["actual_usd"].between(out["lo80_usd"], out["hi80_usd"])
    out["covered95"] = out["actual_usd"].between(out["lo95_usd"], out["hi95_usd"])
    group_cols = [column for column in ["regime", "origin_key", "origin_day", "forecast_origin", "target"] if column in out.columns]
    return (
        out.groupby(group_cols, dropna=False)
        .agg(
            n=("point_usd", "size"),
            mae_log=("realized_log_error", lambda values: float(np.mean(np.abs(values)))),
            rmse_log=("realized_log_error", lambda values: float(np.sqrt(np.mean(np.square(values))))),
            me_log=("realized_log_error", "mean"),
            coverage80=("covered80", "mean"),
            coverage95=("covered95", "mean"),
            median_width80_to_point=("width80_to_point", "median"),
            median_width95_to_point=("width95_to_point", "median"),
            median_hi95_mult=("hi95_mult", "median"),
        )
        .reset_index()
    )


def flatten_json_quantile_policy(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []

    by_origin = payload.get("log_residual_quantiles_by_origin_day", {})
    if isinstance(by_origin, dict):
        for origin_day, item in by_origin.items():
            if isinstance(item, dict):
                rows.append({"policy_scope": "pre_release_origin", "origin_day": origin_day, **item})

    by_regime = payload.get("by_regime_day", {})
    if isinstance(by_regime, dict):
        for regime, by_day in by_regime.items():
            if not isinstance(by_day, dict):
                continue
            for day, item in by_day.items():
                if isinstance(item, dict):
                    rows.append({"policy_scope": "daily_regime_day", "regime": regime, "component_day": day, **item})

    return pd.DataFrame(rows)


def inspect_quantile_artifact(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        out = flatten_json_quantile_policy(path)
    else:
        out = read_table(path)
    if out.empty:
        return out

    candidate_cols = [
        column
        for column in out.columns
        if any(token in column.lower() for token in ["lo80", "hi80", "lo95", "hi95", "q10", "q90", "q025", "q975"])
    ]
    for column in candidate_cols:
        out[column] = pd.to_numeric(out[column], errors="coerce")
        out[f"exp_{column}"] = np.exp(out[column])
    hi_cols = [column for column in candidate_cols if "hi" in column.lower() or "q90" in column.lower() or "q975" in column.lower()]
    for column in hi_cols:
        out[f"warn_{column}_looks_like_ratio_not_log"] = out[column].between(1.05, 3.50)
        out[f"warn_{column}_exp_too_large"] = np.exp(out[column]).gt(3.0)
    return out.replace([np.inf, -np.inf], np.nan)


def compare_artifact_to_db(forecasts: pd.DataFrame, artifact_quantiles: pd.DataFrame) -> pd.DataFrame:
    if artifact_quantiles.empty or "origin_day" not in artifact_quantiles.columns:
        return pd.DataFrame()
    q = artifact_quantiles.loc[artifact_quantiles.get("policy_scope", "pre_release_origin").eq("pre_release_origin")].copy()
    if q.empty:
        return pd.DataFrame()
    for column in ["origin_day", "lo80_log", "hi80_log", "lo95_log", "hi95_log"]:
        if column in q.columns:
            q[column] = pd.to_numeric(q[column], errors="coerce")
    db = forecasts.loc[(forecasts.get("regime") == "pre_release") & (forecasts.get("target") == "opening_weekend")].copy()
    if db.empty:
        return pd.DataFrame()
    db["origin_day"] = pd.to_numeric(db["origin_day"], errors="coerce")
    grouped = (
        db.groupby("origin_day", dropna=False)
        .agg(
            db_n=("point_usd", "size"),
            db_lo80_log=("lo80_log_implied", "median"),
            db_hi80_log=("hi80_log_implied", "median"),
            db_lo95_log=("lo95_log_implied", "median"),
            db_hi95_log=("hi95_log_implied", "median"),
            db_hi95_mult=("hi95_mult", "median"),
        )
        .reset_index()
    )
    merged = grouped.merge(
        q[["origin_day", "lo80_log", "hi80_log", "lo95_log", "hi95_log", "method", "n"]],
        on="origin_day",
        how="left",
    )
    for bound in ["lo80", "hi80", "lo95", "hi95"]:
        merged[f"{bound}_log_delta_db_minus_artifact"] = merged[f"db_{bound}_log"] - merged[f"{bound}_log"]
    return merged


def summarize_calibration_panel(calibration: pd.DataFrame) -> pd.DataFrame:
    out = calibration.copy()
    actual_col = first_existing(out, ["actual_opening_weekend_gross_usd", "actual_ow_usd", "opening_weekend_gross_usd"])
    point_col = first_existing(out, ["primary_point_forecast_usd", "prod_point_forecast_usd", "point_usd", "forecast_usd"])
    if actual_col is None or point_col is None:
        raise ValueError("Calibration panel needs actual and point forecast columns.")

    actual = pd.to_numeric(out[actual_col], errors="coerce")
    point = pd.to_numeric(out[point_col], errors="coerce")
    out["actual_to_point"] = actual / point
    out["log_residual"] = np.log(out["actual_to_point"].where(out["actual_to_point"].gt(0)))
    if "source_count" not in out.columns:
        out["source_count"] = np.nan
    out["source_count_bucket"] = np.where(pd.to_numeric(out["source_count"], errors="coerce").le(1), "1", "2+")
    if "estimate_sources" in out.columns:
        out["source_family"] = out["estimate_sources"].astype(str).str.split(",").str[0].str.strip()
    scale_basis = pd.to_numeric(out[point_col], errors="coerce")
    out["scale_bucket"] = pd.qcut(scale_basis.rank(method="first"), q=3, labels=["small", "mid", "large"])

    group_cols = [column for column in ["origin_day", "source_count_bucket", "source_family", "franchise_group", "scale_bucket"] if column in out.columns]

    def quantile(values: pd.Series, q: float) -> float:
        clean = pd.to_numeric(values, errors="coerce").dropna()
        return float(clean.quantile(q)) if len(clean) else np.nan

    rows = []
    for keys, group in out.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        residual = group["actual_to_point"]
        log_resid = group["log_residual"].dropna()
        row.update(
            {
                "n": int(len(group)),
                "ratio_q05": quantile(residual, 0.05),
                "ratio_q10": quantile(residual, 0.10),
                "ratio_median": quantile(residual, 0.50),
                "ratio_q90": quantile(residual, 0.90),
                "ratio_q95": quantile(residual, 0.95),
                "ratio_q975": quantile(residual, 0.975),
                "log_mean": float(log_resid.mean()) if len(log_resid) else np.nan,
                "log_sd": float(log_resid.std(ddof=1)) if len(log_resid) >= 2 else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols + ["n"], ascending=[True] * len(group_cols) + [False])


def compare_calibration_point_to_db(forecasts: pd.DataFrame, calibration: pd.DataFrame, point_policy_path: Path | None) -> pd.DataFrame:
    if calibration.empty or "release_run_id" not in calibration.columns or "origin_day" not in calibration.columns:
        return pd.DataFrame()
    db = forecasts.loc[(forecasts.get("regime") == "pre_release") & (forecasts.get("target") == "opening_weekend")].copy()
    if db.empty:
        return pd.DataFrame()

    policy = {}
    if point_policy_path and point_policy_path.exists():
        policy = json.loads(point_policy_path.read_text(encoding="utf-8"))
    default_method = str(policy.get("default") or "primary_point_forecast_usd")
    by_origin = policy.get("by_origin_day", {}) if isinstance(policy.get("by_origin_day"), dict) else {}

    cal = calibration.copy()
    cal["release_run_id"] = pd.to_numeric(cal["release_run_id"], errors="coerce")
    cal["origin_day"] = pd.to_numeric(cal["origin_day"], errors="coerce")

    rows = []
    for item in cal.itertuples(index=False):
        origin_day = int(getattr(item, "origin_day"))
        method = str(by_origin.get(str(origin_day)) or by_origin.get(origin_day) or default_method)
        value = getattr(item, method, np.nan) if method in cal.columns else np.nan
        rows.append(
            {
                "release_run_id": getattr(item, "release_run_id"),
                "origin_day": origin_day,
                "artifact_point_method": method,
                "artifact_point_usd": value,
            }
        )
    points = pd.DataFrame(rows)
    db["release_run_id"] = pd.to_numeric(db["release_run_id"], errors="coerce")
    db["origin_day"] = pd.to_numeric(db["origin_day"], errors="coerce")
    merged = db[["release_run_id", "origin_day", "origin_key", "point_usd", "point_model"]].merge(
        points,
        on=["release_run_id", "origin_day"],
        how="left",
    )
    merged["artifact_point_usd"] = pd.to_numeric(merged["artifact_point_usd"], errors="coerce")
    merged["point_delta_usd"] = merged["point_usd"] - merged["artifact_point_usd"]
    merged["point_delta_pct"] = merged["point_delta_usd"] / merged["artifact_point_usd"]
    return merged.replace([np.inf, -np.inf], np.nan)


def print_console_summary(forecasts: pd.DataFrame, artifact_check: pd.DataFrame, suspicious: pd.DataFrame) -> None:
    print("\nInterval multiplier summary by regime/target:")
    summary = (
        forecasts.groupby(["regime", "target"], dropna=False)
        .agg(
            n=("point_usd", "size"),
            hi80_med=("hi80_mult", "median"),
            hi95_med=("hi95_mult", "median"),
            hi95_max=("hi95_mult", "max"),
            lo95_min=("lo95_mult", "min"),
            width95_med=("width95_to_point", "median"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSuspicious rows: {len(suspicious):,}")
    if not artifact_check.empty:
        warn_cols = [column for column in artifact_check.columns if column.startswith("warn_")]
        warning_count = int(artifact_check[warn_cols].fillna(False).any(axis=1).sum()) if warn_cols else 0
        print(f"Artifact unit-warning rows: {warning_count:,}")


def write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def default_artifact_dir(model_version: str | None) -> Path | None:
    if not model_version:
        return None
    path = Path("models") / "boxoffice" / model_version
    return path if path.exists() else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecasts-csv", type=Path)
    parser.add_argument("--database-url")
    parser.add_argument("--model-version")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--quantile-artifact", type=Path)
    parser.add_argument("--daily-quantile-artifact", type=Path)
    parser.add_argument("--calibration-panel", type=Path)
    parser.add_argument("--out-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact_dir = args.artifact_dir or default_artifact_dir(args.model_version)
    out_dir = args.out_dir or Path("data") / "diagnostics" / "interval_debug" / (args.model_version or "csv")
    quantile_artifact = args.quantile_artifact or (artifact_dir / "pre_release_interval_policy.json" if artifact_dir else None)
    daily_artifact = args.daily_quantile_artifact or (artifact_dir / "daily_interval_policy.json" if artifact_dir else None)
    calibration_panel = args.calibration_panel or (artifact_dir / "pre_release_panel.csv" if artifact_dir else None)
    point_policy_path = artifact_dir / "pre_release_point_policy.json" if artifact_dir else None

    forecasts = add_interval_multipliers(load_forecasts(args))
    write(forecasts, out_dir / "db_interval_rows_with_multipliers.csv")
    write(summarize_db_intervals(forecasts), out_dir / "db_interval_summary.csv")
    suspicious = suspicious_rows(forecasts)
    write(suspicious, out_dir / "suspicious_interval_rows.csv")

    scored = score_realized_forecasts(forecasts)
    if not scored.empty:
        write(scored, out_dir / "realized_interval_score_by_origin.csv")

    artifact_frames = []
    for artifact in [quantile_artifact, daily_artifact]:
        if artifact and artifact.exists():
            artifact_frames.append(inspect_quantile_artifact(artifact))
    artifact_check = pd.concat(artifact_frames, ignore_index=True) if artifact_frames else pd.DataFrame()
    if not artifact_check.empty:
        write(artifact_check, out_dir / "artifact_quantile_unit_check.csv")
        comparison = compare_artifact_to_db(forecasts, artifact_check)
        if not comparison.empty:
            write(comparison, out_dir / "artifact_vs_db_pre_release.csv")

    if calibration_panel and calibration_panel.exists():
        calibration = read_table(calibration_panel)
        write(summarize_calibration_panel(calibration), out_dir / "calibration_segment_residuals.csv")
        point_check = compare_calibration_point_to_db(forecasts, calibration, point_policy_path)
        if not point_check.empty:
            write(point_check, out_dir / "calibration_point_vs_db_point.csv")

    print_console_summary(forecasts, artifact_check, suspicious)
    print(f"\nWrote diagnostics to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
