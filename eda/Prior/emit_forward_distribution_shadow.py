#!/usr/bin/env python3
"""Emit immutable forward-shadow distribution forecasts.

The emitter consumes a pre-release forecast panel and writes timestamped,
append-only artifacts for later comparison against realized buckets and market
prices.  It does not place trades and it does not require market prices.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eda.Prior.daily_distribution_validation import (
    OUTPUT_DIR,
    OOF_PATH,
    SOURCE_PATH,
    canonical_grids,
    prepare_oof,
    prepare_source_panel,
    probs_d1_size,
    probs_from_residuals,
    score_probs,
)
from eda.Prior.fallback_adjusted_daily_policy_validation import forecast_with_fallback
from eda.Prior.harden_daily_distribution_policy import (
    d2_quantile_average,
    d3_linear_pool,
    weighted_source_rows,
)
from eda.Prior.locked_distribution_panel_evaluation import (
    QUANTILE_GRID as LOCKED_QUANTILE_GRID,
    d0_distribution,
    d1_distribution,
    source_distribution,
)
from models.boxoffice.distribution_tail_safety import TailSafeDistribution, audited_tail_log_scale


SHADOW_ROOT = REPO_ROOT / "data" / "shadow" / "pre_release_distribution"
BASE_POLICY_PATH = OUTPUT_DIR / "frozen_pre_release_distribution_policy_v1.json"
POLICY_PATH = OUTPUT_DIR / "frozen_pre_release_distribution_policy_v2.json"
POINT_POLICY_PATH = OUTPUT_DIR / "frozen_simplified_daily_point_policy.json"
QUANTILES = np.array([0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99])


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_grid_file(path: Path | None) -> list[tuple[str, np.ndarray, pd.DataFrame]]:
    if path is None:
        return [(grid.name, grid.edges, pd.DataFrame()) for grid in canonical_grids()]
    frame = pd.read_csv(path)
    required = {"grid_id", "bucket_lower_usd", "bucket_upper_usd"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    out = []
    for grid_id, group in frame.groupby("grid_id", sort=False):
        group = group.copy()
        lows = pd.to_numeric(group["bucket_lower_usd"], errors="coerce").fillna(0).to_numpy(dtype=float)
        highs = pd.to_numeric(group["bucket_upper_usd"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        edges = [float(np.nanmin(lows))]
        for value in highs:
            edges.append(float(value) if np.isfinite(value) else np.inf)
        edges = np.asarray(edges, dtype=float)
        if not np.all(np.diff(edges[np.isfinite(edges)]) >= 0):
            raise ValueError(f"Grid {grid_id!r} has non-monotone edges")
        out.append((str(grid_id), edges, group))
    return out


def policy_by_origin(policy: dict[str, Any]) -> dict[int, str]:
    rows = policy.get("daily_distribution_policy", [])
    return {int(row["origin_day"]): str(row["distribution_candidate"]) for row in rows}


def point_for_row(row: pd.Series, panel: pd.DataFrame, point_policy: dict[str, Any]) -> tuple[float, str, str]:
    if "oof_point_forecast_usd" in row.index and pd.notna(row["oof_point_forecast_usd"]):
        return float(row["oof_point_forecast_usd"]), "oof_point_forecast_usd", str(row.get("fallback_level", "oof"))
    origin_day = int(row["origin_day"])
    method_rows = point_policy.get("daily_policy", [])
    method_by_origin = {int(item["origin_day"]): str(item["simplified_method"]) for item in method_rows}
    method = method_by_origin.get(origin_day, "primary_point_forecast_usd")
    result = forecast_with_fallback(panel.loc[[row.name]], method, origin_day)
    value = float(result.values.iloc[0])
    fallback = str(result.fallback_level.iloc[0])
    return value, method, fallback


def source_rows_for_forecast(source: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    frame = source.copy()
    if "opening_weekend_start" in frame.columns:
        frame["opening_weekend_start"] = pd.to_datetime(frame["opening_weekend_start"], errors="coerce")
    mask = pd.Series(True, index=frame.index)
    if "release_run_id" in frame.columns and "release_run_id" in row.index and pd.notna(row["release_run_id"]):
        mask &= pd.to_numeric(frame["release_run_id"], errors="coerce").eq(float(row["release_run_id"]))
    elif "movie_id" in frame.columns and "movie_id" in row.index and pd.notna(row["movie_id"]):
        mask &= pd.to_numeric(frame["movie_id"], errors="coerce").eq(float(row["movie_id"]))
    if "opening_weekend_start" in frame.columns and "opening_weekend_start" in row.index:
        mask &= frame["opening_weekend_start"].eq(pd.Timestamp(row["opening_weekend_start"]))
    if "origin_day" in frame.columns and "origin_day" in row.index:
        mask &= pd.to_numeric(frame["origin_day"], errors="coerce").eq(float(row["origin_day"]))
    out = frame.loc[mask].copy()
    estimate_col = "source_bias_adjusted_estimate_mid_usd"
    if estimate_col not in out.columns and "estimate_mid_usd" in out.columns:
        out[estimate_col] = pd.to_numeric(out["estimate_mid_usd"], errors="coerce")
    return out.loc[pd.to_numeric(out.get(estimate_col), errors="coerce").gt(0)].copy()


def quantiles_from_probs(edges: np.ndarray, probs: np.ndarray, point: float, residuals: np.ndarray) -> dict[float, float]:
    # For D0/D1-style empirical distributions, quantiles are better recovered
    # from residuals than from coarse market buckets.
    if len(residuals):
        return {float(q): float(point * np.exp(np.quantile(residuals, q))) for q in QUANTILES}
    cdf = np.concatenate([[0.0], np.cumsum(probs)])
    return {float(q): float(np.interp(q, cdf, edges)) for q in QUANTILES}


def emit_shadow(
    *,
    forecast_panel: pd.DataFrame,
    source_estimates: pd.DataFrame,
    historical_oof: pd.DataFrame,
    policy: dict[str, Any],
    point_policy: dict[str, Any],
    grids: list[tuple[str, np.ndarray, pd.DataFrame]],
    run_dir: Path,
    artifact_version: str,
    base_policy: dict[str, Any] | None = None,
) -> dict[str, int]:
    run_dir.mkdir(parents=True, exist_ok=False)
    selected_by_origin = policy_by_origin(base_policy or policy)
    point_rows = []
    bucket_rows = []
    cdf_rows = []
    source_train = source_estimates.copy()
    if "release_year" in source_train.columns:
        source_train = source_train.loc[pd.to_numeric(source_train["release_year"], errors="coerce").notna()].copy()
    global_residuals = historical_oof["signed_log_error"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    for _, row in forecast_panel.iterrows():
        origin_day = int(row["origin_day"])
        origin_train = historical_oof.loc[historical_oof["origin_day"].eq(origin_day)].copy()
        if len(origin_train) < 30:
            origin_train = historical_oof.copy()
        origin_residuals = origin_train["signed_log_error"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        point, point_method, point_fallback = point_for_row(row, forecast_panel, point_policy)
        available_sources = source_rows_for_forecast(source_estimates, row)
        selected_candidate = selected_by_origin.get(origin_day, "D0_origin_empirical")
        source_rows, source_fallback_counts = weighted_source_rows(
            available_sources,
            source_train,
            origin_day,
            origin_residuals,
            global_residuals,
            "source_role" if "role" in selected_candidate else "equal",
        )
        size_fit = None
        try:
            from eda.Prior.daily_distribution_validation import fit_size_scale
            size_fit = fit_size_scale(origin_train)
        except (KeyError, ValueError):
            pass
        locked_d0 = d0_distribution(point, origin_residuals)
        locked_d1 = d1_distribution(origin_train, point, size_fit)
        if selected_candidate == "D1_point_size":
            selected_dist = locked_d1
        elif selected_candidate.startswith("D2"):
            selected_dist = source_distribution(
                point,
                available_sources,
                source_train,
                origin_day,
                origin_residuals,
                global_residuals,
                "source_role" if "role" in selected_candidate else "equal",
                True,
                None,
            )
        else:
            selected_dist = locked_d0
        tail_scale = audited_tail_log_scale(origin_residuals)
        candidates = {
            "D0_origin_empirical": (locked_d0, False),
            "D1_point_size": (locked_d1, False),
            "production_distribution_v1_raw": (selected_dist, False),
            "production_distribution_v2_t4_tail_safe": (selected_dist, True),
        }
        for candidate, (distribution, tail_enabled) in candidates.items():
            protected = TailSafeDistribution(
                distribution.quantiles,
                LOCKED_QUANTILE_GRID,
                point,
                tail_scale,
                contamination_weight=float(policy.get("tail_contamination_weight", 0.02)) if tail_enabled else 0.0,
                reference_df=int(policy.get("tail_reference_df", 4)),
            )
            for grid_id, edges, _grid_frame in grids:
                probs = protected.bucket_probabilities(edges)
                cdf = np.asarray(protected.cdf(edges), dtype=float)
                cdf[0], cdf[-1] = 0.0, 1.0
                final_q = protected.ppf(QUANTILES) if tail_enabled else np.interp(QUANTILES, LOCKED_QUANTILE_GRID, distribution.quantiles)
                q_values = dict(zip(QUANTILES, final_q))
                base_q_values = dict(zip(QUANTILES, np.interp(QUANTILES, LOCKED_QUANTILE_GRID, distribution.quantiles)))
                fallback_level = distribution.fallback_level
                base = {
                    "emitted_at_utc": run_dir.name,
                    "artifact_version": artifact_version,
                    "movie_id": row.get("movie_id"),
                    "release_run_id": row.get("release_run_id"),
                    "title": row.get("title"),
                    "opening_weekend_start": row.get("opening_weekend_start"),
                    "forecast_origin_date": row.get("forecast_origin_date"),
                    "origin_day": origin_day,
                    "candidate": candidate,
                    "base_distribution_policy": selected_candidate,
                    "final_distribution_policy": candidate,
                    "tail_safety_enabled": tail_enabled,
                    "tail_contamination_weight": protected.contamination_weight,
                    "tail_reference_family": "student_t" if tail_enabled else None,
                    "tail_reference_df": protected.reference_df if tail_enabled else None,
                    "tail_center_usd": point,
                    "tail_log_scale": tail_scale if tail_enabled else None,
                    "tail_scale_policy": policy.get("tail_scale_policy") if tail_enabled else None,
                    "policy_version": artifact_version,
                    "training_cutoff": row.get("forecast_origin_date"),
                    "grid_id": grid_id,
                    "point_forecast_usd": point,
                    "point_method": point_method,
                    "point_fallback_level": point_fallback,
                    "distribution_fallback_level": fallback_level,
                    "eligible_sources": row.get("eligible_sources", row.get("estimate_sources")),
                    "source_fallback_counts": json.dumps(source_fallback_counts, sort_keys=True),
                }
                point_rows.append({
                    **base,
                    **{f"base_q{int(q * 1000):03d}_usd": value for q, value in base_q_values.items()},
                    **{f"final_q{int(q * 1000):03d}_usd": value for q, value in q_values.items()},
                })
                for bucket_idx, probability in enumerate(probs):
                    bucket_rows.append(
                        {
                            **base,
                            "bucket_index": bucket_idx,
                            "bucket_lower_usd": edges[bucket_idx],
                            "bucket_upper_usd": edges[bucket_idx + 1],
                            "probability": float(probability),
                        }
                    )
                for edge_idx, cdf_value in enumerate(cdf):
                    cdf_rows.append(
                        {
                            **base,
                            "edge_index": edge_idx,
                            "edge_usd": edges[edge_idx],
                            "cdf": float(cdf_value),
                        }
                    )
    pd.DataFrame(point_rows).to_csv(run_dir / "shadow_distribution_points.csv", index=False)
    pd.DataFrame(bucket_rows).to_csv(run_dir / "shadow_bucket_probabilities.csv", index=False)
    pd.DataFrame(cdf_rows).to_csv(run_dir / "shadow_cdf_edges.csv", index=False)
    requirements = pd.DataFrame(
        [
            {
                "required_table": "historical_prediction_market_contract_definitions",
                "required_columns": "market_id,movie_id,market_open_timestamp,market_close_timestamp,settlement_source,settlement_definition,is_three_day_weekend,is_four_day_holiday_weekend,bucket_id,bucket_lower_usd,bucket_upper_usd,lower_inclusive,upper_inclusive,is_open_lower_tail,is_open_upper_tail",
            },
            {
                "required_table": "historical_prediction_market_quotes",
                "required_columns": "market_id,bucket_id,timestamp,bid_yes,ask_yes,bid_no,ask_no,available_size,fees,source",
            },
        ]
    )
    requirements.to_csv(run_dir / "market_data_requirements.csv", index=False)
    return {
        "point_rows": len(point_rows),
        "bucket_probability_rows": len(bucket_rows),
        "cdf_rows": len(cdf_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-panel", type=Path, default=OOF_PATH)
    parser.add_argument("--source-estimates", type=Path, default=SOURCE_PATH)
    parser.add_argument("--historical-oof", type=Path, default=OOF_PATH)
    parser.add_argument("--market-grid", type=Path)
    parser.add_argument("--output-root", type=Path, default=SHADOW_ROOT)
    parser.add_argument("--run-id", default=utc_run_id())
    parser.add_argument("--max-rows", type=int, default=0, help="Optional smoke-test row limit; 0 means all rows.")
    args = parser.parse_args()

    panel = pd.read_csv(args.forecast_panel, low_memory=False)
    if args.max_rows and args.max_rows > 0:
        panel = panel.head(args.max_rows).copy()
    for column in ["opening_weekend_start", "forecast_origin_date"]:
        if column in panel.columns:
            panel[column] = pd.to_datetime(panel[column], errors="coerce")
    source = prepare_source_panel() if args.source_estimates == SOURCE_PATH else pd.read_csv(args.source_estimates, low_memory=False)
    historical = prepare_oof() if args.historical_oof == OOF_PATH else pd.read_csv(args.historical_oof, low_memory=False)
    policy = read_json(POLICY_PATH)
    base_policy = read_json(BASE_POLICY_PATH)
    point_policy = read_json(POINT_POLICY_PATH)
    grids = load_grid_file(args.market_grid)
    run_dir = args.output_root / args.run_id
    counts = emit_shadow(
        forecast_panel=panel,
        source_estimates=source,
        historical_oof=historical,
        policy=policy,
        point_policy=point_policy,
        grids=grids,
        run_dir=run_dir,
        artifact_version=str(policy.get("policy_name", "pre_release_distribution_policy_v1_candidate")),
        base_policy=base_policy,
    )
    manifest = {
        "run_id": args.run_id,
        "emitted_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "forward_shadow_emission",
        "distribution_policy_path": str(POLICY_PATH),
        "base_distribution_policy_path": str(BASE_POLICY_PATH),
        "point_policy_path": str(POINT_POLICY_PATH),
        "forecast_panel_path": str(args.forecast_panel),
        "source_estimates_path": str(args.source_estimates),
        "historical_oof_path": str(args.historical_oof),
        "market_grid_path": str(args.market_grid) if args.market_grid else None,
        "counts": counts,
        "immutability_note": "Do not overwrite this directory; create a new run_id for every emission.",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"Wrote immutable shadow emission to {run_dir}")
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
