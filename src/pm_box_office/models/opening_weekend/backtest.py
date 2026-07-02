"""Production model-selection artifacts for opening-window forecasts."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pm_box_office.models.opening_weekend.registry import DEFAULT_REGISTRY, ModelRegistry, write_registry


DEFAULT_RESULTS_DIR = Path("results/models/opening_window")


def load_metric_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def robust_winner_rows(
    metric_rows: list[dict[str, str]],
    *,
    min_holdout_n: int = 10,
    required_test_years: tuple[int, ...] = (2023, 2024, 2025, 2026),
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in metric_rows:
        if row.get("status") not in ("ok", ""):
            continue
        if row.get("mape_gross", "") == "":
            continue
        key = (
            row.get("target_type", "3_day"),
            row.get("population", ""),
            row.get("interval_method", ""),
            row.get("model", ""),
        )
        grouped[key].append(row)

    winners: list[dict[str, object]] = []
    for (target_type, population, interval_method, model), rows in grouped.items():
        years = sorted({int(row.get("test_start_year", row.get("test_year", 0)) or 0) for row in rows})
        fold_count = len([year for year in years if year in required_test_years])
        total_holdout = sum(int(float(row.get("holdout_n", 0) or 0)) for row in rows)
        low_sample_flag = total_holdout < min_holdout_n or fold_count < len(required_test_years)
        mean_mape = sum(float(row["mape_gross"]) for row in rows) / len(rows)
        worst_mape = max(float(row["mape_gross"]) for row in rows)
        raw_lifts = [
            float(row["raw_estimate_mae_lift_usd"])
            for row in rows
            if row.get("raw_estimate_mae_lift_usd", "") not in ("", None)
        ]
        actual_lifts = [
            float(row["actuals_multiplier_mae_lift_usd"])
            for row in rows
            if row.get("actuals_multiplier_mae_lift_usd", "") not in ("", None)
        ]
        beats_baseline = bool(raw_lifts or actual_lifts) and min(raw_lifts + actual_lifts) > 0.0
        robust_score = mean_mape + 0.25 * worst_mape + (1.0 if low_sample_flag else 0.0) + (1.0 if not beats_baseline else 0.0)
        winners.append(
            {
                "target_type": target_type,
                "population": population,
                "interval_method": interval_method,
                "model": model,
                "fold_count": fold_count,
                "holdout_n": total_holdout,
                "low_sample_flag": low_sample_flag,
                "beats_baseline": beats_baseline,
                "mean_mape_gross": mean_mape,
                "worst_fold_mape_gross": worst_mape,
                "robust_score": robust_score,
            }
        )
    return sorted(winners, key=lambda row: (row["target_type"], row["population"], row["robust_score"]))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "target_type",
        "population",
        "interval_method",
        "model",
        "fold_count",
        "holdout_n",
        "low_sample_flag",
        "beats_baseline",
        "mean_mape_gross",
        "worst_fold_mape_gross",
        "robust_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write deployed opening-window forecast registry/backtest artifacts.")
    parser.add_argument("--database-url", help="PostgreSQL URL reserved for DB-backed backtest persistence.")
    parser.add_argument("--metrics-csv", type=Path, help="day_by_day_metrics_by_horizon.csv from the research run.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--min-holdout-n", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir: Path = args.out_dir / dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    registry: ModelRegistry = DEFAULT_REGISTRY
    write_registry(out_dir / "model_registry.json", registry)
    metric_rows = load_metric_rows(args.metrics_csv) if args.metrics_csv else []
    winner_rows = robust_winner_rows(metric_rows, min_holdout_n=args.min_holdout_n) if metric_rows else []
    write_csv(out_dir / "robust_winner_table.csv", winner_rows)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "created_at": dt.datetime.now(dt.UTC).isoformat(),
                "registry_version": registry.version,
                "metrics_csv": str(args.metrics_csv or ""),
                "robust_winner_count": len(winner_rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote opening-window deployment artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
