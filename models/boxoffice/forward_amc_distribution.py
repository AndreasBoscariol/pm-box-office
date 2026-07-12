#!/usr/bin/env python3
"""Append-only forward scorecard for matched live AMC distribution emissions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pm_box_office.db.connection import connect_database

from .constants import EMISSION_TABLE
from .live_cdf_audit import build_panel, paired_bootstrap, score_panel, summarize


DEFAULT_OUTPUT = Path("data/diagnostics/forward_amc_distribution")


def _json(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def origin_group(origin: object) -> str:
    clock = str(origin or "")
    if clock in {"10:00", "12:00"}:
        return "early"
    if clock in {"14:00", "16:00"}:
        return "mid"
    return "late"


def fetch_pairs(conn: Any, *, model_version: str | None = None, resolved_only: bool = True) -> pd.DataFrame:
    predicates = ["primary.forecast_role = 'production'", "control.forecast_role = 'amc_no_amc_control'", "primary.target = 'opening_weekend'"]
    if resolved_only:
        predicates.append("primary.actual_usd IS NOT NULL")
    params: list[Any] = []
    if model_version:
        predicates.append("primary.model_version = %s")
        params.append(model_version)
    cursor = conn.execute(
        f"""
        SELECT primary.movie_id, primary.release_run_id, primary.opening_weekend_start,
               primary.regime, primary.origin_key, primary.forecast_origin_utc,
               primary.as_of_utc, primary.actual_usd, primary.model_version,
               primary.distribution_payload AS amc_payload,
               control.distribution_payload AS control_payload,
               primary.payload_hash AS amc_emission_hash,
               control.payload_hash AS control_emission_hash
        FROM {EMISSION_TABLE} primary
        JOIN {EMISSION_TABLE} control
          ON control.model_version = primary.model_version
         AND control.release_run_id = primary.release_run_id
         AND control.origin_key = primary.origin_key
         AND control.target = primary.target
         AND control.as_of_utc = primary.as_of_utc
        WHERE {' AND '.join(predicates)}
        ORDER BY primary.opening_weekend_start, primary.forecast_origin_utc
        """,
        tuple(params),
    )
    return pd.DataFrame(cursor.fetchall(), columns=[column[0] for column in cursor.description])


def pair_integrity(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair in pairs.to_dict("records"):
        amc, control = _json(pair["amc_payload"]), _json(pair["control_payload"])
        amc_grid = sorted(key for key in (amc.get("market_bucket_probabilities") or {}) if str(key).startswith("boundaries:"))
        control_grid = sorted(key for key in (control.get("market_bucket_probabilities") or {}) if str(key).startswith("boundaries:"))
        amc_probs = [sum(map(float, value)) for value in (amc.get("market_bucket_probabilities") or {}).values() if isinstance(value, list)]
        control_probs = [sum(map(float, value)) for value in (control.get("market_bucket_probabilities") or {}).values() if isinstance(value, list)]
        rows.append({
            "movie_id": pair["movie_id"], "release_run_id": pair["release_run_id"], "origin_key": pair["origin_key"],
            "same_information_state": amc.get("information_state") == control.get("information_state"),
            "same_policy_version": amc.get("policy_version") == control.get("policy_version"),
            "same_actual_provenance": amc.get("actual_provenance") == control.get("actual_provenance"),
            "same_listing_grid": amc_grid == control_grid,
            "amc_bucket_probabilities_sum_to_one": all(np.isclose(total, 1.0) for total in amc_probs),
            "control_bucket_probabilities_sum_to_one": all(np.isclose(total, 1.0) for total in control_probs),
            "amc_eligible": bool(amc.get("AMC_component")),
            "distinct_emission_hashes": pair["amc_emission_hash"] != pair["control_emission_hash"],
        })
    return pd.DataFrame(rows)


def score_pairs(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = []
    for pair in pairs.to_dict("records"):
        common = {"movie_id": pair["movie_id"], "actual_usd": pair["actual_usd"], "opening_weekend_start": pair["opening_weekend_start"], "origin_key": pair["origin_key"], "regime": pair["regime"]}
        records.extend([
            {**common, "candidate": "candidate", "distribution_payload": pair["amc_payload"]},
            {**common, "candidate": "benchmark", "distribution_payload": pair["control_payload"]},
        ])
    panel = build_panel(pd.DataFrame(records))
    scores = score_panel(panel)
    if not scores.empty:
        scores["day"] = scores["information_state"].map({"pre_weekend": "Live Friday", "after_friday": "Live Saturday", "after_saturday": "Live Sunday"}).fillna(scores["information_state"])
        scores["origin_group"] = scores["clock_origin"].map(origin_group)
    return panel, scores, paired_bootstrap(scores) if not scores.empty else pd.DataFrame()


def write_report(*, pairs: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    integrity = pair_integrity(pairs)
    panel, scores, bootstrap = score_pairs(pairs)
    panel.to_parquet(output_dir / "amc_matched_pair_panel.parquet", index=False)
    summary = summarize(scores)
    if not summary.empty:
        summary.to_csv(output_dir / "amc_summary_by_day_origin.csv", index=False)
    else:
        pd.DataFrame(columns=["candidate", "information_state", "clock_origin"]).to_csv(output_dir / "amc_summary_by_day_origin.csv", index=False)
    bootstrap.to_csv(output_dir / "amc_clustered_bootstrap.csv", index=False)
    integrity.to_csv(output_dir / "amc_pair_integrity_audit.csv", index=False)
    tail = scores.sort_values("bucket_log_loss", ascending=False).head(max(1, int(np.ceil(len(scores) * .01)))) if not scores.empty else scores
    tail.to_csv(output_dir / "amc_tail_loss_cases.csv", index=False)
    resolved_weekends = int(pairs["release_run_id"].nunique()) if not pairs.empty else 0
    stage = "operational validation only" if resolved_weekends < 10 else "descriptive score monitoring" if resolved_weekends < 25 else "interim pooled day-level audit" if resolved_weekends < 40 else "formal promotion or rollback audit"
    integrity_rate = float(integrity.all(axis=1).mean()) if not integrity.empty else 0.0
    (output_dir / "summary.md").write_text(
        f"# Forward AMC distribution validation\n\nResolved release weekends: {resolved_weekends}\n\nReview stage: {stage}\n\nPair-integrity pass rate: {integrity_rate:.1%}\n\nAMC status: production mechanics enabled; forward statistical validation active.\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url")
    parser.add_argument("--model-version")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--canary", action="store_true", help="Validate emitted primary/control pairs without waiting for outcomes.")
    args = parser.parse_args(argv)
    conn = connect_database(args.database_url)
    try:
        pairs = fetch_pairs(conn, model_version=args.model_version, resolved_only=not args.canary)
    finally:
        conn.close()
    if args.canary:
        integrity = pair_integrity(pairs)
        if integrity.empty:
            print("No matched AMC/control emissions found.")
            return 2
        checks = [column for column in integrity.columns if column.startswith(("same_", "amc_", "control_", "distinct_"))]
        failures = int((~integrity[checks].all(axis=1)).sum())
        print(f"Matched pairs: {len(integrity)}; integrity failures: {failures}")
        return 1 if failures else 0
    write_report(pairs=pairs, output_dir=args.output_dir)
    print(f"Wrote forward AMC distribution report to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
