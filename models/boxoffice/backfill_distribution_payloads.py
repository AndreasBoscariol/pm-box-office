"""Backfill UI CDFs for already-persisted eligible production forecasts."""

from __future__ import annotations

import argparse
import json

from src.pm_box_office.db import connect_database

from .artifacts import load_model_artifacts
from .constants import FORECAST_TABLE
from .pre_release_distribution import build_pre_release_distribution_payload


def backfill(*, database_url: str | None, model_version: str, dry_run: bool = False) -> int:
    artifacts = load_model_artifacts(model_version)
    conn = connect_database(database_url)
    try:
        rows = conn.execute(
            f"""
            SELECT forecast_id, point_usd, lo80_usd, hi80_usd, origin_key, forecast_origin_utc, regime
            FROM {FORECAST_TABLE}
            WHERE model_version = %s
              AND target = 'opening_weekend'
              AND point_usd > 0 AND lo80_usd > 0 AND hi80_usd > lo80_usd
              AND distribution_payload IS NULL
            """,
            (model_version,),
        ).fetchall()
        for forecast_id, point, lo80, hi80, origin_key, forecast_origin, regime in rows:
            # Live rows persisted before the live-CDF migration retain their
            # model's final intervals but not its simulation draws.  Preserve
            # the actual promoted pre-release policy where available; label
            # older live rows explicitly as interval-derived rather than
            # falsely presenting them as a regenerated simulation CDF.
            policy = (
                artifacts.pre_release_distribution_policy
                if regime == "pre_release"
                else {"policy_name": "persisted_interval_cdf_backfill_v1", "policy_version": "v1"}
            )
            payload = build_pre_release_distribution_payload(
                point_usd=float(point), lo80_usd=float(lo80), hi80_usd=float(hi80),
                distribution_policy=policy,
                forecast_origin=forecast_origin.isoformat() if forecast_origin else None,
                origin_key=str(origin_key), model_version=model_version, information_state=str(regime),
            )
            if not dry_run:
                conn.execute(
                    f"UPDATE {FORECAST_TABLE} SET distribution_payload = %s::jsonb WHERE forecast_id = %s",
                    (json.dumps(payload), forecast_id),
                )
        if not dry_run:
            conn.commit()
        return len(rows)
    finally:
        conn.close()


def current_coverage(*, database_url: str | None, model_version: str) -> tuple[int, int]:
    """Return current UI releases and those still lacking a persisted CDF."""
    conn = connect_database(database_url)
    try:
        row = conn.execute(
            f"""
            SELECT COUNT(*)::integer,
                   COUNT(*) FILTER (WHERE f.distribution_payload IS NULL)::integer
            FROM analytics.current_release_forecasts current
            JOIN {FORECAST_TABLE} f ON f.forecast_id = current.forecast_id
            WHERE current.model_version = %s AND current.target = 'opening_weekend'
            """,
            (model_version,),
        ).fetchone()
        return int(row[0] or 0), int(row[1] or 0)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-current", action="store_true")
    args = parser.parse_args()
    if args.verify_current:
        total, missing = current_coverage(database_url=args.database_url, model_version=args.model_version)
        print(json.dumps({"eligible_current_releases": total, "missing_distribution_payloads": missing}))
    else:
        print(backfill(database_url=args.database_url, model_version=args.model_version, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
