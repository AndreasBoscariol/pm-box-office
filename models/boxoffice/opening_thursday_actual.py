"""Official opening-Thursday daily gross OW updater.

This module is the shared diagnostic/production implementation for the
promoted The Numbers opening-Thursday actual -> opening-weekend prior update.
It deliberately accepts only official daily actuals; AMC-imputed Thursday
values must use a separate shadow label and must not call this production path.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

BUCKET_THRESHOLDS_USD = np.array(
    [5_000_000, 10_000_000, 15_000_000, 25_000_000, 50_000_000, 75_000_000, 100_000_000, 150_000_000, 200_000_000],
    dtype="float64",
)
POLICY_NAME = "opening_thursday_actual_ratio_update_prod"
POLICY_VERSION = "opening_thursday_actual_ratio_update_v1"


@dataclass(frozen=True)
class OpeningThursdayActualUpdate:
    baseline_ow_usd: float
    opening_thursday_daily_gross_usd: float | None
    updated_ow_usd: float
    update_multiplier: float
    policy_name: str
    policy_version: str | None
    training_cutoff: str | None
    eligibility_status: str
    exclusion_reason: str | None
    production_update_applied: bool
    baseline_distribution: dict[str, Any]
    updated_distribution: dict[str, Any]
    audit: dict[str, Any]


def positive_float(value: Any) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(out) if np.isfinite(out) and float(out) > 0 else math.nan


def maybe_date(value: Any) -> date | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        return None
    return parsed.date()


def maybe_datetime(value: Any) -> datetime | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize(timezone.utc)
    return parsed.to_pydatetime()


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bucket_probabilities(point: float, residuals: np.ndarray, *, smoothing_alpha: float = 0.5) -> list[float]:
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size >= 5:
        bucket_edges = np.r_[0.0, BUCKET_THRESHOLDS_USD, np.inf]
        draws = point * np.exp(residuals)
        counts = np.histogram(draws, bins=bucket_edges)[0].astype("float64")
        probs = counts + float(smoothing_alpha)
        probs = probs / probs.sum()
        return [float(v) for v in probs]
    return []


def distribution_from_residuals(point: float, residuals: list[float] | np.ndarray, *, smoothing_alpha: float = 0.5) -> dict[str, Any]:
    arr = np.asarray(residuals or [], dtype="float64")
    arr = arr[np.isfinite(arr)]
    if not math.isfinite(point) or point <= 0 or arr.size < 5:
        return {
            "point_usd": point if math.isfinite(point) else None,
            "lo80_usd": None,
            "hi80_usd": None,
            "lo95_usd": None,
            "hi95_usd": None,
            "bucket_probabilities": [],
            "residual_pool_n": int(arr.size),
        }
    probs = _bucket_probabilities(point, arr, smoothing_alpha=smoothing_alpha)
    return {
        "point_usd": float(point),
        "lo80_usd": float(point * math.exp(np.quantile(arr, 0.10))),
        "hi80_usd": float(point * math.exp(np.quantile(arr, 0.90))),
        "lo95_usd": float(point * math.exp(np.quantile(arr, 0.025))),
        "hi95_usd": float(point * math.exp(np.quantile(arr, 0.975))),
        "bucket_thresholds_usd": [float(v) for v in BUCKET_THRESHOLDS_USD],
        "bucket_probabilities": probs,
        "residual_pool_n": int(arr.size),
    }


def fit_ratio_update_policy(panel: pd.DataFrame, *, validation_metrics: dict[str, Any] | None = None, min_train_n: int = 20) -> dict[str, Any]:
    required = {"baseline_forecast_usd", "opening_thursday_daily_gross_usd", "actual_opening_weekend_gross_usd", "opening_weekend_start"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Opening Thursday policy panel missing required columns: {sorted(missing)}")
    work = panel.copy()
    for col in ["baseline_forecast_usd", "opening_thursday_daily_gross_usd", "actual_opening_weekend_gross_usd"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.loc[
        work["baseline_forecast_usd"].gt(0)
        & work["opening_thursday_daily_gross_usd"].gt(0)
        & work["actual_opening_weekend_gross_usd"].gt(0)
    ].copy()
    if len(work) < max(3, min_train_n):
        raise ValueError("Insufficient history to freeze opening-Thursday actual policy")
    x = np.log(work["opening_thursday_daily_gross_usd"] / work["baseline_forecast_usd"])
    y = np.log(work["actual_opening_weekend_gross_usd"] / work["baseline_forecast_usd"])
    design = np.column_stack([np.ones(len(work)), x.to_numpy(dtype="float64")])
    alpha, beta = np.linalg.lstsq(design, y.to_numpy(dtype="float64"), rcond=None)[0]
    pred_log = np.log(work["baseline_forecast_usd"]) + alpha + beta * x
    base_residuals = np.log(work["actual_opening_weekend_gross_usd"] / work["baseline_forecast_usd"]).to_numpy(dtype="float64")
    ratio_residuals = (np.log(work["actual_opening_weekend_gross_usd"]) - pred_log).to_numpy(dtype="float64")
    cutoff = pd.to_datetime(work["opening_weekend_start"], errors="coerce").max()
    payload = {
        "policy_name": POLICY_NAME,
        "policy_version": POLICY_VERSION,
        "production_ow_update_enabled": True,
        "selected_update_model": "ratio_update",
        "target": "opening_thursday_daily_gross",
        "source": "the_numbers_daily_box_office",
        "training_cutoff": cutoff.date().isoformat() if pd.notna(cutoff) else None,
        "min_train_n": int(min_train_n),
        "training_n": int(len(work)),
        "training_release_run_n": int(work["release_run_id"].nunique()) if "release_run_id" in work else int(len(work)),
        "parameters": {"alpha": float(alpha), "beta": float(beta)},
        "residual_policy": {
            "method": "empirical_fold_residuals",
            "baseline_residuals_log": [float(v) for v in base_residuals if np.isfinite(v)],
            "ratio_update_residuals_log": [float(v) for v in ratio_residuals if np.isfinite(v)],
            "bucket_thresholds_usd": [float(v) for v in BUCKET_THRESHOLDS_USD],
            "bucket_smoothing_alpha": 0.5,
        },
        "eligible_release_definition": {
            "ow_start_weekday": "Friday",
            "daily_gross_date": "opening_weekend_start_minus_1_day",
            "source": "the_numbers_daily_box_office",
            "gross_positive": True,
            "requires_source_available_by_execution_time": True,
        },
        "fallback_rules": ["retain_baseline_when_missing_invalid_ambiguous_or_unavailable"],
        "validation_metrics": validation_metrics or {},
        "amc_opening_thursday_update_enabled": False,
    }
    canonical = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    payload["artifact_hash"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _row_value(row: pd.Series | dict[str, Any], *columns: str) -> Any:
    for col in columns:
        if isinstance(row, pd.Series):
            if col in row.index:
                return row.get(col)
        elif col in row:
            return row.get(col)
    return None


def _eligible(
    *,
    baseline_ow_usd: float,
    opening_thursday_daily_gross_usd: float,
    row: pd.Series | dict[str, Any] | None,
    execution_time: datetime | None,
    source: str | None,
) -> tuple[bool, str | None, dict[str, Any]]:
    audit: dict[str, Any] = {"baseline_ow_usd": baseline_ow_usd, "source": source}
    if not math.isfinite(baseline_ow_usd) or baseline_ow_usd <= 0:
        return False, "non_positive_baseline_ow", audit
    if not math.isfinite(opening_thursday_daily_gross_usd) or opening_thursday_daily_gross_usd <= 0:
        return False, "missing_or_non_positive_opening_thursday_daily_gross", audit
    if source and "number" not in str(source).lower():
        return False, "non_the_numbers_source", audit
    if row is None:
        return True, None, audit
    release_run_id = _row_value(row, "release_run_id")
    audit["release_run_id"] = release_run_id
    if release_run_id is None or pd.isna(release_run_id):
        return False, "missing_release_run_id", audit
    opening_weekend_start = maybe_date(_row_value(row, "opening_weekend_start", "friday_date"))
    thursday_date = maybe_date(_row_value(row, "opening_thursday_date", "box_office_date", "daily_gross_date"))
    audit["opening_weekend_start"] = opening_weekend_start
    audit["opening_thursday_date"] = thursday_date
    if opening_weekend_start is None:
        return False, "missing_opening_weekend_start", audit
    if opening_weekend_start.weekday() != 4:
        return False, "opening_weekend_start_not_friday", audit
    expected_thursday = opening_weekend_start - pd.Timedelta(days=1)
    if hasattr(expected_thursday, "date"):
        expected_thursday = expected_thursday.date()
    if thursday_date != expected_thursday:
        return False, "daily_gross_not_opening_thursday", audit
    source_available_at = maybe_datetime(_row_value(row, "opening_thursday_fetched_at", "fetched_at", "received_at"))
    audit["source_available_at"] = source_available_at.isoformat() if source_available_at else None
    if execution_time is not None and source_available_at is not None:
        exec_dt = execution_time if execution_time.tzinfo else execution_time.replace(tzinfo=timezone.utc)
        if source_available_at > exec_dt:
            return False, "source_record_not_available_by_execution_time", audit
    conflict_count = _row_value(row, "conflicting_actual_count", "opening_thursday_conflicting_actual_count")
    if conflict_count is not None and not pd.isna(conflict_count) and int(conflict_count) > 0:
        return False, "conflicting_opening_thursday_actual", audit
    return True, None, audit


def update_ow_from_opening_thursday_actual(
    *,
    baseline_ow_usd: float,
    opening_thursday_daily_gross_usd: float | None,
    frozen_policy: dict[str, Any] | None,
    row: pd.Series | dict[str, Any] | None = None,
    execution_time: datetime | None = None,
    source: str | None = None,
) -> OpeningThursdayActualUpdate:
    policy = frozen_policy or {}
    baseline = positive_or_finite(baseline_ow_usd)
    thursday = positive_float(opening_thursday_daily_gross_usd)
    row_for_source: pd.Series | dict[str, Any] = row if row is not None else {}
    source = source or str(_row_value(row_for_source, "opening_thursday_source", "source") or "")
    base_dist = distribution_from_residuals(
        baseline,
        policy.get("residual_policy", {}).get("baseline_residuals_log", []),
        smoothing_alpha=float(policy.get("residual_policy", {}).get("bucket_smoothing_alpha", 0.5)),
    )
    if not policy or not bool(policy.get("production_ow_update_enabled")):
        return _fallback_update(baseline, thursday, policy, base_dist, "missing_or_disabled_opening_thursday_actual_policy")
    if policy.get("selected_update_model") != "ratio_update":
        return _fallback_update(baseline, thursday, policy, base_dist, "unsupported_opening_thursday_update_model")
    ok, reason, audit = _eligible(
        baseline_ow_usd=baseline,
        opening_thursday_daily_gross_usd=thursday,
        row=row,
        execution_time=execution_time,
        source=source,
    )
    if not ok:
        return _fallback_update(baseline, thursday, policy, base_dist, reason, audit=audit)
    params = policy.get("parameters", {})
    alpha = positive_or_finite(params.get("alpha"))
    beta = positive_or_finite(params.get("beta"))
    if not math.isfinite(alpha) or not math.isfinite(beta):
        return _fallback_update(baseline, thursday, policy, base_dist, "invalid_ratio_update_coefficients", audit=audit)
    updated = baseline * math.exp(alpha + beta * math.log(thursday / baseline))
    multiplier = updated / baseline
    update_dist = distribution_from_residuals(
        updated,
        policy.get("residual_policy", {}).get("ratio_update_residuals_log", []),
        smoothing_alpha=float(policy.get("residual_policy", {}).get("bucket_smoothing_alpha", 0.5)),
    )
    audit.update(
        {
            "policy_name": policy.get("policy_name", POLICY_NAME),
            "policy_version": policy.get("policy_version"),
            "alpha": alpha,
            "beta": beta,
            "log_update": math.log(multiplier),
            "dollar_update": updated - baseline,
            "update_multiplier": multiplier,
            "artifact_hash": policy.get("artifact_hash"),
        }
    )
    return OpeningThursdayActualUpdate(
        baseline_ow_usd=baseline,
        opening_thursday_daily_gross_usd=float(thursday),
        updated_ow_usd=float(updated),
        update_multiplier=float(multiplier),
        policy_name=str(policy.get("policy_name", POLICY_NAME)),
        policy_version=policy.get("policy_version"),
        training_cutoff=policy.get("training_cutoff"),
        eligibility_status="eligible",
        exclusion_reason=None,
        production_update_applied=True,
        baseline_distribution=base_dist,
        updated_distribution=update_dist,
        audit=audit,
    )


def positive_or_finite(value: Any) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(out) if np.isfinite(out) else math.nan


def _fallback_update(
    baseline: float,
    thursday: float,
    policy: dict[str, Any],
    base_dist: dict[str, Any],
    reason: str | None,
    *,
    audit: dict[str, Any] | None = None,
) -> OpeningThursdayActualUpdate:
    payload = audit or {}
    payload["fallback_reason"] = reason
    return OpeningThursdayActualUpdate(
        baseline_ow_usd=baseline,
        opening_thursday_daily_gross_usd=float(thursday) if math.isfinite(thursday) else None,
        updated_ow_usd=baseline,
        update_multiplier=1.0,
        policy_name=str(policy.get("policy_name", POLICY_NAME)) if policy else POLICY_NAME,
        policy_version=policy.get("policy_version") if policy else None,
        training_cutoff=policy.get("training_cutoff") if policy else None,
        eligibility_status="ineligible",
        exclusion_reason=reason,
        production_update_applied=False,
        baseline_distribution=base_dist,
        updated_distribution=base_dist,
        audit=payload,
    )
