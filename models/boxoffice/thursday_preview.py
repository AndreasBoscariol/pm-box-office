"""Thursday preview opening-weekend prior updates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SCALE_TOLERANCE_PCT = 0.05
PREVIEW_COLUMNS = ("thursday_preview_gross_usd", "preview_gross_usd", "thursday_preview_actual_usd")
PREVIEW_CUTOFF_COLUMNS = (
    "thursday_preview_cutoff_date",
    "preview_cutoff_date",
    "first_preview_date",
    "preview_date",
)


@dataclass(frozen=True)
class ThursdayPreviewUpdate:
    baseline_ow_usd: float
    preview_gross_usd: float | None
    preview_updated_ow_usd: float
    preview_scale_factor: float
    preview_update_applied: bool
    fallback_reason: str | None
    ow_prior_source: str
    audit: dict[str, Any]


def positive_float(value: Any) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(out) if np.isfinite(out) and float(out) > 0 else math.nan


def finite_float(value: Any) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(out) if np.isfinite(out) else math.nan


def maybe_date(value: Any) -> date | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        return None
    return parsed.date()


def reported_preview_gross(row: pd.Series) -> tuple[float, str | None]:
    for column in PREVIEW_COLUMNS:
        if column in row.index:
            value = positive_float(row.get(column))
            if math.isfinite(value):
                return value, column
    return math.nan, None


def preview_cutoff_date(row: pd.Series) -> tuple[date | None, str | None]:
    for column in PREVIEW_CUTOFF_COLUMNS:
        if column in row.index:
            value = maybe_date(row.get(column))
            if value is not None:
                return value, column
    return None, None


def baseline_timestamp(row: pd.Series) -> tuple[date | None, str | None]:
    for column in ("baseline_forecast_origin_date", "forecast_origin_date", "pre_preview_forecast_origin_date"):
        if column in row.index:
            value = maybe_date(row.get(column))
            if value is not None:
                return value, column
    return None, None


def update_ow_prior_from_reported_preview(
    *,
    baseline_ow_usd: float,
    row: pd.Series,
    policy: dict[str, Any] | None,
) -> ThursdayPreviewUpdate:
    audit: dict[str, Any] = {
        "baseline_ow_usd": baseline_ow_usd,
        "policy_present": bool(policy),
    }
    if not policy:
        return _fallback(baseline_ow_usd, "missing_thursday_preview_policy", audit)
    if not math.isfinite(baseline_ow_usd) or baseline_ow_usd <= 0:
        return _fallback(baseline_ow_usd, "non_positive_baseline_ow", audit)

    preview, preview_col = reported_preview_gross(row)
    audit["preview_column"] = preview_col
    if not math.isfinite(preview):
        return _fallback(baseline_ow_usd, "missing_reported_preview_actual", audit)

    baseline_date, baseline_col = baseline_timestamp(row)
    cutoff_date, cutoff_col = preview_cutoff_date(row)
    audit["baseline_timestamp_column"] = baseline_col
    audit["preview_cutoff_column"] = cutoff_col
    audit["baseline_forecast_date"] = baseline_date
    audit["preview_cutoff_date"] = cutoff_date
    if baseline_date is None or cutoff_date is None:
        return _fallback(baseline_ow_usd, "missing_baseline_or_preview_cutoff_timestamp", audit, preview)
    if baseline_date >= cutoff_date:
        return _fallback(baseline_ow_usd, "baseline_not_pre_preview", audit, preview)

    alpha = finite_float(policy.get("alpha"))
    beta = finite_float(policy.get("beta"))
    if not math.isfinite(alpha) or not math.isfinite(beta):
        return _fallback(baseline_ow_usd, "invalid_thursday_preview_policy_coefficients", audit, preview)

    updated = baseline_ow_usd * math.exp(alpha + beta * math.log(preview / baseline_ow_usd))
    scale = updated / baseline_ow_usd
    audit.update(
        {
            "policy_version": policy.get("policy_version"),
            "training_cutoff": policy.get("training_cutoff"),
            "alpha": alpha,
            "beta": beta,
        }
    )
    return ThursdayPreviewUpdate(
        baseline_ow_usd=float(baseline_ow_usd),
        preview_gross_usd=float(preview),
        preview_updated_ow_usd=float(updated),
        preview_scale_factor=float(scale),
        preview_update_applied=True,
        fallback_reason=None,
        ow_prior_source="thursday_preview_actual",
        audit=audit,
    )


def _fallback(
    baseline_ow_usd: float,
    reason: str,
    audit: dict[str, Any],
    preview_gross_usd: float | None = None,
) -> ThursdayPreviewUpdate:
    baseline = float(baseline_ow_usd) if math.isfinite(baseline_ow_usd) else math.nan
    audit["fallback_reason"] = reason
    return ThursdayPreviewUpdate(
        baseline_ow_usd=baseline,
        preview_gross_usd=preview_gross_usd if preview_gross_usd is not None and math.isfinite(preview_gross_usd) else None,
        preview_updated_ow_usd=baseline,
        preview_scale_factor=1.0,
        preview_update_applied=False,
        fallback_reason=reason,
        ow_prior_source="baseline_consensus",
        audit=audit,
    )


def apply_preview_update_to_predicted_preview(
    *,
    baseline_ow_usd: float,
    predicted_preview_gross_usd: float,
    policy: dict[str, Any] | None,
) -> ThursdayPreviewUpdate:
    row = pd.Series(
        {
            "thursday_preview_gross_usd": predicted_preview_gross_usd,
            "forecast_origin_date": "1900-01-01",
            "thursday_preview_cutoff_date": "1900-01-02",
        }
    )
    update = update_ow_prior_from_reported_preview(
        baseline_ow_usd=baseline_ow_usd,
        row=row,
        policy=policy,
    )
    if update.preview_update_applied:
        return ThursdayPreviewUpdate(
            **{
                **update.__dict__,
                "ow_prior_source": "thursday_amc_preview_nowcast",
            }
        )
    return update
