"""Pre-release opening-weekend forecast inference."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import ModelArtifacts
from .constants import PRE_RELEASE_REGIME
from .intervals import log_normal_interval
from .pre_release_distribution import build_pre_release_distribution_payload
from .schema import DailyComponent, ForecastOrigin, ForecastResult, MovieOpening


def point_method_for_origin(policy: dict[str, Any], origin_day: int) -> str:
    by_origin = policy.get("by_origin_day", {})
    return str(by_origin.get(str(origin_day)) or by_origin.get(origin_day) or policy.get("default") or "primary_point_forecast_usd")


def sigma_for_origin(policy: dict[str, Any], origin_day: int) -> tuple[float, str]:
    by_origin = policy.get("sigma_log_by_origin_day", {})
    value = by_origin.get(str(origin_day), by_origin.get(origin_day))
    if value is not None:
        return float(value), "pre_release_origin_sigma"
    default = policy.get("default_sigma_log", 0.55)
    return float(default), "pre_release_default_sigma"


def select_pre_release_panel_row(panel: pd.DataFrame, movie: MovieOpening, origin: ForecastOrigin) -> pd.Series:
    if origin.regime != PRE_RELEASE_REGIME or origin.origin_day is None:
        raise ValueError("pre-release inference requires a pre_release origin")
    if panel.empty:
        raise ValueError("pre-release panel artifact is empty")
    frame = panel.copy()
    for column in ["movie_id", "release_run_id", "origin_day"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    mask = (
        frame["movie_id"].eq(movie.movie_id)
        & frame["release_run_id"].eq(movie.release_run_id)
        & frame["origin_day"].eq(origin.origin_day)
    )
    if not mask.any():
        mask = frame["release_run_id"].eq(movie.release_run_id) & frame["origin_day"].eq(origin.origin_day)
    rows = frame.loc[mask]
    if rows.empty:
        raise KeyError(f"No pre-release panel row for release_run_id={movie.release_run_id}, origin_day={origin.origin_day}")
    return rows.iloc[-1]


def _positive_float(value: Any) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(out) if np.isfinite(out) and float(out) > 0 else math.nan


def _selected_interval_from_row(row: pd.Series, point: float) -> tuple[dict[str, float], str] | None:
    columns = {
        "lo80_usd": "selected_lo_80",
        "hi80_usd": "selected_hi_80",
        "lo95_usd": "selected_lo_95",
        "hi95_usd": "selected_hi_95",
    }
    if not all(column in row.index for column in columns.values()):
        return None
    interval = {key: _positive_float(row[column]) for key, column in columns.items()}
    if not all(np.isfinite(value) for value in interval.values()):
        return None
    if not (interval["lo95_usd"] <= interval["lo80_usd"] <= point <= interval["hi80_usd"] <= interval["hi95_usd"]):
        return None
    model = row.get("selected_interval_model") if "selected_interval_model" in row.index else None
    interval_model = f"selected_interval_{model}" if pd.notna(model) and str(model) else "selected_interval_panel"
    return interval, interval_model


def _interval_origin_bucket(origin_day: int) -> str:
    if origin_day <= -8:
        return "P_-14_to_-8"
    if origin_day <= -3:
        return "P_-7_to_-3"
    if origin_day == -2:
        return "P_-2"
    if origin_day == -1:
        return "P_-1"
    return "P_0"


def _interval_source_count_bucket(value: Any) -> str:
    count = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "one_source" if np.isfinite(count) and count <= 1 else "multi_source"


def _interval_point_bucket(value: Any) -> str:
    point = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not np.isfinite(point) or point <= 0:
        return "unknown"
    if point < 1_000_000:
        return "lt_1m"
    if point < 5_000_000:
        return "1m_5m"
    if point < 15_000_000:
        return "5m_15m"
    if point < 50_000_000:
        return "15m_50m"
    return "50m_plus"


def _interval_release_scope_bucket(row: pd.Series) -> str:
    release_type = str(row.get("release_type", "") or "").lower()
    width_bucket = str(row.get("release_width_bucket", "") or "").lower()
    if "platform" in release_type or width_bucket in {"limited", "platform"}:
        return "limited_or_platform"
    if width_bucket in {"wide", "large_wide"}:
        return "wide_or_large_wide"
    return "unknown"


def _valid_prediction_interval(interval: dict[str, float]) -> bool:
    return (
        all(np.isfinite(value) and value > 0 for value in interval.values())
        and interval["lo95_usd"] <= interval["lo80_usd"] <= interval["hi80_usd"] <= interval["hi95_usd"]
    )


def _conditional_interval_features(row: pd.Series, origin_day: int, point: float) -> dict[str, str]:
    return {
        "origin_bucket": _interval_origin_bucket(origin_day),
        "source_count_bucket": _interval_source_count_bucket(row.get("source_count")),
        "point_bucket": _interval_point_bucket(point),
        "release_scope_bucket": _interval_release_scope_bucket(row),
    }


def _conditional_interval_from_policy(
    policy: dict[str, Any],
    row: pd.Series,
    origin_day: int,
    point: float,
) -> tuple[dict[str, float], str] | None:
    conditional = policy.get("conditional_log_residual_quantiles")
    if not isinstance(conditional, dict):
        return None
    scopes = conditional.get("scopes")
    if not isinstance(scopes, list):
        return None
    features = _conditional_interval_features(row, origin_day, point)
    payload = None
    for scope in scopes:
        if not isinstance(scope, dict):
            continue
        group_cols = scope.get("group_cols")
        cells = scope.get("cells")
        if not isinstance(group_cols, list) or not isinstance(cells, dict):
            continue
        key = "|".join(str(features.get(str(column), "unknown")) for column in group_cols)
        item = cells.get(key)
        if isinstance(item, dict):
            payload = item
            break
        quantiles = scope.get("global_centered_quantiles")
        if isinstance(quantiles, dict):
            payload = {
                "center_log": float(scope.get("global_center_log", 0.0)),
                "lo80_centered_log": quantiles.get("lo80"),
                "hi80_centered_log": quantiles.get("hi80"),
                "lo95_centered_log": quantiles.get("lo95"),
                "hi95_centered_log": quantiles.get("hi95"),
            }
    if not isinstance(payload, dict):
        return None
    required = ("center_log", "lo80_centered_log", "hi80_centered_log", "lo95_centered_log", "hi95_centered_log")
    if not all(key in payload for key in required):
        return None
    center = float(payload["center_log"])
    values = {
        "lo80": center + float(payload["lo80_centered_log"]),
        "hi80": center + float(payload["hi80_centered_log"]),
        "lo95": center + float(payload["lo95_centered_log"]),
        "hi95": center + float(payload["hi95_centered_log"]),
    }
    if not all(np.isfinite(value) for value in values.values()):
        return None
    interval = {
        "lo80_usd": point * math.exp(values["lo80"]),
        "hi80_usd": point * math.exp(values["hi80"]),
        "lo95_usd": point * math.exp(values["lo95"]),
        "hi95_usd": point * math.exp(values["hi95"]),
    }
    if not _valid_prediction_interval(interval):
        return None
    return interval, "pre_release_raw_conditional_centered_quantile"


def _quantile_interval_from_policy(policy: dict[str, Any], origin_day: int, point: float) -> tuple[dict[str, float], str] | None:
    by_origin = policy.get("log_residual_quantiles_by_origin_day", {})
    payload = by_origin.get(str(origin_day), by_origin.get(origin_day)) if isinstance(by_origin, dict) else None
    if not isinstance(payload, dict):
        return None
    required = ("lo80_log", "hi80_log", "lo95_log", "hi95_log")
    if not all(key in payload for key in required):
        return None
    values = {key: float(payload[key]) for key in required}
    if not all(np.isfinite(value) for value in values.values()):
        return None
    interval = {
        "lo80_usd": point * math.exp(values["lo80_log"]),
        "hi80_usd": point * math.exp(values["hi80_log"]),
        "lo95_usd": point * math.exp(values["lo95_log"]),
        "hi95_usd": point * math.exp(values["hi95_log"]),
    }
    if not _valid_prediction_interval(interval):
        return None
    method = str(payload.get("method") or "empirical_log_residual_quantile")
    interval_model = "pre_release_empirical_quantile"
    return interval, interval_model


def _daily_shape(row: pd.Series, ow_point: float, ow_interval: dict[str, float] | None = None) -> list[DailyComponent]:
    share_columns = {
        "Friday": "friday_share",
        "Saturday": "saturday_share",
        "Sunday": "sunday_share",
    }
    if all(column in row.index and pd.notna(row[column]) for column in share_columns.values()):
        shares = {day: float(row[column]) for day, column in share_columns.items()}
    else:
        shares = {"Friday": 0.42, "Saturday": 0.34, "Sunday": 0.24}
    total = sum(value for value in shares.values() if np.isfinite(value) and value > 0)
    if not np.isfinite(total) or total <= 0:
        shares = {"Friday": 0.42, "Saturday": 0.34, "Sunday": 0.24}
        total = 1.0
    components = []
    for day, share in shares.items():
        point = ow_point * share / total
        ratio = point / ow_point if ow_point > 0 else math.nan
        interval_values = (
            {
                "component_lo80_usd": ow_interval["lo80_usd"] * ratio,
                "component_hi80_usd": ow_interval["hi80_usd"] * ratio,
                "component_lo95_usd": ow_interval["lo95_usd"] * ratio,
                "component_hi95_usd": ow_interval["hi95_usd"] * ratio,
            }
            if ow_interval and np.isfinite(ratio) and ratio > 0
            else {}
        )
        components.append(
            DailyComponent(
                component_day=day,
                component_type="baseline",
                component_point_usd=point,
                component_sigma_log=0.35,
                component_model="daily_shape_model",
                component_source="daily_baseline",
                **interval_values,
            )
        )
    return components


def forecast_pre_release_opening_weekend(
    *,
    movie: MovieOpening,
    origin: ForecastOrigin,
    artifacts: ModelArtifacts,
    run_id: str,
    is_backtest: bool,
) -> list[ForecastResult]:
    row = select_pre_release_panel_row(artifacts.pre_release_panel, movie, origin)
    method = point_method_for_origin(artifacts.pre_release_point_policy, int(origin.origin_day or 0))
    if method not in row.index:
        raise KeyError(f"Point policy selected missing column {method!r}")
    point = _positive_float(row[method])
    origin_day = int(origin.origin_day or 0)
    quantile_interval = _conditional_interval_from_policy(artifacts.pre_release_interval_policy, row, origin_day, point)
    if quantile_interval is None:
        quantile_interval = _quantile_interval_from_policy(artifacts.pre_release_interval_policy, origin_day, point)
    if quantile_interval:
        interval, interval_model = quantile_interval
    else:
        required_model = artifacts.pre_release_interval_policy.get("required_interval_model")
        if required_model:
            raise RuntimeError(
                f"Missing required pre-release interval policy {required_model} "
                f"for origin_day={origin_day}, movie_id={movie.movie_id}"
            )
        selected_interval = _selected_interval_from_row(row, point)
        if selected_interval:
            interval, interval_model = selected_interval
        else:
            sigma, interval_model = sigma_for_origin(artifacts.pre_release_interval_policy, origin_day)
            interval = log_normal_interval(point, sigma)
    actual_ow = _positive_float(row.get("actual_opening_weekend_gross_usd"))
    source_count = int(row.get("source_count")) if pd.notna(row.get("source_count", pd.NA)) else None
    estimate_sources = row.get("estimate_sources") if "estimate_sources" in row.index else None

    components = _daily_shape(row, point, interval)
    distribution_payload = build_pre_release_distribution_payload(
        point_usd=point,
        lo80_usd=interval["lo80_usd"],
        hi80_usd=interval["hi80_usd"],
        distribution_policy=artifacts.pre_release_distribution_policy,
        forecast_origin=origin.forecast_origin,
        origin_key=origin.origin_key,
        model_version=artifacts.model_version,
    )
    results = [
        ForecastResult(
            movie=movie,
            origin=origin,
            target="opening_weekend",
            point_usd=point,
            lo80_usd=interval["lo80_usd"],
            hi80_usd=interval["hi80_usd"],
            lo95_usd=interval["lo95_usd"],
            hi95_usd=interval["hi95_usd"],
            point_model=f"raw_consensus_reference:{method}",
            interval_model=interval_model,
            component_source="consensus",
            model_version=artifacts.model_version,
            run_id=run_id,
            distribution_payload=distribution_payload,
            components=components,
            source_count=source_count,
            estimate_sources=str(estimate_sources) if estimate_sources is not None else None,
            actual_usd=actual_ow if np.isfinite(actual_ow) else None,
            is_backtest=is_backtest,
        )
    ]
    for component in components:
        component_with_interval = component.with_intervals()
        actual = _positive_float(row.get(f"{component.component_day.lower()}_gross_usd"))
        results.append(
            ForecastResult(
                movie=movie,
                origin=origin,
                target=component.component_day.lower(),
                point_usd=component_with_interval.component_point_usd,
                lo80_usd=component_with_interval.component_lo80_usd,
                hi80_usd=component_with_interval.component_hi80_usd,
                lo95_usd=component_with_interval.component_lo95_usd,
                hi95_usd=component_with_interval.component_hi95_usd,
                point_model="daily_shape_model",
                interval_model="daily_shape_allocated_ow_interval" if interval else "daily_shape_log_sigma",
                component_source="daily_baseline",
                model_version=artifacts.model_version,
                run_id=run_id,
                components=[component],
                source_count=source_count,
                estimate_sources=str(estimate_sources) if estimate_sources is not None else None,
                actual_usd=actual if np.isfinite(actual) else None,
                is_backtest=is_backtest,
            )
        )
    return results
