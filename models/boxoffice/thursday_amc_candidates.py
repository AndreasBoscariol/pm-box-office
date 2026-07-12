"""Leakage-safe candidate models for national Thursday preview grosses."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


CANDIDATES = (
    "TH0_asof_direct",
    "TH1_actual_eod_oracle",
    "TH2_predicted_eod_bridge",
    "TH3_pooled_day_bridge",
    "TH4_prior_assisted",
)
STAGE1_FEATURES = (
    "log_s_obs", "n_theatres_observed", "n_snapshots", "log_c_scheduled_known",
    "log_started_show_seats", "log_future_show_advance_seats",
    "fraction_scheduled_showtimes_started", "hours_relative_to_first_preview",
    "premium_format_share", "coverage", "staleness_p50_minutes",
)


def prepare_candidate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        for column in [*STAGE1_FEATURES, "log_eod", "log_preview", "log_baseline_ow"]:
            if column not in out:
                out[column] = pd.Series(dtype="float64")
        return out
    for source, dest in [
        ("s_obs", "log_s_obs"), ("c_scheduled_known", "log_c_scheduled_known"),
        ("started_show_seats", "log_started_show_seats"),
        ("future_show_advance_seats", "log_future_show_advance_seats"),
    ]:
        values = out[source] if source in out else pd.Series(0.0, index=out.index)
        out[dest] = np.log1p(pd.to_numeric(values, errors="coerce").clip(lower=0))
    out["log_eod"] = np.log1p(pd.to_numeric(out.get("s_final_eod"), errors="coerce"))
    out["log_preview"] = np.log(pd.to_numeric(out.get("actual_gross_usd"), errors="coerce"))
    out["log_baseline_ow"] = np.log(pd.to_numeric(out.get("baseline_ow_usd"), errors="coerce"))
    return out.replace([np.inf, -np.inf], np.nan)


def _fit_linear(frame: pd.DataFrame, features: list[str], target: str) -> dict[str, Any] | None:
    clean = frame.dropna(subset=[target, *features])
    if len(clean) < max(3, len(features) + 1):
        return None
    x = np.column_stack([np.ones(len(clean)), *[clean[col].to_numpy("float64") for col in features]])
    y = clean[target].to_numpy("float64")
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    return {"intercept": float(coefficients[0]), "coefficients": dict(zip(features, coefficients[1:].tolist())), "n": len(clean)}


def _predict_linear(model: dict[str, Any], row: pd.Series) -> float:
    value = float(model["intercept"])
    for feature, coefficient in model["coefficients"].items():
        feature_value = pd.to_numeric(pd.Series([row.get(feature)]), errors="coerce").iloc[0]
        if not np.isfinite(feature_value):
            return math.nan
        value += float(coefficient) * float(feature_value)
    return value


def fit_stage2(train: pd.DataFrame, *, estimated_elasticity: bool) -> dict[str, Any] | None:
    clean = train.dropna(subset=["log_eod", "log_preview"])
    if len(clean) < 3:
        return None
    if not estimated_elasticity:
        alpha = float((clean["log_preview"] - clean["log_eod"]).mean())
        return {"alpha": alpha, "beta": 1.0, "elasticity": "fixed", "n": len(clean)}
    model = _fit_linear(clean, ["log_eod"], "log_preview")
    if model is None:
        return None
    return {"alpha": model["intercept"], "beta": model["coefficients"]["log_eod"], "elasticity": "estimated", "n": model["n"]}


def select_stage2(train: pd.DataFrame) -> dict[str, Any] | None:
    """Use estimated elasticity only when leave-one-week-out loss improves."""
    fixed = fit_stage2(train, estimated_elasticity=False)
    estimated = fit_stage2(train, estimated_elasticity=True)
    if fixed is None:
        return estimated
    if estimated is None or "opening_weekend_start" not in train:
        return fixed
    losses = {"fixed": [], "estimated": []}
    for _, holdout in train.groupby(pd.to_datetime(train["opening_weekend_start"]).dt.to_period("W-FRI")):
        fit = train.drop(index=holdout.index)
        for name, elastic in [("fixed", False), ("estimated", True)]:
            model = fit_stage2(fit, estimated_elasticity=elastic)
            if model is None:
                continue
            pred = model["alpha"] + model["beta"] * holdout["log_eod"]
            losses[name].extend(((holdout["log_preview"] - pred) ** 2).dropna().tolist())
    if losses["estimated"] and losses["fixed"] and np.mean(losses["estimated"]) < np.mean(losses["fixed"]):
        return estimated
    return fixed


def forecast_candidates(
    train: pd.DataFrame,
    target: pd.Series,
    *,
    pooled_stage1_train: pd.DataFrame | None = None,
) -> dict[str, dict[str, Any]]:
    train = prepare_candidate_frame(train)
    target = prepare_candidate_frame(pd.DataFrame([target])).iloc[0]
    stage2 = select_stage2(train)
    if stage2 is None:
        return {}
    result: dict[str, dict[str, Any]] = {}

    direct = _fit_linear(train, ["log_s_obs"], "log_preview")
    if direct:
        result[CANDIDATES[0]] = {"log_prediction": _predict_linear(direct, target), "stage1": direct, "stage2": None}
    result[CANDIDATES[1]] = {
        "log_prediction": stage2["alpha"] + stage2["beta"] * float(target["log_eod"]),
        "stage1": "oracle", "stage2": stage2,
    }
    available_features = [feature for feature in STAGE1_FEATURES if feature in train and train[feature].notna().sum() >= 3]
    # Keep the small-sample model identifiable by limiting features to n-2.
    available_features = available_features[: max(1, min(len(available_features), len(train) - 2))]
    stage1 = _fit_linear(train, available_features, "log_eod")
    if stage1:
        pred_eod = _predict_linear(stage1, target)
        result[CANDIDATES[2]] = {"log_prediction": stage2["alpha"] + stage2["beta"] * pred_eod, "pred_log_eod": pred_eod, "stage1": stage1, "stage2": stage2}
    pooled = prepare_candidate_frame(pooled_stage1_train) if pooled_stage1_train is not None and not pooled_stage1_train.empty else train
    pooled_features = [feature for feature in available_features if feature in pooled]
    pooled_stage1 = _fit_linear(pooled, pooled_features, "log_eod")
    if pooled_stage1:
        pred_eod = _predict_linear(pooled_stage1, target)
        result[CANDIDATES[3]] = {"log_prediction": stage2["alpha"] + stage2["beta"] * pred_eod, "pred_log_eod": pred_eod, "stage1": pooled_stage1, "stage2": stage2}
    prior = _fit_linear(train, ["log_s_obs", "log_baseline_ow"], "log_preview")
    if prior:
        result[CANDIDATES[4]] = {"log_prediction": _predict_linear(prior, target), "stage1": prior, "stage2": None, "shadow_only": True}
    return {key: value for key, value in result.items() if np.isfinite(value.get("log_prediction", np.nan))}


def release_week_rolling_predictions(panel: pd.DataFrame, *, min_train_movies: int = 10) -> pd.DataFrame:
    work = prepare_candidate_frame(panel)
    if work.empty:
        return pd.DataFrame()
    work["release_week"] = pd.to_datetime(work["opening_weekend_start"]).dt.to_period("W-FRI").astype(str)
    rows: list[dict[str, Any]] = []
    prior_errors: dict[tuple[str, str], list[float]] = {}
    for _, test_week in work.groupby("release_week", sort=True):
        cutoff = pd.to_datetime(test_week["opening_weekend_start"]).min()
        train = work.loc[
            pd.to_datetime(work["opening_weekend_start"]).lt(cutoff)
            & pd.to_datetime(work.get("preview_published_at"), utc=True, errors="coerce").lt(pd.Timestamp(cutoff, tz="UTC"))
            & work.get("is_primary_training_target", False).fillna(False).astype(bool)
        ]
        if train["movie_id"].nunique() < min_train_movies:
            continue
        for index, target in test_week.iterrows():
            candidates = forecast_candidates(train, target)
            for candidate, payload in candidates.items():
                key = (candidate, str(target.get("forecast_origin")))
                bias = float(np.mean(prior_errors.get(key, [0.0])))
                log_pred = float(payload["log_prediction"] + bias)
                actual_log = float(target["log_preview"])
                residual = actual_log - log_pred
                history = np.asarray(prior_errors.get(key, []), dtype="float64")
                quantiles = np.quantile(history, [0.025, 0.10, 0.90, 0.975]) if history.size >= 5 else [np.nan] * 4
                rows.append({
                    **target.to_dict(), "candidate": candidate,
                    "forecast_preview_gross_usd": math.exp(log_pred), "log_residual": residual,
                    "bias_adjustment": bias, "stage1": payload.get("stage1"), "stage2": payload.get("stage2"),
                    "lo95_usd": math.exp(log_pred + quantiles[0]) if np.isfinite(quantiles[0]) else np.nan,
                    "lo80_usd": math.exp(log_pred + quantiles[1]) if np.isfinite(quantiles[1]) else np.nan,
                    "hi80_usd": math.exp(log_pred + quantiles[2]) if np.isfinite(quantiles[2]) else np.nan,
                    "hi95_usd": math.exp(log_pred + quantiles[3]) if np.isfinite(quantiles[3]) else np.nan,
                })
                prior_errors.setdefault(key, []).append(residual)
    return pd.DataFrame(rows)
