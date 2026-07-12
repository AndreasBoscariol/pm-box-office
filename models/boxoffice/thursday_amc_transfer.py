"""All-day AMC same-day gross nowcast with opening-Thursday slices."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .thursday_amc_candidates import STAGE1_FEATURES, _fit_linear, _predict_linear, fit_stage2, prepare_candidate_frame


FSS_DONOR_DAYS = ("Friday", "Saturday", "Sunday")
NON_THURSDAY_DAYS = ("Monday", "Tuesday", "Wednesday", "Friday", "Saturday", "Sunday")
ALL_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
TRANSFER_MODEL = "AMC_DAILY_GROSS_POOLED_V1"
ALL_DAILY_CONTEXT = "ALL_DAILY_CONTEXT"
OPENING_EVENT_POOL = "OPENING_EVENT_POOL"
WEEKDAY_POOL = "WEEKDAY_POOL"
HYBRID_PARTIAL_POOL = "HYBRID_PARTIAL_POOL"
FSS_POOL = "FSS_POOL"
STAGE2_POOLS = (ALL_DAILY_CONTEXT, OPENING_EVENT_POOL, WEEKDAY_POOL, HYBRID_PARTIAL_POOL, FSS_POOL)
POOL_ALIASES = {
    "S2_ALL_DAILY_CONTEXT": ALL_DAILY_CONTEXT,
    "S2_ALL_CONTEXT": ALL_DAILY_CONTEXT,
    "S2_ALL_NON_THU": WEEKDAY_POOL,
    "S2_FSS": FSS_POOL,
}


def completion_panel(panel: pd.DataFrame) -> pd.DataFrame:
    work = prepare_candidate_frame(panel)
    if work.empty:
        return work
    if "day_of_week" not in work:
        work["day_of_week"] = pd.to_datetime(work["exhibition_date"]).dt.day_name()
    work = _add_release_stage_features(work)
    return work.loc[
        work["s_obs"].gt(0) & work["s_final_eod"].gt(0)
    ].copy()


def canonical_pool_name(donor_pool: str) -> str:
    return POOL_ALIASES.get(str(donor_pool), str(donor_pool))


def _add_release_stage_features(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    exhibition = pd.to_datetime(work["exhibition_date"], errors="coerce") if "exhibition_date" in work else pd.Series(pd.NaT, index=work.index)
    release = pd.to_datetime(work["release_date"], errors="coerce") if "release_date" in work else pd.Series(pd.NaT, index=work.index)
    if "day_of_week" not in work:
        work["day_of_week"] = exhibition.dt.day_name()
    work["release_age_days"] = (exhibition - release).dt.days
    age = pd.to_numeric(work["release_age_days"], errors="coerce")
    work["is_opening_eve"] = age.eq(-1).astype(float)
    work["is_opening_day"] = age.eq(0).astype(float)
    work["is_opening_week"] = age.between(-1, 6).astype(float)
    work["is_holdover"] = age.ge(7).astype(float)
    work["is_weekend"] = work["day_of_week"].isin(FSS_DONOR_DAYS).astype(float)
    work["is_weekday"] = (~work["day_of_week"].isin(FSS_DONOR_DAYS)).astype(float)
    work["is_thursday"] = work["day_of_week"].eq("Thursday").astype(float)
    work["is_friday"] = work["day_of_week"].eq("Friday").astype(float)
    work["is_opening_event"] = (
        work["is_opening_eve"].eq(1.0)
        | work["is_opening_day"].eq(1.0)
        | (work["day_of_week"].eq("Friday") & age.between(0, 1))
    ).astype(float)
    work["opening_event_x_thursday"] = work["is_opening_event"] * work["is_thursday"]
    work["opening_event_x_friday"] = work["is_opening_event"] * work["is_friday"]
    work["opening_event_x_weekend"] = work["is_opening_event"] * work["is_weekend"]
    theaters = work["actual_theaters"] if "actual_theaters" in work else pd.Series(np.nan, index=work.index)
    work["log_actual_theaters"] = np.log1p(pd.to_numeric(theaters, errors="coerce").clip(lower=0))
    if "premium_format_share" in work:
        work["premium_format_share"] = pd.to_numeric(work["premium_format_share"], errors="coerce")
    if "coverage" in work:
        work["coverage"] = pd.to_numeric(work["coverage"], errors="coerce")
    return work


def donor_daily_panel(panel: pd.DataFrame, *, donor_pool: str = "S2_FSS") -> pd.DataFrame:
    work = completion_panel(panel)
    if work.empty:
        return work
    donor_pool = canonical_pool_name(donor_pool)
    if donor_pool in {ALL_DAILY_CONTEXT, HYBRID_PARTIAL_POOL}:
        day_mask = work["day_of_week"].isin(ALL_DAYS)
        stage_mask = pd.Series(True, index=work.index)
    elif donor_pool == OPENING_EVENT_POOL:
        day_mask = work["day_of_week"].isin(ALL_DAYS)
        stage_mask = work["is_opening_event"].eq(1.0)
    elif donor_pool == WEEKDAY_POOL:
        day_mask = work["day_of_week"].isin(("Monday", "Tuesday", "Wednesday", "Thursday"))
        stage_mask = pd.Series(True, index=work.index)
    elif donor_pool == FSS_POOL:
        day_mask = work["day_of_week"].isin(FSS_DONOR_DAYS)
        stage_mask = pd.Series(True, index=work.index)
    else:
        raise ValueError(f"unknown Stage 2 donor pool: {donor_pool}")
    return work.loc[
        day_mask
        & stage_mask
        & work["actual_gross_usd"].gt(0)
    ].copy()


def _stage1_features(work: pd.DataFrame, *, extra_capacity: int = 2) -> list[str]:
    features = [feature for feature in STAGE1_FEATURES if feature in work and work[feature].notna().sum() >= 3]
    return features[: max(1, min(len(features), len(work) - extra_capacity))]


def _fit_context_stage2(train: pd.DataFrame) -> dict[str, Any] | None:
    work = _add_release_stage_features(train)
    features = [
        "log_eod", "is_opening_event", "is_opening_week", "is_holdover",
        "is_weekend", "is_thursday", "log_actual_theaters", "coverage",
        "premium_format_share",
    ]
    features = [feature for feature in features if feature in work and work[feature].notna().sum() >= 3]
    model = _fit_linear(work, features, "log_preview") if "log_eod" in features else None
    if model is None:
        return fit_stage2(work, estimated_elasticity=True)
    return {
        "alpha": model["intercept"],
        "beta": model["coefficients"].get("log_eod", 1.0),
        "elasticity": "estimated_context",
        "coefficients": model["coefficients"],
        "n": model["n"],
    }


def _fit_ridge_linear(frame: pd.DataFrame, features: list[str], target: str, *, alpha: float = 5.0) -> dict[str, Any] | None:
    clean = frame.dropna(subset=[target, *features])
    if len(clean) < max(3, len(features) + 1):
        return None
    x_raw = clean[features].to_numpy("float64")
    y = clean[target].to_numpy("float64")
    means = x_raw.mean(axis=0)
    scales = x_raw.std(axis=0)
    scales[~np.isfinite(scales) | (scales == 0)] = 1.0
    x = (x_raw - means) / scales
    design = np.column_stack([np.ones(len(clean)), x])
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    raw_coefficients = coef[1:] / scales
    intercept = coef[0] - float(np.sum(raw_coefficients * means))
    return {
        "intercept": float(intercept),
        "coefficients": dict(zip(features, raw_coefficients.tolist())),
        "n": int(len(clean)),
        "regularization": "ridge",
        "ridge_alpha": float(alpha),
    }


def _fit_hybrid_partial_pool_stage2(train: pd.DataFrame) -> dict[str, Any] | None:
    work = _add_release_stage_features(train)
    features = [
        "log_eod",
        "is_opening_event",
        "is_opening_week",
        "is_holdover",
        "is_weekend",
        "is_thursday",
        "is_friday",
        "opening_event_x_thursday",
        "opening_event_x_friday",
        "opening_event_x_weekend",
        "log_actual_theaters",
        "coverage",
        "premium_format_share",
    ]
    features = [feature for feature in features if feature in work and work[feature].notna().sum() >= 3]
    if "log_eod" not in features:
        return fit_stage2(work, estimated_elasticity=True)
    model = _fit_ridge_linear(work, features, "log_preview", alpha=5.0)
    if model is None:
        return _fit_context_stage2(work)
    return {
        "alpha": model["intercept"],
        "beta": model["coefficients"].get("log_eod", 1.0),
        "elasticity": "estimated_context",
        "coefficients": model["coefficients"],
        "n": model["n"],
        "regularization": "ridge_partial_pool",
        "ridge_alpha": model["ridge_alpha"],
        "release_stage_features": True,
    }


def _predict_stage2(stage2: dict[str, Any], row: pd.Series, *, log_eod: float) -> float:
    if stage2.get("elasticity") != "estimated_context":
        return float(stage2["alpha"]) + float(stage2["beta"]) * log_eod
    value = float(stage2["alpha"])
    enriched_row: pd.Series | None = None
    for feature, coefficient in stage2.get("coefficients", {}).items():
        if feature == "log_eod":
            feature_value = log_eod
        elif feature == "is_weekend":
            feature_value = 1.0 if row.get("day_of_week") in FSS_DONOR_DAYS else 0.0
        elif feature in {
            "is_opening_eve", "is_opening_day", "is_opening_week", "is_holdover",
            "is_weekday", "is_thursday", "is_friday", "is_opening_event",
            "opening_event_x_thursday", "opening_event_x_friday", "opening_event_x_weekend",
        }:
            if enriched_row is None:
                enriched_row = _add_release_stage_features(pd.DataFrame([row])).iloc[0]
            feature_value = enriched_row.get(feature)
        elif feature == "is_opening_day":
            release_date = pd.to_datetime(row.get("release_date"), errors="coerce")
            exhibition_date = pd.to_datetime(row.get("exhibition_date"), errors="coerce")
            feature_value = 1.0 if pd.notna(release_date) and pd.notna(exhibition_date) and (exhibition_date - release_date).days <= 0 else 0.0
        elif feature == "log_actual_theaters":
            feature_value = math.log1p(max(float(row.get("actual_theaters") or 0.0), 0.0))
        else:
            feature_value = row.get(feature)
        if not np.isfinite(float(feature_value)):
            return math.nan
        value += float(coefficient) * float(feature_value)
    return value


def fit_stage2_for_pool(work: pd.DataFrame, donor_pool: str) -> dict[str, Any] | None:
    donor_pool = canonical_pool_name(donor_pool)
    if donor_pool == HYBRID_PARTIAL_POOL:
        return _fit_hybrid_partial_pool_stage2(work)
    if donor_pool in {ALL_DAILY_CONTEXT, OPENING_EVENT_POOL, WEEKDAY_POOL}:
        return _fit_context_stage2(work)
    return fit_stage2(work, estimated_elasticity=True)


def fit_pooled_transfer(
    panel: pd.DataFrame,
    *,
    stage1_panel: pd.DataFrame | None = None,
    selected_stage2_pool: str = "S2_FSS",
    min_movies: int = 10,
) -> dict[str, Any]:
    selected_stage2_pool = canonical_pool_name(selected_stage2_pool)
    stage1_work = completion_panel(stage1_panel if stage1_panel is not None else panel)
    stage2_work = donor_daily_panel(panel, donor_pool=selected_stage2_pool)
    movie_n = int(stage2_work["movie_id"].nunique()) if not stage2_work.empty else 0
    if movie_n < min_movies:
        return disabled_transfer_policy(movie_n=movie_n, row_n=len(stage2_work), reason="insufficient_donor_history")
    features = _stage1_features(stage1_work)
    stage1 = _fit_linear(stage1_work, features, "log_eod")
    # Transfer uses the pooled estimated elasticity; donor diagnostics determine
    # whether the resulting shadow should be considered transportable.
    stage2 = fit_stage2_for_pool(stage2_work, selected_stage2_pool)
    if stage1 is None or stage2 is None:
        return disabled_transfer_policy(movie_n=movie_n, row_n=len(stage2_work), reason="donor_fit_failed")
    loo = donor_day_leave_one_out(panel, stage1_panel=stage1_work, donor_pool=selected_stage2_pool, exclude_heldout_movies=True)
    residuals = clustered_residual_pool(loo, cluster_key="movie_day_key")
    metrics = donor_day_metrics(loo)
    transportable = bool(
        not metrics.empty
        and metrics["absolute_ME_log"].max() <= 0.08
        and metrics["transfer_RMSE_log_ratio"].max() <= 1.15
    )
    return {
        "enabled": True,
        "model_fitting_enabled": True,
        "shadow_forecast_enabled": True,
        "thursday_specific_fitting_enabled": False,
        "production_ow_update_enabled": False,
        "promotion_status": "shadow_only",
        "policy_version": "amc_pooled_daily_gross_v1",
        "selected_candidate": TRANSFER_MODEL,
        "target_definition": "opening_thursday_daily_gross",
        "training_target_definition": "pooled_same_day_daily_gross",
        "shadow_application_target": "opening_thursday_daily_gross",
        "stage1": stage1,
        "stage2": stage2,
        "bias_adjustment": 0.0,
        "thursday_specific_bias_adjustment": None,
        "stage1_all_day_fitting_enabled": True,
        "stage1_thursday_evaluation_enabled": True,
        "stage1_thursday_specific_coefficients": "disabled",
        "stage2_donor_fitting_enabled": True,
        "stage2_all_day_daily_gross_labels_available": True,
        "stage2_thursday_bias_correction": "disabled_until_more_distinct_thursdays",
        "training_rows_scored": int(len(stage2_work)),
        "training_movies_scored": movie_n,
        "stage1_rows_scored": int(len(stage1_work)),
        "stage1_movies_scored": int(stage1_work["movie_id"].nunique()) if not stage1_work.empty else 0,
        "stage1_calendar_dates_scored": int(stage1_work["exhibition_date"].nunique()) if not stage1_work.empty else 0,
        "selected_stage2_pool": selected_stage2_pool,
        "donor_days": donor_days_for_pool(selected_stage2_pool),
        "stage2_pool_definition": stage2_pool_definition(selected_stage2_pool),
        "release_stage_context_enabled": selected_stage2_pool in {ALL_DAILY_CONTEXT, OPENING_EVENT_POOL, WEEKDAY_POOL, HYBRID_PARTIAL_POOL},
        "partial_pooling_enabled": selected_stage2_pool == HYBRID_PARTIAL_POOL,
        "transfer_transportability_gate_passed": transportable,
        "transfer_residuals": residuals,
        "residual_pools": {"pooled_transfer": residuals},
        "min_residual_pool_n": 5,
        "interval_method": "movie_day_clustered_transfer_uncertainty_envelope",
        "fallback_order": ["pooled_transfer", "no_amc_update"],
    }


def donor_days_for_pool(donor_pool: str) -> list[str]:
    donor_pool = canonical_pool_name(donor_pool)
    if donor_pool == FSS_POOL:
        return list(FSS_DONOR_DAYS)
    if donor_pool == WEEKDAY_POOL:
        return ["Monday", "Tuesday", "Wednesday", "Thursday"]
    return list(ALL_DAYS)


def stage2_pool_definition(donor_pool: str) -> str:
    donor_pool = canonical_pool_name(donor_pool)
    definitions = {
        ALL_DAILY_CONTEXT: "all movie-days with weekday and release-stage context features",
        OPENING_EVENT_POOL: "opening-eve/opening-day movie-days, using Friday openings as strongest donors for opening Thursdays",
        WEEKDAY_POOL: "Monday-Thursday movie-days with release-stage context features",
        HYBRID_PARTIAL_POOL: "all movie-days with ridge-shrunk weekday × release-stage interactions",
        FSS_POOL: "Friday-Sunday benchmark pool",
    }
    return definitions.get(donor_pool, donor_pool)


def disabled_transfer_policy(*, movie_n: int, row_n: int, reason: str) -> dict[str, Any]:
    return {
        "enabled": False, "model_fitting_enabled": False, "shadow_forecast_enabled": False,
        "thursday_specific_fitting_enabled": False, "production_ow_update_enabled": False,
        "promotion_status": "shadow_only", "policy_version": "amc_pooled_daily_gross_v1",
        "selected_candidate": TRANSFER_MODEL,
        "target_definition": "opening_thursday_daily_gross",
        "training_target_definition": "pooled_same_day_daily_gross",
        "shadow_application_target": "opening_thursday_daily_gross",
        "training_rows_scored": row_n,
        "training_movies_scored": movie_n, "disabled_reason": reason,
    }


def predict_transfer_row(row: pd.Series, policy: dict[str, Any]) -> dict[str, float] | None:
    prepared = _add_release_stage_features(prepare_candidate_frame(pd.DataFrame([row]))).iloc[0]
    stage1, stage2 = policy.get("stage1"), policy.get("stage2")
    if not isinstance(stage1, dict) or not isinstance(stage2, dict):
        return None
    pred_log_eod = _predict_linear(stage1, prepared)
    if not math.isfinite(pred_log_eod):
        return None
    log_gross = _predict_stage2(stage2, prepared, log_eod=pred_log_eod)
    return {"pred_log_eod": pred_log_eod, "pred_final_eod_seats": math.expm1(pred_log_eod), "pred_gross_usd": math.exp(log_gross)}


def donor_day_leave_one_out(
    panel: pd.DataFrame,
    *,
    stage1_panel: pd.DataFrame | None = None,
    donor_pool: str = "S2_FSS",
    exclude_heldout_movies: bool = True,
) -> pd.DataFrame:
    donor_pool = canonical_pool_name(donor_pool)
    work = donor_daily_panel(panel, donor_pool=donor_pool)
    stage1_work = completion_panel(stage1_panel if stage1_panel is not None else panel)
    rows: list[dict[str, Any]] = []
    held_days = tuple(day for day in donor_days_for_pool(donor_pool) if day in set(work["day_of_week"].dropna()))
    for held_day in held_days:
        test = work.loc[work["day_of_week"].eq(held_day)]
        held_movies = set(test["movie_id"].dropna().tolist())
        train = work.loc[work["day_of_week"].ne(held_day)]
        stage1_train = stage1_work.loc[stage1_work["day_of_week"].ne(held_day)]
        if exclude_heldout_movies and held_movies:
            train = train.loc[~train["movie_id"].isin(held_movies)]
            stage1_train = stage1_train.loc[~stage1_train["movie_id"].isin(held_movies)]
        within = work.loc[work["day_of_week"].eq(held_day)]
        features = _stage1_features(stage1_train)
        transfer_s1, transfer_s2 = _fit_linear(stage1_train, features, "log_eod"), fit_stage2_for_pool(train, donor_pool)
        within_s1, within_s2 = _fit_linear(within, features, "log_eod"), fit_stage2(within, estimated_elasticity=True)
        if not all([transfer_s1, transfer_s2, within_s1, within_s2]):
            continue
        for _, target in test.iterrows():
            pred_log_eod = _predict_linear(transfer_s1, target)
            within_log_eod = _predict_linear(within_s1, target)
            transfer_log = _predict_stage2(transfer_s2, target, log_eod=pred_log_eod)
            within_log = within_s2["alpha"] + within_s2["beta"] * within_log_eod
            oracle_log = _predict_stage2(transfer_s2, target, log_eod=float(target["log_eod"]))
            movie_day_key = f"{target.get('movie_id')}:{target.get('exhibition_date')}"
            rows.append({
                **target.to_dict(), "held_out_day": held_day,
                "stage2_donor_pool": donor_pool,
                "excluded_heldout_movies": exclude_heldout_movies,
                "movie_day_key": movie_day_key,
                "actual_asof_amc_seats": target["s_obs"],
                "predicted_eod_amc_seats": math.expm1(pred_log_eod),
                "actual_eod_amc_seats": target["s_final_eod"],
                "predicted_daily_gross_from_actual_eod": math.exp(oracle_log),
                "predicted_daily_gross_from_predicted_eod": math.exp(transfer_log),
                "actual_daily_gross": target["actual_gross_usd"],
                "stage1_log_error": target["log_eod"] - pred_log_eod,
                "stage2_log_error": target["log_preview"] - oracle_log,
                "total_log_error": target["log_preview"] - transfer_log,
                "within_day_log_error": target["log_preview"] - within_log,
                "transfer_stage2_beta": transfer_s2["beta"],
                "within_day_stage2_beta": within_s2["beta"],
            })
    return pd.DataFrame(rows)


def temporal_deployment_predictions(
    panel: pd.DataFrame,
    *,
    stage1_panel: pd.DataFrame | None = None,
    donor_pool: str = "S2_FSS",
    min_train_dates: int = 2,
) -> pd.DataFrame:
    donor_pool = canonical_pool_name(donor_pool)
    work = donor_daily_panel(panel, donor_pool=donor_pool).sort_values(["exhibition_date", "movie_id", "forecast_origin"])
    stage1_work = completion_panel(stage1_panel if stage1_panel is not None else panel)
    rows: list[dict[str, Any]] = []
    for date, test in work.groupby("exhibition_date", sort=True):
        train = work.loc[pd.to_datetime(work["exhibition_date"]).lt(pd.Timestamp(date))]
        s1_train = stage1_work.loc[pd.to_datetime(stage1_work["exhibition_date"]).lt(pd.Timestamp(date))]
        if train["exhibition_date"].nunique() < min_train_dates:
            continue
        features = _stage1_features(s1_train)
        stage1, stage2 = _fit_linear(s1_train, features, "log_eod"), fit_stage2_for_pool(train, donor_pool)
        if stage1 is None or stage2 is None:
            continue
        for _, target in test.iterrows():
            pred_log_eod = _predict_linear(stage1, target)
            pred_log_gross = _predict_stage2(stage2, target, log_eod=pred_log_eod)
            if not np.isfinite(pred_log_eod) or not np.isfinite(pred_log_gross):
                continue
            rows.append({
                **target.to_dict(),
                "stage2_donor_pool": donor_pool,
                "prediction_date": date,
                "train_calendar_dates": int(train["exhibition_date"].nunique()),
                "train_movie_days": int(train[["movie_id", "exhibition_date"]].drop_duplicates().shape[0]),
                "movie_day_key": f"{target.get('movie_id')}:{target.get('exhibition_date')}",
                "predicted_eod_amc_seats": math.expm1(pred_log_eod),
                "predicted_daily_gross": math.exp(pred_log_gross),
                "stage1_log_error": target["log_eod"] - pred_log_eod,
                "total_log_error": target["log_preview"] - pred_log_gross,
            })
    return pd.DataFrame(rows)


def clustered_residual_pool(predictions: pd.DataFrame, *, cluster_key: str = "movie_day_key") -> list[float]:
    if predictions.empty or cluster_key not in predictions:
        return []
    clustered = predictions.dropna(subset=["total_log_error"]).groupby(cluster_key)["total_log_error"].mean()
    return clustered.astype(float).tolist()


def donor_day_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows = []
    for day, group in predictions.groupby("held_out_day"):
        transfer = group["total_log_error"]
        within = group["within_day_log_error"]
        rmse = float(np.sqrt(np.mean(transfer ** 2)))
        within_rmse = float(np.sqrt(np.mean(within ** 2)))
        rows.append({
            "held_out_day": day, "n": len(group), "movie_n": group["movie_id"].nunique(),
            "movie_day_n": group[["movie_id", "exhibition_date"]].drop_duplicates().shape[0],
            "calendar_date_n": group["exhibition_date"].nunique(),
            "ME_log": float(transfer.mean()), "absolute_ME_log": float(abs(transfer.mean())),
            "MAE_log": float(transfer.abs().mean()), "RMSE_log": rmse,
            "within_day_RMSE_log": within_rmse,
            "transfer_RMSE_log_ratio": rmse / within_rmse if within_rmse > 0 else np.nan,
            "transfer_elasticity": float(group["transfer_stage2_beta"].median()),
            "within_day_elasticity": float(group["within_day_stage2_beta"].median()),
            "residual_scale": float(transfer.std(ddof=1)),
        })
    return pd.DataFrame(rows)


def donor_pool_comparison(panel: pd.DataFrame, *, stage1_panel: pd.DataFrame | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pool in STAGE2_POOLS:
        preds = donor_day_leave_one_out(panel, stage1_panel=stage1_panel, donor_pool=pool, exclude_heldout_movies=True)
        temporal = temporal_deployment_predictions(panel, stage1_panel=stage1_panel, donor_pool=pool)
        day_metrics = donor_day_metrics(preds)
        for _, metric in day_metrics.iterrows():
            rows.append({"validation": "day_transport_movie_excluded", "stage2_donor_pool": pool, **metric.to_dict()})
        if not temporal.empty:
            residual = temporal["total_log_error"].dropna()
            rows.append({
                "validation": "temporal_next_date",
                "stage2_donor_pool": pool,
                "held_out_day": "ALL",
                "n": int(len(residual)),
                "movie_n": int(temporal["movie_id"].nunique()),
                "movie_day_n": int(temporal[["movie_id", "exhibition_date"]].drop_duplicates().shape[0]),
                "calendar_date_n": int(temporal["exhibition_date"].nunique()),
                "ME_log": float(residual.mean()) if len(residual) else np.nan,
                "absolute_ME_log": float(abs(residual.mean())) if len(residual) else np.nan,
                "MAE_log": float(residual.abs().mean()) if len(residual) else np.nan,
                "RMSE_log": float(np.sqrt(np.mean(residual ** 2))) if len(residual) else np.nan,
            })
    return pd.DataFrame(rows)


def transfer_sample_counts(stage1_panel: pd.DataFrame, stage2_panel: pd.DataFrame, *, thursday_preview_labels: int) -> pd.DataFrame:
    rows = []
    for label, frame in [("stage1_all_day_completion", completion_panel(stage1_panel)), ("stage2_all_day_daily_gross", completion_panel(stage2_panel))]:
        rows.append({
            "sample": label,
            "distinct_calendar_dates": int(frame["exhibition_date"].nunique()) if not frame.empty else 0,
            "distinct_release_weeks": int(pd.to_datetime(frame.get("opening_weekend_start", frame.get("exhibition_date")), errors="coerce").dt.to_period("W-FRI").nunique()) if not frame.empty else 0,
            "distinct_movie_days": int(frame[["movie_id", "exhibition_date"]].drop_duplicates().shape[0]) if not frame.empty else 0,
            "distinct_movies": int(frame["movie_id"].nunique()) if not frame.empty else 0,
            "origin_rows": int(len(frame)),
        })
    thursday = completion_panel(stage1_panel)
    thursday = thursday.loc[thursday["day_of_week"].eq("Thursday")] if not thursday.empty else thursday
    rows.append({
        "sample": "stage1_thursday_completion",
        "distinct_calendar_dates": int(thursday["exhibition_date"].nunique()) if not thursday.empty else 0,
        "distinct_release_weeks": int(pd.to_datetime(thursday.get("opening_weekend_start", thursday.get("exhibition_date")), errors="coerce").dt.to_period("W-FRI").nunique()) if not thursday.empty else 0,
        "distinct_movie_days": int(thursday[["movie_id", "exhibition_date"]].drop_duplicates().shape[0]) if not thursday.empty else 0,
        "distinct_movies": int(thursday["movie_id"].nunique()) if not thursday.empty else 0,
        "origin_rows": int(len(thursday)),
    })
    rows.append({
        "sample": "opening_thursday_daily_gross_labels",
        "distinct_calendar_dates": 0,
        "distinct_release_weeks": 0,
        "distinct_movie_days": int(thursday_preview_labels),
        "distinct_movies": int(thursday_preview_labels),
        "origin_rows": int(thursday_preview_labels),
    })
    return pd.DataFrame(rows)


def donor_bias_by_origin(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    return predictions.groupby(["held_out_day", "forecast_origin"], dropna=False).agg(
        n=("total_log_error", "size"), ME_log=("total_log_error", "mean"),
        MAE_log=("total_log_error", lambda x: x.abs().mean()),
        RMSE_log=("total_log_error", lambda x: float(np.sqrt(np.mean(x ** 2)))),
    ).reset_index()


def donor_nationalization_ratio(panel: pd.DataFrame) -> pd.DataFrame:
    work = donor_daily_panel(panel, donor_pool=WEEKDAY_POOL)
    if work.empty:
        return pd.DataFrame()
    work["gross_per_final_amc_seat"] = work["actual_gross_usd"] / work["s_final_eod"]
    return work.groupby("day_of_week").agg(
        n=("gross_per_final_amc_seat", "size"), movie_n=("movie_id", "nunique"),
        median_gross_per_final_amc_seat=("gross_per_final_amc_seat", "median"),
        mean_gross_per_final_amc_seat=("gross_per_final_amc_seat", "mean"),
        stage2_beta=("log_eod", lambda _: np.nan),
    ).reset_index()
