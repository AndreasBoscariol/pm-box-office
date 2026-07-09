#!/usr/bin/env python3
"""Genre/IP residual correction screens for frozen post-opening baselines.

This runner keeps the AF2+C0 and AS0 point baselines fixed, then tests whether
coarse genre/IP buckets add rolling-origin residual or uncertainty value.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS_DIR = REPO_ROOT / "data" / "diagnostics"
PREDICTIONS_DIR = REPO_ROOT / "data" / "predictions"

AF2_PATH = DIAGNOSTICS_DIR / "af2_internal_calibration_predictions.csv"
AS0_PATH = DIAGNOSTICS_DIR / "as0_corridor_quantile_interval_predictions.csv"
WEEKEND_SHAPE_BASE_CANDIDATES = [
    REPO_ROOT / "data" / "weekend_shape_base.csv",
    DIAGNOSTICS_DIR / "weekend_shape_base.csv",
]

DIAGNOSTIC_MIN_POSITIVE = 10
MODEL_MIN_POSITIVE = 25
ROLLING_MIN_POSITIVE = 8
ROLLING_MIN_NEGATIVE = 20
SHRINK_K = 15.0
INTERVAL_EPSILON = 1.0e-6
MIN_INTERVAL_TRAIN_N = 20

BUCKET_FEATURES = [
    "FranchiseIP",
    "Sequel",
    "Horror",
    "FamilyAnimation",
    "FaithBased",
    "ConcertEvent",
    "RRatedComedyAction",
    "OriginalNonFranchise",
]
POINT_CANDIDATE_FEATURES = [
    "FranchiseIP",
    "Horror",
    "FamilyAnimation",
    "ConcertEvent",
    "FaithBased",
    "RRatedComedyAction",
]
COMPACT_FEATURE_PRIORITY = [
    "FranchiseIP",
    "Horror",
    "FamilyAnimation",
    "ConcertEvent",
    "FaithBased",
    "RRatedComedyAction",
]

TARGET_COLUMNS = [
    "origin",
    "target_name",
    "target_family",
    "release_run_id",
    "title",
    "opening_weekend_start",
    "release_year",
    "release_corridor",
    "Fri_actual",
    "Sat_actual",
    "Sun_actual",
    "OW_actual",
    "Sat_pred",
    "Sun_pred",
    "OW_pred",
    "R_actual",
    "R_pred",
    "q_actual",
    "q_pred",
    "target_residual",
    "friday_surprise",
    "latest_estimate_mid_usd",
]


@dataclass(frozen=True)
class TargetSpec:
    origin: str
    target_name: str
    target_family: str
    baseline_model: str
    prefix: str


TARGET_SPECS = [
    TargetSpec("after_friday", "u_R", "remaining_weekend", "R0_baseline", "R"),
    TargetSpec("after_friday", "u_q", "split", "Q0_baseline", "Q"),
    TargetSpec("after_saturday", "u_SunSat", "sunday_hold", "H0_baseline", "H"),
]


def safe_log_ratio(num: pd.Series | np.ndarray, den: pd.Series | np.ndarray) -> pd.Series:
    num_s = pd.to_numeric(pd.Series(num), errors="coerce")
    den_s = pd.to_numeric(pd.Series(den), errors="coerce")
    out = pd.Series(np.nan, index=num_s.index, dtype="float64")
    mask = (num_s > 0) & (den_s > 0)
    out.loc[mask] = np.log(num_s.loc[mask] / den_s.loc[mask])
    return out


def rmse(values: pd.Series | np.ndarray) -> float:
    clean = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan
    return float(np.sqrt(np.mean(np.square(clean))))


def mae(values: pd.Series | np.ndarray) -> float:
    clean = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan
    return float(clean.abs().mean())


def pct_improvement(baseline: float, candidate: float) -> float:
    if not np.isfinite(baseline) or baseline == 0 or not np.isfinite(candidate):
        return np.nan
    return float((baseline - candidate) / baseline)


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return pd.read_csv(path)


def weekend_shape_base_path() -> Path:
    for path in WEEKEND_SHAPE_BASE_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Missing weekend shape base. Tried: "
        + ", ".join(str(path) for path in WEEKEND_SHAPE_BASE_CANDIDATES)
    )


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower()


def bool_series(series: pd.Series, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(False, index=index)
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes", "y"})


def build_metadata(base: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "release_run_id",
        "title",
        "opening_weekend_start",
        "release_year",
        "release_corridor",
        "genre",
        "distributor",
        "franchise",
        "is_franchise",
        "mpa_rating",
        "is_horror",
        "is_family_animation",
        "is_doc_concert",
        "is_action",
        "franchise_group",
        "action_horror_group",
        "family_group",
        "doc_concert_group",
        "segment_ip_doc",
        "segment_ip_large_doc",
        "latest_estimate_mid_usd",
    ]
    present = [column for column in keep if column in base.columns]
    meta = base[present].copy()
    meta["release_run_id"] = pd.to_numeric(meta["release_run_id"], errors="coerce")
    meta["opening_weekend_start"] = pd.to_datetime(meta.get("opening_weekend_start"), errors="coerce")

    metadata_missing = meta[["genre", "franchise", "is_franchise"]].isna().all(axis=1)
    index = meta.index
    text = normalize_text(
        meta.get("title", pd.Series("", index=index)).astype(str)
        + " "
        + meta.get("genre", pd.Series("", index=index)).astype(str)
        + " "
        + meta.get("distributor", pd.Series("", index=index)).astype(str)
        + " "
        + meta.get("franchise", pd.Series("", index=index)).astype(str)
        + " "
        + meta.get("franchise_group", pd.Series("", index=index)).astype(str)
        + " "
        + meta.get("doc_concert_group", pd.Series("", index=index)).astype(str)
        + " "
        + meta.get("segment_ip_doc", pd.Series("", index=index)).astype(str)
    )
    genre = normalize_text(meta.get("genre", pd.Series("", index=index)))
    rating = normalize_text(meta.get("mpa_rating", pd.Series("", index=index)))
    franchise = meta.get("franchise", pd.Series(np.nan, index=index))
    franchise_text = normalize_text(franchise)

    is_franchise = bool_series(meta.get("is_franchise"), index)
    franchise_ip = (
        is_franchise
        | franchise.notna()
        | text.str.contains(r"franchise/ip|existing ip|reboot|remake|spin.?off|adaptation|comic|toy|game")
    )

    ordered = meta.sort_values(["franchise", "opening_weekend_start", "release_run_id"]).copy()
    ordered["_franchise_key"] = normalize_text(ordered.get("franchise", pd.Series("", index=ordered.index))).str.strip()
    ordered["_prior_in_franchise"] = (
        ordered["_franchise_key"].ne("")
        & (ordered.groupby("_franchise_key")["opening_weekend_start"].rank(method="first") > 1)
    )
    sequel = ordered["_prior_in_franchise"].reindex(meta.index).fillna(False)

    horror = bool_series(meta.get("is_horror"), index) | genre.str.contains("horror")
    family_animation = bool_series(meta.get("is_family_animation"), index) | (
        genre.str.contains("animation") & rating.isin({"g", "pg"})
    )
    faith_based = text.str.contains(
        r"faith|christian|biblical|bible|jesus|church|prayer|testament|the chosen|angel studios|gospel|cabrini"
    )
    concert_event = bool_series(meta.get("is_doc_concert"), index) | text.str.contains(
        r"concert|tour|live|stage|opera|event|fathom|anime"
    )
    r_rated_comedy_action = rating.eq("r") & (
        genre.str.contains("comedy") | genre.str.contains("action") | bool_series(meta.get("is_action"), index)
    )
    original_non_franchise = ~(franchise_ip | sequel | franchise_text.ne(""))

    bucket_values = {
        "FranchiseIP": franchise_ip,
        "Sequel": sequel,
        "Horror": horror,
        "FamilyAnimation": family_animation,
        "FaithBased": faith_based,
        "ConcertEvent": concert_event,
        "RRatedComedyAction": r_rated_comedy_action,
        "OriginalNonFranchise": original_non_franchise,
    }
    for feature, values in bucket_values.items():
        series = pd.Series(values, index=meta.index).astype("boolean")
        series.loc[metadata_missing] = pd.NA
        meta[feature] = series

    return meta.drop(columns=[column for column in ["_franchise_key", "_prior_in_franchise"] if column in meta])


def select_baseline_rows(df: pd.DataFrame, origin: str, calibration_model: str) -> pd.DataFrame:
    out = df.loc[df["origin"].eq(origin)].copy()
    if "calibration_model" in out.columns:
        preferred = out.loc[out["calibration_model"].eq(calibration_model)].copy()
        if not preferred.empty:
            out = preferred
    out["opening_weekend_start"] = pd.to_datetime(out["opening_weekend_start"], errors="coerce")
    out["release_run_id"] = pd.to_numeric(out["release_run_id"], errors="coerce")
    out = out.sort_values(["opening_weekend_start", "release_run_id"]).drop_duplicates("release_run_id", keep="last")
    return out


def prepare_after_friday(af2: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    af = select_baseline_rows(af2, "after_friday", "AF2+C0")
    rename = {
        "friday_gross_usd": "Fri_actual",
        "saturday_gross_usd": "Sat_actual",
        "sunday_gross_usd": "Sun_actual",
        "opening_weekend_gross_usd": "OW_actual",
        "pred_saturday_gross_usd": "Sat_pred",
        "pred_sunday_gross_usd": "Sun_pred",
        "calibrated_pred_ow": "OW_pred",
    }
    af = af.rename(columns=rename)
    if "OW_pred" not in af.columns:
        af["OW_pred"] = af["pred_opening_weekend_gross_usd"]
    raw_remaining = af["Sat_pred"] + af["Sun_pred"]
    frozen_remaining = af["OW_pred"] - af["Fri_actual"]
    scale = frozen_remaining / raw_remaining
    valid_scale = raw_remaining.gt(0) & frozen_remaining.gt(0) & scale.replace([np.inf, -np.inf], np.nan).notna()
    af.loc[valid_scale, "Sat_pred"] = af.loc[valid_scale, "Sat_pred"] * scale.loc[valid_scale]
    af.loc[valid_scale, "Sun_pred"] = af.loc[valid_scale, "Sun_pred"] * scale.loc[valid_scale]
    merged = af.merge(
        meta.drop(columns=["title", "opening_weekend_start", "release_year", "release_corridor"], errors="ignore"),
        on="release_run_id",
        how="left",
        suffixes=("", "_meta"),
    )
    if "latest_estimate_mid_usd" not in merged.columns:
        merged["latest_estimate_mid_usd"] = np.nan
    merged["R_actual"] = merged["Sat_actual"] + merged["Sun_actual"]
    merged["R_pred"] = merged["Sat_pred"] + merged["Sun_pred"]
    merged["q_actual"] = safe_log_ratio(merged["Sun_actual"], merged["Sat_actual"])
    merged["q_pred"] = safe_log_ratio(merged["Sun_pred"], merged["Sat_pred"])
    return merged


def prepare_after_saturday(as0: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    sat = select_baseline_rows(as0, "after_saturday", "AS0")
    rename = {
        "friday_gross_usd": "Fri_actual",
        "saturday_gross_usd": "Sat_actual",
        "sunday_gross_usd": "Sun_actual",
        "opening_weekend_gross_usd": "OW_actual",
        "pred_sunday_gross_usd": "Sun_pred",
        "calibrated_pred_ow": "OW_pred",
    }
    sat = sat.rename(columns=rename)
    if "OW_pred" not in sat.columns:
        sat["OW_pred"] = sat["pred_opening_weekend_gross_usd"]
    sat["Sat_pred"] = sat["Sat_actual"]
    merged = sat.merge(
        meta.drop(columns=["title", "opening_weekend_start", "release_year", "release_corridor"], errors="ignore"),
        on="release_run_id",
        how="left",
        suffixes=("", "_meta"),
    )
    if "latest_estimate_mid_usd" not in merged.columns:
        merged["latest_estimate_mid_usd"] = np.nan
    merged["R_actual"] = merged["Sat_actual"] + merged["Sun_actual"]
    merged["R_pred"] = merged["Sat_actual"] + merged["Sun_pred"]
    merged["q_actual"] = safe_log_ratio(merged["Sun_actual"], merged["Sat_actual"])
    merged["q_pred"] = safe_log_ratio(merged["Sun_pred"], merged["Sat_actual"])
    return merged


def build_targets(after_friday: pd.DataFrame, after_saturday: pd.DataFrame) -> pd.DataFrame:
    frames = []
    af_r = after_friday.copy()
    af_r["origin"] = "after_friday"
    af_r["target_name"] = "u_R"
    af_r["target_family"] = "remaining_weekend"
    af_r["target_residual"] = safe_log_ratio(af_r["R_actual"], af_r["R_pred"])
    frames.append(af_r)

    af_q = after_friday.copy()
    af_q["origin"] = "after_friday"
    af_q["target_name"] = "u_q"
    af_q["target_family"] = "split"
    af_q["target_residual"] = af_q["q_actual"] - af_q["q_pred"]
    frames.append(af_q)

    h = after_saturday.copy()
    h["origin"] = "after_saturday"
    h["target_name"] = "u_SunSat"
    h["target_family"] = "sunday_hold"
    direct = safe_log_ratio(h["Sun_actual"], h["Sun_pred"])
    simplified = h["q_actual"] - h["q_pred"]
    assert np.allclose(direct.dropna(), simplified.loc[direct.dropna().index].dropna(), atol=1e-12)
    h["target_residual"] = direct
    frames.append(h)

    cols = TARGET_COLUMNS + BUCKET_FEATURES
    out = pd.concat([frame.reindex(columns=cols) for frame in frames], ignore_index=True)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.loc[out["target_residual"].notna()].copy()
    out = out.sort_values(["origin", "target_name", "opening_weekend_start", "release_run_id"]).reset_index(drop=True)
    return out


def model_eligible_features(coverage: pd.DataFrame, origin: str, target_name: str) -> set[str]:
    rows = coverage.loc[
        coverage["origin"].eq(origin)
        & coverage["target_name"].eq(target_name)
        & coverage["model_eligible"]
    ]
    return set(rows["feature"])


def build_feature_coverage(targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (origin, target_name), group in targets.groupby(["origin", "target_name"], dropna=False):
        for feature in BUCKET_FEATURES:
            values = group[feature] if feature in group else pd.Series(pd.NA, index=group.index)
            known = values.dropna().astype(bool)
            n_positive = int(known.sum())
            n_negative = int((~known).sum())
            dates = pd.to_datetime(group.loc[values.notna(), "opening_weekend_start"], errors="coerce")
            rows.append(
                {
                    "origin": origin,
                    "target_name": target_name,
                    "feature": feature,
                    "n_total": int(len(group)),
                    "n_positive": n_positive,
                    "n_negative": n_negative,
                    "missing_rate": float(values.isna().mean()) if len(values) else np.nan,
                    "diagnostic_eligible": n_positive >= DIAGNOSTIC_MIN_POSITIVE,
                    "model_eligible": n_positive >= MODEL_MIN_POSITIVE,
                    "first_release": dates.min().date().isoformat() if dates.notna().any() else "",
                    "last_release": dates.max().date().isoformat() if dates.notna().any() else "",
                }
            )
    return pd.DataFrame(rows)


def json_counts(series: pd.Series, labels: pd.Series) -> str:
    if series.empty:
        return "{}"
    counts = labels.loc[series.astype(bool)].value_counts(dropna=True).sort_index()
    return counts.astype(int).to_json()


def build_bucket_counts(meta: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    af_ids = set(targets.loc[targets["origin"].eq("after_friday"), "release_run_id"])
    as_ids = set(targets.loc[targets["origin"].eq("after_saturday"), "release_run_id"])
    for feature in BUCKET_FEATURES:
        values = meta[feature] if feature in meta else pd.Series(pd.NA, index=meta.index)
        known = values.fillna(False).astype(bool)
        rows.append(
            {
                "feature": feature,
                "n_positive_total": int(known.sum()),
                "n_positive_after_friday": int((known & meta["release_run_id"].isin(af_ids)).sum()),
                "n_positive_after_saturday": int((known & meta["release_run_id"].isin(as_ids)).sum()),
                "n_positive_by_year": json_counts(known, meta.get("release_year", pd.Series(index=meta.index))),
                "n_positive_by_corridor": json_counts(
                    known, meta.get("release_corridor", pd.Series(index=meta.index))
                ),
            }
        )
    return pd.DataFrame(rows)


def build_residual_summary(targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (origin, target_name), group in targets.groupby(["origin", "target_name"], dropna=False):
        for bucket in BUCKET_FEATURES:
            if bucket not in group:
                continue
            bucket_group = group.loc[group[bucket].fillna(False).astype(bool), "target_residual"]
            clean = bucket_group.replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(
                {
                    "origin": origin,
                    "target_name": target_name,
                    "bucket": bucket,
                    "n": int(len(clean)),
                    "mean_residual": float(clean.mean()) if len(clean) else np.nan,
                    "median_residual": float(clean.median()) if len(clean) else np.nan,
                    "mae_residual": mae(clean),
                    "rmse_residual": rmse(clean),
                    "mean_abs_residual": mae(clean),
                }
            )
    return pd.DataFrame(rows)


def one_feature_effect(train: pd.DataFrame, feature: str, target_col: str = "target_residual") -> tuple[float, bool, int, int]:
    known = train.loc[train[feature].notna()].copy()
    if known.empty:
        return 0.0, False, 0, 0
    positive = known[feature].astype(bool)
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    supported = n_pos >= ROLLING_MIN_POSITIVE and n_neg >= ROLLING_MIN_NEGATIVE
    if not supported:
        return 0.0, False, n_pos, n_neg
    delta = known.loc[positive, target_col].mean() - known[target_col].mean()
    shrunk = n_pos / (n_pos + SHRINK_K) * delta
    return float(shrunk), True, n_pos, n_neg


def compact_features_for(eligible: set[str]) -> list[str]:
    return [feature for feature in COMPACT_FEATURE_PRIORITY if feature in eligible]


def candidate_name(prefix: str, feature: str | None, compact: bool = False) -> str:
    if compact:
        return f"{prefix}7_compact_eligible_buckets"
    mapping = {
        "FranchiseIP": "1_franchise_ip",
        "Horror": "2_horror",
        "FamilyAnimation": "3_family_animation",
        "ConcertEvent": "4_concert_event",
        "FaithBased": "5_faith_based",
        "RRatedComedyAction": "6_r_rated_comedy_action",
    }
    if feature is None:
        return f"{prefix}0_baseline"
    return f"{prefix}{mapping[feature]}"


def reconstruct_point(row: pd.Series, target_name: str, uhat: float) -> dict[str, float]:
    if target_name == "u_R":
        pred_r = row["R_pred"] * np.exp(uhat)
        sun_share = row["Sun_pred"] / row["R_pred"] if row["R_pred"] > 0 else np.nan
        pred_sun = pred_r * sun_share
        pred_sat = pred_r - pred_sun
        pred_ow = row["Fri_actual"] + pred_r
        return {"pred_R": pred_r, "pred_q": row["q_pred"], "pred_Sat": pred_sat, "pred_Sun": pred_sun, "pred_OW": pred_ow}
    if target_name == "u_q":
        q_new = row["q_pred"] + uhat
        ratio = np.exp(q_new)
        pred_sun = row["R_pred"] * ratio / (1.0 + ratio)
        pred_sat = row["R_pred"] / (1.0 + ratio)
        pred_ow = row["Fri_actual"] + pred_sat + pred_sun
        return {"pred_R": row["R_pred"], "pred_q": q_new, "pred_Sat": pred_sat, "pred_Sun": pred_sun, "pred_OW": pred_ow}
    pred_sun = row["Sun_pred"] * np.exp(uhat)
    pred_ow = row["Fri_actual"] + row["Sat_actual"] + pred_sun
    return {"pred_R": row["Sat_actual"] + pred_sun, "pred_q": np.log(pred_sun / row["Sat_actual"]), "pred_Sat": row["Sat_actual"], "pred_Sun": pred_sun, "pred_OW": pred_ow}


def build_point_predictions_for_candidate(
    group: pd.DataFrame,
    spec: TargetSpec,
    candidate_model: str,
    candidate_features: list[str],
    global_eligible: set[str],
) -> pd.DataFrame:
    rows = []
    group = group.sort_values(["opening_weekend_start", "release_run_id"]).reset_index(drop=True)
    for idx, row in group.iterrows():
        train = group.loc[group["opening_weekend_start"] < row["opening_weekend_start"]]
        uhat = 0.0
        supports: list[str] = []
        if candidate_features:
            for feature in candidate_features:
                if feature not in global_eligible:
                    continue
                if pd.isna(row[feature]) or not bool(row[feature]):
                    continue
                effect, supported, _, _ = one_feature_effect(train, feature)
                if supported:
                    uhat += effect
                    supports.append(feature)
        recon = reconstruct_point(row, spec.target_name, uhat)
        rows.append(
            {
                "origin": spec.origin,
                "target_name": spec.target_name,
                "target_family": spec.target_family,
                "candidate_model": candidate_model,
                "candidate_features": ",".join(candidate_features) if candidate_features else "",
                "release_run_id": row["release_run_id"],
                "title": row["title"],
                "opening_weekend_start": row["opening_weekend_start"],
                "release_year": row["release_year"],
                "release_corridor": row["release_corridor"],
                "train_n": int(len(train)),
                "supported_features": ",".join(supports),
                "predicted_residual": float(uhat),
                "target_residual": row["target_residual"],
                "baseline_target_error_log": row["target_residual"],
                "candidate_target_error_log": row["target_residual"] - uhat,
                "baseline_OW_error_log": np.log(row["OW_actual"] / row["OW_pred"]),
                "candidate_OW_error_log": np.log(row["OW_actual"] / recon["pred_OW"]),
                "baseline_R_error_log": np.log(row["R_actual"] / row["R_pred"]) if row["R_actual"] > 0 and row["R_pred"] > 0 else np.nan,
                "candidate_R_error_log": np.log(row["R_actual"] / recon["pred_R"]) if row["R_actual"] > 0 and recon["pred_R"] > 0 else np.nan,
                "baseline_q_error_log": row["q_actual"] - row["q_pred"],
                "candidate_q_error_log": row["q_actual"] - recon["pred_q"],
                "actual_OW": row["OW_actual"],
                "baseline_OW_pred": row["OW_pred"],
                "candidate_OW_pred": recon["pred_OW"],
                "actual_R": row["R_actual"],
                "baseline_R_pred": row["R_pred"],
                "candidate_R_pred": recon["pred_R"],
                "actual_Sat": row["Sat_actual"],
                "actual_Sun": row["Sun_actual"],
                "baseline_Sat_pred": row["Sat_pred"],
                "baseline_Sun_pred": row["Sun_pred"],
                "candidate_Sat_pred": recon["pred_Sat"],
                "candidate_Sun_pred": recon["pred_Sun"],
                "friday_surprise": row.get("friday_surprise", np.nan),
                "latest_estimate_mid_usd": row.get("latest_estimate_mid_usd", np.nan),
                **{feature: row.get(feature, pd.NA) for feature in BUCKET_FEATURES},
            }
        )
    return pd.DataFrame(rows)


def build_point_predictions(targets: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for spec in TARGET_SPECS:
        group = targets.loc[
            targets["origin"].eq(spec.origin) & targets["target_name"].eq(spec.target_name)
        ].copy()
        eligible = model_eligible_features(coverage, spec.origin, spec.target_name)
        frames.append(build_point_predictions_for_candidate(group, spec, spec.baseline_model, [], eligible))
        for feature in POINT_CANDIDATE_FEATURES:
            frames.append(
                build_point_predictions_for_candidate(
                    group, spec, candidate_name(spec.prefix, feature), [feature], eligible
                )
            )
        compact = compact_features_for(eligible)
        if compact:
            assert "OriginalNonFranchise" not in compact
            assert not ("FranchiseIP" in compact and "Sequel" in compact)
            frames.append(
                build_point_predictions_for_candidate(
                    group, spec, candidate_name(spec.prefix, None, compact=True), compact, eligible
                )
            )
    out = pd.concat(frames, ignore_index=True)
    baseline = out.loc[out["candidate_features"].eq("")]
    assert np.allclose(baseline["predicted_residual"].fillna(0), 0.0)
    assert np.allclose(baseline["candidate_target_error_log"], baseline["target_residual"], equal_nan=True)
    assert np.allclose(baseline["candidate_OW_pred"], baseline["baseline_OW_pred"], equal_nan=True)
    return out


def summarize_point_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(["origin", "target_name", "target_family", "candidate_model", "candidate_features"], dropna=False):
        origin, target_name, target_family, candidate_model, candidate_features = keys
        base = group["baseline_target_error_log"]
        cand = group["candidate_target_error_log"]
        ow_base = group["baseline_OW_error_log"]
        ow_cand = group["candidate_OW_error_log"]
        row = {
            "origin": origin,
            "target_name": target_name,
            "target_family": target_family,
            "candidate_model": candidate_model,
            "candidate_features": candidate_features,
            "n": int(len(group)),
            "mean_train_n": float(group["train_n"].mean()) if len(group) else np.nan,
            "baseline_MAE_log_same_sample": mae(base),
            "candidate_MAE_log": mae(cand),
            "improvement_MAE_log_pct": pct_improvement(mae(base), mae(cand)),
            "baseline_RMSE_log_same_sample": rmse(base),
            "candidate_RMSE_log": rmse(cand),
            "baseline_ME_log_same_sample": float(base.mean()) if len(base) else np.nan,
            "candidate_ME_log": float(cand.mean()) if len(cand) else np.nan,
            "pct_movies_improved_abs_log_error": float((cand.abs() < base.abs()).mean()) if len(group) else np.nan,
            "ow_baseline_MAE_log_same_sample": mae(ow_base),
            "ow_candidate_MAE_log": mae(ow_cand),
            "ow_improvement_MAE_log_pct": pct_improvement(mae(ow_base), mae(ow_cand)),
            "R_baseline_MAE_log_same_sample": mae(group["baseline_R_error_log"]),
            "R_candidate_MAE_log": mae(group["candidate_R_error_log"]),
            "R_improvement_MAE_log_pct": pct_improvement(mae(group["baseline_R_error_log"]), mae(group["candidate_R_error_log"])),
            "q_baseline_MAE_log_same_sample": mae(group["baseline_q_error_log"]),
            "q_candidate_MAE_log": mae(group["candidate_q_error_log"]),
            "q_improvement_MAE_log_pct": pct_improvement(mae(group["baseline_q_error_log"]), mae(group["candidate_q_error_log"])),
            "SunSat_baseline_MAE_log_same_sample": mae(base) if target_name == "u_SunSat" else np.nan,
            "SunSat_candidate_MAE_log": mae(cand) if target_name == "u_SunSat" else np.nan,
            "SunSat_improvement_MAE_log_pct": pct_improvement(mae(base), mae(cand)) if target_name == "u_SunSat" else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["origin", "target_name", "candidate_model"]).reset_index(drop=True)


def target_actual_pred(row: pd.Series, target_name: str) -> tuple[float, float]:
    if target_name == "u_R":
        return float(row["R_actual"]), float(row["R_pred"])
    if target_name == "u_q":
        return float(row["q_actual"]), float(row["q_pred"])
    return float(row["Sun_actual"]), float(row["Sun_pred"])


def rolling_quantiles(values: pd.Series) -> dict[str, float]:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < MIN_INTERVAL_TRAIN_N:
        return {"q025": np.nan, "q10": np.nan, "q90": np.nan, "q975": np.nan}
    qs = clean.quantile([0.025, 0.10, 0.90, 0.975])
    return {"q025": float(qs.loc[0.025]), "q10": float(qs.loc[0.10]), "q90": float(qs.loc[0.90]), "q975": float(qs.loc[0.975])}


def scale_effects(train: pd.DataFrame, features: list[str], target_col: str = "target_residual") -> tuple[dict[str, float], bool]:
    effects = {}
    any_supported = False
    z_train = train.copy()
    z_train["_z"] = np.log(z_train[target_col].abs() + INTERVAL_EPSILON)
    for feature in features:
        effect, supported, _, _ = one_feature_effect(z_train, feature, "_z")
        effects[feature] = effect if supported else 0.0
        any_supported = any_supported or supported
    return effects, any_supported


def row_scale(row: pd.Series, features: list[str], effects: dict[str, float]) -> float:
    total = 0.0
    for feature in features:
        if pd.notna(row.get(feature)) and bool(row.get(feature)):
            total += effects.get(feature, 0.0)
    return float(np.exp(total))


def winkler_score(actual: float, lower: float, upper: float, alpha: float) -> float:
    if not all(np.isfinite(v) for v in [actual, lower, upper]) or lower > upper:
        return np.nan
    score = upper - lower
    if actual < lower:
        score += 2.0 / alpha * (lower - actual)
    elif actual > upper:
        score += 2.0 / alpha * (actual - upper)
    return float(score)


def interval_bounds(row: pd.Series, target_name: str, quantiles: dict[str, float], scale: float) -> dict[str, float]:
    actual, pred = target_actual_pred(row, target_name)
    q025 = quantiles["q025"] * scale
    q10 = quantiles["q10"] * scale
    q90 = quantiles["q90"] * scale
    q975 = quantiles["q975"] * scale
    if target_name in {"u_R", "u_SunSat"}:
        lo80 = pred * np.exp(q10)
        hi80 = pred * np.exp(q90)
        lo95 = pred * np.exp(q025)
        hi95 = pred * np.exp(q975)
    else:
        lo80 = pred + q10
        hi80 = pred + q90
        lo95 = pred + q025
        hi95 = pred + q975
    return {
        "actual_target": actual,
        "baseline_target_pred": pred,
        "lo80": lo80,
        "hi80": hi80,
        "lo95": lo95,
        "hi95": hi95,
        "width80_to_pred": (hi80 - lo80) / pred if target_name != "u_q" and pred > 0 else hi80 - lo80,
        "width95_to_pred": (hi95 - lo95) / pred if target_name != "u_q" and pred > 0 else hi95 - lo95,
    }


def build_interval_predictions_for_candidate(
    group: pd.DataFrame,
    spec: TargetSpec,
    candidate_model: str,
    features: list[str],
    global_eligible: set[str],
) -> pd.DataFrame:
    rows = []
    features = [feature for feature in features if feature in global_eligible]
    group = group.sort_values(["opening_weekend_start", "release_run_id"]).reset_index(drop=True)
    for _, row in group.iterrows():
        train = group.loc[group["opening_weekend_start"] < row["opening_weekend_start"]]
        quantiles = rolling_quantiles(train["target_residual"])
        if any(pd.isna(list(quantiles.values()))):
            continue
        scale = 1.0
        if features:
            effects, _ = scale_effects(train, features)
            train_scales = train.apply(lambda train_row: row_scale(train_row, features, effects), axis=1)
            mean_train_scale = float(train_scales.mean()) if len(train_scales) else 1.0
            if not np.isfinite(mean_train_scale) or mean_train_scale == 0:
                mean_train_scale = 1.0
            normalized_train_scales = train_scales / mean_train_scale
            assert np.isclose(normalized_train_scales.mean(), 1.0, atol=1e-10)
            scale = row_scale(row, features, effects) / mean_train_scale
        bounds = interval_bounds(row, spec.target_name, quantiles, scale)
        rows.append(
            {
                "origin": spec.origin,
                "target_name": spec.target_name,
                "candidate_model": candidate_model,
                "candidate_features": ",".join(features),
                "release_run_id": row["release_run_id"],
                "title": row["title"],
                "opening_weekend_start": row["opening_weekend_start"],
                "release_year": row["release_year"],
                "release_corridor": row["release_corridor"],
                "train_n": int(len(train)),
                "interval_scale": scale,
                **bounds,
            }
        )
    return pd.DataFrame(rows)


def build_interval_predictions(targets: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for spec in TARGET_SPECS:
        group = targets.loc[
            targets["origin"].eq(spec.origin) & targets["target_name"].eq(spec.target_name)
        ].copy()
        eligible = model_eligible_features(coverage, spec.origin, spec.target_name)
        frames.append(build_interval_predictions_for_candidate(group, spec, f"{spec.prefix}I0_baseline", [], eligible))
        for feature in POINT_CANDIDATE_FEATURES:
            frames.append(
                build_interval_predictions_for_candidate(
                    group, spec, f"{candidate_name(spec.prefix, feature)}_scale", [feature], eligible
                )
            )
        compact = compact_features_for(eligible)
        if compact:
            frames.append(
                build_interval_predictions_for_candidate(
                    group, spec, f"{candidate_name(spec.prefix, None, compact=True)}_scale", compact, eligible
                )
            )
    return pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)


def summarize_interval_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baseline_by_target = {
        (origin, target): group
        for (origin, target), group in predictions.loc[predictions["candidate_features"].eq("")].groupby(
            ["origin", "target_name"]
        )
    }
    for keys, group in predictions.groupby(["origin", "target_name", "candidate_model", "candidate_features"], dropna=False):
        origin, target_name, candidate_model, candidate_features = keys
        group = group.copy()
        group["cover80"] = group["actual_target"].between(group["lo80"], group["hi80"])
        group["cover95"] = group["actual_target"].between(group["lo95"], group["hi95"])
        group["lowerMiss80"] = group["actual_target"] < group["lo80"]
        group["upperMiss80"] = group["actual_target"] > group["hi80"]
        group["lowerMiss95"] = group["actual_target"] < group["lo95"]
        group["upperMiss95"] = group["actual_target"] > group["hi95"]
        group["Winkler80"] = group.apply(lambda row: winkler_score(row["actual_target"], row["lo80"], row["hi80"], 0.20), axis=1)
        group["Winkler95"] = group.apply(lambda row: winkler_score(row["actual_target"], row["lo95"], row["hi95"], 0.05), axis=1)

        base = baseline_by_target.get((origin, target_name), pd.DataFrame())
        if not base.empty:
            same_ids = set(group["release_run_id"])
            base_same = base.loc[base["release_run_id"].isin(same_ids)].copy()
            base_same["baseline_Winkler80"] = base_same.apply(lambda row: winkler_score(row["actual_target"], row["lo80"], row["hi80"], 0.20), axis=1)
            base_same["baseline_Winkler95"] = base_same.apply(lambda row: winkler_score(row["actual_target"], row["lo95"], row["hi95"], 0.05), axis=1)
            baseline_w80 = float(base_same["baseline_Winkler80"].mean())
            baseline_w95 = float(base_same["baseline_Winkler95"].mean())
        else:
            baseline_w80 = np.nan
            baseline_w95 = np.nan
        w80 = float(group["Winkler80"].mean()) if len(group) else np.nan
        w95 = float(group["Winkler95"].mean()) if len(group) else np.nan
        rows.append(
            {
                "origin": origin,
                "target_name": target_name,
                "candidate_model": candidate_model,
                "candidate_features": candidate_features,
                "n": int(len(group)),
                "Coverage80": float(group["cover80"].mean()) if len(group) else np.nan,
                "Coverage95": float(group["cover95"].mean()) if len(group) else np.nan,
                "lowerMiss80": float(group["lowerMiss80"].mean()) if len(group) else np.nan,
                "upperMiss80": float(group["upperMiss80"].mean()) if len(group) else np.nan,
                "lowerMiss95": float(group["lowerMiss95"].mean()) if len(group) else np.nan,
                "upperMiss95": float(group["upperMiss95"].mean()) if len(group) else np.nan,
                "Winkler80": w80,
                "Winkler95": w95,
                "baseline_Winkler80_same_sample": baseline_w80,
                "baseline_Winkler95_same_sample": baseline_w95,
                "Winkler80_improvement_pct": pct_improvement(baseline_w80, w80),
                "Winkler95_improvement_pct": pct_improvement(baseline_w95, w95),
                "mean_80_width_to_pred": float(group["width80_to_pred"].mean()) if len(group) else np.nan,
                "mean_95_width_to_pred": float(group["width95_to_pred"].mean()) if len(group) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["origin", "target_name", "candidate_model"]).reset_index(drop=True)


def add_decile(series: pd.Series, label: str) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    if clean.notna().nunique() < 2:
        return pd.Series(pd.NA, index=series.index, dtype="object")
    try:
        return pd.qcut(clean.rank(method="first"), 10, labels=[f"{label}_{i}" for i in range(1, 11)]).astype("object")
    except ValueError:
        return pd.Series(pd.NA, index=series.index, dtype="object")


def build_stability_slices(predictions: pd.DataFrame) -> pd.DataFrame:
    point = predictions.loc[~predictions["candidate_features"].eq("")].copy()
    if point.empty:
        return pd.DataFrame()
    point["consensus_size_decile"] = add_decile(point["latest_estimate_mid_usd"], "consensus")
    point["FridaySurprise_decile"] = add_decile(point["friday_surprise"], "friday_surprise")
    rows = []
    slice_specs = [
        ("release_year", "release_year"),
        ("release_corridor", "release_corridor"),
        ("consensus_size_decile", "consensus_size_decile"),
        ("FridaySurprise_decile", "FridaySurprise_decile"),
    ]
    for feature in BUCKET_FEATURES:
        point[f"bucket_{feature}"] = np.where(point[feature].fillna(False).astype(bool), feature, f"not_{feature}")
        slice_specs.append(("bucket", f"bucket_{feature}"))
    for keys, model_group in point.groupby(["origin", "target_name", "candidate_model"], dropna=False):
        origin, target_name, candidate_model = keys
        for slice_name, column in slice_specs:
            for slice_value, group in model_group.dropna(subset=[column]).groupby(column, dropna=False):
                if len(group) == 0:
                    continue
                base = mae(group["baseline_target_error_log"])
                cand = mae(group["candidate_target_error_log"])
                rows.append(
                    {
                        "origin": origin,
                        "target_name": target_name,
                        "candidate_model": candidate_model,
                        "slice_name": slice_name,
                        "slice_value": slice_value,
                        "n": int(len(group)),
                        "baseline_MAE_log": base,
                        "candidate_MAE_log": cand,
                        "improvement_pct": pct_improvement(base, cand),
                        "candidate_ME_log": float(group["candidate_target_error_log"].mean()),
                        "pct_improved": float((group["candidate_target_error_log"].abs() < group["baseline_target_error_log"].abs()).mean()),
                        "major_slice": len(group) >= 10,
                        "severe_degradation": bool(len(group) >= 10 and np.isfinite(base) and cand > base * 1.10),
                    }
                )
    return pd.DataFrame(rows)


def build_screening_summary(
    coverage: pd.DataFrame,
    residual_summary: pd.DataFrame,
    point_comparison: pd.DataFrame,
    interval_comparison: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in point_comparison.iterrows():
        target = row["target_name"]
        target_ok = row["improvement_MAE_log_pct"] >= 0.02
        if target == "u_R":
            ow_ok = row["ow_improvement_MAE_log_pct"] >= 0.01
        elif target == "u_q":
            ow_ok = row["ow_improvement_MAE_log_pct"] >= 0.0
        else:
            ow_ok = row["ow_improvement_MAE_log_pct"] >= 0.01
        rows.append(
            {
                "screen_type": "point",
                "origin": row["origin"],
                "target_name": target,
                "candidate_model": row["candidate_model"],
                "candidate_features": row["candidate_features"],
                "n": row["n"],
                "target_improvement_pct": row["improvement_MAE_log_pct"],
                "ow_improvement_pct": row["ow_improvement_MAE_log_pct"],
                "pct_improved": row["pct_movies_improved_abs_log_error"],
                "coverage80": np.nan,
                "coverage95": np.nan,
                "winkler95_improvement_pct": np.nan,
                "promotion_candidate": bool(target_ok and ow_ok and row["pct_movies_improved_abs_log_error"] >= 0.50),
            }
        )
    for _, row in interval_comparison.iterrows():
        has_interval_feature = isinstance(row["candidate_features"], str) and bool(row["candidate_features"].strip())
        rows.append(
            {
                "screen_type": "interval",
                "origin": row["origin"],
                "target_name": row["target_name"],
                "candidate_model": row["candidate_model"],
                "candidate_features": row["candidate_features"],
                "n": row["n"],
                "target_improvement_pct": np.nan,
                "ow_improvement_pct": np.nan,
                "pct_improved": np.nan,
                "coverage80": row["Coverage80"],
                "coverage95": row["Coverage95"],
                "winkler95_improvement_pct": row["Winkler95_improvement_pct"],
                "promotion_candidate": bool(
                    has_interval_feature
                    and 0.75 <= row["Coverage80"] <= 0.85
                    and 0.92 <= row["Coverage95"] <= 0.98
                    and row["Winkler95_improvement_pct"] > 1.0e-6
                ),
            }
        )
    summary = pd.DataFrame(rows)
    if not residual_summary.empty:
        diagnostic = residual_summary.merge(
            coverage[["origin", "target_name", "feature", "diagnostic_eligible", "model_eligible"]],
            left_on=["origin", "target_name", "bucket"],
            right_on=["origin", "target_name", "feature"],
            how="left",
        )
        diagnostic["screen_type"] = "bucket_residual_diagnostic"
        diagnostic["candidate_model"] = diagnostic["bucket"]
        diagnostic["candidate_features"] = diagnostic["bucket"]
        diagnostic["target_improvement_pct"] = np.nan
        diagnostic["ow_improvement_pct"] = np.nan
        diagnostic["pct_improved"] = np.nan
        diagnostic["coverage80"] = np.nan
        diagnostic["coverage95"] = np.nan
        diagnostic["winkler95_improvement_pct"] = np.nan
        diagnostic["promotion_candidate"] = False
        diagnostic = diagnostic.rename(columns={"n": "n"})
        keep = [column for column in summary.columns if column in diagnostic.columns]
        summary = pd.concat([summary, diagnostic[keep]], ignore_index=True)
    return summary


def write_outputs(outputs: dict[str, pd.DataFrame]) -> None:
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "coverage": DIAGNOSTICS_DIR / "genre_ip_residual_feature_coverage.csv",
        "bucket_counts": DIAGNOSTICS_DIR / "genre_ip_residual_bucket_counts.csv",
        "residual_summary": DIAGNOSTICS_DIR / "genre_ip_baseline_bucket_residual_summary.csv",
        "screening_summary": DIAGNOSTICS_DIR / "genre_ip_residual_screening_summary.csv",
        "point_comparison": DIAGNOSTICS_DIR / "genre_ip_residual_point_model_comparison.csv",
        "interval_comparison": DIAGNOSTICS_DIR / "genre_ip_residual_interval_scale_comparison.csv",
        "stability_slices": DIAGNOSTICS_DIR / "genre_ip_residual_stability_slices.csv",
        "candidate_predictions": PREDICTIONS_DIR / "genre_ip_residual_candidate_predictions.csv",
    }
    for name, path in paths.items():
        outputs[name].to_csv(path, index=False)


def run_analysis(write: bool = True) -> dict[str, pd.DataFrame]:
    af2 = read_required_csv(AF2_PATH)
    as0 = read_required_csv(AS0_PATH)
    base = read_required_csv(weekend_shape_base_path())
    metadata = build_metadata(base)
    after_friday = prepare_after_friday(af2, metadata)
    after_saturday = prepare_after_saturday(as0, metadata)
    targets = build_targets(after_friday, after_saturday)

    assert set(BUCKET_FEATURES).issuperset(BUCKET_FEATURES)
    assert not any(column.startswith("genre_") for column in targets.columns)
    af_r = targets.loc[targets["target_name"].eq("u_R")]
    recomputed_r = safe_log_ratio(af_r["Sat_actual"] + af_r["Sun_actual"], af_r["Sat_pred"] + af_r["Sun_pred"])
    assert np.allclose(af_r["target_residual"], recomputed_r, equal_nan=True)
    af_q = targets.loc[targets["target_name"].eq("u_q")]
    recomputed_q = safe_log_ratio(af_q["Sun_actual"], af_q["Sat_actual"]) - safe_log_ratio(af_q["Sun_pred"], af_q["Sat_pred"])
    assert np.allclose(af_q["target_residual"], recomputed_q, equal_nan=True)
    h = targets.loc[targets["target_name"].eq("u_SunSat")]
    assert np.allclose(h["target_residual"], safe_log_ratio(h["Sun_actual"], h["Sun_pred"]), equal_nan=True)

    coverage = build_feature_coverage(targets)
    bucket_counts = build_bucket_counts(metadata, targets)
    residual_summary = build_residual_summary(targets)
    point_predictions = build_point_predictions(targets, coverage)
    point_comparison = summarize_point_predictions(point_predictions)
    interval_predictions = build_interval_predictions(targets, coverage)
    interval_comparison = summarize_interval_predictions(interval_predictions)
    stability_slices = build_stability_slices(point_predictions)
    screening_summary = build_screening_summary(coverage, residual_summary, point_comparison, interval_comparison)

    outputs = {
        "targets": targets,
        "metadata": metadata,
        "coverage": coverage,
        "bucket_counts": bucket_counts,
        "residual_summary": residual_summary,
        "screening_summary": screening_summary,
        "point_comparison": point_comparison,
        "interval_comparison": interval_comparison,
        "stability_slices": stability_slices,
        "candidate_predictions": point_predictions,
    }
    assert outputs["coverage"].shape[0] > 0
    assert outputs["point_comparison"].shape[0] > 0
    assert outputs["interval_comparison"].shape[0] > 0
    if write:
        write_outputs(outputs)
    return outputs


def main() -> int:
    outputs = run_analysis(write=True)
    print("Genre/IP residual correction outputs written:")
    print(f"  feature coverage rows: {len(outputs['coverage'])}")
    print(f"  point comparison rows: {len(outputs['point_comparison'])}")
    print(f"  interval comparison rows: {len(outputs['interval_comparison'])}")
    print(f"  candidate prediction rows: {len(outputs['candidate_predictions'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
