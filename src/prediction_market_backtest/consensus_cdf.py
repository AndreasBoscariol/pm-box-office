"""Market-aligned evaluation for frozen-point consensus CDF challengers.

This is deliberately a research-only layer.  It consumes the locked rolling
distribution panel, applies the *same* 98/2 Student-t(4) tail mixture to every
candidate, and evaluates probabilities on grids fixed at a movie's listing
origin.  It never writes a production policy or changes a point forecast.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from models.boxoffice.market_buckets import SyntheticMarketGrid, generate_fixed_grid_ensemble, get_allowed_widths


TAIL_WEIGHT = 0.02
TAIL_DF = 4
QUANTILE_GRID = np.r_[0.001, np.linspace(0.005, 0.995, 199), 0.999]
DEFAULT_LISTING_ORIGINS = (-10, -14, -7, -4)
TARGET_ORIGINS = frozenset((-4, -3, -2, -1))
QCOLS = [f"q_{q:.3f}" for q in QUANTILE_GRID]


@dataclass(frozen=True)
class BaseCdf:
    """Monotone base CDF, centred exactly on the frozen point."""

    quantiles: np.ndarray
    point: float

    @classmethod
    def from_quantiles(cls, quantiles: Iterable[float], point: float) -> "BaseCdf":
        q = np.maximum.accumulate(np.asarray(tuple(quantiles), dtype=float))
        if len(q) != len(QUANTILE_GRID) or not np.all(np.isfinite(q)) or np.any(q <= 0):
            raise ValueError("expected finite positive locked quantiles")
        median = float(np.interp(0.5, QUANTILE_GRID, q))
        if not np.isfinite(point) or point <= 0 or median <= 0:
            raise ValueError("point and median must be positive")
        return cls(np.maximum.accumulate(q * (point / median)), float(point))

    def cdf(self, values: float | np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(values, dtype=float), self.quantiles, QUANTILE_GRID, left=0.0, right=1.0)

    def ppf(self, probabilities: float | np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(probabilities, dtype=float), QUANTILE_GRID, self.quantiles)

    def scaled(self, multiplier: float) -> "BaseCdf":
        multiplier = float(np.clip(multiplier, 0.85, 1.15))
        return BaseCdf(np.maximum.accumulate(self.point + multiplier * (self.quantiles - self.point)), self.point)


@dataclass(frozen=True)
class TailLayeredCdf:
    """``.98 F_base + .02 G_t4`` with a shared frozen-point median."""

    base_cdf: Callable[[float | np.ndarray], np.ndarray]
    base_ppf: Callable[[float | np.ndarray], np.ndarray]
    point: float
    tail_log_scale: float

    def cdf(self, values: float | np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        safe = np.maximum(values, np.finfo(float).tiny)
        tail = student_t.cdf(np.log(safe / self.point) / self.tail_log_scale, TAIL_DF)
        result = (1.0 - TAIL_WEIGHT) * self.base_cdf(values) + TAIL_WEIGHT * tail
        return np.clip(result, 0.0, 1.0)

    def ppf(self, probabilities: float | np.ndarray) -> np.ndarray:
        p = np.asarray(probabilities, dtype=float)
        # The requested summaries are interior quantiles.  A log-spaced support
        # around both component distributions makes their inversion stable
        # while preserving the exact CDF used for market bucket scoring.
        support = np.unique(np.r_[
            self.base_ppf(np.linspace(.0001, .9999, 401)),
            self.point * np.exp(student_t.ppf(np.linspace(.00001, .99999, 401), TAIL_DF) * self.tail_log_scale),
        ])
        mapped = np.maximum.accumulate(self.cdf(support))
        return np.interp(np.clip(p, mapped[0], mapped[-1]), mapped, support)

    def bucket_probabilities(self, boundaries: Iterable[float]) -> np.ndarray:
        edges = np.r_[0.0, np.asarray(tuple(boundaries), dtype=float), np.inf]
        cdf = np.r_[0.0, self.cdf(edges[1:-1]), 1.0]
        probs = np.maximum(0.0, np.diff(np.maximum.accumulate(cdf)))
        return probs / probs.sum()


def blend_base(left: BaseCdf, right: BaseCdf, left_weight: float) -> tuple[Callable[[float | np.ndarray], np.ndarray], Callable[[float | np.ndarray], np.ndarray]]:
    """Return a constrained mixture of base CDFs, not of tail layers."""
    weight = float(np.clip(left_weight, 0.0, 1.0))
    support = np.unique(np.r_[left.quantiles, right.quantiles])
    cdf = np.maximum.accumulate(weight * left.cdf(support) + (1.0 - weight) * right.cdf(support))

    def mixed(values: float | np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(values, dtype=float), support, cdf, left=0.0, right=1.0)

    def inverse(probabilities: float | np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(probabilities, dtype=float), cdf, support)

    return mixed, inverse


def origin_group(origin: int) -> str:
    if origin <= -10:
        return "-14_to_-10"
    if origin <= -5:
        return "-9_to_-5"
    if origin <= -2:
        return "-4_to_-2"
    return "-1"


def source_state(value: object) -> str:
    sources = {item.strip().lower() for item in str(value or "").split(",") if item.strip()}
    core = {"boxofficereport", "boxofficepro"}
    if sources == {"boxofficereport"}:
        return "boxofficereport_only"
    if sources == {"boxofficepro"}:
        return "boxofficepro_only"
    if sources == core:
        return "boxofficereport_boxofficepro"
    if core.issubset(sources) and "toddmthatcher" in sources:
        return "core_plus_todd"
    if core.issubset(sources) and "boxofficetheory" in sources:
        return "core_plus_boxofficetheory"
    return "other_sparse"


def _tail_scale(prior_residuals: np.ndarray) -> float:
    prior = np.asarray(prior_residuals, dtype=float)
    prior = prior[np.isfinite(prior)]
    if len(prior) < 30:
        return 0.35
    return max(float(np.median(np.abs(prior)) / student_t.ppf(.75, TAIL_DF)), 0.03)


def _shape(cdf: TailLayeredCdf) -> dict[str, float]:
    q025, q10, q50, q90, q975 = cdf.ppf([.025, .10, .50, .90, .975])
    lower = max(float(q50 - q10), 1.0)
    upper = max(float(q90 - q50), 1.0)
    return {
        "width_80_usd": float(q90 - q10),
        "skew_80": float(upper / lower),
        "upper_tail_ratio": float((q975 - q90) / upper),
        "lower_tail_ratio": float((q10 - q025) / lower),
    }


def _rps(probabilities: np.ndarray, winner: int) -> float:
    observed = np.zeros(len(probabilities)); observed[winner] = 1.0
    return float(np.mean((np.cumsum(probabilities)[:-1] - np.cumsum(observed)[:-1]) ** 2))


def _continuous_scores(cdf: TailLayeredCdf, actual: float, point: float) -> dict[str, float]:
    p = np.linspace(.001, .999, 401)
    q = cdf.ppf(p)
    error = actual - q
    pinball = np.where(error >= 0, p * error, (1.0 - p) * -error)
    crps = float(2.0 * np.trapezoid(pinball, p))
    return {"ncrps": crps / point, "pit": float(cdf.cdf(actual))}


def _failure_class(*, actual: float, point: float, pit: float, probabilities: np.ndarray, winner: int, point_bucket: int) -> str:
    point_error = abs(np.log(actual / point))
    if point_error >= .35:
        return "location_failure"
    if pit <= .025 or pit >= .975:
        return "scale_too_narrow"
    if winner not in (0, len(probabilities) - 1) and probabilities[winner] < max(probabilities[max(0, winner - 1):winner + 2]):
        return "interior_adjacent_bucket"
    if (pit < .10 and winner < point_bucket) or (pit > .90 and winner > point_bucket):
        return "incorrect_tail_thickness"
    return "unclear_or_well_allocated"


def _grid_rows(anchor: pd.Series, listing_origin: int) -> tuple[SyntheticMarketGrid, ...]:
    return generate_fixed_grid_ensemble(
        release_run_id=int(anchor.get("release_run_id", anchor.movie_id)),
        listing_origin=f"P_{listing_origin}",
        anchor_forecast_usd=float(anchor.frozen_point_forecast_usd),
    )


def _candidate_curves(row: pd.Series, candidates: dict[str, pd.Series], tail_scale: float, *, origin: int, state: str) -> dict[str, TailLayeredCdf]:
    point = float(row.frozen_point_forecast_usd)
    base = {name: BaseCdf.from_quantiles(candidate[QCOLS].to_numpy(float), point) for name, candidate in candidates.items()}
    out: dict[str, TailLayeredCdf] = {}
    for name, curve in base.items():
        out[name] = TailLayeredCdf(curve.cdf, curve.ppf, point, tail_scale)
    if {"D1_point_size", "D2_quantile_equal_centered"}.issubset(base):
        for weight in (.25, .50, .75):
            mixture, inverse = blend_base(base["D2_quantile_equal_centered"], base["D1_point_size"], weight)
            out[f"D2_D1_mix_w{weight:.2f}"] = TailLayeredCdf(mixture, inverse, point, tail_scale)
        state_weight = .75 if state in {"boxofficereport_boxofficepro", "core_plus_todd"} else .50
        mixture, inverse = blend_base(base["D2_quantile_equal_centered"], base["D1_point_size"], state_weight)
        out["D2_D1_source_state_mix"] = TailLayeredCdf(mixture, inverse, point, tail_scale)
    if {"D0_origin_empirical", "D2_quantile_equal_centered"}.issubset(base):
        mixture, inverse = blend_base(base["D2_quantile_equal_centered"], base["D0_origin_empirical"], .75)
        out["D2_D0_mix_w0.75"] = TailLayeredCdf(mixture, inverse, point, tail_scale)
    # A bounded shadow modifier; it is deliberately not a fitted power law.
    if "D2_quantile_equal_centered" in base:
        scale = .95 if point >= 50_000_000 else (1.00 if point >= 15_000_000 else 1.05)
        curve = base["D2_quantile_equal_centered"].scaled(scale)
        out["D2_bounded_point_size_scale_shadow"] = TailLayeredCdf(curve.cdf, curve.ppf, point, tail_scale)
    # Protect validated early origins.  Only the pre-registered shrunk-D2
    # challenger is permitted there; all mixture/scale research is late-only.
    if origin <= -5:
        allowed = {"production_policy_v1", "D2_source_shrunk_k40_equal_centered"}
        return {name: curve for name, curve in out.items() if name in allowed}
    if origin == -1:
        disallowed = {"D2_D0_mix_w0.75", "D2_bounded_point_size_scale_shadow"}
        return {name: curve for name, curve in out.items() if name not in disallowed}
    return out
    return out


def _load_panel(path: str | Path) -> pd.DataFrame:
    panel = pd.read_parquet(path)
    required = {"movie_id", "origin_day", "holdout_year", "frozen_point_forecast_usd", "actual_opening_weekend_gross_usd", "candidate", *QCOLS}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"locked distribution panel is missing {sorted(missing)}")
    return panel


def _merge_features(panel: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    features = oof[[column for column in ["movie_id", "origin_day", "holdout_year", "cross_source_log_disagreement", "aggregate_range_asymmetry_log", "source_count", "estimate_sources"] if column in oof]].drop_duplicates()
    return panel.merge(features, on=["movie_id", "origin_day", "holdout_year"], how="left", suffixes=("", "_oof"))


def _load_actual_grids(path: str | Path | None) -> dict[tuple[int, int], list[tuple[str, tuple[float, ...]]]]:
    """Read exact, already-aligned historical definitions when they exist.

    The current historical panel is intentionally optional: it is an external
    confirmation sample, never a source of candidate selection or weights.
    """
    if path is None or not Path(path).exists():
        return {}
    frame = pd.read_parquet(path) if Path(path).suffix == ".parquet" else pd.read_csv(path)
    required = {"movie_id", "origin", "event_bucket_definitions"}
    if not required.issubset(frame):
        return {}
    grids: dict[tuple[int, int], list[tuple[str, tuple[float, ...]]]] = {}
    for _, row in frame.iterrows():
        try:
            buckets = json.loads(row.event_bucket_definitions)
            boundaries = tuple(float(item["upper"]) for item in buckets[:-1] if item.get("upper") not in (None, ""))
            if len(boundaries) != len(buckets) - 1 or any(a >= b for a, b in zip(boundaries, boundaries[1:])):
                continue
            key = (int(row.movie_id), int(row.origin))
            grids.setdefault(key, []).append((f"actual-{row.get('event_id', len(grids))}", boundaries))
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return grids


def _summarize(scores: pd.DataFrame) -> pd.DataFrame:
    metrics = ["log_loss", "rps", "threshold_brier", "ncrps"]
    if scores.empty:
        return pd.DataFrame()
    # Required aggregation: variants -> movie-origin -> movie -> overall.
    movie_origin = scores.groupby(["candidate", "listing_origin", "movie_id", "origin_day"], as_index=False)[metrics].mean()
    movie = movie_origin.groupby(["candidate", "listing_origin", "movie_id"], as_index=False)[metrics].mean()
    summary = movie.groupby(["candidate", "listing_origin"], as_index=False)[metrics].mean()
    counts = movie.groupby(["candidate", "listing_origin"], as_index=False).agg(movies=("movie_id", "nunique"))
    return summary.merge(counts, on=["candidate", "listing_origin"])


def sequential_diagnostics(scores: pd.DataFrame) -> pd.DataFrame:
    """Measure probability turnover and whether it was supported by new input."""
    rows: list[dict[str, Any]] = []
    keys = ["candidate", "listing_origin", "grid_id", "movie_id"]
    for key, group in scores.sort_values("origin_day").groupby(keys):
        records = list(group.to_dict("records"))
        for previous, current in zip(records, records[1:]):
            if int(current["origin_day"]) != int(previous["origin_day"]) + 1:
                continue
            p0, p1 = np.asarray(previous["probability_vector"], float), np.asarray(current["probability_vector"], float)
            tv = float(.5 * np.abs(p1 - p0).sum())
            added_or_updated = bool(current["source_state"] != previous["source_state"] or current["source_count"] != previous["source_count"])
            point_changed = not np.isclose(float(current["frozen_point_forecast_usd"]), float(previous["frozen_point_forecast_usd"]), rtol=0, atol=1.0)
            score_improvement = float(previous["log_loss"] - current["log_loss"])
            rows.append({
                "candidate": key[0], "listing_origin": key[1], "grid_id": key[2], "movie_id": key[3],
                "from_origin": previous["origin_day"], "to_origin": current["origin_day"], "tv": tv,
                "score_improvement": score_improvement,
                "update_efficiency": score_improvement / tv if tv > 1e-12 else np.nan,
                "realized_bucket_probability_change": float(current["realized_bucket_probability"] - previous["realized_bucket_probability"]),
                "source_added_or_updated": added_or_updated,
                "point_changed": point_changed,
                "unsupported_probability_movement": bool(tv > .05 and not added_or_updated and not point_changed),
                "bucket_leader_reversal": int(current["bucket_leader"] != previous["bucket_leader"]),
            })
    return pd.DataFrame(rows)


def nested_outer_year_selection(scores: pd.DataFrame, *, bootstrap_iterations: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select constrained late-origin challengers using earlier outer years only."""
    target = scores.loc[scores.origin_day.isin(TARGET_ORIGINS)].copy()
    if target.empty:
        return pd.DataFrame(), pd.DataFrame()
    metrics = ["log_loss", "rps", "ncrps"]
    cases = target.groupby(["holdout_year", "listing_origin", "candidate", "movie_id", "origin_day"], as_index=False)[metrics].mean()
    selections: list[dict[str, Any]] = []
    selected: list[pd.DataFrame] = []
    for year in sorted(cases.holdout_year.unique()):
        for group in ("-4_to_-2", "-1"):
            test = cases.loc[(cases.holdout_year == year) & (cases.origin_day.map(origin_group) == group)]
            train = cases.loc[(cases.holdout_year < year) & (cases.origin_day.map(origin_group) == group)]
            if test.empty or train.empty or "production_policy_v1" not in set(train.candidate):
                continue
            means = train.groupby("candidate")[metrics].mean()
            production = means.loc["production_policy_v1"]
            eligible = means.loc[(means.rps <= production.rps * 1.005) & (means.ncrps <= production.ncrps * 1.005)]
            chosen = str(eligible.log_loss.idxmin()) if not eligible.empty else "production_policy_v1"
            selections.append({"holdout_year": int(year), "origin_group": group, "selected_candidate": chosen, "training_movies": int(train.movie_id.nunique()), **{f"training_{metric}": float(means.loc[chosen, metric]) for metric in metrics}})
            subset = test.loc[test.candidate == chosen].copy(); subset["selected_candidate"] = chosen; selected.append(subset)
    selection = pd.DataFrame(selections)
    if not selected:
        return selection, pd.DataFrame()
    selected_cases = pd.concat(selected, ignore_index=True)
    production = cases.loc[cases.candidate == "production_policy_v1"].rename(columns={metric: f"production_{metric}" for metric in metrics})
    paired = selected_cases.merge(production[["holdout_year", "listing_origin", "movie_id", "origin_day", *[f"production_{metric}" for metric in metrics]]], on=["holdout_year", "listing_origin", "movie_id", "origin_day"], how="inner")
    rng = np.random.default_rng(20260711)
    summary: list[dict[str, Any]] = []
    for metric in metrics:
        movie_delta = np.asarray(
            [float((part[metric] - part[f"production_{metric}"]).mean()) for _, part in paired.groupby("movie_id")],
            dtype=float,
        )
        if not len(movie_delta):
            continue
        draws = np.array([movie_delta[rng.integers(0, len(movie_delta), len(movie_delta))].mean() for _ in range(bootstrap_iterations)])
        summary.append({"metric": metric, "movies": len(movie_delta), "delta_selected_minus_production": float(movie_delta.mean()), "ci025": float(np.quantile(draws, .025)), "ci975": float(np.quantile(draws, .975)), "probability_selected_better": float((draws < 0).mean()), "bootstrap_iterations": bootstrap_iterations})
    return selection, pd.DataFrame(summary)


def transition_policy_evaluation(scores: pd.DataFrame, *, bootstrap_iterations: int = 2000) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate the pre-registered contiguous D1 transition policies."""
    synthetic = scores.loc[scores.grid_source.eq("synthetic")].copy()
    keys = ["movie_id", "holdout_year", "origin_day", "listing_origin", "grid_id"]
    production = synthetic.loc[synthetic.candidate.eq("production_policy_v1")]
    d1 = synthetic.loc[synthetic.candidate.eq("D1_point_size")]
    policies = {"current_production": 1, "D1_from_P_-2": -2, "D1_from_P_-3": -3, "D1_from_P_-4": -4}
    assembled: list[pd.DataFrame] = []
    for name, start in policies.items():
        if name == "current_production":
            chosen = production.copy()
        else:
            chosen = pd.concat([production.loc[production.origin_day < start], d1.loc[d1.origin_day >= start]], ignore_index=True)
        chosen["transition_policy"] = name
        assembled.append(chosen)
    panel = pd.concat(assembled, ignore_index=True)
    metrics = ["log_loss", "rps", "ncrps"]
    movie_origin = panel.groupby(["transition_policy", "listing_origin", "movie_id", "origin_day"], as_index=False)[metrics].mean()
    movie = movie_origin.groupby(["transition_policy", "listing_origin", "movie_id"], as_index=False)[metrics].mean()
    overall = movie.groupby(["transition_policy", "listing_origin"], as_index=False)[metrics].mean()
    baseline = overall.loc[overall.transition_policy.eq("current_production")].set_index("listing_origin")
    for metric in metrics:
        overall[f"delta_{metric}_vs_current"] = overall.apply(lambda row: float(row[metric] - baseline.loc[row.listing_origin, metric]), axis=1)
    by_origin = movie_origin.loc[movie_origin.origin_day.isin(TARGET_ORIGINS)].groupby(["transition_policy", "listing_origin", "origin_day"], as_index=False)[metrics].mean()
    base_origin = by_origin.loc[by_origin.transition_policy.eq("current_production")].set_index(["listing_origin", "origin_day"])
    for metric in metrics:
        by_origin[f"delta_{metric}_vs_current"] = by_origin.apply(lambda row: float(row[metric] - base_origin.loc[(row.listing_origin, row.origin_day), metric]), axis=1)
    rng = np.random.default_rng(20260711); boot_rows: list[dict[str, Any]] = []
    for name in policies:
        if name == "current_production":
            continue
        candidate = movie.loc[movie.transition_policy.eq(name)]
        base = movie.loc[movie.transition_policy.eq("current_production")]
        paired = candidate.merge(base, on=["listing_origin", "movie_id"], suffixes=("", "_base"))
        for listing, group in paired.groupby("listing_origin"):
            for metric in metrics:
                delta = (group[metric] - group[f"{metric}_base"]).to_numpy(float)
                draws = np.array([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(bootstrap_iterations)])
                boot_rows.append({"transition_policy": name, "listing_origin": listing, "metric": metric, "movies": len(delta), "delta": float(delta.mean()), "ci025": float(np.quantile(draws, .025)), "ci975": float(np.quantile(draws, .975)), "probability_better": float((draws < 0).mean())})
    folds = panel.loc[panel.origin_day.isin(TARGET_ORIGINS)].groupby(["transition_policy", "listing_origin", "holdout_year", "movie_id"], as_index=False)[metrics].mean()
    fold_summary: list[dict[str, Any]] = []
    for name in policies:
        if name == "current_production": continue
        paired = folds.loc[folds.transition_policy.eq(name)].merge(folds.loc[folds.transition_policy.eq("current_production")], on=["listing_origin", "holdout_year", "movie_id"], suffixes=("", "_base"))
        for (listing, year), group in paired.groupby(["listing_origin", "holdout_year"]):
            fold_summary.append({"transition_policy": name, "listing_origin": listing, "holdout_year": int(year), **{f"delta_{metric}_vs_current": float((group[metric] - group[f"{metric}_base"]).mean()) for metric in metrics}})
    return overall, by_origin, pd.DataFrame(boot_rows), pd.DataFrame(fold_summary)


def run_consensus_cdf_study(
    panel_path: str | Path = "data/diagnostics/fallback_adjusted_daily_policy/locked_rolling_origin_distribution_quantile_panel.parquet",
    output_dir: str | Path = "data/diagnostics/consensus_cdf_fixed_market_v1",
    *,
    oof_path: str | Path | None = None,
    actual_market_path: str | Path | None = "data/diagnostics/prediction_market_historical_price_panel_v6/10_complete_historical_panel.parquet",
    listing_origins: tuple[int, ...] = DEFAULT_LISTING_ORIGINS,
) -> dict[str, Any]:
    """Run the locked fixed-grid diagnostic and write auditable research outputs."""
    panel = _load_panel(panel_path)
    if oof_path is None:
        from eda.Prior.daily_distribution_validation import prepare_oof
        oof = prepare_oof()
    else:
        oof = pd.read_csv(oof_path)
    panel = _merge_features(panel, oof)
    actual_grids = _load_actual_grids(actual_market_path)
    wanted = ["production_policy_v1", "D0_origin_empirical", "D1_point_size", "D2_quantile_equal_centered", "D2_source_shrunk_k40_equal_centered"]
    panel = panel.loc[panel.candidate.isin(wanted)].copy()
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    groups = panel.groupby(["movie_id", "origin_day", "holdout_year"], sort=False)
    lookup = {key: group.set_index("candidate") for key, group in groups}
    records: list[dict[str, Any]] = []
    excluded_anchors = 0

    for (movie_id, listing_origin, year), candidate_rows in lookup.items():
        if int(listing_origin) not in listing_origins or "production_policy_v1" not in candidate_rows.index:
            continue
        anchor = candidate_rows.loc["production_policy_v1"]
        if not get_allowed_widths(float(anchor.frozen_point_forecast_usd)):
            excluded_anchors += 1
            continue
        grids = _grid_rows(anchor, int(listing_origin))
        # The tail scale is fit only from earlier outer years at the same
        # origin, matching the frozen safety rule's no-lookahead constraint.
        prior = oof.loc[(oof.holdout_year < year) & (oof.origin_day == listing_origin), "signed_log_error"].to_numpy(float)
        scale = _tail_scale(prior)
        movie_rows = panel.loc[(panel.movie_id == movie_id) & (panel.holdout_year == year) & (panel.origin_day >= listing_origin)]
        for origin, origin_rows in movie_rows.groupby("origin_day", sort=True):
            by_name = origin_rows.set_index("candidate")
            if "production_policy_v1" not in by_name.index:
                continue
            reference = by_name.loc["production_policy_v1"]
            candidate_input = {name: by_name.loc[name] for name in wanted if name in by_name.index}
            state = source_state(reference.get("estimate_sources", ""))
            curves = _candidate_curves(reference, candidate_input, scale, origin=int(origin), state=state)
            for name, curve in curves.items():
                actual, point = float(reference.actual_opening_weekend_gross_usd), float(reference.frozen_point_forecast_usd)
                continuous = _continuous_scores(curve, actual, point)
                pit = continuous["pit"]
                shape = _shape(curve)
                grid_specs = [("synthetic", grid.grid_id, grid.bucket_count, grid.alignment, tuple(grid.boundaries_usd), int(listing_origin)) for grid in grids]
                # Actual definitions are evaluated only on their exact matched
                # movie/origin rows and are excluded from the primary summary.
                grid_specs.extend(("actual", grid_id, len(boundaries) + 1, "historical_exact", boundaries, -99) for grid_id, boundaries in actual_grids.get((int(movie_id), int(origin)), []))
                for grid_source, grid_id, bucket_count, alignment, boundaries, grid_listing_origin in grid_specs:
                    probs = curve.bucket_probabilities(boundaries)
                    edges = np.r_[0.0, boundaries, np.inf]
                    winner = int(np.searchsorted(edges, actual, side="right") - 1)
                    winner = int(np.clip(winner, 0, len(probs) - 1))
                    point_bucket = int(np.searchsorted(edges, point, side="right") - 1)
                    point_bucket = int(np.clip(point_bucket, 0, len(probs) - 1))
                    records.append({
                        "candidate": name, "movie_id": int(movie_id), "holdout_year": int(year), "origin_day": int(origin),
                        "listing_origin": grid_listing_origin, "grid_source": grid_source, "grid_id": grid_id, "bucket_count": bucket_count,
                        "grid_alignment": alignment, "boundaries_usd": list(boundaries), "probability_vector": probs.tolist(),
                        "actual_opening_weekend_gross_usd": actual, "frozen_point_forecast_usd": point,
                        "realized_bucket": winner, "point_bucket": point_bucket, "bucket_distance_from_point": winner - point_bucket,
                        "bucket_position": "lower_tail" if winner == 0 else ("upper_tail" if winner == len(probs) - 1 else "interior"),
                        "realized_bucket_probability": float(probs[winner]), "log_loss": float(-np.log(max(probs[winner], 1e-12))),
                        "rps": _rps(probs, winner), "threshold_brier": _rps(probs, winner), "ncrps": continuous["ncrps"], "pit": pit,
                        "point_error_log": float(np.log(actual / point)), "source_count": int(reference.get("source_count", 0) or 0),
                        "source_state": state, "inter_source_spread": float(reference.get("cross_source_log_disagreement", np.nan)),
                        "range_asymmetry": float(reference.get("aggregate_range_asymmetry_log", np.nan)), "tail_log_scale": scale,
                        "bucket_leader": int(np.argmax(probs)), "failure_class": _failure_class(actual=actual, point=point, pit=pit, probabilities=probs, winner=winner, point_bucket=point_bucket),
                        **shape,
                    })
    scores = pd.DataFrame(records)
    if not scores.empty:
        scores.to_parquet(output / "01_fixed_listing_grid_scores.parquet", index=False)
    synthetic_scores = scores.loc[scores.grid_source.eq("synthetic")].copy() if not scores.empty else scores
    summary = _summarize(synthetic_scores)
    summary.to_csv(output / "02_movie_equal_market_scores.csv", index=False)
    by_origin = synthetic_scores.groupby(["candidate", "listing_origin", "origin_day"], as_index=False).agg(
        rows=("movie_id", "size"), movies=("movie_id", "nunique"), log_loss=("log_loss", "mean"), rps=("rps", "mean"), ncrps=("ncrps", "mean"), threshold_brier=("threshold_brier", "mean")
    ) if not scores.empty else pd.DataFrame()
    by_origin.to_csv(output / "03_scores_by_origin.csv", index=False)
    failures = synthetic_scores.groupby(["candidate", "origin_day", "bucket_position", "failure_class", "source_state"], as_index=False).agg(
        rows=("movie_id", "size"), mean_log_loss=("log_loss", "mean"), mean_pit=("pit", "mean"), mean_bucket_probability=("realized_bucket_probability", "mean")
    ) if not scores.empty else pd.DataFrame()
    failures.to_csv(output / "04_failure_decomposition.csv", index=False)
    sequential = sequential_diagnostics(synthetic_scores)
    sequential.to_csv(output / "05_sequential_probability_migration.csv", index=False)
    if not sequential.empty:
        sequential.groupby(["candidate", "to_origin"], as_index=False).agg(
            transitions=("movie_id", "size"), mean_tv=("tv", "mean"), mean_update_efficiency=("update_efficiency", "mean"), unsupported_movement_rate=("unsupported_probability_movement", "mean"), leader_reversals=("bucket_leader_reversal", "sum")
        ).to_csv(output / "06_sequential_summary.csv", index=False)
    actual_scores = scores.loc[scores.grid_source.eq("actual")].copy() if not scores.empty else pd.DataFrame()
    if actual_scores.empty:
        pd.DataFrame([{"status": "no_exact_historical_rows_matched", "definitions_loaded": sum(map(len, actual_grids.values()))}]).to_csv(output / "07_actual_market_grid_scores.csv", index=False)
    else:
        _summarize(actual_scores).to_csv(output / "07_actual_market_grid_scores.csv", index=False)
    selection, bootstrap = nested_outer_year_selection(synthetic_scores)
    selection.to_csv(output / "08_nested_outer_year_selection.csv", index=False)
    bootstrap.to_csv(output / "09_movie_clustered_bootstrap.csv", index=False)
    transitions = transition_policy_evaluation(scores)
    for name, frame in zip(("10_transition_policy_summary.csv", "11_transition_policy_by_origin.csv", "12_transition_policy_bootstrap.csv", "13_transition_policy_year_folds.csv"), transitions):
        frame.to_csv(output / name, index=False)
    manifest = {
        "study": "consensus_cdf_fixed_market_v1", "point_policy": "frozen", "tail_weight": TAIL_WEIGHT,
        "tail_distribution": "Student-t(4)", "listing_origin_policy": "canonical_P_-10_with_P_-14_P_-7_P_-4_robustness",
        "grid_bucket_counts": [4, 5, 6], "grid_recentered_at_each_origin": False,
        "actual_market_definitions": "external_confirmation_not_used_for_selection", "actual_definitions_loaded": sum(map(len, actual_grids.values())), "actual_rows_matched": len(actual_scores), "records": len(scores),
        "excluded_listing_anchors_outside_synthetic_domain": excluded_anchors,
        "targeted_origins": sorted(TARGET_ORIGINS),
    }
    (output / "00_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="data/diagnostics/fallback_adjusted_daily_policy/locked_rolling_origin_distribution_quantile_panel.parquet")
    parser.add_argument("--output", default="data/diagnostics/consensus_cdf_fixed_market_v1")
    parser.add_argument("--oof")
    parser.add_argument("--actual-grids")
    args = parser.parse_args()
    print(json.dumps(run_consensus_cdf_study(args.panel, args.output, oof_path=args.oof, actual_market_path=args.actual_grids), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
