#!/usr/bin/env python3
"""Compose live opening-weekend forecasts from baselines and optional daily nowcasts.

This script deliberately treats same-day live models as plug-in daily gross
nowcasts. It does not fit AMC coefficients. For each live regime it combines
known actuals, regime-appropriate baseline daily forecasts, and an optional
same-day plug-in nowcast, then simulates opening-weekend intervals from
component-level log-error uncertainty.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = REPO_ROOT / "data" / "predictions"
DIAGNOSTICS_DIR = REPO_ROOT / "data" / "diagnostics"

DEFAULT_BASELINE_CSV = PREDICTIONS_DIR / "daily_regime_baseline_forecasts.csv"
DEFAULT_PLUGIN_CSV = PREDICTIONS_DIR / "live_daily_plugin_nowcasts.csv"
DEFAULT_OUTPUT_CSV = PREDICTIONS_DIR / "live_weekend_plugin_forecasts.csv"
DEFAULT_COVERAGE_CSV = DIAGNOSTICS_DIR / "live_weekend_plugin_interval_coverage.csv"
DEFAULT_AUDIT_CSV = DIAGNOSTICS_DIR / "live_weekend_plugin_component_audit.csv"
DEFAULT_SIGMA_CSV = DIAGNOSTICS_DIR / "live_weekend_plugin_sigma_diagnostic.csv"

DEFAULT_ORIGINS = ("10:00", "12:00", "14:00", "16:00", "18:00", "20:00", "EOD")
DAYS = ("Friday", "Saturday", "Sunday")
REGIMES = ("live_friday", "live_saturday", "live_sunday")
DEFAULT_REGIME_SIGMA_MULTIPLIERS = {
    "live_friday": 0.90,
    "live_saturday": 1.15,
    "live_sunday": 1.00,
}
DEFAULT_REGIME_TAIL95_MULTIPLIERS = {
    "live_friday": 1.00,
    "live_saturday": 1.00,
    "live_sunday": 1.15,
}

REQUIRED_BASELINE_COLUMNS = {
    "movie_id",
    "title",
    "release_date",
    "pre_fri_usd",
    "pre_sat_usd",
    "pre_sun_usd",
    "after_fri_sat_usd",
    "after_fri_sun_usd",
    "after_sat_sun_usd",
    "actual_fri_usd",
    "actual_sat_usd",
    "actual_sun_usd",
    "actual_ow_usd",
}

REQUIRED_PLUGIN_COLUMNS = {
    "movie_id",
    "regime",
    "forecast_origin",
    "target_day",
    "pred_daily_gross_usd",
    "sigma_log_daily",
    "source",
}

ACTUAL_COLUMNS = {
    "Friday": "actual_fri_usd",
    "Saturday": "actual_sat_usd",
    "Sunday": "actual_sun_usd",
}

BASELINE_COLUMNS = {
    ("live_friday", "Friday"): ("pre_fri_usd", "pre-weekend baseline"),
    ("live_friday", "Saturday"): ("pre_sat_usd", "pre-weekend baseline"),
    ("live_friday", "Sunday"): ("pre_sun_usd", "pre-weekend baseline"),
    ("live_saturday", "Saturday"): ("after_fri_sat_usd", "after-Friday baseline"),
    ("live_saturday", "Sunday"): ("after_fri_sun_usd", "after-Friday baseline"),
    ("live_sunday", "Sunday"): ("after_sat_sun_usd", "after-Saturday baseline"),
}

KNOWN_ACTUAL_DAYS = {
    "live_friday": frozenset(),
    "live_saturday": frozenset({"Friday"}),
    "live_sunday": frozenset({"Friday", "Saturday"}),
}

PLUGIN_TARGET_DAY = {
    "live_friday": "Friday",
    "live_saturday": "Saturday",
    "live_sunday": "Sunday",
}


@dataclass(frozen=True)
class Component:
    day: str
    point: float
    sigma_log: float
    component_type: str
    component_source: str
    sigma_source: str
    raw_plugin_sigma_log: float = np.nan


def normalize_movie_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if np.isfinite(number) and number.is_integer():
        return str(int(number))
    return text


def normalize_day(value: Any) -> str:
    text = str(value).strip().lower()
    lookup = {
        "fri": "Friday",
        "friday": "Friday",
        "sat": "Saturday",
        "saturday": "Saturday",
        "sun": "Sunday",
        "sunday": "Sunday",
    }
    return lookup.get(text, str(value).strip())


def safe_float(value: Any) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(out) if np.isfinite(out) else np.nan


def read_baseline_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required baseline forecast file: {path}")
    frame = pd.read_csv(path)
    missing = REQUIRED_BASELINE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    out = frame.copy()
    out["_movie_id_key"] = out["movie_id"].map(normalize_movie_id)
    for column in [c for c in REQUIRED_BASELINE_COLUMNS if c.endswith("_usd")]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["release_date"] = pd.to_datetime(out["release_date"], errors="coerce").dt.date
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.sort_values(["release_date", "movie_id"]).reset_index(drop=True)


def read_plugin_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=sorted(REQUIRED_PLUGIN_COLUMNS) + ["_movie_id_key"])

    frame = pd.read_csv(path)
    missing = REQUIRED_PLUGIN_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    out = frame.copy()
    out["_movie_id_key"] = out["movie_id"].map(normalize_movie_id)
    out["regime"] = out["regime"].astype(str).str.strip()
    out["forecast_origin"] = out["forecast_origin"].astype(str).str.strip()
    out["target_day"] = out["target_day"].map(normalize_day)
    out["pred_daily_gross_usd"] = pd.to_numeric(out["pred_daily_gross_usd"], errors="coerce")
    out["sigma_log_daily"] = pd.to_numeric(out["sigma_log_daily"], errors="coerce")
    out["source"] = out["source"].fillna("plugin").astype(str)
    out = out.loc[out["regime"].isin(REGIMES) & out["target_day"].isin(DAYS)].copy()
    out = out.dropna(subset=["pred_daily_gross_usd"])
    out = out.loc[out["pred_daily_gross_usd"].gt(0)].copy()
    return out.drop_duplicates(["_movie_id_key", "regime", "forecast_origin", "target_day"], keep="last")


def baseline_component(row: pd.Series, regime: str, day: str) -> tuple[float, str]:
    column, source = BASELINE_COLUMNS[(regime, day)]
    return safe_float(row[column]), source


def log_errors(actual: pd.Series, pred: pd.Series) -> pd.Series:
    actual_num = pd.to_numeric(actual, errors="coerce")
    pred_num = pd.to_numeric(pred, errors="coerce")
    mask = actual_num.gt(0) & pred_num.gt(0)
    out = pd.Series(np.nan, index=actual_num.index, dtype="float64")
    out.loc[mask] = np.log(actual_num.loc[mask] / pred_num.loc[mask])
    return out.replace([np.inf, -np.inf], np.nan)


def build_sigma_table(
    baseline: pd.DataFrame,
    *,
    sigma_floor: float,
    default_sigma: float,
    min_sigma_n: int,
    regime_sigma_multipliers: dict[str, float],
) -> tuple[dict[tuple[str, str], tuple[float, str, int]], pd.DataFrame]:
    rows = []
    for regime in REGIMES:
        for day in DAYS:
            if day in KNOWN_ACTUAL_DAYS[regime]:
                continue
            pred_column, baseline_source = BASELINE_COLUMNS[(regime, day)]
            errors = log_errors(baseline[ACTUAL_COLUMNS[day]], baseline[pred_column])
            for error in errors.dropna():
                rows.append(
                    {
                        "regime": regime,
                        "target_day": day,
                        "baseline_source": baseline_source,
                        "log_error": float(error),
                    }
                )
    errors = pd.DataFrame(rows)
    sigmas: dict[tuple[str, str], tuple[float, str, int]] = {}
    diagnostic_rows = []

    all_errors = errors["log_error"].dropna() if not errors.empty else pd.Series(dtype="float64")
    global_sigma = float(all_errors.std(ddof=1)) if len(all_errors) >= 2 else default_sigma
    global_sigma = max(sigma_floor, global_sigma if np.isfinite(global_sigma) else default_sigma)

    for regime in REGIMES:
        for day in DAYS:
            if day in KNOWN_ACTUAL_DAYS[regime]:
                sigmas[(regime, day)] = (0.0, "actual", 0)
                diagnostic_rows.append(
                    {
                        "regime": regime,
                        "target_day": day,
                        "raw_sigma_log_daily": 0.0,
                        "regime_sigma_multiplier": regime_sigma_multipliers.get(regime, 1.0),
                        "sigma_log_daily": 0.0,
                        "sigma_source": "actual",
                        "sigma_n": 0,
                    }
                )
                continue

            group = errors.loc[errors["regime"].eq(regime) & errors["target_day"].eq(day), "log_error"].dropna()
            day_group = errors.loc[errors["target_day"].eq(day), "log_error"].dropna()
            if len(group) >= min_sigma_n:
                sigma = float(group.std(ddof=1))
                source = "regime_day_baseline_error"
                n = int(len(group))
            elif len(day_group) >= min_sigma_n:
                sigma = float(day_group.std(ddof=1))
                source = "pooled_day_baseline_error"
                n = int(len(day_group))
            elif len(all_errors) >= min_sigma_n:
                sigma = global_sigma
                source = "pooled_all_baseline_error"
                n = int(len(all_errors))
            else:
                sigma = default_sigma
                source = "default_sigma"
                n = int(len(all_errors))

            if not np.isfinite(sigma) or sigma <= 0:
                sigma = default_sigma
                source = "default_sigma"
            raw_sigma = float(max(sigma_floor, sigma))
            multiplier = regime_sigma_multipliers.get(regime, 1.0)
            if not np.isfinite(multiplier) or multiplier <= 0:
                multiplier = 1.0
            calibrated_sigma = float(max(sigma_floor, raw_sigma * multiplier))
            sigmas[(regime, day)] = (calibrated_sigma, source, n)
            diagnostic_rows.append(
                {
                    "regime": regime,
                    "target_day": day,
                    "raw_sigma_log_daily": raw_sigma,
                    "regime_sigma_multiplier": multiplier,
                    "sigma_log_daily": calibrated_sigma,
                    "sigma_source": source,
                    "sigma_n": n,
                }
            )

    sigma_diagnostic = pd.DataFrame(diagnostic_rows)
    return sigmas, sigma_diagnostic


def plugin_lookup(plugin: pd.DataFrame) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    if plugin.empty:
        return {}
    return {
        key: value
        for key, value in plugin.set_index(["_movie_id_key", "regime", "forecast_origin", "target_day"]).to_dict("index").items()
    }


def choose_component(
    row: pd.Series,
    *,
    regime: str,
    origin: str,
    day: str,
    plugins: dict[tuple[str, str, str, str], dict[str, Any]],
    sigmas: dict[tuple[str, str], tuple[float, str, int]],
    sigma_floor: float,
) -> Component:
    if day in KNOWN_ACTUAL_DAYS[regime]:
        return Component(
            day=day,
            point=safe_float(row[ACTUAL_COLUMNS[day]]),
            sigma_log=0.0,
            component_type="actual",
            component_source="known actual",
            sigma_source="actual",
        )

    fallback_sigma, fallback_source, _ = sigmas[(regime, day)]
    target_day = PLUGIN_TARGET_DAY[regime]
    plugin = plugins.get((row["_movie_id_key"], regime, origin, day)) if day == target_day else None
    if plugin is not None:
        raw_sigma = safe_float(plugin.get("sigma_log_daily"))
        effective_sigma = max(
            fallback_sigma,
            sigma_floor,
            raw_sigma if np.isfinite(raw_sigma) and raw_sigma > 0 else fallback_sigma,
        )
        sigma_source = "plugin_sigma_floored_by_baseline" if np.isfinite(raw_sigma) else fallback_source
        return Component(
            day=day,
            point=safe_float(plugin["pred_daily_gross_usd"]),
            sigma_log=float(effective_sigma),
            component_type="plugin",
            component_source=str(plugin.get("source", "plugin")),
            sigma_source=sigma_source,
            raw_plugin_sigma_log=raw_sigma,
        )

    point, baseline_source = baseline_component(row, regime, day)
    return Component(
        day=day,
        point=point,
        sigma_log=float(fallback_sigma),
        component_type="baseline",
        component_source=baseline_source,
        sigma_source=fallback_source,
    )


def simulate_ow(
    components: list[Component],
    *,
    n_sim: int,
    rng: np.random.Generator,
    tail95_multiplier: float,
) -> dict[str, float]:
    draws = np.zeros(n_sim, dtype="float64")
    for component in components:
        if not np.isfinite(component.point) or component.point < 0:
            return {
                "pred_ow_usd": np.nan,
                "ow_lo80": np.nan,
                "ow_hi80": np.nan,
                "ow_lo95": np.nan,
                "ow_hi95": np.nan,
            }
        if component.component_type == "actual" or component.sigma_log <= 0:
            draws += component.point
            continue
        eps = rng.normal(0.0, component.sigma_log, size=n_sim)
        draws += component.point * np.exp(eps)

    median = float(np.quantile(draws, 0.50))
    lo95 = float(np.quantile(draws, 0.025))
    hi95 = float(np.quantile(draws, 0.975))
    if np.isfinite(tail95_multiplier) and tail95_multiplier > 0 and tail95_multiplier != 1.0:
        lo95 = max(0.0, median - tail95_multiplier * (median - lo95))
        hi95 = median + tail95_multiplier * (hi95 - median)

    return {
        "pred_ow_usd": median,
        "ow_lo80": float(np.quantile(draws, 0.10)),
        "ow_hi80": float(np.quantile(draws, 0.90)),
        "ow_lo95": lo95,
        "ow_hi95": hi95,
    }


def format_component_summary(components: list[Component]) -> str:
    return "; ".join(
        f"{component.day} = {component.component_source}"
        if component.component_type != "plugin"
        else f"{component.day} = plug-in nowcast ({component.component_source})"
        for component in components
    )


def build_forecasts(
    baseline: pd.DataFrame,
    plugin: pd.DataFrame,
    *,
    origins: list[str],
    n_sim: int,
    seed: int,
    sigma_floor: float,
    default_sigma: float,
    min_sigma_n: int,
    regime_sigma_multipliers: dict[str, float],
    regime_tail95_multipliers: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sigmas, sigma_diagnostic = build_sigma_table(
        baseline,
        sigma_floor=sigma_floor,
        default_sigma=default_sigma,
        min_sigma_n=min_sigma_n,
        regime_sigma_multipliers=regime_sigma_multipliers,
    )
    plugins = plugin_lookup(plugin)
    rng = np.random.default_rng(seed)

    forecast_rows = []
    audit_rows = []
    for _, row in baseline.iterrows():
        for regime in REGIMES:
            for origin in origins:
                components = [
                    choose_component(
                        row,
                        regime=regime,
                        origin=origin,
                        day=day,
                        plugins=plugins,
                        sigmas=sigmas,
                        sigma_floor=sigma_floor,
                    )
                    for day in DAYS
                ]
                tail95_multiplier = regime_tail95_multipliers.get(regime, 1.0)
                sim = simulate_ow(
                    components,
                    n_sim=n_sim,
                    rng=rng,
                    tail95_multiplier=tail95_multiplier,
                )
                by_day = {component.day: component for component in components}
                plugin_components = [component for component in components if component.component_type == "plugin"]
                plugin_used = bool(plugin_components)
                plugin_source = "|".join(sorted({component.component_source for component in plugin_components}))
                actual_ow = safe_float(row["actual_ow_usd"])
                pred_ow = sim["pred_ow_usd"]
                ow_log_error = (
                    float(np.log(actual_ow / pred_ow))
                    if np.isfinite(actual_ow) and actual_ow > 0 and np.isfinite(pred_ow) and pred_ow > 0
                    else np.nan
                )
                covered_80 = (
                    bool(sim["ow_lo80"] <= actual_ow <= sim["ow_hi80"])
                    if np.isfinite(actual_ow) and np.isfinite(sim["ow_lo80"]) and np.isfinite(sim["ow_hi80"])
                    else np.nan
                )
                covered_95 = (
                    bool(sim["ow_lo95"] <= actual_ow <= sim["ow_hi95"])
                    if np.isfinite(actual_ow) and np.isfinite(sim["ow_lo95"]) and np.isfinite(sim["ow_hi95"])
                    else np.nan
                )

                base = {
                    "movie_id": row["movie_id"],
                    "title": row["title"],
                    "release_date": row["release_date"],
                    "regime": regime,
                    "forecast_origin": origin,
                    "regime_sigma_multiplier": regime_sigma_multipliers.get(regime, 1.0),
                    "regime_tail95_multiplier": tail95_multiplier,
                    "plugin_used": plugin_used,
                    "plugin_source": plugin_source if plugin_used else pd.NA,
                    "fri_component_type": by_day["Friday"].component_type,
                    "sat_component_type": by_day["Saturday"].component_type,
                    "sun_component_type": by_day["Sunday"].component_type,
                    "pred_fri_usd": by_day["Friday"].point,
                    "pred_sat_usd": by_day["Saturday"].point,
                    "pred_sun_usd": by_day["Sunday"].point,
                    "pred_ow_usd": pred_ow,
                    "ow_lo80": sim["ow_lo80"],
                    "ow_hi80": sim["ow_hi80"],
                    "ow_lo95": sim["ow_lo95"],
                    "ow_hi95": sim["ow_hi95"],
                    "actual_fri_usd": row["actual_fri_usd"],
                    "actual_sat_usd": row["actual_sat_usd"],
                    "actual_sun_usd": row["actual_sun_usd"],
                    "actual_ow_usd": actual_ow,
                    "ow_log_error": ow_log_error,
                    "covered_80": covered_80,
                    "covered_95": covered_95,
                }
                forecast_rows.append(base)

                audit = {
                    **base,
                    "component_summary": format_component_summary(components),
                }
                for day, prefix in [("Friday", "fri"), ("Saturday", "sat"), ("Sunday", "sun")]:
                    component = by_day[day]
                    audit[f"{prefix}_component_source"] = component.component_source
                    audit[f"{prefix}_sigma_log"] = component.sigma_log
                    audit[f"{prefix}_sigma_source"] = component.sigma_source
                    audit[f"{prefix}_raw_plugin_sigma_log"] = component.raw_plugin_sigma_log
                audit_rows.append(audit)

    forecasts = pd.DataFrame(forecast_rows)
    audits = pd.DataFrame(audit_rows)
    return forecasts, audits, sigma_diagnostic


def coverage_summary(forecasts: pd.DataFrame) -> pd.DataFrame:
    if forecasts.empty:
        return pd.DataFrame()

    frame = forecasts.copy()
    frame["plugin_source_group"] = frame["plugin_source"].fillna("none")
    rows = []
    for keys, group in frame.groupby(["regime", "forecast_origin", "plugin_used", "plugin_source_group"], dropna=False):
        regime, origin, plugin_used, plugin_source = keys
        valid = group.loc[
            pd.to_numeric(group["actual_ow_usd"], errors="coerce").gt(0)
            & pd.to_numeric(group["pred_ow_usd"], errors="coerce").gt(0)
        ].copy()
        log_error = pd.to_numeric(valid["ow_log_error"], errors="coerce")
        width80 = pd.to_numeric(valid["ow_hi80"], errors="coerce") - pd.to_numeric(valid["ow_lo80"], errors="coerce")
        width95 = pd.to_numeric(valid["ow_hi95"], errors="coerce") - pd.to_numeric(valid["ow_lo95"], errors="coerce")
        rows.append(
            {
                "regime": regime,
                "forecast_origin": origin,
                "plugin_used": bool(plugin_used),
                "plugin_source": plugin_source,
                "n": int(len(group)),
                "n_scored": int(len(valid)),
                "coverage_80": float(valid["covered_80"].mean()) if len(valid) else np.nan,
                "coverage_95": float(valid["covered_95"].mean()) if len(valid) else np.nan,
                "mae_log_ow": float(log_error.abs().mean()) if len(log_error.dropna()) else np.nan,
                "rmse_log_ow": float(np.sqrt(np.mean(np.square(log_error.dropna())))) if len(log_error.dropna()) else np.nan,
                "median_width_80_usd": float(width80.median()) if len(width80.dropna()) else np.nan,
                "median_width_95_usd": float(width95.median()) if len(width95.dropna()) else np.nan,
                "median_width_80_to_pred": float((width80 / valid["pred_ow_usd"]).median()) if len(valid) else np.nan,
                "median_width_95_to_pred": float((width95 / valid["pred_ow_usd"]).median()) if len(valid) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["regime", "forecast_origin", "plugin_used", "plugin_source"])


def parse_origins(raw: str, plugin: pd.DataFrame) -> list[str]:
    origins = [value.strip() for value in raw.split(",") if value.strip()]
    for origin in plugin["forecast_origin"].dropna().astype(str).unique().tolist():
        if origin not in origins:
            origins.append(origin)
    return origins


def format_regime_float_map(values: dict[str, float]) -> str:
    return ",".join(f"{regime}={values[regime]:g}" for regime in REGIMES)


def parse_regime_float_map(raw: str, defaults: dict[str, float], *, label: str) -> dict[str, float]:
    parsed = defaults.copy()
    if not raw:
        return parsed
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"{label} entries must use regime=value format: {item!r}")
        regime, value = [part.strip() for part in item.split("=", 1)]
        if regime not in REGIMES:
            raise ValueError(f"Unknown regime in {label}: {regime!r}; expected one of {REGIMES}")
        number = float(value)
        if not np.isfinite(number) or number <= 0:
            raise ValueError(f"{label} values must be positive finite numbers: {item!r}")
        parsed[regime] = number
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--plugin-csv", type=Path, default=DEFAULT_PLUGIN_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--coverage-csv", type=Path, default=DEFAULT_COVERAGE_CSV)
    parser.add_argument("--component-audit-csv", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--sigma-csv", type=Path, default=DEFAULT_SIGMA_CSV)
    parser.add_argument("--forecast-origins", default=",".join(DEFAULT_ORIGINS))
    parser.add_argument("--n-sim", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--sigma-floor", type=float, default=0.15)
    parser.add_argument("--default-sigma", type=float, default=0.35)
    parser.add_argument("--min-sigma-n", type=int, default=8)
    parser.add_argument(
        "--regime-sigma-multipliers",
        default=format_regime_float_map(DEFAULT_REGIME_SIGMA_MULTIPLIERS),
        help="Comma-separated regime=k values applied to component log sigmas.",
    )
    parser.add_argument(
        "--regime-tail95-multipliers",
        default=format_regime_float_map(DEFAULT_REGIME_TAIL95_MULTIPLIERS),
        help="Comma-separated regime=k values that widen only the simulated 95%% interval around the median.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    baseline = read_baseline_csv(args.baseline_csv)
    plugin = read_plugin_csv(args.plugin_csv)
    origins = parse_origins(args.forecast_origins, plugin)
    regime_sigma_multipliers = parse_regime_float_map(
        args.regime_sigma_multipliers,
        DEFAULT_REGIME_SIGMA_MULTIPLIERS,
        label="regime sigma multipliers",
    )
    regime_tail95_multipliers = parse_regime_float_map(
        args.regime_tail95_multipliers,
        DEFAULT_REGIME_TAIL95_MULTIPLIERS,
        label="regime 95-tail multipliers",
    )

    forecasts, audits, sigma_diagnostic = build_forecasts(
        baseline,
        plugin,
        origins=origins,
        n_sim=args.n_sim,
        seed=args.seed,
        sigma_floor=args.sigma_floor,
        default_sigma=args.default_sigma,
        min_sigma_n=args.min_sigma_n,
        regime_sigma_multipliers=regime_sigma_multipliers,
        regime_tail95_multipliers=regime_tail95_multipliers,
    )
    coverage = coverage_summary(forecasts)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.coverage_csv.parent.mkdir(parents=True, exist_ok=True)
    args.component_audit_csv.parent.mkdir(parents=True, exist_ok=True)
    args.sigma_csv.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(args.output_csv, index=False)
    coverage.to_csv(args.coverage_csv, index=False)
    audits.to_csv(args.component_audit_csv, index=False)
    sigma_diagnostic.to_csv(args.sigma_csv, index=False)

    plugin_note = f"{len(plugin):,} plug-in rows" if not plugin.empty else "no plug-in file/rows; baseline fallback only"
    print(f"Wrote {args.output_csv} ({len(forecasts):,} rows, {plugin_note})")
    print(f"Wrote {args.coverage_csv} ({len(coverage):,} rows)")
    print(f"Wrote {args.component_audit_csv} ({len(audits):,} rows)")
    print(f"Wrote {args.sigma_csv} ({len(sigma_diagnostic):,} rows)")


if __name__ == "__main__":
    main()
