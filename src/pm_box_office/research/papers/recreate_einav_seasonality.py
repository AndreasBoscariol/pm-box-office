#!/usr/bin/env python3
"""Recreate core results from Einav (2007), using reported or local data.

Paper:
    Seasonality in the U.S. motion picture industry

The ACNielsen EDI / Competitive Media Reporting movie-week panel used in the
paper is not bundled with this repository. By default this script recreates the
paper's reported-result artifacts:

* reported industry trends and sample descriptive statistics;
* the benchmark nested-logit estimates with and without movie fixed effects;
* the projection of estimated movie fixed effects on observables;
* summary facts about seasonality, amplification, and timing implications;
* simple SVG figures for the reported amplification and counterfactual results.

If you later add a movie-week panel, pass --panel-csv. In that mode the script
also computes a two-way-fixed-effect 2SLS analogue of the benchmark demand
equation:

    log(s_jt) - log(s_0t) =
        movie FE + season-week FE + beta * age_jt
        + sigma * log(s_jt / (1 - s_0t)) + error_jt

where log(s_jt / (1 - s_0t)) is instrumented with the number of movies in
release in that market week, after residualizing movie and season-week fixed
effects. The local-data estimator is intentionally transparent and dependency
free; it is a useful reproduction scaffold, not a substitute for the original
proprietary data and full econometric replication.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_OUT_DIR = Path("results/papers/einav_seasonality")

REPORTED_INDUSTRY_TRENDS = [
    (1985, 5.54, 167, 103, 3.51, 2.69),
    (1986, 5.70, 222, 120, 3.98, 2.94),
    (1987, 5.78, 220, 120, 4.28, 3.08),
    (1988, 5.84, 230, 144, 4.37, 3.08),
    (1989, 5.40, 213, 115, 4.92, 3.73),
    (1990, 5.45, 223, 119, 4.90, 3.64),
    (1991, 5.20, 224, 125, 4.63, 3.57),
    (1992, 4.97, 214, 121, 4.85, 3.86),
    (1993, 4.83, 227, 144, 5.09, 4.14),
    (1994, 4.74, 233, 142, 5.13, 4.20),
    (1995, 4.80, 246, 149, 5.27, 4.23),
    (1996, 4.74, 262, 152, 5.52, 4.44),
    (1997, 4.81, 250, 150, 5.90, 4.63),
    (1998, 4.91, 264, 139, 6.20, 4.73),
    (1999, 5.00, 244, 143, 6.80, 5.03),
]

REPORTED_TABLE2_STATS = [
    ("total_revenues_current_usd_millions", 1956, 34.8, 20.0, 43.5),
    ("population_share_percent_first_10_weeks", 1956, 2.78, 1.71, 3.16),
    ("first_week_revenues_current_usd_millions", 1956, 10.20, 6.63, 10.80),
    ("production_cost_dec1999_usd_millions", 1604, 29.7, 23.2, 22.6),
    ("advertising_expenditure_dec1999_usd_millions", 1873, 8.47, 7.19, 5.77),
]

REPORTED_TABLE2_LEADERS = [
    ("total_revenues_current_usd_millions", 1, "Titanic", 601),
    ("total_revenues_current_usd_millions", 2, "Star Wars (1999)", 431),
    ("total_revenues_current_usd_millions", 3, "Jurassic Park", 357),
    ("total_revenues_current_usd_millions", 4, "Forrest Gump", 330),
    ("total_revenues_current_usd_millions", 5, "Lion King", 313),
    ("population_share_percent_first_10_weeks", 1, "Titanic", 32),
    ("population_share_percent_first_10_weeks", 2, "Jurassic Park", 28),
    ("population_share_percent_first_10_weeks", 3, "Star Wars", 27),
    ("population_share_percent_first_10_weeks", 4, "Batman", 23),
    ("population_share_percent_first_10_weeks", 5, "Lion King", 23),
    ("first_week_revenues_current_usd_millions", 1, "The Lost World", 107),
    ("first_week_revenues_current_usd_millions", 2, "Star Wars (1999)", 99),
    ("first_week_revenues_current_usd_millions", 3, "Austin Powers 2", 85),
    ("first_week_revenues_current_usd_millions", 4, "Jurassic Park", 82),
    ("first_week_revenues_current_usd_millions", 5, "Independence Day", 79),
    ("production_cost_dec1999_usd_millions", 1, "Titanic", 209),
    ("production_cost_dec1999_usd_millions", 2, "Waterworld", 193),
    ("production_cost_dec1999_usd_millions", 3, "Armageddon", 155),
    ("production_cost_dec1999_usd_millions", 4, "Speed 2", 152),
    ("production_cost_dec1999_usd_millions", 5, "Tarzan", 152),
    ("advertising_expenditure_dec1999_usd_millions", 1, "Toy Story", 43),
    ("advertising_expenditure_dec1999_usd_millions", 2, "Titanic", 35),
    ("advertising_expenditure_dec1999_usd_millions", 3, "The Rookie", 35),
    ("advertising_expenditure_dec1999_usd_millions", 4, "Anastasia", 33),
    ("advertising_expenditure_dec1999_usd_millions", 5, "Forrest Gump", 31),
]

REPORTED_BENCHMARK = [
    ("with_movie_fixed_effects", "decay_beta", -0.220, 0.0014, 16103, 1956, 0.876),
    ("with_movie_fixed_effects", "nested_logit_sigma", 0.524, 0.030, 16103, 1956, 0.876),
    ("without_movie_fixed_effects", "decay_beta", -0.163, 0.005, 16103, 1956, 0.893),
    ("without_movie_fixed_effects", "nested_logit_sigma", 0.577, 0.011, 16103, 1956, 0.893),
]

REPORTED_SEASONALITY_DECOMPOSITION = [
    ("season_week_effect_sd_with_movie_fixed_effects", 0.238),
    ("season_week_effect_sd_without_movie_fixed_effects", 0.356),
    ("quality_endogeneity_amplification_ratio", 0.356 / 0.238),
    ("underlying_demand_share_of_gross_seasonal_variation", 0.238 / 0.356),
]

REPORTED_FIXED_EFFECT_PROJECTIONS = [
    ("log_production_cost", 0.215, 0.021, "**", 0.389, 0.021, "**", 0.384, 0.020, "**"),
    ("log_advertising_expenditure", 0.367, 0.019, "**", None, None, "", None, None, ""),
    ("mpaa_pg", "omitted", None, "", "omitted", None, "", None, None, ""),
    ("mpaa_pg13", -0.125, 0.038, "**", -0.135, 0.043, "**", None, None, ""),
    ("mpaa_r", -0.154, 0.039, "**", -0.144, 0.044, "**", None, None, ""),
    ("mpaa_above_r", -0.442, 0.365, "", -0.716, 0.413, "*", None, None, ""),
    ("genre_action", "omitted", None, "", "omitted", None, "", None, None, ""),
    ("genre_comedy", -0.052, 0.035, "", 0.029, 0.039, "", None, None, ""),
    ("genre_drama", -0.097, 0.037, "**", -0.021, 0.040, "", None, None, ""),
    ("genre_children", -0.070, 0.054, "", -0.037, 0.060, "", None, None, ""),
    ("best_picture_nominee", 0.433, 0.081, "**", None, None, "", None, None, ""),
    ("best_picture_award", 0.467, 0.139, "**", None, None, "", None, None, ""),
]

REPORTED_FIXED_EFFECT_PROJECTION_FOOTER = [
    ("year_fixed_effects", "yes", "yes", "no"),
    ("n", 1542, 1603, 1603),
    ("adjusted_r2", 0.387, 0.208, 0.193),
]

REPORTED_SUBPERIOD_DECAY = [
    ("1985_1989", -0.174, 0.0016, 4035, 572, 0.402),
    ("1990_1994", -0.206, 0.0014, 5521, 666, 0.430),
    ("1995_1999", -0.253, 0.0015, 6547, 754, 0.499),
]

REPORTED_TIMING = [
    ("observed_average_industry_share", 3.92),
    ("benchmark_model_constructed_industry_share", 3.94),
    ("coordinated_optimal_industry_share_upper_bound", 3.98),
    ("labor_day_vs_thanksgiving_small_movie_revenue_ratio", 2.0),
]

REPORTED_HEADLINE_FACTS = [
    ("original_titles", "3523"),
    ("wide_release_titles_used", "1956"),
    ("weekly_observations", "16103"),
    ("sample_revenue_coverage", "94% of original-sample revenues"),
    ("average_first_week_revenue_share", "almost 40%"),
    ("average_first_ten_weeks_revenue_share", "more than 90%"),
    ("main_decomposition", "underlying demand explains about two-thirds of gross seasonal variation"),
    ("main_market_response", "release-date choices amplify underlying seasonality by about 50%"),
    ("summer_pattern", "estimated underlying demand is fairly flat over summer and drops sharply after Labor Day"),
    ("labor_day_implication", "Labor Day combines high estimated demand with soft competition"),
]


def clean_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def float_or_none(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "").replace("$", "")
    if text == "" or text.lower() in {"na", "nan", "none"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def sample_sd(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{clean_key(key): value for key, value in row.items()} for row in reader]


def solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    augmented = [row[:] + [value] for row, value in zip(matrix, rhs)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            raise ValueError("singular matrix")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        pivot_value = augmented[col][col]
        for j in range(col, n + 1):
            augmented[col][j] /= pivot_value
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                augmented[row][j] -= factor * augmented[col][j]
    return [augmented[row][n] for row in range(n)]


def invert_matrix(matrix: list[list[float]]) -> list[list[float]]:
    columns = []
    for i in range(len(matrix)):
        rhs = [0.0] * len(matrix)
        rhs[i] = 1.0
        columns.append(solve_linear_system(matrix, rhs))
    return [[columns[col][row] for col in range(len(matrix))] for row in range(len(matrix))]


def ols(x: list[list[float]], y: list[float]) -> tuple[list[float], list[float], float, float]:
    width = len(x[0])
    gram = [[sum(row[i] * row[j] for row in x) for j in range(width)] for i in range(width)]
    target = [sum(row[i] * value for row, value in zip(x, y)) for i in range(width)]
    try:
        beta = solve_linear_system(gram, target)
    except ValueError:
        for i in range(width):
            gram[i][i] += 1e-9
        beta = solve_linear_system(gram, target)
    inv = invert_matrix(gram)
    predictions = [sum(coef * value for coef, value in zip(beta, row)) for row in x]
    residuals = [actual - pred for actual, pred in zip(y, predictions)]
    sse = sum(value * value for value in residuals)
    y_bar = mean(y)
    tss = sum((value - y_bar) ** 2 for value in y)
    df = max(1, len(y) - width)
    sigma2 = sse / df
    se = [math.sqrt(max(0.0, sigma2 * inv[i][i])) for i in range(width)]
    r2 = 1.0 - sse / tss if tss else 0.0
    return beta, se, r2, sse


def residualize_two_way(
    values: list[float],
    group_a: list[str],
    group_b: list[str],
    iterations: int = 200,
    tolerance: float = 1e-10,
) -> list[float]:
    residuals = values[:]
    previous_ss = sum(value * value for value in residuals)
    for _ in range(iterations):
        for groups in (group_a, group_b):
            totals: dict[str, float] = defaultdict(float)
            counts: dict[str, int] = defaultdict(int)
            for key, value in zip(groups, residuals):
                totals[key] += value
                counts[key] += 1
            means = {key: totals[key] / counts[key] for key in totals}
            residuals = [value - means[key] for value, key in zip(residuals, groups)]
        current_ss = sum(value * value for value in residuals)
        if abs(previous_ss - current_ss) <= tolerance * max(1.0, previous_ss):
            break
        previous_ss = current_ss
    return residuals


def estimate_two_way_effects(
    target: list[float],
    movie_ids: list[str],
    season_weeks: list[str],
    iterations: int = 500,
    tolerance: float = 1e-10,
) -> tuple[dict[str, float], dict[str, float]]:
    movie_effects = {key: 0.0 for key in set(movie_ids)}
    week_effects = {key: 0.0 for key in set(season_weeks)}
    previous_ss = float("inf")
    for _ in range(iterations):
        movie_totals: dict[str, float] = defaultdict(float)
        movie_counts: dict[str, int] = defaultdict(int)
        for value, movie, week in zip(target, movie_ids, season_weeks):
            movie_totals[movie] += value - week_effects[week]
            movie_counts[movie] += 1
        movie_effects = {key: movie_totals[key] / movie_counts[key] for key in movie_totals}

        week_totals: dict[str, float] = defaultdict(float)
        week_counts: dict[str, int] = defaultdict(int)
        for value, movie, week in zip(target, movie_ids, season_weeks):
            week_totals[week] += value - movie_effects[movie]
            week_counts[week] += 1
        week_effects = {key: week_totals[key] / week_counts[key] for key in week_totals}

        week_mean = mean(week_effects.values())
        week_effects = {key: value - week_mean for key, value in week_effects.items()}
        movie_effects = {key: value + week_mean for key, value in movie_effects.items()}

        residual_ss = 0.0
        for value, movie, week in zip(target, movie_ids, season_weeks):
            residual = value - movie_effects[movie] - week_effects[week]
            residual_ss += residual * residual
        if abs(previous_ss - residual_ss) <= tolerance * max(1.0, previous_ss):
            break
        previous_ss = residual_ss
    return movie_effects, week_effects


def find_first(row: dict[str, str], names: list[str]) -> str | None:
    for name in names:
        if name in row and row[name] != "":
            return row[name]
    return None


def prepare_panel(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    aliases = {
        "movie": "movie_id",
        "movieid": "movie_id",
        "title_id": "movie_id",
        "season_week": "calendar_week",
        "week_of_year": "calendar_week",
        "week_dummy": "calendar_week",
        "market_week": "period_id",
        "year_week": "period_id",
        "yearweek": "period_id",
        "weeks_since_release": "age",
        "week_since_release": "age",
        "week_in_release": "age",
        "release_age": "age",
        "box_office": "revenue",
        "boxoffice": "revenue",
        "weekly_revenue": "revenue",
        "share": "market_share",
        "movie_share": "market_share",
        "sjt": "market_share",
        "ticket": "ticket_price",
    }
    normalized_rows = []
    for row in rows:
        normalized: dict[str, str] = {}
        for key, value in row.items():
            normalized[aliases.get(key, key)] = value
        normalized_rows.append(normalized)

    panel = []
    for row in normalized_rows:
        movie_id = find_first(row, ["movie_id", "film_id", "title"])
        season_week = find_first(row, ["calendar_week"])
        age = float_or_none(find_first(row, ["age"]))
        if not movie_id or not season_week or age is None:
            continue

        period_id = find_first(row, ["period_id", "date"])
        if not period_id:
            year = find_first(row, ["year"])
            period_id = f"{year}_{season_week}" if year else None
        if not period_id:
            continue

        market_share = float_or_none(find_first(row, ["market_share"]))
        if market_share is None:
            revenue = float_or_none(find_first(row, ["revenue"]))
            ticket_price = float_or_none(find_first(row, ["ticket_price"]))
            population = float_or_none(find_first(row, ["population"]))
            if revenue is None or ticket_price is None or population is None or ticket_price <= 0 or population <= 0:
                continue
            market_share = revenue / ticket_price / population

        if market_share is None or market_share <= 0:
            continue
        panel.append(
            {
                "movie_id": str(movie_id),
                "season_week": str(season_week),
                "period_id": str(period_id),
                "age": float(age),
                "market_share": float(market_share),
            }
        )

    inside_share_by_period: dict[str, float] = defaultdict(float)
    movie_count_by_period: dict[str, int] = defaultdict(int)
    for row in panel:
        inside_share_by_period[str(row["period_id"])] += float(row["market_share"])
        movie_count_by_period[str(row["period_id"])] += 1

    prepared = []
    for row in panel:
        inside_share = inside_share_by_period[str(row["period_id"])]
        if inside_share <= 0 or inside_share >= 1:
            continue
        market_share = float(row["market_share"])
        y = math.log(market_share) - math.log(1.0 - inside_share)
        within = math.log(market_share / inside_share)
        prepared.append(
            {
                **row,
                "inside_share": inside_share,
                "outside_share": 1.0 - inside_share,
                "dependent_log_share": y,
                "log_within_industry_share": within,
                "movies_in_release": float(movie_count_by_period[str(row["period_id"])]),
            }
        )
    return prepared


def run_panel_estimation(out_dir: Path, panel_csv: Path) -> None:
    rows = read_csv_rows(panel_csv)
    panel = prepare_panel(rows)
    if len(panel) < 20:
        raise SystemExit(
            "Panel data did not contain enough usable rows. Required fields are movie_id, "
            "calendar_week, age, period_id or year, and either market_share or "
            "revenue + ticket_price + population."
        )

    y = [float(row["dependent_log_share"]) for row in panel]
    age = [float(row["age"]) for row in panel]
    within = [float(row["log_within_industry_share"]) for row in panel]
    instrument = [float(row["movies_in_release"]) for row in panel]
    movie_ids = [str(row["movie_id"]) for row in panel]
    season_weeks = [str(row["season_week"]) for row in panel]

    y_r = residualize_two_way(y, movie_ids, season_weeks)
    age_r = residualize_two_way(age, movie_ids, season_weeks)
    within_r = residualize_two_way(within, movie_ids, season_weeks)
    instrument_r = residualize_two_way(instrument, movie_ids, season_weeks)

    first_beta, first_se, first_r2, _ = ols([[a, z] for a, z in zip(age_r, instrument_r)], within_r)
    within_hat = [first_beta[0] * a + first_beta[1] * z for a, z in zip(age_r, instrument_r)]
    second_beta, second_se, second_r2, _ = ols([[a, w] for a, w in zip(age_r, within_hat)], y_r)
    ols_beta_values, ols_se_values, ols_r2, _ = ols([[a, w] for a, w in zip(age_r, within_r)], y_r)

    estimates = [
        {
            "model": "two_way_fe_2sls",
            "parameter": "decay_beta",
            "estimate": second_beta[0],
            "se_naive": second_se[0],
            "r2_residualized": second_r2,
            "n": len(panel),
            "movies": len(set(movie_ids)),
            "season_weeks": len(set(season_weeks)),
        },
        {
            "model": "two_way_fe_2sls",
            "parameter": "nested_logit_sigma",
            "estimate": second_beta[1],
            "se_naive": second_se[1],
            "r2_residualized": second_r2,
            "n": len(panel),
            "movies": len(set(movie_ids)),
            "season_weeks": len(set(season_weeks)),
        },
        {
            "model": "two_way_fe_ols_diagnostic",
            "parameter": "decay_beta",
            "estimate": ols_beta_values[0],
            "se_naive": ols_se_values[0],
            "r2_residualized": ols_r2,
            "n": len(panel),
            "movies": len(set(movie_ids)),
            "season_weeks": len(set(season_weeks)),
        },
        {
            "model": "two_way_fe_ols_diagnostic",
            "parameter": "nested_logit_sigma",
            "estimate": ols_beta_values[1],
            "se_naive": ols_se_values[1],
            "r2_residualized": ols_r2,
            "n": len(panel),
            "movies": len(set(movie_ids)),
            "season_weeks": len(set(season_weeks)),
        },
        {
            "model": "first_stage",
            "parameter": "movies_in_release_instrument",
            "estimate": first_beta[1],
            "se_naive": first_se[1],
            "r2_residualized": first_r2,
            "n": len(panel),
            "movies": len(set(movie_ids)),
            "season_weeks": len(set(season_weeks)),
        },
    ]
    write_csv(
        out_dir / "local_panel_nested_logit_estimates.csv",
        estimates,
        ["model", "parameter", "estimate", "se_naive", "r2_residualized", "n", "movies", "season_weeks"],
    )

    target_for_effects = [
        yy - second_beta[0] * aa - second_beta[1] * ww
        for yy, aa, ww in zip(y, age, within)
    ]
    _, week_effects = estimate_two_way_effects(target_for_effects, movie_ids, season_weeks)
    week_rows = [
        {"season_week": week, "estimated_underlying_demand_effect": value}
        for week, value in sorted(week_effects.items(), key=lambda item: natural_week_key(item[0]))
    ]
    write_csv(out_dir / "local_panel_estimated_week_effects.csv", week_rows, ["season_week", "estimated_underlying_demand_effect"])
    write_panel_descriptives(out_dir, panel)
    write_week_effect_svg(out_dir / "local_panel_week_effects.svg", week_rows)


def natural_week_key(value: str) -> tuple[int, str]:
    numeric = float_or_none(value)
    return (int(numeric) if numeric is not None else 9999, value)


def write_panel_descriptives(out_dir: Path, panel: list[dict[str, object]]) -> None:
    columns = ["age", "market_share", "inside_share", "dependent_log_share", "log_within_industry_share", "movies_in_release"]
    rows = []
    for column in columns:
        values = [float(row[column]) for row in panel]
        rows.append(
            {
                "variable": column,
                "n": len(values),
                "mean": mean(values),
                "std_dev": sample_sd(values),
                "min": min(values),
                "max": max(values),
            }
        )
    write_csv(out_dir / "local_panel_descriptive_stats.csv", rows, ["variable", "n", "mean", "std_dev", "min", "max"])


def write_reported_outputs(out_dir: Path) -> None:
    (out_dir / "README.txt").write_text(
        "\n".join(
            [
                "Einav (2007) seasonality reproduction artifacts",
                "",
                "Default mode recreates reported tables, estimates, and summary figures from the paper.",
                "The original ACNielsen EDI / Competitive Media Reporting movie-week panel is proprietary",
                "and is not included in this repository.",
                "",
                "To compute local-data analogues, rerun:",
                "  python -m pm_box_office.research.papers.recreate_einav_seasonality --panel-csv path/to/movie_week_panel.csv",
                "",
                "Panel columns required for the local nested-logit analogue:",
                "  movie_id, calendar_week, age, period_id or year, and either market_share",
                "  or revenue + ticket_price + population.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_csv(
        out_dir / "reported_industry_trends_table1.csv",
        [
            {
                "year": year,
                "average_ticket_price_dec1999_usd": ticket,
                "number_of_releases": releases,
                "number_of_wide_releases": wide,
                "total_box_office_revenues_dec1999_usd_billions": revenue,
                "admissions_per_capita": admissions,
            }
            for year, ticket, releases, wide, revenue, admissions in REPORTED_INDUSTRY_TRENDS
        ],
        [
            "year",
            "average_ticket_price_dec1999_usd",
            "number_of_releases",
            "number_of_wide_releases",
            "total_box_office_revenues_dec1999_usd_billions",
            "admissions_per_capita",
        ],
    )
    write_csv(
        out_dir / "reported_descriptive_stats_table2.csv",
        [
            {"variable": variable, "titles": titles, "mean": avg, "median": median, "std_dev": sd}
            for variable, titles, avg, median, sd in REPORTED_TABLE2_STATS
        ],
        ["variable", "titles", "mean", "median", "std_dev"],
    )
    write_csv(
        out_dir / "reported_table2_leading_movies.csv",
        [
            {"variable": variable, "rank": rank, "movie": movie, "value": value}
            for variable, rank, movie, value in REPORTED_TABLE2_LEADERS
        ],
        ["variable", "rank", "movie", "value"],
    )
    write_csv(
        out_dir / "reported_benchmark_estimates_figure3.csv",
        [
            {
                "model": model,
                "parameter": parameter,
                "estimate": estimate,
                "se": se,
                "n": n,
                "titles": titles,
                "r2": r2,
            }
            for model, parameter, estimate, se, n, titles, r2 in REPORTED_BENCHMARK
        ],
        ["model", "parameter", "estimate", "se", "n", "titles", "r2"],
    )
    write_csv(
        out_dir / "reported_seasonality_decomposition.csv",
        [{"metric": metric, "value": value} for metric, value in REPORTED_SEASONALITY_DECOMPOSITION],
        ["metric", "value"],
    )
    write_csv(
        out_dir / "reported_fixed_effect_projection_table3.csv",
        [
            {
                "variable": variable,
                "spec1_estimate": spec1,
                "spec1_se": se1,
                "spec1_significance": sig1,
                "spec2_estimate": spec2,
                "spec2_se": se2,
                "spec2_significance": sig2,
                "spec3_estimate": spec3,
                "spec3_se": se3,
                "spec3_significance": sig3,
            }
            for variable, spec1, se1, sig1, spec2, se2, sig2, spec3, se3, sig3 in REPORTED_FIXED_EFFECT_PROJECTIONS
        ],
        [
            "variable",
            "spec1_estimate",
            "spec1_se",
            "spec1_significance",
            "spec2_estimate",
            "spec2_se",
            "spec2_significance",
            "spec3_estimate",
            "spec3_se",
            "spec3_significance",
        ],
    )
    write_csv(
        out_dir / "reported_fixed_effect_projection_footer.csv",
        [{"metric": metric, "spec1": spec1, "spec2": spec2, "spec3": spec3} for metric, spec1, spec2, spec3 in REPORTED_FIXED_EFFECT_PROJECTION_FOOTER],
        ["metric", "spec1", "spec2", "spec3"],
    )
    write_csv(
        out_dir / "reported_subperiod_decay_robustness.csv",
        [
            {"period": period, "decay_beta": beta, "se": se, "n": n, "titles": titles, "r2": r2}
            for period, beta, se, n, titles, r2 in REPORTED_SUBPERIOD_DECAY
        ],
        ["period", "decay_beta", "se", "n", "titles", "r2"],
    )
    write_csv(
        out_dir / "reported_timing_implications.csv",
        [{"metric": metric, "value": value} for metric, value in REPORTED_TIMING],
        ["metric", "value"],
    )
    write_csv(
        out_dir / "reported_headline_facts.csv",
        [{"metric": metric, "value": value} for metric, value in REPORTED_HEADLINE_FACTS],
        ["metric", "value"],
    )
    write_amplification_svg(out_dir / "figure_reported_amplification.svg")
    write_counterfactual_svg(out_dir / "figure_reported_counterfactual_market_share.svg")


def svg_escape(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_amplification_svg(path: Path) -> None:
    bars = [
        ("With movie FE", 0.238, "#2f6f73"),
        ("Without movie FE", 0.356, "#c15b3f"),
    ]
    width, height = 760, 420
    left, bottom, top = 110, 340, 60
    scale = 250 / 0.40
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="40" y="34" font-family="Arial" font-size="20" font-weight="700">Reported seasonal amplification</text>',
        '<text x="40" y="58" font-family="Arial" font-size="13" fill="#555">Standard deviation of season-week effects in Einav Figure 3</text>',
        f'<line x1="{left}" y1="{bottom}" x2="690" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>',
    ]
    for tick in [0.0, 0.1, 0.2, 0.3, 0.4]:
        y = bottom - tick * scale
        parts.append(f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#333"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.1f}" font-family="Arial" font-size="11" text-anchor="end">{tick:.1f}</text>')
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="690" y2="{y:.1f}" stroke="#e8e8e8"/>')
    for i, (label, value, color) in enumerate(bars):
        x = 210 + i * 220
        y = bottom - value * scale
        parts.append(f'<rect x="{x}" y="{y:.1f}" width="100" height="{value * scale:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x + 50}" y="{bottom + 24}" font-family="Arial" font-size="13" text-anchor="middle">{svg_escape(label)}</text>')
        parts.append(f'<text x="{x + 50}" y="{y - 8:.1f}" font-family="Arial" font-size="13" font-weight="700" text-anchor="middle">{value:.3f}</text>')
    parts.append('<text x="40" y="390" font-family="Arial" font-size="13" fill="#333">Omitting movie fixed effects raises seasonal variation by about 50%, showing supply-side amplification.</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_counterfactual_svg(path: Path) -> None:
    bars = [
        ("Observed", 3.92, "#6c7a89"),
        ("Benchmark model", 3.94, "#2f6f73"),
        ("Coordinated optimum", 3.98, "#c15b3f"),
    ]
    width, height = 760, 420
    left, bottom, top = 110, 340, 60
    y_min, y_max = 3.88, 4.00
    scale = 260 / (y_max - y_min)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="40" y="34" font-family="Arial" font-size="20" font-weight="700">Reported industry-share counterfactual</text>',
        '<text x="40" y="58" font-family="Arial" font-size="13" fill="#555">Average industry share from Section 5 timing exercise</text>',
        f'<line x1="{left}" y1="{bottom}" x2="700" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>',
    ]
    for tick in [3.88, 3.92, 3.96, 4.00]:
        y = bottom - (tick - y_min) * scale
        parts.append(f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#333"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.1f}" font-family="Arial" font-size="11" text-anchor="end">{tick:.2f}</text>')
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="700" y2="{y:.1f}" stroke="#e8e8e8"/>')
    for i, (label, value, color) in enumerate(bars):
        x = 170 + i * 175
        y = bottom - (value - y_min) * scale
        parts.append(f'<rect x="{x}" y="{y:.1f}" width="95" height="{(value - y_min) * scale:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x + 47}" y="{bottom + 24}" font-family="Arial" font-size="12" text-anchor="middle">{svg_escape(label)}</text>')
        parts.append(f'<text x="{x + 47}" y="{y - 8:.1f}" font-family="Arial" font-size="13" font-weight="700" text-anchor="middle">{value:.2f}</text>')
    parts.append('<text x="40" y="390" font-family="Arial" font-size="13" fill="#333">The coordinated allocation amplifies releases more, but raises market share only from about 3.94 to 3.98.</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_week_effect_svg(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    values = [float(row["estimated_underlying_demand_effect"]) for row in rows]
    width, height = 900, 440
    left, right, top, bottom = 70, 30, 50, 330
    x_step = (width - left - right) / max(1, len(rows) - 1)
    y_min, y_max = min(values), max(values)
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    pad = (y_max - y_min) * 0.12
    y_min -= pad
    y_max += pad
    def sx(i: int) -> float:
        return left + i * x_step
    def sy(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
    points = " ".join(f"{sx(i):.1f},{sy(value):.1f}" for i, value in enumerate(values))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="36" y="30" font-family="Arial" font-size="20" font-weight="700">Local panel estimated season-week effects</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{width - right}" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>',
        f'<polyline points="{points}" fill="none" stroke="#2f6f73" stroke-width="2"/>',
    ]
    for i, row in enumerate(rows):
        if i % max(1, len(rows) // 12) == 0:
            parts.append(f'<text x="{sx(i):.1f}" y="{bottom + 20}" font-family="Arial" font-size="10" text-anchor="middle">{svg_escape(row["season_week"])}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for recreated result artifacts.")
    parser.add_argument(
        "--panel-csv",
        type=Path,
        help=(
            "Optional movie-week panel. Required columns: movie_id, calendar_week, age, "
            "period_id or year, and either market_share or revenue + ticket_price + population."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_reported_outputs(args.out_dir)
    if args.panel_csv:
        run_panel_estimation(args.out_dir, args.panel_csv)
    print(f"Wrote Einav reproduction artifacts to {args.out_dir}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
