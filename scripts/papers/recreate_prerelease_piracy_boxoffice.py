#!/usr/bin/env python3
"""Recreate results from Ma, Montgomery, Singh, and Smith.

Paper:
    An Empirical Analysis of the Impact of Pre-Release Movie Piracy on
    Box-Office Revenue

The raw Nielsen/Vcdquality movie-week panel used by the paper is not bundled
with this repository. By default this script uses the OCR markdown of the
paper to recreate reported-result artifacts:

* every embedded paper table as a CSV;
* curated CSVs for the headline summary facts, key coefficient estimates,
  revenue-loss calculations, and robustness estimates;
* simple SVG figures for the piracy revenue paths and robustness patterns.

If you later add a movie-week panel, pass --panel-csv. In that mode the script
also computes data-derived descriptive statistics and pooled-OLS analogues of
the paper's exponential revenue models. The published estimates use feasible
GLS/random effects, so local OLS outputs are diagnostic unless you extend this
script with the original GLS estimator and raw data.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable


DEFAULT_PAPER_MD = Path("docs/papers_ocr/SSRN-id1782924.pdf_by_PaddleOCR-VL-1.6.md")

MARKET_POTENTIAL_TERMS = [
    "budget",
    "missing_budget",
    "screen",
    "director_appeal",
    "user_rating",
    "critic_rating",
    "star_appeal",
    "g",
    "r",
    "pg13",
    "warner",
    "universal",
    "paramount",
    "fox",
    "sony",
    "newline",
    "lionsgate",
    "mgm",
    "action",
    "comedy",
    "drama",
    "adventure",
    "horror",
    "thriller",
    "animation",
]

DECAY_TERMS = ["user_rating", "critic_rating", "director_appeal", "star_appeal"]

REPORTED_SUMMARY_FACTS = [
    ("sample_period", "Wide-release U.S. movies from February 2006 through December 2008"),
    ("movies_total", "533"),
    ("movies_with_prerelease_piracy", "52"),
    ("movies_without_prerelease_piracy", "481"),
    ("prerelease_piracy_share", "0.10"),
    ("mean_prerelease_piracy_weeks_before_release", "7.04"),
    ("model_estimation_movies", "475"),
    ("model_estimation_prerelease_piracy_movies", "48"),
    ("known_budget_movies", "375"),
    ("known_budget_prerelease_piracy_movies", "40"),
    ("average_theatrical_run_weeks_used_for_loss", "12"),
    ("headline_revenue_loss_percent", "19.1"),
]

REPORTED_MODEL_ESTIMATES = [
    ("table6_homogeneous_decline", "tau", -0.1929, 0.0222, "***", "piracy effect on rate of decline"),
    ("table6_homogeneous_decline", "rho", -0.7399, 0.1767, "***", "piracy effect on market potential"),
    ("table6_homogeneous_decline", "lambda", 0.7600, 0.0071, "***", "baseline rate of decline"),
    ("table7_heterogeneous_decline", "tau", -0.0965, 0.0208, "***", "piracy effect on rate of decline"),
    ("table7_heterogeneous_decline", "rho", -0.4024, 0.1746, "*", "piracy effect on market potential"),
    ("table7_heterogeneous_decline", "lambda", 0.7503, 0.0064, "***", "average baseline rate of decline"),
    ("table8_piracy_quality", "tau_1", -0.0963, 0.0208, "***", "piracy indicator effect on rate of decline"),
    ("table8_piracy_quality", "tau_2", -0.0162, 0.0126, "", "pirated quality effect on rate of decline"),
    ("table8_piracy_quality", "rho_1", -0.4022, 0.1746, "*", "piracy indicator effect on market potential"),
    ("table8_piracy_quality", "rho_2", -0.0669, 0.1066, "", "pirated quality effect on market potential"),
    ("table9_propensity_score_matching", "tau", -0.1204, 0.0280, "***", "matched-sample piracy effect on rate of decline"),
    ("table9_propensity_score_matching", "rho", -0.4874, 0.2228, "*", "matched-sample piracy effect on market potential"),
    ("table10_timing", "tau_1", -0.0964, 0.0208, "***", "piracy indicator effect on rate of decline"),
    ("table10_timing", "rho_1", -0.3992, 0.1745, "*", "piracy indicator effect on market potential"),
    ("table10_timing", "rho_2", -0.1999, 0.1696, "", "log piracy-week effect on market potential"),
    ("table10_timing", "tau_2", -0.0058, 0.0203, "", "log piracy-week effect on rate of decline"),
    ("table12_known_budget", "tau", -0.1201, 0.0218, "***", "known-budget sample piracy effect on rate of decline"),
    ("table12_known_budget", "rho", -0.4874, 0.1831, "**", "known-budget sample piracy effect on market potential"),
]

REPORTED_ALTERNATIVE_WEEKS = [
    (4, -0.0716, 0.0313, -0.3231, 0.1555),
    (5, -0.1013, 0.0249, -0.4048, 0.1637),
    (6, -0.0965, 0.0208, -0.4024, 0.1746),
    (7, -0.0963, 0.0177, -0.4212, 0.1799),
    (8, -0.0841, 0.0151, -0.4673, 0.1814),
    (9, -0.0937, 0.0140, -0.4654, 0.1859),
]


@dataclass
class ExtractedTable:
    table_id: int
    caption: str
    rows: list[list[str]]


@dataclass
class OLSResult:
    equation: str
    n: int
    r2: float
    rows: list[dict[str, object]]


class TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self.in_table = False
        self.in_cell = False
        self.current_table: list[list[str]] = []
        self.current_row: list[str] = []
        self.current_cell: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self.in_table = True
            self.current_table = []
        elif self.in_table and tag == "tr":
            self.current_row = []
        elif self.in_table and tag in {"td", "th"}:
            self.in_cell = True
            self.current_cell = []

    def handle_endtag(self, tag: str) -> None:
        if self.in_table and tag in {"td", "th"}:
            self.current_row.append(clean_text("".join(self.current_cell)))
            self.in_cell = False
        elif self.in_table and tag == "tr":
            if self.current_row:
                self.current_table.append(self.current_row)
        elif tag == "table" and self.in_table:
            self.tables.append(self.current_table)
            self.in_table = False

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.current_cell.append(data)


def clean_text(value: str) -> str:
    value = html.unescape(re.sub(r"<[^>]+>", " ", value))
    value = value.replace("\xa0", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_numeric_text(value: object) -> str:
    text = str(value).strip()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = re.sub(r"\s+", "", text)
    text = text.replace("M", "e6").replace("K", "e3")
    return text


def float_or_none(value: object) -> float | None:
    if value is None:
        return None
    text = normalize_numeric_text(value)
    if text == "" or text.lower() in {"na", "nan", "none"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def log_positive(value: object) -> float | None:
    number = float_or_none(value)
    if number is None or number <= 0:
        return None
    return math.log(number)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def sample_sd(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


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
    n = len(matrix)
    columns = []
    for i in range(n):
        rhs = [0.0] * n
        rhs[i] = 1.0
        columns.append(solve_linear_system(matrix, rhs))
    return [[columns[col][row] for col in range(n)] for row in range(n)]


def ols_beta(x: list[list[float]], y: list[float]) -> tuple[list[float], list[list[float]], float]:
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
    sse = sum((actual - pred) ** 2 for actual, pred in zip(y, predictions))
    return beta, inv, sse


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_grid_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = max((len(row) for row in rows), default=0)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for row in rows:
            writer.writerow(row + [""] * (width - len(row)))


def extract_tables(markdown: str) -> list[ExtractedTable]:
    matches = list(re.finditer(r"<table\b.*?</table>", markdown, flags=re.DOTALL | re.IGNORECASE))
    extracted = []
    for index, match in enumerate(matches, start=1):
        parser = TableHTMLParser()
        parser.feed(match.group(0))
        rows = parser.tables[0] if parser.tables else []
        next_lines = markdown[match.end() : match.end() + 1200].splitlines()[:20]
        caption = f"Table {index}"
        for line in next_lines:
            text = clean_text(line)
            if re.search(r"\bTable\s+\d+\b", text, flags=re.IGNORECASE):
                caption = short_caption(text)
                break
        extracted.append(ExtractedTable(table_id=index, caption=caption, rows=rows))
    return extracted


def short_caption(text: str) -> str:
    for marker in (
        " Each of these",
        " The descriptive",
        " In Table",
        " Our dataset",
        " With respect",
        " The coefficients",
        " In summary",
        " ### ",
        " The results",
        " ## ",
    ):
        if marker in text:
            text = text.split(marker, 1)[0]
    if len(text) > 220:
        text = text[:217].rstrip() + "..."
    return text


def write_extracted_tables(out_dir: Path, paper_md: Path) -> list[ExtractedTable]:
    if not paper_md.exists():
        print(f"warning: OCR markdown not found, skipping extraction: {paper_md}", file=sys.stderr)
        return []
    markdown = paper_md.read_text(encoding="utf-8")
    tables = extract_tables(markdown)
    table_dir = out_dir / "extracted_paper_tables"
    index_rows = []
    for table in tables:
        filename = f"table_{table.table_id:02d}.csv"
        write_grid_csv(table_dir / filename, table.rows)
        index_rows.append(
            {
                "table_id": table.table_id,
                "caption": table.caption,
                "rows": len(table.rows),
                "columns_max": max((len(row) for row in table.rows), default=0),
                "csv": str(Path("extracted_paper_tables") / filename),
            }
        )
    write_csv(out_dir / "paper_tables_index.csv", index_rows, ["table_id", "caption", "rows", "columns_max", "csv"])
    return tables


def cumulative_revenue_loss(lambda_value: float, rho: float, tau: float, weeks: int = 12) -> float:
    baseline = sum(math.exp(-lambda_value * t) for t in range(1, weeks + 1))
    pirated = sum(math.exp(rho - (lambda_value + tau) * t) for t in range(1, weeks + 1))
    return 1.0 - pirated / baseline


def write_reported_outputs(out_dir: Path) -> None:
    write_csv(
        out_dir / "reported_summary_facts.csv",
        [{"metric": metric, "value": value} for metric, value in REPORTED_SUMMARY_FACTS],
        ["metric", "value"],
    )
    write_csv(
        out_dir / "reported_key_estimates.csv",
        [
            {
                "source": source,
                "parameter": parameter,
                "estimate": estimate,
                "se": se,
                "significance": significance,
                "note": note,
            }
            for source, parameter, estimate, se, significance, note in REPORTED_MODEL_ESTIMATES
        ],
        ["source", "parameter", "estimate", "se", "significance", "note"],
    )
    write_csv(
        out_dir / "reported_alternative_week_thresholds.csv",
        [
            {
                "number_of_weeks": weeks,
                "tau": tau,
                "tau_se": tau_se,
                "rho": rho,
                "rho_se": rho_se,
            }
            for weeks, tau, tau_se, rho, rho_se in REPORTED_ALTERNATIVE_WEEKS
        ],
        ["number_of_weeks", "tau", "tau_se", "rho", "rho_se"],
    )

    loss_rows = []
    for label, lambda_value, rho, tau, reported_loss in [
        ("table6_homogeneous_decline", 0.7600, -0.7399, -0.1929, 0.289),
        ("table7_heterogeneous_decline", 0.7503, -0.4024, -0.0965, 0.191),
    ]:
        calculated = cumulative_revenue_loss(lambda_value, rho, tau)
        loss_rows.append(
            {
                "source": label,
                "lambda": lambda_value,
                "rho": rho,
                "tau": tau,
                "weeks": 12,
                "calculated_loss": f"{calculated:.6f}",
                "reported_loss": f"{reported_loss:.3f}",
            }
        )
    write_csv(
        out_dir / "reported_revenue_loss_calculation.csv",
        loss_rows,
        ["source", "lambda", "rho", "tau", "weeks", "calculated_loss", "reported_loss"],
    )

    write_revenue_paths_svg(out_dir / "figure_revenue_paths.svg")
    write_robustness_svg(out_dir / "figure_alternative_week_thresholds.svg")


def canonicalize_columns(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    aliases = {
        "movieid": "movie_id",
        "movie_id": "movie_id",
        "id": "movie_id",
        "week": "week",
        "t": "week",
        "revenue": "revenue",
        "box_office": "revenue",
        "boxoffice": "revenue",
        "weekly_revenue": "revenue",
        "budget": "budget",
        "missingbudget": "missing_budget",
        "missing_budget": "missing_budget",
        "screen": "screen",
        "screens": "screen",
        "opening_screens": "screen",
        "prerelease_piracy": "prerelease_piracy",
        "pre_release_piracy": "prerelease_piracy",
        "pir": "prerelease_piracy",
        "piracy": "prerelease_piracy",
        "pirated_quality": "pirated_quality",
        "pirqual": "pirated_quality",
        "prerelease_piracy_week": "prerelease_piracy_week",
        "pirweek": "prerelease_piracy_week",
    }
    normalized_rows = []
    for row in rows:
        normalized: dict[str, str] = {}
        for key, value in row.items():
            cleaned = re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")
            normalized[aliases.get(cleaned, cleaned)] = value
        normalized_rows.append(normalized)
    return normalized_rows


def required_columns(rows: list[dict[str, str]], columns: list[str], label: str) -> bool:
    if not rows:
        print(f"warning: {label} is empty", file=sys.stderr)
        return False
    available = set(rows[0])
    missing = [col for col in columns if col not in available]
    if missing:
        print(f"warning: {label} is missing required columns: {', '.join(missing)}", file=sys.stderr)
        return False
    return True


def write_data_descriptives(out_dir: Path, rows: list[dict[str, str]], prefix: str) -> None:
    if not rows:
        return
    output = []
    for col in rows[0]:
        values = [float_or_none(row.get(col)) for row in rows]
        clean = [value for value in values if value is not None]
        if not clean:
            continue
        sorted_values = sorted(clean)
        mid = len(sorted_values) // 2
        median = sorted_values[mid] if len(sorted_values) % 2 else (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
        output.append(
            {
                "variable": col,
                "n": len(clean),
                "mean": f"{mean(clean):.10g}",
                "median": f"{median:.10g}",
                "std_dev": f"{sample_sd(clean):.10g}",
                "min": f"{min(clean):.10g}",
                "max": f"{max(clean):.10g}",
            }
        )
    write_csv(out_dir / f"{prefix}_descriptive_stats.csv", output, ["variable", "n", "mean", "median", "std_dev", "min", "max"])


def value_for_term(row: dict[str, str], term: str) -> float | None:
    if term == "prerelease_piracy_x_week":
        pir = float_or_none(row.get("prerelease_piracy"))
        week = float_or_none(row.get("week"))
        return None if pir is None or week is None else pir * week
    if term == "piracy_quality":
        return float_or_none(row.get("pirated_quality"))
    if term == "piracy_quality_x_week":
        quality = float_or_none(row.get("pirated_quality"))
        week = float_or_none(row.get("week"))
        return None if quality is None or week is None else quality * week
    if term == "log_piracy_week":
        pir_week = float_or_none(row.get("prerelease_piracy_week"))
        return math.log(pir_week) if pir_week is not None and pir_week > 0 else 0.0
    if term == "log_piracy_week_x_week":
        log_week = value_for_term(row, "log_piracy_week")
        week = float_or_none(row.get("week"))
        return None if log_week is None or week is None else log_week * week
    if term.endswith("_x_week"):
        base = term.removesuffix("_x_week")
        value = float_or_none(row.get(base))
        week = float_or_none(row.get("week"))
        return None if value is None or week is None else value * week
    if term == "log_revenue":
        return log_positive(row.get("revenue"))
    if term in {"budget", "screen"}:
        return log_positive(row.get(term))
    return float_or_none(row.get(term))


def fit_panel_model(
    equation: str,
    rows: list[dict[str, str]],
    terms: list[str],
    *,
    min_weeks: int,
    require_known_budget: bool = False,
) -> OLSResult | None:
    movie_weeks: dict[str, set[int]] = {}
    for row in rows:
        movie_id = row.get("movie_id")
        week = float_or_none(row.get("week"))
        if movie_id is not None and week is not None:
            movie_weeks.setdefault(movie_id, set()).add(int(week))
    eligible = {movie_id for movie_id, weeks in movie_weeks.items() if len(weeks) >= min_weeks}

    x: list[list[float]] = []
    y: list[float] = []
    for row in rows:
        if row.get("movie_id") not in eligible:
            continue
        if require_known_budget and (float_or_none(row.get("missing_budget")) or 0.0) != 0.0:
            continue
        y_value = value_for_term(row, "log_revenue")
        if y_value is None:
            continue
        values = [1.0]
        ok = True
        for term in terms:
            value = value_for_term(row, term)
            if value is None or not math.isfinite(value):
                ok = False
                break
            values.append(value)
        if ok:
            x.append(values)
            y.append(y_value)
    if len(y) <= len(terms) + 2:
        print(f"warning: not enough usable rows for {equation}", file=sys.stderr)
        return None

    beta, inv, sse = ols_beta(x, y)
    y_bar = mean(y)
    sst = sum((value - y_bar) ** 2 for value in y)
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    dof = max(1, len(y) - len(beta))
    sigma2 = sse / dof
    names = ["intercept"] + terms
    result_rows = []
    for idx, (name, coef) in enumerate(zip(names, beta)):
        se = math.sqrt(max(0.0, sigma2 * inv[idx][idx]))
        t_stat = coef / se if se else float("nan")
        p_normal = math.erfc(abs(t_stat) / math.sqrt(2.0)) if math.isfinite(t_stat) else float("nan")
        reported_parameter = transform_to_paper_parameter(name, coef)
        result_rows.append(
            {
                "equation": equation,
                "term": name,
                "coef_in_log_revenue_regression": f"{coef:.10g}",
                "paper_parameter_analogue": "" if reported_parameter is None else f"{reported_parameter:.10g}",
                "se": f"{se:.10g}",
                "t_stat": f"{t_stat:.10g}",
                "p_normal_approx": f"{p_normal:.10g}",
                "n": len(y),
                "r2": f"{r2:.10g}",
            }
        )
    return OLSResult(equation=equation, n=len(y), r2=r2, rows=result_rows)


def transform_to_paper_parameter(term: str, coef: float) -> float | None:
    if term == "week":
        return -coef
    if term.endswith("_x_week"):
        return -coef
    return coef


def write_panel_models(out_dir: Path, panel_rows: list[dict[str, str]]) -> None:
    rows = canonicalize_columns(panel_rows)
    if not required_columns(rows, ["movie_id", "week", "revenue", "prerelease_piracy"], "panel CSV"):
        return
    write_data_descriptives(out_dir, rows, "data_panel")

    available_terms = [term for term in MARKET_POTENTIAL_TERMS if term in rows[0]]
    missing_terms = [term for term in MARKET_POTENTIAL_TERMS if term not in rows[0]]
    if missing_terms:
        print(
            "warning: panel CSV lacks some paper controls, OLS analogues will use available controls only: "
            + ", ".join(missing_terms),
            file=sys.stderr,
        )

    models = []
    homogeneous_terms = available_terms + ["week", "prerelease_piracy", "prerelease_piracy_x_week"]
    result = fit_panel_model("homogeneous_decline_pooled_ols", rows, homogeneous_terms, min_weeks=6)
    if result:
        models.append(result)

    hetero_terms = (
        available_terms
        + ["week", "prerelease_piracy", "prerelease_piracy_x_week"]
        + [f"{term}_x_week" for term in DECAY_TERMS if term in rows[0]]
    )
    result = fit_panel_model("heterogeneous_decline_pooled_ols", rows, hetero_terms, min_weeks=6)
    if result:
        models.append(result)

    if "pirated_quality" in rows[0]:
        quality_terms = hetero_terms + ["piracy_quality", "piracy_quality_x_week"]
        result = fit_panel_model("piracy_quality_pooled_ols", rows, quality_terms, min_weeks=6)
        if result:
            models.append(result)

    if "prerelease_piracy_week" in rows[0]:
        timing_terms = hetero_terms + ["log_piracy_week", "log_piracy_week_x_week"]
        result = fit_panel_model("timing_pooled_ols", rows, timing_terms, min_weeks=6)
        if result:
            models.append(result)

    result = fit_panel_model("known_budget_pooled_ols", rows, hetero_terms, min_weeks=6, require_known_budget=True)
    if result:
        models.append(result)

    output = [row for result in models for row in result.rows]
    if output:
        write_csv(
            out_dir / "data_panel_pooled_ols_models.csv",
            output,
            [
                "equation",
                "term",
                "coef_in_log_revenue_regression",
                "paper_parameter_analogue",
                "se",
                "t_stat",
                "p_normal_approx",
                "n",
                "r2",
            ],
        )


def svg_escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def path_points(values: list[tuple[float, float]], sx: float, sy: float, left: float, top: float, ymax: float) -> str:
    commands = []
    for i, (x, y) in enumerate(values):
        px = left + x * sx
        py = top + (ymax - y) * sy
        commands.append(("M" if i == 0 else "L") + f"{px:.2f},{py:.2f}")
    return " ".join(commands)


def write_revenue_paths_svg(path: Path) -> None:
    width, height = 900, 520
    left, right, top, bottom = 72, 34, 58, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    weeks = list(range(1, 13))
    lambda_value = 0.7503
    rho = -0.4024
    tau = -0.0965
    baseline = [(week, math.exp(-lambda_value * week)) for week in weeks]
    pirated = [(week, math.exp(rho - (lambda_value + tau) * week)) for week in weeks]
    ymax = max(y for _, y in baseline + pirated) * 1.08
    sx = plot_w / 11
    sy = plot_h / ymax
    loss = cumulative_revenue_loss(lambda_value, rho, tau)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:13px}.title{font-size:20px;font-weight:700}.axis{stroke:#222;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.legend{font-size:14px}</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text class="title" x="{left}" y="32">Recreated Table 7 revenue path implication</text>',
    ]
    for i in range(6):
        y_value = ymax * i / 5
        y = top + (ymax - y_value) * sy
        elements.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}"/>')
        elements.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{y_value:.2f}</text>')
    elements.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    elements.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    for week in weeks:
        x = left + (week - 1) * sx
        elements.append(f'<text x="{x:.2f}" y="{height - 44}" text-anchor="middle">{week}</text>')
    elements.append(f'<path d="{path_points([(w - 1, y) for w, y in baseline], sx, sy, left, top, ymax)}" fill="none" stroke="#386cb0" stroke-width="3"/>')
    elements.append(f'<path d="{path_points([(w - 1, y) for w, y in pirated], sx, sy, left, top, ymax)}" fill="none" stroke="#d95f02" stroke-width="3"/>')
    elements.append(f'<text class="legend" x="{left + 20}" y="{top + 24}" fill="#386cb0">Post-release piracy baseline</text>')
    elements.append(f'<text class="legend" x="{left + 20}" y="{top + 48}" fill="#d95f02">Pre-release piracy</text>')
    elements.append(f'<text x="{left + 390}" y="{top + 28}">12-week cumulative loss: {loss * 100:.1f}%</text>')
    elements.append(f'<text x="{left + plot_w / 2}" y="{height - 15}" text-anchor="middle">Week in theatrical run</text>')
    elements.append('<text transform="translate(18 280) rotate(-90)" text-anchor="middle">Revenue index</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def write_robustness_svg(path: Path) -> None:
    width, height = 900, 500
    left, right, top, bottom = 78, 34, 58, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [row[0] for row in REPORTED_ALTERNATIVE_WEEKS]
    tau_values = [row[1] for row in REPORTED_ALTERNATIVE_WEEKS]
    rho_values = [row[3] for row in REPORTED_ALTERNATIVE_WEEKS]
    ymin = min(tau_values + rho_values) - 0.04
    ymax = 0.02

    def sx(week: float) -> float:
        return left + (week - min(xs)) / (max(xs) - min(xs)) * plot_w

    def sy(value: float) -> float:
        return top + (ymax - value) / (ymax - ymin) * plot_h

    def line(values: list[float]) -> str:
        return " ".join(
            ("M" if idx == 0 else "L") + f"{sx(week):.2f},{sy(value):.2f}"
            for idx, (week, value) in enumerate(zip(xs, values))
        )

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:13px}.title{font-size:20px;font-weight:700}.axis{stroke:#222;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.legend{font-size:14px}</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text class="title" x="{left}" y="32">Recreated Table 13: alternative week thresholds</text>',
    ]
    for i in range(6):
        value = ymin + (ymax - ymin) * i / 5
        y = sy(value)
        elements.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}"/>')
        elements.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{value:.2f}</text>')
    elements.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    elements.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    for week in xs:
        elements.append(f'<text x="{sx(week):.2f}" y="{height - 44}" text-anchor="middle">{week}</text>')
    elements.append(f'<path d="{line(tau_values)}" fill="none" stroke="#386cb0" stroke-width="3"/>')
    elements.append(f'<path d="{line(rho_values)}" fill="none" stroke="#d95f02" stroke-width="3"/>')
    for week, tau, _, rho, _ in REPORTED_ALTERNATIVE_WEEKS:
        elements.append(f'<circle cx="{sx(week):.2f}" cy="{sy(tau):.2f}" r="4" fill="#386cb0"/>')
        elements.append(f'<circle cx="{sx(week):.2f}" cy="{sy(rho):.2f}" r="4" fill="#d95f02"/>')
    elements.append(f'<text class="legend" x="{left + 20}" y="{top + 24}" fill="#386cb0">tau</text>')
    elements.append(f'<text class="legend" x="{left + 20}" y="{top + 48}" fill="#d95f02">rho</text>')
    elements.append(f'<text x="{left + plot_w / 2}" y="{height - 15}" text-anchor="middle">Minimum number of theatrical weeks retained</text>')
    elements.append('<text transform="translate(22 278) rotate(-90)" text-anchor="middle">Reported estimate</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-md", type=Path, default=DEFAULT_PAPER_MD, help="OCR markdown of the paper.")
    parser.add_argument("--panel-csv", type=Path, help="Optional movie-week panel CSV for local OLS analogues.")
    parser.add_argument("--out", type=Path, default=Path("results/papers/prerelease_piracy_boxoffice"), help="Output directory.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tables = write_extracted_tables(args.out, args.paper_md)
    write_reported_outputs(args.out)

    if args.panel_csv:
        write_panel_models(args.out, read_csv_rows(args.panel_csv))

    print(f"Wrote pre-release piracy reproduction artifacts to {args.out}", file=sys.stderr)
    print(f"Extracted {len(tables)} paper tables from {args.paper_md}", file=sys.stderr)
    if not args.panel_csv:
        print("No --panel-csv supplied; outputs are reported-result recreations plus reusable data-mode code.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
