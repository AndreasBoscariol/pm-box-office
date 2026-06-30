#!/usr/bin/env python3
"""Recreate core results from Ho Kim (2013), using reported or local data.

Paper:
    Is Search Volume a Good Market Predictor?
    Product Quality and the Predictive Performance of Search Volume

The paper's raw 174-movie panel is not bundled with this repository. By
default this script uses the OCR markdown of the paper to recreate the
reported result artifacts:

* every embedded paper table as a CSV;
* curated CSVs for the reported descriptive statistics, partial correlations,
  and focal coefficients;
* simple SVG figures for the theory diagram and the reported correlation and
  coefficient patterns.

If you later add the paper's movie-level and weekly panel data, pass
--movie-csv and optionally --weekly-csv. In that mode the script also computes
data-derived descriptive statistics, partial correlations, and log-linear OLS
analogues of the paper's opening-week and post-launch equations. The paper's
published estimates are 3SLS; the local-data OLS outputs are therefore
diagnostic unless you extend the script with a full simultaneous-equation
estimator.
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


DEFAULT_PAPER_MD = Path("docs/papers_ocr/ssrn-2209537.pdf_by_PaddleOCR-VL-1.6.md")

REPORTED_DESCRIPTIVES = [
    ("No. of user reviews per movie (N=174, up to 10 week after release)", "4835.9", "2203.5", "8272.2", "113.0", "71764.0"),
    ("Average movie rating by users (N=174, up to 10 week after release)", "7.0", "7.1", "1.2", "3.8", "9.5"),
    ("Standard deviation of movie ratings by users (N=174, up to 10 week after release)", "2.6", "2.7", "0.4", "1.4", "3.3"),
    ("No. of critics' reviews per movie (N=172)", "27.8", "29.5", "7.8", "1.0", "39.0"),
    ("Average rating of critic reviews (N=172, 100 point scale)", "57.2", "58.1", "15.3", "14.3", "92.7"),
    ("Standard deviation of ratings of critics reviews (N=171, 100 point scale)", "15.7", "15.6", "3.2", "5.0", "23.1"),
    ("No. of past movies by the focal directors (N=169)", "8.1", "5.0", "8.2", "0.0", "49.0"),
    ("Total U.S. gross box-office revenue of past movies by the focal directors (N=169)", "$774 M", "$337 M", "$1052 M", "$0", "$6518 M"),
    ("Average rating of past movies by the focal directors (N=161)", "6.7", "6.8", "0.6", "4.8", "8.7"),
    ("Standard deviation of ratings of past movies by the focal directors (N=161)", "2.0", "2.0", "0.2", "1.5", "3.4"),
    ("No. of past movies by the first-billing stars (N=156)", "15.3", "14.0", "11.2", "0.0", "51.0"),
    ("Total U.S. gross box-office revenue of past movies by the first-billing stars (N=156)", "$1389 M", "$1026 M", "$1396 M", "0.0", "$7524 M"),
    ("Average rating of past movies by the first-billing stars (N=149)", "6.6", "6.7", "0.5", "4.8", "7.6"),
    ("Standard deviation of ratings of past movies by the first-billing stars (N=149)", "2.0", "2.0", "0.3", "1.7", "3.4"),
    ("Advertising spending (N=174)", "$20 M", "$20 M", "$12 M", "$6.5 K", "$51 M"),
    ("Production budget (N=156)", "$55 M", "$38 M", "$53 M", "$11 K", "$250 M"),
]

REPORTED_PARTIAL_CORRELATIONS = [
    ("whole_balanced_sample", 160, "user_rating_t0", "critic_rating", 0.42, 0.00),
    ("whole_balanced_sample", 160, "cumulative_search_t_minus_1", "critic_rating", 0.06, 0.43),
    ("whole_balanced_sample", 160, "cumulative_search_t_minus_1", "user_rating_t0", 0.06, 0.43),
    ("whole_balanced_sample", 160, "opening_revenue", "critic_rating", 0.19, 0.02),
    ("whole_balanced_sample", 160, "opening_revenue", "user_rating_t0", 0.23, 0.00),
    ("whole_balanced_sample", 160, "opening_revenue", "cumulative_search_t_minus_1", 0.68, 0.00),
    ("upper_half_us_gross", 80, "user_rating_t0", "critic_rating", 0.38, 0.00),
    ("upper_half_us_gross", 80, "cumulative_search_t_minus_1", "critic_rating", 0.00, 0.97),
    ("upper_half_us_gross", 80, "cumulative_search_t_minus_1", "user_rating_t0", -0.02, 0.83),
    ("upper_half_us_gross", 80, "opening_revenue", "critic_rating", 0.07, 0.56),
    ("upper_half_us_gross", 80, "opening_revenue", "user_rating_t0", 0.05, 0.67),
    ("upper_half_us_gross", 80, "opening_revenue", "cumulative_search_t_minus_1", 0.71, 0.00),
    ("lower_half_us_gross", 80, "user_rating_t0", "critic_rating", 0.29, 0.00),
    ("lower_half_us_gross", 80, "cumulative_search_t_minus_1", "critic_rating", 0.03, 0.81),
    ("lower_half_us_gross", 80, "cumulative_search_t_minus_1", "user_rating_t0", 0.07, 0.54),
    ("lower_half_us_gross", 80, "opening_revenue", "critic_rating", -0.00, 0.99),
    ("lower_half_us_gross", 80, "opening_revenue", "user_rating_t0", 0.05, 0.68),
    ("lower_half_us_gross", 80, "opening_revenue", "cumulative_search_t_minus_1", 0.25, 0.03),
]

REPORTED_KEY_COEFFICIENTS = [
    ("table4_opening_search_3sls", "prelaunch_advertising", 0.32, 0.08, 0.00, "positive search driver"),
    ("table4_opening_search_3sls", "avg_rating_director", 5.44, 1.67, 0.00, "H1 support"),
    ("table4_opening_search_3sls", "sd_rating_director", 5.08, 1.39, 0.00, "H2 support"),
    ("table4_opening_search_3sls", "avg_rating_star", 3.48, 2.40, 0.15, "not significant"),
    ("table4_opening_search_3sls", "sd_rating_star", 1.32, 1.77, 0.46, "not significant"),
    ("table7_new_keyword_search", "avg_rating_director", 6.13, 1.65, 0.00, "robust search equation"),
    ("table7_new_keyword_search", "sd_rating_director", 6.48, 1.42, 0.00, "robust search equation"),
    ("table7_new_keyword_revenue", "prelaunch_search", -0.02, 0.09, 0.83, "main effect weak"),
    ("table7_new_keyword_revenue", "prelaunch_search_x_critic_rating", 0.0022, 0.00, 0.00, "quality moderates conversion"),
    ("table7_new_keyword_postlaunch_search", "avg_user_rating_previous_week", 1.16, 0.32, 0.00, "H6 support"),
    ("table7_new_keyword_postlaunch_search", "sd_user_rating_previous_week", 1.17, 0.39, 0.00, "H7 support"),
    ("table7_new_keyword_postlaunch_revenue", "search_x_avg_user_rating_previous_week", 0.04, 0.00, 0.00, "H9 support"),
    ("table7_new_keyword_postlaunch_revenue", "sd_user_rating_previous_week", 0.70, 0.21, 0.00, "quality uncertainty has direct effect under new keyword rule"),
]

REPORTED_KEYWORD_CORRELATIONS = [
    ("user_rating_t0", "critic_rating", 0.42, 0.00),
    ("prelaunch_search_original_rule", "critic_rating", 0.06, 0.50),
    ("prelaunch_search_original_rule", "user_rating_t0", 0.06, 0.44),
    ("prelaunch_search_new_rule", "critic_rating", 0.10, 0.21),
    ("prelaunch_search_new_rule", "user_rating_t0", 0.01, 0.87),
    ("prelaunch_search_new_rule", "prelaunch_search_original_rule", 0.55, 0.00),
    ("opening_revenue", "critic_rating", 0.19, 0.03),
    ("opening_revenue", "user_rating_t0", 0.21, 0.01),
    ("opening_revenue", "prelaunch_search_original_rule", 0.68, 0.00),
    ("opening_revenue", "prelaunch_search_new_rule", 0.37, 0.00),
]


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
            text = clean_text("".join(self.current_cell))
            self.current_row.append(text)
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


def clean_text(value: str) -> str:
    value = html.unescape(re.sub(r"<[^>]+>", " ", value))
    value = value.replace("\xa0", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_numeric_text(value: object) -> str:
    text = str(value).strip()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("M", "e6")
    text = text.replace("K", "e3")
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


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_bar = mean(xs)
    y_bar = mean(ys)
    num = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
    x_ss = sum((x - x_bar) ** 2 for x in xs)
    y_ss = sum((y - y_bar) ** 2 for y in ys)
    if x_ss <= 0.0 or y_ss <= 0.0:
        return None
    return num / math.sqrt(x_ss * y_ss)


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


def residuals_from_controls(y: list[float], controls: list[list[float]]) -> list[float]:
    if not controls:
        center = mean(y)
        return [value - center for value in y]
    x = [[1.0] + row for row in controls]
    beta, _, _ = ols_beta(x, y)
    return [actual - sum(coef * value for coef, value in zip(beta, row)) for actual, row in zip(y, x)]


def partial_correlation(
    rows: list[dict[str, str]],
    x_col: str,
    y_col: str,
    control_cols: list[str],
) -> tuple[int, float | None]:
    xs: list[float] = []
    ys: list[float] = []
    controls: list[list[float]] = []
    for row in rows:
        x = float_or_none(row.get(x_col))
        y = float_or_none(row.get(y_col))
        control_values = [float_or_none(row.get(col)) for col in control_cols]
        if x is None or y is None or any(value is None for value in control_values):
            continue
        xs.append(x)
        ys.append(y)
        controls.append([value for value in control_values if value is not None])
    if len(xs) < len(control_cols) + 3:
        return len(xs), None
    return len(xs), pearson(residuals_from_controls(xs, controls), residuals_from_controls(ys, controls))


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


def ols(
    equation: str,
    rows: list[dict[str, object]],
    y_col: str,
    terms: list[tuple[str, str]],
) -> OLSResult | None:
    x: list[list[float]] = []
    y: list[float] = []
    for row in rows:
        y_value = log_positive(row.get(y_col))
        if y_value is None:
            continue
        values = [1.0]
        ok = True
        for term_name, expression in terms:
            value = evaluate_term(row, expression)
            if value is None or not math.isfinite(value):
                ok = False
                break
            values.append(value)
        if ok:
            x.append(values)
            y.append(y_value)
    if len(y) <= len(terms) + 2:
        return None
    beta, inv, sse = ols_beta(x, y)
    y_bar = mean(y)
    sst = sum((value - y_bar) ** 2 for value in y)
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    dof = max(1, len(y) - len(beta))
    sigma2 = sse / dof
    names = ["Intercept"] + [term_name for term_name, _ in terms]
    result_rows = []
    for idx, (name, coef) in enumerate(zip(names, beta)):
        se = math.sqrt(max(0.0, sigma2 * inv[idx][idx]))
        t_stat = coef / se if se else float("nan")
        p_normal = math.erfc(abs(t_stat) / math.sqrt(2.0)) if math.isfinite(t_stat) else float("nan")
        result_rows.append(
            {
                "equation": equation,
                "term": name,
                "coef": f"{coef:.10g}",
                "se": f"{se:.10g}",
                "t_stat": f"{t_stat:.10g}",
                "p_normal_approx": f"{p_normal:.10g}",
                "n": len(y),
                "r2": f"{r2:.10g}",
            }
        )
    return OLSResult(equation=equation, n=len(y), r2=r2, rows=result_rows)


def evaluate_term(row: dict[str, object], expression: str) -> float | None:
    parts = expression.split("*")
    value = 1.0
    for part in parts:
        part = part.strip()
        if part.startswith("log:"):
            term = log_positive(row.get(part[4:]))
        else:
            term = float_or_none(row.get(part))
        if term is None:
            return None
        value *= term
    return value


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
        prefix_lines = markdown[: match.start()].splitlines()[-25:]
        candidates = []
        for line in prefix_lines:
            text = clean_text(line)
            if re.search(r"\bTable\s+\d+\b", text, flags=re.IGNORECASE) or re.match(r"\([a-z]\)\s+", text):
                candidates.append(text)
        caption = " / ".join(candidates[-2:]) if candidates else f"Table {index}"
        extracted.append(ExtractedTable(table_id=index, caption=caption, rows=rows))
    return extracted


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


def write_reported_outputs(out_dir: Path) -> None:
    descriptive_rows = [
        {"variable": row[0], "mean": row[1], "median": row[2], "std_dev": row[3], "min": row[4], "max": row[5]}
        for row in REPORTED_DESCRIPTIVES
    ]
    write_csv(out_dir / "reported_table2_descriptive_stats.csv", descriptive_rows, ["variable", "mean", "median", "std_dev", "min", "max"])

    partial_rows = [
        {"sample": sample, "n": n, "row_variable": row_var, "column_variable": col_var, "partial_corr": corr, "p_value": p_value}
        for sample, n, row_var, col_var, corr, p_value in REPORTED_PARTIAL_CORRELATIONS
    ]
    write_csv(out_dir / "reported_table3_partial_correlations.csv", partial_rows, ["sample", "n", "row_variable", "column_variable", "partial_corr", "p_value"])

    coefficient_rows = [
        {"source": source, "term": term, "coef": coef, "se": se, "p_value": p_value, "note": note}
        for source, term, coef, se, p_value, note in REPORTED_KEY_COEFFICIENTS
    ]
    write_csv(out_dir / "reported_key_coefficients.csv", coefficient_rows, ["source", "term", "coef", "se", "p_value", "note"])

    keyword_rows = [
        {"row_variable": row_var, "column_variable": col_var, "corr": corr, "p_value": p_value}
        for row_var, col_var, corr, p_value in REPORTED_KEYWORD_CORRELATIONS
    ]
    write_csv(out_dir / "reported_table6_keyword_rule_correlations.csv", keyword_rows, ["row_variable", "column_variable", "corr", "p_value"])

    write_theory_svg(out_dir / "figure1_theory_recreated.svg")
    write_table3_svg(out_dir / "figure_table3_partial_correlations.svg")
    write_coefficients_svg(out_dir / "figure_key_coefficients.svg")


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
        print(f"warning: {prefix} CSV is empty; skipping descriptive stats", file=sys.stderr)
        return
    numeric_cols = []
    for col in rows[0]:
        values = [float_or_none(row.get(col)) for row in rows]
        clean = [value for value in values if value is not None]
        if clean:
            numeric_cols.append((col, clean))
    output = []
    for col, values in numeric_cols:
        sorted_values = sorted(values)
        mid = len(sorted_values) // 2
        median = sorted_values[mid] if len(sorted_values) % 2 else (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
        output.append(
            {
                "variable": col,
                "n": len(values),
                "mean": f"{mean(values):.10g}",
                "median": f"{median:.10g}",
                "std_dev": f"{sample_sd(values):.10g}",
                "min": f"{min(values):.10g}",
                "max": f"{max(values):.10g}",
            }
        )
    write_csv(out_dir / f"{prefix}_descriptive_stats.csv", output, ["variable", "n", "mean", "median", "std_dev", "min", "max"])


def write_data_partial_correlations(out_dir: Path, rows: list[dict[str, str]]) -> None:
    required = [
        "opening_revenue",
        "critic_rating",
        "user_rating_t0",
        "cumulative_search_t_minus_1",
        "opening_screens",
        "prelaunch_ad",
    ]
    if not required_columns(rows, required, "movie CSV"):
        return
    split_col = "total_us_gross" if "total_us_gross" in rows[0] else "opening_revenue"
    numeric_split_rows = [(float_or_none(row.get(split_col)), row) for row in rows]
    numeric_split_rows = [(value, row) for value, row in numeric_split_rows if value is not None]
    if not numeric_split_rows:
        print(f"warning: no numeric values found in {split_col}; skipping split-sample correlations", file=sys.stderr)
        return
    median_value = sorted(value for value, _ in numeric_split_rows)[len(numeric_split_rows) // 2]
    samples = [
        ("whole_sample", rows),
        ("upper_half_gross", [row for value, row in numeric_split_rows if value >= median_value]),
        ("lower_half_gross", [row for value, row in numeric_split_rows if value < median_value]),
    ]
    pairs = [
        ("user_rating_t0", "critic_rating"),
        ("cumulative_search_t_minus_1", "critic_rating"),
        ("cumulative_search_t_minus_1", "user_rating_t0"),
        ("opening_revenue", "critic_rating"),
        ("opening_revenue", "user_rating_t0"),
        ("opening_revenue", "cumulative_search_t_minus_1"),
    ]
    output = []
    controls = ["opening_screens", "prelaunch_ad"]
    for sample_name, sample_rows in samples:
        for x_col, y_col in pairs:
            n, corr = partial_correlation(sample_rows, x_col, y_col, controls)
            output.append(
                {
                    "sample": sample_name,
                    "n": n,
                    "row_variable": x_col,
                    "column_variable": y_col,
                    "controls": ",".join(controls),
                    "partial_corr": "" if corr is None else f"{corr:.10g}",
                }
            )
    write_csv(out_dir / "data_partial_correlations.csv", output, ["sample", "n", "row_variable", "column_variable", "controls", "partial_corr"])


def write_data_opening_ols(out_dir: Path, movie_rows: list[dict[str, str]]) -> None:
    results: list[OLSResult] = []
    search_terms = [
        ("log_prelaunch_ad", "log:prelaunch_ad"),
        ("log_avg_rating_director", "log:avg_rating_director"),
        ("log_sd_rating_director", "log:sd_rating_director"),
        ("log_avg_rating_star", "log:avg_rating_star"),
        ("log_sd_rating_star", "log:sd_rating_star"),
    ]
    result = ols("opening_search_ols", movie_rows, "cumulative_search_t_minus_1", search_terms)
    if result:
        results.append(result)
    revenue_terms = [
        ("log_opening_ad", "log:opening_ad"),
        ("log_opening_screens", "log:opening_screens"),
        ("log_cumulative_search", "log:cumulative_search_t_minus_1"),
        ("log_cumulative_search_x_critic_rating", "log:cumulative_search_t_minus_1*critic_rating"),
        ("log_avg_rating_director", "log:avg_rating_director"),
        ("log_sd_rating_director", "log:sd_rating_director"),
        ("log_avg_rating_star", "log:avg_rating_star"),
        ("log_sd_rating_star", "log:sd_rating_star"),
    ]
    result = ols("opening_revenue_ols", movie_rows, "opening_revenue", revenue_terms)
    if result:
        results.append(result)
    rows = [row for result in results for row in result.rows]
    if rows:
        write_csv(out_dir / "data_opening_week_ols.csv", rows, ["equation", "term", "coef", "se", "t_stat", "p_normal_approx", "n", "r2"])


def build_weekly_lag_rows(movie_rows: list[dict[str, str]], weekly_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    movie_by_id = {row.get("movie_id"): row for row in movie_rows}
    by_movie: dict[str, list[dict[str, str]]] = {}
    for row in weekly_rows:
        movie_id = row.get("movie_id")
        week = float_or_none(row.get("week"))
        if movie_id is None or week is None:
            continue
        by_movie.setdefault(movie_id, []).append(row)
    lagged = []
    for movie_id, rows in by_movie.items():
        sorted_rows = sorted(rows, key=lambda row: float_or_none(row.get("week")) or 0.0)
        previous: dict[str, str] | None = None
        for row in sorted_rows:
            week = float_or_none(row.get("week"))
            if previous is not None and week is not None and week >= 1:
                merged: dict[str, object] = dict(row)
                merged["prev_revenue"] = previous.get("revenue")
                merged["prev_search"] = previous.get("search")
                merged["prev_screens"] = previous.get("screens")
                merged["prev_avg_user_rating"] = previous.get("avg_user_rating")
                merged["prev_sd_user_rating"] = previous.get("sd_user_rating")
                movie = movie_by_id.get(movie_id, {})
                merged["critic_rating"] = movie.get("critic_rating")
                lagged.append(merged)
            previous = row
    return lagged


def write_data_postlaunch_ols(out_dir: Path, movie_rows: list[dict[str, str]], weekly_rows: list[dict[str, str]]) -> None:
    needed = ["movie_id", "week", "revenue", "search", "advertising", "screens", "avg_user_rating", "sd_user_rating"]
    if not required_columns(weekly_rows, needed, "weekly CSV"):
        return
    lagged = build_weekly_lag_rows(movie_rows, weekly_rows)
    results: list[OLSResult] = []
    search_terms = [
        ("log_advertising", "log:advertising"),
        ("log_prev_revenue", "log:prev_revenue"),
        ("log_prev_avg_user_rating", "log:prev_avg_user_rating"),
        ("log_prev_sd_user_rating", "log:prev_sd_user_rating"),
    ]
    result = ols("postlaunch_search_ols", lagged, "search", search_terms)
    if result:
        results.append(result)
    revenue_terms = [
        ("log_advertising", "log:advertising"),
        ("log_screens", "log:screens"),
        ("log_search", "log:search"),
        ("log_search_x_prev_avg_user_rating", "log:search*prev_avg_user_rating"),
        ("critic_rating", "critic_rating"),
        ("log_prev_sd_user_rating", "log:prev_sd_user_rating"),
    ]
    result = ols("postlaunch_revenue_ols", lagged, "revenue", revenue_terms)
    if result:
        results.append(result)
    rows = [row for result in results for row in result.rows]
    if rows:
        write_csv(out_dir / "data_postlaunch_ols.csv", rows, ["equation", "term", "coef", "se", "t_stat", "p_normal_approx", "n", "r2"])


def svg_escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_theory_svg(path: Path) -> None:
    width, height = 980, 420
    boxes = [
        ("Perceived quality", 70, 70, 210, 64, "#e8f1fb"),
        ("Quality uncertainty", 70, 210, 210, 64, "#fff3d8"),
        ("Search volume", 390, 140, 210, 64, "#e9f7ef"),
        ("Updated quality", 680, 70, 210, 64, "#f5eafa"),
        ("Revenue", 680, 210, 210, 64, "#fbecea"),
    ]
    arrows = [
        (280, 102, 390, 172, "+"),
        (280, 242, 390, 172, "+"),
        (600, 172, 680, 242, "+"),
        (785, 134, 785, 210, "moderates"),
    ]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:16px}.title{font-size:22px;font-weight:700}.small{font-size:13px}.box{stroke:#333;stroke-width:1.4;rx:6}</style>",
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#333"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text class="title" x="50" y="34">Figure 1 recreated: quality, uncertainty, search, and sales</text>',
    ]
    for label, x, y, w, h, fill in boxes:
        elements.append(f'<rect class="box" x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}"/>')
        elements.append(f'<text x="{x + w / 2}" y="{y + h / 2 + 6}" text-anchor="middle">{svg_escape(label)}</text>')
    for x1, y1, x2, y2, label in arrows:
        elements.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#333" stroke-width="1.7" marker-end="url(#arrow)"/>')
        elements.append(f'<text class="small" x="{(x1 + x2) / 2 + 8}" y="{(y1 + y2) / 2 - 8}">{svg_escape(label)}</text>')
    elements.append('<text class="small" x="70" y="335">Paper implication: quality uncertainty raises search, but weak quality information can reduce search-to-sales conversion.</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def write_table3_svg(path: Path) -> None:
    rows = [
        ("Whole", 0.68, 0.19, 0.23),
        ("Upper half", 0.71, 0.07, 0.05),
        ("Lower half", 0.25, -0.00, 0.05),
    ]
    write_grouped_bar_svg(
        path,
        "Reported partial correlations with opening revenue",
        rows,
        ["Search", "Critic rating", "User rating"],
        -0.1,
        0.8,
    )


def write_coefficients_svg(path: Path) -> None:
    rows = [
        ("Avg rating director", 5.44),
        ("SD rating director", 5.08),
        ("Avg rating star", 3.48),
        ("SD rating star", 1.32),
        ("Search x critic", 2.2),
        ("Prev user rating", 1.16),
        ("Prev rating SD", 1.17),
        ("Search x prev rating", 4.0),
    ]
    # Interaction terms are scaled for visibility: 0.0022*1000 and 0.04*100.
    write_horizontal_bar_svg(path, "Focal reported coefficients, scaled where needed", rows)


def write_grouped_bar_svg(
    path: Path,
    title: str,
    rows: list[tuple[str, float, float, float]],
    labels: list[str],
    ymin: float,
    ymax: float,
) -> None:
    width, height = 880, 500
    left, right, top, bottom = 80, 30, 55, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = ["#386cb0", "#fdb462", "#7fc97f"]

    def sy(value: float) -> float:
        return top + (ymax - value) / (ymax - ymin) * plot_h

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:13px}.title{font-size:20px;font-weight:700}.axis{stroke:#222;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text class="title" x="{left}" y="32">{svg_escape(title)}</text>',
    ]
    for i in range(6):
        value = ymin + (ymax - ymin) * i / 5
        y = sy(value)
        elements.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}"/>')
        elements.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{value:.1f}</text>')
    zero_y = sy(0.0)
    elements.append(f'<line class="axis" x1="{left}" y1="{zero_y:.2f}" x2="{left + plot_w}" y2="{zero_y:.2f}"/>')
    group_w = plot_w / len(rows)
    bar_w = group_w / 5
    for g, row in enumerate(rows):
        sample, *values = row
        center = left + group_w * (g + 0.5)
        elements.append(f'<text x="{center:.2f}" y="{height - 38}" text-anchor="middle">{svg_escape(sample)}</text>')
        for i, value in enumerate(values):
            x = center + (i - 1.5) * bar_w
            y = sy(max(value, 0.0))
            h = abs(sy(value) - zero_y)
            if value < 0:
                y = zero_y
            elements.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w - 3:.2f}" height="{h:.2f}" fill="{colors[i]}"/>')
            elements.append(f'<text x="{x + bar_w / 2:.2f}" y="{y - 5:.2f}" text-anchor="middle">{value:.2f}</text>')
    for i, label in enumerate(labels):
        lx = left + i * 160
        elements.append(f'<rect x="{lx}" y="{height - 20}" width="14" height="14" fill="{colors[i]}"/>')
        elements.append(f'<text x="{lx + 20}" y="{height - 8}">{svg_escape(label)}</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def write_horizontal_bar_svg(path: Path, title: str, rows: list[tuple[str, float]]) -> None:
    width, height = 900, 500
    left, right, top, bottom = 245, 40, 55, 35
    plot_w = width - left - right
    bar_h = 28
    gap = 17
    max_value = max(value for _, value in rows)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:13px}.title{font-size:20px;font-weight:700}.small{font-size:12px}</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text class="title" x="40" y="32">{svg_escape(title)}</text>',
    ]
    for i, (label, value) in enumerate(rows):
        y = top + i * (bar_h + gap)
        w = value / max_value * plot_w
        elements.append(f'<text x="{left - 12}" y="{y + 19}" text-anchor="end">{svg_escape(label)}</text>')
        elements.append(f'<rect x="{left}" y="{y}" width="{w:.2f}" height="{bar_h}" fill="#5b8def"/>')
        elements.append(f'<text x="{left + w + 8:.2f}" y="{y + 19}">{value:.3g}</text>')
    elements.append(f'<text class="small" x="{left}" y="{height - 10}">Scaled terms: Search x critic = coef*1000; Search x previous rating = coef*100.</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-md", type=Path, default=DEFAULT_PAPER_MD, help="OCR markdown of the paper.")
    parser.add_argument("--movie-csv", type=Path, help="Optional movie-level panel CSV for local calculations.")
    parser.add_argument("--weekly-csv", type=Path, help="Optional weekly panel CSV for post-launch local calculations.")
    parser.add_argument("--out", type=Path, default=Path("results/papers/search_volume_quality"), help="Output directory.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tables = write_extracted_tables(args.out, args.paper_md)
    write_reported_outputs(args.out)

    if args.movie_csv:
        movie_rows = read_csv_rows(args.movie_csv)
        write_data_descriptives(args.out, movie_rows, "data_movie_level")
        write_data_partial_correlations(args.out, movie_rows)
        write_data_opening_ols(args.out, movie_rows)
        if args.weekly_csv:
            weekly_rows = read_csv_rows(args.weekly_csv)
            write_data_descriptives(args.out, weekly_rows, "data_weekly")
            write_data_postlaunch_ols(args.out, movie_rows, weekly_rows)

    print(f"Wrote Ho Kim reproduction artifacts to {args.out}", file=sys.stderr)
    print(f"Extracted {len(tables)} paper tables from {args.paper_md}", file=sys.stderr)
    if not args.movie_csv:
        print("No --movie-csv supplied; outputs are reported-result recreations plus reusable data-mode code.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
