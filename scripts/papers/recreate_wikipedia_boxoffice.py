#!/usr/bin/env python3
"""Recreate the core results from Mestyan, Yasseri, and Kertesz (2013).

The paper predicts opening-weekend movie revenue from Wikipedia activity:

    Early Prediction of Movie Box Office Success Based on Wikipedia Activity
    Big Data, PLoS ONE 8(8): e71226.

This script downloads/parses Dataset S1 and recreates the main quantitative
outputs without third-party Python packages:

* Figure 1-style summary histograms for the 312-movie sample.
* Figure 2-style temporal Pearson correlations.
* Figure 3-style 10-fold cross-validated linear-regression R^2 curves.
* Figure 4-style Asur-Huberman 24-movie comparison curve.
* Figure 5-style actual vs predicted revenue scatter at t = -30 days.

Outputs are written as CSV files and simple SVG plots.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import random
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PLOS_DATASET_URL = (
    "https://journals.plos.org/plosone/article/file?"
    "type=supplementary&id=10.1371/journal.pone.0071226.s002"
)
LEGACY_DATASET_URL = "http://www.phy.bme.hu/SupplementaryDataS1.zip"
ZIP_ROOT = "wikipredict_data_pack"

PREDICTOR_COLUMNS = ("V", "U", "R", "E")
PREDICTOR_LABELS = {
    "V": "Views",
    "U": "Users",
    "R": "Rigor",
    "E": "Edits",
    "T": "Theaters",
}

MODEL_SETS = [
    ("T", ("T",)),
    ("V", ("V",)),
    ("V+T", ("V", "T")),
    ("U+T", ("U", "T")),
    ("V+U+R+E+T", ("V", "U", "R", "E", "T")),
]

COLORS = {
    "V": "#1f77b4",
    "U": "#2ca02c",
    "R": "#9467bd",
    "E": "#ff7f0e",
    "T": "#444444",
    "V+T": "#d62728",
    "U+T": "#17becf",
    "V+U+R+E+T": "#111111",
    "Wikipedia model": "#1f77b4",
    "Twitter reference": "#d62728",
}


@dataclass
class Movie:
    movie_id: str
    title: str
    wiki_title: str
    revenue: float
    theaters: float
    release_date: str
    inception_days: float | None = None


@dataclass
class Sample:
    name: str
    movies: list[Movie]
    predictors: dict[str, dict[int, dict[str, float]]]

    @property
    def days(self) -> list[int]:
        day_sets = [set(self.predictors[m.movie_id]) for m in self.movies]
        return sorted(set.intersection(*day_sets))


def ensure_dataset(zip_path: Path, force_download: bool = False) -> Path:
    if zip_path.exists() and not force_download:
        return zip_path

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in (PLOS_DATASET_URL, LEGACY_DATASET_URL):
        try:
            print(f"Downloading Dataset S1 from {url}", file=sys.stderr)
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                if not any(name.startswith(ZIP_ROOT + "/") for name in zf.namelist()):
                    raise ValueError("download did not contain the expected data pack")
            zip_path.write_bytes(data)
            return zip_path
        except Exception as exc:  # noqa: BLE001 - report both mirror failures.
            errors.append(f"{url}: {exc}")

    joined = "\n  ".join(errors)
    raise RuntimeError(
        "Could not download Dataset S1. Re-run with --zip /path/to/SupplementaryDataS1.zip.\n"
        f"  {joined}"
    )


def open_zip_text(zf: zipfile.ZipFile, member: str) -> io.TextIOWrapper:
    return io.TextIOWrapper(zf.open(member), encoding="utf-8", newline="")


def read_sample(zf: zipfile.ZipFile, sample_dir: str, index_name: str) -> Sample:
    index_member = f"{ZIP_ROOT}/{sample_dir}/{index_name}"
    with open_zip_text(zf, index_member) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        movies = []
        for row in reader:
            inception = row.get("Inception_of_article_(movie_time_days)")
            movies.append(
                Movie(
                    movie_id=row["ID"],
                    title=row["Title"],
                    wiki_title=row["WP_page_title"],
                    revenue=float(row["First_weekend_revenue_USD"]),
                    theaters=float(row["Number_of theaters"]),
                    release_date=row["Date_of_release"],
                    inception_days=float(inception) if inception not in (None, "") else None,
                )
            )

    predictors: dict[str, dict[int, dict[str, float]]] = {}
    for movie in movies:
        member = f"{ZIP_ROOT}/{sample_dir}/wikipedia_predictors/{movie.movie_id}"
        with open_zip_text(zf, member) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            by_day: dict[int, dict[str, float]] = {}
            for row in reader:
                by_day[int(row["Day_(movie_time)"])] = {
                    "V": float(row["Views"]),
                    "U": float(row["Users"]),
                    "R": float(row["Rigor"]),
                    "E": float(row["Edits"]),
                }
            predictors[movie.movie_id] = by_day

    return Sample(sample_dir, movies, predictors)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    x_bar = mean(xs)
    y_bar = mean(ys)
    num = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
    x_ss = sum((x - x_bar) ** 2 for x in xs)
    y_ss = sum((y - y_bar) ** 2 for y in ys)
    if x_ss <= 0.0 or y_ss <= 0.0:
        return None
    return num / math.sqrt(x_ss * y_ss)


def r2_score(y_true: list[float], y_pred: list[float]) -> float | None:
    y_bar = mean(y_true)
    ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y_true, y_pred))
    ss_tot = sum((actual - y_bar) ** 2 for actual in y_true)
    if ss_tot <= 0.0:
        return None
    return 1.0 - ss_res / ss_tot


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


def standardize(
    x_train: list[list[float]], x_apply: list[list[float]]
) -> tuple[list[list[float]], list[list[float]]]:
    if not x_train:
        raise ValueError("cannot fit a model with no training rows")
    width = len(x_train[0])
    centers = [mean(row[j] for row in x_train) for j in range(width)]
    scales = []
    for j, center in enumerate(centers):
        variance = mean((row[j] - center) ** 2 for row in x_train)
        scales.append(math.sqrt(variance) or 1.0)

    def transform(rows: list[list[float]]) -> list[list[float]]:
        return [
            [1.0] + [(row[j] - centers[j]) / scales[j] for j in range(width)]
            for row in rows
        ]

    return transform(x_train), transform(x_apply)


def linear_regression_predict(
    x_train: list[list[float]], y_train: list[float], x_apply: list[list[float]]
) -> list[float]:
    z_train, z_apply = standardize(x_train, x_apply)
    width = len(z_train[0])

    gram = [
        [sum(row[i] * row[j] for row in z_train) for j in range(width)]
        for i in range(width)
    ]
    target = [sum(row[i] * y for row, y in zip(z_train, y_train)) for i in range(width)]

    try:
        beta = solve_linear_system(gram, target)
    except ValueError:
        # The paper uses ordinary least squares. This tiny fallback only handles
        # numerically singular folds without materially changing well-posed fits.
        for i in range(width):
            gram[i][i] += 1e-9
        beta = solve_linear_system(gram, target)

    return [sum(coef * value for coef, value in zip(beta, row)) for row in z_apply]


def feature_matrix(sample: Sample, day: int, features: tuple[str, ...]) -> list[list[float]]:
    rows = []
    for movie in sample.movies:
        day_values = sample.predictors[movie.movie_id][day]
        row = []
        for feature in features:
            if feature == "T":
                row.append(movie.theaters)
            else:
                row.append(day_values[feature])
        rows.append(row)
    return rows


def cross_validated_r2(
    sample: Sample, day: int, features: tuple[str, ...], folds: int, seed: int
) -> tuple[float | None, float | None]:
    x = feature_matrix(sample, day, features)
    y = [movie.revenue for movie in sample.movies]
    indices = list(range(len(sample.movies)))
    random.Random(seed).shuffle(indices)
    test_folds = [indices[i::folds] for i in range(folds)]

    predictions: list[float | None] = [None] * len(sample.movies)
    fold_scores: list[float] = []
    for test_idx in test_folds:
        test_set = set(test_idx)
        train_idx = [idx for idx in indices if idx not in test_set]
        y_hat = linear_regression_predict(
            [x[idx] for idx in train_idx],
            [y[idx] for idx in train_idx],
            [x[idx] for idx in test_idx],
        )
        for idx, pred in zip(test_idx, y_hat):
            predictions[idx] = pred
        fold_r2 = r2_score([y[idx] for idx in test_idx], y_hat)
        if fold_r2 is not None:
            fold_scores.append(fold_r2)

    pooled = r2_score(y, [p for p in predictions if p is not None])
    mean_fold = mean(fold_scores) if fold_scores else None
    return pooled, mean_fold


def in_sample_r2(sample: Sample, day: int, features: tuple[str, ...]) -> float | None:
    x = feature_matrix(sample, day, features)
    y = [movie.revenue for movie in sample.movies]
    y_hat = linear_regression_predict(x, y, x)
    return r2_score(y, y_hat)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_number(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.10g}"


def svg_escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_line_svg(
    path: Path,
    title: str,
    series: dict[str, list[tuple[float, float | None]]],
    x_label: str,
    y_label: str,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    reference_lines: list[tuple[float, str, str]] | None = None,
) -> None:
    width, height = 960, 560
    left, right, top, bottom = 80, 190, 50, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    points = [(x, y) for values in series.values() for x, y in values if y is not None]
    if not points:
        raise ValueError(f"no plottable points for {path}")
    xmin, xmax = x_range or (min(x for x, _ in points), max(x for x, _ in points))
    ymin, ymax = y_range or (min(y for _, y in points), max(y for _, y in points))
    if ymin == ymax:
        ymin -= 1.0
        ymax += 1.0

    def sx(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * plot_w

    def sy(y: float) -> float:
        return top + (ymax - y) / (ymax - ymin) * plot_h

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:13px}.title{font-size:20px;font-weight:700}.axis{stroke:#222;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.legend{font-size:13px}</style>",
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text class="title" x="{left}" y="30">{svg_escape(title)}</text>',
    ]

    for i in range(6):
        frac = i / 5
        x = xmin + (xmax - xmin) * frac
        px = sx(x)
        elements.append(f'<line class="grid" x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{top + plot_h}"/>')
        elements.append(f'<text x="{px:.2f}" y="{height - 35}" text-anchor="middle">{x:.0f}</text>')
    for i in range(6):
        frac = i / 5
        y = ymin + (ymax - ymin) * frac
        py = sy(y)
        elements.append(f'<line class="grid" x1="{left}" y1="{py:.2f}" x2="{left + plot_w}" y2="{py:.2f}"/>')
        elements.append(f'<text x="{left - 10}" y="{py + 4:.2f}" text-anchor="end">{y:.2f}</text>')

    elements.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    elements.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    elements.append(f'<text x="{left + plot_w / 2:.2f}" y="{height - 10}" text-anchor="middle">{svg_escape(x_label)}</text>')
    elements.append(
        f'<text transform="translate(18 {top + plot_h / 2:.2f}) rotate(-90)" text-anchor="middle">{svg_escape(y_label)}</text>'
    )

    if reference_lines:
        for y_value, label, color in reference_lines:
            py = sy(y_value)
            elements.append(
                f'<line x1="{left}" y1="{py:.2f}" x2="{left + plot_w}" y2="{py:.2f}" '
                f'stroke="{color}" stroke-width="1.5" stroke-dasharray="6 5"/>'
            )
            elements.append(f'<text x="{left + plot_w + 8}" y="{py + 4:.2f}" fill="{color}">{svg_escape(label)}</text>')

    legend_x = left + plot_w + 25
    legend_y = top + 10
    for idx, (name, values) in enumerate(series.items()):
        color = COLORS.get(name, "#333333")
        clean = [(sx(x), sy(y)) for x, y in values if y is not None]
        if len(clean) >= 2:
            point_string = " ".join(f"{x:.2f},{y:.2f}" for x, y in clean)
            elements.append(
                f'<polyline points="{point_string}" fill="none" stroke="{color}" '
                f'stroke-width="2.3" stroke-linejoin="round" stroke-linecap="round"/>'
            )
        ly = legend_y + idx * 24
        elements.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 24}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        elements.append(f'<text class="legend" x="{legend_x + 32}" y="{ly + 4}">{svg_escape(name)}</text>')

    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def write_scatter_svg(
    path: Path,
    title: str,
    points: list[tuple[float, float, str]],
    x_label: str,
    y_label: str,
) -> None:
    width, height = 760, 660
    left, right, top, bottom = 80, 45, 50, 75
    plot_w = width - left - right
    plot_h = height - top - bottom
    positive = [(x, y, group) for x, y, group in points if x > 0 and y > 0]
    xs = [math.log10(x) for x, _, _ in positive]
    ys = [math.log10(y) for _, y, _ in positive]
    lo = math.floor(min(xs + ys))
    hi = math.ceil(max(xs + ys))

    def sx_log(value: float) -> float:
        return left + (math.log10(value) - lo) / (hi - lo) * plot_w

    def sy_log(value: float) -> float:
        return top + (hi - math.log10(value)) / (hi - lo) * plot_h

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:13px}.title{font-size:20px;font-weight:700}.axis{stroke:#222;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}</style>",
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text class="title" x="{left}" y="30">{svg_escape(title)}</text>',
    ]
    for exp in range(lo, hi + 1):
        x = left + (exp - lo) / (hi - lo) * plot_w
        y = top + (hi - exp) / (hi - lo) * plot_h
        label = f"1e{exp}"
        elements.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}"/>')
        elements.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}"/>')
        elements.append(f'<text x="{x:.2f}" y="{height - 35}" text-anchor="middle">{label}</text>')
        elements.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{label}</text>')

    elements.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    elements.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    elements.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top}" stroke="#555" stroke-width="1.5" stroke-dasharray="7 5"/>')

    for actual, predicted, group in positive:
        color = "#2ca02c" if group == "Asur-Huberman overlap" else "#222222"
        radius = 4 if group == "Asur-Huberman overlap" else 3
        elements.append(
            f'<circle cx="{sx_log(actual):.2f}" cy="{sy_log(predicted):.2f}" '
            f'r="{radius}" fill="{color}" fill-opacity="0.72"/>'
        )

    elements.append(f'<text x="{left + plot_w / 2:.2f}" y="{height - 10}" text-anchor="middle">{svg_escape(x_label)}</text>')
    elements.append(
        f'<text transform="translate(18 {top + plot_h / 2:.2f}) rotate(-90)" text-anchor="middle">{svg_escape(y_label)}</text>'
    )
    skipped = len(points) - len(positive)
    if skipped:
        elements.append(f'<text x="{left}" y="{height - 52}">Skipped {skipped} non-positive predicted values on log scale.</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def write_histogram_svg(path: Path, sample: Sample) -> None:
    day = 7
    rows = [
        ("Inception", [m.inception_days for m in sample.movies if m.inception_days is not None], False),
        ("Revenue", [m.revenue for m in sample.movies], True),
        ("Theaters", [m.theaters for m in sample.movies], True),
        ("Views", [sample.predictors[m.movie_id][day]["V"] for m in sample.movies], True),
        ("Users", [sample.predictors[m.movie_id][day]["U"] for m in sample.movies], True),
        ("Edits", [sample.predictors[m.movie_id][day]["E"] for m in sample.movies], True),
        ("Rigor", [sample.predictors[m.movie_id][day]["R"] for m in sample.movies], True),
    ]
    width, height = 980, 700
    margin = 36
    panel_w, panel_h = 290, 180
    gap_x, gap_y = 25, 42
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:12px}.title{font-size:20px;font-weight:700}.small{font-size:11px}</style>",
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text class="title" x="{margin}" y="28">Figure 1-style summary histograms, n={len(sample.movies)}</text>',
    ]

    for idx, (name, values, log_scale) in enumerate(rows):
        col = idx % 3
        row = idx // 3
        x0 = margin + col * (panel_w + gap_x)
        y0 = 52 + row * (panel_h + gap_y)
        clean = [v for v in values if v is not None and (not log_scale or v > 0)]
        plotted = [math.log10(v) for v in clean] if log_scale else clean
        bins = 14
        lo = min(plotted)
        hi = max(plotted)
        if lo == hi:
            lo -= 0.5
            hi += 0.5
        counts = [0] * bins
        for value in plotted:
            b = min(bins - 1, int((value - lo) / (hi - lo) * bins))
            counts[b] += 1
        max_count = max(counts) or 1
        elements.append(f'<text x="{x0}" y="{y0 - 9}" font-weight="700">{svg_escape(name)}</text>')
        elements.append(f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="#fafafa" stroke="#ccc"/>')
        for b, count in enumerate(counts):
            bar_w = panel_w / bins - 2
            bar_h = count / max_count * (panel_h - 28)
            bx = x0 + b * panel_w / bins + 1
            by = y0 + panel_h - bar_h - 22
            elements.append(f'<rect x="{bx:.2f}" y="{by:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="#4c78a8"/>')
        x_label = "log10(value)" if log_scale else "value"
        elements.append(f'<text class="small" x="{x0 + panel_w / 2}" y="{y0 + panel_h - 5}" text-anchor="middle">{x_label}</text>')
        elements.append(f'<text class="small" x="{x0 + 4}" y="{y0 + 14}">max bin {max_count}</text>')

    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def build_correlation_rows(sample: Sample) -> list[dict[str, object]]:
    revenues = [movie.revenue for movie in sample.movies]
    rows = []
    for day in sample.days:
        row: dict[str, object] = {"day": day}
        for feature in ("V", "U", "R", "E", "T"):
            if feature == "T":
                values = [movie.theaters for movie in sample.movies]
            else:
                values = [sample.predictors[movie.movie_id][day][feature] for movie in sample.movies]
            row[feature] = format_number(pearson(values, revenues))
        rows.append(row)
    return rows


def build_cv_rows(sample: Sample, folds: int, seed: int) -> list[dict[str, object]]:
    rows = []
    for day in sample.days:
        for model_name, features in MODEL_SETS:
            pooled, mean_fold = cross_validated_r2(sample, day, features, folds, seed)
            rows.append(
                {
                    "day": day,
                    "model": model_name,
                    "features": ",".join(features),
                    "pooled_r2": format_number(pooled),
                    "mean_fold_r2": format_number(mean_fold),
                }
            )
    return rows


def build_asur_rows(sample: Sample) -> list[dict[str, object]]:
    rows = []
    features = ("V", "U", "R", "E", "T")
    for day in sample.days:
        score = in_sample_r2(sample, day, features)
        rows.append({"day": day, "model": "Wikipedia model", "r2": format_number(score)})
    return rows


def predict_minus_30_rows(sample: Sample, overlap_titles: set[str]) -> list[dict[str, object]]:
    day = -30
    features = ("V", "U", "R", "E", "T")
    x = feature_matrix(sample, day, features)
    y = [movie.revenue for movie in sample.movies]
    predictions = linear_regression_predict(x, y, x)
    rows = []
    for movie, pred in zip(sample.movies, predictions):
        group = "Asur-Huberman overlap" if movie.title in overlap_titles else "2010 sample"
        rows.append(
            {
                "ID": movie.movie_id,
                "Title": movie.title,
                "actual_revenue": format_number(movie.revenue),
                "predicted_revenue": format_number(pred),
                "group": group,
            }
        )
    return rows


def numeric_from_rows(rows: list[dict[str, object]], key: str) -> float | None:
    value = rows[0][key]
    if value == "":
        return None
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip",
        type=Path,
        default=Path("data/external/supplementary/SupplementaryDataS1.zip"),
        help="Dataset S1 ZIP path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/papers/wikipedia_boxoffice"),
        help="Output directory for CSV and SVG files.",
    )
    parser.add_argument("--force-download", action="store_true", help="Download the dataset even if --zip already exists.")
    parser.add_argument("--folds", type=int, default=10, help="Number of cross-validation folds for Figure 3.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for the 10-fold split.")
    args = parser.parse_args()

    if args.folds < 2:
        raise SystemExit("--folds must be at least 2")

    zip_path = ensure_dataset(args.zip, args.force_download)
    args.out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        sample_312 = read_sample(zf, "sample_of_312", "sample_of_312")
        sample_24 = read_sample(
            zf, "asur_huberman_sample_of_24", "asur_huberman_sample_of_24"
        )

    correlation_rows = build_correlation_rows(sample_312)
    write_csv(args.out / "correlations_312.csv", correlation_rows, ["day", "V", "U", "R", "E", "T"])

    cv_rows = build_cv_rows(sample_312, args.folds, args.seed)
    write_csv(
        args.out / "r2_312_cross_validated.csv",
        cv_rows,
        ["day", "model", "features", "pooled_r2", "mean_fold_r2"],
    )

    asur_rows = build_asur_rows(sample_24)
    write_csv(args.out / "r2_asur_huberman_24.csv", asur_rows, ["day", "model", "r2"])

    overlap_titles = {movie.title for movie in sample_24.movies}
    prediction_rows = predict_minus_30_rows(sample_312, overlap_titles)
    write_csv(
        args.out / "predictions_minus30_312.csv",
        prediction_rows,
        ["ID", "Title", "actual_revenue", "predicted_revenue", "group"],
    )

    corr_series = {
        feature: [
            (int(row["day"]), float(row[feature]) if row[feature] != "" else None)
            for row in correlation_rows
        ]
        for feature in ("V", "U", "R", "E", "T")
    }
    write_line_svg(
        args.out / "figure2_correlations.svg",
        "Temporal correlation with opening-weekend revenue",
        corr_series,
        "movie time day",
        "Pearson r",
        x_range=(-500, 100),
        y_range=(0.0, 0.9),
    )

    cv_series = {}
    for model_name, _ in MODEL_SETS:
        cv_series[model_name] = [
            (
                int(row["day"]),
                float(row["pooled_r2"]) if row["model"] == model_name and row["pooled_r2"] != "" else None,
            )
            for row in cv_rows
            if row["model"] == model_name
        ]
    write_line_svg(
        args.out / "figure3_cv_r2.svg",
        f"{args.folds}-fold cross-validated R^2 on the 312-movie sample",
        cv_series,
        "movie time day",
        "pooled cross-validated R^2",
        x_range=(-500, 100),
        y_range=(0.0, 0.9),
    )

    asur_series = {
        "Wikipedia model": [
            (int(row["day"]), float(row["r2"]) if row["r2"] != "" else None)
            for row in asur_rows
        ]
    }
    write_line_svg(
        args.out / "figure4_asur_huberman_24.svg",
        "Wikipedia model on the 24-movie Asur-Huberman sample",
        asur_series,
        "movie time day",
        "in-sample R^2",
        x_range=(-500, 100),
        y_range=(0.0, 1.0),
        reference_lines=[(0.98, "Twitter reference 0.98", COLORS["Twitter reference"])],
    )

    scatter_points = [
        (float(row["actual_revenue"]), float(row["predicted_revenue"]), str(row["group"]))
        for row in prediction_rows
    ]
    write_scatter_svg(
        args.out / "figure5_actual_vs_predicted_minus30.svg",
        "Actual vs predicted revenue at t = -30 days",
        scatter_points,
        "actual first-weekend revenue (USD, log scale)",
        "predicted first-weekend revenue (USD, log scale)",
    )

    write_histogram_svg(args.out / "figure1_summary_histograms.svg", sample_312)

    def lookup(rows: list[dict[str, object]], day: int, model: str, column: str) -> str:
        for row in rows:
            if int(row["day"]) == day and row.get("model") == model:
                return str(row[column])
        return ""

    print(f"Read {len(sample_312.movies)} movies from the main sample.")
    print(f"Read {len(sample_24.movies)} movies from the Asur-Huberman comparison sample.")
    print(f"Wrote CSV and SVG outputs to {args.out}.")
    print(
        "Key checks: "
        f"r(V, revenue) at t=-30 = {next(row['V'] for row in correlation_rows if row['day'] == -30)}, "
        f"CV R^2(all predictors) at t=-30 = {lookup(cv_rows, -30, 'V+U+R+E+T', 'pooled_r2')}, "
        f"Asur-Huberman R^2(all predictors) at t=-30 = {lookup(asur_rows, -30, 'Wikipedia model', 'r2')}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
