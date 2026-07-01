#!/usr/bin/env python3
"""Recreate the core results from Mestyan, Yasseri, and Kertesz (2013).

The paper predicts opening-weekend movie revenue from Wikipedia activity:

    Early Prediction of Movie Box Office Success Based on Wikipedia Activity
    Big Data, PLoS ONE 8(8): e71226.

By default this script reads fresh local PostgreSQL data for every movie that
has both The Numbers actuals and a matched Wikipedia page. Pass
`--paper-dataset` to instead download/parse Dataset S1 and recreate the
original paper sample.

The main quantitative outputs are:

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
import datetime as dt
import io
import math
import random
import sys
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pm_box_office.db.connection import connect_database, database_url_from_env


PLOS_DATASET_URL = (
    "https://journals.plos.org/plosone/article/file?"
    "type=supplementary&id=10.1371/journal.pone.0071226.s002"
)
LEGACY_DATASET_URL = "http://www.phy.bme.hu/SupplementaryDataS1.zip"
ZIP_ROOT = "wikipredict_data_pack"
DEFAULT_FRESH_OUT_DIR = Path("results/papers/wikipedia_boxoffice_fresh")
DEFAULT_PAPER_OUT_DIR = Path("results/papers/wikipedia_boxoffice")
DEFAULT_DAY_START = -500
DEFAULT_DAY_END = 100
DEFAULT_QUARTILE_DAYS = (-30, -14, -7, -1, 0, 1, 7)
NEXT_DAY_MODEL_SETS = [
    ("baseline", ("log_current_day_gross", "days_since_release", "day_of_week")),
    (
        "baseline_plus_wikipedia",
        (
            "log_current_day_gross",
            "days_since_release",
            "day_of_week",
            "log1p_V",
            "log1p_U",
            "log1p_R",
            "log1p_E",
            "log1p_V_delta_1",
            "log1p_V_delta_3",
            "log1p_E_delta_3",
        ),
    ),
    (
        "baseline_plus_wikipedia_by_quartile",
        (
            "log_current_day_gross",
            "days_since_release",
            "day_of_week",
            "opening_day_gross_quartile",
            "log1p_V",
            "log1p_U",
            "log1p_R",
            "log1p_E",
            "log1p_V_delta_1",
            "log1p_V_delta_3",
            "log1p_E_delta_3",
            "quartile_x_log1p_V_delta_3",
        ),
    ),
]
OPENING_WEEKEND_TIME_MODEL_SETS = [
    ("paper_raw_all", "raw", ("V", "U", "R", "E", "T")),
    ("log_views_theaters", "log", ("log1p_V", "log1p_T")),
    ("log_all", "log", ("log1p_V", "log1p_U", "log1p_R", "log1p_E", "log1p_T")),
    (
        "log_all_q4_interactions",
        "log",
        (
            "log1p_V",
            "log1p_U",
            "log1p_R",
            "log1p_E",
            "log1p_T",
            "q4_flag",
            "q4_x_log1p_V",
            "q4_x_log1p_U",
            "q4_x_log1p_E",
        ),
    ),
]

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
    "Q1": "#1f77b4",
    "Q2": "#2ca02c",
    "Q3": "#ff7f0e",
    "Q4": "#9467bd",
}


@dataclass
class Movie:
    movie_id: str
    title: str
    wiki_title: str
    revenue: float
    theaters: float
    release_date: str
    opening_day_gross: float | None = None
    release_run_id: int | None = None
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


@dataclass(frozen=True)
class FreshMovieRow:
    movie_id: int
    title: str
    release_year: int | None
    release_run_id: int
    opening_date: dt.date
    opening_theaters: int
    opening_day_gross_usd: int
    opening_weekend_revenue_usd: int
    language: str
    wiki_page_id: int
    wiki_title: str
    match_status: str


def parse_db_date(value: Any) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


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


def load_fresh_candidate_movies(
    conn: Any,
    *,
    language: str,
    release_year: int | None,
    min_release_year: int | None,
    min_opening_theaters: int | None,
    min_opening_day_gross: int | None,
) -> list[FreshMovieRow]:
    sql = """
        WITH opening AS (
            SELECT
                rr.release_run_id,
                rr.movie_id,
                MIN(dbo.box_office_date::date) AS opening_date
            FROM release_runs rr
            JOIN daily_box_office dbo ON dbo.release_run_id = rr.release_run_id
            WHERE dbo.is_preview = 0
              AND dbo.gross_usd IS NOT NULL
            GROUP BY rr.release_run_id, rr.movie_id
        ),
        opening_features AS (
            SELECT
                opening.release_run_id,
                opening.movie_id,
                opening.opening_date,
                opening_day.theaters AS opening_theaters,
                opening_day.gross_usd AS opening_day_gross_usd,
                SUM(weekend.gross_usd) AS opening_weekend_revenue_usd
            FROM opening
            JOIN daily_box_office opening_day
              ON opening_day.release_run_id = opening.release_run_id
             AND opening_day.box_office_date::date = opening.opening_date
            JOIN daily_box_office weekend
              ON weekend.release_run_id = opening.release_run_id
             AND weekend.is_preview = 0
             AND weekend.box_office_date::date >= opening.opening_date
             AND weekend.box_office_date::date < opening.opening_date + INTERVAL '3 days'
            GROUP BY
                opening.release_run_id,
                opening.movie_id,
                opening.opening_date,
                opening_day.theaters,
                opening_day.gross_usd
        )
        SELECT
            m.movie_id,
            m.title,
            m.release_year,
            ofe.release_run_id,
            ofe.opening_date,
            ofe.opening_theaters,
            ofe.opening_day_gross_usd,
            ofe.opening_weekend_revenue_usd,
            mwp.language,
            mwp.wiki_page_id,
            COALESCE(wp.page_title, mwp.wiki_page_id::text) AS wiki_title,
            mwp.match_status
        FROM opening_features ofe
        JOIN movies m ON m.movie_id = ofe.movie_id
        JOIN movie_wiki_pages mwp ON mwp.movie_id = m.movie_id
        LEFT JOIN wiki_pages wp
          ON wp.language = mwp.language
         AND wp.wiki_page_id = mwp.wiki_page_id
        WHERE mwp.language = %s
          AND mwp.match_status IN ('matched', 'manual_override')
          AND mwp.wiki_page_id IS NOT NULL
          AND ofe.opening_theaters IS NOT NULL
          AND ofe.opening_day_gross_usd IS NOT NULL
          AND ofe.opening_weekend_revenue_usd IS NOT NULL
    """
    params: list[Any] = [language]
    if release_year is not None:
        sql += " AND m.release_year = %s"
        params.append(release_year)
    if min_release_year is not None:
        sql += " AND m.release_year >= %s"
        params.append(min_release_year)
    if min_opening_theaters is not None:
        sql += " AND ofe.opening_theaters >= %s"
        params.append(min_opening_theaters)
    if min_opening_day_gross is not None:
        sql += " AND ofe.opening_day_gross_usd >= %s"
        params.append(min_opening_day_gross)
    sql += " ORDER BY ofe.opening_date, m.title, m.movie_id"

    rows = conn.execute(sql, params).fetchall()
    return [
        FreshMovieRow(
            movie_id=int(row[0]),
            title=str(row[1]),
            release_year=int(row[2]) if row[2] is not None else None,
            release_run_id=int(row[3]),
            opening_date=parse_db_date(row[4]),
            opening_theaters=int(row[5]),
            opening_day_gross_usd=int(row[6]),
            opening_weekend_revenue_usd=int(row[7]),
            language=str(row[8]),
            wiki_page_id=int(row[9]),
            wiki_title=str(row[10]),
            match_status=str(row[11]),
        )
        for row in rows
    ]


def load_pageview_increments(
    conn: Any,
    movie: FreshMovieRow,
    *,
    day_start: int,
    day_end: int,
) -> tuple[dict[int, float], int, int]:
    rows = conn.execute(
        """
        SELECT
            (view_date::date - %s::date) AS movie_time_day,
            SUM(views) AS views,
            COUNT(*) AS rows
        FROM wiki_pageviews_daily
        WHERE language = %s
          AND wiki_page_id = %s
          AND agent = 'user'
          AND view_date::date >= %s::date + (%s * INTERVAL '1 day')
          AND view_date::date <= %s::date + (%s * INTERVAL '1 day')
        GROUP BY movie_time_day
        ORDER BY movie_time_day
        """,
        (
            movie.opening_date.isoformat(),
            movie.language,
            movie.wiki_page_id,
            movie.opening_date.isoformat(),
            day_start,
            movie.opening_date.isoformat(),
            day_end,
        ),
    ).fetchall()
    increments: dict[int, float] = {}
    source_rows = 0
    total_views = 0
    for row in rows:
        day = int(row[0])
        views = int(row[1] or 0)
        increments[day] = increments.get(day, 0.0) + float(views)
        source_rows += int(row[2] or 0)
        total_views += views
    return increments, source_rows, total_views


def load_revision_increments(
    conn: Any,
    movie: FreshMovieRow,
    *,
    day_start: int,
    day_end: int,
) -> tuple[dict[int, list[str]], int]:
    rows = conn.execute(
        """
        SELECT
            (rev_date::date - %s::date) AS movie_time_day,
            user_key
        FROM wiki_revisions
        WHERE language = %s
          AND wiki_page_id = %s
          AND is_bot = 0
          AND rev_date::date >= %s::date + (%s * INTERVAL '1 day')
          AND rev_date::date <= %s::date + (%s * INTERVAL '1 day')
        ORDER BY rev_timestamp, rev_id
        """,
        (
            movie.opening_date.isoformat(),
            movie.language,
            movie.wiki_page_id,
            movie.opening_date.isoformat(),
            day_start,
            movie.opening_date.isoformat(),
            day_end,
        ),
    ).fetchall()
    revisions_by_day: dict[int, list[str]] = {}
    for row in rows:
        revisions_by_day.setdefault(int(row[0]), []).append(str(row[1]))
    return revisions_by_day, len(rows)


def build_dense_predictors(
    *,
    pageviews_by_day: dict[int, float],
    revisions_by_day: dict[int, list[str]],
    day_start: int,
    day_end: int,
) -> dict[int, dict[str, float]]:
    predictors: dict[int, dict[str, float]] = {}
    views = 0.0
    edits = 0.0
    rigor = 0.0
    users: set[str] = set()
    previous_user: str | None = None
    for day in range(day_start, day_end + 1):
        views += pageviews_by_day.get(day, 0.0)
        for user_key in revisions_by_day.get(day, []):
            edits += 1.0
            users.add(user_key)
            if previous_user is None or previous_user != user_key:
                rigor += 1.0
            previous_user = user_key
        predictors[day] = {
            "V": views,
            "U": float(len(users)),
            "R": rigor,
            "E": edits,
        }
    return predictors


def load_fresh_sample(
    conn: Any,
    *,
    language: str = "en",
    day_start: int = DEFAULT_DAY_START,
    day_end: int = DEFAULT_DAY_END,
    release_year: int | None = None,
    min_release_year: int | None = None,
    min_opening_theaters: int | None = None,
    min_opening_day_gross: int | None = None,
) -> tuple[Sample, list[dict[str, object]]]:
    if day_end < day_start:
        raise SystemExit("--day-end must be greater than or equal to --day-start")

    candidates = load_fresh_candidate_movies(
        conn,
        language=language,
        release_year=release_year,
        min_release_year=min_release_year,
        min_opening_theaters=min_opening_theaters,
        min_opening_day_gross=min_opening_day_gross,
    )
    movies: list[Movie] = []
    predictors: dict[str, dict[int, dict[str, float]]] = {}
    coverage_rows: list[dict[str, object]] = []

    for candidate in candidates:
        pageviews_by_day, pageview_rows, total_views = load_pageview_increments(
            conn,
            candidate,
            day_start=day_start,
            day_end=day_end,
        )
        revisions_by_day, revision_rows = load_revision_increments(
            conn,
            candidate,
            day_start=day_start,
            day_end=day_end,
        )
        activity_days = sorted(set(pageviews_by_day) | set(revisions_by_day))
        included = bool(activity_days)
        coverage_rows.append(
            {
                "movie_id": candidate.movie_id,
                "title": candidate.title,
                "release_year": candidate.release_year if candidate.release_year is not None else "",
                "release_run_id": candidate.release_run_id,
                "opening_date": candidate.opening_date.isoformat(),
                "opening_theaters": candidate.opening_theaters,
                "opening_day_gross_usd": candidate.opening_day_gross_usd,
                "opening_weekend_revenue_usd": candidate.opening_weekend_revenue_usd,
                "language": candidate.language,
                "wiki_page_id": candidate.wiki_page_id,
                "wiki_title": candidate.wiki_title,
                "match_status": candidate.match_status,
                "pageview_rows": pageview_rows,
                "total_views": total_views,
                "human_revision_rows": revision_rows,
                "first_activity_day": activity_days[0] if activity_days else "",
                "last_activity_day": activity_days[-1] if activity_days else "",
                "included": int(included),
            }
        )
        if not included:
            continue

        movie_id = str(candidate.movie_id)
        movies.append(
            Movie(
                movie_id=movie_id,
                title=candidate.title,
                wiki_title=candidate.wiki_title,
                revenue=float(candidate.opening_weekend_revenue_usd),
                theaters=float(candidate.opening_theaters),
                release_date=candidate.opening_date.isoformat(),
                opening_day_gross=float(candidate.opening_day_gross_usd),
                release_run_id=candidate.release_run_id,
                inception_days=float(activity_days[0]),
            )
        )
        predictors[movie_id] = build_dense_predictors(
            pageviews_by_day=pageviews_by_day,
            revisions_by_day=revisions_by_day,
            day_start=day_start,
            day_end=day_end,
        )

    return Sample("fresh_postgres", movies, predictors), coverage_rows


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
    if len(sample.movies) < 2:
        return None, None
    effective_folds = min(folds, len(sample.movies))
    x = feature_matrix(sample, day, features)
    y = [movie.revenue for movie in sample.movies]
    indices = list(range(len(sample.movies)))
    random.Random(seed).shuffle(indices)
    test_folds = [indices[i::effective_folds] for i in range(effective_folds)]

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
    if not positive:
        path.write_text(
            "\n".join(
                [
                    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                    "<style>text{font-family:Arial,sans-serif;font-size:13px}.title{font-size:20px;font-weight:700}</style>",
                    '<rect width="100%" height="100%" fill="#fff"/>',
                    f'<text class="title" x="{left}" y="30">{svg_escape(title)}</text>',
                    f'<text x="{left}" y="70">No positive predicted revenue values were available for a log-scale scatter.</text>',
                    "</svg>",
                ]
            ),
            encoding="utf-8",
        )
        return
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
        if not plotted:
            plotted = [0.0]
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


def write_placeholder_svg(path: Path, title: str, message: str) -> None:
    width, height = 760, 360
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                "<style>text{font-family:Arial,sans-serif;font-size:13px}.title{font-size:20px;font-weight:700}</style>",
                '<rect width="100%" height="100%" fill="#fff"/>',
                f'<text class="title" x="40" y="42">{svg_escape(title)}</text>',
                f'<text x="40" y="86">{svg_escape(message)}</text>',
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )


def write_simple_bar_svg(
    path: Path,
    *,
    title: str,
    bars: list[tuple[str, float, str]],
    y_label: str,
    note: str = "",
) -> None:
    width, height = 900, 520
    left, right, top, bottom = 80, 45, 62, 95
    plot_w = width - left - right
    plot_h = height - top - bottom
    if not bars:
        write_placeholder_svg(path, title, "No rows were available for this plot.")
        return
    max_value = max(abs(value) for _, value, _ in bars) or 1.0
    has_negative = any(value < 0 for _, value, _ in bars)
    min_value = -max_value if has_negative else 0.0
    max_axis = max_value

    def sy(value: float) -> float:
        return top + (max_axis - value) / (max_axis - min_value) * plot_h

    baseline_y = sy(0.0)
    bar_slot = plot_w / len(bars)
    bar_w = min(46.0, bar_slot * 0.65)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:12px}.title{font-size:20px;font-weight:700}.axis{stroke:#222;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}</style>",
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text class="title" x="{left}" y="34">{svg_escape(title)}</text>',
    ]
    for i in range(5):
        value = min_value + (max_axis - min_value) * i / 4
        y = sy(value)
        elements.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}"/>')
        elements.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{value:.2g}</text>')
    elements.append(f'<line class="axis" x1="{left}" y1="{baseline_y:.2f}" x2="{left + plot_w}" y2="{baseline_y:.2f}"/>')
    elements.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    for idx, (label, value, color) in enumerate(bars):
        x = left + idx * bar_slot + (bar_slot - bar_w) / 2
        y = sy(max(value, 0.0))
        bar_h = abs(sy(value) - baseline_y)
        if value < 0:
            y = baseline_y
        elements.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="{color}" fill-opacity="0.82"/>'
        )
        elements.append(
            f'<text transform="translate({x + bar_w / 2:.2f} {height - 58}) rotate(-35)" text-anchor="end">{svg_escape(label)}</text>'
        )
    elements.append(
        f'<text transform="translate(18 {top + plot_h / 2:.2f}) rotate(-90)" text-anchor="middle">{svg_escape(y_label)}</text>'
    )
    if note:
        elements.append(f'<text x="{left}" y="{height - 18}" fill="#555">{svg_escape(note)}</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def write_opening_day_quartile_distribution_svg(path: Path, rows: list[dict[str, object]]) -> None:
    groups: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        groups[int(row["opening_day_gross_quartile"])].append(float(row["opening_day_gross_usd"]))
    bars = [
        (
            f"Q{quartile}",
            float(len(values)),
            COLORS.get("V", "#1f77b4"),
        )
        for quartile, values in sorted(groups.items())
    ]
    write_simple_bar_svg(
        path,
        title="Opening-day gross quartile distribution",
        bars=bars,
        y_label="movies",
        note="Quartiles are relative to the filtered fresh sample.",
    )


def write_wiki_features_by_quartile_svg(path: Path, rows: list[dict[str, object]], *, preferred_day: int = 0) -> None:
    available_days = sorted({int(row["day"]) for row in rows})
    if not available_days:
        write_placeholder_svg(path, "Wikipedia features by quartile", "No quartile summary rows were available.")
        return
    day = preferred_day if preferred_day in available_days else available_days[-1]
    day_rows = [row for row in rows if int(row["day"]) == day]
    bars: list[tuple[str, float, str]] = []
    feature_colors = {"V": "#1f77b4", "U": "#2ca02c", "R": "#9467bd", "E": "#ff7f0e", "T": "#444444"}
    for quartile in range(1, 5):
        row = next((item for item in day_rows if int(item["opening_day_gross_quartile"]) == quartile), None)
        if row is None or int(row["movies"]) == 0:
            continue
        for feature in ("V", "U", "R", "E", "T"):
            value = row.get(f"mean_{feature}")
            if value == "" or value is None:
                continue
            bars.append((f"Q{quartile} {feature}", math.log1p(float(value)), feature_colors[feature]))
    write_simple_bar_svg(
        path,
        title=f"Wikipedia feature scale by quartile at day {day}",
        bars=bars,
        y_label="log1p(mean value)",
    )


def write_quartile_line_svg(
    path: Path,
    rows: list[dict[str, object]],
    *,
    title: str,
    metric_column: str,
    y_label: str,
    x_range: tuple[float, float],
) -> None:
    series: dict[str, list[tuple[float, float | None]]] = {}
    for quartile in range(1, 5):
        series[f"Q{quartile}"] = [
            (
                int(row["day"]),
                float(row[metric_column]) if row[metric_column] != "" else None,
            )
            for row in rows
            if int(row["opening_day_gross_quartile"]) == quartile
        ]
    if not any(y is not None for values in series.values() for _, y in values):
        write_placeholder_svg(path, title, "Not enough variation was available for this line plot.")
        return
    write_line_svg(path, title, series, "movie time day", y_label, x_range=x_range)


def write_next_day_model_lift_svg(path: Path, metric_rows: list[dict[str, object]]) -> None:
    baseline = {
        (row["segment_type"], row["segment"]): float(row["rmse_log_ratio"])
        for row in metric_rows
        if row["model"] == "baseline" and row["rmse_log_ratio"] != ""
    }
    bars: list[tuple[str, float, str]] = []
    for row in metric_rows:
        if row["model"] != "baseline_plus_wikipedia" or row["segment_type"] != "quartile":
            continue
        key = (row["segment_type"], row["segment"])
        if key not in baseline or row["rmse_log_ratio"] == "":
            continue
        lift = baseline[key] - float(row["rmse_log_ratio"])
        bars.append((str(row["segment"]), lift, "#2ca02c" if lift >= 0 else "#d62728"))
    write_simple_bar_svg(
        path,
        title="Next-day model lift from Wikipedia features",
        bars=bars,
        y_label="baseline RMSE - wiki RMSE",
        note="Positive bars mean Wikipedia features reduced log-ratio RMSE.",
    )


def write_next_day_error_by_release_age_svg(path: Path, metric_rows: list[dict[str, object]]) -> None:
    bars = [
        (str(row["segment"]), float(row["mae_log_ratio"]), "#4c78a8")
        for row in metric_rows
        if row["model"] == "baseline_plus_wikipedia"
        and row["segment_type"] == "release_age_bucket"
        and row["mae_log_ratio"] != ""
    ]
    write_simple_bar_svg(
        path,
        title="Next-day Wikipedia model error by release age",
        bars=bars,
        y_label="MAE log ratio",
    )


def write_prediction_residuals_by_quartile_svg(path: Path, metric_rows: list[dict[str, object]]) -> None:
    bars = [
        (
            str(row["segment"]),
            float(row["mean_residual_log_gross"]),
            "#9467bd" if float(row["mean_residual_log_gross"]) >= 0 else "#ff7f0e",
        )
        for row in metric_rows
        if row["model"] == "baseline_plus_wikipedia"
        and row["segment_type"] == "quartile"
        and row["mean_residual_log_gross"] != ""
    ]
    write_simple_bar_svg(
        path,
        title="Prediction residuals by opening-day quartile",
        bars=bars,
        y_label="mean actual - predicted log gross",
    )


def write_wiki_momentum_scatter_svg(path: Path, panel_rows: list[dict[str, object]]) -> None:
    points = [
        (math.log1p(float(row["V_delta_3"])), float(row["next_day_over_current_day"]), str(row["opening_day_gross_quartile_label"]))
        for row in panel_rows
        if float(row["next_day_over_current_day"]) > 0
    ]
    if not points:
        write_placeholder_svg(path, "Wikipedia momentum vs next-day ratio", "No next-day panel rows were available.")
        return
    width, height = 820, 580
    left, right, top, bottom = 80, 145, 54, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    xmin, xmax = min(x for x, _, _ in points), max(x for x, _, _ in points)
    ymin, ymax = min(y for _, y, _ in points), max(y for _, y, _ in points)
    if xmin == xmax:
        xmin -= 0.5
        xmax += 0.5
    if ymin == ymax:
        ymin -= 0.5
        ymax += 0.5

    def sx(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * plot_w

    def sy(value: float) -> float:
        return top + (ymax - value) / (ymax - ymin) * plot_h

    colors = {"Q1": "#1f77b4", "Q2": "#2ca02c", "Q3": "#ff7f0e", "Q4": "#9467bd"}
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:12px}.title{font-size:20px;font-weight:700}.axis{stroke:#222;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}</style>",
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text class="title" x="{left}" y="34">Wikipedia momentum vs next-day gross ratio</text>',
    ]
    for i in range(5):
        x = xmin + (xmax - xmin) * i / 4
        y = ymin + (ymax - ymin) * i / 4
        px = sx(x)
        py = sy(y)
        elements.append(f'<line class="grid" x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{top + plot_h}"/>')
        elements.append(f'<line class="grid" x1="{left}" y1="{py:.2f}" x2="{left + plot_w}" y2="{py:.2f}"/>')
        elements.append(f'<text x="{px:.2f}" y="{height - 35}" text-anchor="middle">{x:.1f}</text>')
        elements.append(f'<text x="{left - 10}" y="{py + 4:.2f}" text-anchor="end">{y:.2f}</text>')
    elements.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    elements.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    for x, y, label in points:
        elements.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3.5" fill="{colors.get(label, "#333")}" fill-opacity="0.7"/>')
    for idx, label in enumerate(("Q1", "Q2", "Q3", "Q4")):
        y = top + 20 + idx * 22
        elements.append(f'<circle cx="{left + plot_w + 28}" cy="{y}" r="5" fill="{colors[label]}"/>')
        elements.append(f'<text x="{left + plot_w + 42}" y="{y + 4}">{label}</text>')
    elements.append(f'<text x="{left + plot_w / 2:.2f}" y="{height - 10}" text-anchor="middle">log1p(3-day view delta)</text>')
    elements.append(
        f'<text transform="translate(18 {top + plot_h / 2:.2f}) rotate(-90)" text-anchor="middle">next-day / current-day gross</text>'
    )
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


def predict_minus_30_rows(
    sample: Sample,
    overlap_titles: set[str],
    *,
    default_group: str = "2010 sample",
) -> list[dict[str, object]]:
    day = -30
    features = ("V", "U", "R", "E", "T")
    x = feature_matrix(sample, day, features)
    y = [movie.revenue for movie in sample.movies]
    predictions = linear_regression_predict(x, y, x)
    rows = []
    for movie, pred in zip(sample.movies, predictions):
        group = "Asur-Huberman overlap" if movie.title in overlap_titles else default_group
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


def lookup_metric(rows: list[dict[str, object]], day: int, model: str, column: str) -> str:
    for row in rows:
        if int(row["day"]) == day and row.get("model") == model:
            return str(row[column])
    return ""


def parse_day_list(value: str) -> list[int]:
    days: list[int] = []
    for part in value.split(","):
        clean = part.strip()
        if not clean:
            continue
        days.append(int(clean))
    return sorted(set(days))


def median(values: Iterable[float]) -> float | None:
    clean = sorted(values)
    if not clean:
        return None
    midpoint = len(clean) // 2
    if len(clean) % 2:
        return clean[midpoint]
    return (clean[midpoint - 1] + clean[midpoint]) / 2.0


def movie_quartile_basis_value(movie: Movie, basis: str) -> float:
    if basis != "opening_day_gross":
        raise ValueError(f"Unsupported quartile basis {basis!r}; expected 'opening_day_gross'")
    if movie.opening_day_gross is None:
        raise ValueError(f"Movie {movie.movie_id} has no opening-day gross")
    return float(movie.opening_day_gross)


def assign_opening_day_quartiles(sample: Sample, *, basis: str = "opening_day_gross") -> dict[str, int]:
    values = sorted(
        (movie.movie_id, movie_quartile_basis_value(movie, basis))
        for movie in sample.movies
    )
    if not values:
        return {}
    values.sort(key=lambda item: (item[1], item[0]))
    out: dict[str, int] = {}
    n = len(values)
    index = 0
    while index < n:
        end = index
        while end + 1 < n and values[end + 1][1] == values[index][1]:
            end += 1
        midpoint_rank = (index + end) / 2.0
        quartile = min(4, int(math.floor(midpoint_rank * 4 / n)) + 1)
        for pos in range(index, end + 1):
            out[values[pos][0]] = quartile
        index = end + 1
    return out


def release_year(movie: Movie) -> int:
    return dt.date.fromisoformat(movie.release_date).year


def fixed_quartile_thresholds(
    sample: Sample,
    *,
    basis: str = "opening_day_gross",
    train_through_year: int,
) -> list[float]:
    values = sorted(
        movie_quartile_basis_value(movie, basis)
        for movie in sample.movies
        if release_year(movie) <= train_through_year
    )
    if not values:
        return []
    thresholds = []
    for frac in (0.25, 0.50, 0.75):
        index = min(len(values) - 1, max(0, math.ceil(len(values) * frac) - 1))
        thresholds.append(values[index])
    return thresholds


def assign_opening_day_quartiles_from_thresholds(
    sample: Sample,
    thresholds: list[float],
    *,
    basis: str = "opening_day_gross",
) -> dict[str, int]:
    if len(thresholds) != 3:
        raise ValueError("fixed quartile assignment requires exactly three thresholds")
    out = {}
    for movie in sample.movies:
        value = movie_quartile_basis_value(movie, basis)
        if value <= thresholds[0]:
            quartile = 1
        elif value <= thresholds[1]:
            quartile = 2
        elif value <= thresholds[2]:
            quartile = 3
        else:
            quartile = 4
        out[movie.movie_id] = quartile
    return out


def fixed_quartile_threshold_rows(thresholds: list[float], *, basis: str, train_through_year: int) -> list[dict[str, object]]:
    rows = []
    lower = ""
    for index, upper in enumerate(thresholds + [math.inf], start=1):
        rows.append(
            {
                "quartile_basis": basis,
                "train_through_year": train_through_year,
                "opening_day_gross_quartile": index,
                "opening_day_gross_quartile_label": f"Q{index}",
                "lower_bound_usd": lower,
                "upper_bound_usd": "" if math.isinf(upper) else format_number(upper),
            }
        )
        lower = format_number(upper)
    return rows


def opening_day_quartile_rows(
    sample: Sample,
    quartiles: dict[str, int],
    *,
    basis: str,
) -> list[dict[str, object]]:
    rows = []
    for movie in sample.movies:
        quartile = quartiles[movie.movie_id]
        rows.append(
            {
                "movie_id": movie.movie_id,
                "title": movie.title,
                "release_date": movie.release_date,
                "quartile_basis": basis,
                "opening_day_gross_usd": format_number(movie.opening_day_gross),
                "opening_weekend_revenue_usd": format_number(movie.revenue),
                "opening_theaters": format_number(movie.theaters),
                "opening_day_gross_quartile": quartile,
                "opening_day_gross_quartile_label": f"Q{quartile}",
            }
        )
    return rows


def subset_sample(sample: Sample, movie_ids: set[str], name: str) -> Sample:
    movies = [movie for movie in sample.movies if movie.movie_id in movie_ids]
    predictors = {movie.movie_id: sample.predictors[movie.movie_id] for movie in movies}
    return Sample(name, movies, predictors)


def build_wiki_quartile_summary_rows(
    sample: Sample,
    quartiles: dict[str, int],
    *,
    days: list[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for day in days:
        for quartile in range(1, 5):
            movies = [
                movie
                for movie in sample.movies
                if quartiles.get(movie.movie_id) == quartile
                and day in sample.predictors.get(movie.movie_id, {})
            ]
            row: dict[str, object] = {
                "day": day,
                "opening_day_gross_quartile": quartile,
                "opening_day_gross_quartile_label": f"Q{quartile}",
                "movies": len(movies),
            }
            for feature in ("V", "U", "R", "E"):
                values = [sample.predictors[movie.movie_id][day][feature] for movie in movies]
                row[f"mean_{feature}"] = format_number(mean(values)) if values else ""
                row[f"median_{feature}"] = format_number(median(values)) if values else ""
            theater_values = [movie.theaters for movie in movies]
            opening_day_values = [movie.opening_day_gross or 0.0 for movie in movies]
            weekend_values = [movie.revenue for movie in movies]
            row["mean_T"] = format_number(mean(theater_values)) if theater_values else ""
            row["mean_opening_day_gross_usd"] = format_number(mean(opening_day_values)) if opening_day_values else ""
            row["mean_opening_weekend_revenue_usd"] = format_number(mean(weekend_values)) if weekend_values else ""
            rows.append(row)
    return rows


def build_quartile_model_metric_rows(
    sample: Sample,
    quartiles: dict[str, int],
    *,
    folds: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for quartile in range(1, 5):
        movie_ids = {movie_id for movie_id, value in quartiles.items() if value == quartile}
        quartile_sample = subset_sample(sample, movie_ids, f"Q{quartile}")
        revenues = [movie.revenue for movie in quartile_sample.movies]
        for day in sample.days:
            values = [
                quartile_sample.predictors[movie.movie_id][day]["V"]
                for movie in quartile_sample.movies
                if day in quartile_sample.predictors[movie.movie_id]
            ]
            if len(values) == len(revenues) and len(values) >= 2:
                v_corr = pearson(values, revenues)
            else:
                v_corr = None
            pooled, mean_fold = cross_validated_r2(
                quartile_sample,
                day,
                ("V", "U", "R", "E", "T"),
                folds,
                seed,
            )
            rows.append(
                {
                    "day": day,
                    "opening_day_gross_quartile": quartile,
                    "opening_day_gross_quartile_label": f"Q{quartile}",
                    "movies": len(quartile_sample.movies),
                    "v_correlation": format_number(v_corr),
                    "all_predictors_pooled_r2": format_number(pooled),
                    "all_predictors_mean_fold_r2": format_number(mean_fold),
                }
            )
    return rows


def load_daily_actuals_by_movie(conn: Any, sample: Sample) -> dict[str, dict[int, float]]:
    actuals: dict[str, dict[int, float]] = {}
    for movie in sample.movies:
        if movie.release_run_id is None:
            continue
        opening_date = dt.date.fromisoformat(movie.release_date)
        rows = conn.execute(
            """
            SELECT
                dbo.box_office_date::date AS box_office_date,
                SUM(dbo.gross_usd) AS gross_usd
            FROM daily_box_office dbo
            WHERE dbo.release_run_id = %s
              AND dbo.is_preview = 0
              AND dbo.gross_usd IS NOT NULL
              AND dbo.gross_usd > 0
            GROUP BY dbo.box_office_date::date
            ORDER BY dbo.box_office_date::date
            """,
            (movie.release_run_id,),
        ).fetchall()
        by_day: dict[int, float] = {}
        for row in rows:
            date = parse_db_date(row[0])
            by_day[(date - opening_date).days] = float(row[1])
        actuals[movie.movie_id] = by_day
    return actuals


def release_age_bucket(age_days: int) -> str:
    if age_days <= 2:
        return "opening_weekend"
    if age_days <= 6:
        return "first_week"
    if age_days <= 13:
        return "second_week"
    return "later_run"


def predictor_delta(predictors: dict[int, dict[str, float]], day: int, feature: str, window: int) -> float:
    current = predictors[day][feature]
    previous = predictors.get(day - window, {}).get(feature, 0.0)
    return max(0.0, current - previous)


def build_next_day_panel_rows(
    sample: Sample,
    daily_actuals: dict[str, dict[int, float]],
    quartiles: dict[str, int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    movies_by_id = {movie.movie_id: movie for movie in sample.movies}
    for movie_id, actuals_by_day in daily_actuals.items():
        movie = movies_by_id.get(movie_id)
        if movie is None:
            continue
        opening_date = dt.date.fromisoformat(movie.release_date)
        predictors = sample.predictors.get(movie_id, {})
        for day in sorted(actuals_by_day):
            next_day = day + 1
            if day < 0 or next_day not in actuals_by_day or day not in predictors:
                continue
            current_gross = actuals_by_day[day]
            next_gross = actuals_by_day[next_day]
            if current_gross <= 0 or next_gross <= 0:
                continue
            day_values = predictors[day]
            quartile = quartiles[movie_id]
            v_delta_1 = predictor_delta(predictors, day, "V", 1)
            v_delta_3 = predictor_delta(predictors, day, "V", 3)
            v_delta_7 = predictor_delta(predictors, day, "V", 7)
            e_delta_1 = predictor_delta(predictors, day, "E", 1)
            e_delta_3 = predictor_delta(predictors, day, "E", 3)
            exhibition_date = opening_date + dt.timedelta(days=day)
            row = {
                "movie_id": movie.movie_id,
                "title": movie.title,
                "release_date": movie.release_date,
                "box_office_date": exhibition_date.isoformat(),
                "movie_time_day": day,
                "release_age_bucket": release_age_bucket(day),
                "current_day_gross_usd": current_gross,
                "next_day_gross_usd": next_gross,
                "log_current_day_gross": math.log(current_gross),
                "log_next_day_gross": math.log(next_gross),
                "log_next_day_over_current_day": math.log(next_gross / current_gross),
                "next_day_over_current_day": next_gross / current_gross,
                "days_since_release": float(day),
                "day_of_week": float(exhibition_date.isoweekday()),
                "opening_day_gross_usd": movie.opening_day_gross or 0.0,
                "opening_weekend_revenue_usd": movie.revenue,
                "opening_theaters": movie.theaters,
                "opening_day_gross_quartile": quartile,
                "opening_day_gross_quartile_label": f"Q{quartile}",
                "V": day_values["V"],
                "U": day_values["U"],
                "R": day_values["R"],
                "E": day_values["E"],
                "V_delta_1": v_delta_1,
                "V_delta_3": v_delta_3,
                "V_delta_7": v_delta_7,
                "E_delta_1": e_delta_1,
                "E_delta_3": e_delta_3,
                "log1p_V": math.log1p(max(0.0, day_values["V"])),
                "log1p_U": math.log1p(max(0.0, day_values["U"])),
                "log1p_R": math.log1p(max(0.0, day_values["R"])),
                "log1p_E": math.log1p(max(0.0, day_values["E"])),
                "log1p_V_delta_1": math.log1p(v_delta_1),
                "log1p_V_delta_3": math.log1p(v_delta_3),
                "log1p_E_delta_3": math.log1p(e_delta_3),
                "quartile_x_log1p_V_delta_3": quartile * math.log1p(v_delta_3),
            }
            rows.append(row)
    return rows


def numeric_feature_matrix(rows: list[dict[str, object]], features: tuple[str, ...]) -> list[list[float]]:
    return [[float(row[feature]) for feature in features] for row in rows]


def cross_validated_numeric_predictions(
    rows: list[dict[str, object]],
    *,
    target: str,
    features: tuple[str, ...],
    folds: int,
    seed: int,
) -> list[float]:
    if not rows:
        return []
    y = [float(row[target]) for row in rows]
    if len(rows) < 2:
        return [y[0]]
    x = numeric_feature_matrix(rows, features)
    effective_folds = min(folds, len(rows))
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    test_folds = [indices[i::effective_folds] for i in range(effective_folds)]
    predictions = [mean(y)] * len(rows)
    for test_idx in test_folds:
        test_set = set(test_idx)
        train_idx = [idx for idx in indices if idx not in test_set]
        if not train_idx:
            continue
        fold_predictions = linear_regression_predict(
            [x[idx] for idx in train_idx],
            [y[idx] for idx in train_idx],
            [x[idx] for idx in test_idx],
        )
        for idx, pred in zip(test_idx, fold_predictions):
            predictions[idx] = pred
    return predictions


def next_day_prediction_and_metric_rows(
    panel_rows: list[dict[str, object]],
    *,
    folds: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for model_name, features in NEXT_DAY_MODEL_SETS:
        predictions = cross_validated_numeric_predictions(
            panel_rows,
            target="log_next_day_over_current_day",
            features=features,
            folds=folds,
            seed=seed,
        )
        model_predictions: list[dict[str, object]] = []
        for source, pred_log_ratio in zip(panel_rows, predictions):
            current_gross = float(source["current_day_gross_usd"])
            predicted_gross = current_gross * math.exp(pred_log_ratio)
            actual_gross = float(source["next_day_gross_usd"])
            residual = math.log(actual_gross) - math.log(predicted_gross)
            row = {
                "model": model_name,
                "movie_id": source["movie_id"],
                "title": source["title"],
                "box_office_date": source["box_office_date"],
                "movie_time_day": source["movie_time_day"],
                "release_age_bucket": source["release_age_bucket"],
                "opening_day_gross_quartile": source["opening_day_gross_quartile"],
                "opening_day_gross_quartile_label": source["opening_day_gross_quartile_label"],
                "current_day_gross_usd": format_number(current_gross),
                "actual_next_day_gross_usd": format_number(actual_gross),
                "predicted_next_day_gross_usd": format_number(predicted_gross),
                "actual_log_next_day_over_current_day": format_number(source["log_next_day_over_current_day"]),
                "predicted_log_next_day_over_current_day": format_number(pred_log_ratio),
                "residual_log_gross": format_number(residual),
                "absolute_percentage_error": format_number(abs(predicted_gross - actual_gross) / actual_gross),
            }
            prediction_rows.append(row)
            model_predictions.append(row)

        metric_rows.extend(summarize_next_day_metrics(model_name, model_predictions))
    return prediction_rows, metric_rows


def release_year_from_panel_row(row: dict[str, object]) -> int:
    return dt.date.fromisoformat(str(row["release_date"])).year


def time_split_next_day_prediction_and_metric_rows(
    panel_rows: list[dict[str, object]],
    *,
    train_through_year: int,
    test_start_year: int,
    test_end_year: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train_rows = [row for row in panel_rows if release_year_from_panel_row(row) <= train_through_year]
    test_rows = [
        row
        for row in panel_rows
        if test_start_year <= release_year_from_panel_row(row) <= test_end_year
    ]
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    if not train_rows or not test_rows:
        return prediction_rows, metric_rows

    for model_name, features in NEXT_DAY_MODEL_SETS:
        train_x = numeric_feature_matrix(train_rows, features)
        train_y = [float(row["log_next_day_over_current_day"]) for row in train_rows]
        test_x = numeric_feature_matrix(test_rows, features)
        predictions = linear_regression_predict(train_x, train_y, test_x)
        model_predictions: list[dict[str, object]] = []
        for source, pred_log_ratio in zip(test_rows, predictions):
            current_gross = float(source["current_day_gross_usd"])
            predicted_gross = current_gross * math.exp(pred_log_ratio)
            actual_gross = float(source["next_day_gross_usd"])
            residual = math.log(actual_gross) - math.log(predicted_gross)
            row = {
                "model": model_name,
                "train_through_year": train_through_year,
                "test_start_year": test_start_year,
                "test_end_year": test_end_year,
                "train_rows": len(train_rows),
                "test_rows": len(test_rows),
                "movie_id": source["movie_id"],
                "title": source["title"],
                "release_date": source["release_date"],
                "box_office_date": source["box_office_date"],
                "movie_time_day": source["movie_time_day"],
                "release_age_bucket": source["release_age_bucket"],
                "opening_day_gross_quartile": source["opening_day_gross_quartile"],
                "opening_day_gross_quartile_label": source["opening_day_gross_quartile_label"],
                "current_day_gross_usd": format_number(current_gross),
                "actual_next_day_gross_usd": format_number(actual_gross),
                "predicted_next_day_gross_usd": format_number(predicted_gross),
                "actual_log_next_day_over_current_day": format_number(source["log_next_day_over_current_day"]),
                "predicted_log_next_day_over_current_day": format_number(pred_log_ratio),
                "residual_log_gross": format_number(residual),
                "absolute_percentage_error": format_number(abs(predicted_gross - actual_gross) / actual_gross),
            }
            prediction_rows.append(row)
            model_predictions.append(row)

        for metric in summarize_next_day_metrics(model_name, model_predictions):
            metric["train_through_year"] = train_through_year
            metric["test_start_year"] = test_start_year
            metric["test_end_year"] = test_end_year
            metric["train_rows"] = len(train_rows)
            metric["test_rows"] = len(test_rows)
            metric_rows.append(metric)
    return prediction_rows, metric_rows


def opening_weekend_time_feature_row(
    sample: Sample,
    movie: Movie,
    *,
    day: int,
    quartile: int,
) -> dict[str, object]:
    values = sample.predictors[movie.movie_id][day]
    q4_flag = 1.0 if quartile == 4 else 0.0
    log1p_v = math.log1p(max(0.0, values["V"]))
    log1p_u = math.log1p(max(0.0, values["U"]))
    log1p_r = math.log1p(max(0.0, values["R"]))
    log1p_e = math.log1p(max(0.0, values["E"]))
    return {
        "movie_id": movie.movie_id,
        "title": movie.title,
        "release_date": movie.release_date,
        "release_year": release_year(movie),
        "day": day,
        "opening_weekend_revenue_usd": movie.revenue,
        "log_opening_weekend_revenue": math.log(movie.revenue),
        "opening_day_gross_usd": movie.opening_day_gross or 0.0,
        "opening_theaters": movie.theaters,
        "opening_day_gross_quartile": quartile,
        "opening_day_gross_quartile_label": f"Q{quartile}",
        "V": values["V"],
        "U": values["U"],
        "R": values["R"],
        "E": values["E"],
        "T": movie.theaters,
        "log1p_V": log1p_v,
        "log1p_U": log1p_u,
        "log1p_R": log1p_r,
        "log1p_E": log1p_e,
        "log1p_T": math.log1p(max(0.0, movie.theaters)),
        "q4_flag": q4_flag,
        "q4_x_log1p_V": q4_flag * log1p_v,
        "q4_x_log1p_U": q4_flag * log1p_u,
        "q4_x_log1p_E": q4_flag * log1p_e,
    }


def opening_weekend_time_validation_rows(
    sample: Sample,
    *,
    quartiles: dict[str, int],
    days: list[int],
    train_through_year: int,
    test_start_year: int,
    test_end_year: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for day in days:
        rows = [
            opening_weekend_time_feature_row(
                sample,
                movie,
                day=day,
                quartile=quartiles[movie.movie_id],
            )
            for movie in sample.movies
            if day in sample.predictors.get(movie.movie_id, {}) and movie.revenue > 0
        ]
        train_rows = [row for row in rows if int(row["release_year"]) <= train_through_year]
        test_rows = [
            row
            for row in rows
            if test_start_year <= int(row["release_year"]) <= test_end_year
        ]
        if not train_rows or not test_rows:
            continue
        for model_name, target_scale, features in OPENING_WEEKEND_TIME_MODEL_SETS:
            target = "opening_weekend_revenue_usd" if target_scale == "raw" else "log_opening_weekend_revenue"
            predictions = linear_regression_predict(
                numeric_feature_matrix(train_rows, features),
                [float(row[target]) for row in train_rows],
                numeric_feature_matrix(test_rows, features),
            )
            model_predictions: list[dict[str, object]] = []
            for source, prediction in zip(test_rows, predictions):
                actual_gross = float(source["opening_weekend_revenue_usd"])
                predicted_gross = prediction if target_scale == "raw" else math.exp(prediction)
                predicted_gross = max(1.0, predicted_gross)
                actual_log = math.log(actual_gross)
                predicted_log = math.log(predicted_gross)
                row = {
                    "model": model_name,
                    "day": day,
                    "target_scale": target_scale,
                    "train_through_year": train_through_year,
                    "test_start_year": test_start_year,
                    "test_end_year": test_end_year,
                    "train_movies": len(train_rows),
                    "test_movies": len(test_rows),
                    "movie_id": source["movie_id"],
                    "title": source["title"],
                    "release_date": source["release_date"],
                    "opening_day_gross_quartile": source["opening_day_gross_quartile"],
                    "opening_day_gross_quartile_label": source["opening_day_gross_quartile_label"],
                    "actual_opening_weekend_revenue_usd": format_number(actual_gross),
                    "predicted_opening_weekend_revenue_usd": format_number(predicted_gross),
                    "actual_log_opening_weekend_revenue": format_number(actual_log),
                    "predicted_log_opening_weekend_revenue": format_number(predicted_log),
                    "residual_log_revenue": format_number(actual_log - predicted_log),
                    "absolute_percentage_error": format_number(abs(predicted_gross - actual_gross) / actual_gross),
                }
                prediction_rows.append(row)
                model_predictions.append(row)
            metric_rows.extend(
                summarize_opening_weekend_time_metrics(
                    model_name,
                    day,
                    target_scale,
                    model_predictions,
                    train_through_year=train_through_year,
                    test_start_year=test_start_year,
                    test_end_year=test_end_year,
                    train_movies=len(train_rows),
                    test_movies=len(test_rows),
                )
            )
    return prediction_rows, metric_rows


def summarize_opening_weekend_time_metrics(
    model_name: str,
    day: int,
    target_scale: str,
    prediction_rows: list[dict[str, object]],
    *,
    train_through_year: int,
    test_start_year: int,
    test_end_year: int,
    train_movies: int,
    test_movies: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {("overall", "overall"): prediction_rows}
    for row in prediction_rows:
        grouped.setdefault(("quartile", str(row["opening_day_gross_quartile_label"])), []).append(row)

    rows: list[dict[str, object]] = []
    for (segment_type, segment), group in sorted(grouped.items(), key=lambda item: item[0]):
        actual_logs = [float(row["actual_log_opening_weekend_revenue"]) for row in group]
        predicted_logs = [float(row["predicted_log_opening_weekend_revenue"]) for row in group]
        actual_gross = [float(row["actual_opening_weekend_revenue_usd"]) for row in group]
        predicted_gross = [float(row["predicted_opening_weekend_revenue_usd"]) for row in group]
        log_errors = [actual - pred for actual, pred in zip(actual_logs, predicted_logs)]
        apes = [float(row["absolute_percentage_error"]) for row in group]
        gross_errors = [actual - pred for actual, pred in zip(actual_gross, predicted_gross)]
        rows.append(
            {
                "model": model_name,
                "day": day,
                "target_scale": target_scale,
                "train_through_year": train_through_year,
                "test_start_year": test_start_year,
                "test_end_year": test_end_year,
                "train_movies": train_movies,
                "test_movies": test_movies,
                "segment_type": segment_type,
                "segment": segment,
                "rows": len(group),
                "r2_log_revenue": format_number(r2_score(actual_logs, predicted_logs) if len(group) >= 2 else None),
                "rmse_log_revenue": format_number(math.sqrt(mean(error * error for error in log_errors))) if log_errors else "",
                "mae_log_revenue": format_number(mean(abs(error) for error in log_errors)) if log_errors else "",
                "r2_gross": format_number(r2_score(actual_gross, predicted_gross) if len(group) >= 2 else None),
                "rmse_gross": format_number(math.sqrt(mean(error * error for error in gross_errors))) if gross_errors else "",
                "mae_gross": format_number(mean(abs(error) for error in gross_errors)) if gross_errors else "",
                "mape_gross": format_number(mean(apes)) if apes else "",
                "mean_actual_gross": format_number(mean(actual_gross)) if actual_gross else "",
                "mean_predicted_gross": format_number(mean(predicted_gross)) if predicted_gross else "",
            }
        )
    return rows


def summarize_next_day_metrics(
    model_name: str,
    prediction_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {("overall", "overall"): prediction_rows}
    for row in prediction_rows:
        grouped.setdefault(("quartile", str(row["opening_day_gross_quartile_label"])), []).append(row)
        grouped.setdefault(("release_age_bucket", str(row["release_age_bucket"])), []).append(row)

    rows: list[dict[str, object]] = []
    for (segment_type, segment), group in sorted(grouped.items(), key=lambda item: item[0]):
        actual_logs = [float(row["actual_log_next_day_over_current_day"]) for row in group]
        predicted_logs = [float(row["predicted_log_next_day_over_current_day"]) for row in group]
        actual_gross = [float(row["actual_next_day_gross_usd"]) for row in group]
        predicted_gross = [float(row["predicted_next_day_gross_usd"]) for row in group]
        residuals = [float(row["residual_log_gross"]) for row in group]
        apes = [float(row["absolute_percentage_error"]) for row in group]
        errors = [actual - pred for actual, pred in zip(actual_logs, predicted_logs)]
        rows.append(
            {
                "model": model_name,
                "segment_type": segment_type,
                "segment": segment,
                "rows": len(group),
                "r2_log_ratio": format_number(r2_score(actual_logs, predicted_logs) if len(group) >= 2 else None),
                "rmse_log_ratio": format_number(math.sqrt(mean(error * error for error in errors))) if errors else "",
                "mae_log_ratio": format_number(mean(abs(error) for error in errors)) if errors else "",
                "mape_gross": format_number(mean(apes)) if apes else "",
                "mean_residual_log_gross": format_number(mean(residuals)) if residuals else "",
                "mean_actual_next_day_gross_usd": format_number(mean(actual_gross)) if actual_gross else "",
                "mean_predicted_next_day_gross_usd": format_number(mean(predicted_gross)) if predicted_gross else "",
            }
        )
    return rows


def write_analysis_outputs(
    *,
    out_dir: Path,
    sample: Sample,
    correlation_rows: list[dict[str, object]],
    cv_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
    prefix: str,
    folds: int,
    day_start: int,
    day_end: int,
) -> None:
    write_csv(out_dir / f"{prefix}_correlations.csv", correlation_rows, ["day", "V", "U", "R", "E", "T"])
    write_csv(
        out_dir / f"{prefix}_r2_cross_validated.csv",
        cv_rows,
        ["day", "model", "features", "pooled_r2", "mean_fold_r2"],
    )
    write_csv(
        out_dir / f"{prefix}_predictions_minus30.csv",
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
        out_dir / f"{prefix}_figure2_correlations.svg",
        "Temporal correlation with opening-weekend revenue",
        corr_series,
        "movie time day",
        "Pearson r",
        x_range=(day_start, day_end),
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
        out_dir / f"{prefix}_figure3_cv_r2.svg",
        f"{folds}-fold cross-validated R^2 on the {len(sample.movies)}-movie sample",
        cv_series,
        "movie time day",
        "pooled cross-validated R^2",
        x_range=(day_start, day_end),
        y_range=(0.0, 0.9),
    )

    scatter_points = [
        (float(row["actual_revenue"]), float(row["predicted_revenue"]), str(row["group"]))
        for row in prediction_rows
    ]
    write_scatter_svg(
        out_dir / f"{prefix}_figure5_actual_vs_predicted_minus30.svg",
        "Actual vs predicted revenue at t = -30 days",
        scatter_points,
        "actual first-weekend revenue (USD, log scale)",
        "predicted first-weekend revenue (USD, log scale)",
    )

    write_histogram_svg(out_dir / f"{prefix}_figure1_summary_histograms.svg", sample)


def sample_summary_rows(
    sample: Sample,
    *,
    language: str,
    day_start: int,
    day_end: int,
    release_year: int | None,
    min_release_year: int | None,
    min_opening_theaters: int | None,
    min_opening_day_gross: int | None,
) -> list[dict[str, object]]:
    opening_dates = [movie.release_date for movie in sample.movies]
    revenues = [movie.revenue for movie in sample.movies]
    theaters = [movie.theaters for movie in sample.movies]
    return [
        {
            "sample": sample.name,
            "movies": len(sample.movies),
            "language": language,
            "day_start": day_start,
            "day_end": day_end,
            "release_year_filter": release_year if release_year is not None else "",
            "min_release_year_filter": min_release_year if min_release_year is not None else "",
            "min_opening_theaters_filter": min_opening_theaters if min_opening_theaters is not None else "",
            "min_opening_day_gross_filter": min_opening_day_gross if min_opening_day_gross is not None else "",
            "first_opening_date": min(opening_dates) if opening_dates else "",
            "last_opening_date": max(opening_dates) if opening_dates else "",
            "total_opening_weekend_revenue_usd": format_number(sum(revenues)) if revenues else "",
            "mean_opening_weekend_revenue_usd": format_number(mean(revenues)) if revenues else "",
            "mean_opening_day_gross_usd": format_number(mean(movie.opening_day_gross or 0.0 for movie in sample.movies))
            if sample.movies
            else "",
            "mean_opening_theaters": format_number(mean(theaters)) if theaters else "",
        }
    ]


QUARTILE_FIELDNAMES = [
    "movie_id",
    "title",
    "release_date",
    "quartile_basis",
    "opening_day_gross_usd",
    "opening_weekend_revenue_usd",
    "opening_theaters",
    "opening_day_gross_quartile",
    "opening_day_gross_quartile_label",
]

WIKI_QUARTILE_SUMMARY_FIELDNAMES = [
    "day",
    "opening_day_gross_quartile",
    "opening_day_gross_quartile_label",
    "movies",
    "mean_V",
    "median_V",
    "mean_U",
    "median_U",
    "mean_R",
    "median_R",
    "mean_E",
    "median_E",
    "mean_T",
    "mean_opening_day_gross_usd",
    "mean_opening_weekend_revenue_usd",
]

QUARTILE_MODEL_METRIC_FIELDNAMES = [
    "day",
    "opening_day_gross_quartile",
    "opening_day_gross_quartile_label",
    "movies",
    "v_correlation",
    "all_predictors_pooled_r2",
    "all_predictors_mean_fold_r2",
]

NEXT_DAY_PANEL_FIELDNAMES = [
    "movie_id",
    "title",
    "release_date",
    "box_office_date",
    "movie_time_day",
    "release_age_bucket",
    "current_day_gross_usd",
    "next_day_gross_usd",
    "log_current_day_gross",
    "log_next_day_gross",
    "log_next_day_over_current_day",
    "next_day_over_current_day",
    "days_since_release",
    "day_of_week",
    "opening_day_gross_usd",
    "opening_weekend_revenue_usd",
    "opening_theaters",
    "opening_day_gross_quartile",
    "opening_day_gross_quartile_label",
    "V",
    "U",
    "R",
    "E",
    "V_delta_1",
    "V_delta_3",
    "V_delta_7",
    "E_delta_1",
    "E_delta_3",
    "log1p_V",
    "log1p_U",
    "log1p_R",
    "log1p_E",
    "log1p_V_delta_1",
    "log1p_V_delta_3",
    "log1p_E_delta_3",
    "quartile_x_log1p_V_delta_3",
]

NEXT_DAY_METRIC_FIELDNAMES = [
    "model",
    "segment_type",
    "segment",
    "rows",
    "r2_log_ratio",
    "rmse_log_ratio",
    "mae_log_ratio",
    "mape_gross",
    "mean_residual_log_gross",
    "mean_actual_next_day_gross_usd",
    "mean_predicted_next_day_gross_usd",
]

NEXT_DAY_PREDICTION_FIELDNAMES = [
    "model",
    "movie_id",
    "title",
    "box_office_date",
    "movie_time_day",
    "release_age_bucket",
    "opening_day_gross_quartile",
    "opening_day_gross_quartile_label",
    "current_day_gross_usd",
    "actual_next_day_gross_usd",
    "predicted_next_day_gross_usd",
    "actual_log_next_day_over_current_day",
    "predicted_log_next_day_over_current_day",
    "residual_log_gross",
    "absolute_percentage_error",
]

TIME_VALIDATION_THRESHOLD_FIELDNAMES = [
    "quartile_basis",
    "train_through_year",
    "opening_day_gross_quartile",
    "opening_day_gross_quartile_label",
    "lower_bound_usd",
    "upper_bound_usd",
]

TIME_VALIDATION_METRIC_FIELDNAMES = [
    "model",
    "train_through_year",
    "test_start_year",
    "test_end_year",
    "train_rows",
    "test_rows",
    "segment_type",
    "segment",
    "rows",
    "r2_log_ratio",
    "rmse_log_ratio",
    "mae_log_ratio",
    "mape_gross",
    "mean_residual_log_gross",
    "mean_actual_next_day_gross_usd",
    "mean_predicted_next_day_gross_usd",
]

TIME_VALIDATION_PREDICTION_FIELDNAMES = [
    "model",
    "train_through_year",
    "test_start_year",
    "test_end_year",
    "train_rows",
    "test_rows",
    "movie_id",
    "title",
    "release_date",
    "box_office_date",
    "movie_time_day",
    "release_age_bucket",
    "opening_day_gross_quartile",
    "opening_day_gross_quartile_label",
    "current_day_gross_usd",
    "actual_next_day_gross_usd",
    "predicted_next_day_gross_usd",
    "actual_log_next_day_over_current_day",
    "predicted_log_next_day_over_current_day",
    "residual_log_gross",
    "absolute_percentage_error",
]

OPENING_WEEKEND_TIME_VALIDATION_METRIC_FIELDNAMES = [
    "model",
    "day",
    "target_scale",
    "train_through_year",
    "test_start_year",
    "test_end_year",
    "train_movies",
    "test_movies",
    "segment_type",
    "segment",
    "rows",
    "r2_log_revenue",
    "rmse_log_revenue",
    "mae_log_revenue",
    "r2_gross",
    "rmse_gross",
    "mae_gross",
    "mape_gross",
    "mean_actual_gross",
    "mean_predicted_gross",
]

OPENING_WEEKEND_TIME_VALIDATION_PREDICTION_FIELDNAMES = [
    "model",
    "day",
    "target_scale",
    "train_through_year",
    "test_start_year",
    "test_end_year",
    "train_movies",
    "test_movies",
    "movie_id",
    "title",
    "release_date",
    "opening_day_gross_quartile",
    "opening_day_gross_quartile_label",
    "actual_opening_weekend_revenue_usd",
    "predicted_opening_weekend_revenue_usd",
    "actual_log_opening_weekend_revenue",
    "predicted_log_opening_weekend_revenue",
    "residual_log_revenue",
    "absolute_percentage_error",
]


def write_fresh_extension_outputs(
    *,
    out_dir: Path,
    sample: Sample,
    quartile_basis: str,
    quartile_days: list[int],
    folds: int,
    seed: int,
    day_start: int,
    day_end: int,
    daily_actuals: dict[str, dict[int, float]] | None = None,
    time_validation: bool = False,
    opening_weekend_time_validation: bool = False,
    train_through_year: int = 2024,
    test_start_year: int = 2025,
    test_end_year: int = 2026,
) -> None:
    quartiles = assign_opening_day_quartiles(sample, basis=quartile_basis)
    quartile_rows = opening_day_quartile_rows(sample, quartiles, basis=quartile_basis)
    summary_rows = build_wiki_quartile_summary_rows(sample, quartiles, days=quartile_days)
    metric_rows = build_quartile_model_metric_rows(sample, quartiles, folds=folds, seed=seed)

    write_csv(out_dir / "fresh_opening_day_quartiles.csv", quartile_rows, QUARTILE_FIELDNAMES)
    write_csv(out_dir / "fresh_wiki_quartile_summary.csv", summary_rows, WIKI_QUARTILE_SUMMARY_FIELDNAMES)
    write_csv(out_dir / "fresh_quartile_model_metrics.csv", metric_rows, QUARTILE_MODEL_METRIC_FIELDNAMES)

    write_opening_day_quartile_distribution_svg(
        out_dir / "fresh_figure_opening_day_quartile_distribution.svg",
        quartile_rows,
    )
    write_wiki_features_by_quartile_svg(
        out_dir / "fresh_figure_wiki_features_by_quartile.svg",
        summary_rows,
    )
    write_quartile_line_svg(
        out_dir / "fresh_figure_correlations_by_quartile.svg",
        metric_rows,
        title="Views correlation with opening-weekend revenue by quartile",
        metric_column="v_correlation",
        y_label="Pearson r",
        x_range=(day_start, day_end),
    )
    write_quartile_line_svg(
        out_dir / "fresh_figure_cv_r2_by_quartile.svg",
        metric_rows,
        title="All-predictor cross-validated R^2 by opening-day quartile",
        metric_column="all_predictors_pooled_r2",
        y_label="pooled cross-validated R^2",
        x_range=(day_start, day_end),
    )

    if opening_weekend_time_validation:
        thresholds = fixed_quartile_thresholds(
            sample,
            basis=quartile_basis,
            train_through_year=train_through_year,
        )
        if len(thresholds) == 3:
            fixed_quartiles = assign_opening_day_quartiles_from_thresholds(
                sample,
                thresholds,
                basis=quartile_basis,
            )
            ow_prediction_rows, ow_metric_rows = opening_weekend_time_validation_rows(
                sample,
                quartiles=fixed_quartiles,
                days=quartile_days,
                train_through_year=train_through_year,
                test_start_year=test_start_year,
                test_end_year=test_end_year,
            )
            write_csv(
                out_dir / "fresh_opening_weekend_time_validation_metrics.csv",
                ow_metric_rows,
                OPENING_WEEKEND_TIME_VALIDATION_METRIC_FIELDNAMES,
            )
            write_csv(
                out_dir / "fresh_opening_weekend_time_validation_predictions.csv",
                ow_prediction_rows,
                OPENING_WEEKEND_TIME_VALIDATION_PREDICTION_FIELDNAMES,
            )
        else:
            write_csv(
                out_dir / "fresh_opening_weekend_time_validation_metrics.csv",
                [],
                OPENING_WEEKEND_TIME_VALIDATION_METRIC_FIELDNAMES,
            )
            write_csv(
                out_dir / "fresh_opening_weekend_time_validation_predictions.csv",
                [],
                OPENING_WEEKEND_TIME_VALIDATION_PREDICTION_FIELDNAMES,
            )

    if daily_actuals is None:
        return

    panel_rows = build_next_day_panel_rows(sample, daily_actuals, quartiles)
    prediction_rows, next_day_metric_rows = next_day_prediction_and_metric_rows(
        panel_rows,
        folds=folds,
        seed=seed,
    )
    write_csv(out_dir / "fresh_next_day_panel.csv", panel_rows, NEXT_DAY_PANEL_FIELDNAMES)
    write_csv(out_dir / "fresh_next_day_model_metrics.csv", next_day_metric_rows, NEXT_DAY_METRIC_FIELDNAMES)
    write_csv(out_dir / "fresh_next_day_predictions.csv", prediction_rows, NEXT_DAY_PREDICTION_FIELDNAMES)
    write_next_day_model_lift_svg(out_dir / "fresh_figure_next_day_model_lift.svg", next_day_metric_rows)
    write_next_day_error_by_release_age_svg(
        out_dir / "fresh_figure_next_day_error_by_release_age.svg",
        next_day_metric_rows,
    )
    write_wiki_momentum_scatter_svg(
        out_dir / "fresh_figure_wiki_momentum_vs_next_day_ratio.svg",
        panel_rows,
    )
    write_prediction_residuals_by_quartile_svg(
        out_dir / "fresh_figure_prediction_residuals_by_quartile.svg",
        next_day_metric_rows,
    )

    if not time_validation:
        return

    thresholds = fixed_quartile_thresholds(
        sample,
        basis=quartile_basis,
        train_through_year=train_through_year,
    )
    if len(thresholds) != 3:
        write_csv(out_dir / "fresh_time_validation_bucket_thresholds.csv", [], TIME_VALIDATION_THRESHOLD_FIELDNAMES)
        write_csv(out_dir / "fresh_next_day_time_validation_metrics.csv", [], TIME_VALIDATION_METRIC_FIELDNAMES)
        write_csv(out_dir / "fresh_next_day_time_validation_predictions.csv", [], TIME_VALIDATION_PREDICTION_FIELDNAMES)
        return

    fixed_quartiles = assign_opening_day_quartiles_from_thresholds(
        sample,
        thresholds,
        basis=quartile_basis,
    )
    fixed_panel_rows = build_next_day_panel_rows(sample, daily_actuals, fixed_quartiles)
    time_prediction_rows, time_metric_rows = time_split_next_day_prediction_and_metric_rows(
        fixed_panel_rows,
        train_through_year=train_through_year,
        test_start_year=test_start_year,
        test_end_year=test_end_year,
    )
    write_csv(
        out_dir / "fresh_time_validation_bucket_thresholds.csv",
        fixed_quartile_threshold_rows(
            thresholds,
            basis=quartile_basis,
            train_through_year=train_through_year,
        ),
        TIME_VALIDATION_THRESHOLD_FIELDNAMES,
    )
    write_csv(
        out_dir / "fresh_next_day_time_validation_metrics.csv",
        time_metric_rows,
        TIME_VALIDATION_METRIC_FIELDNAMES,
    )
    write_csv(
        out_dir / "fresh_next_day_time_validation_predictions.csv",
        time_prediction_rows,
        TIME_VALIDATION_PREDICTION_FIELDNAMES,
    )


def run_fresh_database(args: argparse.Namespace) -> int:
    if args.folds < 2:
        raise SystemExit("--folds must be at least 2")
    if args.release_year is not None and args.min_release_year is not None:
        raise SystemExit("--release-year and --min-release-year cannot be combined")
    if args.min_opening_theaters is not None and args.min_opening_theaters < 0:
        raise SystemExit("--min-opening-theaters must be non-negative")
    if args.min_opening_day_gross is not None and args.min_opening_day_gross < 0:
        raise SystemExit("--min-opening-day-gross must be non-negative")
    if args.next_day_time_validation and not args.next_day_analysis:
        args.next_day_analysis = True
    if args.test_end_year < args.test_start_year:
        raise SystemExit("--test-end-year must be greater than or equal to --test-start-year")
    if args.plot_format != "svg":
        raise SystemExit("Only --plot-format svg is currently supported")

    out_dir = args.out or DEFAULT_FRESH_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = connect_database(args.database_url)
    daily_actuals: dict[str, dict[int, float]] | None = None
    try:
        sample, coverage_rows = load_fresh_sample(
            conn,
            language=args.language,
            day_start=args.day_start,
            day_end=args.day_end,
            release_year=args.release_year,
            min_release_year=args.min_release_year,
            min_opening_theaters=args.min_opening_theaters,
            min_opening_day_gross=args.min_opening_day_gross,
        )
        if args.next_day_analysis:
            daily_actuals = load_daily_actuals_by_movie(conn, sample)
    finally:
        conn.close()

    if not sample.movies:
        write_csv(
            out_dir / "fresh_coverage_diagnostics.csv",
            coverage_rows,
            [
                "movie_id",
                "title",
                "release_year",
                "release_run_id",
                "opening_date",
                "opening_theaters",
                "opening_day_gross_usd",
                "opening_weekend_revenue_usd",
                "language",
                "wiki_page_id",
                "wiki_title",
                "match_status",
                "pageview_rows",
                "total_views",
                "human_revision_rows",
                "first_activity_day",
                "last_activity_day",
                "included",
            ],
        )
        raise SystemExit(
            "No fresh matched Wikipedia/The Numbers movies had activity in the requested analysis window."
        )

    correlation_rows = build_correlation_rows(sample)
    cv_rows = build_cv_rows(sample, args.folds, args.seed)
    prediction_rows = predict_minus_30_rows(sample, set(), default_group="fresh_postgres")

    write_csv(
        out_dir / "fresh_sample_summary.csv",
        sample_summary_rows(
            sample,
            language=args.language,
            day_start=args.day_start,
            day_end=args.day_end,
            release_year=args.release_year,
            min_release_year=args.min_release_year,
            min_opening_theaters=args.min_opening_theaters,
            min_opening_day_gross=args.min_opening_day_gross,
        ),
        [
            "sample",
            "movies",
            "language",
            "day_start",
            "day_end",
            "release_year_filter",
            "min_release_year_filter",
            "min_opening_theaters_filter",
            "min_opening_day_gross_filter",
            "first_opening_date",
            "last_opening_date",
            "total_opening_weekend_revenue_usd",
            "mean_opening_weekend_revenue_usd",
            "mean_opening_day_gross_usd",
            "mean_opening_theaters",
        ],
    )
    write_csv(
        out_dir / "fresh_coverage_diagnostics.csv",
        coverage_rows,
        [
            "movie_id",
            "title",
            "release_year",
            "release_run_id",
            "opening_date",
            "opening_theaters",
            "opening_day_gross_usd",
            "opening_weekend_revenue_usd",
            "language",
            "wiki_page_id",
            "wiki_title",
            "match_status",
            "pageview_rows",
            "total_views",
            "human_revision_rows",
            "first_activity_day",
            "last_activity_day",
            "included",
        ],
    )
    write_analysis_outputs(
        out_dir=out_dir,
        sample=sample,
        correlation_rows=correlation_rows,
        cv_rows=cv_rows,
        prediction_rows=prediction_rows,
        prefix="fresh",
        folds=args.folds,
        day_start=args.day_start,
        day_end=args.day_end,
    )
    if args.quartile_results or args.next_day_analysis or args.opening_weekend_time_validation:
        write_fresh_extension_outputs(
            out_dir=out_dir,
            sample=sample,
            quartile_basis=args.quartile_basis,
            quartile_days=parse_day_list(args.quartile_days),
            folds=args.folds,
            seed=args.seed,
            day_start=args.day_start,
            day_end=args.day_end,
            daily_actuals=daily_actuals,
            time_validation=args.next_day_time_validation,
            opening_weekend_time_validation=args.opening_weekend_time_validation,
            train_through_year=args.train_through_year,
            test_start_year=args.test_start_year,
            test_end_year=args.test_end_year,
        )

    print(f"Read {len(sample.movies)} fresh matched movies from Postgres.")
    print(f"Wrote fresh Wikipedia box-office artifacts to {out_dir}.")
    if args.quartile_results or args.next_day_analysis or args.opening_weekend_time_validation:
        print("Wrote expanded Wikipedia quartile/next-day artifacts.")
    print(
        "Key checks: "
        f"r(V, revenue) at t=-30 = {next(row['V'] for row in correlation_rows if row['day'] == -30)}, "
        f"CV R^2(all predictors) at t=-30 = {lookup_metric(cv_rows, -30, 'V+U+R+E+T', 'pooled_r2')}."
    )
    return 0


def run_paper_dataset(args: argparse.Namespace) -> int:
    if args.folds < 2:
        raise SystemExit("--folds must be at least 2")

    out_dir = args.out or DEFAULT_PAPER_OUT_DIR
    zip_path = ensure_dataset(args.zip, args.force_download)
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        sample_312 = read_sample(zf, "sample_of_312", "sample_of_312")
        sample_24 = read_sample(
            zf, "asur_huberman_sample_of_24", "asur_huberman_sample_of_24"
        )

    correlation_rows = build_correlation_rows(sample_312)
    cv_rows = build_cv_rows(sample_312, args.folds, args.seed)
    asur_rows = build_asur_rows(sample_24)
    overlap_titles = {movie.title for movie in sample_24.movies}
    prediction_rows = predict_minus_30_rows(sample_312, overlap_titles)

    write_csv(out_dir / "correlations_312.csv", correlation_rows, ["day", "V", "U", "R", "E", "T"])
    write_csv(
        out_dir / "r2_312_cross_validated.csv",
        cv_rows,
        ["day", "model", "features", "pooled_r2", "mean_fold_r2"],
    )
    write_csv(out_dir / "r2_asur_huberman_24.csv", asur_rows, ["day", "model", "r2"])
    write_csv(
        out_dir / "predictions_minus30_312.csv",
        prediction_rows,
        ["ID", "Title", "actual_revenue", "predicted_revenue", "group"],
    )

    write_analysis_outputs(
        out_dir=out_dir,
        sample=sample_312,
        correlation_rows=correlation_rows,
        cv_rows=cv_rows,
        prediction_rows=prediction_rows,
        prefix="figure",
        folds=args.folds,
        day_start=DEFAULT_DAY_START,
        day_end=DEFAULT_DAY_END,
    )
    (out_dir / "figure_figure1_summary_histograms.svg").replace(out_dir / "figure1_summary_histograms.svg")
    (out_dir / "figure_figure2_correlations.svg").replace(out_dir / "figure2_correlations.svg")
    (out_dir / "figure_figure3_cv_r2.svg").replace(out_dir / "figure3_cv_r2.svg")
    (out_dir / "figure_figure5_actual_vs_predicted_minus30.svg").replace(
        out_dir / "figure5_actual_vs_predicted_minus30.svg"
    )

    asur_series = {
        "Wikipedia model": [
            (int(row["day"]), float(row["r2"]) if row["r2"] != "" else None)
            for row in asur_rows
        ]
    }
    write_line_svg(
        out_dir / "figure4_asur_huberman_24.svg",
        "Wikipedia model on the 24-movie Asur-Huberman sample",
        asur_series,
        "movie time day",
        "in-sample R^2",
        x_range=(DEFAULT_DAY_START, DEFAULT_DAY_END),
        y_range=(0.0, 1.0),
        reference_lines=[(0.98, "Twitter reference 0.98", COLORS["Twitter reference"])],
    )

    print(f"Read {len(sample_312.movies)} movies from the main paper sample.")
    print(f"Read {len(sample_24.movies)} movies from the Asur-Huberman comparison sample.")
    print(f"Wrote CSV and SVG outputs to {out_dir}.")
    print(
        "Key checks: "
        f"r(V, revenue) at t=-30 = {next(row['V'] for row in correlation_rows if row['day'] == -30)}, "
        f"CV R^2(all predictors) at t=-30 = {lookup_metric(cv_rows, -30, 'V+U+R+E+T', 'pooled_r2')}, "
        f"Asur-Huberman R^2(all predictors) at t=-30 = {lookup_metric(asur_rows, -30, 'Wikipedia model', 'r2')}."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-dataset",
        action="store_true",
        help="Use the original Dataset S1 ZIP instead of fresh Postgres data.",
    )
    parser.add_argument(
        "--database-url",
        default=database_url_from_env(),
        help="PostgreSQL connection URL. Defaults to DATABASE_URL, POSTGRES_DSN, or .env.",
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=Path("data/external/supplementary/SupplementaryDataS1.zip"),
        help="Dataset S1 ZIP path for --paper-dataset.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for CSV and SVG files.",
    )
    parser.add_argument("--force-download", action="store_true", help="Download the dataset even if --zip already exists.")
    parser.add_argument("--language", default="en", help="Wikipedia language for fresh Postgres mode.")
    parser.add_argument("--day-start", type=int, default=DEFAULT_DAY_START, help="First movie-time day for fresh Postgres mode.")
    parser.add_argument("--day-end", type=int, default=DEFAULT_DAY_END, help="Last movie-time day for fresh Postgres mode.")
    parser.add_argument("--release-year", type=int, help="Optional release-year filter for fresh Postgres mode.")
    parser.add_argument("--min-release-year", type=int, help="Optional minimum release-year filter for fresh Postgres mode.")
    parser.add_argument("--min-opening-theaters", type=int, help="Optional opening-theater threshold for fresh Postgres mode.")
    parser.add_argument("--min-opening-day-gross", type=int, help="Optional opening-day gross threshold for fresh Postgres mode.")
    parser.add_argument("--folds", type=int, default=10, help="Number of cross-validation folds for Figure 3.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for the 10-fold split.")
    parser.add_argument("--quartile-results", action="store_true", help="Write opening-day gross quartile diagnostics for fresh mode.")
    parser.add_argument("--next-day-analysis", action="store_true", help="Write model-ready next-day Wikipedia feature diagnostics for fresh mode.")
    parser.add_argument(
        "--opening-weekend-time-validation",
        action="store_true",
        help="Fit opening-weekend models on older releases and score the requested holdout years.",
    )
    parser.add_argument(
        "--next-day-time-validation",
        action="store_true",
        help="Fit next-day models on older releases and score the requested holdout years.",
    )
    parser.add_argument("--train-through-year", type=int, default=2024, help="Last release year included in time-validation training.")
    parser.add_argument("--test-start-year", type=int, default=2025, help="First release year included in time-validation testing.")
    parser.add_argument("--test-end-year", type=int, default=2026, help="Last release year included in time-validation testing.")
    parser.add_argument(
        "--quartile-basis",
        default="opening_day_gross",
        choices=("opening_day_gross",),
        help="Basis used to assign opening-day performance quartiles.",
    )
    parser.add_argument(
        "--quartile-days",
        default=",".join(str(day) for day in DEFAULT_QUARTILE_DAYS),
        help="Comma-separated movie-time days for quartile feature summaries.",
    )
    parser.add_argument(
        "--plot-format",
        default="svg",
        choices=("svg",),
        help="Plot format for expanded fresh-mode diagnostics.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.paper_dataset:
        return run_paper_dataset(args)
    return run_fresh_database(args)


if __name__ == "__main__":
    raise SystemExit(main())
