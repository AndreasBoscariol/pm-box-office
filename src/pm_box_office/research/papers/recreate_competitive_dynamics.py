#!/usr/bin/env python3
"""Replicate Krider-Weinberg-style timing results from local box-office data.

This script uses the local The Numbers database rather than the original
Variety panel from Krider and Weinberg (1998). It estimates movie life-cycle
decay, opening strength, release timing, and simple timing regressions from the
available `daily_box_office` and `daily_chart_pages` tables.

Default run:
    python -m pm_box_office.research.papers.recreate_competitive_dynamics
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pm_box_office.db.connection import connect_database, database_url_from_env, table_names


DEFAULT_OUT_DIR = Path("results/papers/competitive_dynamics")
DEFAULT_ANALYSIS_START = dt.date(2022, 1, 1)
DEFAULT_GAP_THRESHOLD_DAYS = 14
DEFAULT_MIN_POSITIVE_WEEKS = 4
DEFAULT_WIDE_THEATER_THRESHOLD = 600
THEORY_SEASON_WEEKS = 10.0
THEORY_GRID_STEP = 0.1


@dataclass(frozen=True)
class DailyRow:
    release_run_id: int
    movie_id: int
    movie_url: str
    title: str
    release_year: int | None
    box_office_date: dt.date
    gross_usd: int
    theaters: int
    cumulative_gross_usd: int
    rank: str | None


@dataclass(frozen=True)
class ReleaseEpisode:
    episode_id: str
    release_run_id: int
    movie_id: int
    movie_url: str
    title: str
    release_year: int | None
    episode_index: int
    rows: list[DailyRow]

    @property
    def opening_date(self) -> dt.date:
        return self.rows[0].box_office_date

    @property
    def closing_date(self) -> dt.date:
        return self.rows[-1].box_office_date

    @property
    def max_theaters(self) -> int:
        return max(row.theaters for row in self.rows)

    @property
    def total_gross_usd(self) -> int:
        return sum(row.gross_usd for row in self.rows)

    @property
    def final_cumulative_gross_usd(self) -> int:
        return max(row.cumulative_gross_usd for row in self.rows)


@dataclass(frozen=True)
class WeeklyRow:
    episode_id: str
    week_index: int
    week_number: int
    week_start: dt.date
    week_end: dt.date
    weekly_gross_usd: int
    max_theaters: int
    days_observed: int


@dataclass(frozen=True)
class LifecycleEstimate:
    episode: ReleaseEpisode
    positive_weeks: int
    opening_7_day_gross_usd: int
    decay_beta: float | None
    half_life_weeks: float | None
    lifecycle_r2: float | None
    opening_share: float | None
    opening_attraction: float | None
    chart_coverage_days: int
    delay_weeks: float
    is_wide: bool
    season_label: str | None = None
    season_start: dt.date | None = None
    season_delay_weeks: float | None = None


def parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def date_or_none(value: str | None) -> dt.date | None:
    return parse_date(value) if value else None


def season_for_opening_date(opening_date: dt.date) -> tuple[str, dt.date] | None:
    summer_start = dt.date(opening_date.year, 5, 25)
    summer_end = dt.date(opening_date.year, 9, 5)
    if summer_start <= opening_date <= summer_end:
        return f"summer_{opening_date.year}", summer_start

    holiday_start = dt.date(opening_date.year, 11, 1)
    holiday_end = dt.date(opening_date.year + 1, 1, 5)
    if holiday_start <= opening_date <= holiday_end:
        return f"holiday_{opening_date.year}", holiday_start

    previous_holiday_start = dt.date(opening_date.year - 1, 11, 1)
    previous_holiday_end = dt.date(opening_date.year, 1, 5)
    if previous_holiday_start <= opening_date <= previous_holiday_end:
        return f"holiday_{opening_date.year - 1}", previous_holiday_start

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


def median(values: Iterable[float]) -> float | None:
    values = sorted(values)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean_output_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.iterdir():
        if path.name == "README.txt" or path.suffix in {".csv", ".svg"}:
            path.unlink()


def schema_summary(conn: Any) -> list[dict[str, object]]:
    rows = []
    for table in table_names(conn):
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        rows.append({"table_name": table, "row_count": count})
    return rows


def validate_database(conn: Any) -> None:
    required = {
        "movies",
        "release_runs",
        "daily_box_office",
        "daily_chart_pages",
    }
    existing = set(table_names(conn))
    missing = sorted(required - existing)
    if missing:
        raise SystemExit(f"Database is missing required tables: {', '.join(missing)}")


def default_analysis_window(conn: Any) -> tuple[dt.date, dt.date]:
    row = conn.execute(
        """
        SELECT GREATEST(
            COALESCE((SELECT MAX(box_office_date) FROM daily_box_office), '0001-01-01'),
            COALESCE((SELECT MAX(chart_date) FROM daily_chart_pages), '0001-01-01')
        )
        """
    ).fetchone()
    if not row or not row[0] or str(row[0]) == "0001-01-01":
        raise SystemExit("Database has no usable box-office dates.")
    return DEFAULT_ANALYSIS_START, parse_date(str(row[0]))


def load_daily_rows(conn: Any) -> list[DailyRow]:
    rows = []
    for row in conn.execute(
        """
        SELECT d.release_run_id,
               r.movie_id,
               m.movie_url,
               m.title,
               m.release_year,
               d.box_office_date,
               d.gross_usd,
               d.theaters,
               d.cumulative_gross_usd,
               d.rank
        FROM daily_box_office d
        JOIN release_runs r USING(release_run_id)
        JOIN movies m USING(movie_id)
        WHERE d.is_preview = 0
          AND d.gross_usd IS NOT NULL
          AND d.theaters IS NOT NULL
          AND d.cumulative_gross_usd IS NOT NULL
        ORDER BY d.release_run_id, d.box_office_date
        """
    ):
        rows.append(
            DailyRow(
                release_run_id=int(row[0]),
                movie_id=int(row[1]),
                movie_url=str(row[2]),
                title=str(row[3]),
                release_year=int(row[4]) if row[4] is not None else None,
                box_office_date=parse_date(str(row[5])),
                gross_usd=int(row[6]),
                theaters=int(row[7]),
                cumulative_gross_usd=int(row[8]),
                rank=str(row[9]) if row[9] is not None else None,
            )
        )
    return rows


def split_release_episodes(
    rows: list[DailyRow],
    *,
    gap_threshold_days: int,
) -> list[ReleaseEpisode]:
    grouped: dict[int, list[DailyRow]] = defaultdict(list)
    for row in rows:
        grouped[row.release_run_id].append(row)

    episodes = []
    for release_run_id, run_rows in sorted(grouped.items()):
        run_rows = sorted(run_rows, key=lambda row: row.box_office_date)
        current: list[DailyRow] = []
        episode_index = 1
        previous_date: dt.date | None = None
        for row in run_rows:
            if previous_date is not None and (row.box_office_date - previous_date).days > gap_threshold_days:
                if current:
                    episodes.append(make_episode(release_run_id, episode_index, current))
                    episode_index += 1
                current = []
            current.append(row)
            previous_date = row.box_office_date
        if current:
            episodes.append(make_episode(release_run_id, episode_index, current))
    return episodes


def filter_release_episodes(
    episodes: list[ReleaseEpisode],
    *,
    analysis_start: dt.date,
    analysis_end: dt.date,
) -> list[ReleaseEpisode]:
    return [
        episode
        for episode in episodes
        if analysis_start <= episode.opening_date <= analysis_end
    ]


def make_episode(release_run_id: int, episode_index: int, rows: list[DailyRow]) -> ReleaseEpisode:
    first = rows[0]
    return ReleaseEpisode(
        episode_id=f"{release_run_id}:{episode_index}",
        release_run_id=release_run_id,
        movie_id=first.movie_id,
        movie_url=first.movie_url,
        title=first.title,
        release_year=first.release_year,
        episode_index=episode_index,
        rows=rows,
    )


def aggregate_weekly(episode: ReleaseEpisode) -> list[WeeklyRow]:
    buckets: dict[int, list[DailyRow]] = defaultdict(list)
    opening = episode.opening_date
    for row in episode.rows:
        week_index = (row.box_office_date - opening).days // 7
        buckets[week_index].append(row)

    weekly = []
    for week_index in sorted(buckets):
        rows = buckets[week_index]
        weekly.append(
            WeeklyRow(
                episode_id=episode.episode_id,
                week_index=week_index,
                week_number=week_index + 1,
                week_start=min(row.box_office_date for row in rows),
                week_end=max(row.box_office_date for row in rows),
                weekly_gross_usd=sum(row.gross_usd for row in rows),
                max_theaters=max(row.theaters for row in rows),
                days_observed=len(rows),
            )
        )
    return weekly


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
    if not x or not y or len(x) != len(y):
        raise ValueError("OLS requires non-empty aligned x and y")
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
    f_stat = ((tss - sse) / max(1, width - 1)) / (sse / df) if sse and width > 1 else 0.0
    return beta, se, r2, f_stat


def estimate_lifecycle_from_weekly(
    weekly: list[WeeklyRow],
    *,
    min_positive_weeks: int,
) -> tuple[int, int, float | None, float | None, float | None]:
    positive = [row for row in weekly if row.weekly_gross_usd > 0]
    opening_gross = positive[0].weekly_gross_usd if positive else 0
    if len(positive) < min_positive_weeks:
        return len(positive), opening_gross, None, None, None

    x = [[1.0, float(row.week_index)] for row in positive]
    y = [math.log(row.weekly_gross_usd) for row in positive]
    beta, _, r2, _ = ols(x, y)
    decay_beta = -beta[1]
    if decay_beta <= 0:
        return len(positive), opening_gross, decay_beta, None, r2
    return len(positive), opening_gross, decay_beta, math.log(2.0) / decay_beta, r2


def load_chart_daily_totals(conn: Any) -> dict[dt.date, int]:
    totals = {}
    for row in conn.execute(
        """
        SELECT chart_date, SUM(gross_usd)
        FROM daily_chart_pages
        WHERE gross_usd IS NOT NULL
        GROUP BY chart_date
        """
    ):
        totals[parse_date(str(row[0]))] = int(row[1])
    return totals


def load_chart_movie_gross(conn: Any) -> dict[tuple[str, dt.date], int]:
    values = {}
    for row in conn.execute(
        """
        SELECT movie_url, chart_date, SUM(gross_usd)
        FROM daily_chart_pages
        WHERE gross_usd IS NOT NULL
        GROUP BY movie_url, chart_date
        """
    ):
        values[(str(row[0]), parse_date(str(row[1])))] = int(row[2])
    return values


def opening_share_for_episode(
    episode: ReleaseEpisode,
    chart_totals: dict[dt.date, int],
    chart_movie_gross: dict[tuple[str, dt.date], int],
) -> tuple[float | None, float | None, int]:
    numerator = 0
    denominator = 0
    coverage_days = 0
    for offset in range(7):
        day = episode.opening_date + dt.timedelta(days=offset)
        day_total = chart_totals.get(day)
        if not day_total:
            continue
        coverage_days += 1
        denominator += day_total
        numerator += chart_movie_gross.get((episode.movie_url, day), 0)
    if coverage_days < 3 or denominator <= 0 or numerator <= 0:
        return None, None, coverage_days
    share = numerator / denominator
    attraction = share / (1.0 - share) if 0.0 < share < 1.0 else None
    return share, attraction, coverage_days


def opening_weekend_gross_usd(episode: ReleaseEpisode) -> int:
    opening = episode.opening_date
    return sum(
        row.gross_usd
        for row in episode.rows
        if 0 <= (row.box_office_date - opening).days <= 2
    )


def estimate_lifecycles(
    episodes: list[ReleaseEpisode],
    conn: Any,
    *,
    analysis_start: dt.date,
    min_positive_weeks: int,
    wide_theater_threshold: int,
) -> tuple[list[LifecycleEstimate], dict[str, list[WeeklyRow]]]:
    chart_totals = load_chart_daily_totals(conn)
    chart_movie_gross = load_chart_movie_gross(conn)
    estimates = []
    weekly_by_episode = {}
    for episode in episodes:
        weekly = aggregate_weekly(episode)
        weekly_by_episode[episode.episode_id] = weekly
        positive_weeks, opening_gross, decay_beta, half_life, r2 = estimate_lifecycle_from_weekly(
            weekly,
            min_positive_weeks=min_positive_weeks,
        )
        opening_share, attraction, coverage_days = opening_share_for_episode(
            episode,
            chart_totals,
            chart_movie_gross,
        )
        season = season_for_opening_date(episode.opening_date)
        season_label = season[0] if season else None
        season_start = season[1] if season else None
        estimates.append(
            LifecycleEstimate(
                episode=episode,
                positive_weeks=positive_weeks,
                opening_7_day_gross_usd=opening_gross,
                decay_beta=decay_beta,
                half_life_weeks=half_life,
                lifecycle_r2=r2,
                opening_share=opening_share,
                opening_attraction=attraction,
                chart_coverage_days=coverage_days,
                delay_weeks=(episode.opening_date - analysis_start).days / 7.0,
                is_wide=episode.max_theaters >= wide_theater_threshold,
                season_label=season_label,
                season_start=season_start,
                season_delay_weeks=(episode.opening_date - season_start).days / 7.0
                if season_start
                else None,
            )
        )
    return estimates, weekly_by_episode


def regression_rows(estimates: list[LifecycleEstimate], analysis_start: dt.date, analysis_end: dt.date) -> list[dict[str, object]]:
    timing_sample = [
        estimate
        for estimate in estimates
        if estimate.is_wide
        and estimate.half_life_weeks is not None
        and estimate.season_delay_weeks is not None
        and analysis_start <= estimate.episode.opening_date <= analysis_end
    ]
    outputs: list[dict[str, object]] = []

    specs = [
        (
            "delay_on_opening_weekend_gross_millions_and_half_life",
            [
                ("intercept", lambda e: 1.0),
                ("opening_weekend_gross_millions", lambda e: opening_weekend_gross_usd(e.episode) / 1_000_000.0),
                ("half_life_weeks", lambda e: float(e.half_life_weeks)),
            ],
            timing_sample,
        ),
        (
            "delay_on_opening_gross_millions_and_half_life",
            [
                ("intercept", lambda e: 1.0),
                ("opening_7_day_gross_millions", lambda e: e.opening_7_day_gross_usd / 1_000_000.0),
                ("half_life_weeks", lambda e: float(e.half_life_weeks)),
            ],
            timing_sample,
        ),
        (
            "delay_on_opening_share_and_half_life",
            [
                ("intercept", lambda e: 1.0),
                ("opening_share", lambda e: float(e.opening_share)),
                ("half_life_weeks", lambda e: float(e.half_life_weeks)),
            ],
            [estimate for estimate in timing_sample if estimate.opening_share is not None],
        ),
    ]

    for model, terms, sample in specs:
        if len(sample) < len(terms) + 2:
            outputs.append(
                {
                    "model": model,
                    "term": "insufficient_sample",
                    "estimate": None,
                    "se": None,
                    "t_stat": None,
                    "n": len(sample),
                    "r2": None,
                    "f_stat": None,
                }
            )
            continue
        x = [[fn(estimate) for _, fn in terms] for estimate in sample]
        y = [float(estimate.season_delay_weeks) for estimate in sample]
        beta, se, r2, f_stat = ols(x, y)
        for (term, _), estimate, term_se in zip(terms, beta, se):
            outputs.append(
                {
                    "model": model,
                    "term": term,
                    "estimate": estimate,
                    "se": term_se,
                    "t_stat": estimate / term_se if term_se else None,
                    "n": len(sample),
                    "r2": r2,
                    "f_stat": f_stat,
                }
            )
    return outputs


def model_vs_observed_rows(
    estimates: list[LifecycleEstimate],
    analysis_start: dt.date,
    analysis_end: dt.date,
) -> list[dict[str, object]]:
    sample = [
        estimate
        for estimate in estimates
        if estimate.is_wide
        and estimate.half_life_weeks is not None
        and estimate.season_delay_weeks is not None
        and analysis_start <= estimate.episode.opening_date <= analysis_end
    ]
    if len(sample) < 2:
        return []

    log_gross = [math.log(max(1, estimate.opening_7_day_gross_usd)) for estimate in sample]
    half_lives = [float(estimate.half_life_weeks) for estimate in sample]
    gross_mean, gross_sd = mean(log_gross), sample_sd(log_gross) or 1.0
    half_mean, half_sd = mean(half_lives), sample_sd(half_lives) or 1.0

    strength: dict[str, float] = {}
    for estimate, gross_value, half_value in zip(sample, log_gross, half_lives):
        strength[estimate.episode.episode_id] = (
            (gross_value - gross_mean) / gross_sd
            + (half_value - half_mean) / half_sd
        )

    rows = []
    for i, left in enumerate(sample):
        for right in sample[i + 1 :]:
            if left.season_label != right.season_label:
                continue
            if left.season_delay_weeks == right.season_delay_weeks:
                continue
            left_strength = strength[left.episode.episode_id]
            right_strength = strength[right.episode.episode_id]
            if abs(left_strength - right_strength) < 1e-9:
                continue
            predicted = left if left_strength > right_strength else right
            observed = left if float(left.season_delay_weeks) < float(right.season_delay_weeks) else right
            rows.append(
                {
                    "season_label": left.season_label,
                    "left_episode_id": left.episode.episode_id,
                    "left_title": left.episode.title,
                    "left_opening_date": left.episode.opening_date.isoformat(),
                    "left_season_delay_weeks": left.season_delay_weeks,
                    "left_strength_score": left_strength,
                    "right_episode_id": right.episode.episode_id,
                    "right_title": right.episode.title,
                    "right_opening_date": right.episode.opening_date.isoformat(),
                    "right_season_delay_weeks": right.season_delay_weeks,
                    "right_strength_score": right_strength,
                    "predicted_first_episode_id": predicted.episode.episode_id,
                    "observed_first_episode_id": observed.episode.episode_id,
                    "prediction_correct": int(predicted.episode.episode_id == observed.episode.episode_id),
                    "opening_gap_days": abs((left.episode.opening_date - right.episode.opening_date).days),
                }
            )
    return rows


def standardized_curve_rows(
    estimates: list[LifecycleEstimate],
    weekly_by_episode: dict[str, list[WeeklyRow]],
    *,
    bin_width: float = 0.25,
    max_half_lives: float = 5.0,
) -> list[dict[str, object]]:
    bins: dict[float, list[float]] = defaultdict(list)
    for estimate in estimates:
        if estimate.half_life_weeks is None or estimate.opening_7_day_gross_usd <= 0:
            continue
        weekly = weekly_by_episode[estimate.episode.episode_id]
        for row in weekly:
            if row.weekly_gross_usd <= 0:
                continue
            x = row.week_index / estimate.half_life_weeks
            if x > max_half_lives:
                continue
            bucket = round(round(x / bin_width) * bin_width, 2)
            y = math.log(row.weekly_gross_usd / estimate.opening_7_day_gross_usd)
            bins[bucket].append(y)

    rows = []
    for bucket in sorted(bins):
        values = bins[bucket]
        rows.append(
            {
                "half_lives_after_opening": bucket,
                "mean_log_standardized_gross": mean(values),
                "ideal_exponential_log_decline": -math.log(2.0) * bucket,
                "movie_week_count": len(values),
            }
        )
    return rows


def release_episode_rows(estimates: list[LifecycleEstimate]) -> list[dict[str, object]]:
    rows = []
    for estimate in estimates:
        episode = estimate.episode
        rows.append(
            {
                "episode_id": episode.episode_id,
                "release_run_id": episode.release_run_id,
                "movie_id": episode.movie_id,
                "title": episode.title,
                "release_year": episode.release_year,
                "episode_index": episode.episode_index,
                "opening_date": episode.opening_date.isoformat(),
                "closing_date": episode.closing_date.isoformat(),
                "observed_days": len(episode.rows),
                "max_theaters": episode.max_theaters,
                "is_wide": int(estimate.is_wide),
                "total_observed_gross_usd": episode.total_gross_usd,
                "final_cumulative_gross_usd": episode.final_cumulative_gross_usd,
            }
        )
    return rows


def lifecycle_rows(estimates: list[LifecycleEstimate]) -> list[dict[str, object]]:
    rows = []
    for estimate in estimates:
        episode = estimate.episode
        rows.append(
            {
                "episode_id": episode.episode_id,
                "title": episode.title,
                "release_year": episode.release_year,
                "opening_date": episode.opening_date.isoformat(),
                "delay_weeks": estimate.delay_weeks,
                "season_label": estimate.season_label,
                "season_start": estimate.season_start.isoformat() if estimate.season_start else None,
                "season_delay_weeks": estimate.season_delay_weeks,
                "positive_weeks": estimate.positive_weeks,
                "opening_weekend_gross_usd": opening_weekend_gross_usd(episode),
                "opening_7_day_gross_usd": estimate.opening_7_day_gross_usd,
                "max_theaters": episode.max_theaters,
                "decay_beta": estimate.decay_beta,
                "half_life_weeks": estimate.half_life_weeks,
                "lifecycle_r2": estimate.lifecycle_r2,
                "opening_share": estimate.opening_share,
                "opening_attraction": estimate.opening_attraction,
                "chart_coverage_days": estimate.chart_coverage_days,
                "is_wide": int(estimate.is_wide),
            }
        )
    return rows


def opening_share_rows(estimates: list[LifecycleEstimate]) -> list[dict[str, object]]:
    return [
        {
            "episode_id": estimate.episode.episode_id,
            "title": estimate.episode.title,
            "opening_date": estimate.episode.opening_date.isoformat(),
            "opening_share": estimate.opening_share,
            "opening_attraction": estimate.opening_attraction,
            "chart_coverage_days": estimate.chart_coverage_days,
            "opening_7_day_gross_usd": estimate.opening_7_day_gross_usd,
        }
        for estimate in estimates
        if estimate.opening_share is not None
    ]


def regression_model_rows(
    rows: list[dict[str, object]],
    model: str,
) -> list[dict[str, object]]:
    return [row for row in rows if row["model"] == model]


def regression_value(
    rows: list[dict[str, object]],
    *,
    model: str,
    term: str,
    field: str,
) -> object | None:
    for row in rows:
        if row["model"] == model and row["term"] == term:
            return row[field]
    return None


def paper_vs_database_summary_rows(
    estimates: list[LifecycleEstimate],
    regression: list[dict[str, object]],
    model_rows: list[dict[str, object]],
    analysis_start: dt.date,
    analysis_end: dt.date,
) -> list[dict[str, object]]:
    timing_sample = [
        estimate
        for estimate in estimates
        if estimate.is_wide
        and estimate.half_life_weeks is not None
        and estimate.season_delay_weeks is not None
        and analysis_start <= estimate.episode.opening_date <= analysis_end
    ]
    opening_model = "delay_on_opening_weekend_gross_millions_and_half_life"
    share_model = "delay_on_opening_share_and_half_life"
    accuracy = None
    if model_rows:
        accuracy = sum(int(row["prediction_correct"]) for row in model_rows) / len(model_rows)
    rows = [
        {
            "result": "sample",
            "paper_1990_reported": "24 major summer movies, delay from May 25",
            "database_2022_plus": (
                f"{len(timing_sample)} wide 2022+ summer/holiday release episodes, "
                "delay normalized within season"
            ),
        },
        {
            "result": "opening_gross_t_stat",
            "paper_1990_reported": "-3.6, p < .005",
            "database_2022_plus": regression_value(
                regression,
                model=opening_model,
                term="opening_weekend_gross_millions",
                field="t_stat",
            ),
        },
        {
            "result": "half_life_t_stat_in_opening_gross_model",
            "paper_1990_reported": "-.47, p > .5",
            "database_2022_plus": regression_value(
                regression,
                model=opening_model,
                term="half_life_weeks",
                field="t_stat",
            ),
        },
        {
            "result": "opening_gross_model_f_stat",
            "paper_1990_reported": "7.1, p < .005",
            "database_2022_plus": regression_value(
                regression,
                model=opening_model,
                term="intercept",
                field="f_stat",
            ),
        },
        {
            "result": "opening_share_model_f_stat",
            "paper_1990_reported": "5.6, p < .02",
            "database_2022_plus": regression_value(
                regression,
                model=share_model,
                term="intercept",
                field="f_stat",
            ),
        },
        {
            "result": "pairwise_stronger_opens_earlier_accuracy",
            "paper_1990_reported": "not reported as pairwise accuracy",
            "database_2022_plus": accuracy,
        },
    ]
    return rows


@dataclass(frozen=True)
class TheoryMovie:
    opening_attraction: float
    half_life_weeks: float

    @property
    def decay_beta(self) -> float:
        return math.log(2.0) / self.half_life_weeks


@dataclass(frozen=True)
class DuopolyEquilibrium:
    movie1_opening_week: float
    movie2_opening_week: float
    classification: str


def theory_attraction(movie: TheoryMovie, t: float, opening_week: float) -> float:
    if t < opening_week:
        return 0.0
    return movie.opening_attraction * math.exp(-movie.decay_beta * (t - opening_week))


def duopoly_revenue(
    player: int,
    movie1: TheoryMovie,
    movie2: TheoryMovie,
    *,
    movie1_opening_week: float,
    movie2_opening_week: float,
    season_weeks: float = THEORY_SEASON_WEEKS,
    integration_steps: int = 160,
) -> float:
    start = movie1_opening_week if player == 1 else movie2_opening_week
    if start >= season_weeks:
        return 0.0
    dt_step = (season_weeks - start) / integration_steps
    total = 0.0
    for index in range(integration_steps):
        t = start + (index + 0.5) * dt_step
        a1 = theory_attraction(movie1, t, movie1_opening_week)
        a2 = theory_attraction(movie2, t, movie2_opening_week)
        denominator = 1.0 + a1 + a2
        attraction = a1 if player == 1 else a2
        total += attraction / denominator * dt_step
    return total


def candidate_opening_weeks(
    season_weeks: float = THEORY_SEASON_WEEKS,
    step: float = THEORY_GRID_STEP,
) -> list[float]:
    count = int(round(season_weeks / step))
    return [round(index * step, 10) for index in range(count + 1)]


def best_response_opening_week(
    player: int,
    movie1: TheoryMovie,
    movie2: TheoryMovie,
    *,
    other_opening_week: float,
    season_weeks: float = THEORY_SEASON_WEEKS,
    step: float = THEORY_GRID_STEP,
) -> float:
    best_week = 0.0
    best_revenue = -1.0
    for week in candidate_opening_weeks(season_weeks, step):
        revenue = duopoly_revenue(
            player,
            movie1,
            movie2,
            movie1_opening_week=week if player == 1 else other_opening_week,
            movie2_opening_week=other_opening_week if player == 1 else week,
            season_weeks=season_weeks,
        )
        if revenue > best_revenue + 1e-10:
            best_week = week
            best_revenue = revenue
    return best_week


def is_best_response(
    player: int,
    movie1: TheoryMovie,
    movie2: TheoryMovie,
    *,
    own_opening_week: float,
    other_opening_week: float,
    season_weeks: float = THEORY_SEASON_WEEKS,
    step: float = THEORY_GRID_STEP,
    tolerance: float = 1e-6,
) -> bool:
    actual = duopoly_revenue(
        player,
        movie1,
        movie2,
        movie1_opening_week=own_opening_week if player == 1 else other_opening_week,
        movie2_opening_week=other_opening_week if player == 1 else own_opening_week,
        season_weeks=season_weeks,
    )
    best_week = best_response_opening_week(
        player,
        movie1,
        movie2,
        other_opening_week=other_opening_week,
        season_weeks=season_weeks,
        step=step,
    )
    best = duopoly_revenue(
        player,
        movie1,
        movie2,
        movie1_opening_week=best_week if player == 1 else other_opening_week,
        movie2_opening_week=other_opening_week if player == 1 else best_week,
        season_weeks=season_weeks,
    )
    return actual >= best - tolerance


def classify_duopoly_equilibria(equilibria: list[tuple[float, float]]) -> str:
    if not equilibria:
        return "no_equilibrium_found"
    if len(equilibria) > 1:
        return "dual_equilibria"
    t1, t2 = equilibria[0]
    if abs(t1) < 1e-9 and abs(t2) < 1e-9:
        return "simultaneous_beginning"
    if abs(t1) < 1e-9 and t2 > 0:
        return "movie2_delays"
    if t1 > 0 and abs(t2) < 1e-9:
        return "movie1_delays"
    return "other"


def solve_duopoly_equilibria(
    movie1: TheoryMovie,
    movie2: TheoryMovie,
    *,
    season_weeks: float = THEORY_SEASON_WEEKS,
    step: float = THEORY_GRID_STEP,
) -> list[DuopolyEquilibrium]:
    candidates: list[tuple[float, float]] = []
    movie2_delay = best_response_opening_week(
        2,
        movie1,
        movie2,
        other_opening_week=0.0,
        season_weeks=season_weeks,
        step=step,
    )
    if is_best_response(
        1,
        movie1,
        movie2,
        own_opening_week=0.0,
        other_opening_week=movie2_delay,
        season_weeks=season_weeks,
        step=step,
    ):
        candidates.append((0.0, movie2_delay))

    movie1_delay = best_response_opening_week(
        1,
        movie1,
        movie2,
        other_opening_week=0.0,
        season_weeks=season_weeks,
        step=step,
    )
    if is_best_response(
        2,
        movie1,
        movie2,
        own_opening_week=0.0,
        other_opening_week=movie1_delay,
        season_weeks=season_weeks,
        step=step,
    ):
        candidates.append((movie1_delay, 0.0))

    deduped: list[tuple[float, float]] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    classification = classify_duopoly_equilibria(deduped)
    return [
        DuopolyEquilibrium(
            movie1_opening_week=movie1_week,
            movie2_opening_week=movie2_week,
            classification=classification,
        )
        for movie1_week, movie2_week in deduped
    ]


def primary_duopoly_equilibrium(
    movie1: TheoryMovie,
    movie2: TheoryMovie,
    *,
    season_weeks: float = THEORY_SEASON_WEEKS,
    step: float = THEORY_GRID_STEP,
) -> DuopolyEquilibrium:
    equilibria = solve_duopoly_equilibria(movie1, movie2, season_weeks=season_weeks, step=step)
    if equilibria:
        return equilibria[0]
    return DuopolyEquilibrium(0.0, 0.0, "no_equilibrium_found")


def figure4_theory_rows() -> list[dict[str, object]]:
    rows = []
    movie2 = TheoryMovie(0.5, 3.4)
    for index in range(5, 101, 5):
        attraction = index / 100.0
        movie1 = TheoryMovie(attraction, 3.4)
        equilibria = solve_duopoly_equilibria(movie1, movie2)
        if not equilibria:
            rows.append(
                {
                    "movie1_opening_attraction": attraction,
                    "movie1_equilibrium_week": None,
                    "movie2_equilibrium_week": None,
                    "classification": "no_equilibrium_found",
                }
            )
            continue
        for equilibrium in equilibria:
            rows.append(
                {
                    "movie1_opening_attraction": attraction,
                    "movie1_equilibrium_week": equilibrium.movie1_opening_week,
                    "movie2_equilibrium_week": equilibrium.movie2_opening_week,
                    "classification": equilibrium.classification,
                }
            )
    return rows


def figure5_theory_rows() -> list[dict[str, object]]:
    rows = []
    for half_life_index in range(10, 51, 2):
        half_life = half_life_index / 10.0
        for attraction_index in range(5, 101, 5):
            movie2_attraction = attraction_index / 100.0
            equilibrium = primary_duopoly_equilibrium(
                TheoryMovie(0.5, half_life),
                TheoryMovie(movie2_attraction, half_life),
            )
            rows.append(
                {
                    "half_life_weeks": half_life,
                    "movie2_opening_attraction": movie2_attraction,
                    "classification": equilibrium.classification,
                }
            )
    return rows


def figure8_theory_rows() -> list[dict[str, object]]:
    rows = []
    movie1 = TheoryMovie(0.5, 3.0)
    for half_life_index in range(10, 51, 2):
        half_life = half_life_index / 10.0
        for attraction_index in range(5, 101, 5):
            movie2_attraction = attraction_index / 100.0
            equilibrium = primary_duopoly_equilibrium(
                movie1,
                TheoryMovie(movie2_attraction, half_life),
            )
            rows.append(
                {
                    "movie2_half_life_weeks": half_life,
                    "movie2_opening_attraction": movie2_attraction,
                    "classification": equilibrium.classification,
                }
            )
    return rows


def paper_parameter_equilibrium_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    examples = [
        ("figure_6_small_large", TheoryMovie(0.13, 1.0), TheoryMovie(1.0, 1.0)),
        ("figure_7_small_large", TheoryMovie(0.67, 1.0), TheoryMovie(1.0, 1.0)),
        ("long_legs_identical", TheoryMovie(0.5, 4.0), TheoryMovie(0.5, 4.0)),
    ]
    for label, movie1, movie2 in examples:
        equilibria = solve_duopoly_equilibria(movie1, movie2)
        if not equilibria:
            rows.append(
                {
                    "scenario": label,
                    "movie": "duopoly",
                    "opening_attraction": None,
                    "half_life_weeks": None,
                    "recreated_equilibrium_opening_week": None,
                    "paper_reported_opening_week": None,
                    "classification": "no_equilibrium_found",
                }
            )
        for equilibrium in equilibria:
            rows.extend(
                [
                    {
                        "scenario": label,
                        "movie": "movie1",
                        "opening_attraction": movie1.opening_attraction,
                        "half_life_weeks": movie1.half_life_weeks,
                        "recreated_equilibrium_opening_week": equilibrium.movie1_opening_week,
                        "paper_reported_opening_week": None,
                        "classification": equilibrium.classification,
                    },
                    {
                        "scenario": label,
                        "movie": "movie2",
                        "opening_attraction": movie2.opening_attraction,
                        "half_life_weeks": movie2.half_life_weeks,
                        "recreated_equilibrium_opening_week": equilibrium.movie2_opening_week,
                        "paper_reported_opening_week": None,
                        "classification": equilibrium.classification,
                    },
                ]
            )

    table1_rows = [
        ("table1_group1_equal_short_half_lives", 0.7, 2.0, 0.0),
        ("table1_group1_equal_short_half_lives", 0.5, 2.0, 1.7),
        ("table1_group1_equal_short_half_lives", 0.3, 2.0, 3.7),
        ("table1_group2_different_half_lives", 0.5, 3.0, 0.0),
        ("table1_group2_different_half_lives", 0.5, 2.0, 1.2),
        ("table1_group2_different_half_lives", 0.5, 1.0, 5.4),
        ("table1_group3_long_half_lives", 0.5, 4.0, 0.0),
        ("table1_group3_long_half_lives", 0.5, 3.5, 0.0),
        ("table1_group3_long_half_lives", 0.5, 3.0, 0.9),
        ("table1_group4_equal_long_half_lives", 0.5, 4.0, 0.0),
        ("table1_group4_equal_long_half_lives", 0.5, 4.0, 0.0),
        ("table1_group4_equal_long_half_lives", 0.5, 4.0, 0.0),
    ]
    for index, (scenario, attraction, half_life, opening_week) in enumerate(table1_rows, start=1):
        rows.append(
            {
                "scenario": scenario,
                "movie": f"movie{index}",
                "opening_attraction": attraction,
                "half_life_weeks": half_life,
                "recreated_equilibrium_opening_week": None,
                "paper_reported_opening_week": opening_week,
                "classification": "paper_reported_three_firm_table",
            }
        )
    return rows


def summary_metrics(
    db_path: Path,
    analysis_start: dt.date,
    analysis_end: dt.date,
    estimates: list[LifecycleEstimate],
    regression: list[dict[str, object]],
    model_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    wide = [estimate for estimate in estimates if estimate.is_wide]
    lifecycle = [estimate for estimate in estimates if estimate.half_life_weeks is not None]
    timing = [
        estimate
        for estimate in lifecycle
        if estimate.is_wide
        and estimate.season_delay_weeks is not None
        and analysis_start <= estimate.episode.opening_date <= analysis_end
    ]
    shares = [float(estimate.opening_share) for estimate in estimates if estimate.opening_share is not None]
    accuracy = None
    if model_rows:
        accuracy = sum(int(row["prediction_correct"]) for row in model_rows) / len(model_rows)
    metrics = [
        ("db_path", str(db_path)),
        ("analysis_start", analysis_start.isoformat()),
        ("analysis_end", analysis_end.isoformat()),
        ("release_episodes", len(estimates)),
        ("wide_release_episodes", len(wide)),
        ("episodes_with_lifecycle_estimates", len(lifecycle)),
        ("timing_regression_sample", len(timing)),
        ("episodes_with_opening_share", len(shares)),
        ("median_opening_share", median(shares)),
        ("max_opening_share", max(shares) if shares else None),
        ("model_pairwise_comparisons", len(model_rows)),
        ("model_pairwise_accuracy", accuracy),
    ]
    for row in regression:
        if row["term"] in {
            "opening_weekend_gross_millions",
            "opening_7_day_gross_millions",
            "opening_share",
            "half_life_weeks",
        }:
            metrics.append((f'{row["model"]}_{row["term"]}_t_stat', row["t_stat"]))
    return [{"metric": metric, "value": value} for metric, value in metrics]


def svg_escape(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_placeholder_svg(path: Path, title: str, message: str) -> None:
    path.write_text(
        "\n".join(
            [
                '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="360" viewBox="0 0 760 360">',
                '<rect width="100%" height="100%" fill="#ffffff"/>',
                f'<text x="36" y="42" font-family="Arial" font-size="20" font-weight="700">{svg_escape(title)}</text>',
                f'<text x="36" y="82" font-family="Arial" font-size="14" fill="#555">{svg_escape(message)}</text>',
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )


def write_opening_share_svg(path: Path, rows: list[dict[str, object]]) -> None:
    values = [float(row["opening_share"]) for row in rows if row["opening_share"] is not None]
    if not values:
        write_placeholder_svg(path, "Opening share distribution", "No opening-share observations had enough chart coverage.")
        return
    width, height = 820, 460
    left, right, top, bottom = 70, 30, 70, 350
    max_value = max(0.5, max(values))
    bin_count = 10
    counts = [0] * bin_count
    for value in values:
        index = min(bin_count - 1, int(value / max_value * bin_count))
        counts[index] += 1
    max_count = max(counts) or 1
    bar_w = (width - left - right) / bin_count
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">Opening share distribution</text>',
        '<text x="36" y="58" font-family="Arial" font-size="13" fill="#555">The Numbers chart-gross share over each movie episode opening week</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{width - right}" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>',
    ]
    for i, count in enumerate(counts):
        x = left + i * bar_w + 3
        h = count / max_count * (bottom - top)
        y = bottom - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 6:.1f}" height="{h:.1f}" fill="#4c78a8"/>')
        parts.append(f'<text x="{x + (bar_w - 6) / 2:.1f}" y="{y - 5:.1f}" font-family="Arial" font-size="10" text-anchor="middle">{count}</text>')
    for tick in [0.0, max_value / 2.0, max_value]:
        x = left + tick / max_value * (width - left - right)
        parts.append(f'<text x="{x:.1f}" y="{bottom + 22}" font-family="Arial" font-size="11" text-anchor="middle">{tick:.0%}</text>')
    parts.append('<text x="410" y="405" font-family="Arial" font-size="13" text-anchor="middle">opening share</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_standardized_curve_svg(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        write_placeholder_svg(path, "Standardized log revenue curve", "No lifecycle estimates were available.")
        return
    width, height = 860, 480
    left, right, top, bottom = 80, 40, 70, 360
    xs = [float(row["half_lives_after_opening"]) for row in rows]
    ys = [float(row["mean_log_standardized_gross"]) for row in rows]
    ideal = [float(row["ideal_exponential_log_decline"]) for row in rows]
    x_min, x_max = min(xs), max(xs) or 1.0
    y_min, y_max = min(min(ys), min(ideal)), max(max(ys), max(ideal))
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    pad = (y_max - y_min) * 0.1
    y_min -= pad
    y_max += pad

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min or 1.0) * (width - left - right)

    def sy(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    actual_points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
    ideal_points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ideal))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">Standardized log revenue curve</text>',
        '<text x="36" y="58" font-family="Arial" font-size="13" fill="#555">Weekly grosses divided by opening-week gross and aligned by estimated half-life</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{width - right}" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>',
        f'<polyline points="{ideal_points}" fill="none" stroke="#999" stroke-width="2" stroke-dasharray="5 5"/>',
        f'<polyline points="{actual_points}" fill="none" stroke="#f58518" stroke-width="2.5"/>',
    ]
    for x, y in zip(xs, ys):
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.5" fill="#f58518"/>')
    parts.extend(
        [
            '<text x="610" y="96" font-family="Arial" font-size="12" fill="#f58518">observed mean</text>',
            '<text x="610" y="116" font-family="Arial" font-size="12" fill="#777">ideal exponential</text>',
            '<text x="430" y="415" font-family="Arial" font-size="13" text-anchor="middle">half-lives after opening</text>',
            '<text x="24" y="220" font-family="Arial" font-size="13" transform="rotate(-90 24 220)" text-anchor="middle">log standardized gross</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def write_timing_regression_svg(path: Path, estimates: list[LifecycleEstimate], analysis_start: dt.date, analysis_end: dt.date) -> None:
    sample = [
        estimate
        for estimate in estimates
        if estimate.is_wide
        and estimate.half_life_weeks is not None
        and estimate.season_delay_weeks is not None
        and analysis_start <= estimate.episode.opening_date <= analysis_end
    ]
    if len(sample) < 2:
        write_placeholder_svg(path, "Timing regression scatter", "Not enough wide releases opened in the selected analysis window.")
        return
    width, height = 860, 500
    left, right, top, bottom = 85, 40, 70, 370
    xs = [estimate.opening_7_day_gross_usd / 1_000_000.0 for estimate in sample]
    ys = [float(estimate.season_delay_weeks) for estimate in sample]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    x_pad = (x_max - x_min) * 0.08
    y_pad = (y_max - y_min) * 0.15
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (width - left - right)

    def sy(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    line_beta, _, _, _ = ols([[1.0, x] for x in xs], ys)
    line = [(x_min, line_beta[0] + line_beta[1] * x_min), (x_max, line_beta[0] + line_beta[1] * x_max)]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">Timing versus opening strength</text>',
        '<text x="36" y="58" font-family="Arial" font-size="13" fill="#555">Wide summer/holiday releases; y-axis is delay within each season</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{width - right}" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{sx(line[0][0]):.1f}" y1="{sy(line[0][1]):.1f}" x2="{sx(line[1][0]):.1f}" y2="{sy(line[1][1]):.1f}" stroke="#4c78a8" stroke-width="2"/>',
    ]
    for estimate, x, y in zip(sample, xs, ys):
        radius = 4 + min(8, float(estimate.half_life_weeks or 0) * 0.6)
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{radius:.1f}" fill="#f58518" fill-opacity="0.75" stroke="#8a4b10"/>')
        parts.append(f'<title>{svg_escape(estimate.episode.title)}</title>')
    parts.extend(
        [
            '<text x="430" y="425" font-family="Arial" font-size="13" text-anchor="middle">opening 7-day gross, USD millions</text>',
            '<text x="26" y="220" font-family="Arial" font-size="13" transform="rotate(-90 26 220)" text-anchor="middle">season-relative delay, weeks</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def write_model_vs_observed_svg(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        write_placeholder_svg(path, "Model-vs-observed timing", "No pairwise timing comparisons were available.")
        return
    correct = sum(int(row["prediction_correct"]) for row in rows)
    incorrect = len(rows) - correct
    width, height = 700, 420
    left, top, bottom = 100, 70, 310
    max_count = max(correct, incorrect, 1)
    bars = [("correct", correct, "#54a24b"), ("incorrect", incorrect, "#e45756")]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">Model implication: stronger movies open earlier</text>',
        '<text x="36" y="58" font-family="Arial" font-size="13" fill="#555">Pairwise comparison using composite opening-gross plus half-life strength</text>',
        f'<line x1="{left}" y1="{bottom}" x2="620" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>',
    ]
    for i, (label, count, color) in enumerate(bars):
        x = 180 + i * 190
        h = count / max_count * (bottom - top)
        y = bottom - h
        parts.append(f'<rect x="{x}" y="{y:.1f}" width="110" height="{h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x + 55}" y="{y - 8:.1f}" font-family="Arial" font-size="13" font-weight="700" text-anchor="middle">{count}</text>')
        parts.append(f'<text x="{x + 55}" y="{bottom + 24}" font-family="Arial" font-size="13" text-anchor="middle">{label}</text>')
    accuracy = correct / len(rows)
    parts.append(f'<text x="36" y="380" font-family="Arial" font-size="13">Pairwise accuracy: {accuracy:.1%} across {len(rows)} comparisons.</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def classification_color(classification: str) -> str:
    colors = {
        "simultaneous_beginning": "#54a24b",
        "movie1_delays": "#e45756",
        "movie2_delays": "#4c78a8",
        "dual_equilibria": "#f58518",
        "no_equilibrium_found": "#bab0ac",
        "other": "#b279a2",
    }
    return colors.get(classification, "#bab0ac")


def write_figure4_theory_svg(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        write_placeholder_svg(path, "Figure 4 recreation", "No theory rows were available.")
        return
    width, height = 860, 500
    left, right, top, bottom = 80, 45, 70, 370
    x_values = [float(row["movie1_opening_attraction"]) for row in rows]
    y_values = [
        float(value)
        for row in rows
        for value in (row["movie1_equilibrium_week"], row["movie2_equilibrium_week"])
        if value is not None
    ]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = 0.0, max(4.0, max(y_values) if y_values else 1.0)

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (width - left - right)

    def sy(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    movie1_points = []
    movie2_points = []
    for row in rows:
        attraction = float(row["movie1_opening_attraction"])
        if row["movie1_equilibrium_week"] is not None:
            movie1_points.append((attraction, float(row["movie1_equilibrium_week"])))
        if row["movie2_equilibrium_week"] is not None:
            movie2_points.append((attraction, float(row["movie2_equilibrium_week"])))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">Figure 4 recreation</text>',
        '<text x="36" y="58" font-family="Arial" font-size="13" fill="#555">Equilibrium release timing versus Movie 1 opening attraction; Movie 2 fixed at .5, half-lives 3.4 weeks</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{width - right}" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>',
    ]
    for points, color, label_y, label in [
        (movie1_points, "#e45756", 96, "Movie 1"),
        (movie2_points, "#4c78a8", 116, "Movie 2"),
    ]:
        point_text = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        if point_text:
            parts.append(f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            for x, y in points:
                parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.8" fill="{color}"/>')
        parts.append(f'<text x="650" y="{label_y}" font-family="Arial" font-size="12" fill="{color}">{label}</text>')
    parts.extend(
        [
            '<text x="430" y="425" font-family="Arial" font-size="13" text-anchor="middle">Movie 1 opening attraction</text>',
            '<text x="24" y="220" font-family="Arial" font-size="13" transform="rotate(-90 24 220)" text-anchor="middle">equilibrium opening week</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def write_equilibrium_region_svg(
    path: Path,
    rows: list[dict[str, object]],
    *,
    title: str,
    subtitle: str,
    x_field: str,
    y_field: str,
    x_label: str,
    y_label: str,
) -> None:
    if not rows:
        write_placeholder_svg(path, title, "No theory rows were available.")
        return
    width, height = 860, 520
    left, right, top, bottom = 95, 160, 75, 380
    xs = sorted({float(row[x_field]) for row in rows})
    ys = sorted({float(row[y_field]) for row in rows})
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cell_w = (width - left - right) / len(xs)
    cell_h = (bottom - top) / len(ys)

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min or 1.0) * (width - left - right - cell_w)

    def sy(value: float) -> float:
        return bottom - cell_h - (value - y_min) / (y_max - y_min or 1.0) * (bottom - top - cell_h)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">{svg_escape(title)}</text>',
        f'<text x="36" y="58" font-family="Arial" font-size="13" fill="#555">{svg_escape(subtitle)}</text>',
    ]
    for row in rows:
        x = float(row[x_field])
        y = float(row[y_field])
        color = classification_color(str(row["classification"]))
        parts.append(
            f'<rect x="{sx(x):.1f}" y="{sy(y):.1f}" width="{cell_w + 0.5:.1f}" height="{cell_h + 0.5:.1f}" fill="{color}" fill-opacity="0.86"/>'
        )
    parts.extend(
        [
            f'<rect x="{left}" y="{top}" width="{width - left - right}" height="{bottom - top}" fill="none" stroke="#333"/>',
            f'<text x="{left + (width - left - right) / 2:.1f}" y="435" font-family="Arial" font-size="13" text-anchor="middle">{svg_escape(x_label)}</text>',
            f'<text x="28" y="{top + (bottom - top) / 2:.1f}" font-family="Arial" font-size="13" transform="rotate(-90 28 {top + (bottom - top) / 2:.1f})" text-anchor="middle">{svg_escape(y_label)}</text>',
        ]
    )
    legend = [
        ("simultaneous_beginning", "both begin"),
        ("movie1_delays", "movie 1 delays"),
        ("movie2_delays", "movie 2 delays"),
        ("dual_equilibria", "dual equilibria"),
        ("no_equilibrium_found", "not found"),
    ]
    for index, (classification, label) in enumerate(legend):
        y = 100 + index * 28
        parts.append(f'<rect x="720" y="{y}" width="16" height="16" fill="{classification_color(classification)}"/>')
        parts.append(f'<text x="744" y="{y + 13}" font-family="Arial" font-size="12">{svg_escape(label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_summary_dashboard_svg(path: Path, metrics: list[dict[str, object]]) -> None:
    metric_map = {str(row["metric"]): row["value"] for row in metrics}
    cards = [
        ("Episodes", metric_map.get("release_episodes")),
        ("Wide episodes", metric_map.get("wide_release_episodes")),
        ("Lifecycle estimates", metric_map.get("episodes_with_lifecycle_estimates")),
        ("Timing sample", metric_map.get("timing_regression_sample")),
        ("Opening-share episodes", metric_map.get("episodes_with_opening_share")),
        ("Pairwise accuracy", f'{float(metric_map["model_pairwise_accuracy"]):.1%}' if metric_map.get("model_pairwise_accuracy") is not None else "n/a"),
    ]
    width, height = 860, 390
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="36" y="36" font-family="Arial" font-size="21" font-weight="700">Database competitive dynamics summary</text>',
        f'<text x="36" y="62" font-family="Arial" font-size="13" fill="#555">Window: {svg_escape(metric_map.get("analysis_start"))} to {svg_escape(metric_map.get("analysis_end"))}</text>',
    ]
    for i, (label, value) in enumerate(cards):
        col = i % 3
        row = i // 3
        x = 50 + col * 265
        y = 100 + row * 110
        parts.append(f'<rect x="{x}" y="{y}" width="230" height="78" fill="#f8f8f8" stroke="#d8d8d8"/>')
        parts.append(f'<text x="{x + 16}" y="{y + 28}" font-family="Arial" font-size="13" fill="#555">{svg_escape(label)}</text>')
        parts.append(f'<text x="{x + 16}" y="{y + 58}" font-family="Arial" font-size="24" font-weight="700">{svg_escape(value)}</text>')
    parts.append('<text x="36" y="344" font-family="Arial" font-size="12" fill="#555">All metrics are derived from the local The Numbers database, not from the original Variety sample.</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_readme(
    path: Path,
    *,
    db_path: Path,
    analysis_start: dt.date,
    analysis_end: dt.date,
    gap_threshold_days: int,
    min_positive_weeks: int,
    wide_theater_threshold: int,
    estimates: list[LifecycleEstimate],
) -> None:
    wide_count = sum(1 for estimate in estimates if estimate.is_wide)
    lifecycle_count = sum(1 for estimate in estimates if estimate.half_life_weeks is not None)
    text = [
        "Krider-Weinberg competitive dynamics replication for 2022+ movies",
        "",
        "This directory contains Krider-Weinberg-style lifecycle and release timing",
        "artifacts estimated from the local The Numbers database, plus numerical",
        "recreations of the paper's parameter-space figures. It is not the paper's",
        "original Variety 1990-1992 sample.",
        "",
        f"Database: {db_path}",
        f"Analysis window: {analysis_start.isoformat()} to {analysis_end.isoformat()}",
        "Default sample rule: release episodes opening on or after 2022-01-01",
        "Timing regression sample: wide releases in summer (May 25-Sep 5) or holiday (Nov 1-Jan 5)",
        "Timing regression target: weeks from each movie's season start, pooled across seasons",
        f"Gap threshold for release episode splitting: {gap_threshold_days} days",
        f"Minimum positive weeks for half-life estimation: {min_positive_weeks}",
        f"Wide-release threshold: {wide_theater_threshold} theaters",
        "",
        f"Release episodes: {len(estimates)}",
        f"Wide release episodes: {wide_count}",
        f"Episodes with lifecycle estimates: {lifecycle_count}",
        "",
        "Primary outputs:",
        "  database_2022_plus_lifecycle_estimates.csv",
        "  database_2022_plus_timing_regression_opening_gross.csv",
        "  database_2022_plus_timing_regression_opening_share.csv",
        "  database_2022_plus_opening_share_distribution.csv",
        "  database_2022_plus_standardized_log_revenue_curve.csv",
        "  paper_parameter_equilibrium_table.csv",
        "  paper_vs_database_2022_plus_summary.csv",
        "  figure_b1_database_2022_plus_opening_shares.svg",
        "  figure_b3_database_2022_plus_standardized_curve.svg",
        "  figure_4_recreated_equilibrium_vs_opening_attraction.svg",
        "  figure_5_recreated_equilibrium_regions.svg",
        "  figure_8_recreated_marketability_playability_regions.svg",
    ]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def write_outputs(
    out_dir: Path,
    *,
    db_path: Path,
    conn: Any,
    analysis_start: dt.date,
    analysis_end: dt.date,
    gap_threshold_days: int,
    min_positive_weeks: int,
    wide_theater_threshold: int,
    include_theory_recreation: bool = True,
) -> None:
    daily_rows = load_daily_rows(conn)
    all_episodes = split_release_episodes(daily_rows, gap_threshold_days=gap_threshold_days)
    episodes = filter_release_episodes(
        all_episodes,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
    )
    estimates, weekly_by_episode = estimate_lifecycles(
        episodes,
        conn,
        analysis_start=analysis_start,
        min_positive_weeks=min_positive_weeks,
        wide_theater_threshold=wide_theater_threshold,
    )
    regression = regression_rows(estimates, analysis_start, analysis_end)
    model_rows = model_vs_observed_rows(estimates, analysis_start, analysis_end)
    curve_rows = standardized_curve_rows(estimates, weekly_by_episode)
    share_rows = opening_share_rows(estimates)
    metrics = summary_metrics(db_path, analysis_start, analysis_end, estimates, regression, model_rows)
    comparison_rows = paper_vs_database_summary_rows(
        estimates,
        regression,
        model_rows,
        analysis_start,
        analysis_end,
    )

    write_csv(out_dir / "database_schema_summary.csv", schema_summary(conn), ["table_name", "row_count"])
    write_csv(
        out_dir / "database_release_episodes.csv",
        release_episode_rows(estimates),
        [
            "episode_id",
            "release_run_id",
            "movie_id",
            "title",
            "release_year",
            "episode_index",
            "opening_date",
            "closing_date",
            "observed_days",
            "max_theaters",
            "is_wide",
            "total_observed_gross_usd",
            "final_cumulative_gross_usd",
        ],
    )
    write_csv(
        out_dir / "database_lifecycle_estimates.csv",
        lifecycle_rows(estimates),
        [
            "episode_id",
            "title",
            "release_year",
            "opening_date",
            "delay_weeks",
            "season_label",
            "season_start",
            "season_delay_weeks",
            "positive_weeks",
            "opening_weekend_gross_usd",
            "opening_7_day_gross_usd",
            "max_theaters",
            "decay_beta",
            "half_life_weeks",
            "lifecycle_r2",
            "opening_share",
            "opening_attraction",
            "chart_coverage_days",
            "is_wide",
        ],
    )
    write_csv(
        out_dir / "database_2022_plus_lifecycle_estimates.csv",
        lifecycle_rows(estimates),
        [
            "episode_id",
            "title",
            "release_year",
            "opening_date",
            "delay_weeks",
            "season_label",
            "season_start",
            "season_delay_weeks",
            "positive_weeks",
            "opening_weekend_gross_usd",
            "opening_7_day_gross_usd",
            "max_theaters",
            "decay_beta",
            "half_life_weeks",
            "lifecycle_r2",
            "opening_share",
            "opening_attraction",
            "chart_coverage_days",
            "is_wide",
        ],
    )
    write_csv(
        out_dir / "database_opening_share_distribution.csv",
        share_rows,
        [
            "episode_id",
            "title",
            "opening_date",
            "opening_share",
            "opening_attraction",
            "chart_coverage_days",
            "opening_7_day_gross_usd",
        ],
    )
    write_csv(
        out_dir / "database_2022_plus_opening_share_distribution.csv",
        share_rows,
        [
            "episode_id",
            "title",
            "opening_date",
            "opening_share",
            "opening_attraction",
            "chart_coverage_days",
            "opening_7_day_gross_usd",
        ],
    )
    write_csv(
        out_dir / "database_standardized_log_revenue_curve.csv",
        curve_rows,
        [
            "half_lives_after_opening",
            "mean_log_standardized_gross",
            "ideal_exponential_log_decline",
            "movie_week_count",
        ],
    )
    write_csv(
        out_dir / "database_2022_plus_standardized_log_revenue_curve.csv",
        curve_rows,
        [
            "half_lives_after_opening",
            "mean_log_standardized_gross",
            "ideal_exponential_log_decline",
            "movie_week_count",
        ],
    )
    write_csv(
        out_dir / "database_timing_regression.csv",
        regression,
        ["model", "term", "estimate", "se", "t_stat", "n", "r2", "f_stat"],
    )
    write_csv(
        out_dir / "database_2022_plus_timing_regression_opening_gross.csv",
        regression_model_rows(
            regression,
            "delay_on_opening_weekend_gross_millions_and_half_life",
        ),
        ["model", "term", "estimate", "se", "t_stat", "n", "r2", "f_stat"],
    )
    write_csv(
        out_dir / "database_2022_plus_timing_regression_opening_share.csv",
        regression_model_rows(regression, "delay_on_opening_share_and_half_life"),
        ["model", "term", "estimate", "se", "t_stat", "n", "r2", "f_stat"],
    )
    write_csv(
        out_dir / "database_model_vs_observed_timing.csv",
        model_rows,
        [
            "season_label",
            "left_episode_id",
            "left_title",
            "left_opening_date",
            "left_season_delay_weeks",
            "left_strength_score",
            "right_episode_id",
            "right_title",
            "right_opening_date",
            "right_season_delay_weeks",
            "right_strength_score",
            "predicted_first_episode_id",
            "observed_first_episode_id",
            "prediction_correct",
            "opening_gap_days",
        ],
    )
    write_csv(out_dir / "database_summary_metrics.csv", metrics, ["metric", "value"])
    write_csv(
        out_dir / "paper_vs_database_2022_plus_summary.csv",
        comparison_rows,
        ["result", "paper_1990_reported", "database_2022_plus"],
    )

    write_opening_share_svg(out_dir / "figure_database_opening_share_distribution.svg", share_rows)
    write_opening_share_svg(out_dir / "figure_b1_database_2022_plus_opening_shares.svg", share_rows)
    write_standardized_curve_svg(out_dir / "figure_database_standardized_log_revenue_curve.svg", curve_rows)
    write_standardized_curve_svg(out_dir / "figure_b3_database_2022_plus_standardized_curve.svg", curve_rows)
    write_timing_regression_svg(out_dir / "figure_database_timing_regression.svg", estimates, analysis_start, analysis_end)
    write_model_vs_observed_svg(out_dir / "figure_database_model_vs_observed_timing.svg", model_rows)
    write_summary_dashboard_svg(out_dir / "figure_database_summary_dashboard.svg", metrics)

    if include_theory_recreation:
        figure4_rows = figure4_theory_rows()
        figure5_rows = figure5_theory_rows()
        figure8_rows = figure8_theory_rows()
        write_csv(
            out_dir / "paper_parameter_equilibrium_table.csv",
            paper_parameter_equilibrium_rows(),
            [
                "scenario",
                "movie",
                "opening_attraction",
                "half_life_weeks",
                "recreated_equilibrium_opening_week",
                "paper_reported_opening_week",
                "classification",
            ],
        )
        write_figure4_theory_svg(
            out_dir / "figure_4_recreated_equilibrium_vs_opening_attraction.svg",
            figure4_rows,
        )
        write_equilibrium_region_svg(
            out_dir / "figure_5_recreated_equilibrium_regions.svg",
            figure5_rows,
            title="Figure 5 recreation",
            subtitle="Movie 1 opening attraction fixed at .5; equal half-lives vary together",
            x_field="half_life_weeks",
            y_field="movie2_opening_attraction",
            x_label="half-life, weeks",
            y_label="Movie 2 opening attraction",
        )
        write_equilibrium_region_svg(
            out_dir / "figure_8_recreated_marketability_playability_regions.svg",
            figure8_rows,
            title="Figure 8 recreation",
            subtitle="Movie 1 fixed at opening attraction .5 and half-life 3 weeks; Movie 2 varies",
            x_field="movie2_half_life_weeks",
            y_field="movie2_opening_attraction",
            x_label="Movie 2 half-life, weeks",
            y_label="Movie 2 opening attraction",
        )
    write_readme(
        out_dir / "README.txt",
        db_path=db_path,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
        gap_threshold_days=gap_threshold_days,
        min_positive_weeks=min_positive_weeks,
        wide_theater_threshold=wide_theater_threshold,
        estimates=estimates,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=database_url_from_env(),
        help="PostgreSQL connection URL. Defaults to DATABASE_URL or POSTGRES_DSN.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument("--analysis-start", help="Optional analysis start date, YYYY-MM-DD.")
    parser.add_argument("--analysis-end", help="Optional analysis end date, YYYY-MM-DD.")
    parser.add_argument("--gap-threshold-days", type=int, default=DEFAULT_GAP_THRESHOLD_DAYS)
    parser.add_argument("--min-positive-weeks", type=int, default=DEFAULT_MIN_POSITIVE_WEEKS)
    parser.add_argument("--wide-theater-threshold", type=int, default=DEFAULT_WIDE_THEATER_THRESHOLD)
    parser.add_argument(
        "--skip-theory-recreation",
        action="store_true",
        help="Only write database-derived artifacts; skip recreated paper-parameter figures.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()
    conn = connect_database(args.database_url)
    try:
        validate_database(conn)
        default_start, default_end = default_analysis_window(conn)
        analysis_start = date_or_none(args.analysis_start) or default_start
        analysis_end = date_or_none(args.analysis_end) or default_end
        if analysis_end < analysis_start:
            raise SystemExit("analysis end must be on or after analysis start")
        clean_output_dir(args.out_dir)
        write_outputs(
            args.out_dir,
            db_path=Path("postgres"),
            conn=conn,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
            gap_threshold_days=args.gap_threshold_days,
            min_positive_weeks=args.min_positive_weeks,
            wide_theater_threshold=args.wide_theater_threshold,
            include_theory_recreation=not args.skip_theory_recreation,
        )
    finally:
        conn.close()
    print(f"Wrote database competitive dynamics artifacts to {args.out_dir}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
