"""Shared helpers for paper recreation scripts."""

from __future__ import annotations

import csv
import datetime as dt
import math
from pathlib import Path
from typing import Any, Iterable


def parse_db_date(value: Any) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


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
