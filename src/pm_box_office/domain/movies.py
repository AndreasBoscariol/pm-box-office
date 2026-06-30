"""Movie domain constants and lightweight helpers."""

from __future__ import annotations


def normalize_title(value: str) -> str:
    return " ".join(value.strip().lower().split())

