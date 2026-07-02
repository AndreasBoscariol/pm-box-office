"""Prediction entrypoints for the deployed opening-window forecast model."""

from __future__ import annotations

from pm_box_office.models.opening_weekend.engine import OpeningWindowPredictionEngine
from pm_box_office.models.opening_weekend.registry import DEFAULT_REGISTRY


def opening_window_engine() -> OpeningWindowPredictionEngine:
    return OpeningWindowPredictionEngine(DEFAULT_REGISTRY)
