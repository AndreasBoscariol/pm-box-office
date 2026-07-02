"""Model evaluation entrypoints for deployed opening-window forecast artifacts."""

from __future__ import annotations

from pm_box_office.models.opening_weekend.backtest import build_parser, main, robust_winner_rows


__all__ = ["build_parser", "main", "robust_winner_rows"]
