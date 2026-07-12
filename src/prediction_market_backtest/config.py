"""Dependency-free pipeline configuration loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Config:
    database_url: str | None = None
    gamma_url: str = "https://gamma-api.polymarket.com"
    clob_url: str = "https://clob.polymarket.com"
    random_seed: int = 0
    assumed_latency_seconds: float = 5.0
    maximum_book_age_seconds: float = 60.0
    risk_budget: float = 10.0
    edge_threshold: float = 0.05
    forecast_policy_name: str = "007"
    forecast_policy_version: str = "boxoffice_local_007_weighted_interval_calibration"
    allow_mixed_policy_diagnostic: bool = False


def load_config(path: str | Path | None, overrides: dict[str, Any] | None = None) -> Config:
    data: dict[str, Any] = {}
    if path:
        raw = Path(path).read_text(encoding="utf-8")
        if Path(path).suffix.lower() == ".json":
            data = json.loads(raw)
        else:
            try:
                import yaml  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ValueError("YAML configuration requires PyYAML; JSON works without it") from exc
            data = yaml.safe_load(raw) or {}
    data.update({k: v for k, v in (overrides or {}).items() if v is not None})
    allowed = {field.name for field in fields(Config)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError("unknown configuration keys: " + ", ".join(sorted(unknown)))
    return Config(**data)
