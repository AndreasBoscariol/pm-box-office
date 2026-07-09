#!/usr/bin/env python3
"""CLI wrapper for producing movie forecast timelines."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.boxoffice.produce_movie_forecast_timeline import main


if __name__ == "__main__":
    raise SystemExit(main())
