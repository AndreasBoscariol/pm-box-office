"""Production wrapper around leakage-safe AMC as-of feature construction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class AMCFeatureBuilder:
    conn: Any
    origin_timezone: str = "America/New_York"

    def build_asof_features(
        self,
        *,
        movie_id: int,
        exhibition_date: date,
        forecast_origin: str,
        as_of_utc: pd.Timestamp,
    ) -> pd.DataFrame:
        """Build one movie-day-origin AMC feature row using snapshots observed by ``as_of_utc``."""

        from eda.During import same_day_seat_nowcast_eda as amc_eda

        schedule = amc_eda.fetch_schedule(self.conn)
        snapshots = amc_eda.fetch_snapshots(self.conn)
        if schedule.empty:
            return pd.DataFrame()
        schedule["exhibition_date"] = pd.to_datetime(schedule["exhibition_date"], errors="coerce").dt.date
        snapshots["exhibition_date"] = pd.to_datetime(snapshots["exhibition_date"], errors="coerce").dt.date

        schedule = schedule.loc[
            schedule["movie_id"].eq(movie_id) & schedule["exhibition_date"].eq(exhibition_date)
        ].copy()
        snapshots = snapshots.loc[
            snapshots["movie_id"].eq(movie_id)
            & snapshots["exhibition_date"].eq(exhibition_date)
            & (pd.to_datetime(snapshots["observed_at"], utc=True, errors="coerce") <= pd.Timestamp(as_of_utc).tz_convert("UTC"))
        ].copy()
        if schedule.empty:
            return pd.DataFrame()

        grid = amc_eda.build_origin_grid(schedule, [forecast_origin], self.origin_timezone)
        grid["forecast_origin_utc"] = pd.Timestamp(as_of_utc).tz_convert("UTC")
        asof = amc_eda.latest_snapshots_as_of(snapshots, grid)
        return amc_eda.aggregate_as_of_features(asof)

