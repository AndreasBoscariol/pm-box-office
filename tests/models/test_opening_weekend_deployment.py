from __future__ import annotations

import datetime as dt
import unittest

from pm_box_office.models.opening_weekend.backtest import robust_winner_rows
from pm_box_office.models.opening_weekend.engine import (
    FeatureSnapshot,
    OpeningWindowPredictionEngine,
    SourceAvailability,
    known_actual_offsets_for_as_of,
)
from pm_box_office.models.opening_weekend.targets import resolve_opening_target


class OpeningWeekendDeploymentTests(unittest.TestCase):
    def test_target_window_resolution_supports_three_four_and_five_day_targets(self) -> None:
        friday = resolve_opening_target(dt.date(2026, 7, 3))
        monday_holiday = resolve_opening_target(dt.date(2026, 7, 3), holiday_flag=True)
        wednesday = resolve_opening_target(dt.date(2026, 7, 1))
        explicit = resolve_opening_target(
            dt.date(2026, 7, 3),
            target_end_date=dt.date(2026, 7, 7),
        )

        self.assertEqual(("3_day", 3, dt.date(2026, 7, 5)), (friday.target_type, friday.target_days, friday.target_end_date))
        self.assertEqual(("4_day", 4, dt.date(2026, 7, 6)), (monday_holiday.target_type, monday_holiday.target_days, monday_holiday.target_end_date))
        self.assertEqual(("5_day", 5, dt.date(2026, 7, 5)), (wednesday.target_type, wednesday.target_days, wednesday.target_end_date))
        self.assertEqual(("5_day", 5), (explicit.target_type, explicit.target_days))

    def test_production_routing_by_bop_segment(self) -> None:
        engine = OpeningWindowPredictionEngine()
        target = resolve_opening_target(dt.date(2026, 7, 3))

        large = FeatureSnapshot(1, -1, dt.date(2026, 7, 2), target, 30_000_000.0, (), 0.0, SourceAvailability(bop_available=True))
        mid = FeatureSnapshot(1, -1, dt.date(2026, 7, 2), target, 10_000_000.0, (), 0.0, SourceAvailability(bop_available=True))
        small = FeatureSnapshot(1, -1, dt.date(2026, 7, 2), target, 4_000_000.0, (), 0.0, SourceAvailability(bop_available=True))
        missing = FeatureSnapshot(1, -1, dt.date(2026, 7, 2), target, None, (), 0.0, SourceAvailability())

        self.assertEqual("production", engine.predict(large).deployment_status)
        self.assertEqual("ridge_full_plus_both_competition", engine.predict(large).registry_entry.model_name)
        self.assertEqual("cautious_production", engine.predict(mid).deployment_status)
        self.assertEqual("shadow", engine.predict(small).deployment_status)
        self.assertEqual("shadow", engine.predict(missing).deployment_status)

    def test_actuals_known_switches_to_remaining_gross_nowcast(self) -> None:
        engine = OpeningWindowPredictionEngine()
        target = resolve_opening_target(dt.date(2026, 7, 3))
        snapshot = FeatureSnapshot(
            1,
            1,
            dt.date(2026, 7, 4),
            target,
            30_000_000.0,
            (0,),
            12_000_000.0,
            SourceAvailability(bop_available=True, actuals_available=True),
        )

        prediction = engine.predict(snapshot, expected_remaining_gross=18_000_000.0)

        self.assertEqual("official_actuals_available", prediction.forecast_state)
        self.assertEqual("known_actuals_remainder", prediction.registry_entry.model_family)
        self.assertEqual(18_000_000.0, prediction.remaining_gross_forecast)
        self.assertEqual(30_000_000.0, prediction.point_forecast_usd)

    def test_one_day_lag_actual_availability(self) -> None:
        opening = dt.date(2026, 7, 3)

        self.assertEqual((), known_actual_offsets_for_as_of(target_start_date=opening, target_days=3, as_of_date=opening))
        self.assertEqual((0,), known_actual_offsets_for_as_of(target_start_date=opening, target_days=3, as_of_date=opening + dt.timedelta(days=1)))
        self.assertEqual((0, 1), known_actual_offsets_for_as_of(target_start_date=opening, target_days=3, as_of_date=opening + dt.timedelta(days=2)))
        self.assertEqual((0, 1, 2), known_actual_offsets_for_as_of(target_start_date=opening, target_days=3, as_of_date=opening + dt.timedelta(days=3)))
        self.assertEqual((0, 1, 2, 3, 4), known_actual_offsets_for_as_of(target_start_date=opening, target_days=5, as_of_date=opening + dt.timedelta(days=5)))

    def test_robust_winner_rows_require_baseline_lift_and_flag_low_sample(self) -> None:
        rows = [
            {
                "status": "ok",
                "target_type": "3_day",
                "population": "bop_covered",
                "interval_method": "loo_conformal_abs",
                "model": "candidate",
                "test_start_year": str(year),
                "holdout_n": "4",
                "mape_gross": "0.10",
                "raw_estimate_mae_lift_usd": "1000",
                "actuals_multiplier_mae_lift_usd": "",
            }
            for year in (2023, 2024, 2025, 2026)
        ]
        rows.append(
            {
                "status": "ok",
                "target_type": "3_day",
                "population": "bop_covered",
                "interval_method": "loo_conformal_abs",
                "model": "worse_than_baseline",
                "test_start_year": "2026",
                "holdout_n": "12",
                "mape_gross": "0.08",
                "raw_estimate_mae_lift_usd": "-1",
                "actuals_multiplier_mae_lift_usd": "",
            }
        )

        winners = robust_winner_rows(rows, min_holdout_n=10)
        by_model = {row["model"]: row for row in winners}

        self.assertFalse(by_model["candidate"]["low_sample_flag"])
        self.assertTrue(by_model["candidate"]["beats_baseline"])
        self.assertFalse(by_model["worse_than_baseline"]["beats_baseline"])


if __name__ == "__main__":
    unittest.main()
