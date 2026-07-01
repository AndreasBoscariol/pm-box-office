from __future__ import annotations

import datetime as dt
import unittest
from unittest.mock import Mock, patch

from pm_box_office.research.papers import recreate_competition_opening_weekend as opening
from pm_box_office.web.services import forecast_service


class Cursor:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self.rows = rows

    def fetchall(self) -> list[tuple[object, ...]]:
        return self.rows

    def fetchone(self) -> tuple[object, ...] | None:
        return self.rows[0] if self.rows else None


class FakeConn:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, object]] = []

    def execute(self, sql: str, params: object = None) -> Cursor:
        self.calls.append((sql, params))
        return Cursor(self.rows)


def bop_forecast(
    movie_id: int,
    *,
    prediction_id: int,
    published_date: dt.date,
    target_start_date: dt.date,
    low: int = 10_000_000,
    high: int = 20_000_000,
) -> opening.BoxofficeProForecast:
    return opening.BoxofficeProForecast(
        prediction_id=prediction_id,
        movie_id=movie_id,
        article_id=prediction_id,
        article_url=f"https://www.boxofficepro.com/{prediction_id}/",
        source_movie_title=f"Movie {movie_id}",
        forecast_metric="domestic_opening_weekend",
        source_context="test",
        source_rank=1,
        target_start_date=target_start_date,
        target_end_date=target_start_date + dt.timedelta(days=2),
        range_low_usd=float(low),
        range_high_usd=float(high),
        showtime_market_share_pct=None,
        published_date=published_date,
    )


class ForecastServiceTests(unittest.TestCase):
    def tearDown(self) -> None:
        forecast_service.clear_model_cache()

    def test_search_forecast_movies_reads_bop_matched_candidates(self) -> None:
        conn = FakeConn(
            [
                (
                    7,
                    "Example Movie",
                    2026,
                    "Example Movie Source",
                    dt.date(2026, 7, 3),
                    dt.date(2026, 7, 1),
                    42_000_000.0,
                )
            ]
        )

        rows = forecast_service.search_forecast_movies(conn, query="example")

        self.assertEqual(1, len(rows))
        self.assertEqual(7, rows[0]["movie_id"])
        self.assertEqual("Example Movie", rows[0]["title"])
        self.assertEqual("$42.0M", rows[0]["latest_bop_midpoint_label"])
        self.assertIn("boxofficepro_weekend_predictions", conn.calls[0][0])

    def test_latest_bop_forecast_excludes_future_article_dates(self) -> None:
        target = dt.date(2026, 7, 3)
        forecasts = [
            bop_forecast(1, prediction_id=1, published_date=dt.date(2026, 6, 20), target_start_date=target),
            bop_forecast(1, prediction_id=2, published_date=dt.date(2026, 7, 2), target_start_date=target),
        ]

        selected = opening.latest_forecast(
            forecasts,
            as_of_date=dt.date(2026, 6, 30),
            movie_id=1,
            forecast_metric="domestic_opening_weekend",
            target_start_date=target,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(1, selected.prediction_id)

    def test_build_movie_feature_rows_filters_future_snapshot_dates(self) -> None:
        candidate = forecast_service.ForecastCandidate(
            movie_id=1,
            title="Future Movie",
            release_year=2026,
            opening_date=dt.date(2026, 7, 17),
            latest_forecast_date=dt.date(2026, 7, 1),
            latest_bop_midpoint=25_000_000.0,
            source_movie_title="Future Movie",
        )
        past_row = {"snapshot_day": -14, "as_of_date": "2026-07-03"}
        future_row = {"snapshot_day": -1, "as_of_date": "2026-07-16"}
        with (
            patch.object(
                forecast_service,
                "movie_for_candidate",
                return_value=opening.OpeningWeekendMovie(
                    movie_id=1,
                    title="Future Movie",
                    release_year=2026,
                    release_run_id=0,
                    opening_date=candidate.opening_date,
                    opening_theaters=0,
                    opening_day_gross_usd=0,
                    opening_weekend_revenue_usd=0,
                ),
            ),
            patch.object(forecast_service.opening, "load_daily_grosses", return_value=[]),
            patch.object(forecast_service.opening, "load_wiki_feature_map", return_value={}),
            patch.object(forecast_service.opening, "load_boxofficepro_forecasts", return_value=[]),
            patch.object(
                forecast_service.day_by_day,
                "build_day_by_day_feature_panel",
                return_value=[past_row, future_row],
            ),
        ):
            rows, has_actual = forecast_service.build_movie_feature_rows(
                Mock(),
                candidate,
                today=dt.date(2026, 7, 3),
            )

        self.assertFalse(has_actual)
        self.assertEqual([past_row], rows)

    def test_release_timing_label_handles_future_today_and_released_movies(self) -> None:
        today = dt.date(2026, 7, 1)

        self.assertEqual("2 days until opening", forecast_service.release_timing_label(dt.date(2026, 7, 3), today))
        self.assertEqual("released today", forecast_service.release_timing_label(today, today))
        self.assertEqual("released 68 days ago", forecast_service.release_timing_label(dt.date(2026, 4, 24), today))

    def test_forecast_movie_uses_fallback_model_when_bop_snapshot_is_missing(self) -> None:
        candidate = forecast_service.ForecastCandidate(
            movie_id=1,
            title="Earlier Signal Movie",
            release_year=2026,
            opening_date=dt.date(2026, 7, 17),
            latest_forecast_date=dt.date(2026, 7, 15),
            latest_bop_midpoint=30_000_000.0,
            source_movie_title="Earlier Signal Movie",
        )
        row = {
            "snapshot_day": -14,
            "as_of_date": "2026-07-03",
            "bop_forecast_available": 0.0,
            "bop_forecast_midpoint": 0.0,
            "bop_forecast_published_date": "",
            "wiki_available": 1.0,
            "competitor_count_lag7": 0.0,
            "opening_weekend_revenue_usd": 0.0,
        }
        fitted = opening.FittedModel(
            terms=[],
            centers=[],
            scales=[],
            beta=[__import__("math").log(25_000_000.0)],
        )
        bundle = forecast_service.ForecastModelBundle(
            train_start_year=2022,
            train_end_year=2025,
            min_opening_day_gross=5_000_000,
            models={
                (-14, forecast_service.FALLBACK_MODEL): forecast_service.ForecastDayModel(
                    snapshot_day=-14,
                    model_name=forecast_service.FALLBACK_MODEL,
                    terms=[],
                    train_rows=[],
                    fitted=fitted,
                    bucket_residuals={"missing_bop": [0.1]},
                    global_residuals=[0.1],
                )
            },
        )
        with (
            patch.object(forecast_service, "latest_candidate_for_movie", return_value=candidate),
            patch.object(forecast_service, "get_model_bundle", return_value=bundle),
            patch.object(forecast_service, "build_movie_feature_rows", return_value=([row], False)),
        ):
            payload = forecast_service.forecast_movie(Mock(), movie_id=1, today=dt.date(2026, 7, 3))

        self.assertIsNotNone(payload)
        snapshots = payload["snapshots"]  # type: ignore[index]
        self.assertEqual(1, len(snapshots))
        self.assertEqual("ok", snapshots[0]["status"])
        self.assertEqual(forecast_service.FALLBACK_MODEL, snapshots[0]["model"])
        self.assertEqual("fallback_model", snapshots[0]["prediction_source"])
        self.assertFalse(snapshots[0]["bop_forecast_available"])
        self.assertEqual("-", snapshots[0]["bop_forecast_midpoint_label"])

    def test_model_cache_reuses_empty_training_bundle(self) -> None:
        conn = Mock()
        with patch.object(forecast_service.opening, "load_opening_weekend_movies", return_value=[]) as load_movies:
            first = forecast_service.get_model_bundle(conn)
            second = forecast_service.get_model_bundle(conn)

        self.assertIs(first, second)
        load_movies.assert_called_once()


if __name__ == "__main__":
    unittest.main()
