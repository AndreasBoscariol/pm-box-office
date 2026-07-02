from __future__ import annotations

import datetime as dt
import math
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


def weekend_actuals(*grosses: int | None, estimate_offsets: set[int] | None = None) -> list[dict[str, object]]:
    estimate_offsets = estimate_offsets or set()
    opening_date = dt.date(2026, 7, 3)
    rows: list[dict[str, object]] = []
    for offset in forecast_service.OPENING_WEEKEND_OFFSETS:
        gross = grosses[offset] if offset < len(grosses) else None
        is_estimate = offset in estimate_offsets and gross is not None
        rows.append(
            {
                "offset": offset,
                "box_office_date": (opening_date + dt.timedelta(days=offset)).isoformat(),
                "gross_usd": float(gross) if gross is not None else None,
                "gross_label": forecast_service.money(float(gross)) if gross is not None else "-",
                "is_actual": gross is not None and not is_estimate,
                "is_estimate": is_estimate,
                "status": "estimate" if is_estimate else ("actual" if gross is not None else "missing"),
                "source_url": "",
                "fetched_at": "",
            }
        )
    return rows


class ForecastServiceTests(unittest.TestCase):
    def tearDown(self) -> None:
        forecast_service.clear_model_cache()

    def test_wikipedia_forecast_models_use_recommended_activity_terms(self) -> None:
        full_wiki_terms = {"log1p_V", "log1p_U", "log1p_R", "log1p_E"}
        residual_terms = forecast_service.day_by_day.SNAPSHOT_MODEL_TERMS[
            forecast_service.WIKI_RESIDUAL_MODEL
        ]

        self.assertTrue(full_wiki_terms.issubset(forecast_service.FALLBACK_TERMS))
        self.assertIn("log1p_V", residual_terms)
        self.assertFalse({"log1p_U", "log1p_R", "log1p_E"} & set(residual_terms))

    def test_standalone_wiki_uses_two_million_floor_without_broadening_bop_residual_models(self) -> None:
        opening_date = dt.date(2025, 6, 6)
        grosses = [
            1_000_000,
            2_000_000,
            3_000_000,
            4_000_000,
            5_000_000,
            6_000_000,
            7_000_000,
            8_000_000,
            9_000_000,
            10_000_000,
            11_000_000,
            12_000_000,
        ]
        movies = [
            opening.OpeningWeekendMovie(
                movie_id=index,
                title=f"Movie {index}",
                release_year=2025,
                release_run_id=index,
                opening_date=opening_date + dt.timedelta(days=index * 7),
                opening_theaters=3000,
                opening_day_gross_usd=gross,
                opening_weekend_revenue_usd=gross * 3,
            )
            for index, gross in enumerate(grosses, start=1)
        ]
        panel_rows = [
            {
                "movie_id": movie.movie_id,
                "snapshot_day": -2,
                "release_year": movie.release_year,
                "opening_day_gross_usd": movie.opening_day_gross_usd,
                "opening_weekend_revenue_usd": movie.opening_weekend_revenue_usd,
                "target_log_opening_weekend": math.log(movie.opening_weekend_revenue_usd),
                "target_log_bop_residual": 0.1,
                "bop_forecast_available": 1.0,
                "bop_forecast_midpoint": float(movie.opening_weekend_revenue_usd) * 0.9,
                "bop_forecast_range_width_pct": 0.2,
                "bop_estimate_bucket": "30-60m",
                "bop_estimate_bucket_15_30m": 0.0,
                "bop_estimate_bucket_30_60m": 1.0,
                "bop_estimate_bucket_60_100m": 0.0,
                "bop_estimate_bucket_100m_plus": 0.0,
                "wiki_available": 1.0,
                "log1p_V": 1.0,
                "log1p_U": 1.0,
                "log1p_R": 1.0,
                "log1p_E": 1.0,
                "log1p_opening_theaters": 1.0,
                "competitor_count_lag7": 0.0,
                "log1p_competitor_total_gross_lag7": 0.0,
            }
            for movie in movies
        ]

        def fitted_model(rows: list[dict[str, object]], terms: list[str], target_key: str) -> opening.FittedModel:
            return opening.FittedModel(
                terms=terms,
                centers=[0.0 for _ in terms],
                scales=[1.0 for _ in terms],
                beta=[0.0 for _ in range(len(terms) + 1)],
            )

        def model_residuals(**kwargs: object) -> list[float]:
            return [0.0 for _ in kwargs["rows"]]  # type: ignore[index]

        def interval_residuals(**kwargs: object) -> tuple[dict[str, list[float]], list[float]]:
            residuals = list(kwargs["residuals"])  # type: ignore[arg-type]
            return {}, residuals

        with (
            patch.object(forecast_service.opening, "load_opening_weekend_movies", return_value=movies) as load_movies,
            patch.object(forecast_service.opening, "load_daily_grosses", return_value=[]),
            patch.object(forecast_service.opening, "load_wiki_feature_map", return_value={}),
            patch.object(forecast_service.opening, "load_boxofficepro_forecasts", return_value=[]),
            patch.object(forecast_service.day_by_day, "build_day_by_day_feature_panel", return_value=panel_rows),
            patch.object(forecast_service, "load_amc_feature_map", return_value={}),
            patch.object(forecast_service, "train_weekend_remainder_models", return_value={}) as train_remainder,
            patch.object(forecast_service.opening, "fit_model_for_target", side_effect=fitted_model),
            patch.object(
                forecast_service.day_by_day,
                "loo_prediction_residuals_for_opening_model",
                side_effect=model_residuals,
            ),
            patch.object(
                forecast_service.day_by_day,
                "loo_prediction_residuals_for_model",
                side_effect=model_residuals,
            ),
            patch.object(
                forecast_service.day_by_day,
                "interval_residuals_by_bucket",
                side_effect=interval_residuals,
            ),
        ):
            bundle = forecast_service.get_model_bundle(Mock())

        self.assertEqual(
            forecast_service.DEFAULT_WIKI_MIN_OPENING_DAY_GROSS,
            load_movies.call_args.kwargs["min_opening_day_gross"],
        )
        fallback_model = bundle.models[(-2, forecast_service.FALLBACK_MODEL)]
        wiki_model = bundle.models[(-2, forecast_service.WIKI_RESIDUAL_MODEL)]
        bucket_model = bundle.models[(-2, forecast_service.BUCKET_RESIDUAL_MODEL)]
        raw_model = bundle.models[(-2, forecast_service.RAW_MODEL)]

        self.assertEqual(2_000_000, min(row["opening_day_gross_usd"] for row in fallback_model.train_rows))
        self.assertEqual(5_000_000, min(row["opening_day_gross_usd"] for row in wiki_model.train_rows))
        self.assertEqual(5_000_000, min(row["opening_day_gross_usd"] for row in bucket_model.train_rows))
        self.assertEqual(5_000_000, min(row["opening_day_gross_usd"] for row in raw_model.train_rows))
        remainder_movies = train_remainder.call_args.args[0]
        self.assertEqual(5_000_000, min(movie.opening_day_gross_usd for movie in remainder_movies))

    def test_pre_release_wiki_residual_trains_on_pooled_lead_window(self) -> None:
        opening_date = dt.date(2025, 6, 6)
        movies = [
            opening.OpeningWeekendMovie(
                movie_id=index,
                title=f"Movie {index}",
                release_year=2025,
                release_run_id=index,
                opening_date=opening_date + dt.timedelta(days=index * 7),
                opening_theaters=3000,
                opening_day_gross_usd=6_000_000 + index,
                opening_weekend_revenue_usd=12_000_000 + index,
            )
            for index in range(1, 8)
        ]
        panel_rows = [
            {
                "movie_id": movie.movie_id,
                "snapshot_day": -14 if movie.movie_id <= 3 else -13,
                "release_year": movie.release_year,
                "opening_day_gross_usd": movie.opening_day_gross_usd,
                "opening_weekend_revenue_usd": movie.opening_weekend_revenue_usd,
                "target_log_opening_weekend": math.log(movie.opening_weekend_revenue_usd),
                "target_log_bop_residual": 0.1 * movie.movie_id,
                "bop_forecast_available": 1.0,
                "bop_forecast_midpoint": float(movie.opening_weekend_revenue_usd) * 0.9,
                "bop_forecast_range_width_pct": 0.2,
                "bop_estimate_bucket": "under_15m",
                "bop_estimate_bucket_15_30m": 0.0,
                "bop_estimate_bucket_30_60m": 0.0,
                "bop_estimate_bucket_60_100m": 0.0,
                "bop_estimate_bucket_100m_plus": 0.0,
                "wiki_available": 1.0,
                "log1p_V": float(movie.movie_id),
                "log1p_U": 0.0,
                "log1p_R": 0.0,
                "log1p_E": 0.0,
                "log1p_opening_theaters": 1.0,
                "competitor_count_lag7": 0.0,
                "log1p_competitor_total_gross_lag7": 0.0,
            }
            for movie in movies
        ]

        def fitted_model(rows: list[dict[str, object]], terms: list[str], target_key: str) -> opening.FittedModel:
            return opening.FittedModel(
                terms=terms,
                centers=[0.0 for _ in terms],
                scales=[1.0 for _ in terms],
                beta=[0.0 for _ in range(len(terms) + 1)],
            )

        def model_residuals(**kwargs: object) -> list[float]:
            return [0.0 for _ in kwargs["rows"]]  # type: ignore[index]

        def interval_residuals(**kwargs: object) -> tuple[dict[str, list[float]], list[float]]:
            residuals = list(kwargs["residuals"])  # type: ignore[arg-type]
            return {}, residuals

        with (
            patch.object(forecast_service.opening, "load_opening_weekend_movies", return_value=movies),
            patch.object(forecast_service.opening, "load_daily_grosses", return_value=[]),
            patch.object(forecast_service.opening, "load_wiki_feature_map", return_value={}),
            patch.object(forecast_service.opening, "load_boxofficepro_forecasts", return_value=[]),
            patch.object(forecast_service.day_by_day, "build_day_by_day_feature_panel", return_value=panel_rows),
            patch.object(forecast_service, "load_amc_feature_map", return_value={}),
            patch.object(forecast_service, "train_weekend_remainder_models", return_value={}),
            patch.object(forecast_service.opening, "fit_model_for_target", side_effect=fitted_model),
            patch.object(
                forecast_service.day_by_day,
                "loo_prediction_residuals_for_model",
                side_effect=model_residuals,
            ),
            patch.object(
                forecast_service.day_by_day,
                "interval_residuals_by_bucket",
                side_effect=interval_residuals,
            ),
        ):
            bundle = forecast_service.get_model_bundle(Mock())

        model = bundle.models[(-14, forecast_service.WIKI_RESIDUAL_MODEL)]
        self.assertEqual(forecast_service.day_by_day.SNAPSHOT_MODEL_TERMS[forecast_service.WIKI_RESIDUAL_MODEL], model.terms)
        self.assertEqual(7, len(model.train_rows))
        self.assertEqual({-14, -13}, {int(row["snapshot_day"]) for row in model.train_rows})
        raw_model = bundle.models[(-14, forecast_service.RAW_MODEL)]
        self.assertEqual(7, len(raw_model.train_rows))
        self.assertEqual({-14, -13}, {int(row["snapshot_day"]) for row in raw_model.train_rows})

    def test_search_forecast_movies_reads_bop_matched_candidates(self) -> None:
        conn = FakeConn(
            [
                (
                    7,
                    "Example Movie",
                    2026,
                    "Example Movie Source",
                    dt.date(2026, 7, 3),
                    dt.date(2026, 7, 3),
                    dt.date(2026, 7, 5),
                    3,
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

    def test_latest_candidate_defaults_to_longest_explicit_target_window(self) -> None:
        conn = FakeConn(
            [
                (
                    7,
                    "Minions & Monsters",
                    2026,
                    "Minions & Monsters",
                    dt.date(2026, 7, 1),
                    dt.date(2026, 7, 1),
                    dt.date(2026, 7, 5),
                    5,
                    dt.date(2026, 6, 19),
                    100_000_000.0,
                )
            ]
        )

        candidate = forecast_service.latest_candidate_for_movie(conn, 7)

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual("5_day", candidate.target_type)
        self.assertEqual(dt.date(2026, 7, 5), candidate.target_end_date)
        self.assertIn("target_end_date::date - p.target_start_date::date + 1)::integer DESC", conn.calls[0][0])

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
            target_start_date=dt.date(2026, 7, 17),
            target_end_date=dt.date(2026, 7, 19),
            target_day_count=3,
            target_type="3_day",
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
            target_start_date=dt.date(2026, 7, 17),
            target_end_date=dt.date(2026, 7, 19),
            target_day_count=3,
            target_type="3_day",
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
            "movie_id": 1,
            "opening_date": "2026-07-17",
            "target_start_date": "2026-07-17",
            "target_end_date": "2026-07-19",
            "target_day_count": 3,
            "target_type": "3_day",
            "wiki_available": 1.0,
            "competitor_count_lag7": 0.0,
            "opening_weekend_revenue_usd": 0.0,
        }
        fitted = opening.FittedModel(
            terms=[],
            centers=[],
            scales=[],
            beta=[math.log(25_000_000.0)],
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
            patch.object(
                forecast_service,
                "load_opening_weekend_daily_actuals",
                return_value=weekend_actuals(None, None, None),
            ),
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

    def test_forecast_movie_uses_raw_bop_when_pre_release_estimate_exists_but_model_is_untrained(self) -> None:
        candidate = forecast_service.ForecastCandidate(
            movie_id=1,
            title="Sparse Early BOP Movie",
            release_year=2026,
            opening_date=dt.date(2026, 7, 17),
            target_start_date=dt.date(2026, 7, 17),
            target_end_date=dt.date(2026, 7, 19),
            target_day_count=3,
            target_type="3_day",
            latest_forecast_date=dt.date(2026, 7, 3),
            latest_bop_midpoint=30_000_000.0,
            source_movie_title="Sparse Early BOP Movie",
        )
        row = {
            "snapshot_day": -14,
            "as_of_date": "2026-07-03",
            "bop_forecast_available": 1.0,
            "bop_forecast_midpoint": 30_000_000.0,
            "bop_forecast_published_date": "2026-07-03",
            "movie_id": 1,
            "opening_date": "2026-07-17",
            "target_start_date": "2026-07-17",
            "target_end_date": "2026-07-19",
            "target_day_count": 3,
            "target_type": "3_day",
            "bop_estimate_bucket_under_15m": 0.0,
            "bop_estimate_bucket_15_30m": 1.0,
            "bop_estimate_bucket_30_60m": 0.0,
            "bop_estimate_bucket_60_100m": 0.0,
            "bop_estimate_bucket_100m_plus": 0.0,
            "wiki_available": 1.0,
            "competitor_count_lag7": 1.0,
            "opening_weekend_revenue_usd": 0.0,
        }
        bundle = forecast_service.ForecastModelBundle(
            train_start_year=2022,
            train_end_year=2025,
            min_opening_day_gross=5_000_000,
            models={},
        )
        with (
            patch.object(forecast_service, "latest_candidate_for_movie", return_value=candidate),
            patch.object(forecast_service, "get_model_bundle", return_value=bundle),
            patch.object(forecast_service, "build_movie_feature_rows", return_value=([row], False)),
            patch.object(
                forecast_service,
                "load_opening_weekend_daily_actuals",
                return_value=weekend_actuals(None, None, None),
            ),
        ):
            payload = forecast_service.forecast_movie(Mock(), movie_id=1, today=dt.date(2026, 7, 3))

        snapshots = payload["snapshots"]  # type: ignore[index]
        self.assertEqual("ok", snapshots[0]["status"])
        self.assertEqual(forecast_service.RAW_MODEL, snapshots[0]["model"])
        self.assertEqual("bop", snapshots[0]["prediction_source"])
        self.assertAlmostEqual(
            30_000_000.0,
            snapshots[0]["predicted_opening_weekend_revenue_usd"],
            delta=0.01,
        )
        self.assertEqual(0, snapshots[0]["prediction_interval_train_n"])
        self.assertEqual("-", snapshots[0]["prediction_confidence_label"])
        self.assertEqual("unavailable", snapshots[0]["prediction_confidence_source"])
        self.assertEqual(0, snapshots[0]["model_train_n"])

    def test_forecast_window_includes_reconciliation_day(self) -> None:
        self.assertEqual(-14, min(forecast_service.SNAPSHOT_DAYS))
        self.assertEqual(5, max(forecast_service.SNAPSHOT_DAYS))
        self.assertEqual("reconciliation", forecast_service.forecast_stage_for_snapshot(3))
        self.assertEqual("in_window", forecast_service.forecast_stage_for_snapshot(2))
        self.assertEqual("in_window", forecast_service.forecast_stage_for_snapshot(4, target_day_count=5))

    def test_prediction_certainty_uses_interval_width_and_final_actuals(self) -> None:
        narrow = forecast_service.prediction_certainty(100.0, 90.0, 110.0)
        wide = forecast_service.prediction_certainty(100.0, 50.0, 150.0)
        final = forecast_service.prediction_certainty(None, None, None, is_final_actual=True)

        self.assertGreater(narrow["prediction_certainty"], wide["prediction_certainty"])
        self.assertEqual("83%", narrow["prediction_certainty_label"])
        self.assertEqual("50%", wide["prediction_certainty_label"])
        self.assertEqual("100%", final["prediction_certainty_label"])

    def test_stabilize_intervals_floors_shrunk_rows_and_prevents_later_widening(self) -> None:
        snapshots = [
            {
                "snapshot_day": -3,
                "target_type": "3_day",
                "forecast_state": "pre_release",
                "predicted_opening_weekend_revenue_usd": 100.0,
                "predicted_lower_80_opening_weekend_revenue_usd": 70.0,
                "predicted_upper_80_opening_weekend_revenue_usd": 130.0,
                "prediction_interval_train_n": 10,
                "prediction_interval_source": "global",
                "prediction_interval_method": forecast_service.PRIMARY_INTERVAL_METHOD,
            },
            {
                "snapshot_day": -2,
                "target_type": "3_day",
                "forecast_state": "pre_release",
                "predicted_opening_weekend_revenue_usd": 100.0,
                "predicted_lower_80_opening_weekend_revenue_usd": 99.0,
                "predicted_upper_80_opening_weekend_revenue_usd": 101.0,
                "prediction_interval_train_n": 10,
                "prediction_interval_source": "global",
                "prediction_interval_method": forecast_service.PRIMARY_INTERVAL_METHOD,
            },
            {
                "snapshot_day": 0,
                "target_type": "3_day",
                "forecast_state": "pre_release",
                "predicted_opening_weekend_revenue_usd": 100.0,
                "predicted_lower_80_opening_weekend_revenue_usd": 50.0,
                "predicted_upper_80_opening_weekend_revenue_usd": 150.0,
                "prediction_interval_train_n": 10,
                "prediction_interval_source": "global",
                "prediction_interval_method": forecast_service.PRIMARY_INTERVAL_METHOD,
            },
        ]

        forecast_service.stabilize_prediction_intervals(snapshots)

        widths = [row["prediction_uncertainty_width_pct"] for row in snapshots]
        self.assertEqual([0.6, 0.35, 0.35], widths)
        self.assertEqual("$82", snapshots[1]["predicted_lower_80_label"])
        self.assertEqual("$118", snapshots[1]["predicted_upper_80_label"])

    def test_forecast_chart_shows_all_window_ticks_and_grid(self) -> None:
        snapshots = [
            {
                "status": "ok",
                "snapshot_day": day,
                "predicted_opening_weekend_revenue_usd": 20_000_000.0 + day,
                "predicted_lower_80_opening_weekend_revenue_usd": 18_000_000.0 + day,
                "predicted_upper_80_opening_weekend_revenue_usd": 22_000_000.0 + day,
            }
            for day in (-14, -1, 3)
        ]

        svg = forecast_service.forecast_chart_svg("Chart Movie", snapshots, actual_opening_weekend=None)

        for day in range(-14, 4):
            self.assertIn(f">{day}</text>", svg)
        self.assertIn("days from opening", svg)
        self.assertGreaterEqual(svg.count('stroke="#edf1f3"'), 10)

    def test_daily_actual_lookup_locks_only_non_estimate_rows(self) -> None:
        conn = FakeConn(
            [
                (0, dt.date(2026, 7, 3), 11_000_000, 0, "actual-url", "fetched"),
                (1, dt.date(2026, 7, 4), 12_000_000, 1, "estimate-url", "fetched"),
            ]
        )

        rows = forecast_service.load_opening_weekend_daily_actuals(
            conn,
            movie_id=7,
            opening_date=dt.date(2026, 7, 3),
        )

        self.assertTrue(rows[0]["is_actual"])
        self.assertFalse(rows[1]["is_actual"])
        self.assertTrue(rows[1]["is_estimate"])
        self.assertEqual("missing", rows[2]["status"])
        sql = conn.calls[0][0]
        self.assertIn("dbo.source = 'the_numbers'", sql)
        self.assertIn("dbo.is_preview = 0", sql)
        self.assertIn("dbo.is_estimate", sql)

    def test_partial_weekend_reconciliation_uses_remainder_model(self) -> None:
        bundle = forecast_service.ForecastModelBundle(
            train_start_year=2022,
            train_end_year=2025,
            min_opening_day_gross=5_000_000,
            models={},
            remainder_models={
                (0,): forecast_service.WeekendRemainderModel((0,), 10, 2.0, 1 / 3),
                (0, 1): forecast_service.WeekendRemainderModel((0, 1), 10, 0.5, 2 / 3),
            },
        )

        friday = forecast_service.reconcile_snapshot_prediction(
            bundle=bundle,
            snapshot_day=1,
            base_prediction_usd=30_000_000,
            base_lower_usd=20_000_000,
            base_upper_usd=40_000_000,
            weekend_actuals=weekend_actuals(10_000_000, None, None),
        )
        saturday = forecast_service.reconcile_snapshot_prediction(
            bundle=bundle,
            snapshot_day=2,
            base_prediction_usd=30_000_000,
            base_lower_usd=20_000_000,
            base_upper_usd=40_000_000,
            weekend_actuals=weekend_actuals(10_000_000, 12_000_000, None),
        )
        final = forecast_service.reconcile_snapshot_prediction(
            bundle=bundle,
            snapshot_day=3,
            base_prediction_usd=30_000_000,
            base_lower_usd=20_000_000,
            base_upper_usd=40_000_000,
            weekend_actuals=weekend_actuals(10_000_000, 12_000_000, 8_000_000),
        )

        self.assertEqual(30_000_000, friday["reconciled_opening_weekend_prediction_usd"])
        self.assertEqual(33_000_000, saturday["reconciled_opening_weekend_prediction_usd"])
        self.assertEqual(30_000_000, final["reconciled_opening_weekend_prediction_usd"])
        self.assertEqual("final", final["reconciliation_status"])

    def test_selected_model_prefers_wiki_residual_through_t_minus_2(self) -> None:
        raw = forecast_service.ForecastDayModel(
            snapshot_day=-2,
            model_name=forecast_service.RAW_MODEL,
            terms=[],
            train_rows=[],
            fitted=None,
            bucket_residuals={},
            global_residuals=[],
        )
        wiki_residual = forecast_service.ForecastDayModel(
            snapshot_day=-2,
            model_name=forecast_service.WIKI_RESIDUAL_MODEL,
            terms=[],
            train_rows=[],
            fitted=None,
            bucket_residuals={},
            global_residuals=[],
        )
        bundle = forecast_service.ForecastModelBundle(
            train_start_year=2022,
            train_end_year=2025,
            min_opening_day_gross=5_000_000,
            models={
                (-2, forecast_service.RAW_MODEL): raw,
                (-2, forecast_service.WIKI_RESIDUAL_MODEL): wiki_residual,
            },
        )
        row = {
            "snapshot_day": -2,
            "bop_forecast_available": 1.0,
            "wiki_available": 1.0,
            "competitor_count_lag7": 1.0,
        }

        self.assertIs(wiki_residual, forecast_service.selected_model_for_row(bundle, row))

    def test_selected_model_prefers_pre_release_wiki_residual_over_richer_models(self) -> None:
        history_residual = forecast_service.ForecastDayModel(
            snapshot_day=-2,
            model_name=forecast_service.HISTORY_RESIDUAL_MODEL,
            terms=[],
            train_rows=[],
            fitted=None,
            bucket_residuals={},
            global_residuals=[],
        )
        amc_residual = forecast_service.ForecastDayModel(
            snapshot_day=-2,
            model_name=forecast_service.AMC_RESIDUAL_MODEL,
            terms=[],
            train_rows=[],
            fitted=None,
            bucket_residuals={},
            global_residuals=[],
        )
        wiki_residual = forecast_service.ForecastDayModel(
            snapshot_day=-2,
            model_name=forecast_service.WIKI_RESIDUAL_MODEL,
            terms=[],
            train_rows=[],
            fitted=None,
            bucket_residuals={},
            global_residuals=[],
        )
        bundle = forecast_service.ForecastModelBundle(
            train_start_year=2022,
            train_end_year=2025,
            min_opening_day_gross=5_000_000,
            models={
                (-2, forecast_service.HISTORY_RESIDUAL_MODEL): history_residual,
                (-2, forecast_service.AMC_RESIDUAL_MODEL): amc_residual,
                (-2, forecast_service.WIKI_RESIDUAL_MODEL): wiki_residual,
                (-2, forecast_service.RESIDUAL_MODEL): forecast_service.ForecastDayModel(
                    snapshot_day=-2,
                    model_name=forecast_service.RESIDUAL_MODEL,
                    terms=[],
                    train_rows=[],
                    fitted=None,
                    bucket_residuals={},
                    global_residuals=[],
                ),
            },
        )
        row = {
            "snapshot_day": -2,
            "bop_forecast_available": 1.0,
            "bop_forecast_count_as_of": 2.0,
            "wiki_available": 1.0,
            "competitor_count_lag7": 1.0,
            "amc_available": 1.0,
        }

        self.assertIs(wiki_residual, forecast_service.selected_model_for_row(bundle, row))

    def test_selected_model_uses_pre_release_wiki_residual_at_t_minus_1(self) -> None:
        wiki_residual = forecast_service.ForecastDayModel(
            snapshot_day=-1,
            model_name=forecast_service.WIKI_RESIDUAL_MODEL,
            terms=[],
            train_rows=[],
            fitted=None,
            bucket_residuals={},
            global_residuals=[],
        )
        competition_residual = forecast_service.ForecastDayModel(
            snapshot_day=-1,
            model_name=forecast_service.RESIDUAL_MODEL,
            terms=[],
            train_rows=[],
            fitted=None,
            bucket_residuals={},
            global_residuals=[],
        )
        bundle = forecast_service.ForecastModelBundle(
            train_start_year=2022,
            train_end_year=2025,
            min_opening_day_gross=5_000_000,
            models={
                (-1, forecast_service.WIKI_RESIDUAL_MODEL): wiki_residual,
                (-1, forecast_service.RESIDUAL_MODEL): competition_residual,
            },
        )
        row = {
            "snapshot_day": -1,
            "bop_forecast_available": 1.0,
            "wiki_available": 1.0,
            "competitor_count_lag7": 1.0,
        }

        self.assertIs(wiki_residual, forecast_service.selected_model_for_row(bundle, row))

    def test_selected_model_uses_raw_bop_until_t_minus_3_and_wiki_residual_for_t_minus_2_to_t_minus_1(self) -> None:
        models: dict[tuple[int, str], forecast_service.ForecastDayModel] = {}
        for snapshot_day in range(-14, 0):
            models[(snapshot_day, forecast_service.RAW_MODEL)] = forecast_service.ForecastDayModel(
                snapshot_day=snapshot_day,
                model_name=forecast_service.RAW_MODEL,
                terms=[],
                train_rows=[{"snapshot_day": snapshot_day}],
                fitted=None,
                bucket_residuals={},
                global_residuals=[],
            )
            models[(snapshot_day, forecast_service.WIKI_RESIDUAL_MODEL)] = forecast_service.ForecastDayModel(
                snapshot_day=snapshot_day,
                model_name=forecast_service.WIKI_RESIDUAL_MODEL,
                terms=["log1p_V"],
                train_rows=[{"snapshot_day": snapshot_day}],
                fitted=None,
                bucket_residuals={},
                global_residuals=[],
            )
        bundle = forecast_service.ForecastModelBundle(
            train_start_year=2022,
            train_end_year=2025,
            min_opening_day_gross=5_000_000,
            models=models,
        )

        for snapshot_day in range(-14, 0):
            with self.subTest(snapshot_day=snapshot_day):
                row = {
                    "snapshot_day": snapshot_day,
                    "bop_forecast_available": 1.0,
                    "wiki_available": 1.0,
                    "competitor_count_lag7": 1.0,
                }
                expected_model = (
                    forecast_service.WIKI_RESIDUAL_MODEL
                    if snapshot_day in forecast_service.PRE_RELEASE_WIKI_RESIDUAL_LIVE_DAYS
                    else forecast_service.RAW_MODEL
                )
                self.assertIs(
                    models[(snapshot_day, expected_model)],
                    forecast_service.selected_model_for_row(bundle, row),
                )

    def test_selected_model_uses_configured_competition_switch_day(self) -> None:
        wiki_residual = forecast_service.ForecastDayModel(
            snapshot_day=0,
            model_name=forecast_service.WIKI_RESIDUAL_MODEL,
            terms=[],
            train_rows=[],
            fitted=None,
            bucket_residuals={},
            global_residuals=[],
        )
        competition_residual = forecast_service.ForecastDayModel(
            snapshot_day=0,
            model_name=forecast_service.RESIDUAL_MODEL,
            terms=[],
            train_rows=[],
            fitted=None,
            bucket_residuals={},
            global_residuals=[],
        )
        bundle = forecast_service.ForecastModelBundle(
            train_start_year=2022,
            train_end_year=2025,
            min_opening_day_gross=5_000_000,
            models={
                (0, forecast_service.WIKI_RESIDUAL_MODEL): wiki_residual,
                (0, forecast_service.RESIDUAL_MODEL): competition_residual,
            },
        )
        row = {
            "snapshot_day": 0,
            "bop_forecast_available": 1.0,
            "wiki_available": 1.0,
            "competitor_count_lag7": 1.0,
        }

        with patch.object(forecast_service, "DEFAULT_COMPETITION_RESIDUAL_SWITCH_DAY", 1):
            self.assertIs(wiki_residual, forecast_service.selected_model_for_row(bundle, row))
        with patch.object(forecast_service, "DEFAULT_COMPETITION_RESIDUAL_SWITCH_DAY", 0):
            self.assertIs(competition_residual, forecast_service.selected_model_for_row(bundle, row))

    def test_selected_model_falls_back_to_exact_raw_bop_when_pre_release_wiki_is_untrained(self) -> None:
        nearest_wiki_residual = forecast_service.ForecastDayModel(
            snapshot_day=-2,
            model_name=forecast_service.WIKI_RESIDUAL_MODEL,
            terms=[],
            train_rows=[],
            fitted=None,
            bucket_residuals={},
            global_residuals=[],
        )
        raw = forecast_service.ForecastDayModel(
            snapshot_day=-3,
            model_name=forecast_service.RAW_MODEL,
            terms=[],
            train_rows=[],
            fitted=None,
            bucket_residuals={},
            global_residuals=[],
        )
        bundle = forecast_service.ForecastModelBundle(
            train_start_year=2022,
            train_end_year=2025,
            min_opening_day_gross=5_000_000,
            models={
                (-2, forecast_service.WIKI_RESIDUAL_MODEL): nearest_wiki_residual,
                (-3, forecast_service.RAW_MODEL): raw,
            },
        )
        row = {
            "snapshot_day": -3,
            "bop_forecast_available": 1.0,
            "wiki_available": 1.0,
            "competitor_count_lag7": 1.0,
        }

        self.assertIs(raw, forecast_service.selected_model_for_row(bundle, row))

    def test_selected_model_creates_raw_bop_fallback_when_no_pre_release_model_is_trained(self) -> None:
        bundle = forecast_service.ForecastModelBundle(
            train_start_year=2022,
            train_end_year=2025,
            min_opening_day_gross=5_000_000,
            models={},
        )
        row = {
            "snapshot_day": -14,
            "bop_forecast_available": 1.0,
            "wiki_available": 1.0,
            "competitor_count_lag7": 1.0,
        }

        model = forecast_service.selected_model_for_row(bundle, row)

        self.assertIsNotNone(model)
        assert model is not None
        self.assertEqual(forecast_service.RAW_MODEL, model.model_name)
        self.assertEqual(-14, model.snapshot_day)
        self.assertEqual([], model.global_residuals)

    def test_selected_model_prefers_amc_only_when_amc_model_is_trained(self) -> None:
        raw = forecast_service.ForecastDayModel(
            snapshot_day=1,
            model_name=forecast_service.RAW_MODEL,
            terms=[],
            train_rows=[],
            fitted=None,
            bucket_residuals={},
            global_residuals=[],
        )
        amc = forecast_service.ForecastDayModel(
            snapshot_day=1,
            model_name=forecast_service.AMC_RESIDUAL_MODEL,
            terms=[],
            train_rows=[],
            fitted=None,
            bucket_residuals={},
            global_residuals=[],
        )
        row = {
            "snapshot_day": 1,
            "bop_forecast_available": 1.0,
            "wiki_available": 0.0,
            "competitor_count_lag7": 0.0,
            "amc_available": 1.0,
        }
        without_amc = forecast_service.ForecastModelBundle(2022, 2025, 5_000_000, {(1, forecast_service.RAW_MODEL): raw})
        with_amc = forecast_service.ForecastModelBundle(
            2022,
            2025,
            5_000_000,
            {
                (1, forecast_service.RAW_MODEL): raw,
                (1, forecast_service.AMC_RESIDUAL_MODEL): amc,
            },
        )

        self.assertIs(raw, forecast_service.selected_model_for_row(without_amc, row))
        self.assertIs(amc, forecast_service.selected_model_for_row(with_amc, row))

    def test_model_cache_reuses_empty_training_bundle(self) -> None:
        conn = Mock()
        with patch.object(forecast_service.opening, "load_opening_weekend_movies", return_value=[]) as load_movies:
            first = forecast_service.get_model_bundle(conn)
            second = forecast_service.get_model_bundle(conn)

        self.assertIs(first, second)
        load_movies.assert_called_once()


if __name__ == "__main__":
    unittest.main()
