from __future__ import annotations

import datetime as dt
import math

import pandas as pd
import pytest

from models.boxoffice.live_amc_plugin import (
    LiveAmcPluginConfig,
    amc_interval_cell_key,
    build_amc_interval_policy_from_residuals,
    nowcast_from_panel_row,
    select_target_panel_row,
)
from models.boxoffice.artifacts import ModelArtifacts
from models.boxoffice.schema import MovieOpening
from models.boxoffice.thursday_amc_preview import (
    ThursdayAmcPreviewConfig,
    build_thursday_amc_preview_nowcasts_from_frames,
    preview_nowcast_from_panel_row,
)


def test_nowcast_from_panel_row_uses_historical_pace_and_bridge() -> None:
    rows = []
    for idx, day in enumerate([dt.date(2026, 6, 19), dt.date(2026, 6, 26), dt.date(2026, 7, 3)]):
        rows.append(
            {
                "movie_id": idx + 1,
                "exhibition_date": day,
                "forecast_origin": "14:00",
                "day_of_week": "Friday",
                "s_obs": 50.0,
                "s_final_eod": 100.0,
                "actual_gross_usd": 1_010_000.0,
                "coverage": 0.8,
                "n_snapshots": 10,
                "delay_p50_minutes": 3.0,
                "collection_quality_bucket": "high",
            }
        )
    target = pd.Series(
        {
            "movie_id": 99,
            "exhibition_date": dt.date(2026, 7, 10),
            "forecast_origin": "14:00",
            "day_of_week": "Friday",
            "s_obs": 60.0,
            "s_final_eod": 0.0,
            "actual_gross_usd": float("nan"),
            "coverage": 0.9,
            "n_snapshots": 12,
            "delay_p50_minutes": 4.0,
            "collection_quality_bucket": "high",
        }
    )
    panel = pd.DataFrame(rows + [target.to_dict()])

    result = nowcast_from_panel_row(
        panel,
        target,
        config=LiveAmcPluginConfig(min_train_rows=3, min_sigma_log_daily=0.25),
    )

    assert result is not None
    assert result["feature_quality_bucket"] == "high"
    assert result["amc_snapshot_count"] == 12
    assert math.isclose(float(result["pred_daily_gross_usd"]), 1_200_000.0, rel_tol=0.02)
    assert float(result["sigma_log_daily"]) >= 0.25


def test_select_target_panel_row_prefers_observed_candidate_when_duplicate_mappings_exist() -> None:
    panel = pd.DataFrame(
        [
            {
                "movie_id": 378,
                "amc_movie_id": "moana-72474",
                "exhibition_date": dt.date(2026, 7, 11),
                "forecast_origin": "18:00",
                "s_obs": 9686.0,
                "n_snapshots": 231,
                "c_obs": 55783.0,
                "c_scheduled_known": 55783.0,
                "n_scheduled_showtimes": 5976,
            },
            {
                "movie_id": 378,
                "amc_movie_id": "the-invite-82975",
                "exhibition_date": dt.date(2026, 7, 11),
                "forecast_origin": "18:00",
                "s_obs": 0.0,
                "n_snapshots": 0,
                "c_obs": 0.0,
                "c_scheduled_known": 0.0,
                "n_scheduled_showtimes": 2,
            },
        ]
    )

    row = select_target_panel_row(
        panel,
        movie_id=378,
        exhibition_date=dt.date(2026, 7, 11),
        forecast_origin="18:00",
    )

    assert row is not None
    assert row["amc_movie_id"] == "moana-72474"
    assert row["s_obs"] == 9686.0


def test_amc_interval_policy_builds_specific_and_fallback_cells() -> None:
    residuals = pd.DataFrame(
        [
            {
                "sample_key": "top_hybrid_30",
                "regime": "live_friday",
                "forecast_origin": "16:00",
                "target_day": "Friday",
                "feature_quality_bucket": "high",
                "log_residual": value,
            }
            for value in [-0.20, -0.10, 0.0, 0.10, 0.20]
        ]
    )

    policy = build_amc_interval_policy_from_residuals(
        residuals,
        sample_key="top_hybrid_30",
        min_bucket_n=3,
    )

    specific_key = amc_interval_cell_key(
        {
            "sample_key": "top_hybrid_30",
            "regime": "live_friday",
            "forecast_origin": "16:00",
            "target_day": "Friday",
            "feature_quality_bucket": "high",
        }
    )
    sample_key = amc_interval_cell_key({"sample_key": "top_hybrid_30"})

    assert policy["cells"][specific_key]["n"] == 5
    assert policy["cells"][sample_key]["n"] == 5
    assert policy["cells"][specific_key]["hi95_log"] > 0
    assert policy["n_residual_rows"] == 5
    assert policy["n_unique_movie_days"] == 5


def test_amc_policy_cluster_weights_origins_and_partially_pools_small_cells() -> None:
    rows = []
    for movie_id, residual in [(1, 0.0), (2, 0.2), (3, 3.0)]:
        for origin in (["10:00", "12:00", "14:00", "16:00"] if movie_id == 1 else ["16:00"]):
            rows.append(
                {
                    "sample_key": "top_hybrid_30",
                    "regime": "live_friday",
                    "forecast_origin": origin,
                    "target_day": "Friday",
                    "feature_quality_bucket": "low",
                    "movie_id": movie_id,
                    "exhibition_date": dt.date(2026, 6, 5 + movie_id),
                    "log_residual": residual,
                }
            )
    policy = build_amc_interval_policy_from_residuals(
        pd.DataFrame(rows), sample_key="top_hybrid_30", min_bucket_n=2
    )
    exact = policy["cells"][amc_interval_cell_key({
        "sample_key": "top_hybrid_30", "regime": "live_friday", "forecast_origin": "16:00",
        "target_day": "Friday", "feature_quality_bucket": "low",
    })]

    assert policy["n_residual_rows"] == 6
    assert policy["n_unique_movie_days"] == 3
    assert exact["method"] == "partially_pooled_amc_live_log_residual_quantile"
    assert exact["pooling_weight"] < 0.1
    assert math.exp(exact["hi95_log"]) < math.exp(3.0)
    assert len(policy["diagnostics"]["amc_interval_cell_stability"]) == 3
    assert policy["diagnostics"]["amc_tail_case_audit"][-1]["movie_id"] == 3


def test_thursday_amc_preview_nowcast_builds_shadow_ow_prior_from_seats() -> None:
    target_movie = MovieOpening(1, 10, "Example", dt.date(2026, 7, 10))
    preview_dates = [dt.date(2026, 6, 18), dt.date(2026, 6, 25), dt.date(2026, 7, 2), dt.date(2026, 7, 9)]
    movie_ids = [101, 102, 103, 1]
    schedule = pd.DataFrame(
        [
            {
                "movie_id": movie_id,
                "amc_movie_id": movie_id + 1000,
                "exhibition_date": preview_date,
                "n_scheduled_showtimes": 1,
                "n_scheduled_theatres": 1,
                "n_premium_showtimes": 0,
                "premium_format_share": 0.0,
                "n_timezones_scheduled": 1,
                "weighted_scheduled_showtimes": 1.0,
                "c_scheduled_known": 200.0,
                "n_showtimes_with_known_capacity": 1,
            }
            for movie_id, preview_date in zip(movie_ids, preview_dates, strict=True)
        ]
    )
    snapshots = pd.DataFrame(
        [
            {
                "movie_id": movie_id,
                "amc_movie_id": movie_id + 1000,
                "exhibition_date": preview_date,
                "showtime_id": movie_id * 10,
                "amc_theatre_id": 1,
                "timezone": "America/New_York",
                "local_calendar_start_at": f"{preview_date} 19:00:00",
                "starts_at_utc": pd.Timestamp(f"{preview_date} 23:00:00Z"),
                "business_minute": 19 * 60,
                "showtime_block": "evening",
                "seat_snapshot_id": movie_id * 100,
                "target_offset_minutes": 0,
                "scheduled_for": pd.Timestamp(f"{preview_date} 20:00:00Z"),
                "observed_at": pd.Timestamp(f"{preview_date} 20:00:00Z"),
                "lateness_seconds": 0.0,
                "total_seats": 200.0,
                "filled_or_unavailable_seats": 60.0 if movie_id == 1 else 50.0,
                "fill_rate": 0.30 if movie_id == 1 else 0.25,
                "parse_method": "rsc",
                "analysis_weight": 1.0,
                "is_late_rescue": False,
            }
            for movie_id, preview_date in zip(movie_ids, preview_dates, strict=True)
        ]
    )
    actuals = pd.DataFrame(
        [
            {
                "release_run_id": idx,
                "movie_id": movie_id,
                "title": f"Historical {idx}",
                "release_date": preview_date + dt.timedelta(days=1),
                "exhibition_date": preview_date,
                "day_number": -1,
                "actual_gross_usd": 1_010_000.0,
                "actual_theaters": 3000.0,
                "is_preview": 1,
            }
            for idx, (movie_id, preview_date) in enumerate(zip(movie_ids[:3], preview_dates[:3], strict=True), start=1)
        ]
    )
    artifacts = ModelArtifacts(
        model_version="boxoffice_test",
        artifact_dir=".",
        manifest={},
        thursday_amc_preview_policy={
            "enabled": True,
            "selected_point_model": "multiplicative",
            "min_train_rows": 3,
            "policy_version": "test_thursday_amc",
        },
        thursday_preview_policy={"alpha": math.log(1.2), "beta": 0.0, "policy_version": "test"},
        daily_baseline=pd.DataFrame(
            [{"movie_id": 1, "release_run_id": 10, "pre_fri_usd": 30_000_000, "pre_sat_usd": 35_000_000, "pre_sun_usd": 25_000_000}]
        ),
    )

    nowcasts = build_thursday_amc_preview_nowcasts_from_frames(
        schedule=schedule,
        snapshots=snapshots,
        actuals=actuals,
        movies=[target_movie],
        as_of_utc=dt.datetime(2026, 7, 9, 21, 0, tzinfo=dt.timezone.utc),
        artifacts=artifacts,
        sample_key="top_hybrid_30",
        config=ThursdayAmcPreviewConfig(
            origins=("16:00",),
            live_amc_config=LiveAmcPluginConfig(min_train_rows=3, min_sigma_log_daily=0.25),
        ),
    )

    assert len(nowcasts) == 1
    row = nowcasts.iloc[0]
    assert row["forecast_origin"] == "16:00"
    assert row["predicted_thursday_previews_usd"] == pytest.approx(1_200_000.0, rel=0.02)
    assert row["preview_amc_training_rows"] == 3
    assert bool(row["preview_amc_nowcast_trained"]) is True
    assert row["shadow_ow_prior_source"] == "thursday_amc_preview_nowcast"
    assert row["shadow_ow_prior_usd"] == pytest.approx(108_000_000)
    assert bool(row["thursday_amc_seats_collected"]) is True


def test_preview_nowcast_uses_preview_only_bucket_fallback() -> None:
    rows = []
    for idx, (origin, obs) in enumerate([("18:00", 50.0), ("20:00", 60.0), ("EOD", 70.0)], start=1):
        rows.append(
            {
                "movie_id": idx,
                "exhibition_date": dt.date(2026, 6, 1 + idx),
                "forecast_origin": origin,
                "is_preview": 1,
                "s_obs": obs,
                "s_final_eod": obs * 2,
                "c_obs": 100.0,
                "c_scheduled_known": 200.0,
                "actual_gross_usd": (obs * 2 + 1) * 10_000,
                "coverage": 0.8,
                "n_snapshots": 10,
                "delay_p50_minutes": 3.0,
                "staleness_p50_minutes": 4.0,
                "collection_quality_bucket": "medium",
            }
        )
    target = pd.Series(
        {
            "movie_id": 99,
            "exhibition_date": dt.date(2026, 7, 9),
            "forecast_origin": "16:00",
            "is_preview": 1,
            "s_obs": 80.0,
            "s_final_eod": 0.0,
            "c_obs": 100.0,
            "c_scheduled_known": 200.0,
            "actual_gross_usd": float("nan"),
            "coverage": 0.8,
            "n_snapshots": 11,
            "delay_p50_minutes": 3.0,
            "staleness_p50_minutes": 4.0,
            "collection_quality_bucket": "medium",
        }
    )
    panel = pd.DataFrame(rows + [target.to_dict()])

    result = preview_nowcast_from_panel_row(
        panel,
        target,
        policy={"enabled": True, "selected_point_model": "hybrid"},
        min_train_rows=3,
    )

    assert result is not None
    assert result["preview_amc_training_pool"] == "origin_bucket_late"
    assert result["preview_amc_training_rows"] == 3
    assert result["pred_daily_gross_usd"] > 0


def test_thursday_amc_preview_nowcast_does_not_update_prior_without_preview_training() -> None:
    target_movie = MovieOpening(1, 10, "Example", dt.date(2026, 7, 10))
    preview_dates = [dt.date(2026, 6, 18), dt.date(2026, 6, 25), dt.date(2026, 7, 2), dt.date(2026, 7, 9)]
    movie_ids = [101, 102, 103, 1]
    schedule = pd.DataFrame(
        [
            {
                "movie_id": movie_id,
                "amc_movie_id": movie_id + 1000,
                "exhibition_date": preview_date,
                "n_scheduled_showtimes": 1,
                "n_scheduled_theatres": 1,
                "n_premium_showtimes": 0,
                "premium_format_share": 0.0,
                "n_timezones_scheduled": 1,
                "weighted_scheduled_showtimes": 1.0,
                "c_scheduled_known": 200.0,
                "n_showtimes_with_known_capacity": 1,
            }
            for movie_id, preview_date in zip(movie_ids, preview_dates, strict=True)
        ]
    )
    snapshots = pd.DataFrame(
        [
            {
                "movie_id": movie_id,
                "amc_movie_id": movie_id + 1000,
                "exhibition_date": preview_date,
                "showtime_id": movie_id * 10,
                "amc_theatre_id": 1,
                "timezone": "America/New_York",
                "local_calendar_start_at": f"{preview_date} 19:00:00",
                "starts_at_utc": pd.Timestamp(f"{preview_date} 23:00:00Z"),
                "business_minute": 19 * 60,
                "showtime_block": "evening",
                "seat_snapshot_id": movie_id * 100,
                "target_offset_minutes": 0,
                "scheduled_for": pd.Timestamp(f"{preview_date} 20:00:00Z"),
                "observed_at": pd.Timestamp(f"{preview_date} 20:00:00Z"),
                "lateness_seconds": 0.0,
                "total_seats": 200.0,
                "filled_or_unavailable_seats": 60.0 if movie_id == 1 else 50.0,
                "fill_rate": 0.30 if movie_id == 1 else 0.25,
                "parse_method": "rsc",
                "analysis_weight": 1.0,
                "is_late_rescue": False,
            }
            for movie_id, preview_date in zip(movie_ids, preview_dates, strict=True)
        ]
    )
    actuals = pd.DataFrame(
        [
            {
                "release_run_id": idx,
                "movie_id": movie_id,
                "title": f"Historical {idx}",
                "release_date": preview_date + dt.timedelta(days=1),
                "exhibition_date": preview_date,
                "day_number": 1,
                "actual_gross_usd": 1_010_000.0,
                "actual_theaters": 3000.0,
                "is_preview": 0,
            }
            for idx, (movie_id, preview_date) in enumerate(zip(movie_ids[:3], preview_dates[:3], strict=True), start=1)
        ]
    )
    artifacts = ModelArtifacts(
        model_version="boxoffice_test",
        artifact_dir=".",
        manifest={},
        thursday_preview_policy={"alpha": math.log(1.2), "beta": 0.0, "policy_version": "test"},
        daily_baseline=pd.DataFrame(
            [{"movie_id": 1, "release_run_id": 10, "pre_fri_usd": 30_000_000, "pre_sat_usd": 35_000_000, "pre_sun_usd": 25_000_000}]
        ),
    )

    nowcasts = build_thursday_amc_preview_nowcasts_from_frames(
        schedule=schedule,
        snapshots=snapshots,
        actuals=actuals,
        movies=[target_movie],
        as_of_utc=dt.datetime(2026, 7, 9, 21, 0, tzinfo=dt.timezone.utc),
        artifacts=artifacts,
        sample_key="top_hybrid_30",
        config=ThursdayAmcPreviewConfig(
            origins=("16:00",),
            live_amc_config=LiveAmcPluginConfig(min_train_rows=3, min_sigma_log_daily=0.25),
        ),
    )

    # Thursday must not reuse the Friday/Saturday daily-gross bridge when no
    # classified preview training history or frozen Thursday policy exists.
    assert nowcasts.empty
