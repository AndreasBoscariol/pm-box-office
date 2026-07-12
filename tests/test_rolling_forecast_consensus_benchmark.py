from __future__ import annotations

import pandas as pd

from eda.Prior.rolling_forecast_consensus_benchmark import (
    BenchmarkConfig,
    add_locked_point_forecasts,
    add_range_features_to_consensus_panel,
    build_consensus_panel,
    build_point_candidate_rolling_scores,
    build_range_feature_effect_rolling,
    build_source_common_support_paired_scores,
    build_source_data_coverage_audit,
    build_source_error_correlation,
    build_source_individual_performance,
    build_source_range_calibration,
    build_source_range_resolution,
    build_source_origin_panel,
)


def _config() -> BenchmarkConfig:
    return BenchmarkConfig(
        origin_days=(-1,),
        excluded_estimate_sources=(),
        excluded_release_years=(),
        recency_lambdas=(0.0, 1.0),
        train_years=(2025,),
        test_start_year=2026,
        min_source_reliability_n=1,
        source_bias_shrink_k=10,
        max_source_age_days=(7,),
        min_train_n_for_model_selection=1,
        interval_shrink_k=20,
    )


def _openings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "release_run_id": 10,
                "movie_id": 20,
                "title": "Minions & Monsters",
                "opening_date": pd.Timestamp("2026-07-01"),
                "opening_weekend_start": pd.Timestamp("2026-07-03"),
                "release_year": 2026,
                "release_month": 7,
                "season_bucket": "summer",
                "market": "US_CA",
                "release_type": "wide",
                "release_width_bucket": "wide",
                "is_wide_release": True,
                "is_large_release": True,
                "genre": "Animation",
                "distributor": "Universal",
                "franchise": None,
                "is_franchise": False,
                "opening_weekend_theaters": 4000,
                "opening_weekend_gross_usd": 70_000_000,
            }
        ]
    )


def test_source_origin_panel_excludes_non_3_day_estimates() -> None:
    estimates = pd.DataFrame(
        [
            {
                "eda_estimate_id": 1,
                "estimate_source": "boxofficetheory",
                "source_prediction_id": 100,
                "release_run_id": 10,
                "movie_id": 20,
                "title": "Minions & Monsters",
                "opening_date": pd.Timestamp("2026-07-01"),
                "opening_weekend_start": pd.Timestamp("2026-07-03"),
                "actual_opening_weekend_gross_usd": 70_000_000,
                "release_width_bucket": "wide",
                "genre": "Animation",
                "distributor": "Universal",
                "franchise": None,
                "is_franchise": False,
                "estimate_date": pd.Timestamp("2026-07-02"),
                "target_start_date": pd.NaT,
                "target_end_date": pd.NaT,
                "target_day_count": 5,
                "forecast_metric": "domestic_opening_weekend",
                "estimate_low_usd": 80_000_000,
                "estimate_high_usd": 90_000_000,
                "estimate_mid_usd": 85_000_000,
                "estimate_width_usd": 10_000_000,
                "actual_inside_range": False,
                "days_before_opening_weekend": 1,
            },
            {
                "eda_estimate_id": 2,
                "estimate_source": "boxofficetheory",
                "source_prediction_id": 101,
                "release_run_id": 10,
                "movie_id": 20,
                "title": "Minions & Monsters",
                "opening_date": pd.Timestamp("2026-07-01"),
                "opening_weekend_start": pd.Timestamp("2026-07-03"),
                "actual_opening_weekend_gross_usd": 70_000_000,
                "release_width_bucket": "wide",
                "genre": "Animation",
                "distributor": "Universal",
                "franchise": None,
                "is_franchise": False,
                "estimate_date": pd.Timestamp("2026-07-02"),
                "target_start_date": pd.NaT,
                "target_end_date": pd.NaT,
                "target_day_count": 3,
                "forecast_metric": "domestic_opening_weekend",
                "estimate_low_usd": 60_000_000,
                "estimate_high_usd": 70_000_000,
                "estimate_mid_usd": 65_000_000,
                "estimate_width_usd": 10_000_000,
                "actual_inside_range": True,
                "days_before_opening_weekend": 1,
            },
        ]
    )

    source_panel, audit = build_source_origin_panel(
        _openings(),
        estimates,
        _config(),
        return_alignment_audit=True,
    )

    assert source_panel["source_prediction_id"].tolist() == [101]
    excluded = audit.loc[audit["source_prediction_id"].eq(100)].iloc[0]
    assert excluded["exclusion_reason"] == "non_3_day_target"


def test_source_audit_builders_emit_expected_outputs() -> None:
    openings = _openings()
    openings.loc[0, "release_year"] = 2026
    estimates = pd.DataFrame(
        [
            {
                "eda_estimate_id": 1,
                "estimate_source": "source_a",
                "source_prediction_id": 100,
                "release_run_id": 10,
                "movie_id": 20,
                "title": "Minions & Monsters",
                "opening_date": pd.Timestamp("2026-07-01"),
                "opening_weekend_start": pd.Timestamp("2026-07-03"),
                "actual_opening_weekend_gross_usd": 70_000_000,
                "release_width_bucket": "wide",
                "genre": "Animation",
                "distributor": "Universal",
                "franchise": None,
                "is_franchise": False,
                "estimate_date": pd.Timestamp("2026-07-02"),
                "target_start_date": pd.NaT,
                "target_end_date": pd.NaT,
                "target_day_count": 3,
                "forecast_metric": "domestic_opening_weekend",
                "estimate_low_usd": 60_000_000,
                "estimate_high_usd": 80_000_000,
                "estimate_mid_usd": 70_000_000,
                "estimate_width_usd": 20_000_000,
                "actual_inside_range": True,
                "days_before_opening_weekend": 1,
            },
            {
                "eda_estimate_id": 2,
                "estimate_source": "source_b",
                "source_prediction_id": 101,
                "release_run_id": 10,
                "movie_id": 20,
                "title": "Minions & Monsters",
                "opening_date": pd.Timestamp("2026-07-01"),
                "opening_weekend_start": pd.Timestamp("2026-07-03"),
                "actual_opening_weekend_gross_usd": 70_000_000,
                "release_width_bucket": "wide",
                "genre": "Animation",
                "distributor": "Universal",
                "franchise": None,
                "is_franchise": False,
                "estimate_date": pd.Timestamp("2026-07-02"),
                "target_start_date": pd.NaT,
                "target_end_date": pd.NaT,
                "target_day_count": 3,
                "forecast_metric": "domestic_opening_weekend",
                "estimate_low_usd": 65_000_000,
                "estimate_high_usd": 75_000_000,
                "estimate_mid_usd": 70_000_000,
                "estimate_width_usd": 10_000_000,
                "actual_inside_range": True,
                "days_before_opening_weekend": 1,
            },
        ]
    )

    source_panel, alignment_audit = build_source_origin_panel(
        openings,
        estimates,
        _config(),
        return_alignment_audit=True,
    )
    consensus_panel = add_range_features_to_consensus_panel(build_consensus_panel(source_panel, _config()), source_panel)
    locked_panel = add_locked_point_forecasts(consensus_panel)

    assert not build_source_data_coverage_audit(source_panel, alignment_audit, openings).empty
    assert not build_source_individual_performance(source_panel).empty
    assert not build_source_common_support_paired_scores(source_panel, locked_panel).empty
    assert not build_source_error_correlation(source_panel).empty
    assert not build_source_range_calibration(source_panel).empty
    assert not build_source_range_resolution(source_panel).empty
    assert not build_range_feature_effect_rolling(locked_panel).empty
    assert not build_point_candidate_rolling_scores(consensus_panel).empty
