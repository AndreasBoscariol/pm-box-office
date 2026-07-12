from __future__ import annotations

import math

import numpy as np
import pandas as pd

from eda.During.thursday_preview_lift_diagnostic import (
    build_analysis_panel,
    build_downstream_live_friday_predictions,
    build_downstream_influence_audit,
    build_rolling_predictions,
    build_training_policy_predictions,
    downstream_live_friday_bootstrap,
    downstream_influence_bucket_metrics,
    downstream_leave_one_out,
    downstream_top_influence_removed_metrics,
    evaluate_downstream_live_friday,
    evaluate_models,
    evaluate_training_policies,
    ols_alpha_beta,
    paired_improvement_bootstrap,
    training_policy_paired_bootstrap,
    validate_preview_gross,
)


def _preview_row(
    release_run_id: int = 1,
    movie_id: int = 10,
    *,
    title: str = "Example",
    opening_weekend_start: str = "2026-07-03",
    opening_date: str = "2026-07-03",
    preview_date: str = "2026-07-02",
    preview_gross: float = 10.0,
    actual: float = 120.0,
    daily_id: int = 1,
) -> dict[str, object]:
    return {
        "release_run_id": release_run_id,
        "movie_id": movie_id,
        "title": title,
        "opening_date": opening_date,
        "opening_weekend_start": opening_weekend_start,
        "release_year": int(opening_weekend_start[:4]),
        "release_width_bucket": "wide",
        "release_type": "movie_page_full_run",
        "opening_weekend_theaters": 3000,
        "actual_opening_weekend_gross_usd": actual,
        "daily_box_office_id": daily_id,
        "preview_date": preview_date,
        "preview_gross_usd": preview_gross,
    }


def _panel_row(
    release_run_id: int = 1,
    movie_id: int = 10,
    *,
    origin_day: int = -2,
    forecast_origin_date: str = "2026-07-01",
    latest_estimate_date: str = "2026-07-01",
    baseline: float = 100.0,
) -> dict[str, object]:
    return {
        "release_run_id": release_run_id,
        "movie_id": movie_id,
        "title": "Example",
        "origin_day": origin_day,
        "opening_weekend_start": "2026-07-03",
        "forecast_origin_date": forecast_origin_date,
        "latest_estimate_date": latest_estimate_date,
        "primary_point_forecast_usd": baseline,
        "primary_point_method": "test_method",
        "source_count": 2,
    }


def test_leakage_free_baseline_selection_uses_latest_prior_safe_row() -> None:
    preview = pd.DataFrame([_preview_row()])
    panel = pd.DataFrame(
        [
            _panel_row(origin_day=-3, forecast_origin_date="2026-06-30", latest_estimate_date="2026-06-30", baseline=90),
            _panel_row(origin_day=-2, forecast_origin_date="2026-07-01", latest_estimate_date="2026-07-01", baseline=100),
            _panel_row(origin_day=-1, forecast_origin_date="2026-07-02", latest_estimate_date="2026-07-02", baseline=200),
        ]
    )

    analysis, audit = build_analysis_panel(preview, panel)

    assert len(analysis) == 1
    assert analysis.iloc[0]["baseline_forecast_usd"] == 100
    assert analysis.iloc[0]["baseline_origin_day"] == -2
    assert bool(audit.iloc[0]["included"]) is True


def test_preview_deduplication_and_unclear_aggregation() -> None:
    duplicate = pd.DataFrame(
        [
            _preview_row(daily_id=1, preview_gross=10),
            _preview_row(daily_id=2, preview_gross=10),
        ]
    )
    duplicate_summary = validate_preview_gross(duplicate)
    assert duplicate_summary.iloc[0]["preview_validation_status"] == "duplicate_preview_rows"
    assert duplicate_summary.iloc[0]["preview_gross_usd"] == 10

    unclear = pd.DataFrame(
        [
            _preview_row(daily_id=1, preview_date="2026-07-02", preview_gross=10),
            _preview_row(daily_id=2, preview_date="2026-07-01", preview_gross=2),
        ]
    )
    unclear_summary = validate_preview_gross(unclear)
    assert unclear_summary.iloc[0]["preview_validation_status"] == "unclear_preview_aggregation"
    assert math.isnan(unclear_summary.iloc[0]["preview_gross_usd"])


def test_cohort_filtering_and_x_e_calculation() -> None:
    preview = pd.DataFrame(
        [
            _preview_row(release_run_id=1, movie_id=10, preview_gross=10, actual=120),
            _preview_row(release_run_id=2, movie_id=20, preview_gross=5, actual=0),
            _preview_row(release_run_id=3, movie_id=30, preview_gross=5, actual=80),
        ]
    )
    panel = pd.DataFrame(
        [
            _panel_row(release_run_id=1, movie_id=10, baseline=100),
            _panel_row(release_run_id=2, movie_id=20, baseline=100),
            _panel_row(release_run_id=3, movie_id=30, baseline=-1),
        ]
    )

    analysis, audit = build_analysis_panel(preview, panel)

    assert analysis["movie_id"].tolist() == [10]
    row = analysis.iloc[0]
    assert math.isclose(row["e_log"], math.log(120 / 100))
    assert math.isclose(row["x_log_preview_ratio"], math.log(10 / 100))
    reasons = audit.set_index("movie_id")["exclusion_reason"].to_dict()
    assert reasons[20] == "invalid_actual"
    assert reasons[30] == "non_positive_baseline"


def _synthetic_analysis_panel(n: int = 23) -> pd.DataFrame:
    rows = []
    for i in range(n):
        x = -3.0 + i * 0.1
        e = 0.2 + 0.5 * x
        baseline = 100.0
        rows.append(
            {
                "release_run_id": i + 1,
                "movie_id": i + 100,
                "title": f"Movie {i}",
                "opening_weekend_start": pd.Timestamp("2022-01-07") + pd.Timedelta(days=7 * i),
                "release_year": 2022 if i < 10 else 2023,
                "actual_opening_weekend_gross_usd": baseline * math.exp(e),
                "preview_gross_usd": baseline * math.exp(x),
                "baseline_forecast_usd": baseline,
                "e_log": e,
                "x_log_preview_ratio": x,
            }
        )
    return pd.DataFrame(rows)


def test_rolling_training_uses_only_prior_movies_and_minimum_sample() -> None:
    panel = _synthetic_analysis_panel(22)
    rolling = build_rolling_predictions(panel, min_train_n=20)

    assert rolling.loc[:19, "scored"].eq(False).all()
    assert bool(rolling.loc[20, "scored"]) is True
    assert rolling.loc[20, "rolling_train_n"] == 20
    assert math.isclose(rolling.loc[20, "historical_preview_bias_log"], panel.iloc[:20]["e_log"].mean())

    alpha, beta = ols_alpha_beta(panel.iloc[:20]["x_log_preview_ratio"], panel.iloc[:20]["e_log"])
    expected = 100.0 * math.exp(alpha + beta * panel.loc[20, "x_log_preview_ratio"])
    assert math.isclose(rolling.loc[20, "rolling_alpha"], alpha)
    assert math.isclose(rolling.loc[20, "rolling_beta"], beta)
    assert math.isclose(rolling.loc[20, "forecast_preview_amount_update_usd"], expected)


def test_model_metrics_and_bootstrap_are_paired() -> None:
    panel = _synthetic_analysis_panel(23)
    rolling = build_rolling_predictions(panel, min_train_n=20)

    metrics = evaluate_models(rolling).set_index("model")
    assert metrics.loc["consensus", "n"] == 3
    assert metrics.loc["preview_status_bias", "n"] == 3
    assert metrics.loc["preview_amount_update", "n"] == 3
    assert metrics.loc["preview_amount_update", "RMSE_log"] < metrics.loc["preview_status_bias", "RMSE_log"]
    assert metrics.loc["preview_amount_update", "MAE_log"] < metrics.loc["preview_status_bias", "MAE_log"]
    assert metrics.loc["preview_amount_update", "proportion_improved_vs_preview_status_bias"] == 1.0

    boot = paired_improvement_bootstrap(rolling, resamples=200, seed=7)
    assert set(boot["loss_metric"]) == {"abs_log_error", "squared_log_error", "abs_dollar_error"}
    assert np.isfinite(boot["mean_loss_difference"]).all()


def test_training_policy_predictions_use_common_test_rows_and_correct_windows() -> None:
    pre = _synthetic_analysis_panel(3)
    pre["release_year"] = [2017, 2018, 2019]
    pre["opening_weekend_start"] = pd.to_datetime(["2017-01-06", "2018-01-05", "2019-01-04"])
    current = _synthetic_analysis_panel(23)
    current["release_year"] = [2023] * 10 + [2024] * 10 + [2025] * 3
    current["opening_weekend_start"] = pd.date_range("2023-01-06", periods=23, freq="7D")
    current["release_run_id"] += 100
    current["movie_id"] += 1000
    panel = pd.concat([pre, current], ignore_index=True)

    predictions = build_training_policy_predictions(panel, min_current_train_n=20)

    assert set(predictions["training_policy"]) == {"post_2022_only", "pooled_non_covid"}
    counts = predictions.groupby(["release_run_id", "movie_id"])["training_policy"].nunique()
    assert counts.eq(2).all()
    first_current = predictions.loc[predictions["training_policy"].eq("post_2022_only")].iloc[0]
    first_pooled = predictions.loc[predictions["training_policy"].eq("pooled_non_covid")].iloc[0]
    assert first_current["current_train_n"] == 20
    assert first_current["training_n"] == 20
    assert first_pooled["current_train_n"] == 20
    assert first_pooled["pooled_pre_2020_train_n"] == 3
    assert first_pooled["training_n"] == 23

    metrics = evaluate_training_policies(predictions)
    assert set(metrics["training_policy"]) == {"post_2022_only", "pooled_non_covid"}
    boot = training_policy_paired_bootstrap(predictions, resamples=100, seed=11)
    assert set(boot["loss_metric"]) == {"abs_log_error", "squared_log_error", "abs_dollar_error"}


def test_downstream_live_friday_rescales_daily_components_before_scoring() -> None:
    policy = pd.DataFrame(
        [
            {
                "release_run_id": 1,
                "movie_id": 10,
                "title": "Example",
                "opening_weekend_start": pd.Timestamp("2026-07-03"),
                "release_year": 2026,
                "training_policy": "post_2022_only",
                "forecast_preview_amount_update_usd": 200.0,
            }
        ]
    )
    daily = pd.DataFrame(
        [
            {
                "release_run_id": 1,
                "movie_id": 10,
                "pre_fri_usd": 50.0,
                "pre_sat_usd": 30.0,
                "pre_sun_usd": 20.0,
                "actual_ow_usd": 180.0,
                "total_forecast_usd": 100.0,
            }
        ]
    )

    predictions = build_downstream_live_friday_predictions(policy, daily)

    assert len(predictions) == 1
    row = predictions.iloc[0]
    assert row["component_rescale_factor"] == 2.0
    assert row["updated_friday_component_usd"] == 100.0
    assert row["updated_saturday_component_usd"] == 60.0
    assert row["updated_sunday_component_usd"] == 40.0
    assert row["forecast_original_consensus_prior_usd"] == 100.0
    assert row["forecast_thursday_preview_prior_usd"] == 200.0

    metrics = evaluate_downstream_live_friday(predictions).set_index("model")
    assert metrics.loc["thursday_preview_updated_prior", "MAE_log"] < metrics.loc["original_consensus_prior", "MAE_log"]
    boot = downstream_live_friday_bootstrap(predictions, resamples=20, seed=3)
    assert set(boot["loss_metric"]) == {"abs_log_error", "squared_log_error", "abs_dollar_error"}


def test_downstream_influence_audit_reports_pairwise_and_leave_one_out_metrics() -> None:
    predictions = pd.DataFrame(
        [
            {
                "release_run_id": 1,
                "movie_id": 10,
                "title": "Small",
                "release_year": 2025,
                "forecast_original_consensus_prior_usd": 100.0,
                "forecast_thursday_preview_prior_usd": 120.0,
                "actual_opening_weekend_gross_usd": 125.0,
            },
            {
                "release_run_id": 2,
                "movie_id": 20,
                "title": "Large",
                "release_year": 2025,
                "forecast_original_consensus_prior_usd": 100_000_000.0,
                "forecast_thursday_preview_prior_usd": 130_000_000.0,
                "actual_opening_weekend_gross_usd": 105_000_000.0,
            },
        ]
    )

    audit = build_downstream_influence_audit(predictions)

    assert {
        "movie_id",
        "release_year",
        "original_forecast",
        "updated_forecast",
        "actual_ow",
        "baseline_size_bucket",
        "delta_abs_log_error",
        "delta_abs_dollar_error",
    }.issubset(audit.columns)
    assert audit.loc[audit["movie_id"].eq(10), "delta_abs_log_error"].iloc[0] < 0
    assert audit.loc[audit["movie_id"].eq(20), "delta_abs_dollar_error"].iloc[0] > 0

    buckets = downstream_influence_bucket_metrics(audit)
    assert set(buckets["baseline_size_bucket"]) == {"lt_5m", "100m_plus"}
    loo = downstream_leave_one_out(audit)
    assert len(loo) == 2
    removed = downstream_top_influence_removed_metrics(audit)
    assert set(removed["sample"]) == {"full_sample", "top_dollar_influence_removed"}
