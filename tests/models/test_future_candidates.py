from __future__ import annotations

from datetime import date

import pandas as pd

from models.boxoffice.future_candidates import (
    _candidate_rows,
    _estimate_queries,
    build_daily_baseline,
    build_pre_release_panel,
    normalize_title_key,
    virtual_release_run_id,
)


class _FakeRegclassCursor:
    def __init__(self, exists: bool) -> None:
        self.exists = exists

    def fetchone(self) -> tuple[bool]:
        return (self.exists,)


class _FakeRelationConn:
    def __init__(self, relations: set[str]) -> None:
        self.relations = relations

    def execute(self, _sql: str, params: tuple[str, ...]) -> _FakeRegclassCursor:
        relation = params[0].split(".")[-1]
        return _FakeRegclassCursor(relation in self.relations)


def test_normalize_title_key_collapses_release_qualifiers() -> None:
    assert normalize_title_key("Evil Dead Burn (Wide)") == "evil dead burn"
    assert normalize_title_key("Evil Dead Burn (2026)") == "evil dead burn"


def test_future_candidate_rows_accept_bare_empty_source_frames() -> None:
    candidates = _candidate_rows(pd.DataFrame(), pd.DataFrame())

    assert candidates.empty
    assert list(candidates.columns) == ["title_key", "opening_weekend_start", "movie_id", "title"]


def test_future_estimate_queries_include_boxofficetheory_substack() -> None:
    queries = _estimate_queries(_FakeRelationConn({"boxofficetheory_substack_predictions"}))

    assert len(queries) == 1
    assert "'boxofficetheory_substack'::text AS estimate_source" in queries[0]
    assert "FROM boxofficetheory_substack_predictions p" in queries[0]
    assert "p.forecast_metric ILIKE '%%opening%%'" in queries[0]
    assert "p.opening_weekend_day_count = 3" in queries[0]


def test_future_estimate_queries_require_3_day_boxofficepro_window() -> None:
    queries = _estimate_queries(_FakeRelationConn({"boxofficepro_weekend_predictions"}))

    assert len(queries) == 1
    assert "p.target_end_date IS NOT NULL" in queries[0]
    assert "(p.target_end_date - p.target_start_date) = 2" in queries[0]


def test_future_estimates_promote_first_weekend_despite_generic_source_label() -> None:
    queries = _estimate_queries(
        _FakeRelationConn({"boxofficereport_weekend_predictions", "eda_movie_openings"})
    )

    assert len(queries) == 1
    assert "OR NOT EXISTS" in queries[0]
    assert "opening.opening_weekend_start < p.target_start_date::date" in queries[0]


def test_future_candidate_prefers_amc_identity_over_estimate_identity() -> None:
    estimates = pd.DataFrame(
        [
            {
                "estimate_source": "boxofficepro",
                "source_movie_id": 6407,
                "source_movie_title": "Evil Dead Burn",
                "estimate_date": date(2026, 7, 8),
                "opening_weekend_start": date(2026, 7, 10),
                "estimate_low_usd": 25_000_000,
                "estimate_high_usd": 30_000_000,
                "estimate_mid_usd": 27_500_000,
                "title_key": "evil dead burn",
            }
        ]
    )
    amc = pd.DataFrame(
        [
            {
                "movie_id": 376,
                "title": "Evil Dead Burn",
                "opening_weekend_start": date(2026, 7, 10),
                "title_key": "evil dead burn",
            }
        ]
    )

    candidates = _candidate_rows(estimates, amc)
    release_run_id = virtual_release_run_id(376, date(2026, 7, 10))
    candidates["release_run_id"] = release_run_id

    panel = build_pre_release_panel(candidates, estimates)
    daily = build_daily_baseline(candidates, estimates, as_of_date=date(2026, 7, 9))

    assert candidates.iloc[0]["movie_id"] == 376
    assert panel["release_run_id"].unique().tolist() == [release_run_id]
    assert panel.loc[panel["origin_day"].eq(-1), "dollar_median_consensus_usd"].iloc[0] == 27_500_000
    assert daily.iloc[0]["movie_id"] == 376
    assert daily.iloc[0]["total_forecast_usd"] == 27_500_000
