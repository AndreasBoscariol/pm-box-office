from __future__ import annotations

from eda.create_plotting_tables import estimate_union_sql


class _FakeRegclassCursor:
    def __init__(self, value: str | None) -> None:
        self.value = value

    def fetchone(self) -> tuple[str | None]:
        return (self.value,)


class _FakeRelationConn:
    def __init__(self, relations: set[str]) -> None:
        self.relations = relations

    def execute(self, _sql: str, params: tuple[str, ...]) -> _FakeRegclassCursor:
        relation = params[0].split(".")[-1]
        return _FakeRegclassCursor(relation if relation in self.relations else None)


def test_estimate_union_includes_boxofficetheory_substack() -> None:
    sql = estimate_union_sql(_FakeRelationConn({"boxofficetheory_substack_predictions"}))

    assert "'boxofficetheory_substack'::text AS estimate_source" in sql
    assert "FROM boxofficetheory_substack_predictions p" in sql
    assert "LEFT JOIN boxofficetheory_substack_posts post" in sql
    assert "p.opening_weekend_pinpoint_usd::numeric" in sql
    assert "p.opening_weekend_day_count" in sql


def test_estimate_union_includes_edwarddouglas_substack() -> None:
    sql = estimate_union_sql(_FakeRelationConn({"edwarddouglas_substack_predictions"}))

    assert "'edwarddouglas_substack'::text AS estimate_source" in sql
    assert "FROM edwarddouglas_substack_predictions p" in sql
    assert "LEFT JOIN edwarddouglas_substack_posts post" in sql
    assert "p.weekend_forecast_usd::numeric" in sql


def test_estimate_union_carries_target_day_count_for_3_day_filtering() -> None:
    sql = estimate_union_sql(
        _FakeRelationConn(
            {
                "boxofficepro_weekend_predictions",
                "boxofficetheory_predictions",
                "the_numbers_prediction_rows",
                "the_numbers_prediction_images",
                "the_numbers_prediction_articles",
            }
        )
    )

    assert "(p.target_end_date - p.target_start_date + 1)::integer AS target_day_count" in sql
    assert "THEN p.opening_weekend_day_count" in sql
    assert "3::integer AS target_day_count" in sql
