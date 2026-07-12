import csv
import json
from pathlib import Path

import pandas as pd

from prediction_market_backtest.historical_panel import build_phase9c_historical_panel


class FakeGamma:
    def events(self, **filters):
        yield {
            "markets": [
                {"id": "m1", "clobTokenIds": json.dumps(["101", "102"])},
                {"id": "m2", "clobTokenIds": json.dumps(["201", "202"])},
            ]
        }


class FakeClob:
    def price_history(self, token_id, *, interval="max", fidelity=1):
        return [{"t": "2026-01-08T12:00:00+00:00", "p": "0.40" if token_id == "101" else "0.70"}]


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_phase9c_builds_locked_panel_without_lookahead(tmp_path):
    review = tmp_path / "review"
    review.mkdir()
    buckets = json.dumps([
        {"market_id": "m1", "lower": None, "upper": "10000000", "include_lower": False, "include_upper": False},
        {"market_id": "m2", "lower": "10000000", "upper": None, "include_lower": True, "include_upper": False},
    ])
    decision = {
        "event_id": "e1",
        "event_title": "Movie Opening Weekend",
        "decision": "approved",
        "reviewer": "tester",
        "reviewed_at": "2026-01-01T00:00:00+00:00",
        "matched_movie_id": "1",
        "approved_movie_id": "1",
        "locked": "true",
        "source_warnings": "",
    }
    queue = {"event_id": "e1", "event_slug": "movie-opening-weekend", "bucket_table_json": buckets}
    _write(review / "01_prioritized_review_queue.csv", [queue])
    _write(review / "03_reviewed_event_decisions.csv", [decision])
    _write(review / "04_approved_events.csv", [decision])
    panel = tmp_path / "panel.csv"
    _write(panel, [{
        "movie_id": "1",
        "origin_day": "-1",
        "forecast_origin_date": "2026-01-09",
        "primary_point_forecast_usd": "12000000",
        "actual_opening_weekend_gross_usd": "13000000",
        "release_year": "2026",
        "opening_weekend_start": "2026-01-10",
    }])
    result = build_phase9c_historical_panel(review, tmp_path / "prices", tmp_path / "diag", panel, gamma=FakeGamma(), clob=FakeClob(), bootstrap_iterations=25)
    assert result["paired_movie_origin_rows"] == 1
    assert result["independent_movies"] == 1
    assert result["sample_gate"] == "blocked_small_sample"
    assert len(pd.read_parquet(tmp_path / "prices" / "10_complete_historical_panel.parquet")) == 1
    assert len(pd.read_parquet(tmp_path / "prices" / "14_primary_latest_origin_panel.parquet")) == 1
    audit_rows = list(csv.DictReader((tmp_path / "prices" / "13_event_recovery_audit.csv").open()))
    assert audit_rows[0]["included_in_panel"] == "true"
    paper_rows = list(csv.DictReader((tmp_path / "prices" / "16_historical_price_based_paper_backtest_summary.csv").open()))
    assert paper_rows[0]["label"] == "historical_price_based_paper_backtest"
    decision_rows = list(csv.DictReader((tmp_path / "diag" / "03_decision.csv").open()))
    assert decision_rows[0]["diagnostic_decision"] == "diagnostic_inconclusive_small_sample"
