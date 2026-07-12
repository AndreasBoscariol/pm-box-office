from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

from prediction_market_backtest.collector import ClobCollector, MemorySink
from prediction_market_backtest.policy007 import (build_chronological_pool, generate_draws, validate_policy_panel,
                                                   weighted_quantile)


def _panel() -> pd.DataFrame:
    rows=[]
    for movie in range(1,26):
        rows.append(dict(movie_id=movie,release_run_id=movie,origin_day=-1,forecast_origin_date=f"2024-02-{movie:02d}",
            opening_weekend_start=f"2024-02-{movie:02d}",primary_point_forecast_usd=10.0,
            actual_opening_weekend_gross_usd=10.0+movie,source_count=2,release_width_bucket="wide",release_type="wide"))
    return pd.DataFrame(rows)


def test_weighted_quantile_and_chronological_pool() -> None:
    assert weighted_quantile([1,2,3],[1,1,8],[.5,.9]).tolist()==[3,3]
    frame=_panel(); pool=build_chronological_pool(frame,frame.iloc[-1])
    assert max(pool.member_ids)<25
    assert pool.normalized_weights.sum()==pytest.approx(1)
    assert pool.effective_sample_size==pytest.approx(24)
    draws=generate_draws(10,pool,seed=7,n_draws=50000)
    assert len(draws)==50000 and np.all(draws>=0)
    assert np.array_equal(draws,generate_draws(10,pool,seed=7,n_draws=50000))


def test_policy_lock_rejects_missing_005_and_mixed() -> None:
    for policies in ([],["005"],["007","005"]):
        with pytest.raises(ValueError): validate_policy_panel(policies)
    validate_policy_panel(["007","005"],comparison_diagnostic=True)


def test_collector_gap_invalidates_then_reconciles() -> None:
    async def scenario() -> None:
        sink=MemorySink()
        async def rest(token): return {"bids":[{"price":".4","size":"2"}],"asks":[{"price":".6","size":"2"}],"sequence":9}
        collector=ClobCollector(["123456789012345678901234567890"],sink,rest_book=rest)
        token=next(iter(collector.books))
        await collector.process({"event_type":"book","asset_id":token,"sequence":1,"bids":[],"asks":[]})
        await collector.process({"event_type":"price_change","asset_id":token,"sequence":3,"price_changes":[{"side":"BUY","price":".4","size":"1"}]})
        assert collector.health.sequence_gaps==1
        assert collector.books[token].valid
        assert collector.books[token].sequence==9
        assert sink.events[0][0]==token
    asyncio.run(scenario())
