from datetime import datetime,timedelta,timezone
import pytest

from prediction_market_backtest.historical_panel import PriceObservation,normalize_price_history,pair_complete_event
from prediction_market_backtest.trading_lock import TradingContext,require_trading_preconditions,research_probability_difference


def test_price_normalization_dedup_pairing_and_shared_state():
    token="12345678901234567890";rows=normalize_price_history(token,[{"t":100,"p":"0.4"},{"t":100,"p":"0.4"}]);assert len(rows)==1
    now=datetime.fromtimestamp(200,timezone.utc);other=PriceObservation("b",datetime.fromtimestamp(150,timezone.utc),.7)
    result=pair_complete_event([token,"b"],[*rows,other],now,staleness=timedelta(hours=1));assert sum(result["projected_vector"])==pytest.approx(1) and result["shared_market_state_id"]


def test_trading_lock_and_research_only_difference():
    context=TradingContext("provisional","failed","approved","locked","fresh")
    with pytest.raises(PermissionError):require_trading_preconditions(context)
    row=research_probability_difference(.6,.5);assert not row["tradable"] and row["recommended_size"]==0
