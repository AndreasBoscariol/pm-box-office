"""Central hard lock separating diagnostics from tradable actions."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True,slots=True)
class TradingContext:
    prospective_probability_status:str
    clob_capture_acceptance:str
    event_semantics:str
    movie_match:str
    market_book:str
    signal_status:str="research_only"


def require_trading_preconditions(context:TradingContext)->None:
    required={"prospective_probability_status":"approved_probability_model","clob_capture_acceptance":"passed","event_semantics":"approved","movie_match":"locked","market_book":"fresh","signal_status":"tradable"}
    failures=[f"{field}={getattr(context,field)}" for field,value in required.items() if getattr(context,field)!=value]
    if failures:raise PermissionError("trading disabled: "+", ".join(failures))


def research_probability_difference(model_probability:float,market_probability:float)->dict[str,float|str|bool]:return {"probability_difference":model_probability-market_probability,"signal_status":"research_only","tradable":False,"recommended_size":0.0}
