import pytest
from prediction_market_backtest.capture_runtime import run_capture_acceptance


def test_capture_rejects_missing_or_non_string_precision_unsafe_tokens():
    import asyncio
    with pytest.raises(ValueError):asyncio.run(run_capture_acceptance([],"/tmp/no-capture",duration_hours=0))
    with pytest.raises(ValueError):asyncio.run(run_capture_acceptance(["1.23"],"/tmp/no-capture",duration_hours=0))
