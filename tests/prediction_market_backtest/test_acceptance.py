from prediction_market_backtest.acceptance import validate_capture_acceptance


def test_capture_requires_full_duration_and_recovery_evidence():
    manifest={"start_utc":"2026-01-01T00:00:00+00:00","end_utc":"2026-01-02T00:06:00+00:00","reconnect_successes":1,
        "sequence_gaps":1,"state_invalidations":1,"rest_recovery_successes":1,"snapshots":10,"unrecovered_failures":0,"database_write_errors":0}
    assert validate_capture_acceptance(manifest)==(True,())
    manifest["end_utc"]="2026-01-02T00:05:00+00:00"
    passed,errors=validate_capture_acceptance(manifest);assert not passed and "failed_insufficient_duration" in errors
