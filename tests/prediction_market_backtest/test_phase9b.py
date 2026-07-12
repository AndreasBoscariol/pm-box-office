import json
from pathlib import Path

import pytest

from prediction_market_backtest.acceptance import acceptance_failure_codes, manifest_duration_hours
from prediction_market_backtest.acceptance_closeout import OUTPUTS, closeout_capture
from prediction_market_backtest.phase9_status import _sample_label


def complete_manifest():
    return {"status":"completed_pending_validation","start_utc":"2026-01-01T00:00:00+00:00","end_utc":"2026-01-02T00:06:00+00:00","subscribed_tokens":["1"],"disconnects":1,"reconnect_successes":1,"sequence_gaps":1,"state_invalidations":1,"rest_recovery_successes":1,"snapshots":1,"unrecovered_failures":0,"reconciliation_failures":0,"database_write_errors":0,"corrupted_books_used_downstream":0}


def test_duration_is_derived_and_all_conditions_required():
    manifest=complete_manifest();assert manifest_duration_hours(manifest)==pytest.approx(24.1);assert acceptance_failure_codes(manifest)==()
    manifest["rest_recovery_successes"]=0;assert "failed_rest_recovery" in acceptance_failure_codes(manifest)


def test_closeout_writes_formal_package(tmp_path:Path):
    (tmp_path/"01_run_manifest.json").write_text(json.dumps(complete_manifest()))
    (tmp_path/"raw_events.jsonl").write_text(json.dumps({"message":{"event_type":"book"}})+"\n")
    (tmp_path/"snapshots.jsonl").write_text(json.dumps({"token_id":"1","valid":True,"book_hash":"x"})+"\n")
    result=closeout_capture(tmp_path);assert result["decision"]=="passed"
    assert all((tmp_path/name).exists() for name in OUTPUTS)


def test_active_run_cannot_be_closed(tmp_path:Path):
    manifest=complete_manifest();manifest["status"]="running";(tmp_path/"01_run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError):closeout_capture(tmp_path)


def test_diagnostic_sample_labels():
    assert [_sample_label(n) for n in (9,10,20,30)]==["blocked_small_sample","exploratory","diagnostic_small_sample","diagnostic"]
