import json
from datetime import datetime,timezone
from pathlib import Path
from decimal import Decimal
import pandas as pd
import pytest

from prediction_market_backtest.prospective import cohort_inventory,verify_frozen_artifact
from prediction_market_backtest.prospective_actuals import ActualRevision,append_actual,scheduled_look


def test_artifact_verification_and_tamper_failure(tmp_path):
    source=Path("models/boxoffice/pre_release_pointscale_cal_001/manifest.json");target=tmp_path/"artifact";target.mkdir();(target/"manifest.json").write_bytes(source.read_bytes())
    _,verification=verify_frozen_artifact(target);assert verification.valid
    payload=json.loads((target/"manifest.json").read_text());payload["calibration_parameters"]["alpha"]+=.01;(target/"manifest.json").write_text(json.dumps(payload))
    with pytest.raises(RuntimeError,match="checksum"):verify_frozen_artifact(target)


def test_cohort_is_strictly_post_cutoff_and_market_independent():
    manifest={"latest_included_release_date":"2026-07-03"};frame=pd.DataFrame([
        {"movie_id":1,"title":"old","opening_weekend_start":"2026-07-03","origin_day":-1,"release_width_bucket":"wide"},
        {"movie_id":2,"title":"new","opening_weekend_start":"2026-07-10","origin_day":-1,"release_width_bucket":"wide"}])
    rows=cohort_inventory(frame,manifest,datetime.now(timezone.utc));assert [row.movie_id for row in rows]==[2] and rows[0].polymarket_available is None


def test_actual_revisions_append_and_decision_schedule(tmp_path):
    path=tmp_path/"actuals.csv";first=ActualRevision(1,Decimal("10"),"source","2026-01-01T00:00:00Z","2026-01-01T01:00:00Z",3,"USD","domestic_us_canada")
    append_actual(path,first);append_actual(path,first);assert len(path.read_text().splitlines())==2
    assert scheduled_look(29)[2].startswith("inconclusive")
    assert scheduled_look(30)==(30,.975,"formal_first_look")
    assert scheduled_look(50,"continue_to_50")==(50,.95,"formal_final_look")
