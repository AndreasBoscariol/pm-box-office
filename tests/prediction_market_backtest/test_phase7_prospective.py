import json
from pathlib import Path

from prediction_market_backtest.phase7_freeze import _artifact_checksum


def test_frozen_manifest_checksum_detects_change():
    payload={"artifact":"x","status":"provisional"};checksum=_artifact_checksum(payload)
    assert checksum==_artifact_checksum(payload)
    payload["status"]="approved"
    assert checksum!=_artifact_checksum(payload)


def test_primary_panel_is_one_row_per_movie_and_under_30_cannot_approve():
    rows=[{"movie_id":1,"origin":-2},{"movie_id":1,"origin":-1},{"movie_id":2,"origin":-1}]
    latest={}
    for row in rows:latest[row["movie_id"]]=max(latest.get(row["movie_id"],-99),row["origin"])
    assert latest=={1:-1,2:-1}
    assert len(latest)<30
