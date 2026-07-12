import csv
from pathlib import Path

import pytest

from prediction_market_backtest.review_import import import_review_csv


def _write(path:Path,**changes):
    row={"event_id":"1","reviewer_decision":"approved","reviewer":"alice","approved_movie_id":"42","locked":"true",
         "full_resolution_rules":"rules","bucket_table_json":"[]","parsed_target":"domestic","automated_warnings":"","reviewer_notes":"ok",**changes}
    with path.open("w",newline="") as handle:writer=csv.DictWriter(handle,fieldnames=row);writer.writeheader();writer.writerow(row)


def test_approval_requires_reviewer_movie_lock_and_no_warnings(tmp_path):
    path=tmp_path/"review.csv";_write(path);decision=import_review_csv(path)["1"]
    assert decision.locked and decision.approved_movie_id==42
    for changes in ({"reviewer":""},{"approved_movie_id":""},{"locked":"false"},{"automated_warnings":"gap"}):
        _write(path,**changes)
        with pytest.raises(ValueError):import_review_csv(path)


def test_locked_decision_cannot_change(tmp_path):
    path=tmp_path/"review.csv";_write(path);existing=import_review_csv(path)
    _write(path,reviewer_decision="rejected_duplicate")
    with pytest.raises(ValueError,match="locked"):import_review_csv(path,existing)
