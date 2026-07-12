from prediction_market_backtest.operations import sync_prospective_actuals


def test_actual_sync_fails_closed_without_publication_timestamp(tmp_path):
    panel=tmp_path/"panel.csv";panel.write_text("movie_id,opening_weekend_start,actual_opening_weekend_gross_usd\n1,2026-07-10,10\n")
    result=sync_prospective_actuals(panel,tmp_path/"out")
    assert result["inserted"]==0 and result["reason"]=="actual_available_at_missing_fail_closed"
