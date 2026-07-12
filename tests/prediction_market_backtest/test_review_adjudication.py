from prediction_market_backtest.review_adjudication import _decide,_match_movie


def test_review_rubric_never_approves_warnings_or_missing_match():
    base={"event_title":"Film Opening Weekend Box Office","full_resolution_rules":"three-day opening weekend","proposed_movie_id":"1","automated_warnings":""}
    assert _decide(base)[0]=="approved"
    assert _decide({**base,"proposed_movie_id":""})[0]=="rejected_no_internal_movie"
    assert _decide({**base,"automated_warnings":"overlap at 1"})[0]=="rejected_overlapping_buckets"
    assert _decide({**base,"automated_warnings":"gap at boundary 2"})[0]=="rejected_boundary_ambiguity"

def test_title_database_matching_handles_year_and_format_qualifiers():
    movies=[{"movie_id":12,"title":"28 Years Later (2025)","release_year":2025,"release_date":None},{"movie_id":378,"title":"Moana (IMAX)","release_year":2026,"release_date":None}]
    assert _match_movie({"event_title":"28 Years Later Opening Weekend Box Office","full_resolution_rules":"June 2025"},movies)["movie_id"]==12
    assert _match_movie({"event_title":"Moana (2026) Opening Weekend Box Office","full_resolution_rules":"July 2026"},movies)["movie_id"]==378
