from prediction_market_backtest.event_review import automated_filter, parse_event_buckets


def _event(description="Domestic 3-day opening weekend. If exactly between brackets, use the higher range bracket."):
    questions=["less than $20m","between $20m and $25m","greater than $25m"]
    return {"id":"1","title":"Example Opening Weekend Box Office","description":description,
        "markets":[{"id":str(i),"question":question,"description":description} for i,question in enumerate(questions)]}


def test_real_style_boundary_rule_is_exhaustive():
    buckets,errors=parse_event_buckets(_event())
    assert not errors
    assert [sum(bucket.contains(value) for bucket in buckets) for value in (0,20_000_000,25_000_000)]==[1,1,1]


def test_filter_rejects_extended_and_single_threshold():
    event=_event("Domestic 5-day opening weekend")
    assert automated_filter(event)==(False,"unsupported extended duration")
    event=_event();event["markets"]=event["markets"][:1]
    assert automated_filter(event)==(False,"single threshold or incomplete bucket set")
