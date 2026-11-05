from trip_agent.checker.consistency import BeliefView, TruthView, check_itinerary
from trip_agent.env.facts import true_facts
from trip_agent.env.models import Itinerary, ItineraryItem, parse_clock


def _with(itinerary: Itinerary, **changes) -> Itinerary:
    items = [i.model_copy(update=changes.get(i.id, {})) for i in itinerary.items]
    return Itinerary(hotel_id=changes.get("hotel_id", itinerary.hotel_id), items=items)


def _names(report):
    return sorted({v.name for v in report.violations})


def test_reference_itineraries_are_valid(catalog, scenarios):
    for s in scenarios:
        report = check_itinerary(s.reference_itinerary, s, catalog, TruthView(catalog))
        assert report.valid, [v.message for v in report.violations]
        assert report.fraction_satisfied == 1.0


def test_detects_each_kind_of_violation(catalog, monday):
    ref = monday.reference_itinerary
    truth = TruthView(catalog)
    late_museum = _with(ref, ref4={"start": parse_clock("15:30")})  # ends 17:30, closes 17:00
    assert "opening_hours" in _names(check_itinerary(late_museum, monday, catalog, truth))

    tight = _with(ref, ref3={"start": parse_clock("12:35")})  # lunch ends 12:30, travel 10 min
    assert _names(check_itinerary(tight, monday, catalog, truth)) == ["travel"]

    overlap = _with(ref, ref3={"start": parse_clock("12:00")})
    assert {"overlap", "travel"} <= set(_names(check_itinerary(overlap, monday, catalog, truth)))

    no_lunch = Itinerary(hotel_id=ref.hotel_id, items=[i for i in ref.items if i.id != "ref2"])
    assert "meal" in _names(check_itinerary(no_lunch, monday, catalog, truth))

    no_museum = Itinerary(hotel_id=ref.hotel_id, items=[i for i in ref.items if i.id != "ref4"])
    assert "required_activity" in _names(check_itinerary(no_museum, monday, catalog, truth))

    pricey = monday.model_copy(deep=True)
    pricey.constraints.budget = 400
    assert _names(check_itinerary(ref, pricey, catalog, truth)) == ["budget"]

    no_hotel = Itinerary(hotel_id=None, items=ref.items)
    assert "hotel" in _names(check_itinerary(no_hotel, monday, catalog, truth))


def test_closed_weekday(catalog, scenarios):
    tuesday = scenarios[1]
    items = [*tuesday.reference_itinerary.items]
    items.append(ItineraryItem(id="x", day=0, place_id="a_art_museum", kind="visit", start=parse_clock("19:30")))
    report = check_itinerary(Itinerary(hotel_id="h_harbor", items=items), tuesday, catalog, TruthView(catalog))
    assert any(v.name == "opening_hours" and "closed on Tuesday" in v.message for v in report.violations)


def test_hotel_neighborhood_rule(catalog, scenarios):
    tuesday = scenarios[1]
    wrong = Itinerary(hotel_id="h_copley", items=tuesday.reference_itinerary.items)
    report = check_itinerary(wrong, tuesday, catalog, TruthView(catalog))
    assert "hotel" in _names(report)


def test_belief_view_reports_unknown_facts_without_revealing_truth(catalog, monday):
    report = check_itinerary(monday.reference_itinerary, monday, catalog, BeliefView({}, {}))
    assert not report.valid
    unknown = {k for v in report.violations for k in v.unknown_facts}
    assert "a_art_museum.close_time" in unknown
    assert report.total_cost is None


def test_belief_view_uses_wrong_values_when_told(catalog, monday):
    facts = true_facts(catalog)
    facts["a_art_museum.close_time"] = parse_clock("19:00")  # optimistic lie
    late = _with(monday.reference_itinerary, ref4={"start": parse_clock("15:30")})
    believed = check_itinerary(late, monday, catalog, BeliefView(facts, {}))
    assert believed.valid  # looks fine to the agent
    verified = check_itinerary(late, monday, catalog, BeliefView(facts, {"a_art_museum.close_time": 17 * 60}))
    assert "opening_hours" in _names(verified)
