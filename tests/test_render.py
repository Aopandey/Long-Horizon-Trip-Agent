from trip_agent.corruption.corrupt import CorruptionPlan
from trip_agent.env.models import parse_clock
from trip_agent.tools.environment import TOOL_NAMES, TripEnvironment
from trip_agent.tools.render import brief, render


def _env(catalog, scenario, altered=None):
    plan = CorruptionPlan(scenario_id=scenario.id, seed=0, rate=0.2, altered=altered or {})
    return TripEnvironment(catalog, scenario, plan, max_tool_calls=100)


def test_every_tool_has_a_compact_text_form(catalog, monday):
    env = _env(catalog, monday, {"a_art_museum.close_time": parse_clock("19:00")})
    calls = [
        ("search_places", {"kind": "attraction"}),
        ("search_places", {"kind": "hotel"}),
        ("search_places", {"kind": "restaurant", "meal_type": "dinner"}),
        ("get_travel_time", {"from_id": "h_copley", "to_id": "a_aquarium"}),
        ("verify_fact", {"fact_key": "a_art_museum.close_time"}),
        ("set_hotel", {"hotel_id": "h_copley"}),
        ("add_item", {"day": 0, "place_id": "a_aquarium", "start": "10:00"}),
        ("move_item", {"item_id": "c1", "start": "10:30"}),
        ("check_itinerary", {}),
        ("view_itinerary", {}),
        ("remove_item", {"item_id": "c1"}),
        ("add_item", {"day": 9, "place_id": "a_aquarium", "start": "10:00"}),
        ("finish", {}),
    ]
    assert {name for name, _ in calls} == TOOL_NAMES
    for name, args in calls:
        result = env.call(name, args)
        full, short = render(name, result), brief(name, result)
        assert not full.lstrip().startswith("{"), f"{name} rendered as JSON"
        assert "\n" not in short and short.startswith("[old]")
        assert len(short) <= len(full)
        if "tool_calls_remaining" in result:
            assert f"calls left: {result['tool_calls_remaining']}" in full


def test_text_keeps_the_facts_the_agent_needs(catalog, monday):
    env = _env(catalog, monday, {"a_art_museum.close_time": parse_clock("19:00")})
    found = env.call("search_places", {"place_ids": ["a_art_museum"]})
    text = render("search_places", found)
    assert "a_art_museum" in text and "open 10:00-19:00" in text and "(closed Tue)" in text
    assert "120 min visit" in text and "$25/person" in text
    assert "a_art_museum 10:00-19:00 $25" in brief("search_places", found)
    checked = env.call("verify_fact", {"fact_key": "a_art_museum.close_time"})
    assert "= 17:00" in render("verify_fact", checked) and "WRONG" in render("verify_fact", checked)
    assert "search was wrong" in brief("verify_fact", checked)


def test_search_text_is_much_shorter_than_json(catalog, monday):
    import json

    result = _env(catalog, monday).call("search_places", {"kind": "attraction", "limit": 10})
    assert len(render("search_places", result)) < 0.7 * len(json.dumps(result))


def test_planner_notes_and_notices_are_kept_in_full_form(catalog, monday):
    env = _env(catalog, monday)
    env.call("search_places", {"kind": "hotel"})
    repeat = env.call("search_places", {"kind": "hotel"})
    text = render("search_places", repeat, note="[planner] verify the hotel price")
    assert "Same call as step 1" in text and text.endswith("[planner] verify the hotel price")
    assert "planner" not in brief("search_places", repeat)
