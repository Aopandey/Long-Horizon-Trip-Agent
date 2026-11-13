from trip_agent.corruption.corrupt import CorruptionPlan, clean_plan
from trip_agent.env.models import parse_clock
from trip_agent.tools.environment import TOOL_NAMES, TOOL_SPECS, TripEnvironment


def _env(catalog, scenario, altered=None, limit=50):
    plan = CorruptionPlan(scenario_id=scenario.id, seed=0, rate=0.2, altered=altered or {})
    return TripEnvironment(catalog, scenario, plan, max_tool_calls=limit)


def test_tool_specs_are_well_formed():
    assert len(TOOL_SPECS) == len(TOOL_NAMES) == 10
    for spec in TOOL_SPECS:
        assert spec["type"] == "function"
        assert spec["function"]["parameters"]["type"] == "object"


def test_search_shows_corrupted_value_and_verify_shows_truth(catalog, monday):
    env = _env(catalog, monday, {"a_art_museum.close_time": parse_clock("19:00")})
    found = env.call("search_places", {"place_ids": ["a_art_museum"]})
    assert found["results"][0]["close"] == "19:00"
    assert env.state.observed["a_art_museum.close_time"] == parse_clock("19:00")
    checked = env.call("verify_fact", {"fact_key": "a_art_museum.close_time"})
    assert checked["verified_value"] == "17:00"
    assert checked["previously_observed"] == "19:00"
    assert checked["matches_observed"] is False
    assert env.verify_calls == 1
    assert env.state.verified_step["a_art_museum.close_time"] == 2


def test_travel_time_returns_its_fact_key(catalog, monday):
    env = _env(catalog, monday)
    result = env.call("get_travel_time", {"from_id": "a_art_museum", "to_id": "h_copley"})
    assert result["minutes"] == 5
    assert result["fact_key"].startswith("travel:")
    assert env.call("verify_fact", {"fact_key": result["fact_key"]})["matches_observed"] is True


def test_item_editing_and_dependencies(catalog, monday):
    env = _env(catalog, monday)
    env.call("set_hotel", {"hotel_id": "h_copley"})
    added = env.call("add_item", {"day": 0, "place_id": "a_art_museum", "start": "15:00"})
    item_id = added["added"]["item_id"]
    assert added["added"]["end"] == "17:00"
    assert "a_art_museum.close_time" in added["depends_on_facts"]
    assert any(k.startswith("travel:") for k in added["depends_on_facts"])
    moved = env.call("move_item", {"item_id": item_id, "start": "14:00"})
    assert moved["moved"]["start"] == "14:00"
    assert env.call("remove_item", {"item_id": item_id}) == {"removed": item_id, "tool_calls_remaining": 46}
    assert [e.action for e in env.state.history] == ["set_hotel", "add", "move", "remove"]


def test_bad_calls_return_errors_and_still_cost_budget(catalog, monday):
    env = _env(catalog, monday, limit=10)
    assert "error" in env.call("add_item", {"day": 5, "place_id": "a_aquarium", "start": "10:00"})
    assert "error" in env.call("add_item", {"day": 0, "place_id": "r_pier", "start": "12:00"})  # meal_type missing
    assert "error" in env.call("add_item", {"day": 0, "place_id": "a_aquarium", "start": "ten"})
    assert "error" in env.call("teleport", {})
    assert "error" in env.call("verify_fact", {"fact_key": "nope"})
    assert env.budget.used == 5
    extra = env.call("view_itinerary", {"unexpected": 1})
    assert extra["ignored_arguments"] == ["unexpected"]


def test_budget_exhaustion_ends_the_episode(catalog, monday):
    env = _env(catalog, monday, limit=2)
    env.call("view_itinerary", {})
    last = env.call("view_itinerary", {})
    assert "notice" in last and env.done and env.stop_reason == "budget_exhausted"
    assert "error" in env.call("view_itinerary", {})
    assert env.budget.used == 2


def test_check_itinerary_uses_beliefs(catalog, monday):
    env = _env(catalog, monday)
    env.call("set_hotel", {"hotel_id": "h_copley"})
    env.call("add_item", {"day": 0, "place_id": "a_aquarium", "start": "10:00"})
    result = env.call("check_itinerary", {})
    assert result["valid"] is False
    unknown = {k for v in result["violations"] for k in v.get("unknown_facts", [])}
    assert "a_aquarium.close_time" in unknown


def test_task_brief_mentions_constraints(catalog, monday):
    brief = TripEnvironment(catalog, monday, clean_plan(monday.id), 40).task_brief()
    for text in ("2 people", "$500.00", "Fenway Art Museum (a_art_museum)", "lunch", "40 tool calls", "Monday"):
        assert text in brief


def test_search_matches_words_not_whole_sentences(catalog, monday):
    env = _env(catalog, monday)
    hits = env.call("search_places", {"kind": "hotel", "query": "Boston hotel near the Seaport harbor"})
    assert [r["id"] for r in hits["results"]][0] == "h_harbor"
    fallback = env.call("search_places", {"kind": "hotel", "query": "Boston hotel near Freedom Trail"})
    assert {r["id"] for r in fallback["results"]} == {"h_copley", "h_harbor"} and "note" in fallback
    museums = env.call("search_places", {"query": "museum"})
    assert {r["id"] for r in museums["results"]} == {"a_art_museum", "a_science"}


def test_empty_search_points_elsewhere_and_repeats_are_flagged(catalog, monday):
    env = _env(catalog, monday)
    empty = env.call("search_places", {"kind": "restaurant", "meal_type": "dinner", "neighborhood": "Back Bay"})
    assert empty["results"] == [] and "North End" in empty["note"] and "Seaport" in empty["note"]
    again = env.call("search_places", {"neighborhood": "Back Bay", "kind": "restaurant", "meal_type": "dinner"})
    assert "repeat_notice" in again and env.budget.used == 2
