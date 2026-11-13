from __future__ import annotations

import pytest

from trip_agent.agent.llm.scripted import call
from trip_agent.env.example import example_catalog, example_scenarios
from trip_agent.env.models import format_clock


@pytest.fixture
def catalog():
    return example_catalog()


@pytest.fixture
def scenarios():
    return example_scenarios()


@pytest.fixture
def monday(scenarios):
    return scenarios[0]


def reference_script(scenario, check=True):
    """Tool-call turns that rebuild the scenario's reference itinerary, then check and finish."""
    turns = [[call("set_hotel", hotel_id=scenario.reference_itinerary.hotel_id)]]
    for item in scenario.reference_itinerary.items:
        args = {"day": item.day, "place_id": item.place_id, "start": format_clock(item.start)}
        if item.meal_type:
            args["meal_type"] = item.meal_type.value
        turns.append([call("search_places", place_ids=[item.place_id]), call("add_item", **args)])
    if check:
        turns.append([call("check_itinerary")])
    turns.append([call("finish", notes="done")])
    return turns
