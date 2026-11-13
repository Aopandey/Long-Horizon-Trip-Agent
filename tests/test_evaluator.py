from conftest import reference_script

from trip_agent.agent.llm.scripted import ScriptedClient, call
from trip_agent.agent.loop import run_episode
from trip_agent.agent.policies import make_policy
from trip_agent.corruption.corrupt import CorruptionPlan
from trip_agent.env.models import parse_clock
from trip_agent.eval.evaluator import score_run
from trip_agent.eval.records import RunRecord
from trip_agent.tools.environment import TripEnvironment

LIE = {"a_art_museum.close_time": parse_clock("19:00")}


def _run(catalog, scenario, turns, altered=None, variant="standard"):
    plan = CorruptionPlan(scenario_id=scenario.id, seed=0, rate=0.2, altered=altered or {})
    env = TripEnvironment(catalog, scenario, plan, max_tool_calls=80)
    result = run_episode(env, make_policy(variant), ScriptedClient(turns))
    record = RunRecord(
        run_key=f"t/{scenario.id}/x/{variant}/r0",
        experiment="t",
        scenario_id=scenario.id,
        split=scenario.split,
        horizon=scenario.horizon,
        condition="corrupted" if altered else "clean",
        variant=variant,
        repeat=0,
        seed=0,
        model="scripted",
        stop_reason=result.stop_reason,
        tool_calls=result.tool_calls,
        final_itinerary=result.itinerary,
        corruption=plan,
        observed=env.state.observed,
        verified=env.state.verified,
        observed_step=env.state.observed_step,
        verified_step=env.state.verified_step,
        history=env.state.history,
    )
    return score_run(record, scenario, catalog)


def _late_museum_script(verify: bool):
    turns = [
        [call("set_hotel", hotel_id="h_copley")],
        [call("search_places", place_ids=["a_liberty_walk", "r_hanover", "a_aquarium", "a_art_museum", "r_pier"])],
        [call("add_item", day=0, place_id="a_liberty_walk", start="09:30")],
        [call("add_item", day=0, place_id="r_hanover", start="11:30", meal_type="lunch")],
        [call("add_item", day=0, place_id="a_aquarium", start="13:00")],
        [call("add_item", day=0, place_id="a_art_museum", start="16:00")],  # wrong if close is really 17:00
        [call("add_item", day=0, place_id="r_pier", start="18:30", meal_type="dinner")],
    ]
    if verify:
        turns += [
            [call("verify_fact", fact_key="a_art_museum.close_time")],
            [call("move_item", item_id="c4", start="14:45")],
            [call("move_item", item_id="c3", start="13:00")],
        ]
    return turns + [[call("finish")]]


def test_valid_reference_run_scores_clean(catalog, monday):
    score = _run(catalog, monday, reference_script(monday))
    assert score.valid and score.fraction_satisfied == 1.0
    assert score.first_error_position is None and score.false_reliance_items == 0


def test_false_information_is_measured(catalog, monday):
    score = _run(catalog, monday, _late_museum_script(verify=False), LIE)
    assert not score.valid
    assert score.violations_by_check.get("opening_hours") == 1
    assert score.false_reliance_items == 1
    assert score.first_error_position == 4  # the museum is the 4th commitment
    assert score.conflicts_detected == 0 and score.recovery_success_rate is None


def test_detection_and_repair_are_measured(catalog, monday):
    score = _run(catalog, monday, _late_museum_script(verify=True), LIE)
    assert score.valid, score.violations_by_check
    assert score.conflicts_detected == 1 and score.conflicts_resolved == 1
    assert score.recovery_success_rate == 1.0
    assert score.false_reliance_items == 0
    assert score.first_error_position == 4  # the original mistake still counts as the first error
