import ast
from pathlib import Path

from conftest import reference_script

from trip_agent.agent.llm.scripted import ScriptedClient, call
from trip_agent.agent.loop import EpisodeLimits, run_episode
from trip_agent.agent.policies import PolicySettings, RecoveryPolicy, make_policy
from trip_agent.agent.prompts import system_prompt
from trip_agent.checker.consistency import TruthView, check_itinerary
from trip_agent.corruption.corrupt import CorruptionPlan, clean_plan
from trip_agent.env.models import parse_clock
from trip_agent.tools.environment import TripEnvironment

LIE = {"a_art_museum.close_time": parse_clock("19:00")}


def _env(catalog, scenario, altered=None, limit=60):
    plan = CorruptionPlan(scenario_id=scenario.id, seed=0, rate=0.2, altered=altered or {})
    return TripEnvironment(catalog, scenario, plan, max_tool_calls=limit)


def test_prompts_nest():
    assert system_prompt("standard") in system_prompt("verification")
    assert system_prompt("verification") in system_prompt("recovery")


def test_reference_script_finishes_in_every_variant(catalog, monday):
    for variant in ("standard", "verification", "recovery"):
        env = _env(catalog, monday)
        result = run_episode(env, make_policy(variant), ScriptedClient(reference_script(monday)))
        assert result.stop_reason == "finished"
        assert len(result.itinerary.items) == 5
        truth = check_itinerary(result.itinerary, monday, catalog, TruthView(catalog))
        assert truth.valid, [v.message for v in truth.violations]
        # The script never looked up travel times or the hotel price, so its own check cannot pass.
        assert env.last_check is not None and not env.last_check.valid


def test_history_is_append_only(catalog, monday):
    llm = ScriptedClient(reference_script(monday))
    run_episode(_env(catalog, monday), make_policy("recovery"), llm)
    for earlier, later in zip(llm.requests, llm.requests[1:], strict=False):
        assert later[: len(earlier)] == earlier


def test_idle_model_is_nudged_then_stopped(catalog, monday):
    env = _env(catalog, monday)
    result = run_episode(
        env,
        make_policy("standard"),
        ScriptedClient(policy=lambda m: "I would visit a museum."),
        EpisodeLimits(max_idle_turns=2),
    )
    assert result.stop_reason == "no_tool_calls"
    assert result.turns == 3


def test_max_turns(catalog, monday):
    env = _env(catalog, monday, limit=1000)
    result = run_episode(
        env,
        make_policy("standard"),
        ScriptedClient(policy=lambda m: [call("view_itinerary")]),
        EpisodeLimits(max_turns=4),
    )
    assert result.stop_reason == "max_turns" and result.turns == 4


def test_verification_flags_tight_unverified_facts(catalog, monday):
    env = _env(catalog, monday, LIE)
    policy = make_policy("verification")
    env.call("set_hotel", {"hotel_id": "h_copley"})
    env.call("search_places", {"place_ids": ["a_art_museum"]})
    result = env.call(
        "add_item", {"day": 0, "place_id": "a_art_museum", "start": "16:00"}
    )  # believed end 18:00 vs 19:00
    note = policy.after_tool(env, "add_item", {}, result)
    assert note and "a_art_museum.close_time" in note


def _schedule_late_museum(env):
    env.call("set_hotel", {"hotel_id": "h_copley"})
    env.call("search_places", {"place_ids": ["a_art_museum", "r_pier"]})
    museum = env.call("add_item", {"day": 0, "place_id": "a_art_museum", "start": "16:00"})["added"]["item_id"]
    dinner = env.call("add_item", {"day": 0, "place_id": "r_pier", "start": "18:30", "meal_type": "dinner"})["added"][
        "item_id"
    ]
    return museum, dinner


def test_recovery_removes_only_items_the_corrected_fact_breaks(catalog, monday):
    env = _env(catalog, monday, LIE)
    policy = RecoveryPolicy()
    museum, dinner = _schedule_late_museum(env)
    result = env.call("verify_fact", {"fact_key": "a_art_museum.close_time"})
    note = policy.after_tool(env, "verify_fact", {}, result)
    assert museum not in env.state.items and dinner in env.state.items
    assert "Repair 1/3" in note and dinner in note  # downstream item is named for re-checking
    assert env.state.checkpoints and env.state.history[-1].action == "repair_remove"


def test_recovery_caps_repairs_per_conflict(catalog, monday):
    env = _env(catalog, monday, LIE, limit=200)
    policy = RecoveryPolicy(PolicySettings(max_repairs_per_conflict=3))
    env.call("set_hotel", {"hotel_id": "h_copley"})
    env.call("search_places", {"place_ids": ["a_art_museum"]})
    env.call("verify_fact", {"fact_key": "a_art_museum.close_time"})
    notes = []
    for _ in range(4):  # keep re-adding the museum too late; each failed check is one repair attempt
        env.call("add_item", {"day": 0, "place_id": "a_art_museum", "start": "16:00"})
        result = env.call("check_itinerary", {})
        notes.append(policy.after_tool(env, "check_itinerary", {}, result) or "")
    assert sum("Repair" in n for n in notes) == 3
    assert "still unresolved" in notes[3]
    policy.finalize(env)
    assert "check:opening_hours:a_art_museum" in env.unresolved


def test_standard_policy_adds_nothing(catalog, monday):
    env = _env(catalog, monday, LIE)
    museum, _ = _schedule_late_museum(env)
    result = env.call("verify_fact", {"fact_key": "a_art_museum.close_time"})
    assert make_policy("standard").after_tool(env, "verify_fact", {}, result) is None
    assert museum in env.state.items


def test_agent_code_cannot_reach_storage_or_evaluator():
    """Truth isolation: the agent package must not import the evaluator or the database layer."""
    root = Path(__file__).resolve().parents[1] / "trip_agent" / "agent"
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert not name.startswith(("trip_agent.storage", "trip_agent.eval")), f"{path} imports {name}"


def test_clean_condition_has_no_lies(catalog, monday):
    env = TripEnvironment(catalog, monday, clean_plan(monday.id), 10)
    found = env.call("search_places", {"place_ids": ["a_art_museum"]})
    assert found["results"][0]["close"] == "17:00"
