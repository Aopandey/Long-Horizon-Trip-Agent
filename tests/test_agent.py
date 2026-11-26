import ast
from pathlib import Path

from conftest import reference_script

from trip_agent.agent.llm.scripted import ScriptedClient, call
from trip_agent.agent.loop import EpisodeLimits, run_episode
from trip_agent.agent.policies import PolicySettings, RecoveryPolicy, make_policy, risky_facts
from trip_agent.agent.prompts import system_prompt
from trip_agent.checker.consistency import TruthView, check_itinerary
from trip_agent.corruption.corrupt import CorruptionPlan, clean_plan
from trip_agent.env.models import format_clock, parse_clock
from trip_agent.state import dependency_graph
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


def test_without_a_window_the_history_is_append_only(catalog, monday):
    llm = ScriptedClient(reference_script(monday))
    run_episode(_env(catalog, monday), make_policy("recovery"), llm, EpisodeLimits(recent_turns=None))
    for earlier, later in zip(llm.requests, llm.requests[1:], strict=False):
        assert later[: len(earlier)] == earlier


def test_history_window_shortens_old_turns_in_blocks(catalog, monday):
    turns = [[call("search_places", kind="attraction", limit=10)] for _ in range(20)] + [[call("finish")]]
    llm = ScriptedClient(turns)
    env = _env(catalog, monday, limit=100)
    run_episode(env, make_policy("standard"), llm, EpisodeLimits(recent_turns=3))
    head = llm.requests[0][:2]
    for request in llm.requests:
        assert request[:2] == head  # system prompt and task never change
        tool_msgs = [m for m in request if m["role"] == "tool"]
        full = [m for m in tool_msgs if not m["content"].startswith("[old]")]
        assert len(full) < 6  # at most 2 x window in full
        assert all(m["content"].startswith("[old] search:") for m in tool_msgs[: len(tool_msgs) - len(full)])
    # Between shortening steps the prompt only grows at the end, so the server can reuse its cache.
    extensions = sum(
        later[: len(earlier)] == earlier for earlier, later in zip(llm.requests, llm.requests[1:], strict=False)
    )
    assert extensions >= len(llm.requests) * 0.6
    last = llm.requests[-1]
    windowed = sum(len(m.get("content", "")) for m in last)
    unwindowed_llm = ScriptedClient(turns)
    run_episode(
        _env(catalog, monday, limit=100), make_policy("standard"), unwindowed_llm, EpisodeLimits(recent_turns=None)
    )
    assert windowed < 0.5 * sum(len(m.get("content", "")) for m in unwindowed_llm.requests[-1])


def test_old_assistant_text_is_dropped_but_tool_calls_stay(catalog, monday):
    class Talkative(ScriptedClient):
        def chat(self, messages, tools, *, seed=None):
            response = super().chat(messages, tools, seed=seed)
            response.content = "Let me think about this at length. " * 20
            return response

    llm = Talkative([[call("view_itinerary")] for _ in range(10)] + [[call("finish")]])
    run_episode(_env(catalog, monday, limit=100), make_policy("standard"), llm, EpisodeLimits(recent_turns=2))
    assistants = [m for m in llm.requests[-1] if m["role"] == "assistant"]
    assert assistants[0]["content"] == "" and assistants[0]["tool_calls"][0]["function"]["name"] == "view_itinerary"
    assert assistants[-1]["content"].startswith("Let me think")


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


def _reference_plan(env, scenario):
    """Rebuild the reference itinerary with every lookup its check needs; returns the check result."""
    ref = scenario.reference_itinerary
    env.call("search_places", {"place_ids": [ref.hotel_id] + [i.place_id for i in ref.items]})
    env.call("set_hotel", {"hotel_id": ref.hotel_id})
    for item in ref.items:
        args = {"day": item.day, "place_id": item.place_id, "start": format_clock(item.start)}
        if item.meal_type:
            args["meal_type"] = item.meal_type.value
        env.call("add_item", args)
    stops = [ref.hotel_id] + [i.place_id for i in sorted(ref.items, key=lambda i: i.start)] + [ref.hotel_id]
    for a, b in zip(stops, stops[1:], strict=False):
        env.call("get_travel_time", {"from_id": a, "to_id": b})
    return env.call("check_itinerary", {})


def test_verification_stays_quiet_while_building(catalog, monday):
    env = _env(catalog, monday, LIE)
    policy = make_policy("verification")
    for tool, args in [
        ("set_hotel", {"hotel_id": "h_copley"}),
        ("search_places", {"place_ids": ["a_art_museum"]}),
        ("add_item", {"day": 0, "place_id": "a_art_museum", "start": "16:00"}),  # believed end 18:00 vs 19:00
        ("move_item", {"item_id": "c1", "start": "16:30"}),
        ("check_itinerary", {}),  # not valid yet: nothing to verify
    ]:
        assert policy.after_tool(env, tool, args, env.call(tool, args)) is None


def test_verification_lists_risky_facts_once_the_plan_passes(catalog, monday):
    env = _env(catalog, monday, LIE)
    policy = make_policy("verification")
    check = _reference_plan(env, monday)
    assert check["valid"]
    note = policy.after_tool(env, "check_itinerary", {}, check)
    assert note.startswith("[planner] The plan passes")
    risky = policy.risky(env)
    assert [f.score for f in risky] == sorted(f.score for f in risky) and all(f.score < 1 for f in risky)
    assert risky[0].key in note
    # Only facts the plan relies on, and fewer than all of them.
    deps = set(dependency_graph(env.state.itinerary(), catalog)) | {f"{env.state.hotel_id}.price_per_night"}
    assert {f.key for f in risky} <= deps and len(risky) < len(deps)
    # A verified fact leaves the list; once everything risky is verified the planner says so.
    env.call("verify_fact", {"fact_key": risky[0].key})
    assert risky[0].key not in {f.key for f in policy.risky(env)}
    for fact in policy.risky(env):
        env.call("verify_fact", {"fact_key": fact.key})
    assert "Finish" in policy.after_tool(env, "check_itinerary", {}, env.call("check_itinerary", {}))


def test_risky_facts_follow_the_margins(catalog, monday):
    env = _env(catalog, monday, LIE)
    _reference_plan(env, monday)
    state = env.state

    def keys(**settings):
        found = risky_facts(state.itinerary(), monday, catalog, state.belief(), set(), PolicySettings(**settings))
        return {f.key for f in found}

    default = keys()
    assert "h_copley.price_per_night" in default  # the largest cost goes first when the budget is tight
    assert not {k for k in keys(price_error_fraction=0.0) if "price" in k}
    assert not {k for k in keys(hours_slack_minutes=1) if k.endswith(("open_time", "close_time"))}
    assert keys(hours_slack_minutes=600, travel_slack_minutes=600) >= default


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
    notes = []
    for _ in range(4):  # keep re-adding the museum too late and re-verifying the hours it relies on
        env.call("add_item", {"day": 0, "place_id": "a_art_museum", "start": "16:00"})
        result = env.call("verify_fact", {"fact_key": "a_art_museum.close_time"})
        notes.append(policy.after_tool(env, "verify_fact", {}, result) or "")
    assert sum("Repair" in n for n in notes) == 3
    assert "still unresolved" in notes[3]
    policy.finalize(env)
    assert "fact:a_art_museum.close_time" in env.unresolved


def test_recovery_leaves_ordinary_conflicts_to_the_model(catalog, monday):
    env = TripEnvironment(catalog, monday, clean_plan(monday.id), 60)
    policy = RecoveryPolicy()
    env.call("set_hotel", {"hotel_id": "h_copley"})
    env.call("search_places", {"place_ids": ["a_art_museum"]})
    museum = env.call("add_item", {"day": 0, "place_id": "a_art_museum", "start": "16:00"})["added"]["item_id"]
    result = env.call("check_itinerary", {})  # the museum closes at 17:00: a scheduling mistake, no false fact
    assert not result["valid"]
    assert policy.after_tool(env, "check_itinerary", {}, result) is None
    assert museum in env.state.items
    assert not [e for e in env.events if e["type"] == "repair"]


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
