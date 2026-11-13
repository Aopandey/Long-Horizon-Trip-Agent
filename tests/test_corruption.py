from trip_agent.corruption.corrupt import build_plan, clean_plan, eligible_facts
from trip_agent.env.facts import fact_kind, true_facts


def test_exact_rate_and_determinism(catalog):
    a = build_plan(catalog, "s1", rate=0.2, seed=0)
    b = build_plan(catalog, "s1", rate=0.2, seed=0)
    assert a == b
    assert len(a.altered) == round(0.2 * len(eligible_facts(catalog)))
    assert a.eligible_count == len(eligible_facts(catalog))


def test_different_scenarios_or_seeds_change_the_selection(catalog):
    base = set(build_plan(catalog, "s1").altered)
    assert set(build_plan(catalog, "s2").altered) != base
    assert set(build_plan(catalog, "s1", seed=1).altered) != base


def test_every_alteration_is_optimistic_and_different(catalog):
    truth = true_facts(catalog)
    plan = build_plan(catalog, "s1", rate=1.0)
    assert len(plan.altered) == plan.eligible_count
    for key, value in plan.altered.items():
        assert value != truth[key]
        kind = fact_kind(key)
        if kind == "price":
            assert value < truth[key]
        elif kind == "travel_time":
            assert value < truth[key]
        elif key.endswith(".close_time") and truth[key] < 1440:
            assert value > truth[key]
        elif key.endswith(".open_time") and truth[key] > 0:
            assert value < truth[key]


def test_free_places_are_not_eligible(catalog):
    assert "a_liberty_walk.price_per_person" not in eligible_facts(catalog)


def test_clean_plan_changes_nothing(catalog):
    plan = clean_plan("s1")
    assert plan.altered == {}
    assert plan.observed("x", 5) == 5
