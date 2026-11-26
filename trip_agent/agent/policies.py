"""The three agent variants.

All share the model, tools and budget. They differ only in the system prompt and in what the
scaffold adds after a tool call:
- standard: nothing.
- verification: build first, verify last. Once the plan passes the agent's own check, the scaffold
  lists the unverified facts the plan relies on with little margin, riskiest first.
- recovery: the same, plus dependency-aware repair. When verify_fact shows that a search result was
  wrong, the scaffold checkpoints, removes only the items the correct value breaks (state only; the
  message history stays append-only), tells the model what to re-plan, and caps repairs per conflict.
  Ordinary scheduling conflicts are left to the model, as in the verification variant.
"""

from __future__ import annotations

from itertools import pairwise
from typing import NamedTuple

from pydantic import BaseModel

from trip_agent.agent import prompts
from trip_agent.checker.consistency import FactView, check_itinerary
from trip_agent.env.facts import place_fact, resolve_travel_key
from trip_agent.env.models import Catalog, Hotel, Itinerary, Scenario, duration_of, format_clock, rooms_needed
from trip_agent.state import CommitmentEvent, dependency_graph
from trip_agent.tools.environment import TripEnvironment


class PolicySettings(BaseModel):
    hours_slack_minutes: int = 120
    travel_slack_minutes: int = 20
    travel_error_ratio: float = 1.0
    price_error_fraction: float = 0.6
    max_repairs_per_conflict: int = 3


class RiskyFact(NamedTuple):
    key: str
    score: float  # spare margin / margin needed; below 1 an error in this fact could break the plan
    reason: str


def risky_facts(
    itinerary: Itinerary,
    scenario: Scenario,
    catalog: Catalog,
    belief: FactView,
    verified: set[str] | dict,
    s: PolicySettings,
) -> list[RiskyFact]:
    """Unverified facts the plan relies on whose error could break it, riskiest first.

    Wrong search results make plans look easier than they are (later closing, earlier opening,
    lower prices, shorter travel), so a fact matters only when the plan uses up the margin it gives:
    - opening hours: fewer than hours_slack_minutes between the item and the opening or closing time;
    - travel: fewer spare minutes on the leg than max(travel_slack_minutes, travel_error_ratio x minutes);
    - prices: the fewest prices, largest first, such that the rest being price_error_fraction too low
      still fits the budget.
    """
    c = scenario.constraints
    found: dict[str, RiskyFact] = {}

    def keep(key: str, spare: float, needed: float, reason: str) -> None:
        score = max(spare, 0) / needed if needed > 0 else 1.0
        if key in verified or score >= 1:
            return
        if key not in found or score < found[key].score:
            found[key] = RiskyFact(key, score, reason)

    for day in range(c.num_days):
        items = itinerary.day_items(day)
        stops = [(itinerary.hotel_id, None, c.day_start)]
        for item in items:
            end = item.start + (duration_of(item, catalog) or 0)
            stops.append((item.place_id, item.start, end))
            for attribute in ("open_time", "close_time"):
                key = place_fact(item.place_id, attribute)
                value = belief.value(key)
                if value is not None:
                    spare = int(value) - end if attribute == "close_time" else item.start - int(value)
                    keep(key, spare, s.hours_slack_minutes, f"{spare} min to spare")
        if not items:
            continue
        stops.append((itinerary.hotel_id, c.day_end, None))
        for (a, _, leave), (b, arrive, _) in pairwise(stops):
            key = resolve_travel_key(catalog, a, b) if a and b else None
            value = belief.value(key) if key else None
            if value is not None:
                spare = arrive - leave - int(value)
                needed = max(s.travel_slack_minutes, s.travel_error_ratio * int(value))
                keep(key, spare, needed, f"{spare} min to spare")

    report = check_itinerary(itinerary, scenario, catalog, belief)
    if report.total_cost is not None:
        spare = c.budget - report.total_cost
        cost: dict[str, float] = {}
        hotel = catalog.place(itinerary.hotel_id) if itinerary.hotel_id else None
        if isinstance(hotel, Hotel):
            key = place_fact(hotel.id, "price_per_night")
            cost[key] = float(belief.value(key) or 0) * rooms_needed(c.group_size, hotel) * c.hotel_nights
        for item in itinerary.items:
            key = place_fact(item.place_id, "price_per_person")
            cost[key] = cost.get(key, 0.0) + float(belief.value(key) or 0) * c.group_size
        at_risk = sorted(
            ((v * s.price_error_fraction, k) for k, v in cost.items() if v > 0 and k not in verified), reverse=True
        )
        remaining = sum(p for p, _ in at_risk)
        for possible_error, key in at_risk:
            if remaining <= spare:
                break
            keep(key, spare, remaining, f"${spare:.0f} of budget to spare")
            remaining -= possible_error
    return sorted(found.values(), key=lambda f: (f.score, f.key))


class StandardPolicy:
    name = "standard"

    def __init__(self, settings: PolicySettings | None = None):
        self.settings = settings or PolicySettings()

    def system_prompt(self) -> str:
        return prompts.system_prompt(self.name)

    def after_tool(self, env: TripEnvironment, tool: str, args: dict, result: dict) -> str | None:
        return None

    def finalize(self, env: TripEnvironment) -> None:
        return None


class VerificationPolicy(StandardPolicy):
    """Build first, verify last.

    Nothing is flagged while the plan is being built. Once check_itinerary passes on the agent's
    beliefs, the planner lists the unverified facts the plan relies on with little margin, riskiest
    first, so verification spends calls only on facts that could break a finished plan.
    """

    name = "verification"
    max_listed = 10

    def after_tool(self, env, tool, args, result):
        if "error" in result:
            return None
        if tool == "check_itinerary" and result.get("valid"):
            return self.verify_list(env)
        if tool == "verify_fact" and result.get("matches_observed") is False:
            return "[planner] This value differs from the search result. Fix the items that used it, then check again."
        return None

    def risky(self, env: TripEnvironment) -> list[RiskyFact]:
        state = env.state
        return risky_facts(state.itinerary(), env.scenario, env.catalog, state.belief(), state.verified, self.settings)

    def verify_list(self, env: TripEnvironment) -> str:
        risky = self.risky(env)
        if not risky:
            return "[planner] The plan passes and every fact it relies on is verified or has enough margin. Finish."
        shown = ", ".join(f"{f.key} ({f.reason})" for f in risky[: self.max_listed])
        more = f" and {len(risky) - self.max_listed} more" if len(risky) > self.max_listed else ""
        return (
            f"[planner] The plan passes on what you have seen. Verify these facts it relies on with little "
            f"margin, riskiest first: {shown}{more}. Then call check_itinerary again."
        )


class RecoveryPolicy(VerificationPolicy):
    """Verification plus dependency-aware repair of items broken by a corrected fact.

    Repair is triggered only when verify_fact shows that a search result was wrong. Ordinary
    scheduling conflicts are left to the model, exactly as in the verification variant.
    """

    name = "recovery"

    def __init__(self, settings: PolicySettings | None = None):
        super().__init__(settings)
        self.attempts: dict[str, int] = {}
        self.unresolved: set[str] = set()

    def after_tool(self, env, tool, args, result):
        if "error" in result:
            return None
        if tool == "check_itinerary":
            env.state.checkpoint(env.step, "checked")
        if tool == "verify_fact" and result.get("matches_observed") is False:
            return self._repair_fact(env, result["fact_key"])
        return super().after_tool(env, tool, args, result)

    # A corrected fact: remove only the items it now breaks.
    def _repair_fact(self, env: TripEnvironment, key: str) -> str:
        itinerary = env.state.itinerary()
        dependents = dependency_graph(itinerary, env.catalog).get(key, set())
        if not dependents:
            return f"[planner] Corrected {key}. No scheduled item depends on it yet."
        report = check_itinerary(itinerary, env.scenario, env.catalog, env.state.belief())
        broken = sorted(
            {i for v in report.violations if key in v.fact_keys and not v.unknown_facts for i in v.item_ids}
            & dependents
        )
        if not broken:
            return f"[planner] Corrected {key}. Items {sorted(dependents)} still fit with the correct value."
        return self._remove(env, f"fact:{key}", broken, f"corrected {key}")

    def _bump(self, conflict: str, env: TripEnvironment) -> bool:
        """Count a repair attempt; False once the conflict has used up its attempts."""
        if conflict in self.unresolved:
            return False
        self.attempts[conflict] = self.attempts.get(conflict, 0) + 1
        if self.attempts[conflict] > self.settings.max_repairs_per_conflict:
            self.unresolved.add(conflict)
            env.events.append({"type": "conflict_unresolved", "step": env.step, "conflict": conflict})
            return False
        return True

    def _remove(self, env: TripEnvironment, conflict: str, item_ids: list[str], why: str) -> str | None:
        state, c = env.state, env.scenario.constraints
        item_ids = [i for i in item_ids if i in state.items]
        if not item_ids:
            return None
        if not self._bump(conflict, env):
            limit = self.settings.max_repairs_per_conflict
            return f"[planner] {conflict} is still unresolved after {limit} repairs. Report it in `unresolved`."
        state.checkpoint(env.step, f"before repair of {conflict}")
        removed, downstream = [], set()
        for item_id in item_ids:
            item = state.items[item_id]
            downstream.update(state.downstream(item_id))
            del state.items[item_id]
            state.history.append(CommitmentEvent(step=env.step, action="repair_remove", item_id=item_id, item=item))
            label = f"{item_id} ({item.place_id} day {item.day} {format_clock(item.start)}"
            label += ", required)" if item.place_id in c.required_attractions else ")"
            removed.append(label)
        downstream -= set(item_ids)
        env.events.append(
            {
                "type": "repair",
                "step": env.step,
                "conflict": conflict,
                "removed": item_ids,
                "downstream": sorted(downstream),
            }
        )
        attempt = self.attempts[conflict]
        limit = self.settings.max_repairs_per_conflict
        text = f"[planner] Repair {attempt}/{limit} for {conflict} ({why}): removed {', '.join(removed)}."
        if downstream:
            text += f" Later items on the same day depend on their timing: {sorted(downstream)}; re-check them."
        text += " They are already removed (do not call remove_item for them)."
        return text + " Re-add them at feasible times, then call check_itinerary."

    def finalize(self, env: TripEnvironment) -> None:
        """Submit the best checked state if planning ended in a worse one, and report open conflicts."""
        state = env.state
        belief = state.belief()

        def violations(itinerary) -> int:
            return len(check_itinerary(itinerary, env.scenario, env.catalog, belief).violations)

        current = violations(state.itinerary())
        best = min(
            (cp for cp in state.checkpoints if cp.reason == "checked"),
            key=lambda cp: violations(cp.itinerary),
            default=None,
        )
        if best is not None and violations(best.itinerary) < current:
            state.restore(best.itinerary, env.step)
            env.events.append({"type": "restore_checkpoint", "step": best.step})
        env.unresolved = list(dict.fromkeys([*env.unresolved, *sorted(self.unresolved)]))


POLICIES = {"standard": StandardPolicy, "verification": VerificationPolicy, "recovery": RecoveryPolicy}


def make_policy(variant: str, settings: PolicySettings | None = None) -> StandardPolicy:
    try:
        return POLICIES[variant](settings)
    except KeyError as exc:
        raise ValueError(f"unknown variant {variant!r}; choose from {sorted(POLICIES)}") from exc
