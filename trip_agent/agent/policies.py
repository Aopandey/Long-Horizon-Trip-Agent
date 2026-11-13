"""The three agent variants.

All share the model, tools and budget. They differ only in the system prompt and in what the
scaffold adds after a tool call:
- standard: nothing.
- verification: flags unverified facts that a new commitment depends on with little slack.
- recovery: the same, plus dependency-aware repair. When a corrected fact or a failed check
  breaks items, the scaffold checkpoints, removes the broken items (state only; the message
  history stays append-only), tells the model what to re-plan, and caps repairs per conflict.
"""

from __future__ import annotations

from pydantic import BaseModel

from trip_agent.agent import prompts
from trip_agent.checker.consistency import check_itinerary
from trip_agent.env.facts import fact_kind, place_fact, resolve_travel_key
from trip_agent.env.models import ItemKind, duration_of, format_clock
from trip_agent.state import CommitmentEvent, dependency_graph, item_dependencies
from trip_agent.tools.environment import TripEnvironment

REPAIRABLE_CHECKS = {"opening_hours", "travel", "overlap", "day_window", "duplicate"}


class PolicySettings(BaseModel):
    hours_slack_minutes: int = 120
    travel_slack_minutes: int = 20
    budget_slack_fraction: float = 0.3
    max_repairs_per_conflict: int = 3


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
    name = "verification"

    def after_tool(self, env, tool, args, result):
        if "error" in result:
            return None
        if tool in ("add_item", "move_item"):
            item_id = (result.get("added") or result.get("moved") or {}).get("item_id")
            critical = self.critical_unverified(env, item_id) if item_id else []
            if critical:
                return (
                    "[planner] Unverified facts this item relies on: "
                    + ", ".join(critical)
                    + ". Verify before building on them."
                )
        if tool == "set_hotel":
            key = place_fact(result["hotel_id"], "price_per_night")
            if key not in env.state.verified:
                return f"[planner] The hotel price drives the budget. Verify {key}."
        if tool == "verify_fact" and result.get("matches_observed") is False:
            return "[planner] This value differs from the search result. Re-check items that used it and replan them."
        return None

    def critical_unverified(self, env: TripEnvironment, item_id: str) -> list[str]:
        s, state, catalog = self.settings, env.state, env.catalog
        item = state.items.get(item_id)
        if item is None:
            return []
        itinerary, belief = state.itinerary(), state.belief()
        c = env.scenario.constraints
        end = item.start + (duration_of(item, catalog) or 0)
        required = item.kind == ItemKind.visit and item.place_id in c.required_attractions
        out: list[str] = []
        for key in item_dependencies(itinerary, item, catalog):
            if key in state.verified:
                continue
            value = belief.value(key)
            kind = fact_kind(key)
            if kind == "opening_hours" and value is not None:
                slack = int(value) - end if key.endswith(".close_time") else item.start - int(value)
                if slack < s.hours_slack_minutes:
                    out.append(f"{key} (slack {slack} min)")
                elif required:
                    out.append(f"{key} (required attraction)")
            elif kind == "travel_time" and value is not None:
                slack = self._travel_slack(env, item, key, int(value))
                if slack is not None and slack < s.travel_slack_minutes:
                    out.append(f"{key} (slack {slack} min)")
            elif kind == "price":
                report = check_itinerary(itinerary, env.scenario, catalog, belief)
                if report.total_cost is not None and c.budget - report.total_cost < s.budget_slack_fraction * c.budget:
                    out.append(f"{key} (budget slack ${c.budget - report.total_cost:.0f})")
        return out

    @staticmethod
    def _travel_slack(env: TripEnvironment, item, key: str, minutes: int) -> int | None:
        """Spare minutes on the leg identified by key, measured on the current schedule."""
        catalog, c = env.catalog, env.scenario.constraints
        itinerary = env.state.itinerary()
        # (place, arrive by, leave at); the day starts and ends at the hotel.
        stops = [(itinerary.hotel_id, None, c.day_start)]
        stops += [(i.place_id, i.start, i.start + duration_of(i, catalog)) for i in itinerary.day_items(item.day)]
        stops.append((itinerary.hotel_id, c.day_end, None))
        for (a, _, leave_a), (b, arrive_b, _) in zip(stops, stops[1:], strict=False):
            if a and b and resolve_travel_key(catalog, a, b) == key:
                return arrive_b - leave_a - minutes
        return None


class RecoveryPolicy(VerificationPolicy):
    name = "recovery"

    def __init__(self, settings: PolicySettings | None = None):
        super().__init__(settings)
        self.attempts: dict[str, int] = {}
        self.unresolved: set[str] = set()

    def after_tool(self, env, tool, args, result):
        if "error" in result:
            return None
        notes = []
        base = super().after_tool(env, tool, args, result)
        if tool == "verify_fact" and result.get("matches_observed") is False:
            notes.append(self._repair_fact(env, result["fact_key"]))
        elif tool == "check_itinerary":
            env.state.checkpoint(env.step, "checked")
            repair = self._repair_checks(env)
            if repair:
                notes.append(repair)
        elif base:
            notes.append(base)
        return "\n".join(n for n in notes if n) or None

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

    # A failed check: remove the item each repairable violation points at.
    def _repair_checks(self, env: TripEnvironment) -> str | None:
        report = env.last_check
        if report is None:
            return None
        to_remove: dict[str, list[str]] = {}
        budget_note = None
        for v in report.violations:
            if v.unknown_facts:
                continue
            if v.name in REPAIRABLE_CHECKS and v.item_ids:
                culprit = v.item_ids[-1]
                item = env.state.items.get(culprit)
                if item is None:
                    continue
                conflict = f"check:{v.name}:{item.place_id}"
                to_remove.setdefault(conflict, []).append(culprit)
            elif v.name == "budget":
                if self._bump("check:budget", env):
                    priciest = self._priciest(env)
                    budget_note = f"[planner] Over budget. Most expensive items: {priciest}. Replace or drop one."
        notes = [self._remove(env, conflict, sorted(set(ids)), "failed check") for conflict, ids in to_remove.items()]
        if budget_note:
            notes.append(budget_note)
        return "\n".join(n for n in notes if n) or None

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
        item_ids = [i for i in item_ids if i in state.items]  # another conflict may have removed it already
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

    @staticmethod
    def _priciest(env: TripEnvironment) -> list[str]:
        belief, group = env.state.belief(), env.scenario.constraints.group_size
        costs = []
        for item in env.state.items.values():
            value = belief.value(place_fact(item.place_id, "price_per_person"))
            if value is not None:
                costs.append((float(value) * group, item.id, item.place_id))
        return [f"{iid} {pid} ${cost:.0f}" for cost, iid, pid in sorted(costs, reverse=True)[:3]]

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
