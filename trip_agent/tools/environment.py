"""The tool layer the agent acts through.

Search tools return facts filtered through the scenario's corruption plan; verify_fact is the
only tool that reads the trusted value. Every tool call, including verification and checking,
consumes one unit of the shared tool-call budget.

add_item and move_item also report what the agent's own check finds wrong with that item right
away (believed facts only), so fixing a schedule does not need a check_itinerary call per move.
"""

from __future__ import annotations

import inspect
import json
import re
from typing import Any

from trip_agent.checker.consistency import check_itinerary
from trip_agent.corruption.corrupt import CorruptionPlan
from trip_agent.env.facts import FactValue, is_clock_fact, place_fact, resolve_travel_key, true_facts
from trip_agent.env.models import (
    Attraction,
    Catalog,
    Hotel,
    ItemKind,
    ItineraryItem,
    MealType,
    Restaurant,
    Scenario,
    duration_of,
    format_clock,
    parse_clock,
    rooms_needed,
)
from trip_agent.state import AgentState, CommitmentEvent, item_dependencies

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
FACT_KEY_HELP = (
    "Fact keys: '<place_id>.open_time', '<place_id>.close_time', '<place_id>.price_per_person', "
    "'<hotel_id>.price_per_night', or the 'fact_key' returned by get_travel_time (travel:<from>:<to>)."
)


# Words that say nothing about which place is meant ("Boston hotel near the harbor" -> "harbor").
_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "the",
    "of",
    "in",
    "at",
    "to",
    "for",
    "with",
    "near",
    "by",
    "on",
    "or",
    "best",
    "good",
    "nice",
    "cheap",
    "budget",
    "friendly",
    "affordable",
    "popular",
    "famous",
    "boston",
    "city",
    "area",
    "place",
    "places",
    "hotel",
    "hotels",
    "restaurant",
    "restaurants",
    "attraction",
    "attractions",
    "visit",
    "find",
}


def _rank_by_query(places: list, query: str) -> list:
    """Places sharing at least one meaningful word with the query, best matches first."""
    words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 1 and w not in _QUERY_STOPWORDS}
    if not words:
        return list(places)
    scored = []
    for place in places:
        text = " ".join([place.id.replace("_", " "), place.name, place.neighborhood, *place.tags]).lower()
        tokens = set(re.findall(r"[a-z0-9]+", text))
        score = sum(1 for w in words if w in tokens or any(t.startswith(w) for t in tokens))
        if score:
            scored.append((-score, place.id, place))
    return [p for _, _, p in sorted(scored)]


class ToolError(Exception):
    pass


def _fn(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required or []},
        },
    }


TOOL_SPECS: list[dict] = [
    _fn(
        "search_places",
        "Search attractions, restaurants or hotels. Returns hours, prices and durations as reported by the "
        "search service (search results can be wrong; verify_fact returns the checked value).",
        {
            "kind": {"type": "string", "enum": ["attraction", "restaurant", "hotel"]},
            "query": {"type": "string", "description": "Text to match in the name, id or tags"},
            "neighborhood": {"type": "string"},
            "meal_type": {"type": "string", "enum": ["breakfast", "lunch", "dinner"]},
            "place_ids": {"type": "array", "items": {"type": "string"}, "description": "Look up exact ids"},
            "limit": {"type": "integer", "description": "Max results (default 10)"},
        },
    ),
    _fn(
        "get_travel_time",
        "Travel time in minutes between two places (as reported by the search service).",
        {"from_id": {"type": "string"}, "to_id": {"type": "string"}},
        ["from_id", "to_id"],
    ),
    _fn(
        "verify_fact",
        "Check one fact against the trusted source and get its correct value. " + FACT_KEY_HELP,
        {"fact_key": {"type": "string"}},
        ["fact_key"],
    ),
    _fn(
        "set_hotel",
        "Choose the hotel for the whole stay. Every day starts and ends there.",
        {"hotel_id": {"type": "string"}},
        ["hotel_id"],
    ),
    _fn(
        "add_item",
        "Add a visit (attraction) or a meal (restaurant) to the itinerary. Duration comes from the place.",
        {
            "day": {"type": "integer", "description": "Day index, 0 = first day"},
            "place_id": {"type": "string"},
            "start": {"type": "string", "description": "Start time HH:MM (24-hour)"},
            "meal_type": {"type": "string", "enum": ["breakfast", "lunch", "dinner"], "description": "For meals"},
        },
        ["day", "place_id", "start"],
    ),
    _fn("remove_item", "Remove an item from the itinerary.", {"item_id": {"type": "string"}}, ["item_id"]),
    _fn(
        "move_item",
        "Change the day and/or start time of an existing item.",
        {"item_id": {"type": "string"}, "day": {"type": "integer"}, "start": {"type": "string"}},
        ["item_id"],
    ),
    _fn(
        "check_itinerary",
        "Check the current itinerary against every constraint using the facts you have observed or verified.",
        {},
    ),
    _fn("view_itinerary", "Show the current itinerary day by day.", {}),
    _fn(
        "finish",
        "Submit the itinerary and end planning. List any constraints you could not satisfy.",
        {
            "notes": {"type": "string"},
            "unresolved": {"type": "array", "items": {"type": "string"}},
        },
    ),
]
TOOL_NAMES = {spec["function"]["name"] for spec in TOOL_SPECS}
REPEATABLE_LOOKUPS = {"search_places", "get_travel_time", "verify_fact"}


class ToolBudget:
    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def consume(self) -> bool:
        if self.used >= self.limit:
            return False
        self.used += 1
        return True


def _fmt(key: str, value: FactValue | None) -> Any:
    if value is None:
        return None
    return format_clock(int(value)) if is_clock_fact(key) else value


class TripEnvironment:
    def __init__(self, catalog: Catalog, scenario: Scenario, plan: CorruptionPlan, max_tool_calls: int):
        self.catalog = catalog
        self.scenario = scenario
        self.plan = plan
        self.state = AgentState()
        self.budget = ToolBudget(max_tool_calls)
        self._truth = true_facts(catalog)
        self.events: list[dict] = []
        self.step = 0
        self.verify_calls = 0
        self.done = False
        self.stop_reason: str | None = None
        self.finish_notes: str | None = None
        self.unresolved: list[str] = []
        self.last_check = None
        self._seen_calls: dict[str, int] = {}
        self._placements: dict[str, dict[tuple[int, int], int]] = {}  # item id -> (day, start) -> first step

    # ------------------------------------------------------------------ dispatch
    def call(self, name: str, arguments: dict | None) -> dict:
        if self.done:
            return {"error": "planning is over; no more tool calls are executed"}
        if not self.budget.consume():
            self._stop("budget_exhausted")
            return {"error": "tool-call budget exhausted"}
        self.step += 1
        arguments = arguments if isinstance(arguments, dict) else {}
        try:
            if name not in TOOL_NAMES:
                raise ToolError(f"unknown tool {name!r}; available: {sorted(TOOL_NAMES)}")
            handler = getattr(self, f"_tool_{name}")
            accepted = set(inspect.signature(handler).parameters)
            ignored = sorted(set(arguments) - accepted)
            result = handler(**{k: v for k, v in arguments.items() if k in accepted})
            if ignored:
                result["ignored_arguments"] = ignored
        except ToolError as exc:
            result = {"error": str(exc)}
        except TypeError as exc:  # missing required arguments
            result = {"error": f"bad arguments for {name}: {exc}"}
        if name in REPEATABLE_LOOKUPS and "error" not in result:
            signature = json.dumps([name, arguments], sort_keys=True, default=str)
            if signature in self._seen_calls:
                result["repeat_notice"] = (
                    f"Same call as step {self._seen_calls[signature]}; the answer has not changed. "
                    "Use the results you already have or try a different call."
                )
            self._seen_calls.setdefault(signature, self.step)
        result["tool_calls_remaining"] = self.budget.remaining
        if self.budget.remaining == 0 and not self.done:
            self._stop("budget_exhausted")
            result["notice"] = "Tool-call budget exhausted. The current itinerary is submitted as it is."
        self.events.append({"type": "tool", "step": self.step, "tool": name, "args": arguments, "result": result})
        return result

    def _stop(self, reason: str) -> None:
        self.done = True
        self.stop_reason = self.stop_reason or reason

    # ------------------------------------------------------------------ helpers
    def _observe(self, key: str) -> FactValue:
        value = self.plan.observed(key, self._truth[key])
        self.state.record_observed(key, value, self.step)
        return value

    def _place(self, place_id: Any, kinds: tuple[type, ...] | None = None):
        place = self.catalog.place(str(place_id))
        if place is None:
            raise ToolError(f"unknown place id {place_id!r}; use search_places to find ids")
        if kinds and not isinstance(place, kinds):
            raise ToolError(f"{place_id} is a {place.kind}")
        return place

    def _day(self, day: Any) -> int:
        try:
            day = int(day)
        except (TypeError, ValueError) as exc:
            raise ToolError(f"day must be an integer, got {day!r}") from exc
        n = self.scenario.constraints.num_days
        if not 0 <= day < n:
            raise ToolError(f"day must be between 0 and {n - 1}")
        return day

    @staticmethod
    def _clock(value: Any) -> int:
        try:
            return parse_clock(value)
        except ValueError as exc:
            raise ToolError(f"start must be HH:MM (24-hour), got {value!r}") from exc

    def _describe(self, place) -> dict:
        out: dict[str, Any] = {
            "id": place.id,
            "kind": place.kind,
            "name": place.name,
            "neighborhood": place.neighborhood,
        }
        if place.tags:
            out["tags"] = place.tags
        if isinstance(place, Attraction | Restaurant):
            out["open"] = format_clock(int(self._observe(place_fact(place.id, "open_time"))))
            out["close"] = format_clock(int(self._observe(place_fact(place.id, "close_time"))))
            if place.hours.closed_weekdays:
                out["closed_on"] = [WEEKDAYS[d] for d in place.hours.closed_weekdays]
            out["price_per_person"] = self._observe(place_fact(place.id, "price_per_person"))
        if isinstance(place, Attraction):
            out["visit_minutes"] = place.visit_minutes
        if isinstance(place, Restaurant):
            out["meals"] = [m.value for m in place.meals]
            out["meal_minutes"] = place.meal_minutes
        if isinstance(place, Hotel):
            out["price_per_night"] = self._observe(place_fact(place.id, "price_per_night"))
            out["max_guests_per_room"] = place.max_guests_per_room
        return out

    def _item_view(self, item: ItineraryItem) -> dict:
        place = self.catalog.place(item.place_id)
        out = {
            "item_id": item.id,
            "day": item.day,
            "place_id": item.place_id,
            "name": place.name if place else None,
            "kind": item.kind.value,
            "start": format_clock(item.start),
            "end": format_clock(item.start + (duration_of(item, self.catalog) or 0)),
        }
        if item.meal_type:
            out["meal_type"] = item.meal_type.value
        return out

    def _conflicts(self, item_id: str, limit: int = 4) -> list[str]:
        """What the agent's own check finds wrong with this item now, using believed facts only."""
        report = check_itinerary(self.state.itinerary(), self.scenario, self.catalog, self.state.belief())
        return [f"{v.name}: {v.message}" for v in report.violations if item_id in v.item_ids][:limit]

    def _placed(self, item: ItineraryItem) -> int | None:
        """Remember where an item has been; return the step it was first placed here, if it was."""
        seen = self._placements.setdefault(item.id, {})
        earlier = seen.get((item.day, item.start))
        seen.setdefault((item.day, item.start), self.step)
        return earlier

    def _record(self, action: str, item: ItineraryItem | None = None, depends_on=(), **extra) -> None:
        deps = list(depends_on)
        if item is not None and item.id in self.state.items:
            deps = item_dependencies(self.state.itinerary(), item, self.catalog)
        self.state.history.append(
            CommitmentEvent(
                step=self.step,
                action=action,
                item_id=item.id if item else None,
                item=item.model_copy() if item else None,
                depends_on=deps,
                verified_at_commit=sorted(k for k in deps if k in self.state.verified),
                **extra,
            )
        )

    # ------------------------------------------------------------------ tools
    def _tool_search_places(self, kind=None, query=None, neighborhood=None, meal_type=None, place_ids=None, limit=10):
        places = sorted(self.catalog.places.values(), key=lambda p: p.id)
        if place_ids:
            wanted = {str(p) for p in (place_ids if isinstance(place_ids, list) else [place_ids])}
            places = [p for p in places if p.id in wanted]
        if kind:
            places = [p for p in places if p.kind == kind]
        if neighborhood:
            places = [p for p in places if p.neighborhood.lower() == str(neighborhood).lower()]
        if meal_type:
            places = [p for p in places if isinstance(p, Restaurant) and meal_type in {m.value for m in p.meals}]
        note = None
        if query:
            ranked = _rank_by_query(places, str(query))
            if ranked:
                places = ranked
            else:
                note = "No place matched the query words; showing every result for the other filters."
        try:
            limit = max(1, min(int(limit), 25))
        except (TypeError, ValueError):
            limit = 10
        result = {"results": [self._describe(p) for p in places[:limit]], "total_matches": len(places)}
        if not places and neighborhood:
            elsewhere = self._tool_neighborhoods(kind, meal_type)
            note = f"No matches in {neighborhood}. Matching places exist in: {', '.join(elsewhere) or 'none'}."
        if note:
            result["note"] = note
        if len(places) > limit:
            result["more"] = f"{len(places) - limit} more; raise limit or filter by kind, neighborhood or meal_type"
        return result

    def _tool_neighborhoods(self, kind=None, meal_type=None) -> list[str]:
        """Neighborhoods that have places of this kind (and meal type), for empty-result hints."""
        hoods = set()
        for place in self.catalog.places.values():
            if kind and place.kind != kind:
                continue
            if meal_type and not (isinstance(place, Restaurant) and meal_type in {m.value for m in place.meals}):
                continue
            hoods.add(place.neighborhood)
        return sorted(hoods)

    def _tool_get_travel_time(self, from_id, to_id):
        a, b = self._place(from_id), self._place(to_id)
        if a.id == b.id:
            return {"from": a.id, "to": b.id, "minutes": self.catalog.same_place_minutes, "fact_key": None}
        key = resolve_travel_key(self.catalog, a.id, b.id)
        if key is None:
            raise ToolError(f"no travel data between {a.id} and {b.id}")
        leg = self.catalog.leg(a.id, b.id)
        return {
            "from": a.id,
            "to": b.id,
            "minutes": self._observe(key),
            "fact_key": key,
            "transit_cost_per_person": leg.cost_per_person,
        }

    def _tool_verify_fact(self, fact_key):
        key = str(fact_key).strip()
        if key not in self._truth:
            raise ToolError(f"unknown fact key {key!r}. {FACT_KEY_HELP}")
        self.verify_calls += 1
        true_value = self._truth[key]
        previous = self.state.observed.get(key)
        self.state.record_verified(key, true_value, self.step)
        return {
            "fact_key": key,
            "verified_value": _fmt(key, true_value),
            "previously_observed": _fmt(key, previous),
            "matches_observed": None if previous is None else previous == true_value,
        }

    def _tool_set_hotel(self, hotel_id):
        hotel = self._place(hotel_id, (Hotel,))
        self.state.hotel_id = hotel.id
        c = self.scenario.constraints
        self._record("set_hotel", hotel_id=hotel.id, depends_on=[place_fact(hotel.id, "price_per_night")])
        return {"hotel_id": hotel.id, "rooms_needed": rooms_needed(c.group_size, hotel), "nights": c.hotel_nights}

    def _tool_add_item(self, day, place_id, start, meal_type=None):
        place = self._place(place_id, (Attraction, Restaurant))
        meal = None
        if isinstance(place, Restaurant):
            if meal_type is None:
                raise ToolError(
                    f"meal_type is required for restaurants; {place.id} serves {[m.value for m in place.meals]}"
                )
            try:
                meal = MealType(str(meal_type))
            except ValueError as exc:
                raise ToolError("meal_type must be breakfast, lunch or dinner") from exc
        item = ItineraryItem(
            id=self.state.new_item_id(),
            day=self._day(day),
            place_id=place.id,
            kind=ItemKind.visit if isinstance(place, Attraction) else ItemKind.meal,
            meal_type=meal,
            start=self._clock(start),
        )
        self.state.items[item.id] = item
        self._record("add", item)
        self._placed(item)
        return {
            "added": self._item_view(item),
            "depends_on_facts": self.state.history[-1].depends_on,
            "conflicts": self._conflicts(item.id),
        }

    def _tool_remove_item(self, item_id):
        item = self.state.items.pop(str(item_id), None)
        if item is None:
            raise ToolError(f"no item {item_id!r}; current items: {sorted(self.state.items)}")
        self._record("remove", item)
        return {"removed": item.id}

    def _tool_move_item(self, item_id, day=None, start=None):
        item = self.state.items.get(str(item_id))
        if item is None:
            raise ToolError(f"no item {item_id!r}; current items: {sorted(self.state.items)}")
        if day is None and start is None:
            raise ToolError("give a new day and/or start")
        updated = item.model_copy(
            update={
                "day": self._day(day) if day is not None else item.day,
                "start": self._clock(start) if start is not None else item.start,
            }
        )
        self.state.items[item.id] = updated
        self._record("move", updated)
        result = {"moved": self._item_view(updated), "conflicts": self._conflicts(item.id)}
        earlier = self._placed(updated)
        if earlier is not None and (updated.day, updated.start) != (item.day, item.start):
            result["repeat_notice"] = (
                f"{item.id} was already at this time (step {earlier}). Moving items back and forth will not fix "
                "this: reorder the day or choose another place."
            )
        return result

    def _tool_check_itinerary(self):
        report = check_itinerary(self.state.itinerary(), self.scenario, self.catalog, self.state.belief())
        self.last_check = report
        return report.for_agent()

    def _tool_view_itinerary(self):
        c = self.scenario.constraints
        itin = self.state.itinerary()
        scheduled = {i.place_id for i in itin.items}
        return {
            "hotel_id": itin.hotel_id,
            "days": [
                {
                    "day": d,
                    "date": c.date_of(d).isoformat(),
                    "weekday": WEEKDAYS[c.weekday(d)],
                    "items": [self._item_view(i) for i in itin.day_items(d)],
                }
                for d in range(c.num_days)
            ],
            "required_attractions_missing": [a for a in c.required_attractions if a not in scheduled],
        }

    def _tool_finish(self, notes="", unresolved=None):
        self.finish_notes = str(notes or "")
        if unresolved is None:
            unresolved = []
        self.unresolved = [str(u) for u in unresolved] if isinstance(unresolved, list) else [str(unresolved)]
        self._stop("finished")
        return {"finished": True}

    # ------------------------------------------------------------------ prompt material
    def task_brief(self) -> str:
        c = self.scenario.constraints
        names = []
        for aid in c.required_attractions:
            place = self.catalog.place(aid)
            names.append(f"{place.name} ({aid})" if place else aid)
        meals = ", ".join(
            f"{m.value} (start between {format_clock(w.start)} and {format_clock(w.end)})"
            for m, w in ((m, c.meal_windows[m]) for m in c.meals_per_day)
        )
        days = "; ".join(f"day {d} = {c.date_of(d).isoformat()} ({WEEKDAYS[c.weekday(d)]})" for d in range(c.num_days))
        hotel_rule = f" in one of: {', '.join(c.hotel_neighborhoods)}" if c.hotel_neighborhoods else ""
        lines = [
            f"Plan a {c.num_days}-day trip in {self.catalog.city} (synthetic data) for {c.group_size} people.",
            f"Days: {days}.",
            f"Total budget: ${c.budget:.2f} for the whole group. Hotel cost = price per night x rooms x "
            f"{c.hotel_nights} nights; attractions and meals are priced per person.",
            f"Each day runs {format_clock(c.day_start)}-{format_clock(c.day_end)} and starts and ends at the hotel; "
            "leave enough travel time between stops.",
            f"Choose one hotel for the whole stay{hotel_rule}.",
            f"Required attractions (visit each once): {', '.join(names) if names else 'none'}.",
            f"Required meals every day: {meals}.",
            f"You have {self.budget.limit} tool calls in total. Every call, including verify_fact and "
            "check_itinerary, uses one.",
            "When the itinerary is complete and checked, call finish.",
        ]
        return "\n".join(lines)
