"""Deterministic itinerary checker.

One checker, two views: the agent's check_itinerary tool runs it on a BeliefView (only facts
the agent observed or verified), and the evaluator runs it on a TruthView. The logic is shared
so the two can never disagree about what a constraint means, and the belief view never
reveals hidden correct values.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from trip_agent.env.facts import FactValue, place_fact, resolve_travel_key, true_facts
from trip_agent.env.models import (
    Attraction,
    Catalog,
    Hotel,
    ItemKind,
    Itinerary,
    ItineraryItem,
    Restaurant,
    Scenario,
    duration_of,
    format_clock,
    rooms_needed,
)

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


class FactView(Protocol):
    def value(self, key: str) -> FactValue | None: ...


class TruthView:
    def __init__(self, catalog: Catalog):
        self._facts = true_facts(catalog)

    def value(self, key: str) -> FactValue | None:
        return self._facts.get(key)


class BeliefView:
    """What the agent currently believes: verified values override observed ones."""

    def __init__(self, observed: dict[str, FactValue], verified: dict[str, FactValue]):
        self._observed = observed
        self._verified = verified

    def value(self, key: str) -> FactValue | None:
        if key in self._verified:
            return self._verified[key]
        return self._observed.get(key)


class Check(BaseModel):
    name: str
    ok: bool
    message: str
    item_ids: list[str] = Field(default_factory=list)
    fact_keys: list[str] = Field(default_factory=list)
    unknown_facts: list[str] = Field(default_factory=list)


class CheckReport(BaseModel):
    checks: list[Check]
    total_cost: float | None
    cost_breakdown: dict[str, float | None]

    @property
    def violations(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    @property
    def valid(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def fraction_satisfied(self) -> float:
        return sum(c.ok for c in self.checks) / len(self.checks) if self.checks else 0.0

    def for_agent(self, max_violations: int = 15) -> dict:
        """Compact summary returned by the check_itinerary tool."""
        violations = self.violations
        return {
            "valid": self.valid,
            "satisfied": f"{sum(c.ok for c in self.checks)}/{len(self.checks)}",
            "estimated_total_cost": None if self.total_cost is None else round(self.total_cost, 2),
            "violations": [
                {
                    "check": v.name,
                    "message": v.message,
                    "items": v.item_ids,
                    **({"unknown_facts": v.unknown_facts} if v.unknown_facts else {}),
                }
                for v in violations[:max_violations]
            ],
            **({"more_violations": len(violations) - max_violations} if len(violations) > max_violations else {}),
        }


def check_itinerary(itinerary: Itinerary, scenario: Scenario, catalog: Catalog, view: FactView) -> CheckReport:
    c = scenario.constraints
    checks: list[Check] = []

    def add(name: str, ok: bool, message: str, items=(), facts=(), unknown=()) -> None:
        checks.append(
            Check(
                name=name,
                ok=ok,
                message=message,
                item_ids=list(items),
                fact_keys=[f for f in facts if f],
                unknown_facts=list(unknown),
            )
        )

    # Hotel.
    hotel = catalog.place(itinerary.hotel_id) if itinerary.hotel_id else None
    if not isinstance(hotel, Hotel):
        hotel = None
        add("hotel", False, "no hotel selected" if not itinerary.hotel_id else f"{itinerary.hotel_id} is not a hotel")
    elif c.hotel_neighborhoods and hotel.neighborhood not in c.hotel_neighborhoods:
        add("hotel", False, f"hotel {hotel.id} is in {hotel.neighborhood}, allowed: {c.hotel_neighborhoods}")
    else:
        add("hotel", True, f"hotel {hotel.id}")

    # Structural validity of each item.
    valid_items: list[ItineraryItem] = []
    visited: dict[str, str] = {}
    for item in sorted(itinerary.items, key=lambda i: (i.day, i.start, i.id)):
        place = catalog.place(item.place_id)
        problem = None
        if place is None:
            problem = f"unknown place {item.place_id}"
        elif item.kind == ItemKind.visit and not isinstance(place, Attraction):
            problem = f"{item.place_id} is not an attraction"
        elif item.kind == ItemKind.meal and not isinstance(place, Restaurant):
            problem = f"{item.place_id} is not a restaurant"
        elif item.kind == ItemKind.meal and (item.meal_type is None or item.meal_type not in place.meals):
            problem = f"{item.place_id} does not serve {item.meal_type.value if item.meal_type else 'that meal'}"
        elif not 0 <= item.day < c.num_days:
            problem = f"day {item.day} is outside the trip (days 0..{c.num_days - 1})"
        add("place", problem is None, problem or f"{item.id} ok", [item.id])
        if problem:
            continue
        if isinstance(place, Attraction):
            if place.id in visited:
                add("duplicate", False, f"{place.id} already visited in {visited[place.id]}", [item.id])
            visited.setdefault(place.id, item.id)
        valid_items.append(item)

    # Opening hours and day window.
    for item in valid_items:
        place = catalog.place(item.place_id)
        assert isinstance(place, Attraction | Restaurant)
        end = item.start + duration_of(item, catalog)
        open_key, close_key = place_fact(place.id, "open_time"), place_fact(place.id, "close_time")
        weekday = c.weekday(item.day)
        if weekday in place.hours.closed_weekdays:
            add(
                "opening_hours", False, f"{place.id} is closed on {WEEKDAYS[weekday]}", [item.id], [open_key, close_key]
            )
        else:
            open_v, close_v = view.value(open_key), view.value(close_key)
            unknown = [k for k, v in ((open_key, open_v), (close_key, close_v)) if v is None]
            if unknown:
                add("opening_hours", False, f"{place.id} hours unknown", [item.id], [open_key, close_key], unknown)
            else:
                ok = open_v <= item.start and end <= close_v
                add(
                    "opening_hours",
                    ok,
                    f"{place.id} {format_clock(item.start)}-{format_clock(end)}, open "
                    f"{format_clock(int(open_v))}-{format_clock(int(close_v))}",
                    [item.id],
                    [open_key, close_key],
                )
        in_day = c.day_start <= item.start and end <= c.day_end
        add(
            "day_window",
            in_day,
            f"{item.id} {format_clock(item.start)}-{format_clock(end)}, day "
            f"{format_clock(c.day_start)}-{format_clock(c.day_end)}",
            [item.id],
        )

    # Required meals.
    for day in range(c.num_days):
        for meal in c.meals_per_day:
            window = c.meal_windows[meal]
            candidates = [i for i in valid_items if i.day == day and i.kind == ItemKind.meal and i.meal_type == meal]
            ok = any(window.start <= i.start <= window.end for i in candidates)
            add(
                "meal",
                ok,
                f"day {day} {meal.value} {'ok' if ok else 'missing'} (must start "
                f"{format_clock(window.start)}-{format_clock(window.end)})",
                [i.id for i in candidates],
            )

    # Required activities.
    for attraction_id in c.required_attractions:
        matches = [i.id for i in valid_items if i.place_id == attraction_id and i.kind == ItemKind.visit]
        add("required_activity", bool(matches), f"{attraction_id} {'scheduled' if matches else 'missing'}", matches)

    # Sequencing: overlaps and travel feasibility, day by day, starting and ending at the hotel.
    legs_used: list[tuple[str, str]] = []

    def travel(a: str, t_a: int, b: str, t_b: int, ids: list[str]) -> None:
        if a == b:
            minutes, key = catalog.same_place_minutes, None
        else:
            key = resolve_travel_key(catalog, a, b)
            if key is None:
                add("travel", False, f"no travel data {a} -> {b}", ids)
                return
            minutes = view.value(key)
            legs_used.append((a, b))
            if minutes is None:
                add("travel", False, f"travel time {a} -> {b} unknown", ids, [key], [key])
                return
        gap = t_b - t_a
        add("travel", gap >= minutes, f"{a} -> {b}: {gap} min available, {int(minutes)} min needed", ids, [key])

    for day in range(c.num_days):
        day_items = sorted((i for i in valid_items if i.day == day), key=lambda i: (i.start, i.id))
        prev: ItineraryItem | None = None
        prev_end = c.day_start
        for item in day_items:
            if prev is not None:
                add(
                    "overlap",
                    prev_end <= item.start,
                    f"{prev.id} ends {format_clock(prev_end)}, {item.id} starts {format_clock(item.start)}",
                    [prev.id, item.id],
                )
                travel(prev.place_id, prev_end, item.place_id, item.start, [prev.id, item.id])
            elif hotel is not None:
                travel(hotel.id, c.day_start, item.place_id, item.start, [item.id])
            prev, prev_end = item, item.start + duration_of(item, catalog)
        if hotel is not None and prev is not None:
            travel(prev.place_id, prev_end, hotel.id, c.day_end, [prev.id])

    # Budget.
    unknown_prices: list[str] = []
    price_keys: list[str] = []

    def price(key: str) -> float:
        price_keys.append(key)
        value = view.value(key)
        if value is None:
            unknown_prices.append(key)
            return 0.0
        return float(value)

    breakdown: dict[str, float | None] = {}
    if hotel is not None:
        breakdown["hotel"] = (
            price(place_fact(hotel.id, "price_per_night")) * rooms_needed(c.group_size, hotel) * c.hotel_nights
        )
    breakdown["attractions"] = sum(
        price(place_fact(i.place_id, "price_per_person")) * c.group_size
        for i in valid_items
        if i.kind == ItemKind.visit
    )
    breakdown["meals"] = sum(
        price(place_fact(i.place_id, "price_per_person")) * c.group_size for i in valid_items if i.kind == ItemKind.meal
    )
    breakdown["transit"] = sum(catalog.leg(a, b).cost_per_person * c.group_size for a, b in legs_used)
    total = None if unknown_prices else sum(v for v in breakdown.values() if v is not None)
    if unknown_prices:
        add("budget", False, "cost unknown: some prices not observed", facts=price_keys, unknown=unknown_prices)
    else:
        add("budget", total <= c.budget + 1e-6, f"total {total:.2f} vs budget {c.budget:.2f}", facts=price_keys)

    return CheckReport(checks=checks, total_cost=total, cost_breakdown=breakdown)
