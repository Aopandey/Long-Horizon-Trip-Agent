"""Fact keys: the unit of misinformation and verification.

Only these attributes are "eligible" facts. They can be corrupted in search responses and
checked with verify_fact. Everything else in the catalog (names, durations, meal types,
neighborhoods, room capacity, closed weekdays, travel cost) is always reported truthfully.

Key format:
  <place_id>.open_time | <place_id>.close_time | <place_id>.price_per_person   (attractions, restaurants)
  <place_id>.price_per_night                                                   (hotels)
  travel:<from_id>:<to_id>                                                     (stored direction of the leg)
"""

from __future__ import annotations

from typing import Literal

from trip_agent.env.models import Attraction, Catalog, Hotel, Restaurant, TravelLeg

FactKind = Literal["opening_hours", "price", "travel_time"]
FactValue = int | float

PLACE_ATTRIBUTES = ("open_time", "close_time", "price_per_person", "price_per_night")


def place_fact(place_id: str, attribute: str) -> str:
    return f"{place_id}.{attribute}"


def travel_fact(leg: TravelLeg) -> str:
    return f"travel:{leg.from_id}:{leg.to_id}"


def fact_kind(key: str) -> FactKind:
    if key.startswith("travel:"):
        return "travel_time"
    attribute = key.rsplit(".", 1)[-1]
    if attribute in ("open_time", "close_time"):
        return "opening_hours"
    return "price"


def is_clock_fact(key: str) -> bool:
    return key.endswith(".open_time") or key.endswith(".close_time")


def true_facts(catalog: Catalog) -> dict[str, FactValue]:
    """Every eligible fact in the catalog with its correct value."""
    facts: dict[str, FactValue] = {}
    for place in catalog.places.values():
        if isinstance(place, Attraction | Restaurant):
            facts[place_fact(place.id, "open_time")] = place.hours.open
            facts[place_fact(place.id, "close_time")] = place.hours.close
            facts[place_fact(place.id, "price_per_person")] = place.price_per_person
        elif isinstance(place, Hotel):
            facts[place_fact(place.id, "price_per_night")] = place.price_per_night
    for leg in catalog.travel:
        facts[travel_fact(leg)] = leg.minutes
    return facts


def resolve_travel_key(catalog: Catalog, from_id: str, to_id: str) -> str | None:
    """Fact key of the leg used between two places, or None when no leg exists or a == b."""
    if from_id == to_id:
        return None
    leg = catalog.leg(from_id, to_id)
    return travel_fact(leg) if leg else None
