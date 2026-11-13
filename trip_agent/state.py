"""Structured agent state kept by the scaffold (not by the model).

Holds what the agent has observed and verified, its current itinerary, the commitment
history, checkpoints, and the fact -> commitment dependency graph used by the recovery variant.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from trip_agent.checker.consistency import BeliefView
from trip_agent.env.facts import FactValue, place_fact, resolve_travel_key
from trip_agent.env.models import Catalog, Itinerary, ItineraryItem


class Checkpoint(BaseModel):
    step: int
    reason: str
    itinerary: Itinerary


class CommitmentEvent(BaseModel):
    """One change to the itinerary. The evaluator replays these to find the first error."""

    step: int
    action: str  # add | remove | move | set_hotel | repair_remove | restore
    item_id: str | None = None
    item: ItineraryItem | None = None
    hotel_id: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    verified_at_commit: list[str] = Field(default_factory=list)
    snapshot: Itinerary | None = None  # set on restore, so the evaluator can replay


class AgentState(BaseModel):
    observed: dict[str, FactValue] = Field(default_factory=dict)
    verified: dict[str, FactValue] = Field(default_factory=dict)
    observed_step: dict[str, int] = Field(default_factory=dict)  # first tool step that showed the fact
    verified_step: dict[str, int] = Field(default_factory=dict)
    hotel_id: str | None = None
    items: dict[str, ItineraryItem] = Field(default_factory=dict)
    next_item: int = 1
    history: list[CommitmentEvent] = Field(default_factory=list)
    checkpoints: list[Checkpoint] = Field(default_factory=list)

    # Facts and beliefs.
    def record_observed(self, key: str, value: FactValue, step: int) -> None:
        self.observed[key] = value
        self.observed_step.setdefault(key, step)

    def record_verified(self, key: str, value: FactValue, step: int) -> None:
        self.verified[key] = value
        self.verified_step.setdefault(key, step)

    def belief(self) -> BeliefView:
        return BeliefView(self.observed, self.verified)

    # Itinerary.
    def itinerary(self) -> Itinerary:
        return Itinerary(hotel_id=self.hotel_id, items=[i.model_copy() for i in self.items.values()])

    def new_item_id(self) -> str:
        item_id = f"c{self.next_item}"
        self.next_item += 1
        return item_id

    def checkpoint(self, step: int, reason: str) -> Checkpoint:
        cp = Checkpoint(step=step, reason=reason, itinerary=self.itinerary())
        self.checkpoints.append(cp)
        return cp

    def restore(self, itinerary: Itinerary, step: int) -> None:
        self.hotel_id = itinerary.hotel_id
        self.items = {i.id: i.model_copy() for i in itinerary.items}
        self.history.append(
            CommitmentEvent(
                step=step, action="restore", hotel_id=self.hotel_id, snapshot=itinerary.model_copy(deep=True)
            )
        )

    def same_day(self, item_id: str) -> list[ItineraryItem]:
        day = self.items[item_id].day
        return sorted((i for i in self.items.values() if i.day == day), key=lambda i: (i.start, i.id))

    def downstream(self, item_id: str) -> list[str]:
        """Same-day items scheduled after this one: their timing chains through it."""
        item = self.items[item_id]
        return [i.id for i in self.same_day(item_id) if (i.start, i.id) > (item.start, item.id)]


def item_dependencies(itinerary: Itinerary, item: ItineraryItem, catalog: Catalog) -> list[str]:
    """Eligible facts a commitment relies on, given its current neighbors.

    Own opening hours and price, and the travel legs to the previous and next stop on the
    same day (the hotel at either end). The hotel price belongs to the set_hotel commitment.
    """
    deps = [
        place_fact(item.place_id, "open_time"),
        place_fact(item.place_id, "close_time"),
        place_fact(item.place_id, "price_per_person"),
    ]
    day_items = itinerary.day_items(item.day)
    idx = next(i for i, other in enumerate(day_items) if other.id == item.id)
    prev_place = day_items[idx - 1].place_id if idx > 0 else itinerary.hotel_id
    next_place = day_items[idx + 1].place_id if idx + 1 < len(day_items) else itinerary.hotel_id
    for a, b in ((prev_place, item.place_id), (item.place_id, next_place)):
        if a and b:
            key = resolve_travel_key(catalog, a, b)
            if key:
                deps.append(key)
    return list(dict.fromkeys(deps))


def dependency_graph(itinerary: Itinerary, catalog: Catalog) -> dict[str, set[str]]:
    """fact key -> ids of the items that currently depend on it."""
    graph: dict[str, set[str]] = {}
    for item in itinerary.items:
        for key in item_dependencies(itinerary, item, catalog):
            graph.setdefault(key, set()).add(item.id)
    return graph
