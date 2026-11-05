"""Data model for the synthetic trip-planning environment.

The synthetic catalog and scenarios are supplied as JSON that validates against these
models (see docs/DATA_FORMAT.md). Clock times are "HH:MM" strings in JSON and integer
minutes after midnight in Python. Weekdays follow Python's date.weekday(): 0 = Monday.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from enum import Enum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    PrivateAttr,
    WithJsonSchema,
    model_validator,
)

MINUTES_PER_DAY = 24 * 60


def parse_clock(value: object) -> int:
    """Parse "HH:MM" (or an int of minutes) into minutes after midnight. 24:00 is allowed."""
    if isinstance(value, bool):
        raise ValueError("clock time must be 'HH:MM'")
    if isinstance(value, int):
        minutes = value
    elif isinstance(value, str):
        try:
            hours, mins = value.strip().split(":")
            minutes = int(hours) * 60 + int(mins)
            if not 0 <= int(mins) < 60:
                raise ValueError
        except ValueError as exc:
            raise ValueError(f"clock time must be 'HH:MM', got {value!r}") from exc
    else:
        raise ValueError(f"clock time must be 'HH:MM', got {value!r}")
    if not 0 <= minutes <= MINUTES_PER_DAY:
        raise ValueError(f"clock time out of range: {value!r}")
    return minutes


def format_clock(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


Clock = Annotated[
    int,
    BeforeValidator(parse_clock),
    PlainSerializer(format_clock, return_type=str, when_used="json"),
    WithJsonSchema({"type": "string", "pattern": r"^([01]\d|2[0-4]):[0-5]\d$", "examples": ["09:30"]}),
]

PlaceId = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_\-]*$", max_length=64)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MealType(str, Enum):
    breakfast = "breakfast"
    lunch = "lunch"
    dinner = "dinner"


class OpeningHours(_Strict):
    open: Clock
    close: Clock
    closed_weekdays: list[Annotated[int, Field(ge=0, le=6)]] = Field(
        default_factory=list, description="0 = Monday ... 6 = Sunday"
    )

    @model_validator(mode="after")
    def _open_before_close(self) -> OpeningHours:
        if self.open >= self.close:
            raise ValueError("open must be before close (overnight hours are not supported)")
        return self


class _Place(_Strict):
    id: PlaceId
    name: str
    neighborhood: str
    tags: list[str] = Field(default_factory=list)
    lat: float | None = None
    lon: float | None = None


class Attraction(_Place):
    kind: Literal["attraction"] = "attraction"
    hours: OpeningHours
    visit_minutes: int = Field(gt=0)
    price_per_person: float = Field(ge=0)


class Restaurant(_Place):
    kind: Literal["restaurant"] = "restaurant"
    hours: OpeningHours
    meals: list[MealType] = Field(min_length=1)
    meal_minutes: int = Field(default=60, gt=0)
    price_per_person: float = Field(ge=0)


class Hotel(_Place):
    kind: Literal["hotel"] = "hotel"
    price_per_night: float = Field(ge=0, description="Per room, per night")
    max_guests_per_room: int = Field(default=2, ge=1)


Place = Attraction | Restaurant | Hotel


class TravelLeg(_Strict):
    from_id: PlaceId
    to_id: PlaceId
    minutes: int = Field(ge=0)
    cost_per_person: float = Field(default=0.0, ge=0)


class Catalog(_Strict):
    """The trusted environment. Only the tool layer and the evaluator read it directly."""

    version: str = "1"
    city: str = "Boston"
    synthetic: Literal[True] = Field(default=True, description="All data must be synthetic")
    attractions: list[Attraction]
    restaurants: list[Restaurant]
    hotels: list[Hotel]
    travel: list[TravelLeg]
    symmetric_travel: bool = Field(default=True, description="If a->b is missing, use b->a (the same fact, same key)")
    same_place_minutes: int = Field(default=0, ge=0)

    _places: dict[str, Place] = PrivateAttr(default_factory=dict)
    _legs: dict[tuple[str, str], TravelLeg] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _index(self) -> Catalog:
        places: dict[str, Place] = {}
        for place in [*self.attractions, *self.restaurants, *self.hotels]:
            if place.id in places:
                raise ValueError(f"duplicate place id: {place.id}")
            places[place.id] = place
        legs: dict[tuple[str, str], TravelLeg] = {}
        for leg in self.travel:
            for pid in (leg.from_id, leg.to_id):
                if pid not in places:
                    raise ValueError(f"travel leg references unknown place: {pid}")
            key = (leg.from_id, leg.to_id)
            if key in legs:
                raise ValueError(f"duplicate travel leg: {leg.from_id} -> {leg.to_id}")
            legs[key] = leg
        self._places = places
        self._legs = legs
        return self

    def place(self, place_id: str) -> Place | None:
        return self._places.get(place_id)

    @property
    def places(self) -> dict[str, Place]:
        return dict(self._places)

    def leg(self, from_id: str, to_id: str) -> TravelLeg | None:
        """The stored leg used for travel between two places, honoring symmetric_travel."""
        leg = self._legs.get((from_id, to_id))
        if leg is None and self.symmetric_travel:
            leg = self._legs.get((to_id, from_id))
        return leg


class MealWindow(_Strict):
    start: Clock
    end: Clock


def _default_meal_windows() -> dict[MealType, MealWindow]:
    return {
        MealType.breakfast: MealWindow(start=parse_clock("07:00"), end=parse_clock("10:00")),
        MealType.lunch: MealWindow(start=parse_clock("11:30"), end=parse_clock("14:30")),
        MealType.dinner: MealWindow(start=parse_clock("17:30"), end=parse_clock("21:30")),
    }


class Constraints(_Strict):
    start_date: date
    num_days: int = Field(ge=1)
    nights: int | None = Field(default=None, ge=0, description="Hotel nights; defaults to num_days")
    group_size: int = Field(ge=1)
    budget: float = Field(gt=0, description="Total budget for the whole group")
    day_start: Clock = parse_clock("09:00")
    day_end: Clock = parse_clock("22:00")
    required_attractions: list[PlaceId] = Field(default_factory=list)
    meals_per_day: list[MealType] = Field(default_factory=lambda: [MealType.lunch, MealType.dinner])
    meal_windows: dict[MealType, MealWindow] = Field(default_factory=_default_meal_windows)
    hotel_neighborhoods: list[str] | None = Field(
        default=None, description="Allowed hotel neighborhoods; null means any"
    )
    max_tool_calls: int | None = Field(default=None, ge=1, description="Overrides the experiment budget")

    @model_validator(mode="after")
    def _fill_meal_windows(self) -> Constraints:
        defaults = _default_meal_windows()
        for meal in MealType:
            self.meal_windows.setdefault(meal, defaults[meal])
        if self.day_start >= self.day_end:
            raise ValueError("day_start must be before day_end")
        return self

    @property
    def hotel_nights(self) -> int:
        return self.num_days if self.nights is None else self.nights

    def weekday(self, day: int) -> int:
        return (self.start_date + timedelta(days=day)).weekday()

    def date_of(self, day: int) -> date:
        return self.start_date + timedelta(days=day)


class ItemKind(str, Enum):
    visit = "visit"
    meal = "meal"


class ItineraryItem(_Strict):
    id: str
    day: int = Field(ge=0)
    place_id: PlaceId
    kind: ItemKind
    meal_type: MealType | None = None
    start: Clock


class Itinerary(_Strict):
    hotel_id: PlaceId | None = None
    items: list[ItineraryItem] = Field(default_factory=list)

    def day_items(self, day: int) -> list[ItineraryItem]:
        return sorted((i for i in self.items if i.day == day), key=lambda i: (i.start, i.id))


class Scenario(_Strict):
    id: PlaceId
    split: Literal["dev", "eval"]
    horizon: int = Field(ge=1, description="Number of required dependent commitments")
    description: str = ""
    constraints: Constraints
    reference_itinerary: Itinerary = Field(description="Hidden valid itinerary proving feasibility")
    notes: str = ""

    def required_commitments(self) -> int:
        c = self.constraints
        return len(c.required_attractions) + c.num_days * len(c.meals_per_day)


def duration_of(item: ItineraryItem, catalog: Catalog) -> int | None:
    """Item duration from the catalog. Durations are never corrupted."""
    place = catalog.place(item.place_id)
    if isinstance(place, Attraction):
        return place.visit_minutes
    if isinstance(place, Restaurant):
        return place.meal_minutes
    return None


def rooms_needed(group_size: int, hotel: Hotel) -> int:
    return math.ceil(group_size / hotel.max_guests_per_room)
