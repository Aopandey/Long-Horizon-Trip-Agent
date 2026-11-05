"""A tiny hand-made example dataset.

It exists only to show the data format and to smoke-test the pipeline. It is NOT experiment
data: the team supplies the real synthetic catalog and scenarios. All names are invented.
"""

from __future__ import annotations

from datetime import date
from itertools import combinations

from trip_agent.env.models import (
    Attraction,
    Catalog,
    Constraints,
    Hotel,
    Itinerary,
    ItineraryItem,
    MealType,
    OpeningHours,
    Restaurant,
    Scenario,
    TravelLeg,
    parse_clock,
)

# Symmetric neighborhood-to-neighborhood travel minutes; 5 minutes within a neighborhood.
_NEIGHBORHOOD_MINUTES = {
    frozenset({"Back Bay", "Downtown"}): 15,
    frozenset({"Back Bay", "Seaport"}): 20,
    frozenset({"Back Bay", "North End"}): 20,
    frozenset({"Back Bay", "Cambridge"}): 20,
    frozenset({"Downtown", "Seaport"}): 10,
    frozenset({"Downtown", "North End"}): 10,
    frozenset({"Downtown", "Cambridge"}): 20,
    frozenset({"Seaport", "North End"}): 15,
    frozenset({"Seaport", "Cambridge"}): 25,
    frozenset({"North End", "Cambridge"}): 20,
}


def _hours(open_: str, close: str, closed: list[int] | None = None) -> OpeningHours:
    return OpeningHours(open=parse_clock(open_), close=parse_clock(close), closed_weekdays=closed or [])


def example_catalog() -> Catalog:
    attractions = [
        Attraction(
            id="a_art_museum",
            name="Fenway Art Museum",
            neighborhood="Back Bay",
            tags=["art", "museum"],
            hours=_hours("10:00", "17:00", closed=[1]),
            visit_minutes=120,
            price_per_person=25,
        ),
        Attraction(
            id="a_aquarium",
            name="Harbor Aquarium",
            neighborhood="Downtown",
            tags=["family"],
            hours=_hours("09:00", "17:00"),
            visit_minutes=90,
            price_per_person=30,
        ),
        Attraction(
            id="a_liberty_walk",
            name="Liberty Trail Walk",
            neighborhood="Downtown",
            tags=["history", "outdoor"],
            hours=_hours("08:00", "20:00"),
            visit_minutes=90,
            price_per_person=0,
        ),
        Attraction(
            id="a_science",
            name="Riverside Science Center",
            neighborhood="Cambridge",
            tags=["science", "museum"],
            hours=_hours("09:00", "17:00"),
            visit_minutes=120,
            price_per_person=28,
        ),
    ]
    restaurants = [
        Restaurant(
            id="r_hanover",
            name="Hanover Trattoria",
            neighborhood="North End",
            tags=["italian"],
            hours=_hours("11:00", "22:00"),
            meals=[MealType.lunch, MealType.dinner],
            meal_minutes=60,
            price_per_person=35,
        ),
        Restaurant(
            id="r_pier",
            name="Pier Chowder House",
            neighborhood="Seaport",
            tags=["seafood"],
            hours=_hours("11:30", "21:00"),
            meals=[MealType.lunch, MealType.dinner],
            meal_minutes=60,
            price_per_person=30,
        ),
        Restaurant(
            id="r_boylston",
            name="Boylston Bistro",
            neighborhood="Back Bay",
            tags=["cafe"],
            hours=_hours("07:00", "15:00"),
            meals=[MealType.breakfast, MealType.lunch],
            meal_minutes=45,
            price_per_person=20,
        ),
    ]
    hotels = [
        Hotel(
            id="h_copley", name="Copley Row Hotel", neighborhood="Back Bay", price_per_night=220, max_guests_per_room=2
        ),
        Hotel(
            id="h_harbor", name="Seaport Harbor Inn", neighborhood="Seaport", price_per_night=260, max_guests_per_room=2
        ),
    ]
    places = [*attractions, *restaurants, *hotels]
    travel = []
    for a, b in combinations(places, 2):
        same = a.neighborhood == b.neighborhood
        minutes = 5 if same else _NEIGHBORHOOD_MINUTES[frozenset({a.neighborhood, b.neighborhood})]
        travel.append(TravelLeg(from_id=a.id, to_id=b.id, minutes=minutes))
    return Catalog(attractions=attractions, restaurants=restaurants, hotels=hotels, travel=travel)


def _item(i: int, place: str, start: str, meal: MealType | None = None) -> ItineraryItem:
    return ItineraryItem(
        id=f"ref{i}", day=0, place_id=place, kind="meal" if meal else "visit", meal_type=meal, start=parse_clock(start)
    )


def example_scenarios() -> list[Scenario]:
    monday = Scenario(
        id="example-h5-monday",
        split="dev",
        horizon=5,
        description="One day, three required attractions, lunch and dinner (example only).",
        constraints=Constraints(
            start_date=date(2026, 6, 1),  # a Monday
            num_days=1,
            group_size=2,
            budget=500,
            required_attractions=["a_liberty_walk", "a_aquarium", "a_art_museum"],
        ),
        reference_itinerary=Itinerary(
            hotel_id="h_copley",
            items=[
                _item(1, "a_liberty_walk", "09:30"),
                _item(2, "r_hanover", "11:30", MealType.lunch),
                _item(3, "a_aquarium", "13:00"),
                _item(4, "a_art_museum", "15:00"),
                _item(5, "r_pier", "18:00", MealType.dinner),
            ],
        ),
    )
    tuesday = Scenario(
        id="example-h5-tuesday",
        split="eval",
        horizon=5,
        description="The art museum is closed on Tuesdays; hotel must be in Seaport (example only).",
        constraints=Constraints(
            start_date=date(2026, 6, 2),  # a Tuesday
            num_days=1,
            group_size=2,
            budget=560,
            required_attractions=["a_aquarium", "a_liberty_walk", "a_science"],
            hotel_neighborhoods=["Seaport"],
        ),
        reference_itinerary=Itinerary(
            hotel_id="h_harbor",
            items=[
                _item(1, "a_aquarium", "09:30"),
                _item(2, "r_hanover", "11:30", MealType.lunch),
                _item(3, "a_liberty_walk", "13:00"),
                _item(4, "a_science", "15:00"),
                _item(5, "r_pier", "18:00", MealType.dinner),
            ],
        ),
    )
    return [monday, tuesday]
