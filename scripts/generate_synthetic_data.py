"""Generate synthetic trip-planning data (stage 1: a tiny hand-made world)."""

import itertools
import json
import math
from pathlib import Path

# ---- 1. The world: places we invent by hand -------------------------------
# x, y = position in km on a made-up map of Boston (x = east, y = north)

ATTRACTIONS = [
    # id,                name,                          neighborhood, x,    y,    open,    close,   visit, price
    ("a_harbor_aquarium", "Harborlight Aquarium",       "Downtown",   0.1,  0.0, "09:00", "18:00", 120,   32),
    ("a_freedom_steps",   "Freedom Steps Walking Trail", "Downtown",  -0.2,  0.2, "08:00", "20:00",  90,    0),
    ("a_crystal_globe",   "Crystal Globe Map Room",     "Back Bay",  -2.0, -0.6, "10:00", "12:30",  60,    9),
]

RESTAURANTS = [
    # id,                 name,                       neighborhood, x,    y,    open,    close,   meals,               minutes, price
    ("r_quill_cafe",      "Quill and Kettle Cafe",    "Downtown",   0.0,  0.1, "07:00", "15:00", ["breakfast", "lunch"], 45,     16),
    ("r_market_chowder",  "Marketplace Chowder Hall", "Downtown",   0.2, -0.1, "11:00", "22:00", ["lunch", "dinner"],    60,     34),
]

HOTELS = [
    # id,              name,                 neighborhood, x,    y,    price_per_night, guests_per_room
    ("h_copley_grand", "Copley Grand Hotel", "Back Bay",  -1.9, -0.7, 349,             2),
]


# ---- 2. Travel times: worked out from distance, not typed by hand ----------
def travel_minutes(a, b):
    km = math.dist((a["x"], a["y"]), (b["x"], b["y"])) * 1.3   # streets are not straight lines
    if km <= 1.2:
        return max(5, round(km * 13) + 3), 0.0                 # walk: about 13 min per km, free
    return round(8 + km * 4.5), 2.40                           # transit: wait + ride, $2.40


def build_catalog():
    places = []   # every place with its coordinates, for the travel step
    attractions, restaurants, hotels = [], [], []

    for pid, name, hood, x, y, op, cl, visit, price in ATTRACTIONS:
        attractions.append({
            "id": pid, "name": name, "neighborhood": hood, "kind": "attraction",
            "hours": {"open": op, "close": cl, "closed_weekdays": []},
            "visit_minutes": visit, "price_per_person": float(price),
        })
        places.append({"id": pid, "x": x, "y": y})

    for pid, name, hood, x, y, op, cl, meals, minutes, price in RESTAURANTS:
        restaurants.append({
            "id": pid, "name": name, "neighborhood": hood, "kind": "restaurant",
            "hours": {"open": op, "close": cl, "closed_weekdays": []},
            "meals": meals, "meal_minutes": minutes, "price_per_person": float(price),
        })
        places.append({"id": pid, "x": x, "y": y})

    for pid, name, hood, x, y, price, guests in HOTELS:
        hotels.append({
            "id": pid, "name": name, "neighborhood": hood, "kind": "hotel",
            "price_per_night": float(price), "max_guests_per_room": guests,
        })
        places.append({"id": pid, "x": x, "y": y})

    travel = []
    for a, b in itertools.combinations(places, 2):   # every pair, once
        minutes, cost = travel_minutes(a, b)
        travel.append({"from_id": a["id"], "to_id": b["id"], "minutes": minutes, "cost_per_person": cost})

    return {
        "version": "1", "city": "Boston", "synthetic": True,
        "symmetric_travel": True, "same_place_minutes": 0,
        "attractions": attractions, "restaurants": restaurants, "hotels": hotels,
        "travel": travel,
    }

# ---- The scheduler: work out start times for a chosen order -----------------
DAY_START, DAY_END = "09:00", "22:00"
MEAL_WINDOWS = {"lunch": ("11:30", "14:30"), "dinner": ("17:30", "21:30")}


def to_min(text):            # "09:30" -> 570 minutes after midnight
    h, m = text.split(":")
    return int(h) * 60 + int(m)


def to_clock(minutes):       # 570 -> "09:30"
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def schedule_day(catalog, hotel_id, stops):
    """Give each stop the earliest start time that follows all the rules.

    stops: list of (place_id, meal) where meal is "lunch", "dinner" or None.
    Returns a list of (place_id, meal, start, end), or None if impossible.
    """
    places = {p["id"]: p for p in catalog["attractions"] + catalog["restaurants"]}
    legs = {}
    for leg in catalog["travel"]:
        legs[(leg["from_id"], leg["to_id"])] = legs[(leg["to_id"], leg["from_id"])] = leg["minutes"]

    now, here, plan = to_min(DAY_START), hotel_id, []
    for place_id, meal in stops:
        p = places[place_id]
        arrive = now + (0 if here == place_id else legs[(here, place_id)])
        start = max(arrive, to_min(p["hours"]["open"]))       # not before it opens
        if meal:
            start = max(start, to_min(MEAL_WINDOWS[meal][0]))  # not before the meal window
        start = -(-start // 5) * 5                             # round up to 5 minutes
        if meal and start > to_min(MEAL_WINDOWS[meal][1]):
            return None                                        # meal starts too late
        duration = p.get("visit_minutes") or p["meal_minutes"]
        end = start + duration
        if end > to_min(p["hours"]["close"]):
            return None                                        # still there after closing
        plan.append((place_id, meal, start, end))
        now, here = end, place_id
    if now + legs[(here, hotel_id)] > to_min(DAY_END):
        return None                                            # cannot get back in time
    return plan

# ---- 3. One scenario, with a reference plan written by hand ----------------
def build_scenario(catalog):
    hotel = "h_copley_grand"
    order = [                                  # we choose the order, the code finds the times
        ("a_crystal_globe", None),
        ("r_quill_cafe", "lunch"),
        ("a_harbor_aquarium", None),
        ("a_freedom_steps", None),
        ("r_market_chowder", "dinner"),
    ]
    plan = schedule_day(catalog, hotel, order)
    if plan is None:
        raise SystemExit("That order is impossible")

    items = []
    for n, (place_id, meal, start, end) in enumerate(plan, start=1):
        print(f"  {to_clock(start)}-{to_clock(end)}  {place_id}")
        item = {"id": f"ref{n}", "day": 0, "place_id": place_id, "start": to_clock(start)}
        item["kind"] = "meal" if meal else "visit"
        if meal:
            item["meal_type"] = meal
        items.append(item)

    return {
        "id": "dev-h05-001",
        "split": "dev",
        "horizon": 5,
        "description": "One day, three required attractions, lunch and dinner.",
        "constraints": {
            "start_date": "2026-06-01",
            "num_days": 1,
            "group_size": 2,
            "budget": 600.0,
            "required_attractions": ["a_crystal_globe", "a_harbor_aquarium", "a_freedom_steps"],
            "meals_per_day": ["lunch", "dinner"],
        },
        "reference_itinerary": {"hotel_id": hotel, "items": items},
    }




# ---- 4. Write the files -----------------------------------------------------
def main():
    out = Path("data")
    (out / "scenarios").mkdir(parents=True, exist_ok=True)

    catalog = build_catalog()
    (out / "catalog.json").write_text(json.dumps(catalog, indent=2))

    scenario = build_scenario(catalog)
    (out / "scenarios" / f"{scenario['id']}.json").write_text(json.dumps(scenario, indent=2))

    print(f"Wrote {len(catalog['travel'])} travel legs and 1 scenario to {out}/")


if __name__ == "__main__":
    main()