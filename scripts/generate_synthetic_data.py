"""Generate synthetic trip-planning data (stage 5: the full-size catalog)."""

import itertools
import json
import math
import random
from pathlib import Path

SEED = 5130   # change this and every random choice changes; keep it and the data is identical

# ---- 1. The world: places we invent by hand -------------------------------
# Neighborhood centres in km (x = east, y = north), roughly Boston's layout.
NEIGHBORHOODS = {
    "Downtown": (0.0, 0.0),
    "North End": (0.3, 1.0),
    "Beacon Hill": (-0.8, 0.5),
    "Back Bay": (-2.0, -0.6),
    "Fenway": (-3.2, -1.0),
    "South End": (-1.3, -1.6),
    "Seaport": (1.3, -0.8),
    "Cambridge": (-2.2, 2.0),
}

# Weekdays: 0 = Monday ... 6 = Sunday. The last column lists the days a place is closed.
# Attractions: (id, name, neighborhood, tags, open, close, visit_minutes, price, closed_weekdays)
# Restaurants: (id, name, neighborhood, tags, open, close, meals, meal_minutes, price, closed_weekdays)
# Hotels:      (id, name, neighborhood, tags, price_per_night, max_guests_per_room)
ATTRACTIONS = [
    # Downtown
    ("a_lantern_hall", "Old Lantern Meeting Hall", "Downtown", ["history"], "09:30", "17:00", 60, 12, []),
    ("a_harbor_aquarium", "Harborlight Aquarium", "Downtown", ["family", "animals"], "09:00", "18:00", 120, 32, []),
    ("a_freedom_steps", "Freedom Steps Walking Trail", "Downtown", ["outdoor", "history"], "08:00", "20:00", 90, 0, []),
    ("a_mint_vault", "Colonial Mint Vault Tour", "Downtown", ["history", "tour"], "10:00", "13:00", 75, 18, [0]),
    ("a_city_carillon", "City Carillon Tower", "Downtown", ["view", "music"], "11:00", "15:00", 120, 15, []),
    # North End
    ("a_copper_steeple", "Copper Steeple Church", "North End", ["history", "architecture"], "09:00", "17:00", 45, 5, [6]),
    ("a_northpoint_house", "Northpoint Colonial House", "North End", ["museum", "history"], "10:00", "16:00", 60, 8, [0]),
    ("a_hillside_burial", "Hillside Burying Ground", "North End", ["outdoor", "history"], "09:00", "17:00", 45, 0, []),
    ("a_bakery_tour", "Hanover Lane Bakery Tour", "North End", ["food", "tour"], "13:00", "15:30", 90, 28, [1]),
    ("a_wharf_gardens", "Long Wharf Gardens", "North End", ["outdoor", "park"], "07:00", "21:00", 60, 0, []),
    # Beacon Hill
    ("a_golden_dome", "Golden Dome Statehouse Tour", "Beacon Hill", ["history", "tour"], "10:00", "15:30", 60, 0, [5, 6]),
    ("a_gaslight_walk", "Gaslight Lane Architecture Walk", "Beacon Hill", ["outdoor", "architecture"], "09:00", "19:00", 75, 10, []),
    ("a_hilltop_athenaeum", "Hilltop Athenaeum Library", "Beacon Hill", ["library", "history"], "10:00", "16:00", 60, 10, [6]),
    ("a_puppet_theater", "Acorn Street Puppet Theater", "Beacon Hill", ["family", "theater"], "14:00", "17:00", 90, 20, [0, 1]),
    ("a_common_gardens", "Public Common Gardens", "Beacon Hill", ["outdoor", "park"], "06:00", "22:00", 60, 0, []),
    # Back Bay
    ("a_court_library", "Copley Court Library", "Back Bay", ["library", "architecture"], "09:00", "20:00", 60, 0, [6]),
    ("a_skyview_deck", "Skyview Observation Deck", "Back Bay", ["view"], "10:00", "21:00", 75, 29, []),
    ("a_glass_church", "Stained Glass Chapel", "Back Bay", ["architecture", "history"], "10:00", "16:30", 45, 10, [0]),
    ("a_crystal_globe", "Crystal Globe Map Room", "Back Bay", ["museum"], "10:00", "12:30", 60, 9, [1]),
    ("a_swan_boats", "Lagoon Swan Boats", "Back Bay", ["outdoor", "family", "water"], "10:00", "16:00", 45, 6, []),
    # Fenway
    ("a_fine_arts", "Fenway Museum of Fine Arts", "Fenway", ["art", "museum"], "10:00", "17:00", 150, 27, [1]),
    ("a_venetian_court", "Venetian Courtyard Museum", "Fenway", ["art", "museum"], "11:00", "15:00", 120, 22, [1]),
    ("a_ballpark_tour", "Green Wall Ballpark Tour", "Fenway", ["sports", "tour"], "09:00", "12:00", 75, 25, []),
    ("a_fens_garden", "Back Fens Victory Garden", "Fenway", ["outdoor", "park"], "08:00", "19:00", 45, 0, []),
    ("a_symphony_tour", "Symphony Hall Backstage Tour", "Fenway", ["music", "tour"], "12:00", "15:00", 90, 15, [5, 6]),
    # South End
    ("a_brickyard_market", "Brick Yard Open Market", "South End", ["market", "food"], "10:00", "16:00", 60, 0, [0, 1]),
    ("a_ink_gallery", "Ink Block Gallery", "South End", ["art"], "11:00", "18:00", 60, 8, [0]),
    ("a_cocoa_works", "Cocoa Works Chocolate Lab", "South End", ["food", "tour"], "14:00", "17:00", 120, 35, [2]),
    ("a_union_squares", "Union Park Squares Walk", "South End", ["outdoor", "architecture"], "08:00", "20:00", 45, 0, []),
    ("a_jazz_matinee", "Tremont Jazz Cellar Matinee", "South End", ["music"], "15:00", "19:00", 120, 30, [0]),
    # Seaport
    ("a_tea_ship", "Tea Crate Ship Museum", "Seaport", ["history", "museum"], "10:00", "17:00", 75, 32, []),
    ("a_harborfront_art", "Harborfront Contemporary Art", "Seaport", ["art", "museum"], "10:00", "17:00", 90, 20, [0]),
    ("a_discovery_museum", "Seaport Children's Discovery Museum", "Seaport", ["family", "museum"], "09:00", "16:30", 120, 20, []),
    ("a_harborwalk", "Fan Pier Harborwalk", "Seaport", ["outdoor", "water"], "07:00", "21:00", 45, 0, []),
    ("a_harbor_cruise", "Inner Harbor Cruise", "Seaport", ["tour", "water"], "11:00", "14:00", 90, 36, []),
    # Cambridge
    ("a_science_dome", "Riverside Science Dome", "Cambridge", ["science", "museum"], "09:00", "17:00", 120, 29, []),
    ("a_innovation_museum", "Innovation Technology Museum", "Cambridge", ["science", "museum"], "10:00", "17:00", 90, 15, [0]),
    ("a_college_yard", "Old College Yard Tour", "Cambridge", ["history", "tour", "outdoor"], "10:00", "15:00", 60, 0, [6]),
    ("a_glass_botanicals", "Glass Botanical Collection", "Cambridge", ["museum", "science"], "09:00", "12:00", 90, 15, []),
    ("a_river_kayak", "Charles River Kayak Rental", "Cambridge", ["outdoor", "water"], "10:00", "19:00", 90, 25, []),
]

# (id, name, neighborhood, tags, open, close, meals, meal_minutes, price, closed_weekdays)
RESTAURANTS = [
    ("r_quill_cafe", "Quill and Kettle Cafe", "Downtown", ["cafe"], "07:00", "15:00", ["breakfast", "lunch"], 45, 16, []),
    ("r_market_chowder", "Marketplace Chowder Hall", "Downtown", ["seafood"], "11:00", "22:00", ["lunch", "dinner"], 60, 34, []),
    ("r_vault_steak", "The Vault Steakhouse", "Downtown", ["steak", "fine dining"], "17:00", "22:30", ["dinner"], 90, 72, [6]),
    ("r_nonna_lucia", "Nonna Lucia Trattoria", "North End", ["italian"], "11:30", "22:00", ["lunch", "dinner"], 60, 38, []),
    ("r_prince_pastry", "Prince Street Pasticceria", "North End", ["bakery", "cafe"], "07:00", "16:00", ["breakfast", "lunch"], 45, 14, []),
    ("r_lantern_osteria", "Lantern Osteria", "North End", ["italian", "fine dining"], "17:00", "23:00", ["dinner"], 90, 64, [0]),
    ("r_acorn_cafe", "Acorn Corner Cafe", "Beacon Hill", ["cafe"], "07:00", "14:30", ["breakfast", "lunch"], 45, 15, [6]),
    ("r_charles_bistro", "Charles Street Bistro", "Beacon Hill", ["french"], "11:30", "21:30", ["lunch", "dinner"], 60, 41, []),
    ("r_gaslight_tavern", "Gaslight Tavern", "Beacon Hill", ["pub"], "12:00", "23:00", ["lunch", "dinner"], 60, 30, [0]),
    ("r_newbury_greens", "Newbury Greens", "Back Bay", ["salad", "vegetarian"], "11:00", "16:00", ["lunch"], 45, 19, []),
    ("r_boylston_brasserie", "Boylston Brasserie", "Back Bay", ["french", "fine dining"], "11:30", "22:30", ["lunch", "dinner"], 75, 52, []),
    ("r_copley_diner", "Copley Square Diner", "Back Bay", ["diner"], "07:00", "21:00", ["breakfast", "lunch", "dinner"], 45, 21, []),
    ("r_kenmore_grill", "Kenmore Grill", "Fenway", ["american", "burgers"], "11:00", "23:00", ["lunch", "dinner"], 60, 27, []),
    ("r_fens_noodle", "Fens Noodle Bar", "Fenway", ["asian", "noodles"], "11:30", "21:00", ["lunch", "dinner"], 45, 18, [1]),
    ("r_gallery_cafe", "Gallery Garden Cafe", "Fenway", ["cafe"], "11:00", "15:30", ["lunch"], 45, 22, [1]),
    ("r_tremont_table", "Tremont Table", "South End", ["american", "fine dining"], "17:30", "22:30", ["dinner"], 90, 58, [0]),
    ("r_brick_oven", "Brick Oven Bakery", "South End", ["bakery", "cafe"], "07:00", "15:00", ["breakfast", "lunch"], 45, 13, []),
    ("r_union_cantina", "Union Cantina", "South End", ["mexican"], "11:30", "22:00", ["lunch", "dinner"], 60, 29, []),
    ("r_pier_chowder", "Pier Six Chowder House", "Seaport", ["seafood"], "11:30", "21:30", ["lunch", "dinner"], 60, 39, []),
    ("r_dockside_tacos", "Dockside Taco Stand", "Seaport", ["mexican", "quick"], "11:00", "20:00", ["lunch", "dinner"], 45, 16, [0]),
    ("r_lighthouse_oyster", "Lighthouse Oyster Bar", "Seaport", ["seafood", "fine dining"], "17:00", "22:30", ["dinner"], 90, 68, []),
    ("r_square_deli", "Square Deli and Bagels", "Cambridge", ["deli", "cafe"], "07:00", "15:00", ["breakfast", "lunch"], 45, 14, []),
    ("r_riverbend_curry", "Riverbend Curry House", "Cambridge", ["indian"], "11:30", "22:00", ["lunch", "dinner"], 60, 24, []),
    ("r_kendall_kitchen", "Kendall Test Kitchen", "Cambridge", ["fusion"], "11:00", "21:00", ["lunch", "dinner"], 60, 33, [5, 6]),
]

# (id, name, neighborhood, tags, price_per_night, max_guests_per_room)
HOTELS = [
    ("h_harborview_inn", "Harborview Inn", "Downtown", ["waterfront"], 289, 2),
    ("h_northgate", "Northgate Guesthouse", "North End", ["boutique"], 215, 2),
    ("h_beacon_rowhouse", "Beacon Rowhouse Hotel", "Beacon Hill", ["historic"], 305, 2),
    ("h_copley_grand", "Copley Grand Hotel", "Back Bay", ["luxury"], 349, 2),
    ("h_fenway_lodge", "Fenway Lodge", "Fenway", ["budget"], 172, 2),
    ("h_southend_suites", "South End Family Suites", "South End", ["family", "suites"], 259, 4),
    ("h_seaport_tower", "Seaport Tower Suites", "Seaport", ["modern", "suites"], 329, 4),
    ("h_riverbank", "Riverbank Hotel", "Cambridge", ["riverside"], 189, 2),
]



# ---- 2. Travel times: worked out from distance, not typed by hand ----------
def travel_minutes(a, b):
    km = math.dist((a["x"], a["y"]), (b["x"], b["y"])) * 1.3   # streets are not straight lines
    if km <= 1.2:
        return max(5, round(km * 13) + 3), 0.0                 # walk: about 13 min per km, free
    minutes = round(8 + km * 4.5)                              # transit: wait + ride
    if (a["neighborhood"] == "Cambridge") != (b["neighborhood"] == "Cambridge"):
        minutes += 6                                           # crossing the river
    return minutes, 2.40


def build_catalog(rng):
    places = []   # every place with its position, for the travel step
    attractions, restaurants, hotels = [], [], []

    def position(hood):   # a random spot near the neighborhood centre
        cx, cy = NEIGHBORHOODS[hood]
        return cx + rng.uniform(-0.35, 0.35), cy + rng.uniform(-0.35, 0.35)

    for pid, name, hood, tags, op, cl, visit, price, closed in ATTRACTIONS:
        attractions.append({
            "id": pid, "name": name, "neighborhood": hood, "tags": tags, "kind": "attraction",
            "hours": {"open": op, "close": cl, "closed_weekdays": closed},
            "visit_minutes": visit, "price_per_person": float(price),
        })
        x, y = position(hood)
        places.append({"id": pid, "neighborhood": hood, "x": x, "y": y})

    for pid, name, hood, tags, op, cl, meals, minutes, price, closed in RESTAURANTS:
        restaurants.append({
            "id": pid, "name": name, "neighborhood": hood, "tags": tags, "kind": "restaurant",
            "hours": {"open": op, "close": cl, "closed_weekdays": closed},
            "meals": meals, "meal_minutes": minutes, "price_per_person": float(price),
        })
        x, y = position(hood)
        places.append({"id": pid, "neighborhood": hood, "x": x, "y": y})

    for pid, name, hood, tags, price, guests in HOTELS:
        hotels.append({
            "id": pid, "name": name, "neighborhood": hood, "tags": tags, "kind": "hotel",
            "price_per_night": float(price), "max_guests_per_room": guests,
        })
        x, y = position(hood)
        places.append({"id": pid, "neighborhood": hood, "x": x, "y": y})

    travel = []
    for a, b in itertools.combinations(places, 2):   # every pair, once
        minutes, cost = travel_minutes(a, b)
        travel.append({"from_id": a["id"], "to_id": b["id"], "minutes": minutes, "cost_per_person": cost})

    catalog = {
        "version": "1", "city": "Boston", "synthetic": True,
        "symmetric_travel": True, "same_place_minutes": 0,
        "attractions": attractions, "restaurants": restaurants, "hotels": hotels,
        "travel": travel,
    }
    positions = {p["id"]: (p["x"], p["y"]) for p in places}
    return catalog, positions


# ---- Lookup tables: built once, so the search does not re-read the catalog -
def make_index(catalog, positions):
    places = {p["id"]: p for p in catalog["attractions"] + catalog["restaurants"] + catalog["hotels"]}
    legs = {}
    for leg in catalog["travel"]:
        a, b = leg["from_id"], leg["to_id"]
        legs[(a, b)] = legs[(b, a)] = (leg["minutes"], leg["cost_per_person"])
    for pid in places:
        legs[(pid, pid)] = (0, 0.0)             # staying at the same place: no travel
    return {"places": places, "legs": legs, "positions": positions}


# ---- The scheduler: work out start times for a chosen order -----------------
DAY_START, DAY_END = "09:00", "22:00"
MEAL_WINDOWS = {"lunch": ("11:30", "14:30"), "dinner": ("17:30", "21:30")}


def to_min(text):            # "09:30" -> 570 minutes after midnight
    h, m = text.split(":")
    return int(h) * 60 + int(m)


def to_clock(minutes):       # 570 -> "09:30"
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def schedule_day(index, hotel_id, stops, weekday):
    """Give each stop the earliest start time that follows all the rules.

    stops: list of (place_id, meal) where meal is "lunch", "dinner" or None.
    weekday: 0 = Monday ... 6 = Sunday (places can be closed on some days).
    Returns a list of (place_id, meal, start, end), or None if impossible.
    """
    places, legs = index["places"], index["legs"]
    now, here, plan = to_min(DAY_START), hotel_id, []
    for place_id, meal in stops:
        p = places[place_id]
        if weekday in p["hours"]["closed_weekdays"]:
            return None                                        # closed that day
        start = max(now + legs[(here, place_id)][0], to_min(p["hours"]["open"]))
        if meal:
            start = max(start, to_min(MEAL_WINDOWS[meal][0]))  # not before the meal window
        start = -(-start // 5) * 5                             # round up to 5 minutes
        if meal and start > to_min(MEAL_WINDOWS[meal][1]):
            return None                                        # meal starts too late
        end = start + (p.get("visit_minutes") or p["meal_minutes"])
        if end > to_min(p["hours"]["close"]):
            return None                                        # still there after closing
        plan.append((place_id, meal, start, end))
        now, here = end, place_id
    if now + legs[(here, hotel_id)][0] > to_min(DAY_END):
        return None                                            # cannot get back in time
    return plan


# ---- The cost: what one day costs the group (the hotel is added separately) -
def day_cost(index, hotel_id, plan, group_size):
    """Tickets + meals + transit fares for one day."""
    places, legs = index["places"], index["legs"]
    total = sum(places[place_id]["price_per_person"] for place_id, _, _, _ in plan) * group_size
    route = [hotel_id] + [place_id for place_id, _, _, _ in plan] + [hotel_id]
    total += sum(legs[(a, b)][1] for a, b in zip(route, route[1:])) * group_size
    return total


def hotel_cost(index, hotel_id, group_size, nights):
    hotel = index["places"][hotel_id]
    rooms = math.ceil(group_size / hotel["max_guests_per_room"])
    return hotel["price_per_night"] * rooms * nights


# ---- The search: try every order and restaurant, keep the cheapest ---------
def find_day_plan(index, hotel_id, attractions, weekday, group_size):
    """Try every order of the attractions, every place for lunch and dinner, and the
    8 nearest restaurants for each meal. Return (cost, plan) for the cheapest plan
    that works, or None if nothing works."""
    places, positions = index["places"], index["positions"]
    cx = sum(positions[a][0] for a in attractions) / len(attractions)   # middle of the day's attractions
    cy = sum(positions[a][1] for a in attractions) / len(attractions)

    def nearest(meal):   # 24 restaurants x 24 would be slow; the 8 nearest are plenty
        options = [p for p in places.values() if p["kind"] == "restaurant" and meal in p["meals"]]
        options.sort(key=lambda r: math.dist(positions[r["id"]], (cx, cy)))
        return options[:8]

    slots = len(attractions) + 2                   # 3 attractions + lunch + dinner = 5 slots
    best = None
    for order in itertools.permutations(attractions):                        # 6 orders
        for lunch_at, dinner_at in itertools.combinations(range(slots), 2):  # 10 ways
            for lunch in nearest("lunch"):
                for dinner in nearest("dinner"):
                    stops, visits = [], iter(order)
                    for slot in range(slots):
                        if slot == lunch_at:
                            stops.append((lunch["id"], "lunch"))
                        elif slot == dinner_at:
                            stops.append((dinner["id"], "dinner"))
                        else:
                            stops.append((next(visits), None))
                    plan = schedule_day(index, hotel_id, stops, weekday)
                    if plan is None:
                        continue                   # this combination breaks a rule
                    cost = day_cost(index, hotel_id, plan, group_size)
                    if best is None or cost < best[0]:
                        best = (cost, plan)
    return best


# ---- 4. One scenario (still hand-picked; step 12 makes many at random) -----
def build_scenario(index):
    hotel = "h_copley_grand"
    required = ["a_crystal_globe", "a_harbor_aquarium", "a_freedom_steps"]
    weekday = 0                                # 2026-06-01 is a Monday
    result = find_day_plan(index, hotel, required, weekday, group_size=2)
    if result is None:
        raise SystemExit("No plan works for these attractions")
    cost, plan = result
    cost = round(cost + hotel_cost(index, hotel, group_size=2, nights=1), 2)

    rng = random.Random(1)                     # fixed seed: same budget every run
    slack = rng.uniform(0.08, 0.15)            # leave 8-15% spare money
    budget = math.ceil(cost / (1 - slack) / 5) * 5   # round up to the next $5
    print(f"  plan cost ${cost}, budget ${budget}")

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
            "budget": float(budget),
            "required_attractions": required,
            "meals_per_day": ["lunch", "dinner"],
        },
        "reference_itinerary": {"hotel_id": hotel, "items": items},
    }


# ---- 5. Write the files -----------------------------------------------------
def main():
    out = Path("data")
    (out / "scenarios").mkdir(parents=True, exist_ok=True)

    catalog, positions = build_catalog(random.Random(SEED))
    (out / "catalog.json").write_text(json.dumps(catalog, indent=2))
    index = make_index(catalog, positions)

    scenario = build_scenario(index)
    (out / "scenarios" / f"{scenario['id']}.json").write_text(json.dumps(scenario, indent=2))

    print(f"Wrote {len(index['places'])} places, {len(catalog['travel'])} travel legs and 1 scenario to {out}/")


if __name__ == "__main__":
    main()