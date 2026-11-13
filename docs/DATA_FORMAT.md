# Synthetic data format

The agent runs on a synthetic Boston-like catalog and a set of scenarios. This page is the contract for whoever builds that data. A complete, valid example lives in `examples/tiny/` (regenerate it with `trip-agent example-data examples/tiny`). JSON Schemas are in `docs/schemas/`.

All data must be synthetic. Use invented names, not real businesses.

## Folder layout

```
data/
  catalog.json            one catalog: every attraction, restaurant, hotel and travel leg
  scenarios/
    dev-h05-001.json      one scenario per file (any file names)
    eval-h30-010.json
```

A single `scenarios.json` containing a list also works.

Check a dataset before using it:

```bash
trip-agent validate data/
```

It checks the schema, every id reference, that each reference itinerary is feasible under the true facts, that horizons match, and that travel data covers every pair of places. Then upload it:

```bash
trip-agent import-data data/ --backend supabase --dataset main
```

## Conventions

- **Ids:** lowercase letters, digits, `_` and `-`, starting with a letter or digit. Ids are unique across the whole catalog. Prefixes like `a_`, `r_`, `h_` help but are not required.
- **Times:** `"HH:MM"`, 24-hour. `"24:00"` means midnight at the end of the day. Overnight opening hours are not supported.
- **Weekdays:** integers, `0` = Monday ... `6` = Sunday.
- **Money:** plain numbers in one currency.

## catalog.json

```json
{
  "version": "1",
  "city": "Boston",
  "synthetic": true,
  "symmetric_travel": true,
  "same_place_minutes": 0,
  "attractions": [ ... ],
  "restaurants": [ ... ],
  "hotels": [ ... ],
  "travel": [ ... ]
}
```

### Attraction

| Field | Type | Notes |
| --- | --- | --- |
| `id`, `name`, `neighborhood` | string | |
| `tags` | list of strings | Optional. Search matches on these words. |
| `hours` | `{"open": "10:00", "close": "17:00", "closed_weekdays": [1]}` | A visit must start at or after `open` and end at or before `close`. |
| `visit_minutes` | integer > 0 | Fixed visit duration. |
| `price_per_person` | number >= 0 | |
| `lat`, `lon` | number | Optional, unused by the checker. |

### Restaurant

Same as an attraction, except `visit_minutes` is replaced by:

| Field | Type | Notes |
| --- | --- | --- |
| `meals` | list of `"breakfast"`, `"lunch"`, `"dinner"` | Which meals it serves. |
| `meal_minutes` | integer > 0 | Default 60. |

### Hotel

| Field | Type | Notes |
| --- | --- | --- |
| `id`, `name`, `neighborhood`, `tags` | | As above. |
| `price_per_night` | number | Per room. |
| `max_guests_per_room` | integer | Default 2. Rooms needed = ceil(group size / this). |

### Travel leg

```json
{"from_id": "h_copley", "to_id": "a_art_museum", "minutes": 20, "cost_per_person": 0}
```

With `"symmetric_travel": true`, one leg per pair covers both directions. Every pair of places the agent might visit in a row needs a leg (including hotel to place). The simplest approach is one leg for every pair. `validate` warns about missing pairs.

## Which facts can be corrupted

Only these values can be altered in search results in the corrupted condition, and only these can be checked with `verify_fact`:

| Fact | Key |
| --- | --- |
| Opening time | `<place_id>.open_time` |
| Closing time | `<place_id>.close_time` |
| Attraction or restaurant price | `<place_id>.price_per_person` |
| Hotel price | `<hotel_id>.price_per_night` |
| Travel time | `travel:<from_id>:<to_id>` (the stored direction of the leg) |

Everything else (names, durations, meals served, neighborhoods, closed weekdays, room capacity, travel cost) is always shown truthfully. Free places (price 0) and legs under 4 minutes are never corrupted, because the perturbation could not make them look better.

You do not write corruptions yourself. They are generated from the true catalog with a fixed seed.

## Scenario file

```json
{
  "id": "dev-h05-001",
  "split": "dev",
  "horizon": 5,
  "description": "One day, three required attractions, lunch and dinner.",
  "constraints": {
    "start_date": "2026-06-01",
    "num_days": 1,
    "nights": null,
    "group_size": 2,
    "budget": 500,
    "day_start": "09:00",
    "day_end": "22:00",
    "required_attractions": ["a_liberty_walk", "a_aquarium", "a_art_museum"],
    "meals_per_day": ["lunch", "dinner"],
    "meal_windows": {"lunch": {"start": "11:30", "end": "14:30"}},
    "hotel_neighborhoods": null,
    "max_tool_calls": null
  },
  "reference_itinerary": {
    "hotel_id": "h_copley",
    "items": [
      {"id": "ref1", "day": 0, "place_id": "a_liberty_walk", "kind": "visit", "start": "09:30"},
      {"id": "ref2", "day": 0, "place_id": "r_hanover", "kind": "meal", "meal_type": "lunch", "start": "11:30"}
    ]
  }
}
```

| Field | Notes |
| --- | --- |
| `split` | `"dev"` for tuning, `"eval"` for the final runs. Keep them separate. |
| `horizon` | The number of required dependent commitments. It should equal the number of items in the reference itinerary; `validate` warns if not. |
| `start_date`, `num_days` | Days are numbered 0 to `num_days - 1`. Weekdays come from the dates, which matters for `closed_weekdays`. |
| `nights` | Hotel nights. `null` means `num_days`. |
| `budget` | Total for the whole group: hotel (price x rooms x nights) + attractions and meals (price x group size) + travel cost. |
| `day_start`, `day_end` | Each day starts and ends at the hotel. Travel from the hotel to the first item and back from the last item must fit. |
| `meals_per_day` | Every day needs one of each, starting inside its window. Windows you leave out use the defaults: breakfast 07:00-10:00, lunch 11:30-14:30, dinner 17:30-21:30. |
| `hotel_neighborhoods` | Allowed hotel neighborhoods, or `null` for any. |
| `max_tool_calls` | Overrides the experiment's budget formula for this scenario. Usually `null`. |
| `reference_itinerary` | A hidden valid solution. It proves the scenario is feasible and is never shown to the agent. The agent's answer does not need to match it. |

A horizon of H is usually `len(required_attractions) + num_days x len(meals_per_day)`. For example, H=10 could be 2 days, 6 required attractions, and lunch and dinner each day.

## What the checker enforces

The same rules apply to the agent's check and to the final evaluation:

1. A hotel is chosen, exists, and is in an allowed neighborhood.
2. Each item is a real place of the right kind, on a day of the trip; restaurants serve the chosen meal.
3. No attraction is visited twice.
4. Each item fits inside opening hours and is not on a closed weekday.
5. Each item fits inside the day (`day_start` to `day_end`).
6. Each required meal happens every day, at a restaurant that serves it, starting inside its window.
7. Every required attraction is visited.
8. Items on the same day do not overlap, and the gap between consecutive stops (hotel included) covers the travel time.
9. The total cost is within the budget.

## Keep the horizon groups comparable

The proposal asks for similar constraint density and slack at every horizon. Practical targets:

- Budget slack (budget minus reference cost, as a share of the budget) in a similar range for every horizon, for example 8-15%. `validate` prints it per scenario.
- A similar share of tight items, such as visits ending within 30 minutes of closing, at every horizon.
- A catalog large enough that the agent has real choices, for example 20 or more attractions, 15 or more restaurants and 5 or more hotels across 5 or 6 neighborhoods.
