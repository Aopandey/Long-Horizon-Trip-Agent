"""Text forms of tool results for the model's context.

The tool layer returns structured dicts (kept in full in the run logs). The model sees them as
compact text instead of JSON, in two forms:
- full: everything the agent needs, used for recent turns;
- brief: one line, used once a turn is older than the history window.
Both are the same for every variant.
"""

from __future__ import annotations

import json
from typing import Any

_TRAILERS = ("repeat_notice", "notice", "warning")


def _money(value: Any) -> str:
    try:
        return f"${float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def _place_line(p: dict) -> str:
    parts = [p.get("id", "?"), p.get("name", ""), p.get("neighborhood", "")]
    if p.get("kind") == "hotel":
        parts += [
            f"{_money(p.get('price_per_night'))}/night per room",
            f"max {p.get('max_guests_per_room')} guests/room",
        ]
    else:
        hours = f"open {p.get('open')}-{p.get('close')}"
        if p.get("closed_on"):
            hours += f" (closed {', '.join(d[:3] for d in p['closed_on'])})"
        parts.append(hours)
        if p.get("kind") == "restaurant":
            parts += [",".join(p.get("meals", [])), f"{p.get('meal_minutes')} min"]
        else:
            parts.append(f"{p.get('visit_minutes')} min visit")
        parts.append(f"{_money(p.get('price_per_person'))}/person")
    if p.get("tags"):
        parts.append("tags " + ",".join(p["tags"]))
    return " | ".join(str(x) for x in parts if x != "")


def _item_line(i: dict) -> str:
    meal = f" {i['meal_type']}" if i.get("meal_type") else ""
    return f"{i.get('item_id')} day {i.get('day')} {i.get('start')}-{i.get('end')} {i.get('place_id')}{meal}"


def _conflict_lines(result: dict) -> list[str]:
    if "conflicts" not in result:
        return []
    return ["conflicts: " + (" | ".join(result["conflicts"]) or "none for this item")]


def render(tool: str, result: dict, note: str | None = None) -> str:
    """Full text form of a tool result."""
    lines: list[str] = []
    if "error" in result:
        lines.append(f"error: {result['error']}")
    elif tool == "search_places":
        results = result.get("results", [])
        lines.append(f"{len(results)} of {result.get('total_matches', len(results))} matches:")
        lines += [_place_line(p) for p in results]
        for key in ("note", "more"):
            if result.get(key):
                lines.append(str(result[key]))
    elif tool == "get_travel_time":
        line = f"travel {result.get('from')} -> {result.get('to')}: {result.get('minutes')} min"
        if result.get("fact_key"):
            line += f" (fact {result['fact_key']})"
        if result.get("transit_cost_per_person"):
            line += f", transit {_money(result['transit_cost_per_person'])}/person"
        lines.append(line)
    elif tool == "verify_fact":
        line = f"verified {result.get('fact_key')} = {result.get('verified_value')}"
        if result.get("matches_observed") is False:
            line += f" (search showed {result.get('previously_observed')}: WRONG)"
        elif result.get("matches_observed") is True:
            line += " (matches search)"
        else:
            line += " (not seen in search)"
        lines.append(line)
    elif tool == "set_hotel":
        rooms, nights = result.get("rooms_needed"), result.get("nights")
        lines.append(f"hotel set: {result.get('hotel_id')} ({rooms} room(s) x {nights} night(s))")
    elif tool == "add_item":
        lines.append(f"added {_item_line(result.get('added', {}))}")
        if result.get("depends_on_facts"):
            lines.append("relies on: " + ", ".join(result["depends_on_facts"]))
        lines += _conflict_lines(result)
    elif tool == "move_item":
        lines.append(f"moved {_item_line(result.get('moved', {}))}")
        lines += _conflict_lines(result)
    elif tool == "remove_item":
        lines.append(f"removed {result.get('removed')}")
    elif tool == "check_itinerary":
        status = "VALID" if result.get("valid") else "NOT valid"
        cost = result.get("estimated_total_cost")
        cost_text = _money(cost) if cost is not None else "unknown"
        lines.append(f"check: {status}, {result.get('satisfied')} constraints satisfied, estimated cost {cost_text}")
        for v in result.get("violations", []):
            line = f"- {v.get('check')}: {v.get('message')}"
            if v.get("items"):
                line += f" [{', '.join(v['items'])}]"
            if v.get("unknown_facts"):
                line += f" unknown: {', '.join(v['unknown_facts'])}"
            lines.append(line)
        if result.get("more_violations"):
            lines.append(f"... and {result['more_violations']} more")
    elif tool == "view_itinerary":
        lines.append(f"itinerary, hotel {result.get('hotel_id') or 'not set'}:")
        for day in result.get("days", []):
            items = "; ".join(
                f"{i.get('item_id')} {i.get('start')}-{i.get('end')} {i.get('place_id')}"
                + (f" {i['meal_type']}" if i.get("meal_type") else "")
                for i in day.get("items", [])
            )
            lines.append(f"day {day.get('day')} {day.get('date')} {day.get('weekday')}: {items or 'empty'}")
        if result.get("required_attractions_missing"):
            lines.append("required not scheduled: " + ", ".join(result["required_attractions_missing"]))
    elif tool == "finish":
        lines.append("finished")
    else:
        lines.append(json.dumps({k: v for k, v in result.items() if k != "tool_calls_remaining"}, default=str))
    if result.get("ignored_arguments"):
        lines.append(f"ignored arguments: {', '.join(result['ignored_arguments'])}")
    lines += [str(result[k]) for k in _TRAILERS if result.get(k)]
    if "tool_calls_remaining" in result:
        lines.append(f"calls left: {result['tool_calls_remaining']}")
    if note:
        lines.append(note)
    return "\n".join(lines)


def brief(tool: str, result: dict) -> str:
    """One-line form of a tool result, for turns older than the history window."""
    if "error" in result:
        return f"[old] error: {str(result['error'])[:80]}"
    if tool == "search_places":
        places = []
        for p in result.get("results", []):
            if p.get("kind") == "hotel":
                places.append(f"{p.get('id')} {_money(p.get('price_per_night'))}/night")
            else:
                places.append(f"{p.get('id')} {p.get('open')}-{p.get('close')} {_money(p.get('price_per_person'))}")
        return "[old] search: " + ("; ".join(places) or "no matches")
    if tool == "get_travel_time":
        return f"[old] travel {result.get('from')}->{result.get('to')} {result.get('minutes')} min"
    if tool == "verify_fact":
        wrong = " (search was wrong)" if result.get("matches_observed") is False else ""
        return f"[old] verified {result.get('fact_key')} = {result.get('verified_value')}{wrong}"
    if tool == "set_hotel":
        return f"[old] hotel set: {result.get('hotel_id')}"
    if tool == "add_item":
        return f"[old] added {_item_line(result.get('added', {}))}"
    if tool == "move_item":
        return f"[old] moved {_item_line(result.get('moved', {}))}"
    if tool == "remove_item":
        return f"[old] removed {result.get('removed')}"
    if tool == "check_itinerary":
        status = "VALID" if result.get("valid") else "NOT valid"
        return f"[old] check: {status} ({result.get('satisfied')}); details trimmed"
    if tool == "view_itinerary":
        return "[old] viewed itinerary (call view_itinerary for the current one)"
    if tool == "finish":
        return "[old] finished"
    return f"[old] {tool}"
