"""Independent evaluation against the correct catalog.

Runs after an episode ends, from the stored record. Nothing here is sent back to the agent.

Definitions (provisional; the team may refine them):
- valid: every check passes under the true facts.
- false reliance: a commitment relied on a corrupted fact the agent had seen but not yet verified.
- first error position: 1-based index, among add/move commitments in order, of the first one that
  relied on false information or broke a true constraint at the time it was made.
- errors after first: how many later commitments were also errors (propagation along the horizon).
- conflict detected: the agent saw a corrupted value and later verified it.
- conflict resolved: a detected fact is not involved in any true violation in the final itinerary.
"""

from __future__ import annotations

from collections import Counter

from trip_agent.checker.consistency import TruthView, check_itinerary
from trip_agent.env.facts import place_fact
from trip_agent.env.models import Catalog, Itinerary, Scenario
from trip_agent.eval.records import RunRecord, RunScore
from trip_agent.state import CommitmentEvent, item_dependencies

TIMING_CHECKS = {"opening_hours", "travel", "overlap", "day_window", "duplicate", "place"}


def _relied_on_false(record: RunRecord, keys: list[str], step: int) -> list[str]:
    plan = record.corruption
    out = []
    for key in keys:
        if not plan.is_corrupted(key):
            continue
        seen = record.observed_step.get(key)
        verified = record.verified_step.get(key)
        if seen is not None and seen <= step and (verified is None or verified > step):
            out.append(key)
    return out


def _replay(history: list[CommitmentEvent]):
    """Yield (event, itinerary after the event) in order."""
    hotel = None
    items: dict[str, object] = {}
    for event in history:
        if event.action == "set_hotel":
            hotel = event.hotel_id
        elif event.action in ("add", "move") and event.item is not None:
            items[event.item.id] = event.item
        elif event.action in ("remove", "repair_remove") and event.item_id:
            items.pop(event.item_id, None)
        elif event.action == "restore" and event.snapshot is not None:
            hotel = event.snapshot.hotel_id
            items = {i.id: i for i in event.snapshot.items}
        yield event, Itinerary(hotel_id=hotel, items=list(items.values()))


def score_run(record: RunRecord, scenario: Scenario, catalog: Catalog) -> RunScore:
    truth = TruthView(catalog)
    final = record.final_itinerary or Itinerary()
    report = check_itinerary(final, scenario, catalog, truth)
    violations = report.violations
    end_step = record.tool_calls

    # Reliance on false information in the submitted itinerary.
    false_items = 0
    for item in final.items:
        if _relied_on_false(record, item_dependencies(final, item, catalog), end_step):
            false_items += 1
    hotel_false = bool(
        final.hotel_id and _relied_on_false(record, [place_fact(final.hotel_id, "price_per_night")], end_step)
    )

    # Error onset and propagation over the commitment sequence.
    commitments = on_false = errors_after = 0
    first_error = None
    for event, itinerary in _replay(record.history):
        if event.action not in ("add", "move") or event.item is None:
            continue
        commitments += 1
        false_now = _relied_on_false(record, event.depends_on, event.step)
        on_false += bool(false_now)
        broken = any(
            event.item.id in v.item_ids and v.name in TIMING_CHECKS
            for v in check_itinerary(itinerary, scenario, catalog, truth).violations
        )
        if false_now or broken:
            if first_error is None:
                first_error = commitments
            else:
                errors_after += 1

    # Detection and resolution of corrupted facts.
    plan = record.corruption
    observed_corrupt = [k for k in record.observed_step if plan.is_corrupted(k)]
    detected = [k for k in observed_corrupt if k in record.verified_step]
    resolved = [k for k in detected if not any(k in v.fact_keys for v in violations)]

    return RunScore(
        run_key=record.run_key,
        experiment=record.experiment,
        valid=report.valid,
        fraction_satisfied=round(report.fraction_satisfied, 4),
        n_checks=len(report.checks),
        n_violations=len(violations),
        violations_by_check=dict(Counter(v.name for v in violations)),
        true_total_cost=report.total_cost,
        final_items=len(final.items),
        false_reliance_items=false_items,
        false_reliance_fraction=round(false_items / len(final.items), 4) if final.items else None,
        hotel_price_false=hotel_false,
        commitments=commitments,
        commitments_on_false_info=on_false,
        first_error_position=first_error,
        errors_after_first=errors_after,
        corrupted_facts_observed=len(observed_corrupt),
        conflicts_detected=len(detected),
        conflicts_resolved=len(resolved),
        recovery_success_rate=round(len(resolved) / len(detected), 4) if detected else None,
    )
