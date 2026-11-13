"""Controlled misinformation.

A corruption plan is a pure function of (catalog, scenario id, seed, rate): exactly
round(rate * N) eligible facts are altered, chosen by a stable hash ranking, and each is
perturbed the same way every time. So every agent variant sees identical false values, and a
fact never changes during a run. The correct catalog is never modified.

Perturbations are "optimistic", like the proposal's examples: later closing or earlier opening
times, lower prices, shorter travel times. They make infeasible plans look feasible.
"""

from __future__ import annotations

import hashlib
import random

from pydantic import BaseModel, Field

from trip_agent.env.facts import FactValue, fact_kind, true_facts
from trip_agent.env.models import MINUTES_PER_DAY, Catalog

CLOCK_SHIFTS = (60, 90, 120, 150, 180)
PRICE_FACTORS = (0.45, 0.75)
TRAVEL_FACTORS = (0.25, 0.5)
MIN_TRAVEL_MINUTES = 4


class CorruptionPlan(BaseModel):
    scenario_id: str
    seed: int
    rate: float
    eligible_count: int = 0
    altered: dict[str, FactValue] = Field(default_factory=dict)

    def observed(self, key: str, true_value: FactValue) -> FactValue:
        return self.altered.get(key, true_value)

    def is_corrupted(self, key: str) -> bool:
        return key in self.altered


def clean_plan(scenario_id: str) -> CorruptionPlan:
    return CorruptionPlan(scenario_id=scenario_id, seed=0, rate=0.0)


def _digest(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def eligible_facts(catalog: Catalog) -> dict[str, FactValue]:
    """Facts that can be perturbed without becoming a no-op (free places and very short legs are skipped)."""
    eligible = {}
    for key, value in true_facts(catalog).items():
        kind = fact_kind(key)
        if kind == "price" and value <= 0:
            continue
        if kind == "travel_time" and value < MIN_TRAVEL_MINUTES:
            continue
        eligible[key] = value
    return eligible


def perturb(key: str, value: FactValue, rng: random.Random) -> FactValue:
    kind = fact_kind(key)
    if kind == "opening_hours":
        shift = rng.choice(CLOCK_SHIFTS)
        if key.endswith(".close_time"):
            new = min(MINUTES_PER_DAY, int(value) + shift)
            return new if new != value else int(value) - shift
        new = max(0, int(value) - shift)
        return new if new != value else int(value) + shift
    if kind == "price":
        return round(float(value) * rng.uniform(*PRICE_FACTORS), 2)
    return max(1, round(int(value) * rng.uniform(*TRAVEL_FACTORS)))


def build_plan(catalog: Catalog, scenario_id: str, *, rate: float = 0.2, seed: int = 0) -> CorruptionPlan:
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be in [0, 1]")
    eligible = eligible_facts(catalog)
    ranked = sorted(eligible, key=lambda k: (_digest(scenario_id, seed, k), k))
    chosen = ranked[: round(rate * len(eligible))]
    altered = {k: perturb(k, eligible[k], random.Random(_digest("value", scenario_id, seed, k))) for k in chosen}
    return CorruptionPlan(scenario_id=scenario_id, seed=seed, rate=rate, eligible_count=len(eligible), altered=altered)
