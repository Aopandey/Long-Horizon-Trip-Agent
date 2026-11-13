"""Persisted records: one RunRecord per (scenario, condition, variant, repeat) and its RunScore."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from trip_agent.corruption.corrupt import CorruptionPlan
from trip_agent.env.facts import FactValue
from trip_agent.env.models import Itinerary
from trip_agent.state import CommitmentEvent

Condition = Literal["clean", "corrupted"]


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class RunRecord(BaseModel):
    run_key: str
    experiment: str
    scenario_id: str
    split: str
    horizon: int
    condition: Condition
    variant: str
    repeat: int
    seed: int
    model: str
    status: Literal["completed", "error"] = "completed"
    error: str | None = None
    stop_reason: str | None = None
    turns: int = 0
    tool_calls: int = 0
    tool_call_limit: int = 0
    verify_calls: int = 0
    malformed_tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_latency_ms: float = 0.0
    wall_ms: float = 0.0
    final_itinerary: Itinerary | None = None
    finish_notes: str | None = None
    unresolved: list[str] = Field(default_factory=list)
    corruption: CorruptionPlan
    observed: dict[str, FactValue] = Field(default_factory=dict)
    verified: dict[str, FactValue] = Field(default_factory=dict)
    observed_step: dict[str, int] = Field(default_factory=dict)
    verified_step: dict[str, int] = Field(default_factory=dict)
    history: list[CommitmentEvent] = Field(default_factory=list)
    created_at: str = Field(default_factory=now_iso)


class RunScore(BaseModel):
    run_key: str
    experiment: str
    valid: bool
    fraction_satisfied: float
    n_checks: int
    n_violations: int
    violations_by_check: dict[str, int]
    true_total_cost: float | None
    final_items: int
    false_reliance_items: int
    false_reliance_fraction: float | None
    hotel_price_false: bool
    commitments: int
    commitments_on_false_info: int
    first_error_position: int | None
    errors_after_first: int
    corrupted_facts_observed: int
    conflicts_detected: int
    conflicts_resolved: int
    recovery_success_rate: float | None
    created_at: str = Field(default_factory=now_iso)
