"""Experiment grid: scenarios x conditions x variants x repeats, resumable and paired.

The LLM sampling seed depends on (scenario, condition, repeat) but not on the variant, so the
three variants are compared on identical scenarios, identical corruptions and identical seeds.
"""

from __future__ import annotations

import hashlib
import os
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from trip_agent.agent.llm.base import LLMClient
from trip_agent.agent.llm.ollama import OllamaClient, OllamaSettings
from trip_agent.agent.loop import EpisodeLimits, run_episode
from trip_agent.agent.policies import POLICIES, PolicySettings, make_policy
from trip_agent.corruption.corrupt import build_plan, clean_plan
from trip_agent.env.loader import dataset_hash, load_catalog, load_scenarios
from trip_agent.env.models import Catalog, Scenario
from trip_agent.eval.evaluator import score_run
from trip_agent.eval.records import Condition, RunRecord, RunScore
from trip_agent.storage.base import Repository, make_repository
from trip_agent.tools.environment import TripEnvironment


class LLMConfig(BaseModel):
    provider: Literal["ollama"] = "ollama"
    model: str = "qwen3.5:9b"
    host: str | None = None  # default: $OLLAMA_HOST or http://localhost:11434
    temperature: float | None = 0.6
    top_p: float | None = None
    num_ctx: int = 32768
    num_predict: int | None = 2048
    think: bool | None = False
    keep_thinking_in_history: bool = False
    timeout_s: float = 600.0

    def ollama_settings(self) -> OllamaSettings:
        return OllamaSettings(
            model=os.environ.get("TRIP_AGENT_MODEL", self.model),
            host=self.host or os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            temperature=self.temperature,
            top_p=self.top_p,
            num_ctx=self.num_ctx,
            num_predict=self.num_predict,
            think=self.think,
            timeout_s=self.timeout_s,
        )


class BudgetConfig(BaseModel):
    base: int = 20
    per_commitment: int = 5

    def limit(self, scenario: Scenario) -> int:
        return scenario.constraints.max_tool_calls or self.base + self.per_commitment * scenario.horizon


class CorruptionConfig(BaseModel):
    rate: float = Field(default=0.2, ge=0, le=1)
    seed: int = 0


class StorageConfig(BaseModel):
    backend: Literal["local", "supabase"] = "local"
    root: str = "runs_local"
    dataset: str = "main"


class ExperimentConfig(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9_\-.]+$")
    split: Literal["dev", "eval"] = "dev"
    data_dir: str | None = Field(default=None, description="Read the dataset from disk instead of storage")
    scenario_ids: list[str] | None = None
    horizons: list[int] | None = None
    variants: list[str] = Field(default_factory=lambda: ["standard", "verification", "recovery"])
    conditions: list[Condition] = Field(default_factory=lambda: ["clean", "corrupted"])
    repeats: int = Field(default=3, ge=1)
    seed: int = 0
    corruption: CorruptionConfig = Field(default_factory=CorruptionConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    policy: PolicySettings = Field(default_factory=PolicySettings)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    max_turns: int = 200
    history_window: int | None = Field(
        default=6, ge=1, description="Model turns kept in full; older ones shrink to one line. null keeps all."
    )
    concurrency: int = Field(default=1, ge=1)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    @field_validator("variants")
    @classmethod
    def _known_variants(cls, value: list[str]) -> list[str]:
        unknown = set(value) - set(POLICIES)
        if unknown:
            raise ValueError(f"unknown variants {sorted(unknown)}; choose from {sorted(POLICIES)}")
        return value

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))


@dataclass(frozen=True)
class RunSpec:
    scenario: Scenario
    condition: Condition
    variant: str
    repeat: int
    seed: int
    run_key: str


def _stable_int(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "big")


def select_scenarios(cfg: ExperimentConfig, scenarios: list[Scenario]) -> list[Scenario]:
    chosen = [s for s in scenarios if s.split == cfg.split]
    if cfg.scenario_ids:
        wanted = set(cfg.scenario_ids)
        chosen = [s for s in chosen if s.id in wanted]
    if cfg.horizons:
        chosen = [s for s in chosen if s.horizon in set(cfg.horizons)]
    return sorted(chosen, key=lambda s: (s.horizon, s.id))


def plan_runs(cfg: ExperimentConfig, scenarios: list[Scenario]) -> list[RunSpec]:
    specs = []
    for scenario in select_scenarios(cfg, scenarios):
        for condition in cfg.conditions:
            for repeat in range(cfg.repeats):
                seed = cfg.seed + _stable_int(scenario.id, condition, repeat) % 1_000_000
                for variant in cfg.variants:
                    key = f"{cfg.name}/{scenario.id}/{condition}/{variant}/r{repeat}"
                    specs.append(RunSpec(scenario, condition, variant, repeat, seed, key))
    return specs


def execute_run(spec: RunSpec, catalog: Catalog, cfg: ExperimentConfig, llm: LLMClient) -> tuple[RunRecord, list[dict]]:
    scenario = spec.scenario
    plan = (
        build_plan(catalog, scenario.id, rate=cfg.corruption.rate, seed=cfg.corruption.seed)
        if spec.condition == "corrupted"
        else clean_plan(scenario.id)
    )
    limit = cfg.budget.limit(scenario)
    env = TripEnvironment(catalog, scenario, plan, max_tool_calls=limit)
    base = dict(
        run_key=spec.run_key,
        experiment=cfg.name,
        scenario_id=scenario.id,
        split=scenario.split,
        horizon=scenario.horizon,
        condition=spec.condition,
        variant=spec.variant,
        repeat=spec.repeat,
        seed=spec.seed,
        model=getattr(llm, "model", "unknown"),
        tool_call_limit=limit,
        corruption=plan,
    )
    try:
        result = run_episode(
            env,
            make_policy(spec.variant, cfg.policy),
            llm,
            EpisodeLimits(
                max_turns=cfg.max_turns,
                context_limit=cfg.llm.num_ctx,
                keep_thinking_in_history=cfg.llm.keep_thinking_in_history,
                recent_turns=cfg.history_window,
            ),
            seed=spec.seed,
        )
    except Exception as exc:  # keep the grid going; the run is retried on resume
        env.events.append({"type": "error", "error": repr(exc), "traceback": traceback.format_exc()})
        return RunRecord(**base, status="error", error=repr(exc), tool_calls=env.budget.used), env.events
    state = env.state
    record = RunRecord(
        **base,
        stop_reason=result.stop_reason,
        turns=result.turns,
        tool_calls=result.tool_calls,
        verify_calls=result.verify_calls,
        malformed_tool_calls=result.malformed_tool_calls,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        llm_latency_ms=result.llm_latency_ms,
        wall_ms=result.wall_ms,
        final_itinerary=result.itinerary,
        finish_notes=result.finish_notes,
        unresolved=result.unresolved,
        observed=state.observed,
        verified=state.verified,
        observed_step=state.observed_step,
        verified_step=state.verified_step,
        history=state.history,
    )
    return record, result.events


def make_llm(cfg: ExperimentConfig) -> LLMClient:
    return OllamaClient(cfg.llm.ollama_settings())


def load_dataset(cfg: ExperimentConfig, repo: Repository) -> tuple[Catalog, list[Scenario]]:
    if cfg.data_dir:
        return load_catalog(Path(cfg.data_dir)), load_scenarios(Path(cfg.data_dir))
    return repo.load_catalog(), repo.load_scenarios()


def run_experiment(
    cfg: ExperimentConfig,
    repo: Repository | None = None,
    llm_factory: Callable[[ExperimentConfig], LLMClient] = make_llm,
    progress: Callable[[str], None] = print,
    limit_runs: int | None = None,
) -> dict:
    repo = repo or make_repository(cfg.storage.backend, root=cfg.storage.root, dataset=cfg.storage.dataset)
    catalog, scenarios = load_dataset(cfg, repo)
    specs = plan_runs(cfg, scenarios)
    if not specs:
        raise ValueError(f"no scenarios match split={cfg.split} ids={cfg.scenario_ids} horizons={cfg.horizons}")
    repo.save_experiment(cfg.name, cfg.model_dump(mode="json"), dataset_hash(catalog, scenarios))
    done = repo.completed_run_keys(cfg.name)
    todo = [s for s in specs if s.run_key not in done][:limit_runs]
    progress(
        f"{cfg.name}: {len(specs)} runs planned, {len(done & {s.run_key for s in specs})} done, {len(todo)} to run now"
    )
    llm = llm_factory(cfg)
    counts = {"completed": 0, "error": 0}

    def work(spec: RunSpec) -> RunRecord:
        record, events = execute_run(spec, catalog, cfg, llm)
        repo.save_run(record, events)
        if record.status == "completed":
            repo.save_score(score_run(record, spec.scenario, catalog))
        return record

    with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
        futures = {pool.submit(work, spec): spec for spec in todo}
        for i, future in enumerate(as_completed(futures), 1):
            record = future.result()
            counts[record.status] += 1
            progress(
                f"[{i}/{len(todo)}] {record.run_key}: {record.status} {record.stop_reason or record.error or ''} "
                f"({record.tool_calls}/{record.tool_call_limit} calls, {record.wall_ms / 1000:.1f}s)"
            )
    return {"planned": len(specs), "ran": len(todo), **counts}


def evaluate_experiment(cfg: ExperimentConfig, repo: Repository | None = None) -> list[RunScore]:
    """Re-score every completed run (use after changing the evaluator)."""
    repo = repo or make_repository(cfg.storage.backend, root=cfg.storage.root, dataset=cfg.storage.dataset)
    catalog, scenarios = load_dataset(cfg, repo)
    by_id = {s.id: s for s in scenarios}
    scores = []
    for record in repo.load_runs(cfg.name):
        if record.status != "completed" or record.scenario_id not in by_id:
            continue
        score = score_run(record, by_id[record.scenario_id], catalog)
        repo.save_score(score)
        scores.append(score)
    return scores


def result_rows(cfg: ExperimentConfig, repo: Repository | None = None) -> list[dict]:
    """One flat row per scored run: run metadata joined with its score."""
    repo = repo or make_repository(cfg.storage.backend, root=cfg.storage.root, dataset=cfg.storage.dataset)
    scores = {s.run_key: s for s in repo.load_scores(cfg.name)}
    rows = []
    for record in repo.load_runs(cfg.name):
        score = scores.get(record.run_key)
        if score is None:
            continue
        row = record.model_dump(
            include={
                "run_key",
                "scenario_id",
                "horizon",
                "condition",
                "variant",
                "repeat",
                "model",
                "stop_reason",
                "tool_calls",
                "tool_call_limit",
                "verify_calls",
                "turns",
                "prompt_tokens",
                "completion_tokens",
                "wall_ms",
            }
        )
        row.update(score.model_dump(exclude={"run_key", "experiment", "created_at", "violations_by_check"}))
        rows.append(row)
    return rows
