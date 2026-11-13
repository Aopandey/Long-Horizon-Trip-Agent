"""Storage interface. The agent never touches storage; only the runner and evaluator do."""

from __future__ import annotations

from typing import Protocol

from trip_agent.env.models import Catalog, Scenario
from trip_agent.eval.records import RunRecord, RunScore


class Repository(Protocol):
    dataset: str

    # Dataset (the trusted catalog plus scenarios with hidden reference itineraries).
    def save_dataset(self, catalog: Catalog, scenarios: list[Scenario]) -> None: ...
    def load_catalog(self) -> Catalog: ...
    def load_scenarios(self, split: str | None = None) -> list[Scenario]: ...

    # Experiments and runs.
    def save_experiment(self, name: str, config: dict, dataset_hash: str) -> None: ...
    def completed_run_keys(self, experiment: str) -> set[str]: ...
    def save_run(self, record: RunRecord, events: list[dict]) -> None: ...
    def load_runs(self, experiment: str) -> list[RunRecord]: ...
    def load_events(self, run_key: str) -> list[dict]: ...
    def save_score(self, score: RunScore) -> None: ...
    def load_scores(self, experiment: str) -> list[RunScore]: ...


def make_repository(backend: str, *, root: str = "runs_local", dataset: str = "main") -> Repository:
    if backend == "local":
        from trip_agent.storage.local import LocalRepository

        return LocalRepository(root, dataset=dataset)
    if backend == "supabase":
        from trip_agent.storage.supabase_store import SupabaseRepository

        return SupabaseRepository.from_env(dataset=dataset)
    raise ValueError(f"unknown storage backend {backend!r} (use 'local' or 'supabase')")
