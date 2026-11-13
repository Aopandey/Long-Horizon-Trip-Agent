"""Supabase (Postgres) repository. Schema: supabase/migrations/*_init.sql.

Uses the secret (service-role) key, which bypasses row-level security. RLS is enabled on every
table with no policies, so the public anon key can read nothing. Keep the secret key in .env.
"""

from __future__ import annotations

import json
import os

from trip_agent.env.models import Catalog, Scenario
from trip_agent.envfile import load_dotenv
from trip_agent.eval.records import RunRecord, RunScore

PAGE = 1000
EVENT_CHUNK = 500


class SupabaseRepository:
    def __init__(self, client, dataset: str = "main"):
        self.client = client
        self.dataset = dataset

    @classmethod
    def from_env(cls, dataset: str = "main") -> SupabaseRepository:
        try:
            from supabase import create_client
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install the supabase extra: pip install -e '.[supabase]'") from exc
        load_dotenv()
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise RuntimeError("set SUPABASE_URL and SUPABASE_SERVICE_KEY (see .env.example)")
        return cls(create_client(url, key), dataset=dataset)

    # ------------------------------------------------------------------ helpers
    def _select_all(self, table: str, columns: str = "*", **eq) -> list[dict]:
        rows, start = [], 0
        while True:
            query = self.client.table(table).select(columns)
            for column, value in eq.items():
                query = query.eq(column, value)
            batch = query.range(start, start + PAGE - 1).execute().data or []
            rows.extend(batch)
            if len(batch) < PAGE:
                return rows
            start += PAGE

    def _upsert(self, table: str, rows: list[dict], on_conflict: str) -> None:
        for i in range(0, len(rows), EVENT_CHUNK):
            self.client.table(table).upsert(rows[i : i + EVENT_CHUNK], on_conflict=on_conflict).execute()

    # ------------------------------------------------------------------ dataset
    def save_dataset(self, catalog: Catalog, scenarios: list[Scenario]) -> None:
        meta = catalog.model_dump(mode="json", exclude={"attractions", "restaurants", "hotels", "travel"})
        self._upsert("datasets", [{"name": self.dataset, "catalog_meta": meta}], "name")
        places = [
            {
                "dataset": self.dataset,
                "id": p.id,
                "kind": p.kind,
                "name": p.name,
                "neighborhood": p.neighborhood,
                "data": p.model_dump(mode="json"),
            }
            for p in catalog.places.values()
        ]
        self._upsert("places", places, "dataset,id")
        legs = [{"dataset": self.dataset, **leg.model_dump(mode="json")} for leg in catalog.travel]
        self._upsert("travel_legs", legs, "dataset,from_id,to_id")
        rows = [
            {
                "dataset": self.dataset,
                "id": s.id,
                "split": s.split,
                "horizon": s.horizon,
                "data": s.model_dump(mode="json"),
            }
            for s in scenarios
        ]
        self._upsert("scenarios", rows, "dataset,id")

    def load_catalog(self) -> Catalog:
        meta_rows = self._select_all("datasets", name=self.dataset)
        if not meta_rows:
            raise RuntimeError(f"dataset {self.dataset!r} not found in Supabase; run `trip-agent import-data`")
        places = self._select_all("places", "kind,data", dataset=self.dataset)
        legs = self._select_all("travel_legs", "from_id,to_id,minutes,cost_per_person", dataset=self.dataset)
        by_kind = {"attraction": [], "restaurant": [], "hotel": []}
        for row in places:
            by_kind[row["kind"]].append(row["data"])
        return Catalog.model_validate(
            {
                **meta_rows[0]["catalog_meta"],
                "attractions": by_kind["attraction"],
                "restaurants": by_kind["restaurant"],
                "hotels": by_kind["hotel"],
                "travel": legs,
            }
        )

    def load_scenarios(self, split: str | None = None) -> list[Scenario]:
        filters = {"dataset": self.dataset}
        if split:
            filters["split"] = split
        rows = self._select_all("scenarios", "data", **filters)
        return sorted((Scenario.model_validate(r["data"]) for r in rows), key=lambda s: s.id)

    # ------------------------------------------------------------------ experiments and runs
    def save_experiment(self, name: str, config: dict, dataset_hash: str) -> None:
        row = {"name": name, "dataset": self.dataset, "dataset_hash": dataset_hash, "config": config}
        self._upsert("experiments", [row], "name")

    def completed_run_keys(self, experiment: str) -> set[str]:
        rows = self._select_all("runs", "run_key", experiment=experiment, status="completed")
        return {r["run_key"] for r in rows}

    def save_run(self, record: RunRecord, events: list[dict]) -> None:
        row = {
            "run_key": record.run_key,
            "experiment": record.experiment,
            "scenario_id": record.scenario_id,
            "split": record.split,
            "horizon": record.horizon,
            "condition": record.condition,
            "variant": record.variant,
            "repeat": record.repeat,
            "seed": record.seed,
            "model": record.model,
            "status": record.status,
            "stop_reason": record.stop_reason,
            "tool_calls": record.tool_calls,
            "verify_calls": record.verify_calls,
            "turns": record.turns,
            "prompt_tokens": record.prompt_tokens,
            "completion_tokens": record.completion_tokens,
            "wall_ms": record.wall_ms,
            "record": record.model_dump(mode="json"),
        }
        self._upsert("runs", [row], "run_key")
        self.client.table("run_events").delete().eq("run_key", record.run_key).execute()
        event_rows = [
            {"run_key": record.run_key, "seq": i, "type": e.get("type", "event"), "data": e}
            for i, e in enumerate(events)
        ]
        self._upsert("run_events", _jsonable(event_rows), "run_key,seq")

    def load_runs(self, experiment: str) -> list[RunRecord]:
        rows = self._select_all("runs", "record", experiment=experiment)
        return sorted((RunRecord.model_validate(r["record"]) for r in rows), key=lambda r: r.run_key)

    def load_events(self, run_key: str) -> list[dict]:
        rows = self._select_all("run_events", "seq,data", run_key=run_key)
        return [r["data"] for r in sorted(rows, key=lambda r: r["seq"])]

    def save_score(self, score: RunScore) -> None:
        row = {
            "run_key": score.run_key,
            "experiment": score.experiment,
            "valid": score.valid,
            "fraction_satisfied": score.fraction_satisfied,
            "false_reliance_items": score.false_reliance_items,
            "first_error_position": score.first_error_position,
            "recovery_success_rate": score.recovery_success_rate,
            "score": score.model_dump(mode="json"),
        }
        self._upsert("run_scores", [row], "run_key")

    def load_scores(self, experiment: str) -> list[RunScore]:
        rows = self._select_all("run_scores", "score", experiment=experiment)
        return sorted((RunScore.model_validate(r["score"]) for r in rows), key=lambda s: s.run_key)


def _jsonable(rows: list[dict]) -> list[dict]:
    return json.loads(json.dumps(rows, default=str))
