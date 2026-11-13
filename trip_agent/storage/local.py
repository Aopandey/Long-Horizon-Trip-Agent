"""File-based repository: JSON and JSONL under one root folder. Good for laptops and tests."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from trip_agent.env.loader import load_catalog, load_scenarios, write_dataset
from trip_agent.env.models import Catalog, Scenario
from trip_agent.eval.records import RunRecord, RunScore, now_iso


def _safe(key: str) -> str:
    return key.replace("/", "__")


class LocalRepository:
    def __init__(self, root: str | Path, dataset: str = "main"):
        self.root = Path(root)
        self.dataset = dataset
        self._lock = threading.Lock()

    @property
    def _data_dir(self) -> Path:
        return self.root / "datasets" / self.dataset

    def _exp_dir(self, experiment: str) -> Path:
        return self.root / "experiments" / _safe(experiment)

    def _write(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text)
        tmp.replace(path)

    # Dataset.
    def save_dataset(self, catalog: Catalog, scenarios: list[Scenario]) -> None:
        write_dataset(self._data_dir, catalog, scenarios)

    def load_catalog(self) -> Catalog:
        return load_catalog(self._data_dir)

    def load_scenarios(self, split: str | None = None) -> list[Scenario]:
        scenarios = load_scenarios(self._data_dir)
        return [s for s in scenarios if split is None or s.split == split]

    # Experiments and runs.
    def save_experiment(self, name: str, config: dict, dataset_hash: str) -> None:
        path = self._exp_dir(name) / "experiment.json"
        meta = {"name": name, "dataset": self.dataset, "dataset_hash": dataset_hash, "config": config}
        if path.exists():
            meta["created_at"] = json.loads(path.read_text()).get("created_at")
        meta.setdefault("created_at", now_iso())
        meta["updated_at"] = now_iso()
        self._write(path, json.dumps(meta, indent=2, default=str))

    def completed_run_keys(self, experiment: str) -> set[str]:
        return {r.run_key for r in self.load_runs(experiment) if r.status == "completed"}

    def save_run(self, record: RunRecord, events: list[dict]) -> None:
        base = self._exp_dir(record.experiment)
        with self._lock:
            self._write(base / "runs" / f"{_safe(record.run_key)}.json", record.model_dump_json(indent=1))
            self._write(
                base / "events" / f"{_safe(record.run_key)}.jsonl",
                "".join(json.dumps(e, default=str) + "\n" for e in events),
            )

    def load_runs(self, experiment: str) -> list[RunRecord]:
        folder = self._exp_dir(experiment) / "runs"
        return (
            [RunRecord.model_validate_json(p.read_text()) for p in sorted(folder.glob("*.json"))]
            if folder.exists()
            else []
        )

    def load_events(self, run_key: str) -> list[dict]:
        experiment = run_key.split("/", 1)[0]
        path = self._exp_dir(experiment) / "events" / f"{_safe(run_key)}.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []

    def save_score(self, score: RunScore) -> None:
        with self._lock:
            self._write(
                self._exp_dir(score.experiment) / "scores" / f"{_safe(score.run_key)}.json",
                score.model_dump_json(indent=1),
            )

    def load_scores(self, experiment: str) -> list[RunScore]:
        folder = self._exp_dir(experiment) / "scores"
        return (
            [RunScore.model_validate_json(p.read_text()) for p in sorted(folder.glob("*.json"))]
            if folder.exists()
            else []
        )
