from collections import defaultdict

from conftest import reference_script

from trip_agent.agent.llm.scripted import ScriptedClient
from trip_agent.eval.stats import compare_variants, mcnemar_exact, paired_bootstrap, summarize
from trip_agent.runner.experiment import ExperimentConfig, plan_runs, result_rows, run_experiment
from trip_agent.storage.local import LocalRepository
from trip_agent.storage.supabase_store import SupabaseRepository


class FakeSupabase:
    """Just enough of the supabase-py query builder for the repository's calls."""

    def __init__(self):
        self.tables = defaultdict(list)

    def table(self, name):
        return _Query(self, name)


class _Query:
    def __init__(self, db, name):
        self.db, self.name, self.filters, self.op, self.payload, self.conflict, self.span = (
            db,
            name,
            [],
            None,
            None,
            None,
            None,
        )

    def select(self, columns="*"):
        self.op, self.columns = "select", None if columns == "*" else columns.split(",")
        return self

    def upsert(self, rows, on_conflict=None):
        self.op, self.payload, self.conflict = "upsert", rows, on_conflict.split(",")
        return self

    def delete(self):
        self.op = "delete"
        return self

    def eq(self, column, value):
        self.filters.append((column, value))
        return self

    def range(self, start, end):
        self.span = (start, end)
        return self

    def execute(self):
        rows = self.db.tables[self.name]
        match = [r for r in rows if all(r.get(c) == v for c, v in self.filters)]
        if self.op == "upsert":
            for new in self.payload:
                key = tuple(new[c] for c in self.conflict)
                rows[:] = [r for r in rows if tuple(r.get(c) for c in self.conflict) != key] + [dict(new)]
            match = self.payload
        elif self.op == "delete":
            rows[:] = [r for r in rows if r not in match]
        elif self.op == "select":
            if self.span:
                match = match[self.span[0] : self.span[1] + 1]
            if self.columns:
                match = [{c: r.get(c) for c in self.columns} for r in match]
        return type("Result", (), {"data": match})()


def _cfg(tmp_path, **extra):
    return ExperimentConfig(
        name="t", split="dev", repeats=2, storage={"backend": "local", "root": str(tmp_path)}, **extra
    )


def _scripted_factory(scenarios):
    by_id = {s.id: s for s in scenarios}

    class PerRun:
        model = "scripted"

        def __init__(self):
            self.clients = {}

        def chat(self, messages, tools, *, seed=None):
            brief = messages[1]["content"]
            key = (brief, len(self.clients))
            sid = next(s for s in by_id.values() if s.constraints.budget and f"${s.constraints.budget:.2f}" in brief).id
            if len(messages) == 2:  # a new episode starts
                self.clients[key] = ScriptedClient(reference_script(by_id[sid]))
                self.current = self.clients[key]
            return self.current.chat(messages, tools, seed=seed)

    return lambda cfg: PerRun()


def test_plan_runs_pairs_seeds_across_variants(scenarios):
    cfg = ExperimentConfig(name="t", split="dev", repeats=3)
    specs = plan_runs(cfg, scenarios)
    assert len(specs) == 1 * 2 * 3 * 3
    seeds = defaultdict(set)
    for s in specs:
        seeds[(s.scenario.id, s.condition, s.repeat)].add(s.seed)
    assert all(len(v) == 1 for v in seeds.values())


def test_experiment_runs_scores_and_resumes(tmp_path, catalog, scenarios):
    repo = LocalRepository(tmp_path)
    repo.save_dataset(catalog, scenarios)
    cfg = _cfg(tmp_path)
    summary = run_experiment(cfg, repo, llm_factory=_scripted_factory(scenarios), progress=lambda m: None)
    assert summary == {"planned": 12, "ran": 12, "completed": 12, "error": 0}
    again = run_experiment(cfg, repo, llm_factory=_scripted_factory(scenarios), progress=lambda m: None)
    assert again["ran"] == 0
    rows = result_rows(cfg, repo)
    assert len(rows) == 12 and all(r["valid"] for r in rows)
    assert repo.load_events(rows[0]["run_key"])
    table = summarize(rows)
    assert {r["variant"] for r in table} == {"standard", "verification", "recovery"}
    comparison = compare_variants(rows)
    assert all(c["n"] == 2 and c["mean_diff"] == 0 for c in comparison)


def test_supabase_repository_round_trip(catalog, scenarios):
    repo = SupabaseRepository(FakeSupabase(), dataset="example")
    repo.save_dataset(catalog, scenarios)
    assert repo.load_catalog() == catalog
    assert [s.id for s in repo.load_scenarios(split="eval")] == ["example-h5-tuesday"]

    from trip_agent.corruption.corrupt import clean_plan
    from trip_agent.eval.records import RunRecord, RunScore

    record = RunRecord(
        run_key="e/s/clean/standard/r0",
        experiment="e",
        scenario_id="s",
        split="dev",
        horizon=5,
        condition="clean",
        variant="standard",
        repeat=0,
        seed=1,
        model="m",
        corruption=clean_plan("s"),
    )
    repo.save_experiment("e", {"a": 1}, "hash")
    repo.save_run(record, [{"type": "tool", "step": 1}, {"type": "llm"}])
    repo.save_run(record, [{"type": "tool", "step": 1}])  # re-saving replaces events
    assert repo.completed_run_keys("e") == {record.run_key}
    assert repo.load_runs("e") == [record]
    assert repo.load_events(record.run_key) == [{"type": "tool", "step": 1}]
    score = RunScore(
        run_key=record.run_key,
        experiment="e",
        valid=True,
        fraction_satisfied=1.0,
        n_checks=3,
        n_violations=0,
        violations_by_check={},
        true_total_cost=1.0,
        final_items=0,
        false_reliance_items=0,
        false_reliance_fraction=None,
        hotel_price_false=False,
        commitments=0,
        commitments_on_false_info=0,
        first_error_position=None,
        errors_after_first=0,
        corrupted_facts_observed=0,
        conflicts_detected=0,
        conflicts_resolved=0,
        recovery_success_rate=None,
    )
    repo.save_score(score)
    assert repo.load_scores("e") == [score]


def test_stats_helpers():
    result = paired_bootstrap([0, 0, 1, 1], [1, 1, 1, 1], n_boot=2000)
    assert result["mean_diff"] == 0.5 and result["ci_low"] <= 0.5 <= result["ci_high"]
    assert mcnemar_exact([False] * 10, [True] * 10)["p_value"] < 0.01
    assert mcnemar_exact([True, False], [True, False])["p_value"] == 1.0
