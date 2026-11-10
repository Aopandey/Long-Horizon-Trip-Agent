"""Command line: `trip-agent --help`."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Annotated

import typer

from trip_agent.envfile import load_dotenv

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Long-horizon trip-planning agent (CS 5130).")
DataDir = Annotated[Path, typer.Argument(help="Folder with catalog.json and scenarios/*.json")]


@app.callback()
def _main() -> None:
    load_dotenv()


@app.command()
def schema(out: Annotated[Path, typer.Option(help="Output folder")] = Path("docs/schemas")) -> None:
    """Write JSON Schemas for catalog.json and scenario files."""
    from trip_agent.env.models import Catalog, Scenario

    out.mkdir(parents=True, exist_ok=True)
    for name, model in (("catalog", Catalog), ("scenario", Scenario)):
        (out / f"{name}.schema.json").write_text(json.dumps(model.model_json_schema(), indent=2) + "\n")
    typer.echo(f"wrote {out}/catalog.schema.json and {out}/scenario.schema.json")


@app.command("example-data")
def example_data(out: Annotated[Path, typer.Argument(help="Output folder")] = Path("examples/tiny")) -> None:
    """Write the tiny example dataset (format demo only, not experiment data)."""
    from trip_agent.env.example import example_catalog, example_scenarios
    from trip_agent.env.loader import write_dataset

    write_dataset(out, example_catalog(), example_scenarios())
    typer.echo(f"wrote example dataset to {out}")


@app.command()
def validate(
    data_dir: Annotated[Path | None, typer.Argument(help="Folder with catalog.json and scenarios/*.json")] = None,
    backend: Annotated[str | None, typer.Option(help="Validate the copy in storage instead: local or supabase")] = None,
    dataset: Annotated[str, typer.Option(help="Dataset name in storage")] = "main",
    root: Annotated[str, typer.Option(help="Root folder for the local backend")] = "runs_local",
) -> None:
    """Validate a dataset: schema, references, horizon counts, and that every reference itinerary is feasible."""
    from trip_agent.env.loader import DataError, dataset_hash, validate_dataset

    try:
        catalog, scenarios = _load_dataset(data_dir, backend, dataset, root)
    except DataError as exc:
        typer.secho(f"invalid data:\n{exc}", fg="red")
        raise typer.Exit(1) from exc
    report = validate_dataset(catalog, scenarios)
    for line in report.info:
        typer.echo(f"  {line}")
    for line in report.warnings:
        typer.secho(f"warning: {line}", fg="yellow")
    for line in report.errors:
        typer.secho(f"error: {line}", fg="red")
    splits = {s: sum(x.split == s for x in scenarios) for s in ("dev", "eval")}
    typer.echo(
        f"{len(catalog.places)} places, {len(catalog.travel)} travel legs, {len(scenarios)} scenarios {splits}, "
        f"hash {dataset_hash(catalog, scenarios)}"
    )
    if not report.ok:
        raise typer.Exit(1)
    typer.secho("dataset OK", fg="green")


@app.command("import-data")
def import_data(
    data_dir: DataDir,
    backend: Annotated[str, typer.Option(help="local or supabase")] = "supabase",
    dataset: Annotated[str, typer.Option(help="Dataset name in storage")] = "main",
    root: Annotated[str, typer.Option(help="Root folder for the local backend")] = "runs_local",
) -> None:
    """Validate a dataset and upload it to storage."""
    from trip_agent.env.loader import load_catalog, load_scenarios, validate_dataset
    from trip_agent.storage.base import make_repository

    catalog, scenarios = load_catalog(data_dir), load_scenarios(data_dir)
    report = validate_dataset(catalog, scenarios)
    if not report.ok:
        for line in report.errors:
            typer.secho(f"error: {line}", fg="red")
        raise typer.Exit(1)
    make_repository(backend, root=root, dataset=dataset).save_dataset(catalog, scenarios)
    typer.secho(
        f"imported {len(catalog.places)} places and {len(scenarios)} scenarios into {backend}:{dataset}", fg="green"
    )


@app.command("check-llm")
def check_llm(
    model: Annotated[str, typer.Option()] = "qwen3.5:9b",
    host: Annotated[str | None, typer.Option(help="Default: $OLLAMA_HOST or localhost")] = None,
) -> None:
    """Check that Ollama is reachable and the model is pulled and supports tools."""
    from trip_agent.agent.llm.ollama import OllamaError
    from trip_agent.runner.experiment import LLMConfig, OllamaClient

    settings = LLMConfig(model=model, host=host).ollama_settings()
    client = OllamaClient(settings)
    try:
        info = client.model_info()
    except (OllamaError, Exception) as exc:
        typer.secho(f"{settings.host}: {exc}", fg="red")
        raise typer.Exit(1) from exc
    capabilities = info.get("capabilities", [])
    typer.echo(f"{settings.host} {settings.model}: capabilities {capabilities}")
    if "tools" not in capabilities:
        typer.secho("this model does not report tool-calling support", fg="red")
        raise typer.Exit(1)
    typer.secho("OK", fg="green")


@app.command()
def smoke(
    data_dir: Annotated[Path | None, typer.Option(help="Dataset folder (default: built-in example)")] = None,
    scenario: Annotated[str | None, typer.Option(help="Scenario id (default: first)")] = None,
    variant: Annotated[str, typer.Option()] = "recovery",
    condition: Annotated[str, typer.Option()] = "corrupted",
    model: Annotated[str, typer.Option()] = "qwen3.5:9b",
    max_tool_calls: Annotated[int | None, typer.Option()] = None,
    think: Annotated[bool, typer.Option(help="Enable model thinking")] = False,
    show_messages: Annotated[bool, typer.Option(help="Print the tool calls as they happen")] = True,
    backend: Annotated[str | None, typer.Option(help="Read the dataset from storage: local or supabase")] = None,
    dataset: Annotated[str, typer.Option(help="Dataset name in storage")] = "main",
) -> None:
    """Run one episode against the live model and print the score. Nothing is stored."""
    from trip_agent.agent.llm.ollama import OllamaClient
    from trip_agent.agent.loop import EpisodeLimits, run_episode
    from trip_agent.agent.policies import make_policy
    from trip_agent.corruption.corrupt import build_plan, clean_plan
    from trip_agent.env.example import example_catalog, example_scenarios
    from trip_agent.eval.evaluator import score_run
    from trip_agent.eval.records import RunRecord
    from trip_agent.runner.experiment import BudgetConfig, LLMConfig
    from trip_agent.tools.environment import TripEnvironment

    if data_dir or backend:
        catalog, scenarios = _load_dataset(data_dir, backend, dataset, "runs_local")
    else:
        catalog, scenarios = example_catalog(), example_scenarios()
    chosen = next((s for s in scenarios if s.id == scenario), None) if scenario else scenarios[0]
    if chosen is None:
        typer.secho(f"no scenario {scenario!r}", fg="red")
        raise typer.Exit(1)
    plan = build_plan(catalog, chosen.id) if condition == "corrupted" else clean_plan(chosen.id)
    limit = max_tool_calls or BudgetConfig().limit(chosen)
    env = TripEnvironment(catalog, chosen, plan, max_tool_calls=limit)
    llm_cfg = LLMConfig(model=model, think=think)
    client = OllamaClient(llm_cfg.ollama_settings())
    typer.echo(
        f"{chosen.id} | {variant} | {condition} ({len(plan.altered)} facts altered) | {client.model} | {limit} calls"
    )

    if show_messages:
        original = env.call

        def traced(name, arguments):
            result = original(name, arguments)
            short = json.dumps(result, default=str)
            typer.echo(f"  {env.step:>3} {name}({json.dumps(arguments, default=str)[:120]}) -> {short[:160]}")
            return result

        env.call = traced  # type: ignore[method-assign]

    result = run_episode(env, make_policy(variant), client, EpisodeLimits(context_limit=llm_cfg.num_ctx), seed=0)
    record = RunRecord(
        run_key=f"smoke/{chosen.id}/{condition}/{variant}/r0",
        experiment="smoke",
        scenario_id=chosen.id,
        split=chosen.split,
        horizon=chosen.horizon,
        condition=condition,
        variant=variant,
        repeat=0,
        seed=0,
        model=client.model,
        stop_reason=result.stop_reason,
        tool_calls=result.tool_calls,
        tool_call_limit=limit,
        verify_calls=result.verify_calls,
        final_itinerary=result.itinerary,
        corruption=plan,
        observed=env.state.observed,
        verified=env.state.verified,
        observed_step=env.state.observed_step,
        verified_step=env.state.verified_step,
        history=env.state.history,
    )
    score = score_run(record, chosen, catalog)
    typer.echo(
        f"stop={result.stop_reason} turns={result.turns} calls={result.tool_calls} verify={result.verify_calls} "
        f"malformed={result.malformed_tool_calls} tokens={result.prompt_tokens}+{result.completion_tokens} "
        f"time={result.wall_ms / 1000:.1f}s"
    )
    typer.echo(json.dumps(score.model_dump(exclude={"created_at"}), indent=2))


def _load_dataset(data_dir: Path | None, backend: str | None, dataset: str, root: str):
    """A dataset from a folder, or from storage when a backend is given."""
    from trip_agent.env.loader import DataError, load_catalog, load_scenarios
    from trip_agent.storage.base import make_repository

    if data_dir:
        return load_catalog(data_dir), load_scenarios(data_dir)
    if not backend:
        raise DataError("give a data folder, or --backend local|supabase to read the dataset from storage")
    try:
        repo = make_repository(backend, root=root, dataset=dataset)
        return repo.load_catalog(), repo.load_scenarios()
    except (RuntimeError, ValueError) as exc:
        raise DataError(f"{backend}:{dataset}: {exc}") from exc


def _cfg(config: Path):
    from trip_agent.runner.experiment import ExperimentConfig

    return ExperimentConfig.from_yaml(config)


@app.command()
def run(
    config: Annotated[Path, typer.Argument(help="Experiment YAML")],
    limit: Annotated[int | None, typer.Option(help="Run at most this many pending runs")] = None,
    allow_eval: Annotated[bool, typer.Option(help="Required for split: eval (run it once, after freezing)")] = False,
) -> None:
    """Run (or resume) an experiment grid. Completed runs are skipped."""
    from trip_agent.runner.experiment import run_experiment

    cfg = _cfg(config)
    if cfg.split == "eval" and not allow_eval:
        typer.secho(
            "This config uses the eval split. Freeze prompts and settings first, then pass --allow-eval.", fg="red"
        )
        raise typer.Exit(1)
    summary = run_experiment(cfg, progress=typer.echo, limit_runs=limit)
    typer.echo(json.dumps(summary))


@app.command()
def evaluate(config: Annotated[Path, typer.Argument(help="Experiment YAML")]) -> None:
    """Re-score every completed run of an experiment."""
    from trip_agent.runner.experiment import evaluate_experiment

    scores = evaluate_experiment(_cfg(config))
    typer.echo(f"scored {len(scores)} runs")


@app.command()
def report(
    config: Annotated[Path, typer.Argument(help="Experiment YAML")],
    out: Annotated[Path, typer.Option(help="Output folder for CSV and JSON")] = Path("reports"),
) -> None:
    """Summary tables and paired comparisons against the standard agent."""
    from trip_agent.eval.stats import compare_variants, summarize
    from trip_agent.runner.experiment import result_rows

    cfg = _cfg(config)
    rows = result_rows(cfg)
    if not rows:
        typer.secho("no scored runs yet", fg="yellow")
        raise typer.Exit(1)
    folder = out / cfg.name
    folder.mkdir(parents=True, exist_ok=True)
    _write_csv(folder / "runs.csv", rows)
    summary = summarize(rows)
    _write_csv(folder / "summary.csv", summary)
    comparisons = {m: compare_variants(rows, metric=m) for m in ("valid", "fraction_satisfied", "false_reliance_items")}
    (folder / "comparisons.json").write_text(json.dumps(comparisons, indent=2))
    for row in summary:
        typer.echo(
            f"{row['condition']:>9} H={row['horizon']:<3} {row['variant']:<12} n={row['n']:<3} "
            f"valid={row['valid']} satisfied={row['fraction_satisfied']} false_reliance={row['false_reliance_items']} "
            f"verify={row['verify_calls']} calls={row['tool_calls']}"
        )
    typer.echo(f"wrote {folder}/runs.csv, summary.csv, comparisons.json")


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    app()
