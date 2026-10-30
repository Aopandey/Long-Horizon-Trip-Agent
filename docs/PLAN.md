# Implementation and tuning plan

**Status:** Proposed for team review. Nothing here is final until the team agrees on it. Open items are listed at the end.

## Context

This plan turns the [project proposal](../README.md) into a buildable system. The repo currently holds only documents, so the plan covers the code layout, how the three agent variants differ, how we tune the agent, and how we run the experiments on the school server.

Research question: how does misinformation in tool responses affect itinerary validity as the planning horizon grows, and do targeted verification and dependency-aware recovery reduce cascading failures?

Fixed by the proposal:
- Synthetic Boston catalog: attractions, restaurants, hotels, prices, hours, visit durations, travel times.
- 40 evaluation scenarios, 10 each at horizons 5, 10, 20, 30 dependent commitments.
- Clean and corrupted conditions. Corrupted means 20% of eligible opening-hour, price and travel-time facts are altered in search responses.
- 3 variants of the same agent: standard, verification, verification + recovery.
- 3 repeats. 40 x 2 x 3 x 3 = 720 runs.
- Hard separation: a trusted store, a corruption layer in front of search tools, a deterministic checker that sees only the agent's observed or verified facts, and an independent evaluator that uses the truth.

Proposed decisions (the proposal leaves these open):
- **"Training" means tuning without weight updates.** We use a pretrained LLM. Prompts, the verification policy and the recovery logic are tuned on separate dev scenarios only.
- **Hosted model API: Claude Haiku 5.5 (`claude-haiku-5-5`) or Gemini.** The team is choosing between the two. We pick one on the dev set, freeze it, and use that single model for all three variants and all 720 runs, as the proposal requires. No open-weight model and no GPU inference. The school server runs the experiment batch (see Server workflow).
- **Python with a custom agent loop**, no heavy agent framework.

## Architecture

A custom loop over each provider's native tool calling keeps budgets, logging and the three variants fully under our control. The variants differ only by policy and state, never by model or tools.

```
trip_agent/
  env/        models.py (pydantic), catalog.py, generator.py (seeded Boston catalog),
              travel.py (travel-time matrix), scenario.py (scenario + hidden reference itinerary)
  corruption/ corrupt.py (seeded, per-scenario, variant-independent)
  tools/      search_*, get_travel_time, verify_fact, add/revise/remove_item, check_itinerary, budget.py
  checker/    consistency.py (FactView interface: BeliefView vs TruthView)
  agent/      llm/{base,anthropic_client,gemini_client}.py, loop.py, state.py (facts, commitments, dependency graph, checkpoints),
              variants/{standard,verification,recovery}.py, prompts/
  eval/       evaluator.py, metrics.py, logschema.py, stats.py (paired bootstrap CIs)
  runner/     run_experiment.py, configs/*.yaml, launch scripts
data/dev/  data/eval/   frozen JSON scenarios (committed, with content hash)
tests/      docs/PLAN.md
```

Design rules:
1. **One checker, two views.** `consistency.py` takes a `FactView`. The agent's `check_itinerary` tool passes a `BeliefView` (observed or verified facts only). The evaluator passes a `TruthView`. Same logic, no divergence, and the truth never leaks into planning.
2. **Truth isolation.** Only `verify_fact` and the evaluator may import the trusted store. An import-boundary test makes sure the agent package cannot touch it.
3. **Reproducible corruption.** A fact is corrupted iff `hash(scenario_id, fact_key, seed) < 0.20` among eligible fact types. It uses the same perturbation every time (for example close time +2h, price x0.7, travel time /3), so every variant sees identical lies and a fact never changes mid-run.
4. **Feasible by construction.** Build the hidden reference itinerary first, then derive the constraints from it: budget = reference cost + a fixed slack %, required activities from the reference, a fixed number of schedule-slack minutes per day. This keeps constraint density and slack roughly constant across horizons. Horizon H is the number of required commitments linked by time, location or remaining budget. Day count scales with H.
5. **Verification shares the budget.** `ToolBudget` counts every call (search, verify, edit). The total limit scales with horizon and is calibrated on dev.
6. **Recovery is scaffold-managed, not left to the LLM.** In the recovery variant, `add_item` requires a `based_on` list of fact ids. `state.py` builds the fact-to-commitment dependency graph and saves a checkpoint before each commitment. When `verify_fact` or the checker exposes a mismatch, the scaffold invalidates the dependents, rolls back to the checkpoint, and re-prompts only for the affected span. At most 3 repair attempts per conflict, capped by the total budget. Rollback is state-level only: the message history stays append-only (rule 10).
7. **Variants.**
   - Standard: tools plus a plain system prompt.
   - Verification: same, plus a verification policy with ranked fact priority (hotel cost, required activities, tight-slack travel and hours) and ordinary retry or replan on conflict.
   - Recovery: verification policy plus the dependency graph, checkpoints and targeted repair.
8. **Logging.** Every run writes JSONL: each tool call, a corrupted-or-not flag stored out-of-band for the evaluator only, commitment order, tokens, latency. The metrics need this: first-error position, propagation, reliance on false information, recovery success.
9. **LLM client.** One `LLMClient` interface (messages and tool schemas in; text, tool calls, token usage and stop reason out) with a native adapter per provider, each using that provider's official SDK. No OpenAI-compatibility shim. Provider quirks stay inside the adapters. The Claude Haiku 5.5 adapter must:
   - omit `temperature`, `top_p` and `top_k` (non-default values return a 400);
   - set `output_config.effort` explicitly (the default is `medium`);
   - read response blocks by `type`, and pass `thinking` blocks back unmodified with tool results;
   - leave room in `max_tokens`, because thinking counts toward it;
   - use `tool_choice: auto` (a forced tool call skips thinking);
   - treat `stop_reason: "refusal"` as a failed step and log it (there is no server-side fallback).
10. **Append-only history.** On Haiku 5.5 a thinking block is only valid while everything before it is unchanged, and editing earlier turns, `system` or `tools` and then replaying thinking blocks returns a 400 on accounts created on or after 2026-08-31. So in every variant the history is never edited or truncated, and `system` and the tool list stay fixed within a run. Recovery appends a message naming the corrected fact and the invalidated commitments instead of rewriting the past. Using the same mechanism in all three variants also keeps the comparison fair.

## Metrics and statistics

Fully valid itinerary rate; % constraints satisfied; reliance on false information; first-error position and propagation; recovery success rate; verification calls, total calls, latency, token cost. Paired comparisons by scenario, bootstrap confidence intervals, McNemar on validity.

## Tuning protocol

1. Generate **dev scenarios** (about 40, same four horizons, separate seeds) and the **eval scenarios** (40). Hash and freeze both. Never read the eval set during tuning.
2. On dev only, tune: system prompts, verification priority weights, tool-call budget per horizon, repair limit, and the model's effort level. This is also where we choose between Haiku and Gemini (see Model choice and API notes). Temperature is not a tuning knob: Haiku 5.5 rejects it, so run-to-run variation is measured with the 3 repeats instead. Log each config and result in `runs/dev/` so choices are traceable.
3. Sanity targets on dev before freezing: the clean-condition standard agent solves most H=5 scenarios (the pipeline works); the corrupted condition measurably hurts the standard agent at larger H (the effect exists to study); the recovery logic does not loop.
4. Freeze: tag `eval-freeze` with config, prompts, model name and scenario hash. Then run the 720 evaluation runs once. Report anything that changes afterward as a deviation.

## Model choice and API notes

Candidates: Claude Haiku 5.5 (`claude-haiku-5-5`) and Gemini. One of them is used for everything, so the choice is made once, on dev data, before the freeze.

Selection criteria, measured on dev scenarios with identical prompts and tools: tool-call reliability (valid calls, few malformed or looping turns), fully valid itinerary rate in the clean condition at H=5 and H=10, cost per run, latency, and rate-limit headroom at the concurrency we need. Pick the cheapest model that clears the clean-condition sanity bar; the experiment needs a model that can do the task when information is correct, so that the effect of misinformation is what we measure.

Claude Haiku 5.5 notes:
- Price at the time of writing: $0.10 per million input tokens and $0.50 per million output tokens for prompts up to 100K tokens, and $0.50 / $2.50 above that. Cache reads cost 0.1x the input rate. Confirm current prices before budgeting.
- The whole history is resent on every tool call, so cost grows with horizon. Long runs at H=30 may cross 100K tokens and switch rate cards. Use prompt caching on the fixed prefix (system prompt, tool schemas, catalog-independent instructions) and keep that prefix byte-stable.
- Estimate total cost as (input tokens + output tokens per run, from dev) x 720, per rate card. Measure it on dev before the freeze, then give the runner a hard spend cap so a loop cannot burn the budget.
- Adapter requirements are listed in design rules 9 and 10.

Gemini notes: written when needed, against Google's official SDK and docs. Confirm the exact model ID, tool-calling format and any sampling-parameter limits from Google's documentation first. The `LLMClient` interface keeps this isolated to one adapter file.

Secrets: API keys live in environment variables or a git-ignored `.env`. Commit a `.env.example` with placeholder names only. Never put a key in a config file, log, or commit.

## Server workflow

The model runs in the provider's cloud, so the school server is a place to run the long experiment batch reliably, not to host a model.

1. The server pulls the repo, builds a venv, and loads API keys from its environment. It needs outbound HTTPS to the chosen provider.
2. `runner/run_experiment.py` reads a YAML config (variants, conditions, scenarios, repeats, seeds, concurrency, spend cap) and runs many agents concurrently, with retry and backoff on rate-limit and server errors. Runs are resumable: each (scenario, condition, variant, repeat) writes its own log and is skipped when done. This also lets teammates split the 720 runs by scenario shard if the server is unavailable.
3. Results sync back as small JSONL and CSV into `runs/` (large files git-ignored, summary tables committed).

## Phases

0. **Setup.** Repo scaffold, `pyproject.toml`, lint, pytest, docs. Create API keys, confirm who pays and what rate limits apply, and confirm server access.
1. **Environment.** Models, catalog generator, travel matrix. Tests for data validity.
2. **Scenarios.** Reference itinerary builder, constraint derivation, horizon scaling. Dev and eval sets generated and frozen. Tests that every scenario is feasible.
3. **Tools, corruption, checker.** Search tools, verify tool, budget, corruption layer, `FactView` checker. Tests: determinism, 20% rate, identical across variants, truth isolation.
4. **Agent.** LLM client, loop, standard variant. Smoke run on H=5 clean.
5. **Variants.** Verification policy, then dependency graph, checkpoints and repair. Unit tests for invalidation and the 3-attempt cap.
6. **Evaluator, metrics, stats.** Independent scorer, JSONL schema, bootstrap CIs. Validate against hand-built itineraries.
7. **Dev tuning** (choose Haiku or Gemini, tune effort and prompts), then **freeze** and the 720-run evaluation on the server.
8. **Analysis and report.** Tables, plots by horizon and condition, error-propagation case studies.

Suggested work tracks (the team assigns owners):
- A: environment and scenarios.
- B: tools, corruption and checker.
- C: LLM client, agent and variants.
- D: evaluator, metrics, runner and server.

## Open items

- Ask the school or TA: does the server allow outbound HTTPS to the model provider, job scheduler (Slurm?) and time limits, storage quota, Python version, and whether sessions persist for long batches.
- API access: who creates and pays for the keys, whether the school provides credits, and the rate-limit tier for the chosen provider.
- Haiku vs Gemini is undecided; it is settled on dev data before the freeze.
- Whether our Anthropic accounts were created on or after 2026-08-31 (affects how strictly the append-only history is enforced; we follow it either way).
- The proposal gives no deadline, so the phases have no dates yet.
