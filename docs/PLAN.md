# Implementation and tuning plan

**Status:** The core implementation, the dataset and Supabase are done; dev tuning has started. Teammate setup is in [INSTRUCTIONS.md](../INSTRUCTIONS.md), the data format in [DATA_FORMAT.md](DATA_FORMAT.md). Open items are listed at the end.

## Context

Research question: how does misinformation in tool responses affect itinerary validity as the planning horizon grows, and do targeted verification and dependency-aware recovery reduce cascading failures?

Fixed by the proposal:
- Synthetic Boston catalog: attractions, restaurants, hotels, prices, hours, visit durations, travel times.
- 40 evaluation scenarios, 10 each at horizons 5, 10, 20, 30 dependent commitments.
- Clean and corrupted conditions. Corrupted means 20% of eligible opening-hour, price and travel-time facts are altered in search responses.
- 3 variants of the same agent: standard, verification, verification + recovery.
- 3 repeats. 40 x 2 x 3 x 3 = 720 runs.
- Hard separation: a trusted store, a corruption layer in front of search tools, a deterministic checker that sees only the agent's observed or verified facts, and an independent evaluator that uses the truth.

Decisions:
- **Model: Qwen3.5 9B (`qwen3.5:9b`) served by Ollama**, on a laptop or the school server. One model for all three variants and all 720 runs.
- **"Training" means tuning without weight updates.** Prompts, the verification policy, the recovery logic, the tool budget and the sampling settings are tuned on dev scenarios only.
- **Storage: Supabase (Postgres)** for the dataset, experiments, runs, event logs and scores. A local JSON backend does the same for offline work and tests.
- **Python with a custom agent loop**, no agent framework.
- **Synthetic data comes from the team**, validated by `trip-agent validate` before use.

## Architecture

```
trip_agent/
  env/         models.py (pydantic data model), facts.py (fact keys), loader.py (load + validate), example.py
  corruption/  corrupt.py: seeded, exact-rate, per-scenario misinformation
  checker/     consistency.py: one checker, BeliefView (agent) and TruthView (evaluator)
  tools/       environment.py: 10 tools, shared tool-call budget, corruption applied to search results
  state.py     observed/verified facts, itinerary, commitment history, checkpoints, dependency graph
  agent/       llm/ (ollama.py, scripted.py), prompts.py, policies.py (3 variants), loop.py (episode)
  eval/        records.py, evaluator.py (independent scoring), stats.py (bootstrap, McNemar)
  storage/     local.py, supabase_store.py (schema in supabase/migrations/)
  runner/      experiment.py: grid, paired seeds, resumable runs, concurrency
  cli.py       trip-agent validate | import-data | check-llm | smoke | run | evaluate | report
configs/       example.yaml, dev.yaml, eval.yaml
```

Design rules:
1. **One checker, two views.** The agent's `check_itinerary` tool checks with a `BeliefView` (only facts it observed or verified). The evaluator checks with a `TruthView`. Same code, so the two can never disagree about a constraint, and the belief view never reveals a hidden value.
2. **Truth isolation.** Only `verify_fact` and the evaluator read correct values. A test makes sure the agent package does not import storage or the evaluator.
3. **Reproducible corruption.** Exactly round(rate x N) eligible facts are altered, chosen by a stable hash of (scenario, seed, fact key). Each is perturbed the same way every time, in the "optimistic" direction of the proposal's examples: later closing, lower prices, shorter travel. Every variant sees identical lies, and a fact never changes mid-run.
4. **Feasible by construction.** Each scenario ships a hidden reference itinerary. `trip-agent validate` checks that it satisfies every constraint under the true facts.
5. **Verification shares the budget.** Every tool call counts, including `verify_fact` and `check_itinerary`. The limit is `base + per_commitment x horizon` (default 20 + 5H), tuned on dev.
6. **Dependencies are computed by the scaffold, not the model.** An item depends on its own hours and price and on the travel legs to its neighbors on that day. The hotel price belongs to the hotel choice. A 9B model cannot be trusted to list its own dependencies.
7. **Variants differ only in the system prompt and in what the scaffold adds after a tool call.**
   - Standard: base prompt, nothing added.
   - Verification: adds a verification policy and flags unverified facts that a new commitment relies on with little slack.
   - Recovery: the same, plus targeted repair. When a corrected fact or a failed check breaks items, it checkpoints, removes only the broken items, names the later same-day items that depend on them, and caps repairs at 3 per conflict. At the end it submits the best checked checkpoint if planning ended in a worse state.
8. **Bounded context.** Tool results reach the model as compact text, not JSON. Only the last 6 model turns are kept in full (`history_window` in the config); older turns keep the tool calls the model made, drop its narration, and shrink each result to one line. Turns are shortened in blocks so the model server can reuse its prompt cache. Without this, the prompt grew about 390 tokens per tool call and 4- and 6-day trips would overflow a 32K context. Repairs change the scaffold state and are reported in the tool result. All three variants use the same mechanism.
9. **Equal help for everyone.** Search matching by word, hints on empty results, and notices on repeated identical lookups live in the tool layer, so every variant gets them.
10. **Everything is logged.** Each run stores every LLM turn (tokens, latency, thinking text), every tool call and result, the commitment history with the facts each commitment relied on, and the corruption plan.

## Metrics

Computed by `eval/evaluator.py` after a run, from the stored record:
- **valid**: every constraint passes under the true facts.
- **fraction_satisfied**: share of constraint checks that pass.
- **false_reliance_items**: final items that rely on a corrupted fact the agent saw but never verified.
- **first_error_position**: index of the first commitment that relied on false information or broke a true constraint.
- **errors_after_first**: later commitments that were also errors (propagation).
- **conflicts_detected / resolved, recovery_success_rate**: corrupted facts the agent verified, and how many no longer cause a violation at the end.
- Cost: tool calls, verify calls, turns, prompt and completion tokens, latency.

`trip-agent report` writes per-run and summary CSVs. Each variant is compared with the standard agent on paired runs (same scenario, condition and repeat), with a bootstrap CI and an exact McNemar test on validity.

## Tuning protocol

1. Get the dataset: a **dev** split (about 40 scenarios, the same four horizons) and the **eval** split (40). Validate both, import them, and record the dataset hash.
2. On dev only (`configs/dev.yaml`), tune: system prompts, verification slack thresholds, the repair limit, the tool budget, temperature, `num_ctx`, and whether thinking is on. Give each attempt its own experiment name so results stay traceable.
3. Sanity targets on dev before freezing: the clean-condition standard agent solves most H=5 scenarios; the corrupted condition measurably hurts the standard agent at larger H; the recovery logic does not loop.
4. Freeze: copy the chosen settings into `configs/eval.yaml`, commit, and tag `eval-freeze`. Run `trip-agent run configs/eval.yaml --allow-eval` once. Report anything that changes afterward as a deviation.

## Model notes (Qwen3.5 9B on Ollama)

- `qwen3.5:9b` is a 7.6 GB download (9.7B parameters, NVFP4 weights, with a vision projector). On a Mac it used 9.0 GB of GPU memory at a 32K context. Pull it once with `ollama pull qwen3.5:9b`; `trip-agent check-llm` confirms it is available and supports tools.
- **Context:** Ollama's default context window is small; the config sets `num_ctx: 32768`. With the bounded context (design rule 8), replaying the first 26 dev runs cut the largest prompt from 28K to about 14K tokens, and a 6-day trip is projected at about 21K tokens with a budget of 20 + 5H, or 29K with 20 + 8H. If `context_near_limit` events appear, lower `history_window` or raise `num_ctx`.
- **Thinking:** off by default for speed. Try `think: true` on dev; it may improve planning at a large time cost.
- **Sampling:** temperature and per-run seeds are passed to Ollama. The seed depends on (scenario, condition, repeat) but not the variant, so variants are paired.
- **Throughput:** runs go through one Ollama server. To run several at once, set `OLLAMA_NUM_PARALLEL` on the server and raise `concurrency` in the config.
- **First live check:** with the tiny example and `qwen3:8b` standing in, the pipeline ran end to end: tool calls parsed, repairs fired, scores computed. The first two runs showed a small model wasting its budget on exact-match searches and repeated lookups, which led to rule 9.

## Server workflow

1. On the school server: install Ollama, `ollama pull qwen3.5:9b`, clone the repo, create the venv, and copy `.env` with `OLLAMA_HOST` and the Supabase keys.
2. Run inside `tmux` or a batch job: `trip-agent run configs/dev.yaml`. Runs are resumable, so an interrupted batch continues where it stopped. Errored runs are retried on the next start.
3. Results land in Supabase, so anyone on the team can look at them while the batch runs (`run_results` view), and `trip-agent report` works from any machine with the keys.

## Phases

0. **Setup (done).** Package, CLI, lint, tests, configs, Supabase schema.
1. **Core system (done).** Data model, corruption, checker, tools, agent loop, three variants, evaluator, stats, storage, runner.
2. **Dataset (done).** 80 scenarios (40 dev, 40 eval) in `data/`, rebuilt by `scripts/generate_synthetic_data.py`; `validate` passes, hash `f3270941decf0d9a`.
3. **Supabase (done).** Schema applied, dataset imported as `main`, round trip verified (same hash from files and from Supabase).
4. **Model (in progress).** `qwen3.5:9b` works on a Mac (9 GB of memory, about 12 minutes per 1-day run); teammates' GPUs are being checked with `check-llm` and `smoke`.
5. **Dev tuning (next).** Each teammate runs a named config (for example `dev-archer-v1`) against Supabase; iterate until the sanity targets hold.
6. **Freeze and evaluate.** Tag, run the 720 eval runs once.
7. **Analysis and report.** Tables and plots by horizon and condition; error-propagation case studies from the event logs.

## Open items

- School server: GPU and memory (plan for at least 10-12 GB of GPU memory, more for longer contexts), whether Ollama can be installed and runs this model's NVFP4 weights on that GPU (run `check-llm` and `smoke` there first), scheduler and time limits, outbound access to Supabase.
- Supabase project: who owns it, and who gets the secret key.
- Metric definitions in `eval/evaluator.py` are provisional; the team should confirm them before the freeze.
- The proposal gives no deadline, so the phases have no dates yet.
