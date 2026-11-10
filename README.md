# Long-Horizon Trip Agent

A CS 5130 research project on AI planning reliability under dependent constraints and deliberately inaccurate tool responses.

**Status:** Core implementation built: environment, misinformation layer, checker, tools, three agent variants, evaluator, Supabase storage and experiment runner. Next: the team's synthetic dataset, then dev tuning. See [docs/PLAN.md](docs/PLAN.md).

## Project proposal

[Download the CS 5130 project proposal](CS%205130%20Project%20Proposal.docx).

## Presentation and speaker script

The proposal presentation contains 10 slides with a 10-minute timing plan. The PowerPoint includes the spoken script and delivery cues in its speaker notes.

- [PowerPoint with speaker notes](presentation/long-horizon-trip-planning-proposal-with-script.pptx)
- [Speaker script (PDF)](presentation/speaker-script.pdf)
- [Speaker script (Markdown)](presentation/speaker-script.md)

## Research question

**How does misinformation affect itinerary validity as planning horizon increases, and can targeted verification and explicit recovery reduce cascading failures?**

The project addresses Theme 6's horizon difficulty. Trip planning provides a controlled environment for studying how an early mistake can invalidate later decisions.

## What we plan to build

An AI agent will construct multi-day itineraries subject to dates, group size, budget, opening hours, travel time, meals, hotel location, and required activities.

Some search responses will deliberately contain incorrect facts, such as:
- A museum closing at 5 PM being reported as closing at 7 PM.
- A hotel price being reported below its actual cost.
- A 45-minute journey being reported as taking 15 minutes.

The agent must decide what to check, build a consistent itinerary, and repair affected decisions when it discovers a problem. The planned output is an itinerary with a cost breakdown and verification status, or a report of unresolved constraints.

## Planned workflow

1. **Prepare the environment.** Create a synthetic catalog modeled on Boston with attractions, restaurants, hotels, prices, opening hours, visit durations, and travel times. Keep the correct facts in a trusted store.
2. **Introduce controlled misinformation.** Alter selected facts returned by ordinary search tools. Keep the altered facts consistent throughout a scenario and identical across the agent variants being compared.
3. **Plan and verify.** Let the agent choose search, scheduling, and verification actions. A separate reliable verification tool checks individual facts against the trusted store. Verification consumes the same limited tool-call budget as planning.
4. **Check the current itinerary.** A deterministic consistency checker checks budget, overlaps, opening hours, travel feasibility, and required activities against the agent's currently observed or verified facts. It does not automatically reveal the hidden correct information.
5. **Recover from failures.** The full recovery variant stores checkpoints and dependencies between facts and decisions. A corrected fact triggers reconsideration of the affected activities and their downstream consequences. The proposal allows up to three repair attempts per conflict, with a total tool-call limit.
6. **Evaluate independently.** After the agent finishes, a separate evaluator uses the correct dataset to score the itinerary and analyze the execution logs. Final evaluation does not send answers back into planning.

### What dependency-aware recovery means

If a museum visit must move, the agent identifies which later travel, lunch, and activity choices depend on its timing. It revises those choices and rechecks the affected schedule and total budget.

The planned comparison uses **three versions of the same agent**, evaluated separately. It does not require three agents collaborating in one run.

## Agent variants

| Variant | Planned behavior |
| --- | --- |
| Standard agent | Uses the available tools without an explicit verification or dependency-based repair policy. It may still verify facts or correct mistakes naturally. |
| Verification agent | Explicitly prioritizes checking important facts and uses ordinary retries or replanning. |
| Verification and recovery agent | Uses the same verification policy and additionally tracks dependencies and performs targeted repairs. |

All variants will use the same model, tools, scenarios, and resource limits.

## Planned dataset and experiments

We will create our own controlled synthetic dataset. Separate development cases will support debugging; the evaluation scenarios will be kept separate.

| Setting | Proposal plan |
| --- | --- |
| Evaluation scenarios | 40 feasible scenarios |
| Planning horizons | 5, 10, 20, and 30 required dependent itinerary commitments; 10 scenarios per horizon |
| Information conditions | Clean responses and corrupted responses |
| Corrupted condition | Alter 20% of eligible opening-hour, price, or travel-time fact values |
| Agent variants | 3 |
| Repeated runs | 3 per scenario, condition, and variant |
| Total planned runs | 40 scenarios × 2 conditions × 3 variants × 3 repeats = 720 |

A planning commitment is a scheduling decision linked to earlier choices through time, location, or remaining budget. Tool calls are counted separately from horizon.

Each scenario will have a hidden valid reference itinerary to establish feasibility. The correct environment stays unchanged when search responses are corrupted. Final scoring checks whether the submitted itinerary satisfies the actual constraints; it does not require matching the reference itinerary exactly.

Constraint density and budget and scheduling slack will be kept approximately comparable across horizon groups. Comparisons will pair results from the same scenarios and report uncertainty intervals.

## Evaluation metrics

- Fully valid itinerary rate.
- Percentage of constraints satisfied.
- Reliance on false information.
- First-error position and propagation into later decisions.
- Recovery success rate.
- Verification calls, total tool calls, latency, and token/API cost.

## Team

- Avinash Pandey
- Daniel Zhang
- Youssef Yacoub
- Naksha

## Implementation

| Piece | Choice |
| --- | --- |
| Language | Python 3.11+ |
| Model | Qwen3.5 9B (`qwen3.5:9b`) served by [Ollama](https://ollama.com), on a laptop or the school server |
| Agent | Custom tool-calling loop; the three variants differ only in prompt and scaffold policy |
| Storage | Supabase (Postgres), or local JSON files for offline work |
| Data | Synthetic catalog and scenarios supplied by the team: [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) |

The design, metrics and tuning protocol are in [docs/PLAN.md](docs/PLAN.md).

### Setup

```bash
git clone https://github.com/Aopandey/Long-Horizon-Trip-Agent.git
cd Long-Horizon-Trip-Agent
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,supabase]"
cp .env.example .env          # then fill in OLLAMA_HOST and the Supabase keys
ollama pull qwen3.5:9b        # 7.6 GB download, about 9 GB of memory when loaded
```

On macOS, clone the repo outside iCloud-synced folders (Desktop and Documents). Inside them, iCloud can hide the venv's `.pth` file (`No module named 'trip_agent'`), and with Optimize Mac Storage it can evict source and venv files, which makes Python and git hang while they wait for a download.

### Try it

Running the model on your own computer for dev tuning: follow [INSTRUCTIONS.md](INSTRUCTIONS.md).

```bash
pytest -m "not ollama"                       # unit tests, no model needed
trip-agent check-llm                         # Ollama reachable, model pulled, tools supported
trip-agent example-data examples/tiny        # tiny format demo (not experiment data)
trip-agent validate examples/tiny
trip-agent smoke --variant recovery --condition corrupted   # one live episode, scored
```

### Supabase

1. Create a Supabase project. Put its URL and secret (service-role) key in `.env`.
2. Apply the schema: paste `supabase/migrations/20261105000000_init.sql` into the SQL editor, or run `supabase link` then `supabase db push`.
3. Load data: `trip-agent import-data data/ --backend supabase --dataset main`.

Row-level security is on for every table with no policies, so the public anon key cannot read anything. Only the secret key used by the runner can.

### Run experiments

```bash
trip-agent run configs/dev.yaml              # resumable: completed runs are skipped
trip-agent report configs/dev.yaml           # reports/<name>/runs.csv, summary.csv, comparisons.json
trip-agent run configs/eval.yaml --allow-eval   # once, after freezing settings
```

### Code map

```
trip_agent/env/         data model, fact keys, loading and validation, tiny example
trip_agent/corruption/  seeded misinformation
trip_agent/checker/     one checker for the agent (beliefs) and the evaluator (truth)
trip_agent/tools/       the 10 agent tools and the shared tool-call budget
trip_agent/agent/       Ollama client, prompts, the three variants, episode loop
trip_agent/eval/        run records, independent scoring, statistics
trip_agent/storage/     local and Supabase repositories
trip_agent/runner/      experiment grid and configs
supabase/migrations/    database schema
```

All environment information is synthetic and intended for controlled experiments rather than current travel advice.
