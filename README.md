# Long-Horizon Trip Agent

A CS 5130 research project on AI planning reliability under dependent constraints and deliberately inaccurate tool responses.

**Status:** Proposal and team discussion. Implementation has not started. The team will choose the implementation language, models, frameworks, APIs, and supporting tools together.

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

## Implementation decisions

The team will discuss the language, model/provider, agent framework or custom implementation, storage format, testing tools, and experiment tooling before implementation. No installation or execution instructions are available yet.

A proposed build and tuning plan, for the team to review, is in [docs/PLAN.md](docs/PLAN.md).

All environment information is synthetic and intended for controlled experiments rather than current travel advice.
