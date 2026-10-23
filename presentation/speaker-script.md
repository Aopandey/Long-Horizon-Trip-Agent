# Speaker script

## Long-Horizon Trip Planning

**Team:** Avinash Pandey, Daniel Zhang, Youssef Yacoub, Naksha

**Length:** 10 slides, approximately 10 minutes.

Read the spoken script aloud. Delivery cues are reminders, not lines to read. The time windows total 10:00; rehearse at a comfortable pace and leave room for short pauses.

[Companion PowerPoint](long-horizon-trip-planning-proposal-with-script.pptx)

| Slide | Time window | Duration |
| --- | --- | --- |
| 1. Long-Horizon Trip Planning | 0:00 to 0:30 | 30 sec |
| 2. Research question | 0:30 to 1:25 | 55 sec |
| 3. One false fact can invalidate later stops | 1:25 to 2:30 | 65 sec |
| 4. Controlled data and known constraints | 2:30 to 3:40 | 70 sec |
| 5. System architecture | 3:40 to 4:45 | 65 sec |
| 6. Verification and recovery | 4:45 to 5:55 | 70 sec |
| 7. Three versions of the same agent | 5:55 to 7:00 | 65 sec |
| 8. Experimental design | 7:00 to 8:05 | 65 sec |
| 9. Evaluation and failure analysis | 8:05 to 9:10 | 65 sec |
| 10. Implementation plan and expected contribution | 9:10 to 10:00 | 50 sec |

## Slide 1: Long-Horizon Trip Planning

**Timing:** 0:00 to 0:30 (30 seconds)

**Delivery cue:** Introduce the project and team. Keep the opening brief.

### What to say

Our project studies whether an AI agent can keep a long plan correct when some of its information is wrong. We use trip planning because choices about hotels, travel, activities, and meals depend on one another. We will compare ordinary planning with explicit verification and recovery. This presentation describes the proposed system and evaluation. The implementation and experiments have not started.

## Slide 2: Research question

**Timing:** 0:30 to 1:25 (55 seconds)

**Delivery cue:** Pause after the main question, then explain what horizon means.

### What to say

The main question is how misinformation affects itinerary validity as planning horizon increases, and whether targeted verification and explicit recovery reduce cascading failures. Theme 6 gives us the horizon focus. Here, horizon means required scheduling commitments linked through time, location, or remaining budget. A tool lookup is not itself another scheduling commitment.

Harder scenarios will contain twenty or thirty dependent commitments. The challenge is to preserve consistency across that chain. One incorrect hotel price can distort later spending, while an incorrect travel time can invalidate several later appointments. We will investigate both the final outcome and the point where the first error begins to spread.

## Slide 3: One false fact can invalidate later stops

**Timing:** 1:25 to 2:30 (65 seconds)

**Delivery cue:** Point to the 15-minute versus 45-minute travel-time cells.

### What to say

This is an illustrative scenario, not an experimental result. Lunch ends at three thirty. A search tool reports a fifteen-minute trip to a museum. The required museum visit lasts sixty minutes, and the museum closes at five. With the reported travel time, the agent plans to arrive at three forty-five and finish at four forty-five. That appears feasible.

The actual journey takes forty-five minutes. Arrival would be four fifteen, so the same visit would finish at five fifteen, after closing. The schedule can look internally consistent while still being wrong because its supporting fact is wrong. A checker that only sees the reported travel time cannot automatically discover the true duration. This motivates a separate verification action and a way to repair choices after the correction.

## Slide 4: Controlled data and known constraints

**Timing:** 2:30 to 3:40 (70 seconds)

**Delivery cue:** Distinguish environment facts from the trip's requirements.

### What to say

We will create a synthetic environment modeled on Boston. The catalog will contain attractions, restaurants, hotels, opening hours, visit durations, prices, and travel-time tables. Scenario inputs will specify dates, group size, budget, required activities, and scheduling restrictions. Each evaluation scenario will include a hidden valid reference itinerary. That establishes feasibility, but the evaluator will accept any itinerary satisfying the actual constraints.

We will use separate development cases to debug the system, keeping evaluation cases separate. Across horizon groups, we will keep constraint density and budget and scheduling slack approximately comparable. Otherwise, a failure on a longer task might reflect tighter constraints rather than a longer chain of decisions. The environment is synthetic and fixed, so we can repeat the same experiment without relying on changing live travel information.

## Slide 5: System architecture

**Timing:** 3:40 to 4:45 (65 seconds)

**Delivery cue:** Trace the search path, the verification path, and the final scoring path.

### What to say

The correct environment supplies facts to two separate paths. Ordinary search passes selected results through the misinformation layer. The verification tool reads correct individual facts directly. The planning agent receives trip constraints and chooses its next tool dynamically. It can search, verify, or edit the itinerary.

Structured state records observed facts and scheduling commitments. Consistency checks and recovery operate on that state. They use the facts the agent currently knows, including any verified corrections. They do not silently receive all the correct answers. The agent eventually submits an itinerary or reports unresolved constraints. Independent final scoring then uses the actual environment and execution logs. There is no scoring feedback into the planning loop. This separation lets us measure whether the agent itself finds and handles incorrect information.

## Slide 6: Verification and recovery

**Timing:** 4:45 to 5:55 (70 seconds)

**Delivery cue:** Connect the repair steps back to the museum example.

### What to say

Verification asks whether a fact is correct. Recovery asks what must change after a fact or decision fails. We will prioritize verification of consequential facts such as hotel prices, facts supporting required activities, and long journeys. Each check consumes the shared tool-call budget, so checking everything has a cost.

The full recovery method additionally records which facts and commitments support later decisions. When a correction arrives, it identifies the affected choices, restores a consistent checkpoint when needed, revises those choices, and checks their consequences. In the museum example, an earlier lunch could preserve a full visit before closing, provided the preceding activities still fit. Recovery must also recheck the budget. We allow up to three repair attempts per conflict and a total call limit, preventing the agent from retrying indefinitely.

## Slide 7: Three versions of the same agent

**Timing:** 5:55 to 7:00 (65 seconds)

**Delivery cue:** Explain that the rows represent experimental versions run separately.

### What to say

We will compare three versions of the same agent in separate runs. The standard version has the available tools without an explicit verification or dependency-based repair policy. It may still check facts or correct mistakes naturally. The verification version explicitly prioritizes important facts and uses ordinary retries or replanning. The full version uses the same verification policy and adds checkpoints, dependency tracking, and targeted repair. The comparison between the second and third versions isolates the contribution of the recovery mechanism more clearly than a baseline-only comparison.

All versions will use the same model, tools, scenarios, and resource limits. These are experimental variants, not three collaborating agents. The study therefore remains a single-agent planning project with supporting code for checking and evaluation.

## Slide 8: Experimental design

**Timing:** 7:00 to 8:05 (65 seconds)

**Delivery cue:** Point to the horizon groups, then explain how the total reaches 720.

### What to say

The evaluation set contains forty feasible base scenarios, with ten at each horizon of five, ten, twenty, and thirty required dependent commitments. Each scenario has a clean condition and a corrupted condition. In the corrupted condition, we change twenty percent of eligible opening-hour, price, or travel-time fact values returned by search tools. The actual environment remains unchanged.

Corrupted facts stay consistent within a scenario and identical across the versions being compared. We run three agent versions and repeat each scenario-condition combination three times. Forty times two times three times three gives seven hundred twenty planned runs. We will compare matched scenarios and log the misinformation each agent actually encounters. Tool calls remain a separate cost measure rather than our definition of horizon.

## Slide 9: Evaluation and failure analysis

**Timing:** 8:05 to 9:10 (65 seconds)

**Delivery cue:** Emphasize fully valid itineraries before discussing the diagnostic metrics.

### What to say

The main outcome is fully valid itinerary rate. An itinerary succeeds only when it satisfies all hard constraints against the correct environment. We will also measure the percentage of constraints satisfied and reliance on false information.

Execution logs will help us locate the first incorrect commitment, identify which later choices inherit that error, and determine whether attempted repairs succeed. Verification calls, total tool calls, latency, and token or API cost will show the price of improved reliability. We will report paired comparisons and uncertainty intervals instead of presenting a few successful examples. We are not claiming that the recovery method will win. The experiment could show that verification prevents most failures, that repair helps mainly at longer horizons, or that its extra cost limits the benefit.

## Slide 10: Implementation plan and expected contribution

**Timing:** 9:10 to 10:00 (50 seconds)

**Delivery cue:** Finish with the research contribution and the tool choices still open.

### What to say

The planned deliverables are the controlled dataset and corruption layer, the planning and verification system, the recovery mechanism, and an independent evaluation report.

The main implementation challenges are detecting plausible false facts, keeping state consistent after edits, limiting repetitive repairs, and preventing hidden scoring data from leaking into planning. Our intended contribution is evidence about how misinformation propagates through dependent decisions and whether targeted verification and repair limit that damage. Models, frameworks, APIs, storage, and supporting tools remain choices for the team to discuss. The immediate next step is to agree on those choices and build the development environment before running the evaluation.

## Source

[Team project proposal](https://github.com/Aopandey/Long-Horizon-Trip-Agent/blob/main/CS%205130%20Project%20Proposal.docx). The museum schedule on slide 3 is an illustrative example, not experimental results. All experiment counts describe the planned study.

