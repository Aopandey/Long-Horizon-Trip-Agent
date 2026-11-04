# Five-minute speaker script

## Long-Horizon Trip Planning

**Format:** Exactly 3 slides. Target speaking time: 4:40, with a 20-second buffer in a 5-minute slot.

Delivery cues are reminders and are not part of the spoken script. Rehearse at a natural pace and use the buffer for brief pauses or transitions.

[Companion slides](long-horizon-trip-planning-5-minute-update.pptx)

## Slide 1: Long-Horizon Trip Planning

**Timing:** 0:00 to 1:20 (80 seconds)

**Delivery cue:** Introduce the question, then use the travel-time example to explain the dependency problem.

Hi everyone. Our project is Long-Horizon Trip Planning. We are studying whether an AI agent can keep a trip plan correct when it has to make many connected decisions and some of its information is wrong.

The agent must satisfy requirements such as budget, opening hours, travel time, meals, and required activities. These choices depend on one another. Choosing a hotel affects travel times, and those travel times affect when activities can start.

For example, suppose the agent is told that a museum is fifteen minutes away, but the journey actually takes forty-five minutes. A visit that appeared to finish before closing could now finish too late. The arithmetic can be correct while the plan still fails because a supporting fact is wrong.

This connects to Theme 6's focus on planning horizon. Our question is whether checking important facts and repairing the decisions that depend on them can reduce these failures as the plan gets longer.

## Slide 2: Planning, verification and recovery

**Timing:** 1:20 to 3:00 (100 seconds)

**Delivery cue:** Trace search and verification into the agent, then show how state supports repairs and independent scoring.

We will use a controlled synthetic environment modeled on Boston. It will contain attractions, restaurants, hotels, prices, opening hours, visit durations, and travel times. A separate scenario specifies the user's trip requirements.

The agent can search for information, edit the itinerary, or verify a specific fact. Ordinary search results will sometimes contain deliberately incorrect values. A separate reliable verification tool can retrieve the correct value. Each verification uses the same limited tool-call budget as planning, so the agent must choose what to check.

The agent will keep a structured itinerary and a record of the facts supporting its decisions. A consistency checker uses the information the agent currently knows. It will not automatically reveal every incorrect fact.

When a correction invalidates an earlier choice, the recovery method identifies the affected activities, revises them, and checks the downstream schedule and budget. In the museum example, it might move lunch earlier so the full visit still fits. We will limit repairs to three attempts per conflict.

After the agent submits its plan, an independent evaluator checks it against the correct dataset. That scoring information does not feed back into planning.

## Slide 3: Evaluation plan and next steps

**Timing:** 3:00 to 4:40 (100 seconds)

**Delivery cue:** Explain the three versions, the 720-run calculation, the success measure, and the next steps.

We will compare three versions of the same agent in separate runs. The standard version uses the available tools and may correct mistakes naturally. The second explicitly verifies important facts and uses ordinary retries. The third uses the same verification policy and adds checkpoints and repairs based on dependencies. This lets us separate the benefit of verification from the additional benefit of recovery.

The proposed evaluation has forty feasible scenarios, with ten each requiring five, ten, twenty, or thirty dependent scheduling decisions. Each scenario has a clean condition and a condition with twenty percent of eligible facts corrupted. Every version faces the same facts and resource limits.

With three versions and three repeated runs, the total is seven hundred twenty planned runs. Our main outcome is the percentage of fully valid itineraries. We will also measure constraints satisfied, false facts accepted, error propagation, repair success, tool calls, latency, and API cost.

Our next steps are to agree on the tools, build the dataset and checker, and pilot the evaluation. These are planned experiments. We will use the results to assess whether verification and recovery improve reliability and whether the benefit justifies their cost.

**Buffer:** 4:40 to 5:00.

Source: [Team project proposal](https://github.com/Aopandey/Long-Horizon-Trip-Agent/blob/main/CS%205130%20Project%20Proposal.docx). The travel-time example is illustrative. The study design and run counts describe planned experiments.

