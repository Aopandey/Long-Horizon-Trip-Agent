"""One planning episode: the model acts through tools until it finishes or runs out of budget.

The message history is append-only in every variant: nothing already sent is edited or removed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from trip_agent.agent.llm.base import LLMClient
from trip_agent.agent.policies import StandardPolicy
from trip_agent.agent.prompts import NUDGE
from trip_agent.env.models import Itinerary
from trip_agent.tools.environment import TOOL_SPECS, TripEnvironment


@dataclass
class EpisodeLimits:
    max_turns: int = 200
    max_idle_turns: int = 2
    context_limit: int | None = None
    keep_thinking_in_history: bool = False


@dataclass
class EpisodeResult:
    itinerary: Itinerary
    stop_reason: str
    turns: int
    tool_calls: int
    verify_calls: int
    malformed_tool_calls: int
    prompt_tokens: int
    completion_tokens: int
    llm_latency_ms: float
    wall_ms: float
    finish_notes: str | None
    unresolved: list[str]
    events: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)


def run_episode(
    env: TripEnvironment,
    policy: StandardPolicy,
    llm: LLMClient,
    limits: EpisodeLimits | None = None,
    *,
    seed: int | None = None,
) -> EpisodeResult:
    limits = limits or EpisodeLimits()
    started = time.perf_counter()
    messages: list[dict] = [
        {"role": "system", "content": policy.system_prompt()},
        {"role": "user", "content": env.task_brief()},
    ]
    turns = idle = malformed = prompt_tokens = completion_tokens = 0
    llm_ms = 0.0

    while not env.done:
        if turns >= limits.max_turns:
            env._stop("max_turns")
            break
        turns += 1
        response = llm.chat(messages, TOOL_SPECS, seed=seed)
        prompt_tokens += response.prompt_tokens
        completion_tokens += response.completion_tokens
        llm_ms += response.latency_ms
        env.events.append(
            {
                "type": "llm",
                "turn": turns,
                "content": response.content[:4000],
                "thinking": (response.thinking or "")[:8000] or None,
                "tool_calls": [
                    {"name": c.name, "arguments": c.arguments, "parse_error": c.parse_error}
                    for c in response.tool_calls
                ],
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "latency_ms": round(response.latency_ms, 1),
            }
        )
        if limits.context_limit and response.prompt_tokens > 0.9 * limits.context_limit:
            env.events.append({"type": "context_near_limit", "turn": turns, "prompt_tokens": response.prompt_tokens})
        messages.append(response.assistant_message(limits.keep_thinking_in_history))

        if not response.tool_calls:
            idle += 1
            if idle > limits.max_idle_turns:
                env._stop("no_tool_calls")
                break
            messages.append({"role": "user", "content": NUDGE})
            continue
        idle = 0

        for call in response.tool_calls:
            if env.done:
                break
            if call.parse_error or not call.name:
                malformed += 1
            result = env.call(call.name, call.arguments)
            if call.parse_error:
                result["warning"] = call.parse_error
            note = policy.after_tool(env, call.name, call.arguments, result)
            content = json.dumps(result, default=str)
            if note:
                content += "\n" + note
            messages.append({"role": "tool", "tool_name": call.name, "content": content})

    policy.finalize(env)
    return EpisodeResult(
        itinerary=env.state.itinerary(),
        stop_reason=env.stop_reason or "unknown",
        turns=turns,
        tool_calls=env.budget.used,
        verify_calls=env.verify_calls,
        malformed_tool_calls=malformed,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        llm_latency_ms=round(llm_ms, 1),
        wall_ms=round((time.perf_counter() - started) * 1000, 1),
        finish_notes=env.finish_notes,
        unresolved=env.unresolved,
        events=env.events,
        messages=messages,
    )
