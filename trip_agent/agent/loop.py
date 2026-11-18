"""One planning episode: the model acts through tools until it finishes or runs out of budget.

Context management (the same for every variant): tool results are sent as compact text, and only
the most recent turns are kept in full. Older turns keep the tool calls the model made, drop its
narration, and shrink each tool result to one line. Turns are shortened in blocks, so the start of
the prompt stays stable between blocks and the model server can reuse its cache.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from trip_agent.agent.llm.base import LLMClient
from trip_agent.agent.policies import StandardPolicy
from trip_agent.agent.prompts import NUDGE
from trip_agent.env.models import Itinerary
from trip_agent.tools.environment import TOOL_SPECS, TripEnvironment
from trip_agent.tools.render import brief, render


@dataclass
class EpisodeLimits:
    max_turns: int = 200
    max_idle_turns: int = 2
    context_limit: int | None = None
    keep_thinking_in_history: bool = False
    recent_turns: int | None = 6  # turns kept in full; None keeps the whole history in full


@dataclass
class _Turn:
    full: list[dict] = field(default_factory=list)
    brief: list[dict] = field(default_factory=list)

    def add(self, full: dict, short: dict | None = None) -> None:
        self.full.append(full)
        self.brief.append(short if short is not None else full)


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
    head: list[dict] = [
        {"role": "system", "content": policy.system_prompt()},
        {"role": "user", "content": env.task_brief()},
    ]
    history: list[_Turn] = []
    shortened = 0  # history[:shortened] is sent in brief form

    def prompt() -> list[dict]:
        messages = list(head)
        for i, turn in enumerate(history):
            messages += turn.brief if i < shortened else turn.full
        return messages

    turns = idle = malformed = prompt_tokens = completion_tokens = 0
    llm_ms = 0.0

    while not env.done:
        if turns >= limits.max_turns:
            env._stop("max_turns")
            break
        turns += 1
        window = limits.recent_turns
        if window and len(history) - shortened >= 2 * window:
            shortened = len(history) - window
        response = llm.chat(prompt(), TOOL_SPECS, seed=seed)
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
        turn = _Turn()
        history.append(turn)
        assistant = response.assistant_message(limits.keep_thinking_in_history)
        turn.add(assistant, {k: v for k, v in assistant.items() if k != "thinking"} | {"content": ""})

        if not response.tool_calls:
            idle += 1
            if idle > limits.max_idle_turns:
                env._stop("no_tool_calls")
                break
            turn.add({"role": "user", "content": NUDGE})
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
            turn.add(
                {"role": "tool", "tool_name": call.name, "content": render(call.name, result, note)},
                {"role": "tool", "tool_name": call.name, "content": brief(call.name, result)},
            )

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
        messages=head + [m for turn in history for m in turn.full],
    )
