"""Deterministic stand-in for an LLM, for tests and dry runs of the pipeline."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from trip_agent.agent.llm.base import LLMResponse, ToolCall


def call(name: str, **arguments) -> ToolCall:
    return ToolCall(name=name, arguments=arguments)


class ScriptedClient:
    """Replays a fixed list of turns, or asks a policy function for the next turn.

    Each turn is a list of ToolCall (or a plain string for a text-only reply). After the script
    runs out it replies with a finish call so an episode always ends.
    """

    model = "scripted"

    def __init__(self, turns: Iterable[list[ToolCall] | str] | None = None, policy: Callable | None = None):
        self._turns = list(turns or [])
        self._policy = policy
        self.requests: list[list[dict]] = []

    def chat(self, messages: list[dict], tools: list[dict], *, seed: int | None = None) -> LLMResponse:
        self.requests.append(list(messages))
        if self._policy is not None:
            turn = self._policy(messages)
        elif self._turns:
            turn = self._turns.pop(0)
        else:
            turn = [call("finish", notes="script finished")]
        if isinstance(turn, str):
            return LLMResponse(content=turn, prompt_tokens=10, completion_tokens=5)
        return LLMResponse(content="", tool_calls=list(turn), prompt_tokens=10, completion_tokens=5)
