"""Provider-neutral LLM interface.

Messages use Ollama's chat format (role/content/tool_calls/tool_name), which is close to the
OpenAI shape. An adapter for another provider converts to and from this format internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str | None = None
    parse_error: str | None = None


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    raw: dict = field(default_factory=dict)

    def assistant_message(self, include_thinking: bool = False) -> dict:
        message: dict = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            message["tool_calls"] = [{"function": {"name": c.name, "arguments": c.arguments}} for c in self.tool_calls]
        if include_thinking and self.thinking:
            message["thinking"] = self.thinking
        return message


class LLMClient(Protocol):
    model: str

    def chat(self, messages: list[dict], tools: list[dict], *, seed: int | None = None) -> LLMResponse: ...
