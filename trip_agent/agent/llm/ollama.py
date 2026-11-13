"""Ollama /api/chat adapter (default model: qwen3.5:9b)."""

from __future__ import annotations

import json
import re
import time

import httpx
from pydantic import BaseModel

from trip_agent.agent.llm.base import LLMResponse, ToolCall

_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


class OllamaSettings(BaseModel):
    model: str = "qwen3.5:9b"
    host: str = "http://localhost:11434"
    temperature: float | None = 0.6
    top_p: float | None = None
    num_ctx: int = 32768
    num_predict: int | None = 2048
    think: bool | None = False
    keep_alive: str | None = "30m"
    timeout_s: float = 600.0
    max_retries: int = 3


class OllamaError(RuntimeError):
    pass


def _parse_arguments(raw) -> tuple[dict, str | None]:
    if isinstance(raw, dict):
        return raw, None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw) if raw.strip() else {}
            return (parsed, None) if isinstance(parsed, dict) else ({}, "arguments are not a JSON object")
        except json.JSONDecodeError as exc:
            return {}, f"invalid JSON arguments: {exc}"
    return {}, "missing arguments"


def parse_tool_calls(message: dict) -> tuple[list[ToolCall], str]:
    """Native tool calls, or <tool_call>{...}</tool_call> blocks left in the text by small models."""
    calls: list[ToolCall] = []
    for raw in message.get("tool_calls") or []:
        fn = raw.get("function", raw)
        args, err = _parse_arguments(fn.get("arguments"))
        calls.append(ToolCall(name=str(fn.get("name", "")), arguments=args, id=raw.get("id"), parse_error=err))
    content = message.get("content") or ""
    if not calls and "<tool_call>" in content:
        for match in _TOOL_CALL_TAG.finditer(content):
            try:
                blob = json.loads(match.group(1))
            except json.JSONDecodeError as exc:
                calls.append(ToolCall(name="", arguments={}, parse_error=f"invalid tool_call JSON: {exc}"))
                continue
            args, err = _parse_arguments(blob.get("arguments", blob.get("parameters", {})))
            calls.append(ToolCall(name=str(blob.get("name", "")), arguments=args, parse_error=err))
        content = _TOOL_CALL_TAG.sub("", content).strip()
    return calls, content


class OllamaClient:
    def __init__(self, settings: OllamaSettings | None = None, transport: httpx.BaseTransport | None = None):
        self.settings = settings or OllamaSettings()
        self.model = self.settings.model
        self._http = httpx.Client(base_url=self.settings.host, timeout=self.settings.timeout_s, transport=transport)

    def _payload(self, messages: list[dict], tools: list[dict], seed: int | None) -> dict:
        s = self.settings
        options = {"num_ctx": s.num_ctx}
        for name in ("temperature", "top_p", "num_predict"):
            if getattr(s, name) is not None:
                options[name] = getattr(s, name)
        if seed is not None:
            options["seed"] = seed
        payload = {"model": s.model, "messages": messages, "tools": tools, "stream": False, "options": options}
        if s.think is not None:
            payload["think"] = s.think
        if s.keep_alive is not None:
            payload["keep_alive"] = s.keep_alive
        return payload

    def chat(self, messages: list[dict], tools: list[dict], *, seed: int | None = None) -> LLMResponse:
        payload = self._payload(messages, tools, seed)
        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries):
            started = time.perf_counter()
            try:
                response = self._http.post("/api/chat", json=payload)
            except httpx.TransportError as exc:
                last_error = exc
                time.sleep(min(2**attempt, 10))
                continue
            if response.status_code >= 500:
                last_error = OllamaError(f"Ollama {response.status_code}: {response.text[:300]}")
                time.sleep(min(2**attempt, 10))
                continue
            if response.status_code >= 400:  # bad request: retrying will not help
                raise OllamaError(f"Ollama {response.status_code}: {response.text[:500]}")
            data = response.json()
            break
        else:
            raise OllamaError(f"Ollama request failed after {self.settings.max_retries} attempts: {last_error}")
        message = data.get("message") or {}
        calls, content = parse_tool_calls(message)
        return LLMResponse(
            content=content,
            tool_calls=calls,
            thinking=message.get("thinking"),
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
            latency_ms=(time.perf_counter() - started) * 1000,
            raw={k: v for k, v in data.items() if k != "message"},
        )

    def model_info(self) -> dict:
        """Capabilities of the configured model (raises if it is not pulled)."""
        response = self._http.post("/api/show", json={"model": self.settings.model})
        if response.status_code == 404:
            raise OllamaError(f"model {self.settings.model!r} is not available; run: ollama pull {self.settings.model}")
        response.raise_for_status()
        return response.json()
