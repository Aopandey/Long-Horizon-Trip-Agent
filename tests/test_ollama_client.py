import json
import os

import httpx
import pytest

from trip_agent.agent.llm.ollama import OllamaClient, OllamaError, OllamaSettings, parse_tool_calls


def _client(handler):
    return OllamaClient(OllamaSettings(model="qwen3.5:9b", max_retries=2), transport=httpx.MockTransport(handler))


def test_request_payload_and_response_parsing():
    seen = {}

    def handler(request: httpx.Request):
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "thinking": "plan first",
                    "tool_calls": [{"function": {"name": "search_places", "arguments": {"kind": "hotel"}}}],
                },
                "prompt_eval_count": 120,
                "eval_count": 15,
                "done": True,
            },
        )

    response = _client(handler).chat([{"role": "user", "content": "hi"}], [{"type": "function"}], seed=7)
    assert seen["model"] == "qwen3.5:9b" and seen["stream"] is False and seen["think"] is False
    assert seen["options"]["seed"] == 7 and seen["options"]["num_ctx"] == 32768
    assert response.tool_calls[0].name == "search_places" and response.tool_calls[0].arguments == {"kind": "hotel"}
    assert response.prompt_tokens == 120 and response.completion_tokens == 15 and response.thinking == "plan first"
    assert response.assistant_message() == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "search_places", "arguments": {"kind": "hotel"}}}],
    }


def test_text_tool_calls_are_recovered():
    calls, content = parse_tool_calls(
        {"content": 'Sure.\n<tool_call>{"name": "view_itinerary", "arguments": {}}</tool_call>'}
    )
    assert [c.name for c in calls] == ["view_itinerary"] and content == "Sure."
    calls, _ = parse_tool_calls({"tool_calls": [{"function": {"name": "x", "arguments": "{bad"}}]})
    assert calls[0].parse_error


def test_server_errors_are_retried_and_bad_requests_are_not():
    attempts = {"n": 0}

    def flaky(request):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(503, text="loading")
        return httpx.Response(200, json={"message": {"content": "ok"}})

    assert _client(flaky).chat([], []).content == "ok" and attempts["n"] == 2
    with pytest.raises(OllamaError, match="400"):
        _client(lambda r: httpx.Response(400, text="bad")).chat([], [])


@pytest.mark.ollama
def test_live_model_answers_with_a_tool_call():
    """Needs Ollama running with $TRIP_AGENT_MODEL (default qwen3.5:9b). Run: pytest -m ollama"""
    client = OllamaClient(OllamaSettings(model=os.environ.get("TRIP_AGENT_MODEL", "qwen3.5:9b"), num_ctx=4096))
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Current time",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    response = client.chat([{"role": "user", "content": "What time is it? Use the tool."}], tools, seed=0)
    assert response.tool_calls and response.tool_calls[0].name == "get_time"
