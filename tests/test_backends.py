"""The Claude backend against the real Anthropic SDK, with the HTTP layer mocked."""

from __future__ import annotations

import json

import pytest

from datalens.inference.backends import AnthropicBackend, SamplingConfig
from datalens.prompts import build_prompt

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")  # the HTTP client the Anthropic SDK is built on


def _sse(text: str, stop_reason: str) -> bytes:
    usage = {
        "input_tokens": 12,
        "output_tokens": 1,
        "cache_read_input_tokens": 900,
        "cache_creation_input_tokens": 0,
    }
    message = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": usage,
    }
    events = [
        {"type": "message_start", "message": message},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Count."},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "c2ln"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 40},
        },
        {"type": "message_stop"},
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def _backend(stop_reason: str = "end_turn", **kwargs):
    requests = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        body = _sse("```sql\nSELECT COUNT(*) FROM customers\n```", stop_reason)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    client = anthropic.Anthropic(
        api_key="test", max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
    )
    return AnthropicBackend("claude-test", client=client, **kwargs), requests


def _prompt():
    return build_prompt("CREATE TABLE customers (id INTEGER);", "How many customers are there?")


def test_request_caches_the_schema_and_sends_no_sampling_parameters():
    backend, requests = _backend(effort="high")
    [generation] = backend.generate([_prompt()], SamplingConfig(n=1, temperature=None, max_tokens=16000))

    [request] = requests
    assert request["model"] == "claude-test"
    assert request["max_tokens"] == 16000
    assert request["stream"] is True
    assert request["output_config"] == {"effort": "high"}
    assert "temperature" not in request and "thinking" not in request
    schema, question = request["messages"][0]["content"]
    assert schema["cache_control"] == {"type": "ephemeral"}
    assert "CREATE TABLE customers" in schema["text"]
    assert "How many customers are there?" in question["text"]
    assert "cache_control" not in question

    # Only the text block is the answer; thinking is not part of the completion.
    [completion] = generation.completions
    assert completion.text == "```sql\nSELECT COUNT(*) FROM customers\n```"
    assert completion.output_tokens == 40
    assert completion.finish_reason == "end_turn"
    assert generation.input_tokens == 912
    assert generation.cached_input_tokens == 900


def test_samples_are_separate_requests_and_temperature_is_passed_through():
    backend, requests = _backend()
    [generation] = backend.generate([_prompt()], SamplingConfig(n=3, temperature=0.7))
    assert len(requests) == 3
    assert all(r["temperature"] == 0.7 for r in requests)
    assert len(generation.completions) == 3
    assert generation.input_tokens == 3 * 912


def test_refusal_is_an_empty_completion():
    backend, _ = _backend(stop_reason="refusal")
    [generation] = backend.generate([_prompt()], SamplingConfig(n=1, temperature=None))
    assert generation.completions[0].text == ""
    assert generation.completions[0].finish_reason == "refusal"
