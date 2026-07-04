"""Full-app /v1/messages: non-stream, streaming, tool round-trip, errors."""

import json

import pytest

pytest.importorskip("aiohttp")
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from tests.hermes_cli.claude_max.conftest import ResultMessage  # noqa: E402


async def _app_client(fake_sdk):
    from hermes_cli.claude_max.app import build_app
    client = TestClient(TestServer(build_app()))
    await client.start_server()
    return client


def _text_turn(text, stop="end_turn"):
    return [
        {"type": "message_start", "message": {"id": "m", "model": "x",
            "usage": {"input_tokens": 5, "output_tokens": 0}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": 4}},
        {"type": "message_stop"},
    ]


def _parse_sse(text):
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        data = None
        for ln in block.splitlines():
            if ln.startswith("data:"):
                data = json.loads(ln.split(":", 1)[1].strip())
        if data:
            events.append(data)
    return events


@pytest.mark.asyncio
async def test_non_stream_text(fake_sdk):
    fake_sdk.queue_script(_text_turn("Hello there") + [ResultMessage(result="Hello there")])
    client = await _app_client(fake_sdk)
    try:
        resp = await client.post("/v1/messages", json={
            "model": "claude-opus-4-8",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        })
        assert resp.status == 200
        body = await resp.json()
        assert body["type"] == "message" and body["role"] == "assistant"
        assert body["model"] == "claude-opus-4-8"
        assert body["content"][0]["text"] == "Hello there"
        assert body["stop_reason"] == "end_turn"
        assert body["usage"]["input_tokens"] == 5
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_stream_sse_sequence(fake_sdk):
    fake_sdk.queue_script(_text_turn("streamed") + [ResultMessage(result="streamed")])
    client = await _app_client(fake_sdk)
    try:
        resp = await client.post("/v1/messages", json={
            "model": "opus", "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status == 200
        assert resp.headers["Content-Type"].startswith("text/event-stream")
        text = await resp.text()
        events = _parse_sse(text)
        kinds = [e["type"] for e in events]
        assert kinds[0] == "message_start" and kinds[-1] == "message_stop"
        assert events[0]["message"]["model"] == "opus"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_tool_round_trip_reuses_session(fake_sdk):
    # One script covers the whole turn: tool_use response, then (after the
    # blocked handler is delivered) the continuation.
    fake_sdk.queue_script([
        {"type": "message_start", "message": {"id": "m1", "model": "x", "usage": {"input_tokens": 9}}},
        {"type": "content_block_start", "index": 0, "content_block": {
            "type": "tool_use", "id": "toolu_9", "name": "mcp__hermes__get_weather",
            "input": {"city": "Lisbon"}}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 4}},
        {"type": "message_stop"},
        ("await_tool", "get_weather", {"city": "Lisbon"}),
    ] + _text_turn("It is sunny.") + [ResultMessage(result="It is sunny.")])

    client = await _app_client(fake_sdk)
    try:
        tools = [{"name": "get_weather", "input_schema": {"type": "object"}}]
        # Request 1: model asks for the tool.
        r1 = await client.post("/v1/messages", json={
            "model": "opus", "tools": tools,
            "messages": [{"role": "user", "content": "weather in Lisbon?"}],
        })
        b1 = await r1.json()
        assert b1["stop_reason"] == "tool_use"
        tu = [b for b in b1["content"] if b["type"] == "tool_use"][0]
        assert tu["name"] == "get_weather" and tu["id"] == "toolu_9"

        # Request 2: hermes returns the tool_result → same session continues.
        r2 = await client.post("/v1/messages", json={
            "model": "opus", "tools": tools,
            "messages": [
                {"role": "user", "content": "weather in Lisbon?"},
                {"role": "assistant", "content": b1["content"]},
                {"role": "user", "content": [{"type": "tool_result",
                    "tool_use_id": "toolu_9", "content": "Sunny 22C"}]},
            ],
        })
        b2 = await r2.json()
        assert b2["stop_reason"] == "end_turn"
        assert "sunny" in b2["content"][0]["text"].lower()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_auth_error_maps_to_401(fake_sdk):
    fake_sdk.queue_script([ResultMessage(is_error=True, subtype="success",
                                         result="Not logged in · Please run /login")])
    client = await _app_client(fake_sdk)
    try:
        resp = await client.post("/v1/messages", json={
            "model": "opus", "messages": [{"role": "user", "content": "hi"}]})
        assert resp.status == 401
        body = await resp.json()
        assert body["error"]["type"] == "authentication_error"
    finally:
        await client.close()
