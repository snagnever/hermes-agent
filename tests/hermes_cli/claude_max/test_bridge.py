"""ToolBridge future mechanics + ClaudeBridge options/streaming (fake SDK)."""

import asyncio

import pytest

from hermes_cli.claude_max.bridge import ClaudeBridge, ToolBridge
from hermes_cli.claude_max.translator import build_options_inputs


# ---------- ToolBridge ----------

@pytest.mark.asyncio
async def test_handler_blocks_until_result_delivered():
    tb = ToolBridge()
    tb.note_tool_use("toolu_1", "mcp__hermes__get_weather", {"city": "Lisbon"})
    handler = tb.make_handler("get_weather")

    task = asyncio.create_task(handler({"city": "Lisbon"}))
    await asyncio.sleep(0.01)
    assert not task.done()  # blocked

    tb.deliver_result("toolu_1", [{"type": "text", "text": "Sunny"}])
    result = await asyncio.wait_for(task, timeout=1)
    assert result == {"content": [{"type": "text", "text": "Sunny"}], "is_error": False}


@pytest.mark.asyncio
async def test_result_delivered_before_handler_invoked():
    # Sequential parallel execution: result may arrive before the handler runs.
    tb = ToolBridge()
    tb.note_tool_use("toolu_2", "mcp__hermes__get_time", {"city": "Lisbon"})
    tb.deliver_result("toolu_2", [{"type": "text", "text": "12:00"}])

    handler = tb.make_handler("get_time")
    result = await asyncio.wait_for(handler({"city": "Lisbon"}), timeout=1)
    assert result["content"][0]["text"] == "12:00"


@pytest.mark.asyncio
async def test_abort_all_unblocks_with_error():
    tb = ToolBridge()
    tb.note_tool_use("toolu_3", "mcp__hermes__x", {})
    handler = tb.make_handler("x")
    task = asyncio.create_task(handler({}))
    await asyncio.sleep(0.01)
    tb.abort_all("boom")
    result = await asyncio.wait_for(task, timeout=1)
    assert result["is_error"] is True


# ---------- ClaudeBridge options ----------

@pytest.mark.asyncio
async def test_bridge_builds_isolated_toolless_options(fake_sdk):
    tr = build_options_inputs({
        "model": "claude-opus-4-8",
        "system": "be nice",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "get_weather", "description": "wx",
                   "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}],
    })
    bridge = ClaudeBridge(tr)
    await bridge.connect()
    opts = bridge._client.options
    assert opts.model == "claude-opus-4-8"
    assert opts.tools == []                      # no built-in tools
    assert opts.setting_sources == []            # isolation
    assert opts.include_partial_messages is True
    assert opts.system_prompt == "be nice"
    assert opts.env["ANTHROPIC_API_KEY"] == ""
    assert opts.env["MCP_TOOL_TIMEOUT"] == "600000"
    assert opts.allowed_tools == ["mcp__hermes__get_weather"]
    assert "hermes" in opts.mcp_servers
    await bridge.close()
    assert bridge._client.connected is False


# ---------- ClaudeBridge streaming ----------

def _msg_events(text="hi", stop="end_turn"):
    return [
        {"type": "message_start", "message": {"id": "msg_1", "model": "claude-haiku-4-5-20251001",
            "usage": {"input_tokens": 5, "output_tokens": 0}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": 3}},
        {"type": "message_stop"},
    ]


@pytest.mark.asyncio
async def test_stream_plain_text_ends_at_result(fake_sdk):
    tr = build_options_inputs({"model": "opus", "messages": [{"role": "user", "content": "hi"}]})
    bridge = ClaudeBridge(tr)
    await bridge.connect()
    from tests.hermes_cli.claude_max.conftest import ResultMessage
    bridge._client.script(_msg_events("hello") + [ResultMessage(result="hello")])
    await bridge.query("hi")

    types_seen = [ev["type"] async for ev in bridge.stream_one_response()]
    assert types_seen[0] == "message_start" and types_seen[-1] == "message_stop"
    assert bridge.is_done  # ResultMessage consumed
    await bridge.close()


@pytest.mark.asyncio
async def test_stream_tool_turn_boundary_then_continuation(fake_sdk):
    tr = build_options_inputs({
        "model": "opus",
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
    })
    bridge = ClaudeBridge(tr)
    await bridge.connect()
    from tests.hermes_cli.claude_max.conftest import ResultMessage

    tool_use_events = [
        {"type": "message_start", "message": {"id": "m1", "model": "x", "usage": {"input_tokens": 9}}},
        {"type": "content_block_start", "index": 0, "content_block": {
            "type": "tool_use", "id": "toolu_9", "name": "mcp__hermes__get_weather", "input": {"city": "Lisbon"}}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 4}},
        {"type": "message_stop"},
        # handler blocks HERE until deliver; then continuation:
        ("await_tool", "get_weather", {"city": "Lisbon"}),
    ] + _msg_events("It is sunny.") + [ResultMessage(result="It is sunny.")]
    bridge._client.script(tool_use_events)
    await bridge.query("weather?")

    # Response 1: ends at tool_use message_stop
    r1 = [ev["type"] async for ev in bridge.stream_one_response()]
    assert r1[-1] == "message_stop"
    assert not bridge.is_done  # tools pending
    assert bridge.tools.has_pending() is False  # handler not invoked until we pull again

    # hermes returns the tool_result -> resolve the blocked handler
    bridge.deliver_tool_result("toolu_9", [{"type": "text", "text": "Sunny 22C"}])

    # Response 2: continuation streams to completion
    r2 = [ev["type"] async for ev in bridge.stream_one_response()]
    assert "message_start" in r2 and r2[-1] == "message_stop"
    assert bridge.is_done
    await bridge.close()
