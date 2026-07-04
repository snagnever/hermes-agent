"""Translator: anthropic request <-> SDK options, raw SDK stream -> clean
Anthropic SSE / final message, usage + error mapping. Pure functions."""

import json

from hermes_cli.claude_max import translator as T


# ---------- request → options ----------

def test_build_options_system_join_and_tools():
    payload = {
        "model": "claude-opus-4-8",
        "system": [{"type": "text", "text": "You are helpful."},
                   {"type": "text", "text": "Be terse."}],
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "get_weather", "description": "wx",
                   "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}],
        "max_tokens": 1024,
        "temperature": 0.7,   # unsupported — must be accepted silently
    }
    tr = T.build_options_inputs(payload)
    assert tr.model == "claude-opus-4-8"
    assert "You are helpful." in tr.system_prompt and "Be terse." in tr.system_prompt
    assert tr.tool_specs[0]["name"] == "get_weather"
    assert tr.allowed_tools == ["mcp__hermes__get_weather"]


def test_tool_prefix_helpers():
    assert T.add_tool_prefix("foo") == "mcp__hermes__foo"
    assert T.strip_tool_prefix("mcp__hermes__foo") == "foo"
    assert T.strip_tool_prefix("bar") == "bar"  # non-prefixed passthrough


# ---------- stream translation ----------

def _sse_events(lines):
    """Parse emitted SSE text back into (event, data) tuples."""
    out = []
    for block in "".join(lines).split("\n\n"):
        block = block.strip()
        if not block:
            continue
        ev = dat = None
        for ln in block.splitlines():
            if ln.startswith("event:"):
                ev = ln.split(":", 1)[1].strip()
            elif ln.startswith("data:"):
                dat = json.loads(ln.split(":", 1)[1].strip())
        out.append((ev, dat))
    return out


def test_stream_message_start_rewrites_model_and_sanitizes():
    tx = T.StreamTranslator(requested_model="claude-opus-4-8")
    raw = {"type": "message_start", "message": {
        "id": "msg_orig", "type": "message", "role": "assistant",
        "model": "claude-haiku-4-5-20251001", "content": [],
        "stop_reason": None, "stop_sequence": None,
        "diagnostics": None, "stop_details": None,   # non-standard → stripped
        "usage": {"input_tokens": 10, "output_tokens": 1,
                  "cache_read_input_tokens": 2, "cache_creation_input_tokens": 3,
                  "service_tier": "standard", "inference_geo": "x", "iterations": []},
    }}
    (ev, data), = _sse_events(tx.translate(raw))
    assert ev == "message_start"
    assert data["message"]["model"] == "claude-opus-4-8"  # echoed
    assert "diagnostics" not in data["message"]
    assert "stop_details" not in data["message"]
    u = data["message"]["usage"]
    assert u == {"input_tokens": 10, "output_tokens": 1,
                 "cache_read_input_tokens": 2, "cache_creation_input_tokens": 3}


def test_stream_tool_use_name_unprefixed():
    tx = T.StreamTranslator(requested_model="opus")
    raw = {"type": "content_block_start", "index": 1, "content_block": {
        "type": "tool_use", "id": "toolu_1", "name": "mcp__hermes__get_weather",
        "input": {}, "caller": {"type": "direct"}}}
    (ev, data), = _sse_events(tx.translate(raw))
    assert data["content_block"]["name"] == "get_weather"
    assert "caller" not in data["content_block"]


def test_stream_message_delta_keeps_stop_reason_sanitizes_usage():
    tx = T.StreamTranslator(requested_model="opus")
    raw = {"type": "message_delta",
           "delta": {"stop_reason": "tool_use", "stop_sequence": None, "stop_details": None},
           "usage": {"input_tokens": 5, "output_tokens": 20, "iterations": [1],
                     "output_tokens_details": {"thinking_tokens": 3}},
           "context_management": {"applied_edits": []}}
    (ev, data), = _sse_events(tx.translate(raw))
    assert data["delta"]["stop_reason"] == "tool_use"
    assert "stop_details" not in data["delta"]
    assert "context_management" not in data
    assert data["usage"]["output_tokens"] == 20
    assert "iterations" not in data["usage"]


def test_stream_drops_empty_signature_thinking_noise_but_keeps_text():
    tx = T.StreamTranslator(requested_model="opus")
    passthru = tx.translate({"type": "content_block_delta", "index": 0,
                             "delta": {"type": "text_delta", "text": "hi"}})
    (ev, data), = _sse_events(passthru)
    assert data["delta"]["text"] == "hi"


# ---------- non-stream accumulation ----------

def test_accumulator_builds_final_message():
    acc = T.MessageAccumulator(requested_model="claude-opus-4-8")
    for raw in [
        {"type": "message_start", "message": {"id": "msg_x", "model": "claude-haiku-4-5-20251001",
            "usage": {"input_tokens": 8, "output_tokens": 0}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello "}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "world"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {
            "type": "tool_use", "id": "toolu_9", "name": "mcp__hermes__get_weather", "input": {}}},
        {"type": "content_block_delta", "index": 1, "delta": {
            "type": "input_json_delta", "partial_json": '{"city":"Lisbon"}'}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 25}},
        {"type": "message_stop"},
    ]:
        acc.feed(raw)
    msg = acc.result()
    assert msg["type"] == "message" and msg["role"] == "assistant"
    assert msg["model"] == "claude-opus-4-8"
    assert msg["stop_reason"] == "tool_use"
    texts = [b for b in msg["content"] if b["type"] == "text"]
    tools = [b for b in msg["content"] if b["type"] == "tool_use"]
    assert texts[0]["text"] == "Hello world"
    assert tools[0]["name"] == "get_weather"  # unprefixed
    assert tools[0]["input"] == {"city": "Lisbon"}
    assert msg["usage"]["input_tokens"] == 8 and msg["usage"]["output_tokens"] == 25


# ---------- error mapping ----------

def test_map_result_error_auth():
    status, body = T.map_result_error(
        {"is_error": True, "subtype": "success",
         "result": "Not logged in · Please run /login"})
    assert status == 401
    assert body["error"]["type"] == "authentication_error"
    assert "login" in body["error"]["message"].lower()


def test_map_result_error_generic():
    status, body = T.map_result_error(
        {"is_error": True, "subtype": "error_during_execution", "result": "boom"})
    assert status == 500
    assert body["error"]["type"] == "api_error"


def test_map_result_error_usage_quota():
    status, body = T.map_result_error(
        {"is_error": True, "result": "400 You're out of extra usage. Add more..."})
    assert status == 429
    assert body["error"]["type"] == "rate_limit_error"


def test_anthropic_error_shape():
    body = T.anthropic_error_body("rate_limit_error", "slow down")
    assert body == {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}
