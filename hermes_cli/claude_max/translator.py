"""Translation between the Anthropic Messages API (what hermes speaks) and the
Claude Agent SDK (what backs the proxy).

Pure functions + small stateful helpers — no SDK/aiohttp imports, so this is
fully unit-testable with scripted event dicts.

Directions:
  * request  → :func:`build_options_inputs` (anthropic request → SDK options)
  * SDK stream → :class:`StreamTranslator` (raw StreamEvent.event dicts → clean
    Anthropic SSE) and :class:`MessageAccumulator` (→ final non-stream Message)
  * errors   → :func:`map_result_error`, :func:`anthropic_error_body`

Design notes are grounded in scripts/CLAUDE_MAX_SPIKE_FINDINGS.md:
  - SDK stream events carry non-standard fields (diagnostics, stop_details,
    caller, inference_geo, context_management, iterations, …) → sanitize.
  - Tools are exposed to the model as ``mcp__hermes__<name>`` → strip on the way
    out, re-add when correlating tool_results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

MCP_SERVER_NAME = "hermes"
TOOL_PREFIX = f"mcp__{MCP_SERVER_NAME}__"

# Usage fields hermes reads; everything else the SDK adds is dropped.
_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def add_tool_prefix(name: str) -> str:
    return name if name.startswith(TOOL_PREFIX) else f"{TOOL_PREFIX}{name}"


def strip_tool_prefix(name: str) -> str:
    return name[len(TOOL_PREFIX):] if name.startswith(TOOL_PREFIX) else name


def sanitize_usage(usage: Any) -> dict:
    """Reduce an SDK usage dict to the Anthropic-standard token fields."""
    if not isinstance(usage, dict):
        return {}
    return {k: usage[k] for k in _USAGE_KEYS if k in usage}


# ---------------------------------------------------------------------------
# request → SDK options
# ---------------------------------------------------------------------------


@dataclass
class TranslatedRequest:
    model: str
    system_prompt: Optional[str]
    tool_specs: list[dict]           # [{name, description, input_schema}]
    allowed_tools: list[str]         # ["mcp__hermes__<name>", ...]
    max_tokens: Optional[int] = None
    stream: bool = False
    messages: list[dict] = field(default_factory=list)


def _system_to_prompt(system: Any) -> Optional[str]:
    if system is None:
        return None
    if isinstance(system, str):
        return system or None
    if isinstance(system, list):
        parts = [
            str(b.get("text", ""))
            for b in system
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        joined = "\n\n".join(p for p in parts if p)
        return joined or None
    return None


def build_options_inputs(payload: dict) -> TranslatedRequest:
    """Anthropic request payload → the pieces the bridge needs to configure a
    ClaudeSDKClient. Unsupported params (temperature/top_p/metadata) are simply
    ignored — accepted for compatibility, no effect on the SDK."""
    tools = payload.get("tools") or []
    tool_specs: list[dict] = []
    allowed: list[str] = []
    for t in tools:
        if not isinstance(t, dict) or not t.get("name"):
            continue
        name = str(t["name"])
        tool_specs.append({
            "name": name,
            "description": t.get("description", "") or "",
            "input_schema": t.get("input_schema") or {"type": "object", "properties": {}},
        })
        allowed.append(add_tool_prefix(name))
    return TranslatedRequest(
        model=str(payload.get("model") or "").strip() or "sonnet",
        system_prompt=_system_to_prompt(payload.get("system")),
        tool_specs=tool_specs,
        allowed_tools=allowed,
        max_tokens=payload.get("max_tokens"),
        stream=bool(payload.get("stream")),
        messages=list(payload.get("messages") or []),
    )


# ---------------------------------------------------------------------------
# raw SDK stream events → clean Anthropic events
# ---------------------------------------------------------------------------


def _clean_content_block(block: dict) -> dict:
    """Strip non-standard fields; unprefix tool_use names."""
    if not isinstance(block, dict):
        return block
    btype = block.get("type")
    if btype == "tool_use":
        return {
            "type": "tool_use",
            "id": block.get("id"),
            "name": strip_tool_prefix(str(block.get("name", ""))),
            "input": block.get("input", {}),
        }
    if btype == "text":
        return {"type": "text", "text": block.get("text", "")}
    if btype == "thinking":
        return {"type": "thinking", "thinking": block.get("thinking", ""),
                "signature": block.get("signature", "")}
    if btype == "redacted_thinking":
        return {"type": "redacted_thinking", "data": block.get("data", "")}
    return block


def clean_event(raw: dict, requested_model: str) -> Optional[dict]:
    """Return an Anthropic-spec event dict, or None to drop it."""
    if not isinstance(raw, dict):
        return None
    etype = raw.get("type")

    if etype == "message_start":
        msg = raw.get("message") or {}
        return {"type": "message_start", "message": {
            "id": msg.get("id"),
            "type": "message",
            "role": "assistant",
            "model": requested_model,           # echo request model
            "content": [],
            "stop_reason": msg.get("stop_reason"),
            "stop_sequence": msg.get("stop_sequence"),
            "usage": sanitize_usage(msg.get("usage")),
        }}

    if etype == "content_block_start":
        return {"type": "content_block_start", "index": raw.get("index"),
                "content_block": _clean_content_block(raw.get("content_block") or {})}

    if etype == "content_block_delta":
        # deltas (text_delta/input_json_delta/thinking_delta/signature_delta)
        # are already spec-shaped; pass through untouched.
        return {"type": "content_block_delta", "index": raw.get("index"),
                "delta": raw.get("delta") or {}}

    if etype == "content_block_stop":
        return {"type": "content_block_stop", "index": raw.get("index")}

    if etype == "message_delta":
        delta = raw.get("delta") or {}
        return {"type": "message_delta",
                "delta": {"stop_reason": delta.get("stop_reason"),
                          "stop_sequence": delta.get("stop_sequence")},
                "usage": sanitize_usage(raw.get("usage"))}

    if etype == "message_stop":
        return {"type": "message_stop"}

    if etype == "ping":
        return {"type": "ping"}

    return None  # drop unknown/internal events


def _sse(event: dict) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


class StreamTranslator:
    """Translate raw SDK stream event dicts into Anthropic SSE text chunks."""

    def __init__(self, requested_model: str) -> None:
        self.requested_model = requested_model

    def translate(self, raw: dict) -> list[str]:
        ev = clean_event(raw, self.requested_model)
        return [_sse(ev)] if ev is not None else []

    def ping(self) -> str:
        return _sse({"type": "ping"})


class MessageAccumulator:
    """Assemble a final (non-stream) Anthropic Message dict from raw events."""

    def __init__(self, requested_model: str) -> None:
        self.requested_model = requested_model
        self._id: Optional[str] = None
        self._usage: dict = {}
        self._stop_reason: Optional[str] = None
        self._stop_sequence: Optional[str] = None
        self._blocks: dict[int, dict] = {}
        self._json_buf: dict[int, str] = {}
        self._order: list[int] = []

    def feed(self, raw: dict) -> None:
        etype = raw.get("type")
        if etype == "message_start":
            msg = raw.get("message") or {}
            self._id = msg.get("id")
            self._usage.update(sanitize_usage(msg.get("usage")))
        elif etype == "content_block_start":
            idx = raw.get("index")
            self._blocks[idx] = _clean_content_block(raw.get("content_block") or {})
            self._order.append(idx)
        elif etype == "content_block_delta":
            idx = raw.get("index")
            delta = raw.get("delta") or {}
            dt = delta.get("type")
            blk = self._blocks.setdefault(idx, {"type": "text", "text": ""})
            if dt == "text_delta":
                blk["text"] = blk.get("text", "") + delta.get("text", "")
            elif dt == "thinking_delta":
                blk["thinking"] = blk.get("thinking", "") + delta.get("thinking", "")
            elif dt == "signature_delta":
                blk["signature"] = blk.get("signature", "") + delta.get("signature", "")
            elif dt == "input_json_delta":
                self._json_buf[idx] = self._json_buf.get(idx, "") + delta.get("partial_json", "")
        elif etype == "content_block_stop":
            idx = raw.get("index")
            if idx in self._json_buf:
                blk = self._blocks.get(idx)
                if blk is not None and blk.get("type") == "tool_use":
                    try:
                        blk["input"] = json.loads(self._json_buf[idx] or "{}")
                    except json.JSONDecodeError:
                        blk["input"] = {}
        elif etype == "message_delta":
            delta = raw.get("delta") or {}
            if delta.get("stop_reason") is not None:
                self._stop_reason = delta["stop_reason"]
            if delta.get("stop_sequence") is not None:
                self._stop_sequence = delta["stop_sequence"]
            self._usage.update(sanitize_usage(raw.get("usage")))

    def result(self) -> dict:
        content = [self._blocks[i] for i in self._order if i in self._blocks]
        usage = {k: self._usage.get(k, 0) for k in ("input_tokens", "output_tokens")}
        for k in ("cache_read_input_tokens", "cache_creation_input_tokens"):
            if k in self._usage:
                usage[k] = self._usage[k]
        return {
            "id": self._id or "msg_claude_max",
            "type": "message",
            "role": "assistant",
            "model": self.requested_model,
            "content": content,
            "stop_reason": self._stop_reason,
            "stop_sequence": self._stop_sequence,
            "usage": usage,
        }


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


def anthropic_error_body(err_type: str, message: str) -> dict:
    return {"type": "error", "error": {"type": err_type, "message": message}}


def map_result_error(result_message: Any) -> tuple[int, dict]:
    """Map an SDK error ResultMessage (dict or object) → (status, error body).

    Not-logged-in surfaces as is_error=True, subtype='success', with the real
    message in ``result`` (see spike findings).
    """
    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    text = str(_get(result_message, "result", "") or "")
    subtype = str(_get(result_message, "subtype", "") or "")
    low = text.lower()

    if any(s in low for s in ("not logged in", "/login", "unauthorized", "authentication",
                              "expired", "oauth")):
        return 401, anthropic_error_body(
            "authentication_error",
            text or "Claude authentication failed — run `claude` and /login.")
    if any(s in low for s in ("rate limit", "overloaded", "too many requests", "429",
                              "out of extra usage", "usage limit", "out of usage",
                              "quota")):
        return 429, anthropic_error_body("rate_limit_error", text or "usage limit reached")
    return 500, anthropic_error_body(
        "api_error", text or f"claude-max error ({subtype or 'unknown'})")
