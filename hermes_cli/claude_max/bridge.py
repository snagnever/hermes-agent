"""ClaudeBridge — one ClaudeSDKClient per conversation, driven so the stateless
Anthropic API can sit in front of the stateful SDK.

Two pieces:
  * ToolBridge — the model calls caller-defined MCP tools whose handlers BLOCK
    (await a future) until the HTTP client returns the matching tool_result on
    the next request. Correlates handler invocations (name + args) to the
    tool_use ids seen in the stream, so results delivered by id resolve the
    right blocked handler. Handlers run sequentially (spike exp 4), so results
    delivered before a handler is invoked are stored and consumed on invocation.
  * ClaudeBridge — owns the SDK client + a persistent ``receive_response``
    iterator, segmented into per-HTTP-response chunks: a chunk ends at the
    assistant ``message_stop`` when a tool is pending (stop_reason=tool_use) or
    at the final ``ResultMessage``.

Grounded in scripts/CLAUDE_MAX_SPIKE_FINDINGS.md.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Optional

from hermes_cli.claude_max.translator import (
    MCP_SERVER_NAME,
    TranslatedRequest,
    strip_tool_prefix,
)

logger = logging.getLogger(__name__)

_DEFAULT_MCP_TIMEOUT_MS = 600_000  # allow long hermes tool executions
_MAX_TURNS = 100_000              # effectively unbounded; the proxy owns turns


def _canonical(args: Any) -> str:
    try:
        return json.dumps(args or {}, sort_keys=True, default=str)
    except Exception:
        return str(args)


class ToolBridge:
    """Correlates blocking MCP handlers with tool_results delivered by id."""

    def __init__(self) -> None:
        # (tool_name, canonical_args) -> FIFO of tool_use ids seen in the stream
        self._id_queue: dict[tuple[str, str], list[str]] = {}
        self._futures: dict[str, asyncio.Future] = {}   # tool_use_id -> future
        self._results: dict[str, dict] = {}             # tool_use_id -> result (early)

    def note_tool_use(self, tool_use_id: str, prefixed_name: str, input_data: Any) -> None:
        """Record a tool_use block observed in the stream so a later handler
        invocation can be mapped to this id."""
        key = (strip_tool_prefix(prefixed_name), _canonical(input_data))
        self._id_queue.setdefault(key, []).append(tool_use_id)

    def make_handler(self, tool_name: str):
        bridge = self

        async def _handler(args: Any) -> dict:
            key = (tool_name, _canonical(args))
            ids = bridge._id_queue.get(key)
            tool_use_id = ids.pop(0) if ids else None
            if tool_use_id is not None and tool_use_id in bridge._results:
                return bridge._results.pop(tool_use_id)
            fut: asyncio.Future = asyncio.get_event_loop().create_future()
            # If we couldn't correlate an id yet, park under a synthetic key so
            # abort_all still reaches it; deliver_result matches by id when known.
            park_id = tool_use_id or f"__uncorrelated__:{key[0]}:{len(bridge._futures)}"
            bridge._futures[park_id] = fut
            try:
                return await fut
            finally:
                bridge._futures.pop(park_id, None)

        return _handler

    def deliver_result(self, tool_use_id: str, content_blocks: list, is_error: bool = False) -> None:
        result = {"content": content_blocks, "is_error": is_error}
        fut = self._futures.get(tool_use_id)
        if fut is not None and not fut.done():
            fut.set_result(result)
        else:
            # Handler not invoked yet (sequential execution) — store for pickup.
            self._results[tool_use_id] = result

    def abort_all(self, message: str = "aborted") -> None:
        err = {"content": [{"type": "text", "text": message}], "is_error": True}
        for fut in list(self._futures.values()):
            if not fut.done():
                fut.set_result(err)
        self._futures.clear()

    def has_pending(self) -> bool:
        return any(not f.done() for f in self._futures.values())


class ClaudeBridge:
    """One SDK client for one conversation; yields per-response event chunks."""

    def __init__(self, translated: TranslatedRequest, *, mcp_timeout_ms: int = _DEFAULT_MCP_TIMEOUT_MS) -> None:
        self._tr = translated
        self._mcp_timeout_ms = mcp_timeout_ms
        self.tools = ToolBridge()
        self._client: Any = None
        self._it: Optional[AsyncIterator[Any]] = None
        self._last_stop_reason: Optional[str] = None
        self.final_result: Any = None   # ResultMessage when the turn completes
        self.error: Optional[str] = None
        self._closed = False

    # ---- options ----
    def _build_options(self, sdk):
        tool_defs = [
            sdk.tool(spec["name"], spec["description"], spec["input_schema"])(
                self.tools.make_handler(spec["name"])
            )
            for spec in self._tr.tool_specs
        ]
        kwargs: dict[str, Any] = {
            "model": self._tr.model,
            "tools": [],  # no built-in tools — proxy owns all tool behavior
            "setting_sources": [],
            "include_partial_messages": True,
            "max_turns": _MAX_TURNS,
            "env": {
                "ANTHROPIC_API_KEY": "",
                "MCP_TOOL_TIMEOUT": str(self._mcp_timeout_ms),
                "MCP_TIMEOUT": str(self._mcp_timeout_ms),
            },
        }
        if self._tr.system_prompt:
            kwargs["system_prompt"] = self._tr.system_prompt
        if tool_defs:
            server = sdk.create_sdk_mcp_server(MCP_SERVER_NAME, tools=tool_defs)
            kwargs["mcp_servers"] = {MCP_SERVER_NAME: server}
            kwargs["allowed_tools"] = list(self._tr.allowed_tools)
        return sdk.ClaudeAgentOptions(**kwargs)

    # ---- lifecycle ----
    async def connect(self) -> None:
        import claude_agent_sdk as sdk

        options = self._build_options(sdk)
        self._client = sdk.ClaudeSDKClient(options=options)
        await self._client.connect()

    async def query(self, text: str) -> None:
        """Send a user turn and (re)open the response iterator."""
        await self._client.query(text)
        self._it = self._client.receive_response().__aiter__()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.tools.abort_all("session closed")
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception:
                logger.debug("claude-max bridge disconnect failed", exc_info=True)

    async def interrupt(self) -> None:
        if self._client is not None:
            try:
                await self._client.interrupt()
            except Exception:
                logger.debug("claude-max bridge interrupt failed", exc_info=True)
        self.tools.abort_all("interrupted")

    # ---- per-response streaming ----
    async def stream_one_response(self) -> AsyncIterator[dict]:
        """Yield raw Anthropic stream-event dicts for ONE HTTP response.

        Ends at the assistant message_stop when a tool is pending
        (stop_reason=tool_use) or at the final ResultMessage.
        """
        if self._it is None:
            return
        import claude_agent_sdk as sdk

        while True:
            try:
                msg = await self._it.__anext__()
            except StopAsyncIteration:
                self.final_result = self.final_result or True
                return

            if isinstance(msg, sdk.ResultMessage):
                self.final_result = msg
                if getattr(msg, "is_error", False):
                    self.error = f"{getattr(msg, 'subtype', 'error')}: {getattr(msg, 'result', '') or ''}"
                return

            if isinstance(msg, sdk.StreamEvent):
                raw = msg.event or {}
                etype = raw.get("type")
                if etype == "content_block_start":
                    blk = raw.get("content_block") or {}
                    if blk.get("type") == "tool_use":
                        self.tools.note_tool_use(
                            blk.get("id"), str(blk.get("name", "")), blk.get("input", {})
                        )
                elif etype == "message_delta":
                    sr = (raw.get("delta") or {}).get("stop_reason")
                    if sr is not None:
                        self._last_stop_reason = sr
                yield raw
                if etype == "message_stop" and self._last_stop_reason == "tool_use":
                    self._last_stop_reason = None
                    return  # boundary: tools pending, response N ends here
            # ignore AssistantMessage/other high-level messages

    def deliver_tool_result(self, tool_use_id: str, content_blocks: list, is_error: bool = False) -> None:
        self.tools.deliver_result(tool_use_id, content_blocks, is_error)

    @property
    def is_done(self) -> bool:
        return self.final_result is not None
