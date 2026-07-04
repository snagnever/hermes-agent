"""POST /v1/messages handler: routes a request to its live SDK session (Flow
A/B/C), drives one response chunk, and returns it as Anthropic SSE (stream) or
a final Message (non-stream)."""

from __future__ import annotations

import json
import logging
from typing import Any

from aiohttp import web

from hermes_cli.claude_max.server import REGISTRY_KEY
from hermes_cli.claude_max.translator import (
    MessageAccumulator,
    StreamTranslator,
    build_options_inputs,
    map_result_error,
)

logger = logging.getLogger(__name__)


def _pending_tool_ids(assistant_content: list, stop_reason: str | None) -> set:
    if stop_reason != "tool_use":
        return set()
    return {b.get("id") for b in assistant_content
            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id")}


async def _perform_action(mr, query_text_source) -> None:
    bridge = mr.session.bridge
    if mr.kind == "cold":
        await bridge.connect()
        await bridge.query(mr.query_text or "")
    elif mr.kind == "continue":
        await bridge.query(mr.query_text or "")
    elif mr.kind == "tool_result":
        for tool_use_id, content, is_error in mr.deliveries:
            bridge.deliver_tool_result(tool_use_id, content, is_error)


async def handle_messages(request: "web.Request") -> "web.StreamResponse":
    registry = request.app[REGISTRY_KEY]
    try:
        payload = await request.json()
    except Exception:
        return web.json_response(
            {"type": "error",
             "error": {"type": "invalid_request_error", "message": "invalid JSON body"}},
            status=400,
        )

    tr = build_options_inputs(payload)
    system = payload.get("system")
    messages = payload.get("messages") or []

    mr = registry.match(payload)
    session = mr.session
    if session.lock is None:
        import asyncio
        session.lock = asyncio.Lock()

    async with session.lock:
        try:
            await _perform_action(mr, tr)
        except Exception as exc:
            logger.exception("claude-max action failed")
            registry.remove(session)
            return web.json_response(
                {"type": "error", "error": {"type": "api_error", "message": f"session error: {exc}"}},
                status=500,
            )

        bridge = session.bridge
        it = bridge.stream_one_response().__aiter__()

        # Peek the first event so we can return an error JSON (auth, etc.)
        # before committing to a streamed response.
        try:
            first = await it.__anext__()
        except StopAsyncIteration:
            first = None

        if first is None:
            if bridge.error:
                status, body = map_result_error(getattr(bridge, "final_result", None) or
                                                {"is_error": True, "result": bridge.error})
                registry.remove(session)
                return web.json_response(body, status=status)
            # Empty success — return a minimal end_turn message.
            body = MessageAccumulator(tr.model).result()
            body["stop_reason"] = "end_turn"
            registry.record_response(session, system, messages, body["content"], set())
            return web.json_response(body)

        acc = MessageAccumulator(tr.model)

        if tr.stream:
            return await _stream_response(request, session, registry, system, messages,
                                          tr, it, first, acc)
        return await _aggregate_response(session, registry, system, messages, tr, it, first, acc)


async def _finalize(session, registry, system, messages, acc) -> None:
    msg = acc.result()
    registry.record_response(
        session, system, messages, msg["content"],
        _pending_tool_ids(msg["content"], msg.get("stop_reason")),
    )


async def _aggregate_response(session, registry, system, messages, tr, it, first, acc):
    acc.feed(first)
    async for raw in it:
        acc.feed(raw)
    await _finalize(session, registry, system, messages, acc)
    return web.json_response(acc.result())


async def _stream_response(request, session, registry, system, messages, tr, it, first, acc):
    translator = StreamTranslator(tr.model)
    resp = web.StreamResponse(status=200, headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    })
    await resp.prepare(request)

    async def _write(raw):
        acc.feed(raw)
        for chunk in translator.translate(raw):
            await resp.write(chunk.encode("utf-8"))

    try:
        await _write(first)
        async for raw in it:
            await _write(raw)
        await resp.write_eof()
    except (ConnectionResetError, Exception) as exc:  # noqa: BLE001
        # Client disconnected mid-stream: keep the session (hermes may
        # reconnect with tool results); do not interrupt if tools pending.
        logger.debug("claude-max stream ended early: %s", exc)
    await _finalize(session, registry, system, messages, acc)
    return resp
