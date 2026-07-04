"""Conversation → live-SDK-session registry.

The Anthropic API is stateless (full history every request) but a
ClaudeSDKClient is a stateful conversation that cannot replay arbitrary
histories. This registry pins each hermes conversation to one ClaudeBridge and
routes each incoming request to the right one:

  * Flow A (tool continuation): the last message carries tool_result blocks →
    match the session whose pending tool_use ids they answer; deliver results
    to the blocked handlers.
  * Flow B (next user turn): the request is a known session's history plus one
    new user message → query() that message on the matched session.
  * Flow C (cache miss): no match (fresh conversation, proxy restart, or hermes
    compacted/edited the history) → new session; if there is prior assistant
    history, flatten it into a synthetic first prompt (documented fidelity
    loss: thinking continuity + cache metrics reset).

Fingerprints are computed over normalized (system, messages) with volatile
fields stripped so cache_control annotations / string-vs-block content don't
cause spurious misses.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 1800.0
DEFAULT_MAX_SESSIONS = 4


# ---------------------------------------------------------------------------
# normalization / fingerprinting
# ---------------------------------------------------------------------------

def _norm_content(content: Any) -> list:
    """Normalize a message's content to a list of blocks with volatile fields
    stripped (cache_control, thinking signatures)."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return []
    out = []
    for b in content:
        if not isinstance(b, dict):
            continue
        b = {k: v for k, v in b.items() if k != "cache_control"}
        if b.get("type") == "thinking":
            b.pop("signature", None)
        out.append(b)
    return out


def _norm_system(system: Any) -> list:
    if system is None:
        return []
    if isinstance(system, str):
        return [{"type": "text", "text": system}]
    if isinstance(system, list):
        return [{k: v for k, v in b.items() if k != "cache_control"}
                for b in system if isinstance(b, dict)]
    return []


def normalize_history(system: Any, messages: list) -> bytes:
    payload = {
        "system": _norm_system(system),
        "messages": [
            {"role": m.get("role"), "content": _norm_content(m.get("content"))}
            for m in (messages or []) if isinstance(m, dict)
        ],
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")


def fingerprint(system: Any, messages: list) -> str:
    return hashlib.sha256(normalize_history(system, messages)).hexdigest()


# ---------------------------------------------------------------------------
# request shape helpers
# ---------------------------------------------------------------------------

def _blocks(message: dict) -> list:
    c = message.get("content")
    return c if isinstance(c, list) else []


def tool_result_ids(message: dict) -> list[str]:
    return [b.get("tool_use_id") for b in _blocks(message)
            if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id")]


def tool_results(message: dict) -> list[tuple[str, list, bool]]:
    """Return [(tool_use_id, content_blocks, is_error)] from a user message."""
    out = []
    for b in _blocks(message):
        if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id"):
            content = b.get("content")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            elif not isinstance(content, list):
                content = [{"type": "text", "text": str(content)}]
            out.append((b["tool_use_id"], content, bool(b.get("is_error"))))
    return out


def is_tool_result_turn(messages: list) -> bool:
    return bool(messages) and messages[-1].get("role") == "user" and bool(tool_result_ids(messages[-1]))


def message_text(message: dict) -> str:
    c = message.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n\n".join(str(b.get("text", "")) for b in c
                           if isinstance(b, dict) and b.get("type") == "text")
    return ""


def flatten_history(system: Any, messages: list) -> str:
    """Cold-start: render prior turns into a synthetic prompt for a fresh
    session. Thinking blocks are dropped; tool calls/results are rendered
    readably. The final user message is appended as the live turn."""
    lines: list[str] = []
    if messages[:-1]:
        lines.append("[Prior conversation transcript — reconstructed]")
        for m in messages[:-1]:
            role = m.get("role", "?")
            for b in _norm_content(m.get("content")):
                bt = b.get("type")
                if bt == "text" and b.get("text", "").strip():
                    lines.append(f"{role}: {b['text']}")
                elif bt == "tool_use":
                    lines.append(f"{role}: → {b.get('name')}({json.dumps(b.get('input', {}))})")
                elif bt == "tool_result":
                    rc = b.get("content")
                    txt = rc if isinstance(rc, str) else json.dumps(rc)
                    lines.append(f"{role}: ← {txt}")
        lines.append("[End transcript — continue below]")
    lines.append(message_text(messages[-1]) if messages else "")
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

@dataclass
class Session:
    bridge: Any
    history_fp: str = ""
    pending_tool_ids: set = field(default_factory=set)
    last_used: float = 0.0
    lock: Any = None  # asyncio.Lock, created lazily by the handler layer

    def touch(self, now: float) -> None:
        self.last_used = now


@dataclass
class MatchResult:
    kind: str                       # "tool_result" | "continue" | "cold"
    session: Session
    created: bool = False
    query_text: Optional[str] = None
    deliveries: list = field(default_factory=list)  # [(id, content, is_error)]


class SessionRegistry:
    def __init__(
        self,
        bridge_factory: Callable[[Any], Any],
        *,
        ttl: float = DEFAULT_TTL_SECONDS,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._factory = bridge_factory
        self._ttl = ttl
        self._max = max_sessions
        self._clock = clock
        self._sessions: list[Session] = []

    # -- lookup --
    def _find_by_pending(self, ids: list[str]) -> Optional[Session]:
        want = set(ids)
        for s in self._sessions:
            if want and want.issubset(s.pending_tool_ids):
                return s
        return None

    def _find_by_prefix_fp(self, system: Any, messages: list) -> Optional[Session]:
        fp = fingerprint(system, messages[:-1])
        for s in self._sessions:
            if s.history_fp and s.history_fp == fp:
                return s
        return None

    def match(self, payload: dict) -> MatchResult:
        """Decide Flow A/B/C for an incoming request and return the session to
        drive. The caller (handler) performs the async query/deliver + updates
        session.history_fp / pending_tool_ids after streaming."""
        now = self._clock()
        self._evict_expired(now)
        system = payload.get("system")
        messages = payload.get("messages") or []
        translated_req = payload  # translator builds options separately

        # Flow A: tool continuation
        if is_tool_result_turn(messages):
            ids = tool_result_ids(messages[-1])
            s = self._find_by_pending(ids)
            if s is not None:
                s.touch(now)
                return MatchResult("tool_result", s, deliveries=tool_results(messages[-1]))
            # No pinned session (proxy restart mid-tool) → cold start below.

        # Flow B: next user turn on a known session
        elif messages and messages[-1].get("role") == "user":
            s = self._find_by_prefix_fp(system, messages)
            if s is not None:
                s.touch(now)
                return MatchResult("continue", s, query_text=message_text(messages[-1]))

        # Flow C: cold start (new session)
        self._evict_for_capacity(now)
        bridge = self._factory(translated_req)
        session = Session(bridge=bridge, last_used=now)
        self._sessions.append(session)
        query_text = flatten_history(system, messages) if messages else ""
        return MatchResult("cold", session, created=True, query_text=query_text)

    # -- post-response bookkeeping (called by handler) --
    def record_response(self, session: Session, system: Any, messages: list,
                        assistant_content: list, pending_tool_ids: set) -> None:
        """Update the session fingerprint to include the assistant turn we just
        produced, so the NEXT request (Flow B) matches on this prefix."""
        full = list(messages) + [{"role": "assistant", "content": assistant_content}]
        session.history_fp = fingerprint(system, full)
        session.pending_tool_ids = set(pending_tool_ids)
        session.touch(self._clock())

    # -- eviction --
    def _evict_expired(self, now: float) -> None:
        alive = []
        for s in self._sessions:
            if now - s.last_used > self._ttl:
                self._close(s)
            else:
                alive.append(s)
        self._sessions = alive

    def _evict_for_capacity(self, now: float) -> None:
        while len(self._sessions) >= self._max:
            lru = min(self._sessions, key=lambda s: s.last_used)
            self._sessions.remove(lru)
            self._close(lru)

    def _close(self, session: Session) -> None:
        close = getattr(session.bridge, "close", None)
        if close is None:
            return
        try:
            import asyncio
            res = close()
            if asyncio.iscoroutine(res):
                # best-effort: schedule on the running loop if any
                try:
                    asyncio.get_running_loop().create_task(res)
                except RuntimeError:
                    asyncio.new_event_loop().run_until_complete(res)
        except Exception:
            logger.debug("session close failed", exc_info=True)

    def remove(self, session: Session) -> None:
        if session in self._sessions:
            self._sessions.remove(session)
        self._close(session)

    @property
    def size(self) -> int:
        return len(self._sessions)
