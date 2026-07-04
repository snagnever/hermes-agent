"""Self-contained fake `claude_agent_sdk` for claude-max tests.

This branch parts from main (no shared fake-SDK fixture), so the stub lives
here. It models the behaviors the bridge depends on (validated in the Task-0
spike): raw StreamEvent passthrough, in-process MCP tool handlers that BLOCK
until a result is delivered, and the emit-tool_use → await-handler →
emit-continuation flow across a response boundary.

Scripting model: `client.script(steps)` where each step is either
  - a raw Anthropic stream event dict            → yielded as StreamEvent
  - ("await_tool", tool_name, args)              → invokes the registered MCP
        handler (which blocks on the bridge's ToolBridge future) and waits for
        its result before continuing
  - a ResultMessage instance                     → ends the turn
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest


# ---- message / event classes -------------------------------------------------

@dataclass
class TextBlock:
    text: str


@dataclass
class ThinkingBlock:
    thinking: str = ""
    signature: str = ""


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict


@dataclass
class AssistantMessage:
    content: list
    model: str = "claude-haiku-4-5-20251001"


@dataclass
class ResultMessage:
    subtype: str = "success"
    is_error: bool = False
    session_id: str = "sess-fake"
    result: str | None = None
    usage: dict | None = None


@dataclass
class StreamEvent:
    event: dict
    session_id: str = "sess-fake"
    uuid: str = "uuid-fake"
    parent_tool_use_id: str | None = None


@dataclass
class SdkMcpTool:
    name: str
    description: str
    input_schema: Any
    handler: Any


def _tool(name, description, input_schema, annotations=None):
    def deco(fn):
        return SdkMcpTool(name=name, description=description,
                          input_schema=input_schema, handler=fn)
    return deco


def _create_sdk_mcp_server(name, version="1.0.0", tools=None):
    return {"type": "sdk", "name": name,
            "instance": types.SimpleNamespace(tools={t.name: t for t in (tools or [])})}


@dataclass
class PermissionResultAllow:
    updated_input: dict | None = None
    behavior: str = "allow"


@dataclass
class PermissionResultDeny:
    message: str = ""
    interrupt: bool = False
    behavior: str = "deny"


class FakeClient:
    """Mimics ClaudeSDKClient with a scriptable multi-response turn."""

    def __init__(self, options=None):
        self.options = options
        self.connected = False
        self.queries: list = []
        self.interrupted = False
        self._steps: list = []

    # -- lifecycle
    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.connected = False

    async def interrupt(self):
        self.interrupted = True

    async def query(self, prompt):
        self.queries.append(prompt)

    # -- scripting
    def script(self, steps: list) -> None:
        """Replace the step list for the next receive_response drain."""
        self._steps = list(steps)

    def _mcp_handlers(self) -> dict:
        servers = getattr(self.options, "mcp_servers", None) or {}
        out = {}
        for srv in servers.values():
            inst = srv.get("instance") if isinstance(srv, dict) else None
            for tname, t in getattr(inst, "tools", {}).items():
                out[tname] = t.handler
        return out

    async def receive_response(self):
        handlers = self._mcp_handlers()
        steps, self._steps = self._steps, []
        for step in steps:
            if isinstance(step, ResultMessage):
                yield step
                return
            if isinstance(step, tuple) and step and step[0] == "await_tool":
                _, tool_name, args = step
                handler = handlers.get(tool_name)
                if handler is not None:
                    # Blocks on the bridge's ToolBridge future until a result
                    # is delivered — exactly like Claude Code calling the tool.
                    await handler(args)
                continue
            # raw event dict
            yield StreamEvent(event=step)


def _client_factory(options=None):
    return FakeClient(options)


@pytest.fixture
def fake_sdk(monkeypatch):
    """Inject a stub claude_agent_sdk; yield the module for scripting access."""
    mod = types.ModuleType("claude_agent_sdk")
    mod.ClaudeSDKClient = _client_factory
    mod.ClaudeAgentOptions = lambda **kw: types.SimpleNamespace(**kw)
    mod.StreamEvent = StreamEvent
    mod.AssistantMessage = AssistantMessage
    mod.ResultMessage = ResultMessage
    mod.TextBlock = TextBlock
    mod.ThinkingBlock = ThinkingBlock
    mod.ToolUseBlock = ToolUseBlock
    mod.SdkMcpTool = SdkMcpTool
    mod.tool = _tool
    mod.create_sdk_mcp_server = _create_sdk_mcp_server
    mod.PermissionResultAllow = PermissionResultAllow
    mod.PermissionResultDeny = PermissionResultDeny
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", mod)
    return mod
