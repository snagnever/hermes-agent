"""Stub claude_agent_sdk so ClaudeAgentSession tests run without the
package, the claude binary, or network."""

import asyncio
import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest


@dataclass
class TextBlock:
    text: str


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict


@dataclass
class AssistantMessage:
    content: list
    model: str = "claude-opus-4-8"
    usage: dict | None = None


@dataclass
class ResultMessage:
    subtype: str = "success"
    is_error: bool = False
    session_id: str = "sess-fake-1"
    result: str | None = None
    usage: dict | None = None
    total_cost_usd: float | None = None
    num_turns: int = 1


@dataclass
class PermissionResultAllow:
    updated_input: dict | None = None


@dataclass
class PermissionResultDeny:
    message: str = ""
    interrupt: bool = False


class FakeClient:
    """Mimics ClaudeSDKClient: connect/query/receive_response/interrupt/disconnect."""

    def __init__(self, options=None):
        self.options = options
        self.connected = False
        self.queries: list[str] = []
        self.interrupted = False
        self._scripted: list[list[Any]] = []  # one message-list per turn

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.connected = False

    async def query(self, prompt: str):
        self.queries.append(prompt)

    async def interrupt(self):
        self.interrupted = True

    async def receive_response(self):
        messages = self._scripted.pop(0) if self._scripted else []
        for m in messages:
            await asyncio.sleep(0)  # yield control like a real stream
            yield m


class FakeSDK:
    def __init__(self, module):
        self.module = module
        self.clients: list[FakeClient] = []
        self._pending_scripts: list[list[Any]] = []

    def script_turn(self, messages: list):
        """Queue one turn's message stream for the next/current client."""
        if self.clients:
            self.clients[-1]._scripted.append(messages)
        else:
            self._pending_scripts.append(messages)


@pytest.fixture
def fake_claude_sdk(monkeypatch):
    module = types.ModuleType("claude_agent_sdk")
    sdk = FakeSDK(module)

    def _client_factory(options=None):
        client = FakeClient(options)
        client._scripted = list(sdk._pending_scripts)
        sdk._pending_scripts = []
        sdk.clients.append(client)
        return client

    module.ClaudeSDKClient = _client_factory
    module.ClaudeAgentOptions = lambda **kw: types.SimpleNamespace(**kw)
    module.TextBlock = TextBlock
    module.ToolUseBlock = ToolUseBlock
    module.AssistantMessage = AssistantMessage
    module.ResultMessage = ResultMessage
    module.PermissionResultAllow = PermissionResultAllow
    module.PermissionResultDeny = PermissionResultDeny
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
    return sdk
