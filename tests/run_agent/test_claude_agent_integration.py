"""Integration test for the claude_agent_sdk runtime path through AIAgent.

Mirrors test_codex_app_server_integration.py: verifies that
api_mode='claude_agent_sdk' is accepted, run_conversation() takes the
early-return path, projected messages land in the messages list, and the
returned dict has the codex-shaped keys. Uses a stubbed ClaudeAgentSession
so no Claude Code subprocess / SDK / network is required.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import run_agent
from agent.transports.claude_agent_session import ClaudeAgentSession, ClaudeTurnResult


@pytest.fixture
def fake_claude_session(monkeypatch):
    def fake_run_turn(self, user_input, **kwargs):
        return ClaudeTurnResult(
            final_text=f"echo: {user_input}",
            projected_messages=[
                {"role": "assistant", "content": None,
                 "tool_calls": [{"id": "t1", "type": "function",
                                 "function": {"name": "Bash", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "t1", "content": "ok"},
                {"role": "assistant", "content": f"echo: {user_input}"},
            ],
            tool_iterations=1,
            session_id="sess-stub-1",
            usage={},
        )

    monkeypatch.setattr(ClaudeAgentSession, "run_turn", fake_run_turn)
    monkeypatch.setattr(ClaudeAgentSession, "ensure_started", lambda self: None)


def _make_claude_agent():
    return run_agent.AIAgent(
        api_key="stub",
        base_url="https://stub.invalid",
        provider="anthropic",
        api_mode="claude_agent_sdk",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def test_api_mode_is_claude_agent_sdk():
    agent = _make_claude_agent()
    assert agent.api_mode == "claude_agent_sdk"


def test_provider_claude_agent_implies_sdk_without_explicit_api_mode():
    # Picking the claude-agent provider (no api_mode passed) must resolve to
    # the SDK runtime — the provider is self-describing.
    agent = run_agent.AIAgent(
        api_key="stub",
        base_url="",
        provider="claude-agent",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    assert agent.api_mode == "claude_agent_sdk"
    assert agent.provider == "claude-agent"


def test_run_conversation_dispatches_claude_agent_sdk():
    with patch("agent.claude_runtime.run_claude_agent_sdk_turn") as run_turn:
        run_turn.return_value = {
            "final_response": "ok", "messages": [], "api_calls": 1,
            "completed": True, "partial": False, "error": None,
            "agent_persisted": True,
        }
        agent = _make_claude_agent()
        result = agent.run_conversation("hi")
    assert run_turn.call_count == 1
    assert result["final_response"] == "ok"


def test_run_conversation_returns_claude_shape(fake_claude_session):
    agent = _make_claude_agent()
    with patch.object(agent, "_spawn_background_review", return_value=None):
        result = agent.run_conversation("hello there")
    assert result["final_response"] == "echo: hello there"
    assert result["completed"] is True
    assert result["partial"] is False
    assert result["error"] is None
    assert result["api_calls"] == 1
    assert result["claude_session_id"] == "sess-stub-1"
    # Projected messages spliced into the transcript.
    assert any(m.get("role") == "tool" for m in result["messages"])


def test_should_retire_drops_session(monkeypatch):
    closes = {"count": 0}

    def fake_run_turn(self, user_input, **kwargs):
        return ClaudeTurnResult(
            final_text="", interrupted=True,
            error="turn timed out after 600.0s", should_retire=True, usage={},
        )

    monkeypatch.setattr(ClaudeAgentSession, "ensure_started", lambda self: None)
    monkeypatch.setattr(ClaudeAgentSession, "run_turn", fake_run_turn)
    monkeypatch.setattr(ClaudeAgentSession, "close",
                        lambda self, *a, **k: closes.__setitem__("count", closes["count"] + 1))

    agent = _make_claude_agent()
    with patch.object(agent, "_spawn_background_review", return_value=None):
        result = agent.run_conversation("hi")

    assert closes["count"] == 1
    assert getattr(agent, "_claude_session", "MISSING") is None
    assert result["partial"] is True
    assert result["error"] == "turn timed out after 600.0s"
