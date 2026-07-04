"""Auxiliary tasks (title generation, compression, …) route through the
Claude Agent SDK on a Haiku model when there's no HTTP aux provider."""

from types import SimpleNamespace

import agent.auxiliary_client as ac


class _FakeTurn:
    def __init__(self, text="Title Here", error=None):
        self.final_text = text
        self.error = error
        self.usage = {"input_tokens": 5, "output_tokens": 2}


class _FakeSession:
    last = None

    def __init__(self, **kw):
        self.kw = kw
        self.turns = []
        self.closed = False
        _FakeSession.last = self

    def run_turn(self, text, turn_timeout=None):
        self.turns.append((text, turn_timeout))
        return _FakeTurn()

    def close(self):
        self.closed = True


def test_oneshot_builds_openai_shape(monkeypatch):
    monkeypatch.setattr(
        "agent.transports.claude_agent_session.ClaudeAgentSession", _FakeSession
    )
    resp = ac._claude_agent_oneshot_response(
        [
            {"role": "system", "content": "You write titles."},
            {"role": "user", "content": "chat about cats"},
        ],
        model="claude-haiku-4-5",
        timeout=30,
    )
    # OpenAI-ChatCompletion shape that extract_content_or_reasoning reads.
    assert resp.choices[0].message.content == "Title Here"
    assert ac.extract_content_or_reasoning(resp) == "Title Here"
    # system → system_prompt, user text → the turn; no hermes tools / no probe.
    s = _FakeSession.last
    assert s.kw["system_prompt"] == "You write titles."
    assert s.kw["enable_hermes_tools"] is False
    assert s.kw["auto_approve"] is True
    assert s.turns == [("chat about cats", 30)]
    assert s.closed is True


def test_oneshot_raises_on_turn_error(monkeypatch):
    class _ErrSession(_FakeSession):
        def run_turn(self, text, turn_timeout=None):
            return _FakeTurn(text="", error="not logged in")

    monkeypatch.setattr(
        "agent.transports.claude_agent_session.ClaudeAgentSession", _ErrSession
    )
    import pytest

    with pytest.raises(RuntimeError, match="not logged in"):
        ac._claude_agent_oneshot_response([{"role": "user", "content": "x"}])


def test_call_llm_routes_explicit_claude_agent_provider(monkeypatch):
    hit = {}

    def _fake(messages, model=None, timeout=None):
        hit["msgs"] = messages
        hit["model"] = model
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

    monkeypatch.setattr(ac, "_claude_agent_oneshot_response", _fake)
    resp = ac.call_llm(
        task="title_generation", provider="claude-agent",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert resp.choices[0].message.content == "ok"
    assert hit["msgs"] == [{"role": "user", "content": "hi"}]


def test_call_llm_routes_on_api_mode(monkeypatch):
    monkeypatch.setattr(
        ac, "_claude_agent_oneshot_response",
        lambda messages, model=None, timeout=None: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        ),
    )
    resp = ac.call_llm(
        task="title_generation", api_mode="claude_agent_sdk",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert resp.choices[0].message.content == "ok"


def test_call_llm_fallback_when_no_provider_but_claude_installed(monkeypatch):
    # No HTTP client resolvable, but the claude CLI is present → route to SDK
    # instead of raising "No LLM provider configured".
    monkeypatch.setattr(ac, "_get_cached_client", lambda *a, **k: (None, None))
    monkeypatch.setattr(ac, "_claude_agent_aux_available", lambda: True)
    monkeypatch.setattr(
        ac, "_claude_agent_oneshot_response",
        lambda messages, model=None, timeout=None: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="fallback-title"))]
        ),
    )
    resp = ac.call_llm(
        task="title_generation", provider="auto",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert resp.choices[0].message.content == "fallback-title"


def test_call_llm_still_raises_without_claude(monkeypatch):
    import pytest

    monkeypatch.setattr(ac, "_get_cached_client", lambda *a, **k: (None, None))
    monkeypatch.setattr(ac, "_claude_agent_aux_available", lambda: False)
    with pytest.raises(RuntimeError, match="No LLM provider configured"):
        ac.call_llm(
            task="title_generation", provider="auto",
            messages=[{"role": "user", "content": "hi"}],
        )
