import sys


def _make_session(tmp_path, **kw):
    from agent.transports.claude_agent_session import ClaudeAgentSession

    defaults = dict(model="claude-opus-4-8", cwd=str(tmp_path), enable_hermes_tools=False)
    defaults.update(kw)
    return ClaudeAgentSession(**defaults)


def test_import_without_sdk_installed(monkeypatch, tmp_path):
    """Module import and construction must not require claude_agent_sdk."""
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    session = _make_session(tmp_path)
    assert session is not None  # lazy import: no error until ensure_started


def test_ensure_started_connects_client(fake_claude_sdk, tmp_path):
    session = _make_session(tmp_path)
    session.ensure_started()
    try:
        assert len(fake_claude_sdk.clients) == 1
        assert fake_claude_sdk.clients[0].connected
        assert session.is_alive()
    finally:
        session.close()


def test_close_disconnects_and_stops_loop(fake_claude_sdk, tmp_path):
    session = _make_session(tmp_path)
    session.ensure_started()
    session.close()
    assert not fake_claude_sdk.clients[0].connected
    assert not session.is_alive()
    session.close()  # idempotent


def test_options_strip_api_key_and_set_bypass(fake_claude_sdk, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    session = _make_session(tmp_path, auto_approve=True)
    session.ensure_started()
    try:
        opts = fake_claude_sdk.clients[0].options
        assert opts.model == "claude-opus-4-8"
        assert opts.permission_mode == "bypassPermissions"
        assert opts.setting_sources == []
        assert opts.env.get("ANTHROPIC_API_KEY", "") == ""
    finally:
        session.close()
