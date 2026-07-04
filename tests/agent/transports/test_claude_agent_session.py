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


def test_run_turn_returns_text_and_usage(fake_claude_sdk, tmp_path):
    m = fake_claude_sdk.module  # stub classes live on the injected module
    AssistantMessage, ResultMessage, TextBlock = (
        m.AssistantMessage, m.ResultMessage, m.TextBlock,
    )

    session = _make_session(tmp_path)
    session.ensure_started()
    try:
        fake_claude_sdk.script_turn([
            AssistantMessage(content=[TextBlock("Hello from Claude")]),
            ResultMessage(
                session_id="sess-abc",
                result="Hello from Claude",
                usage={"input_tokens": 10, "output_tokens": 5,
                       "cache_read_input_tokens": 2, "cache_creation_input_tokens": 1},
            ),
        ])
        result = session.run_turn("hi")
        assert result.error is None
        assert result.final_text == "Hello from Claude"
        assert result.session_id == "sess-abc"
        assert result.usage["input_tokens"] == 10
        assert fake_claude_sdk.clients[0].queries == ["hi"]
    finally:
        session.close()


def test_run_turn_projects_tool_use(fake_claude_sdk, tmp_path):
    m = fake_claude_sdk.module
    AssistantMessage, ResultMessage, TextBlock, ToolUseBlock = (
        m.AssistantMessage, m.ResultMessage, m.TextBlock, m.ToolUseBlock,
    )

    session = _make_session(tmp_path)
    seen = []
    session.tool_progress_callback = lambda name, preview, args: seen.append(name)
    session.ensure_started()
    try:
        fake_claude_sdk.script_turn([
            AssistantMessage(content=[ToolUseBlock(id="t1", name="Bash", input={"command": "ls"})]),
            AssistantMessage(content=[TextBlock("done")]),
            ResultMessage(session_id="s", result="done", usage={}),
        ])
        result = session.run_turn("list files")
        assert result.tool_iterations == 1
        assert seen == ["Bash"]
        assert result.final_text == "done"
        roles = [msg["role"] for msg in result.projected_messages]
        assert "assistant" in roles and "tool" in roles
    finally:
        session.close()


def test_run_turn_error_is_captured_not_raised(fake_claude_sdk, tmp_path):
    ResultMessage = fake_claude_sdk.module.ResultMessage

    session = _make_session(tmp_path)
    session.ensure_started()
    try:
        fake_claude_sdk.script_turn([
            ResultMessage(subtype="error_during_execution", is_error=True,
                          session_id="s", result=None, usage=None),
        ])
        result = session.run_turn("boom")
        assert result.error is not None
        assert "error_during_execution" in result.error
    finally:
        session.close()
