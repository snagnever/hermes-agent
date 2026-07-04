from types import SimpleNamespace
from unittest.mock import MagicMock


def _fake_agent():
    """A lightweight real-attribute agent so usage accounting runs
    deterministically (a MagicMock would make estimate_usage_cost flaky)."""
    return SimpleNamespace(
        session_api_calls=0,
        _iters_since_skill=0,
        _skill_nudge_interval=0,
        valid_tool_names=set(),
        model="claude-opus-4-8",
        provider="claude-agent",
        base_url="",
        api_key="",
        session_id=None,
        _session_db=None,
        _session_db_created=False,
        _claude_session=None,
        session_cwd="/tmp",
        system_prompt=None,
        tool_progress_callback=None,
        context_compressor=None,
        # session token counters
        session_prompt_tokens=0,
        session_completion_tokens=0,
        session_total_tokens=0,
        session_input_tokens=0,
        session_output_tokens=0,
        session_cache_read_tokens=0,
        session_cache_write_tokens=0,
        session_reasoning_tokens=0,
        session_estimated_cost_usd=0.0,
        session_cost_status=None,
        session_cost_source=None,
        # hooks that must be callable no-ops
        _sync_external_memory_for_turn=lambda **kw: None,
        _spawn_background_review=lambda **kw: None,
    )


def test_turn_returns_codex_shaped_dict():
    from agent.claude_runtime import run_claude_agent_sdk_turn
    from agent.transports.claude_agent_session import ClaudeTurnResult

    turn = ClaudeTurnResult(final_text="hi there", session_id="s1",
                            usage={"input_tokens": 3, "output_tokens": 2})
    fake_session = MagicMock()
    fake_session.run_turn.return_value = turn

    agent = _fake_agent()
    agent._claude_session = fake_session

    out = run_claude_agent_sdk_turn(
        agent,
        user_message="hello",
        original_user_message="hello",
        messages=[{"role": "user", "content": "hello"}],
        effective_task_id="t1",
    )
    assert out["final_response"] == "hi there"
    assert out["completed"] is True
    assert out["partial"] is False
    assert out["error"] is None
    assert out["api_calls"] == 1
    assert out["agent_persisted"] is True
    assert out["claude_session_id"] == "s1"
    assert agent.session_api_calls == 1


def test_turn_error_marks_partial():
    from agent.claude_runtime import run_claude_agent_sdk_turn
    from agent.transports.claude_agent_session import ClaudeTurnResult

    turn = ClaudeTurnResult(error="boom")
    fake_session = MagicMock()
    fake_session.run_turn.return_value = turn
    agent = _fake_agent()
    agent._claude_session = fake_session

    out = run_claude_agent_sdk_turn(
        agent, user_message="x", original_user_message="x",
        messages=[], effective_task_id="t1",
    )
    assert out["completed"] is False
    assert out["partial"] is True
    assert out["error"] == "boom"


def test_turn_projects_messages_and_ticks_skill_counter():
    from agent.claude_runtime import run_claude_agent_sdk_turn
    from agent.transports.claude_agent_session import ClaudeTurnResult

    projected = [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "t1", "type": "function",
                         "function": {"name": "Bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "[executed]"},
        {"role": "assistant", "content": "done"},
    ]
    turn = ClaudeTurnResult(final_text="done", session_id="s", tool_iterations=1,
                            projected_messages=projected, usage={})
    fake_session = MagicMock()
    fake_session.run_turn.return_value = turn
    agent = _fake_agent()
    agent._claude_session = fake_session

    messages = [{"role": "user", "content": "run ls"}]
    out = run_claude_agent_sdk_turn(
        agent, user_message="run ls", original_user_message="run ls",
        messages=messages, effective_task_id="t1",
    )
    assert out["messages"] is messages
    assert len(messages) == 4  # user + 3 projected
    assert agent._iters_since_skill == 1
