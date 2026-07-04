from hermes_cli.runtime_provider import (
    _VALID_API_MODES,
    _maybe_apply_claude_agent_runtime,
)


def test_claude_agent_sdk_is_valid_api_mode():
    assert "claude_agent_sdk" in _VALID_API_MODES


def test_gate_rewrites_for_anthropic_provider_when_opted_in():
    out = _maybe_apply_claude_agent_runtime(
        provider="anthropic",
        api_mode="anthropic_messages",
        model_cfg={"anthropic_runtime": "claude_agent_sdk"},
    )
    assert out == "claude_agent_sdk"


def test_gate_rewrites_for_claude_agent_provider():
    out = _maybe_apply_claude_agent_runtime(
        provider="claude-agent",
        api_mode="chat_completions",
        model_cfg={"anthropic_runtime": "claude_agent_sdk"},
    )
    assert out == "claude_agent_sdk"


def test_claude_agent_provider_routes_without_flag():
    # Selecting the claude-agent provider (e.g. via /model) always routes
    # through the SDK, independent of any config flag.
    for cfg in (None, {}, {"anthropic_runtime": ""}, {"anthropic_runtime": "auto"}):
        out = _maybe_apply_claude_agent_runtime(
            provider="claude-agent", api_mode="chat_completions", model_cfg=cfg
        )
        assert out == "claude_agent_sdk"


def test_gate_noop_when_unset_or_auto():
    for cfg in (None, {}, {"anthropic_runtime": ""}, {"anthropic_runtime": "auto"}):
        out = _maybe_apply_claude_agent_runtime(
            provider="anthropic", api_mode="anthropic_messages", model_cfg=cfg
        )
        assert out == "anthropic_messages"


def test_gate_noop_for_other_providers():
    out = _maybe_apply_claude_agent_runtime(
        provider="openrouter",
        api_mode="chat_completions",
        model_cfg={"anthropic_runtime": "claude_agent_sdk"},
    )
    assert out == "chat_completions"
