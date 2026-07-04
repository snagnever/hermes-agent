"""claude-agent has no HTTP endpoint, so the vision/local-server probes must
never touch a base_url for it (a stale one would hang startup on connect)."""

from agent.image_routing import (
    _resolve_inference_base_url,
    _should_probe_ollama_vision,
)


def test_should_not_probe_for_claude_agent_even_with_stale_base_url():
    # A leftover LM Studio URL from a previous provider must not be probed.
    for prov in ("claude-agent", "claude-sdk", "claude-subscription"):
        assert _should_probe_ollama_vision(prov, "http://192.168.68.107:1234/v1") is False


def test_resolve_base_url_empty_for_claude_agent_provider_arg():
    cfg = {"model": {"provider": "claude-agent", "base_url": "http://192.168.68.107:1234/v1"}}
    assert _resolve_inference_base_url(cfg, "claude-agent") == ""


def test_resolve_base_url_empty_for_claude_agent_from_config():
    # Even when provider arg is empty, the config provider gates it.
    cfg = {"model": {"provider": "claude-agent", "base_url": "http://192.168.68.107:1234/v1"}}
    assert _resolve_inference_base_url(cfg, "") == ""


def test_resolve_base_url_unchanged_for_normal_provider():
    cfg = {"model": {"provider": "lmstudio", "base_url": "http://localhost:1234/v1"}}
    assert _resolve_inference_base_url(cfg, "lmstudio") == "http://localhost:1234/v1"
