"""claude-max provider resolution: runtime short-circuit, picker, validation."""

import hermes_cli.runtime_provider as RP
from hermes_cli.models import validate_requested_model
from hermes_cli.providers import (
    determine_api_mode,
    normalize_provider,
)


def test_runtime_provider_short_circuits_and_autostarts(monkeypatch):
    started = {"n": 0}

    def fake_ensure(*a, **k):
        started["n"] += 1
        return 8646

    monkeypatch.setattr("hermes_cli.claude_max.manager.ensure_server_running", fake_ensure)

    for requested in ("claude-max", "claudemax", "cmax"):
        rt = RP.resolve_runtime_provider(requested=requested)
        assert rt["provider"] == "claude-max"
        assert rt["api_mode"] == "anthropic_messages"
        assert rt["base_url"] == "http://127.0.0.1:8646"
        assert rt["api_key"].startswith("sk-ant-")
    assert started["n"] == 3  # auto-start invoked each time (idempotent server-side)


def test_aliases_and_api_mode():
    assert normalize_provider("claudemax") == "claude-max"
    assert normalize_provider("cmax") == "claude-max"
    assert determine_api_mode("claude-max") == "anthropic_messages"


def test_validation_accepts_without_probe():
    for model in ("opus", "claude-opus-4-8"):
        v = validate_requested_model(model, "claude-max",
                                     api_key="sk-ant-x", base_url="http://127.0.0.1:8646",
                                     api_mode="anthropic_messages")
        assert v["accepted"] is True
        assert v["message"] is None


def test_provider_known_to_picker():
    from hermes_cli.providers import get_provider, _LABEL_OVERRIDES

    assert get_provider("claude-max") is not None
    assert _LABEL_OVERRIDES["claude-max"] == "Claude Max (subscription proxy)"


def test_claude_max_appears_in_model_picker_when_cli_installed(monkeypatch):
    # The /model picker lists claude-max when the `claude` CLI is available
    # (its auth is in the CLI, not env). Mock the check for hermeticity.
    monkeypatch.setattr(
        "hermes_cli.claude_max.manager.check_claude_binary",
        lambda *a, **k: (True, "claude 2.x"),
    )
    from hermes_cli.model_switch import list_picker_providers

    rows = list_picker_providers()
    row = next((r for r in rows if r.get("slug") == "claude-max"), None)
    assert row is not None, f"claude-max missing; got {[r.get('slug') for r in rows]}"
    assert row["name"] == "Claude Max (subscription proxy)"
    assert row["models"] and any("opus" in m for m in row["models"])


def test_claude_max_hidden_from_picker_without_cli(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.claude_max.manager.check_claude_binary",
        lambda *a, **k: (False, "not found"),
    )
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("HERMES_CLAUDE_MAX_PORT", raising=False)
    from hermes_cli.model_switch import list_picker_providers

    rows = list_picker_providers()
    assert not any(r.get("slug") == "claude-max" for r in rows)
