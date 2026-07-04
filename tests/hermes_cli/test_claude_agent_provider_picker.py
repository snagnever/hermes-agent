"""The claude-agent provider is a first-class /model picker option that
resolves to the claude_agent_sdk runtime."""

from hermes_cli.providers import (
    HERMES_OVERLAYS,
    TRANSPORT_TO_API_MODE,
    determine_api_mode,
    get_provider,
    normalize_provider,
)


def test_overlay_registered_with_sdk_transport():
    overlay = HERMES_OVERLAYS.get("claude-agent")
    assert overlay is not None
    assert overlay.transport == "claude_agent_sdk"
    assert overlay.auth_type == "oauth_external"
    assert "CLAUDE_CODE_OAUTH_TOKEN" in overlay.extra_env_vars


def test_transport_maps_to_claude_agent_sdk_api_mode():
    assert TRANSPORT_TO_API_MODE["claude_agent_sdk"] == "claude_agent_sdk"


def test_aliases_normalize_to_claude_agent():
    assert normalize_provider("claude-sdk") == "claude-agent"
    assert normalize_provider("claude-subscription") == "claude-agent"


def test_determine_api_mode_resolves_sdk():
    assert determine_api_mode("claude-agent") == "claude_agent_sdk"
    assert determine_api_mode("claude-sdk") == "claude_agent_sdk"


def test_provider_is_known_to_picker():
    # get_provider returns a real ProviderDef (not None) so the switch/picker
    # treats claude-agent as a known provider.
    assert get_provider("claude-agent") is not None


def test_label_is_subscription_friendly():
    from hermes_cli.providers import _LABEL_OVERRIDES

    assert _LABEL_OVERRIDES["claude-agent"] == "Claude (subscription)"


def test_resolve_runtime_provider_short_circuits_without_credentials():
    # /model switch path: resolve_runtime_provider must return a keyless SDK
    # runtime instead of raising "Unknown provider" from the credential pool.
    from hermes_cli.runtime_provider import resolve_runtime_provider

    for requested in ("claude-agent", "claude-sdk", "claude-subscription"):
        rt = resolve_runtime_provider(requested=requested, target_model="opus")
        assert rt["provider"] == "claude-agent"
        assert rt["api_mode"] == "claude_agent_sdk"
        assert rt["base_url"] == ""
        assert rt["api_key"]  # a non-empty sentinel, never an empty string


def test_switch_validation_accepts_claude_agent_models():
    from hermes_cli.models import validate_requested_model

    for model in ("opus", "sonnet", "claude-opus-4-8"):
        v = validate_requested_model(
            model, "claude-agent",
            api_key="no-key-required", base_url="", api_mode="claude_agent_sdk",
        )
        assert v.get("accepted") is True
        assert v.get("recognized") is True
        # No network-probe warning — there is no /models endpoint by design.
        assert v.get("message") is None
