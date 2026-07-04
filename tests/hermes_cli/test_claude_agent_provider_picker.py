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
