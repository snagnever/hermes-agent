"""Registry contract for the claude-agent provider plugin."""


def test_claude_agent_profile_registered():
    from providers import get_provider_profile

    profile = get_provider_profile("claude-agent")
    assert profile is not None
    assert profile.api_mode == "claude_agent_sdk"
    assert profile.auth_type == "oauth_external"
    assert profile.supports_health_check is False  # no REST /models probe
    assert "CLAUDE_CODE_OAUTH_TOKEN" in profile.env_vars
    assert profile.supports_vision is True


def test_claude_agent_aliases_resolve():
    from providers import get_provider_profile

    for alias in ("claude-sdk", "claude-subscription"):
        assert get_provider_profile(alias) is get_provider_profile("claude-agent")


def test_claude_agent_fallback_models_nonempty():
    from providers import get_provider_profile

    profile = get_provider_profile("claude-agent")
    assert len(profile.fallback_models) >= 3
    assert any("opus" in m for m in profile.fallback_models)


def test_claude_agent_fetch_models_returns_none():
    from providers import get_provider_profile

    profile = get_provider_profile("claude-agent")
    # No REST catalog on subscription OAuth — picker must fall back.
    assert profile.fetch_models() is None
