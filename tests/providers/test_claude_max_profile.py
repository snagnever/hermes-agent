"""Registry contract for the claude-max provider plugin."""


def test_claude_max_profile_registered():
    from providers import get_provider_profile

    p = get_provider_profile("claude-max")
    assert p is not None
    assert p.api_mode == "anthropic_messages"
    assert p.auth_type == "oauth_external"
    assert p.supports_health_check is False
    assert p.base_url.startswith("http://127.0.0.1:")


def test_claude_max_aliases_resolve():
    from providers import get_provider_profile

    for alias in ("claudemax", "claude-max-proxy", "cmax"):
        assert get_provider_profile(alias) is get_provider_profile("claude-max")


def test_claude_max_fetch_models_is_static_no_http():
    from providers import get_provider_profile

    p = get_provider_profile("claude-max")
    models = p.fetch_models()  # no base_url/api_key → must NOT hit the network
    assert isinstance(models, list) and models
    assert any("opus" in m for m in models)


def test_claude_max_fallback_models_present():
    from providers import get_provider_profile

    p = get_provider_profile("claude-max")
    assert len(p.fallback_models) >= 3
    assert any("opus" in m for m in p.fallback_models)
