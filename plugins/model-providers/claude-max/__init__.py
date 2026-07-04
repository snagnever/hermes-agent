"""Claude Max — Claude subscription via a local Anthropic-compatible proxy.

Unlike the plain ``anthropic`` provider (which needs an API key), ``claude-max``
points hermes at a local server (``hermes_cli.claude_max``) that speaks the
Anthropic Messages API but is backed by the Claude Agent SDK + your Claude
subscription. Selecting this provider auto-starts the server; hermes' entire
anthropic transport (streaming, tools, usage) then works unchanged.

There is no REST /models catalog on the subscription path, so ``fetch_models``
returns the static list from ``hermes_cli.claude_max.models_catalog`` (no HTTP).
"""

import os

from providers import register_provider
from providers.base import ProviderProfile

# Keep discovery-time imports light: the port constant is inlined (mirrors
# hermes_cli.claude_max.DEFAULT_PORT) so the plugin needn't import the package
# just to compute base_url; the model list is a pure-data leaf module.
_DEFAULT_PORT = 8646


def _port() -> int:
    raw = os.environ.get("HERMES_CLAUDE_MAX_PORT", "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return _DEFAULT_PORT


class ClaudeMaxProfile(ProviderProfile):
    def fetch_models(self, *, api_key=None, base_url=None, timeout=8.0):
        # Static catalog — the subscription path has no /models endpoint.
        from hermes_cli.claude_max.models_catalog import MODEL_IDS

        return list(MODEL_IDS)


claude_max = ClaudeMaxProfile(
    name="claude-max",
    aliases=("claudemax", "claude-max-proxy", "cmax"),
    display_name="Claude Max (local proxy)",
    description="Claude subscription via a local Anthropic-compatible proxy (Claude Agent SDK)",
    signup_url="https://claude.ai/settings/subscription",
    api_mode="anthropic_messages",
    env_vars=("CLAUDE_CODE_OAUTH_TOKEN", "HERMES_CLAUDE_MAX_PORT"),
    base_url=f"http://127.0.0.1:{_port()}",
    auth_type="oauth_external",
    supports_health_check=False,  # server may not be running yet; auto-start handles it
    supports_vision=True,
    fallback_models=(
        "claude-opus-4-8",
        "claude-sonnet-5",
        "claude-haiku-4-5",
        "opus",
        "sonnet",
    ),
)

register_provider(claude_max)
