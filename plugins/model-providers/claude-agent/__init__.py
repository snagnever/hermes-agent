"""Claude via the Claude Agent SDK (subscription OAuth) provider profile.

Auth is external: the Claude Code CLI's login (`claude` -> /login) or a
CLAUDE_CODE_OAUTH_TOKEN minted by `claude setup-token`. There is no REST
/models catalog on this path, so fetch_models returns None and the picker
uses fallback_models.
"""

from providers import register_provider
from providers.base import ProviderProfile


class ClaudeAgentProfile(ProviderProfile):
    def fetch_models(self, *, api_key=None, base_url=None, timeout=8.0):
        return None  # no REST catalog on subscription OAuth


claude_agent = ClaudeAgentProfile(
    name="claude-agent",
    aliases=("claude-sdk", "claude-subscription"),
    display_name="Claude (subscription)",
    description="Claude via the Claude Agent SDK using your Claude subscription",
    signup_url="https://claude.ai/settings/subscription",
    api_mode="claude_agent_sdk",
    env_vars=("CLAUDE_CODE_OAUTH_TOKEN",),
    base_url="",              # no HTTP endpoint — SDK spawns Claude Code
    auth_type="oauth_external",
    supports_health_check=False,
    supports_vision=True,
    fallback_models=(
        "claude-opus-4-8",
        "claude-sonnet-5",
        "claude-haiku-4-5",
        "opus",    # SDK aliases resolve to the subscription's latest
        "sonnet",
    ),
)

register_provider(claude_agent)
