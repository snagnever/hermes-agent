"""claude-max — a local Anthropic-Messages-API-compatible server backed by the
Claude Agent SDK (Claude subscription), so hermes' native anthropic transport
uses the subscription as if it were api.anthropic.com.

This package is intentionally light at import time (constants only) so the
provider plugin and CLI can import it cheaply. Server/SDK-heavy code lives in
submodules imported on demand.
"""

from __future__ import annotations

import os

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8646
PID_FILENAME = "claude-max-proxy.pid"
LOG_FILENAME = "claude-max-proxy.log"

PORT_ENV_VAR = "HERMES_CLAUDE_MAX_PORT"
HOST_ENV_VAR = "HERMES_CLAUDE_MAX_HOST"

# A dummy sk-ant-* key so hermes' anthropic adapter takes the x-api-key path;
# the local server accepts any key (localhost-only bind is the trust boundary).
LOCAL_API_KEY = "sk-ant-claude-max-local"


def resolved_port() -> int:
    """Effective port: HERMES_CLAUDE_MAX_PORT override, else DEFAULT_PORT."""
    raw = os.environ.get(PORT_ENV_VAR, "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return DEFAULT_PORT


def resolved_host() -> str:
    """Effective bind host — loopback only unless overridden (discouraged)."""
    return os.environ.get(HOST_ENV_VAR, "").strip() or DEFAULT_HOST


def base_url(port: int | None = None) -> str:
    return f"http://{DEFAULT_HOST}:{port or resolved_port()}"
