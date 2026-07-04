"""Claude Agent SDK session adapter for the claude_agent_sdk runtime.

Sibling of codex_app_server_session.py: Hermes' AIAgent loop is
synchronous, the Claude Agent SDK is asyncio — so we run one event loop
on a dedicated daemon thread per session and drive coroutines with
run_coroutine_threadsafe. One ClaudeAgentSession per AIAgent instance,
reused across turns (the SDK client keeps conversation context).

Auth is the Claude Code credential chain (subscription login or
CLAUDE_CODE_OAUTH_TOKEN). ANTHROPIC_API_KEY is deliberately blanked in
the child env so a stray key never flips billing to pay-per-token.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

_START_TIMEOUT = 30.0


@dataclass
class ClaudeTurnResult:
    """Result of one user→assistant→tools turn through the Agent SDK."""

    final_text: str = ""
    projected_messages: list[dict] = field(default_factory=list)
    tool_iterations: int = 0
    interrupted: bool = False
    error: Optional[str] = None
    session_id: Optional[str] = None
    usage: Optional[dict[str, Any]] = None
    should_retire: bool = False


class ClaudeAgentSession:
    def __init__(
        self,
        *,
        model: str,
        cwd: str,
        system_prompt: Optional[str] = None,
        approval_callback=None,
        auto_approve: bool = False,
        enable_hermes_tools: bool = True,
        env: Optional[dict[str, str]] = None,
    ) -> None:
        self._model = model
        self._cwd = cwd
        self._system_prompt = system_prompt
        self._approval_callback = approval_callback
        self._auto_approve = auto_approve
        self._enable_hermes_tools = enable_hermes_tools
        self._extra_env = dict(env or {})
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._client: Any = None
        self._session_id: Optional[str] = None
        self._closed = False
        # Set by run_turn (Task 5); consumed by request_interrupt (Task 7).
        self._active_result: Optional[ClaudeTurnResult] = None
        # Optional (tool_name, preview, args) display hook, set by the runtime.
        self.tool_progress_callback: Any = None

    # ---------- lifecycle ----------

    def ensure_started(self) -> None:
        if self._client is not None and not self._closed:
            return
        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "claude_agent_sdk runtime requires the 'claude-agent-sdk' "
                "package: pip install claude-agent-sdk (and the Claude Code "
                "CLI, logged in with your Claude subscription)"
            ) from exc

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="claude-agent-sdk", daemon=True
        )
        self._thread.start()
        self._closed = False

        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

        options = ClaudeAgentOptions(**self._build_options_kwargs())
        self._client = ClaudeSDKClient(options=options)
        self._run(self._client.connect(), timeout=_START_TIMEOUT)

    def _build_options_kwargs(self) -> dict[str, Any]:
        # Subscription-auth hygiene: never let a stray API key hijack billing.
        child_env = {"ANTHROPIC_API_KEY": "", **self._extra_env}
        kwargs: dict[str, Any] = {
            "model": self._model,
            "cwd": self._cwd,
            "setting_sources": [],  # isolate from the user's personal ~/.claude config
            "env": child_env,
            "permission_mode": "bypassPermissions" if self._auto_approve else "default",
            "mcp_servers": self._mcp_server_config(),
        }
        if self._system_prompt:
            kwargs["system_prompt"] = self._system_prompt
        if self._session_id:
            kwargs["resume"] = self._session_id  # survive session respawn
        if not self._auto_approve and self._approval_callback is not None:
            kwargs["can_use_tool"] = self._make_permission_handler()
        return kwargs

    def _mcp_server_config(self) -> dict[str, Any]:
        if not self._enable_hermes_tools:
            return {}
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        return {
            "hermes-tools": {
                "type": "stdio",
                "command": sys.executable,
                "args": ["-m", "agent.transports.hermes_tools_mcp_server"],
                "env": {
                    "PYTHONPATH": repo_root,
                    "HERMES_QUIET": "1",
                    "HERMES_REDACT_SECRETS": "true",
                },
            }
        }

    def _make_permission_handler(self):
        # Filled in by Task 6; lifecycle task ships a permissive stub.
        async def _handler(tool_name, input_data, context):  # pragma: no cover
            from claude_agent_sdk import PermissionResultAllow

            return PermissionResultAllow(updated_input=input_data)

        return _handler

    def _run(self, coro, timeout: float):
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout)

    def is_alive(self) -> bool:
        return (
            not self._closed
            and self._client is not None
            and self._thread is not None
            and self._thread.is_alive()
        )

    def close(self, timeout: float = 5.0) -> None:
        if self._closed:
            return
        self._closed = True
        if self._client is not None and self._loop is not None:
            try:
                self._run(self._client.disconnect(), timeout=timeout)
            except Exception:
                logger.debug("claude-agent disconnect failed", exc_info=True)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._client = None

    def __enter__(self) -> "ClaudeAgentSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
