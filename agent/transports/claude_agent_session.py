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
        approval_callback = self._approval_callback

        async def _handler(tool_name: str, input_data: dict, context: Any):
            import asyncio as _asyncio

            from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

            # Hermes' own tools re-enter Hermes dispatch, which applies its
            # own approval policy — don't double-gate.
            if tool_name.startswith("mcp__hermes-tools__"):
                return PermissionResultAllow(updated_input=input_data)
            if approval_callback is None:
                # Gateway/cron: no UI to prompt through — fail closed,
                # mirroring the codex runtime's default.
                return PermissionResultDeny(
                    message="no Hermes approval UI available; denied by policy",
                )
            command, description = _describe_tool_for_approval(tool_name, input_data)
            try:
                # Hermes approval callbacks are per-thread/UI-bound and block;
                # run off the event loop. Signature matches the codex path:
                # (command, description, allow_permanent=False) -> choice str.
                choice = await _asyncio.to_thread(
                    approval_callback, command, description, allow_permanent=False
                )
            except Exception:
                logger.debug("hermes approval callback raised", exc_info=True)
                return PermissionResultDeny(message="approval callback error")
            if str(choice).strip().lower() in _APPROVE_CHOICES:
                return PermissionResultAllow(updated_input=input_data)
            return PermissionResultDeny(message=f"denied by user ({choice})")

        return _handler

    def _run(self, coro, timeout: float):
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout)

    # ---------- turn driving ----------

    def run_turn(self, user_input: Any, *, turn_timeout: float = 600.0) -> ClaudeTurnResult:
        """Send one user message; block until the SDK's ResultMessage."""
        result = ClaudeTurnResult()
        try:
            self.ensure_started()
        except (RuntimeError, TimeoutError) as exc:
            result.error = f"claude-agent-sdk startup failed: {exc}"
            result.should_retire = True
            return result
        text = user_input if isinstance(user_input, str) else _coerce_input_text(user_input)
        self._active_result = result
        try:
            self._run(self._turn_coro(text, result), timeout=turn_timeout)
        except TimeoutError:
            result.error = f"claude-agent-sdk turn timed out after {turn_timeout}s"
            result.should_retire = True
        except Exception as exc:
            result.error = f"claude-agent-sdk turn failed: {exc}"
            result.should_retire = True
        finally:
            self._active_result = None
        if result.session_id:
            self._session_id = result.session_id
        return result

    async def _turn_coro(self, text: str, result: ClaudeTurnResult) -> None:
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            TextBlock,
            ToolUseBlock,
        )

        await self._client.query(text)
        last_text_parts: list[str] = []
        async for message in self._client.receive_response():
            if isinstance(message, AssistantMessage):
                text_parts: list[str] = []
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        result.tool_iterations += 1
                        self._emit_tool_progress(block)
                        result.projected_messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": _json_dumps(block.input),
                                },
                            }],
                        })
                        # Claude Code executes the tool itself; project a
                        # synthetic ack so the transcript stays well-formed.
                        result.projected_messages.append({
                            "role": "tool",
                            "tool_call_id": block.id,
                            "content": "[executed inside Claude Code runtime]",
                        })
                if text_parts:
                    last_text_parts = text_parts
                    result.projected_messages.append(
                        {"role": "assistant", "content": "".join(text_parts)}
                    )
            elif isinstance(message, ResultMessage):
                result.session_id = message.session_id
                result.usage = message.usage or {}
                if message.is_error:
                    result.error = f"claude-agent-sdk: {message.subtype}"
                final = message.result or "".join(last_text_parts)
                result.final_text = final or ""

    def _emit_tool_progress(self, block) -> None:
        callback = self.tool_progress_callback
        if callback is None:
            return
        try:
            preview = _json_dumps(block.input)[:120]
            callback(block.name, preview, dict(block.input or {}))
        except Exception:
            logger.debug("tool progress callback raised", exc_info=True)

    def request_interrupt(self) -> None:
        """Thread-safe: interrupt the in-flight turn (no-op when idle)."""
        if self._client is None or self._loop is None or self._closed:
            return
        active = self._active_result
        if active is not None:
            active.interrupted = True
        asyncio.run_coroutine_threadsafe(self._client.interrupt(), self._loop)

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


# Hermes approval choices that grant permission. The Hermes approval
# callback returns 'once' | 'session' | 'always' | 'deny' (see
# codex_app_server_session._approval_choice_to_codex_decision); the extra
# synonyms keep alternate callbacks working.
_APPROVE_CHOICES = frozenset(
    {"once", "session", "always", "approve", "approved", "yes", "allow"}
)


def _describe_tool_for_approval(tool_name: str, input_data: Any) -> tuple[str, str]:
    """Render a Claude Code tool call into (command, description) for the
    Hermes approval prompt, mirroring how the codex path labels exec/patch
    requests so the user sees what's actually about to run."""
    data = input_data if isinstance(input_data, dict) else {}
    if tool_name in {"Bash", "BashOutput"} and data.get("command"):
        command = str(data["command"])
    elif data.get("file_path"):
        command = f"{tool_name}: {data['file_path']}"
    else:
        rendered = _json_dumps(data)[:160]
        command = f"{tool_name} {rendered}".strip()
    return command, f"Claude Code requests {tool_name}"


def check_claude_binary(claude_bin: str = "claude") -> tuple[bool, str]:
    """Verify the Claude Code CLI is installed. Returns (ok, message)."""
    import subprocess

    try:
        proc = subprocess.run(
            [claude_bin, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return False, (
            f"claude CLI not found at {claude_bin!r}. Install with: "
            f"npm install -g @anthropic-ai/claude-code — then run `claude` "
            f"once and log in with your Claude subscription."
        )
    except subprocess.TimeoutExpired:
        return False, "claude --version timed out"
    if proc.returncode != 0:
        return False, f"claude --version exited {proc.returncode}: {proc.stderr.strip()}"
    return True, proc.stdout.strip()


def _json_dumps(obj: Any) -> str:
    import json

    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


def _coerce_input_text(user_input: Any) -> str:
    """Collapse Hermes/OpenAI rich content into plain turn text — mirrors
    codex_app_server_session._coerce_turn_input_text so images degrade to
    their text parts identically."""
    if isinstance(user_input, list):
        parts: list[str] = []
        for item in user_input:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item)
                continue
            if not isinstance(item, dict):
                if item is not None:
                    parts.append(str(item))
                continue
            item_type = item.get("type")
            if item_type in {"text", "input_text"}:
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            elif item_type in {"image", "image_url", "input_image"}:
                parts.append("[image attached]")
        return "\n\n".join(p for p in parts if p).strip()
    return "" if user_input is None else str(user_input)
