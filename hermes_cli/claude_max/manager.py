"""Lifecycle manager for the claude-max local server: preflight, auto-start on
demand, health, PID tracking, stop/status.

Uses stdlib urllib for health checks so it can run inside the main hermes
process (which may not have aiohttp loaded). Spawns the server as a detached
subprocess (``python -m hermes_cli.claude_max.server``).
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from hermes_cli.claude_max import (
    LOG_FILENAME,
    PID_FILENAME,
    resolved_host,
    resolved_port,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def check_claude_binary(claude_bin: str = "claude") -> tuple[bool, str]:
    """Verify the Claude Code CLI is installed. Returns (ok, message)."""
    try:
        proc = subprocess.run(
            [claude_bin, "--version"],
            capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return False, (
            f"claude CLI not found at {claude_bin!r}. Install: "
            f"npm install -g @anthropic-ai/claude-code — then run `claude` and "
            f"log in with your Claude subscription."
        )
    except subprocess.TimeoutExpired:
        return False, "claude --version timed out"
    if proc.returncode != 0:
        return False, f"claude --version exited {proc.returncode}: {proc.stderr.strip()}"
    return True, proc.stdout.strip()


def _preflight() -> None:
    ok, msg = check_claude_binary()
    if not ok:
        raise RuntimeError(msg)
    try:
        import claude_agent_sdk  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "claude-max requires the claude-agent-sdk: "
            "pip install 'hermes-agent[claude-max]'"
        ) from exc
    try:
        import aiohttp  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "claude-max requires aiohttp: pip install 'hermes-agent[claude-max]'"
        ) from exc


# ---------------------------------------------------------------------------
# paths / pid record
# ---------------------------------------------------------------------------

def _hermes_home() -> Path:
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home())


def _pid_path() -> Path:
    return _hermes_home() / PID_FILENAME


def _log_path() -> Path:
    logs = _hermes_home() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    return logs / LOG_FILENAME


def _read_pid_record() -> Optional[dict]:
    try:
        return json.loads(_pid_path().read_text())
    except (FileNotFoundError, ValueError):
        return None


def _write_pid_record(pid: int, port: int, host: str) -> None:
    _pid_path().write_text(json.dumps(
        {"pid": pid, "port": port, "host": host, "started_at": time.time()}
    ))


def _clear_pid_record() -> None:
    try:
        _pid_path().unlink()
    except FileNotFoundError:
        pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------

def _health_ok(port: int, host: str = "127.0.0.1", timeout: float = 1.5) -> bool:
    url = f"http://{host}:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return False
            body = json.loads(resp.read().decode())
            return body.get("status") == "ok"
    except (urllib.error.URLError, OSError, ValueError):
        return False


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------

def ensure_server_running(port: Optional[int] = None, *, timeout: float = 15.0) -> int:
    """Return the port of a healthy claude-max server, starting one if needed.

    Idempotent: if a server already answers /health, returns immediately.
    Raises RuntimeError with an actionable message if preflight fails or the
    server doesn't come up in time.
    """
    host = resolved_host()
    port = port or resolved_port()

    if _health_ok(port, host):
        return port

    _preflight()

    log_path = _log_path()
    logf = open(log_path, "ab")
    proc = subprocess.Popen(
        [sys.executable, "-m", "hermes_cli.claude_max.server",
         "--host", host, "--port", str(port)],
        stdin=subprocess.DEVNULL, stdout=logf, stderr=logf,
        start_new_session=True,
        cwd=os.getcwd(),
    )
    _write_pid_record(proc.pid, port, host)
    logger.info("claude-max: spawned server pid=%s port=%s (logs: %s)", proc.pid, port, log_path)

    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = _log_tail(log_path)
            _clear_pid_record()
            raise RuntimeError(
                f"claude-max server exited immediately (code {proc.returncode}).\n{tail}"
            )
        if _health_ok(port, host):
            return port
        time.sleep(0.25)

    raise RuntimeError(
        f"claude-max server did not become healthy within {timeout}s "
        f"(logs: {log_path})"
    )


def _log_tail(path: Path, lines: int = 15) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except Exception:
        return "(no logs)"


def get_status() -> dict:
    rec = _read_pid_record()
    port = (rec or {}).get("port") or resolved_port()
    host = (rec or {}).get("host") or resolved_host()
    healthy = _health_ok(port, host)
    pid = (rec or {}).get("pid")
    return {
        "running": healthy,
        "healthy": healthy,
        "port": port,
        "host": host,
        "pid": pid,
        "pid_alive": bool(pid and _pid_alive(pid)),
        "log": str(_log_path()),
    }


def stop_server(*, timeout: float = 5.0) -> bool:
    rec = _read_pid_record()
    if not rec or not rec.get("pid"):
        return False
    pid = int(rec["pid"])
    if not _pid_alive(pid):
        _clear_pid_record()
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        _clear_pid_record()
        return False
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pid_alive(pid):
            break
        time.sleep(0.1)
    if _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    _clear_pid_record()
    return True
