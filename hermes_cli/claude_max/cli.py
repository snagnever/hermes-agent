"""`hermes claude-max serve|stop|status` — manage the local Claude-subscription
proxy. Auto-start also happens implicitly when the claude-max provider is used;
these commands are for explicit control + diagnostics.
"""

from __future__ import annotations

import json as _json


def build_claude_max_parser(subparsers, *, cmd_claude_max) -> None:
    p = subparsers.add_parser(
        "claude-max",
        help="Manage the local Claude-subscription proxy (Anthropic-compatible)",
    )
    p.set_defaults(func=cmd_claude_max)
    sub = p.add_subparsers(dest="claude_max_command")

    serve = sub.add_parser("serve", help="Run the proxy in the foreground")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    sub.add_parser("start", help="Start the proxy in the background (auto-start)")
    sub.add_parser("stop", help="Stop the background proxy")
    sub.add_parser("status", help="Show proxy status")


def cmd_claude_max(args) -> int:
    command = getattr(args, "claude_max_command", None) or "status"

    if command == "serve":
        from hermes_cli.claude_max.server import run_server

        run_server(getattr(args, "host", None), getattr(args, "port", None))
        return 0

    if command == "start":
        from hermes_cli.claude_max.manager import ensure_server_running

        try:
            port = ensure_server_running()
        except RuntimeError as exc:
            print(f"claude-max: {exc}")
            return 1
        print(f"claude-max proxy running on http://127.0.0.1:{port}")
        return 0

    if command == "stop":
        from hermes_cli.claude_max.manager import stop_server

        print("claude-max proxy stopped" if stop_server() else "claude-max proxy was not running")
        return 0

    # status (default)
    from hermes_cli.claude_max.manager import get_status

    status = get_status()
    print(_json.dumps(status, indent=2))
    return 0 if status["running"] else 1
