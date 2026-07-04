"""Wires the session registry + real ClaudeBridge factory into the aiohttp app.

Kept separate from server.py so the skeleton (server.create_app) stays
importable without the SDK; build_app() pulls in the bridge/handler.
"""

from __future__ import annotations

from hermes_cli.claude_max.bridge import ClaudeBridge
from hermes_cli.claude_max.handler import handle_messages
from hermes_cli.claude_max.server import create_app
from hermes_cli.claude_max.sessions import SessionRegistry
from hermes_cli.claude_max.translator import build_options_inputs


def _bridge_factory(payload: dict) -> ClaudeBridge:
    return ClaudeBridge(build_options_inputs(payload))


def build_app():
    registry = SessionRegistry(_bridge_factory)
    return create_app(registry=registry, messages_handler=handle_messages)
