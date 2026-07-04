"""Local Anthropic-Messages-API-compatible origin server backed by the Claude
Agent SDK. hermes' anthropic transport points here (base_url) and the entire
existing anthropic path (streaming, tools, usage) works unchanged.

Unlike ``hermes_cli/proxy`` (a credential-forwarding proxy to an upstream),
this is an ORIGIN server: it terminates the Anthropic API and produces
responses from the SDK. Routes: POST /v1/messages, GET /v1/models, GET /health.

Run standalone (used by the auto-start manager):
    python -m hermes_cli.claude_max.server --host 127.0.0.1 --port 8646
"""

from __future__ import annotations

import argparse
import ipaddress
import logging
from typing import Any, Optional

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover
    web = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

from hermes_cli.claude_max import DEFAULT_HOST, DEFAULT_PORT, resolved_host, resolved_port
from hermes_cli.claude_max.models_catalog import models_response

logger = logging.getLogger(__name__)

if AIOHTTP_AVAILABLE:
    REGISTRY_KEY: "web.AppKey" = web.AppKey("registry", object)
    MESSAGES_HANDLER_KEY: "web.AppKey" = web.AppKey("messages_handler", object)

_AIOHTTP_HINT = (
    "aiohttp is required for the claude-max proxy. Install with: "
    "pip install 'hermes-agent[claude-max]' (or pip install aiohttp)."
)


def anthropic_error(status: int, err_type: str, message: str) -> "web.Response":
    """Anthropic-shaped error JSON: {"type":"error","error":{"type","message"}}."""
    return web.json_response(
        {"type": "error", "error": {"type": err_type, "message": message}},
        status=status,
    )


async def handle_health(request: "web.Request") -> "web.Response":
    return web.json_response({"status": "ok", "service": "claude-max"})


async def handle_models(request: "web.Request") -> "web.Response":
    raw = request.query.get("limit")
    limit: Optional[int]
    try:
        limit = int(raw) if raw is not None else None
    except ValueError:
        limit = None
    return web.json_response(models_response(limit))


async def handle_messages(request: "web.Request") -> "web.StreamResponse":
    """Placeholder — real implementation wired in the /v1/messages task.

    Kept here so the route exists in the skeleton; returns a clear error until
    the session-bridge handler is installed via create_app(registry=...).
    """
    handler = request.app.get(MESSAGES_HANDLER_KEY)
    if handler is None:
        return anthropic_error(
            503, "api_error",
            "claude-max server started without a message handler (not yet wired).",
        )
    return await handler(request)


async def _error_middleware(request: "web.Request", handler):
    """Turn 404/405 and uncaught errors into Anthropic-shaped error JSON."""
    try:
        return await handler(request)
    except web.HTTPNotFound:
        return anthropic_error(404, "not_found_error", f"Unknown route: {request.path}")
    except web.HTTPMethodNotAllowed:
        return anthropic_error(405, "invalid_request_error", "Method not allowed")
    except web.HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("claude-max unhandled error")
        return anthropic_error(500, "api_error", f"internal error: {exc}")


def create_app(registry: Any = None, messages_handler: Any = None) -> "web.Application":
    """Build the aiohttp app. ``registry``/``messages_handler`` are injected by
    the /v1/messages wiring task; models + health work without them."""
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError(_AIOHTTP_HINT)
    app = web.Application(middlewares=[web.middleware(_error_middleware)])
    app[REGISTRY_KEY] = registry
    app[MESSAGES_HANDLER_KEY] = messages_handler
    app.router.add_post("/v1/messages", handle_messages)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/health", handle_health)
    return app


def _require_loopback(host: str) -> None:
    """Refuse non-loopback binds — the socket has no auth; localhost is the
    trust boundary."""
    try:
        ip = ipaddress.ip_address(host)
        is_loopback = ip.is_loopback
    except ValueError:
        is_loopback = host in {"localhost"}
    if not is_loopback:
        raise SystemExit(
            f"claude-max refuses to bind non-loopback host {host!r} "
            f"(the server is unauthenticated; use 127.0.0.1)."
        )


def run_server(host: Optional[str] = None, port: Optional[int] = None) -> None:
    """Blocking server run (foreground). Builds the real registry-backed app."""
    if not AIOHTTP_AVAILABLE:
        raise SystemExit(_AIOHTTP_HINT)
    host = host or resolved_host()
    port = port or resolved_port()
    _require_loopback(host)

    # Lazy: the session-bridge app factory (imports the SDK) lives in the
    # wiring task; fall back to the skeleton app until then.
    try:
        from hermes_cli.claude_max.app import build_app  # type: ignore

        app = build_app()
    except Exception:
        app = create_app()

    logger.info("claude-max serving on http://%s:%d", host, port)
    web.run_app(app, host=host, port=port, print=None)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="claude-max-server")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)
    run_server(args.host, args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
