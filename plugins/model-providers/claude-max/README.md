# claude-max — Claude subscription as a drop-in Anthropic endpoint

Runs a small **local HTTP server** that speaks the Anthropic Messages API but
is backed by the **Claude Agent SDK + your Claude subscription**. hermes' normal
`anthropic` transport points at it, so streaming, tool calling, vision, usage
accounting, and every TUI feature work **exactly as if you were using
api.anthropic.com** — but billed to your subscription instead of an API key.

## claude-max vs claude-agent

- **`claude-agent`** (the other approach) hands the whole turn to a Claude Code
  subprocess — Claude Code owns the tool loop; hermes' own tools/aux are limited.
- **`claude-max`** (this) is an **API facade**: hermes keeps its own agent loop
  and tool surface; the proxy only translates HTTP ↔ SDK. Nothing in hermes
  changes — it just sees an Anthropic endpoint.

## Setup

1. `npm install -g @anthropic-ai/claude-code`, run `claude`, `/login` with your
   subscription (or `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token`).
2. `pip install 'hermes-agent[claude-max]'` (claude-agent-sdk + aiohttp).
3. Use it — either:
   - `/model opus --provider claude-max` (shows as **"Claude Max (subscription
     proxy)"** in the picker), or
   - in `config.yaml`:
     ```yaml
     model:
       default: claude-opus-4-8
       provider: claude-max
     ```

Selecting the provider **auto-starts** the proxy (detached, health-checked,
PID at `~/.hermes/claude-max-proxy.pid`, logs at
`~/.hermes/logs/claude-max-proxy.log`). Manage it explicitly with
`hermes claude-max serve|start|stop|status`.

## How it works

The proxy serves `POST /v1/messages` (stream + non-stream) and a static
`GET /v1/models`. Each conversation is pinned to one `ClaudeSDKClient`. Tools
from the request are registered as in-process MCP tools whose handlers **block**
until hermes returns the matching `tool_result` on the next request — so the
stateless Anthropic tool loop drives the stateful SDK. Streaming reuses the
SDK's raw Anthropic stream events (sanitized to spec).

## Config / env

- `HERMES_CLAUDE_MAX_PORT` — server port (default `8646`).
- `HERMES_CLAUDE_MAX_SESSION_TTL` — idle session TTL seconds (default `1800`).
- `HERMES_CLAUDE_MAX_MAX_SESSIONS` — max concurrent pinned sessions (default `4`).

## Caveats

- **No `temperature` / `top_p`** — the SDK doesn't expose sampling params
  (accepted and ignored).
- **Cold-start fidelity**: if the proxy restarts, or hermes compacts/edits the
  history, a conversation can't be matched to a live session; it falls back to
  flattening prior turns into a synthetic transcript in a fresh session
  (thinking-block continuity is lost, cache metrics reset). Normal
  turn-by-turn use stays pinned and lossless.
- **Localhost only, unauthenticated** — the server binds `127.0.0.1` and
  accepts any API key; the loopback bind is the trust boundary. It refuses
  non-loopback binds.
- **Subscription limits apply** — usage draws from your Claude plan; hitting the
  cap surfaces as an Anthropic `rate_limit_error` (429).
- One in-flight request per conversation (the SDK client is single-stream);
  concurrent conversations use separate sessions.
