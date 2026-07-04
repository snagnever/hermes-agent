# claude-agent — Claude via your Claude subscription (Agent SDK)

Runs Claude models through the Claude Agent SDK, authenticated by your
Claude subscription (Claude Code OAuth) instead of a pay-per-token
`ANTHROPIC_API_KEY`. This is the Anthropic-sanctioned path for
subscription-backed use via the Agent SDK.

## Setup

1. `npm install -g @anthropic-ai/claude-code` and run `claude` once → `/login`
   with your Claude subscription (or export `CLAUDE_CODE_OAUTH_TOKEN` from
   `claude setup-token` for headless hosts).
2. `pip install claude-agent-sdk` (or `hermes-agent[claude-agent]`) into
   Hermes' environment.
3. `hermes doctor` — the "Claude Code CLI (claude_agent_sdk runtime)" check
   should pass.

## Using it

**As a switchable option (recommended)** — keep your normal default model and
switch into Claude-subscription when you want it. `claude-agent` shows up in
the `/model` picker as **"Claude (subscription)"** with opus / sonnet / haiku,
or switch directly:

```
/model opus --provider claude-agent
```

Selecting the `claude-agent` provider always routes through the Agent SDK —
no config flag needed. Switch back with `/model <your-model>` any time.

**As your default** — set it in `config.yaml`:

```yaml
model:
  default: claude-opus-4-8
  provider: claude-agent
```

Or reroute the plain `anthropic` provider through the SDK without changing
provider, via the opt-in flag:

```yaml
model:
  provider: anthropic
  anthropic_runtime: claude_agent_sdk   # auto (default) leaves the API-key path unchanged
```

## How it works

Claude Code owns the tool loop (terminal, file ops, patching run in its
own runtime). Hermes' own tool surface (web search, browser, vision,
kanban, skills…) is injected into the session via the `hermes-tools` stdio
MCP server (`agent/transports/hermes_tools_mcp_server.py`). Per-turn token
usage/cost is projected back into Hermes' accounting from the SDK's
`ResultMessage`. Structurally this mirrors the `codex_app_server` runtime.

## Auth hygiene

`ANTHROPIC_API_KEY` is deliberately blanked in the SDK subprocess env so a
stray key can never silently flip billing from your subscription to
pay-per-token API usage.

## Caveats

- No `temperature` / `top_p` control — the SDK does not expose sampling params.
- `delegate_task` / `memory` / `todo` Hermes agent-loop tools are unavailable
  on this runtime (same limitation as the `codex_app_server` runtime).
- Auxiliary tasks (title generation, compression, memory extraction) run
  through the Claude Agent SDK on a cheap model (Haiku) automatically when no
  HTTP aux provider is configured and the `claude` CLI is present — so a
  subscription-only setup gets them with no extra key. Each such call spawns
  a short-lived Claude Code subprocess (~5-7s), so they're best-effort helpers
  rather than hot-path calls. To use a faster API-key backend for aux instead
  (avoiding the subprocess spawn), configure one explicitly:

  ```yaml
  auxiliary:
    provider: anthropic        # or openrouter, etc. — needs its own key
    model: claude-haiku-4-5
  ```

  Streaming and tool-calling auxiliary tasks (e.g. the vision helper) are not
  routed through the SDK and still need an HTTP provider.
- There is no REST `/models` catalog on this path, so the model picker uses
  the profile's `fallback_models` (SDK aliases `opus`/`sonnet` resolve to your
  subscription's current models).
- Hermes tools cold-start: the injected `hermes-tools` MCP server takes
  ~15-20s to boot (it imports Hermes and registers its tool surface), and
  Claude Code loads MCP servers asynchronously. A first message sent within
  that window may not see `mcp__hermes-tools__*` tools yet — they become
  available once the server finishes connecting (typically by the time you
  send your first message in an interactive session). Claude Code's own
  tools (Bash, file ops) are available immediately.
- `claude-agent-sdk` is versioned alongside Claude Code; if the SDK's
  `ClaudeAgentOptions` fields drift, pin/upgrade both together.
