# claude-max Task-0 spike findings (claude-agent-sdk 0.2.110)

Validated against the real SDK + logged-in `claude` CLI via `scripts/claude_max_spike.py`.

## Gate: PASSED — primary blocking-handler tool bridge is viable

### EXP 1 — raw stream fidelity
- `StreamEvent.event` carries spec-order Anthropic SSE: `message_start` → (`content_block_start`/`content_block_delta*`/`content_block_stop`) per block → `message_delta` (with `stop_reason` + `usage`) → `message_stop`.
- `message_start.message` has `id` (`msg_…`), `model` (**full id** e.g. `claude-haiku-4-5-20251001`), `usage`.
- **Non-standard extra fields present** and must be stripped by the translator: `diagnostics`, `stop_details`, `caller`, `inference_geo`, `context_management`, `output_tokens_details`, `iterations`, `cache_creation`, `service_tier`, `diagnostics`. → translator sanitizes each event to the clean Anthropic schema; rewrites `model` to echo the request model and mints a fresh `msg_<uuid>` id.
- The model emits a `thinking` content block even for trivial prompts (empty signature). Passthrough is fine; may strip empty thinking blocks for cleanliness.

### EXP 2 — tool name prefixing (CONFIRMED)
- Model-visible tool name is `mcp__hermes__<toolname>` (server registered as `hermes`). Translator **strips** the `mcp__hermes__` prefix when emitting `tool_use` to the client, **re-adds** when correlating `tool_result`s. Tool ids are standard `toolu_*`. Full input arrives on the `ToolUseBlock` (`{'city':'Lisbon'}`); the stream `content_block_start` shows `input: {}` then `input_json_delta`.

### EXP 3 — blocking handler bridge (THE GATE — PASSED)
- Assistant message (ThinkingBlock + ToolUseBlock) completes with `message_delta{stop_reason:"tool_use"}` + `message_stop` **while the handler is still blocked** (released=False). → response N is finalized with `stop_reason=tool_use` before the tool result exists.
- Handler blocked **90s with no timeout kill** when `env={"MCP_TOOL_TIMEOUT":"600000","MCP_TIMEOUT":"600000"}`. → long hermes tool executions are safe.
- After resolving the handler's future, a **fresh** `message_start…message_stop` cycle streams the continuation (response N+1). Continuation usage showed `cache_creation_input_tokens` populated (context cached across the block).

### EXP 4 — parallel tool calls (SEQUENTIAL execution)
- Two tool_use blocks stream together in ONE assistant message (both `content_block_start` at ~2.5s).
- Handlers are invoked **SEQUENTIALLY** (3.0s gap == first handler's sleep). → **Design consequence**: the ToolBridge must STORE delivered tool_results (keyed by name+args, re-keyed by tool_use id) and let each handler consume its result when it is eventually invoked; do NOT assume all handlers are concurrently blocked when the tool_results arrive.

### EXP 6 — interrupt recovery (PASSED)
- `client.interrupt()` while a handler blocks → turn ends `is_error=True`; the same client is immediately reusable ("recovered"). → eviction/cleanup can interrupt safely.

### EXP 7 — auth error shape (reused prior finding)
- Not-logged-in surfaces as `ResultMessage(is_error=True, subtype="success")` with the message in `result` ("Not logged in · Please run /login"). → map to Anthropic `authentication_error` (401).

### EXP 8 — isolation (CONFIRMED)
- `system_prompt` applies; `setting_sources=[]` → no project CLAUDE.md / user settings leak.

## Live E2E results (against the real subscription)

Server auto-started via `hermes claude-max start`, then:
- `GET /v1/models` → catalog, `max_input_tokens: 200000`. ✓
- non-stream `POST /v1/messages` → real "PONG", `stop_reason=end_turn`, usage populated. ✓
- **anthropic python SDK** (hermes' actual transport) `messages.stream()` → "1 2 3 4 5". ✓
- **anthropic SDK tool round-trip** → r1 `stop_reason=tool_use` (get_weather{city:Lisbon}); r2 with tool_result → continuation "sunny 24°C". ✓
- `hermes -z ... --provider claude-max` → auto-started, resolved runtime, reached Claude (turn blocked only by subscription quota exhaustion — surfaced correctly). ✓

**Bug found + fixed via E2E**: the stream's `content_block_start` for tool_use carries an EMPTY `input` (real args stream via `input_json_delta`), so correlating handler→tool_use_id by (name, args) failed and the handler blocked forever. Fixed to correlate by **tool NAME FIFO** (handlers run sequentially in stream order).

## Design implications locked
1. **Bridge = blocking MCP handlers** (primary; no fallback needed). `tools=[]` + `create_sdk_mcp_server("hermes", …)` + `allowed_tools=["mcp__hermes__*"]` + `include_partial_messages=True` + `setting_sources=[]` + `env` MCP-timeout knobs + big `max_turns`.
2. **One HTTP response = one assistant message cycle** (`message_start`→`message_stop`), terminated at `stop_reason=tool_use` (tools pending) or `ResultMessage` (final).
3. **Translator sanitizes** raw events to clean Anthropic schema; rewrites model + id; strips `mcp__hermes__` from tool_use names.
4. **ToolBridge stores results** for sequential handler consumption; correlate by (name, args) FIFO, re-key by tool_use id from stream events.
5. **Session bridge**: pin conversation → live ClaudeSDKClient; deliver tool_results to resolve blocked handlers (Flow A); `query()` for next user turn (Flow B); cold-start flatten on cache miss (Flow C).
