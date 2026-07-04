"""claude-max proxy design spike — validates SDK behaviors the proxy depends on.

Dev-only harness (requires: claude CLI logged in, claude-agent-sdk installed).
Run individual experiments:

    uv run python scripts/claude_max_spike.py 1        # raw stream fidelity
    uv run python scripts/claude_max_spike.py 2 3 ...  # or several

Findings feed the Task-0 gate: experiment 3 decides whether the blocking-MCP-
handler tool bridge is viable (primary design) or we fall back to
deny-interrupt re-injection.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
)

try:
    from claude_agent_sdk import StreamEvent
except ImportError:  # pragma: no cover
    StreamEvent = None

MODEL = "claude-haiku-4-5"


def _dump(tag: str, obj) -> None:
    try:
        text = json.dumps(obj, default=str)[:600]
    except Exception:
        text = repr(obj)[:600]
    print(f"  [{tag}] {text}")


async def exp1_stream_fidelity() -> None:
    print("=== EXP 1: raw stream fidelity (no tools) ===")
    opts = ClaudeAgentOptions(
        model=MODEL, tools=[], setting_sources=[],
        include_partial_messages=True, max_turns=1,
        env={"ANTHROPIC_API_KEY": ""},
    )
    client = ClaudeSDKClient(options=opts)
    await client.connect()
    try:
        await client.query("Say exactly: hello world")
        kinds: list[str] = []
        async for msg in client.receive_response():
            if StreamEvent is not None and isinstance(msg, StreamEvent):
                ev = msg.event
                kinds.append(ev.get("type", "?"))
                if ev.get("type") in {"message_start", "message_delta", "message_stop",
                                      "content_block_start", "content_block_stop"}:
                    _dump(ev.get("type"), ev)
            elif isinstance(msg, ResultMessage):
                _dump("ResultMessage", {"is_error": msg.is_error,
                                        "usage_keys": list((msg.usage or {}).keys())})
        print(f"  event sequence: {kinds}")
    finally:
        await client.disconnect()


def _make_weather_tool(block_event: asyncio.Event | None = None, results: dict | None = None):
    @tool("get_weather", "Get the current weather for a city", {"city": str})
    async def get_weather(args):
        print(f"  [handler] get_weather invoked args={args} t={time.time():.1f}")
        if block_event is not None:
            print("  [handler] blocking until event set...")
            await block_event.wait()
            print(f"  [handler] unblocked t={time.time():.1f}")
        payload = (results or {}).get("get_weather", "Sunny, 22C")
        return {"content": [{"type": "text", "text": payload}]}

    return get_weather


async def exp2_tool_prefixing() -> None:
    print("=== EXP 2: tool name prefixing ===")
    server = create_sdk_mcp_server("hermes", tools=[_make_weather_tool()])
    opts = ClaudeAgentOptions(
        model=MODEL, tools=[], mcp_servers={"hermes": server},
        allowed_tools=["mcp__hermes__get_weather"], setting_sources=[],
        include_partial_messages=True, env={"ANTHROPIC_API_KEY": ""},
    )
    client = ClaudeSDKClient(options=opts)
    await client.connect()
    try:
        await client.query("What's the weather in Lisbon? Use the get_weather tool.")
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for b in msg.content:
                    if isinstance(b, ToolUseBlock):
                        print(f"  ToolUseBlock name={b.name!r} id={b.id!r} input={b.input}")
            elif StreamEvent is not None and isinstance(msg, StreamEvent):
                ev = msg.event
                if ev.get("type") == "content_block_start" and \
                        (ev.get("content_block") or {}).get("type") == "tool_use":
                    _dump("stream tool_use start", ev)
            elif isinstance(msg, ResultMessage):
                _dump("Result", {"is_error": msg.is_error})
    finally:
        await client.disconnect()


async def exp3_blocking_handler(block_seconds: float = 90.0) -> None:
    print(f"=== EXP 3: blocking handler bridge (block {block_seconds}s) ===")
    release = asyncio.Event()
    server = create_sdk_mcp_server("hermes", tools=[_make_weather_tool(release)])
    opts = ClaudeAgentOptions(
        model=MODEL, tools=[], mcp_servers={"hermes": server},
        allowed_tools=["mcp__hermes__get_weather"], setting_sources=[],
        include_partial_messages=True,
        env={"ANTHROPIC_API_KEY": "", "MCP_TOOL_TIMEOUT": "600000", "MCP_TIMEOUT": "600000"},
    )
    client = ClaudeSDKClient(options=opts)
    await client.connect()
    saw_stop = asyncio.Event()
    timeline: list[tuple[float, str]] = []
    t0 = time.time()

    async def reader():
        async for msg in client.receive_response():
            ts = time.time() - t0
            if StreamEvent is not None and isinstance(msg, StreamEvent):
                et = msg.event.get("type", "?")
                timeline.append((ts, et))
                if et == "message_delta":
                    _dump(f"message_delta@{ts:.1f}s", msg.event)
                if et == "message_stop":
                    print(f"  message_stop at {ts:.1f}s (released={release.is_set()})")
                    saw_stop.set()
            elif isinstance(msg, AssistantMessage):
                blocks = [type(b).__name__ for b in msg.content]
                timeline.append((ts, f"Assistant({blocks})"))
                print(f"  AssistantMessage at {ts:.1f}s blocks={blocks}")
            elif isinstance(msg, ResultMessage):
                timeline.append((ts, "Result"))
                print(f"  ResultMessage at {ts:.1f}s is_error={msg.is_error}")

    reader_task = asyncio.create_task(reader())
    await client.query("What's the weather in Lisbon? Use the get_weather tool, then summarize in one line.")
    try:
        await asyncio.wait_for(saw_stop.wait(), timeout=60)
        print("  ✓ assistant tool_use message completed WHILE handler blocked")
    except asyncio.TimeoutError:
        print("  ✗ no message_stop before handler resolution")
    print(f"  holding blocked {block_seconds}s to test timeouts...")
    await asyncio.sleep(block_seconds)
    release.set()
    try:
        await asyncio.wait_for(reader_task, timeout=120)
        print("  ✓ continuation completed after release")
    except asyncio.TimeoutError:
        print("  ✗ reader did not finish after release")
    finally:
        await client.disconnect()
    print("  timeline tail:", [(f"{t:.1f}s", e) for t, e in timeline[-8:]])


async def exp4_parallel_tools() -> None:
    print("=== EXP 4: parallel tool calls ===")
    invocations: list[tuple[str, float]] = []
    t0 = time.time()

    @tool("get_weather", "Get weather for a city", {"city": str})
    async def w(args):
        invocations.append((f"weather:{args.get('city')}", time.time() - t0))
        await asyncio.sleep(3)
        return {"content": [{"type": "text", "text": "Sunny"}]}

    @tool("get_time", "Get local time for a city", {"city": str})
    async def tm(args):
        invocations.append((f"time:{args.get('city')}", time.time() - t0))
        await asyncio.sleep(3)
        return {"content": [{"type": "text", "text": "12:00"}]}

    server = create_sdk_mcp_server("hermes", tools=[w, tm])
    opts = ClaudeAgentOptions(
        model=MODEL, tools=[], mcp_servers={"hermes": server},
        allowed_tools=["mcp__hermes__get_weather", "mcp__hermes__get_time"],
        setting_sources=[], include_partial_messages=True,
        env={"ANTHROPIC_API_KEY": ""},
    )
    client = ClaudeSDKClient(options=opts)
    await client.connect()
    try:
        await client.query("Call BOTH get_weather and get_time for Lisbon in the SAME response, then summarize.")
        order: list[tuple[float, str]] = []
        async for msg in client.receive_response():
            if StreamEvent is not None and isinstance(msg, StreamEvent):
                ev = msg.event
                if ev.get("type") == "content_block_start" and \
                        (ev.get("content_block") or {}).get("type") == "tool_use":
                    order.append((time.time() - t0, ev["content_block"].get("name") or "?"))
            elif isinstance(msg, ResultMessage):
                break
        print(f"  stream tool_use starts: {[(f'{ts:.1f}s', n) for ts, n in order]}")
        print(f"  handler invocations:    {[(n, f'{ts:.1f}s') for n, ts in invocations]}")
        if len(invocations) >= 2:
            gap = abs(invocations[1][1] - invocations[0][1])
            print(f"  handler start gap: {gap:.2f}s -> {'CONCURRENT' if gap < 2.5 else 'SEQUENTIAL'}")
    finally:
        await client.disconnect()


async def exp6_interrupt_recovery() -> None:
    print("=== EXP 6: interrupt while handler blocked ===")
    release = asyncio.Event()
    server = create_sdk_mcp_server("hermes", tools=[_make_weather_tool(release)])
    opts = ClaudeAgentOptions(
        model=MODEL, tools=[], mcp_servers={"hermes": server},
        allowed_tools=["mcp__hermes__get_weather"], setting_sources=[],
        include_partial_messages=True, env={"ANTHROPIC_API_KEY": ""},
    )
    client = ClaudeSDKClient(options=opts)
    await client.connect()
    try:
        await client.query("Weather in Lisbon? Use get_weather.")

        async def drain():
            async for msg in client.receive_response():
                if isinstance(msg, ResultMessage):
                    print(f"  turn1 Result is_error={msg.is_error}")

        task = asyncio.create_task(drain())
        await asyncio.sleep(8)
        print("  sending interrupt...")
        await client.interrupt()
        release.set()
        try:
            await asyncio.wait_for(task, timeout=30)
        except asyncio.TimeoutError:
            print("  drain timed out post-interrupt")
        await client.query("Say exactly: recovered")
        text = ""
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for b in msg.content:
                    if isinstance(b, TextBlock):
                        text += b.text
            elif isinstance(msg, ResultMessage):
                break
        print(f"  post-interrupt text: {text!r} -> {'✓ usable' if 'recovered' in text.lower() else '✗ broken'}")
    finally:
        await client.disconnect()


async def exp8_isolation() -> None:
    print("=== EXP 8: system prompt + isolation ===")
    opts = ClaudeAgentOptions(
        model=MODEL, tools=[], setting_sources=[],
        system_prompt="You are TESTBOT. Always begin your reply with the word TESTBOT.",
        include_partial_messages=False, max_turns=1,
        env={"ANTHROPIC_API_KEY": ""},
    )
    client = ClaudeSDKClient(options=opts)
    await client.connect()
    try:
        await client.query("Who are you? One short line. Do you see any project CLAUDE.md instructions? yes/no.")
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for b in msg.content:
                    if isinstance(b, TextBlock):
                        print(f"  reply: {b.text[:200]!r}")
            elif isinstance(msg, ResultMessage):
                break
    finally:
        await client.disconnect()


EXPERIMENTS = {
    "1": exp1_stream_fidelity,
    "2": exp2_tool_prefixing,
    "3": exp3_blocking_handler,
    "4": exp4_parallel_tools,
    "6": exp6_interrupt_recovery,
    "8": exp8_isolation,
}


async def main() -> None:
    picks = sys.argv[1:] or list(EXPERIMENTS)
    for p in picks:
        fn = EXPERIMENTS.get(p)
        if fn is None:
            print(f"(no experiment {p!r})")
            continue
        try:
            await fn()
        except Exception as exc:
            print(f"  !! experiment {p} raised: {type(exc).__name__}: {exc}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
