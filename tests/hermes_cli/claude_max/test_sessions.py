"""Session registry: fingerprint stability + Flow A/B/C matching + eviction."""

from hermes_cli.claude_max import sessions as S


def _fake_bridge_factory():
    made = []

    def factory(_translated):
        b = type("B", (), {"closed": False, "close": lambda self: setattr(self, "closed", True)})()
        made.append(b)
        return b

    return factory, made


# ---------- fingerprint ----------

def test_fingerprint_ignores_cache_control_and_string_vs_blocks():
    a = S.fingerprint("sys", [{"role": "user", "content": "hi"}])
    b = S.fingerprint(
        [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
        [{"role": "user", "content": [{"type": "text", "text": "hi",
                                       "cache_control": {"type": "ephemeral"}}]}],
    )
    assert a == b


def test_fingerprint_distinguishes_different_history():
    a = S.fingerprint("sys", [{"role": "user", "content": "hi"}])
    b = S.fingerprint("sys", [{"role": "user", "content": "bye"}])
    assert a != b


# ---------- Flow C: cold start ----------

def test_cold_start_new_session_single_user():
    factory, made = _fake_bridge_factory()
    reg = S.SessionRegistry(factory)
    r = reg.match({"system": "s", "messages": [{"role": "user", "content": "hello"}]})
    assert r.kind == "cold" and r.created is True
    assert r.query_text == "hello"
    assert reg.size == 1


def test_cold_start_flattens_prior_history_on_miss():
    factory, _ = _fake_bridge_factory()
    reg = S.SessionRegistry(factory)
    msgs = [
        {"role": "user", "content": "what's the weather?"},
        {"role": "assistant", "content": [{"type": "text", "text": "Let me check."}]},
        {"role": "user", "content": "and tomorrow?"},
    ]
    r = reg.match({"system": "s", "messages": msgs})
    assert r.kind == "cold"
    assert "Prior conversation transcript" in r.query_text
    assert "and tomorrow?" in r.query_text


# ---------- Flow B: continuation ----------

def test_flow_b_matches_prior_fingerprint():
    factory, _ = _fake_bridge_factory()
    reg = S.SessionRegistry(factory)
    # Seed a session as if it answered turn 1.
    r1 = reg.match({"system": "s", "messages": [{"role": "user", "content": "hi"}]})
    reg.record_response(r1.session, "s",
                        [{"role": "user", "content": "hi"}],
                        [{"type": "text", "text": "hello!"}], set())
    # Turn 2: same history + new user message → continue on the same session.
    msgs2 = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "hello!"}]},
        {"role": "user", "content": "how are you?"},
    ]
    r2 = reg.match({"system": "s", "messages": msgs2})
    assert r2.kind == "continue"
    assert r2.session is r1.session
    assert r2.query_text == "how are you?"
    assert reg.size == 1  # reused, not created


# ---------- Flow A: tool continuation ----------

def test_flow_a_matches_pending_tool_ids():
    factory, _ = _fake_bridge_factory()
    reg = S.SessionRegistry(factory)
    r1 = reg.match({"system": "s", "messages": [{"role": "user", "content": "weather?"}]})
    # Session streamed an assistant tool_use with id toolu_7:
    assistant = [{"type": "tool_use", "id": "toolu_7", "name": "get_weather", "input": {"city": "Lisbon"}}]
    reg.record_response(r1.session, "s",
                        [{"role": "user", "content": "weather?"}],
                        assistant, {"toolu_7"})
    # hermes returns the tool_result.
    msgs2 = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": assistant},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_7",
                                      "content": "Sunny 22C"}]},
    ]
    r2 = reg.match({"system": "s", "messages": msgs2})
    assert r2.kind == "tool_result"
    assert r2.session is r1.session
    assert r2.deliveries == [("toolu_7", [{"type": "text", "text": "Sunny 22C"}], False)]


def test_flow_a_miss_falls_back_to_cold():
    factory, _ = _fake_bridge_factory()
    reg = S.SessionRegistry(factory)
    msgs = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_x", "name": "w", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_x", "content": "?"}]},
    ]
    r = reg.match({"system": "s", "messages": msgs})  # no pinned session
    assert r.kind == "cold"


# ---------- eviction ----------

def test_lru_eviction_at_capacity():
    factory, made = _fake_bridge_factory()
    clock = {"t": 0.0}
    reg = S.SessionRegistry(factory, max_sessions=2, clock=lambda: clock["t"])
    clock["t"] = 1
    reg.match({"messages": [{"role": "user", "content": "a"}]})
    clock["t"] = 2
    reg.match({"messages": [{"role": "user", "content": "b"}]})
    clock["t"] = 3
    reg.match({"messages": [{"role": "user", "content": "c"}]})  # evicts LRU (a)
    assert reg.size == 2
    assert made[0].closed is True  # first session closed


def test_ttl_eviction():
    factory, made = _fake_bridge_factory()
    clock = {"t": 0.0}
    reg = S.SessionRegistry(factory, ttl=10, clock=lambda: clock["t"])
    reg.match({"messages": [{"role": "user", "content": "a"}]})
    clock["t"] = 100  # past TTL
    reg.match({"messages": [{"role": "user", "content": "b"}]})
    assert reg.size == 1
    assert made[0].closed is True
