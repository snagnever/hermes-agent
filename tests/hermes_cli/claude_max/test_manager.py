"""Auto-start manager: already-running short-circuit, spawn+poll, failure."""

import pytest

from hermes_cli.claude_max import manager as M


def test_check_claude_binary_missing():
    ok, msg = M.check_claude_binary("definitely-not-real-xyz")
    assert ok is False and "not found" in msg


def test_ensure_running_short_circuits_when_healthy(monkeypatch):
    calls = {"spawn": 0}
    monkeypatch.setattr(M, "_health_ok", lambda *a, **k: True)
    monkeypatch.setattr(M.subprocess, "Popen",
                        lambda *a, **k: calls.__setitem__("spawn", calls["spawn"] + 1))
    port = M.ensure_server_running(port=9999)
    assert port == 9999
    assert calls["spawn"] == 0  # never spawned


def test_ensure_running_spawns_and_polls(monkeypatch, tmp_path):
    health = {"ok": [False, False, True]}  # unhealthy until 3rd poll

    def fake_health(*a, **k):
        return health["ok"].pop(0) if health["ok"] else True

    monkeypatch.setattr(M, "_health_ok", fake_health)
    monkeypatch.setattr(M, "_preflight", lambda: None)
    monkeypatch.setattr(M, "_log_path", lambda: tmp_path / "log")
    monkeypatch.setattr(M, "_write_pid_record", lambda *a, **k: None)

    class FakeProc:
        pid = 4321
        def poll(self):
            return None  # alive

    monkeypatch.setattr(M.subprocess, "Popen", lambda *a, **k: FakeProc())
    monkeypatch.setattr(M.time, "sleep", lambda s: None)
    port = M.ensure_server_running(port=9998, timeout=5)
    assert port == 9998


def test_ensure_running_surfaces_immediate_exit(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "_health_ok", lambda *a, **k: False)
    monkeypatch.setattr(M, "_preflight", lambda: None)
    logp = tmp_path / "log"
    logp.write_text("Traceback: boom\n")
    monkeypatch.setattr(M, "_log_path", lambda: logp)
    monkeypatch.setattr(M, "_write_pid_record", lambda *a, **k: None)
    monkeypatch.setattr(M, "_clear_pid_record", lambda: None)

    class DeadProc:
        pid = 1
        returncode = 1
        def poll(self):
            return 1  # exited

    monkeypatch.setattr(M.subprocess, "Popen", lambda *a, **k: DeadProc())
    monkeypatch.setattr(M.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="exited immediately"):
        M.ensure_server_running(port=9997, timeout=5)
