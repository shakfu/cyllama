"""Tests for the stable-diffusion CLI Ctrl-C isolation harness.

These exercise the supervision logic (`_supervise`) and the isolation gate
(`_should_isolate`) without spawning a real generation -- SD's generate() has
no in-process abort hook, so the CLI runs it in a child process the parent can
kill (cyllama issue #8). The harness is plain process plumbing, so it is tested
with fake process objects and env toggles rather than a multi-GB model.
"""

import pytest

# The harness lives in the SD CLI module; importing it is cheap (SD-specific
# imports inside the module are lazy).
mod = pytest.importorskip("cyllama.sd.__main__")


class _FakeProc:
    """Minimal subprocess.Popen stand-in for _supervise."""

    def __init__(self, wait):
        self._wait = wait  # callable(timeout) -> int, may raise
        self._alive = True
        self.events = []

    def wait(self, timeout=None):
        return self._wait(timeout)

    def terminate(self):
        self.events.append("terminate")
        self._alive = False

    def kill(self):
        self.events.append("kill")
        self._alive = False

    def poll(self):
        return None if self._alive else 0


class TestSupervise:
    def test_returns_child_exit_code(self):
        proc = _FakeProc(lambda timeout=None: 7)
        proc._alive = False  # already exited
        assert mod._supervise(proc) == 7
        assert proc.events == []  # no terminate/kill on clean exit

    def test_keyboardinterrupt_terminates_child_and_returns_130(self):
        calls = {"n": 0}

        def wait(timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise KeyboardInterrupt  # the Ctrl-C
            return 0  # child exits after terminate()

        proc = _FakeProc(wait)
        rc = mod._supervise(proc)
        assert rc == 130
        assert "terminate" in proc.events
        assert "kill" not in proc.events  # terminated gracefully within grace

    def test_keyboardinterrupt_force_kills_unresponsive_child(self):
        import subprocess

        calls = {"n": 0}

        def wait(timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise KeyboardInterrupt
            if timeout is not None:
                raise subprocess.TimeoutExpired(cmd="child", timeout=timeout)
            return 0

        proc = _FakeProc(wait)
        rc = mod._supervise(proc)
        assert rc == 130
        assert proc.events == ["terminate", "kill"]


class TestShouldIsolate:
    def test_isolated_command_with_no_markers(self, monkeypatch):
        monkeypatch.delenv("_CYLLAMA_SD_CHILD", raising=False)
        monkeypatch.delenv("CYLLAMA_SD_NO_ISOLATE", raising=False)
        assert mod._should_isolate("txt2img") is True

    def test_child_marker_disables_isolation(self, monkeypatch):
        monkeypatch.setenv("_CYLLAMA_SD_CHILD", "1")
        monkeypatch.delenv("CYLLAMA_SD_NO_ISOLATE", raising=False)
        assert mod._should_isolate("txt2img") is False

    def test_opt_out_disables_isolation(self, monkeypatch):
        monkeypatch.delenv("_CYLLAMA_SD_CHILD", raising=False)
        monkeypatch.setenv("CYLLAMA_SD_NO_ISOLATE", "1")
        assert mod._should_isolate("txt2img") is False

    def test_non_isolated_command_runs_in_process(self, monkeypatch):
        monkeypatch.delenv("_CYLLAMA_SD_CHILD", raising=False)
        monkeypatch.delenv("CYLLAMA_SD_NO_ISOLATE", raising=False)
        assert mod._should_isolate("info") is False
