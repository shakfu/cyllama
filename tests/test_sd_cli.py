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


class TestMemoryPlacementFlags:
    """`--offload-to-cpu` & friends must map to weight *placement*, not to a
    VRAM budget.

    Upstream (stable-diffusion.cpp master-731) replaced the per-component
    `keep_*_on_cpu` fields with two separate mechanisms: `params_backend`
    placements and the `max_vram` graph-cut budget. Routing the legacy flags
    to `max_vram=-1` changes what they mean -- the weights stay on the GPU and
    every module is handed a budget computed as if it were the only resident
    one, which OOMs on small cards instead of offloading.
    """

    @staticmethod
    def _params(argv):
        import argparse

        parser = argparse.ArgumentParser()
        mod.add_common_memory_args(parser)
        return mod.create_context_params(parser.parse_args(argv))

    def test_no_flags_leaves_placement_unset(self):
        params = self._params([])
        assert params.max_vram is None
        assert params.params_backend is None
        assert params.backend is None
        assert params.auto_fit is False

    def test_offload_to_cpu_places_all_weights_on_cpu(self):
        params = self._params(["--offload-to-cpu"])
        assert params.params_backend == "cpu"
        assert params.max_vram is None

    def test_per_component_flags_map_to_module_keys(self):
        params = self._params(["--clip-on-cpu", "--vae-on-cpu"])
        assert params.params_backend == "te=cpu,vae=cpu"
        params = self._params(["--control-net-cpu"])
        assert params.params_backend == "control-net=cpu"

    def test_offload_to_cpu_subsumes_per_component_flags(self):
        params = self._params(["--offload-to-cpu", "--vae-on-cpu"])
        assert params.params_backend == "cpu"

    def test_explicit_params_backend_wins_over_legacy_flags(self):
        params = self._params(["--params-backend", "vae=cpu", "--clip-on-cpu"])
        assert params.params_backend == "vae=cpu"

    def test_max_vram_is_independent_of_placement(self):
        params = self._params(["--max-vram", "4", "--clip-on-cpu"])
        assert params.max_vram == "4"
        assert params.params_backend == "te=cpu"

    def test_backend_and_auto_fit(self):
        params = self._params(["--backend", "diffusion=cuda0,te=cpu", "--auto-fit"])
        assert params.backend == "diffusion=cuda0,te=cpu"
        assert params.auto_fit is True
