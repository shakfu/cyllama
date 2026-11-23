"""Backend library discovery for repaired wheels."""

import os

from cyllama._internal import backend_dl


def test_windows_path_excludes_cwd_from_dll_search(tmp_path, monkeypatch):
    # the legacy search order checks the cwd before PATH, where the CUDA runtime is found
    calls = []
    (tmp_path / "cyllama.libs").mkdir()
    monkeypatch.setattr(backend_dl, "_libs_loaded", False)
    monkeypatch.setattr(backend_dl.sys, "platform", "win32")
    monkeypatch.setattr(backend_dl, "_exclude_cwd_from_dll_search", lambda: calls.append(1))
    monkeypatch.setenv("PATH", "")
    backend_dl.libs_to_load(str(tmp_path))
    assert calls == [1]
    assert os.environ["PATH"].split(os.pathsep)[0] == str(tmp_path / "cyllama.libs")


def test_libs_to_load_runs_once(tmp_path, monkeypatch):
    monkeypatch.setattr(backend_dl, "_libs_loaded", False)
    backend_dl.libs_to_load(str(tmp_path))
    assert backend_dl.libs_to_load(str(tmp_path)) == []
