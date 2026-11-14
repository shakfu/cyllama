"""Tri-state GGML_* env flags in `scripts/manage.py`: unset, 0/1, and invalid values."""

import importlib.util
import logging
import sys
from pathlib import Path

import pytest

MANAGE_PY = Path(__file__).resolve().parent.parent / "scripts" / "manage.py"


def _load_manage():
    if "cyllama_manage" in sys.modules:
        return sys.modules["cyllama_manage"]
    spec = importlib.util.spec_from_file_location("cyllama_manage", MANAGE_PY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["cyllama_manage"] = module
    spec.loader.exec_module(module)
    return module


manage = _load_manage()


@pytest.fixture
def options(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for var in ("GGML_NATIVE", "GGML_OPENMP"):
        monkeypatch.delenv(var, raising=False)

    def run(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        return manage.LlamaCppBuilder().get_backend_cmake_options()

    return run


@pytest.mark.parametrize("var", ["GGML_NATIVE", "GGML_OPENMP"])
def test_unset_leaves_upstream_default(options, var):
    assert var not in options()


@pytest.mark.parametrize("var", ["GGML_NATIVE", "GGML_OPENMP"])
@pytest.mark.parametrize("value, expected", [("1", "ON"), ("0", "OFF")])
def test_zero_one(options, var, value, expected):
    assert options(**{var: value})[var] == expected


@pytest.mark.parametrize("var", ["GGML_NATIVE", "GGML_OPENMP"])
def test_invalid_value_warns(options, var, caplog):
    with caplog.at_level(logging.WARNING):
        assert options(**{var: "ON"})[var] == "OFF"
    assert f"Invalid boolean value for {var}" in caplog.text
