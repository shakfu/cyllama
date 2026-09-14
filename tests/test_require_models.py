"""CYLLAMA_REQUIRE_MODELS=1 turns a missing test model from a skip into a failure."""

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

CONFTEST = (Path(__file__).parent / "conftest.py").read_text()

MODEL_TEST = """
def test_needs_model(model_path):
    pass
"""


@pytest.fixture
def suite(pytester, monkeypatch, tmp_path):
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(MODEL_TEST)
    monkeypatch.setenv("CYLLAMA_MODELS_DIR", str(tmp_path / "no-models"))
    monkeypatch.delenv("CYLLAMA_REQUIRE_MODELS", raising=False)
    return pytester


def test_missing_model_skips_by_default(suite):
    result = suite.runpytest_subprocess("-p", "no:cacheprovider")
    result.assert_outcomes(skipped=1)


def test_missing_model_fails_when_required(suite, monkeypatch):
    monkeypatch.setenv("CYLLAMA_REQUIRE_MODELS", "1")
    result = suite.runpytest_subprocess("-p", "no:cacheprovider")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*CYLLAMA_REQUIRE_MODELS=1 but test model is missing*"])


def test_present_model_runs_when_required(suite, monkeypatch, tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "Llama-3.2-1B-Instruct-Q8_0.gguf").touch()
    monkeypatch.setenv("CYLLAMA_MODELS_DIR", str(models))
    monkeypatch.setenv("CYLLAMA_REQUIRE_MODELS", "1")
    result = suite.runpytest_subprocess("-p", "no:cacheprovider")
    result.assert_outcomes(passed=1)
