"""Thread-count defaults: physical cores, reaching llama.cpp from the high-level API."""

import os
from unittest.mock import patch

import pytest

from cyllama.api import LLM, GenerationConfig
from cyllama.utils.platform import physical_cores, resolve_n_threads


def test_physical_cores():
    assert 1 <= physical_cores() <= (os.cpu_count() or 1)


@pytest.mark.parametrize(
    "n_threads, share, expected",
    [(3, 1, 3), (3, 4, 3), (-1, 1, 16), (-1, 2, 8), (-1, 32, 1)],
)
def test_resolve_n_threads(n_threads, share, expected):
    with patch("cyllama.utils.platform.physical_cores", return_value=16):
        assert resolve_n_threads(n_threads, share) == expected


@pytest.mark.parametrize("field", ["n_threads", "n_threads_batch"])
@pytest.mark.parametrize("value", [0, -2])
def test_config_rejects_invalid_threads(field, value):
    with pytest.raises(ValueError, match=field):
        GenerationConfig(**{field: value})


def test_config_threads_in_dict():
    d = GenerationConfig(n_threads=2, n_threads_batch=3).to_dict()
    assert (d["n_threads"], d["n_threads_batch"]) == (2, 3)


def test_llm_applies_threads(model_path):
    llm = LLM(model_path, verbose=False)
    try:
        llm("Hi", config=GenerationConfig(max_tokens=1))
        assert (llm._ctx.n_threads(), llm._ctx.n_threads_batch()) == (physical_cores(), physical_cores())

        # The context is reused for a same-size request; the new counts still apply.
        ctx = llm._ctx
        llm("Hi", config=GenerationConfig(max_tokens=1, n_threads=2, n_threads_batch=3))
        assert llm._ctx is ctx
        assert (ctx.n_threads(), ctx.n_threads_batch()) == (2, 3)
    finally:
        llm.close()
