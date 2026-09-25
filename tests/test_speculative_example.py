"""Runs the draft/verify loop in tests/examples/speculative_example.py.

pytest does not collect tests/examples, so without this the example can
drift from the API unnoticed. The model drafts for itself, which makes the
expected results exact: greedy verification must reproduce plain greedy
decoding, and a correctly aligned draft is almost always accepted.
"""

import importlib.util
from pathlib import Path

import pytest

from cyllama.llama.llama_cpp import Speculative, SpeculativeParams

_spec = importlib.util.spec_from_file_location(
    "speculative_example", Path(__file__).parent / "examples" / "speculative_example.py"
)
example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example)

N_PREDICT = 48


@pytest.fixture(scope="module")
def setup(model_path):
    model, ctx_tgt = example.load(model_path, n_ctx=512, n_gpu_layers=0)
    _, ctx_dft = example.load(model_path, n_ctx=512, n_gpu_layers=0)
    prompt = model.get_vocab().tokenize(
        "Write a short story about a lighthouse keeper.", add_special=True, parse_special=False
    )
    baseline = example.greedy_generate(ctx_tgt, prompt, N_PREDICT)
    return ctx_tgt, ctx_dft, prompt, baseline


@pytest.mark.parametrize(
    "params",
    [SpeculativeParams(n_max=4), SpeculativeParams(n_max=8, p_min=0.75)],
    ids=["n_max=4", "n_max=8,p_min=0.75"],
)
def test_matches_greedy_decoding(setup, params):
    ctx_tgt, ctx_dft, prompt, baseline = setup
    spec = Speculative(params, ctx_tgt, ctx_dft)
    res = example.speculative_generate(ctx_tgt, spec, params, prompt, N_PREDICT)

    assert res.tokens == baseline.tokens
    assert len(res.tokens) == N_PREDICT
    # a draft misaligned by one position is almost never accepted
    assert res.acceptance > 0.9
    # each round yields its accepted draft tokens plus one target token
    assert res.n_accepted + res.n_rounds >= len(res.tokens)
