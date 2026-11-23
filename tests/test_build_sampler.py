"""build_sampler: one sampler chain for LLM, grammar generation, BatchGenerator, ServerSlot and chat."""

import gc

import pytest

import cyllama.llama.llama_cpp as cy
from cyllama.api import GenerationConfig
from cyllama.llama.generation import build_sampler


@pytest.fixture(scope="module")
def vocab(model_path):
    model = cy.LlamaModel(model_path, verbose=False)
    yield model.get_vocab()
    model.close()
    gc.collect()


def _names(sampler):
    return [sampler.chain_get(i).name() for i in range(len(sampler))]


def test_default_chain(vocab):
    names = _names(build_sampler(vocab, GenerationConfig(), 512))
    assert names[-5:] == ["min-p", "top-k", "top-p", "temp", "dist"]


def test_greedy(vocab):
    assert _names(build_sampler(vocab, GenerationConfig(temperature=0.0), 512))[-1] == "greedy"


def test_penalties_and_dry_come_first(vocab):
    config = GenerationConfig(repeat_penalty=1.1, dry_multiplier=0.8)
    names = _names(build_sampler(vocab, config, 512))
    assert names.index("penalties") < names.index("dry") < names.index("min-p")


def test_grammar_link_first(vocab):
    names = _names(build_sampler(vocab, GenerationConfig(), 512, grammar=('root ::= "yes"', "root")))
    assert names[0] == "grammar"


def test_mirostat(vocab):
    assert _names(build_sampler(vocab, GenerationConfig(mirostat=2), 512))[-1] == "mirostat-v2"


def test_batch_generator_honours_penalties(model_path, monkeypatch):
    # BatchGenerator built its own chain and dropped penalties, DRY and mirostat
    import cyllama.batching as batching

    seen = []
    real = batching.build_sampler
    monkeypatch.setattr(batching, "build_sampler", lambda *a, **k: seen.append(_names(real(*a, **k))) or real(*a, **k))
    with batching.BatchGenerator(model_path, n_ctx=256, batch_size=64, n_seq_max=2) as gen:
        gen.generate_batch(["Hi"], GenerationConfig(max_tokens=2, repeat_penalty=1.2))
    assert "penalties" in seen[0]
    gc.collect()
