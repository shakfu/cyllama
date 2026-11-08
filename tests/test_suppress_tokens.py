"""LlamaSampler.add_suppress_tokens(): the model's tokenizer.ggml.suppress_tokens are never sampled."""

import gc

import pytest

import cyllama.llama.llama_cpp as cy
from conftest import MODELS_DIR

# the only local model that declares suppress tokens: its 50,276 text tokens
CHATTERBOX = MODELS_DIR / "chatterbox-turbo-Q8_0.gguf"
needs_chatterbox = pytest.mark.skipif(not CHATTERBOX.exists(), reason="needs chatterbox-turbo-Q8_0.gguf")


def test_no_suppress_tokens_adds_no_link(model_path):
    model = cy.LlamaModel(model_path, verbose=False)
    vocab = model.get_vocab()
    assert vocab.get_suppress_tokens() == []
    sampler = cy.LlamaSampler()
    sampler.add_suppress_tokens(vocab)
    assert len(sampler) == 0
    sampler.close()
    model.close()


@needs_chatterbox
def test_suppressed_tokens_are_never_sampled():
    model = cy.LlamaModel(str(CHATTERBOX), verbose=False)
    vocab = model.get_vocab()
    suppress = set(vocab.get_suppress_tokens())
    assert len(suppress) == 50276

    params = cy.LlamaContextParams()
    params.n_ctx = 256
    ctx = cy.LlamaContext(model, params)
    ctx.decode(cy.llama_batch_get_one(vocab.tokenize("Hello there", add_special=False, parse_special=False)))

    def draws(suppressed: bool) -> list:
        out = []
        for seed in range(20):
            sampler = cy.LlamaSampler()
            if suppressed:
                sampler.add_suppress_tokens(vocab)
            # temp 100 flattens the distribution; 88% of the vocab is suppressed
            sampler.add_temp(100.0)
            sampler.add_dist(seed)
            out.append(sampler.sample(ctx, -1))
            sampler.close()
        return out

    # control: without the bias, suppressed tokens are drawn (16 of 20 at b12110)
    assert any(t in suppress for t in draws(False))
    assert not any(t in suppress for t in draws(True))

    sampler = cy.LlamaSampler()
    sampler.add_suppress_tokens(vocab)
    assert sampler.chain_get(0).name() == "logit-bias"
    sampler.close()
    ctx.close()
    model.close()
    gc.collect()


@needs_chatterbox
def test_llm_sampler_applies_suppress_tokens():
    from cyllama import LLM, GenerationConfig

    with LLM(str(CHATTERBOX), verbose=False) as llm:
        llm("Hello", config=GenerationConfig(max_tokens=1))
        assert llm._sampler.chain_get(0).name() == "logit-bias"
    del llm
    gc.collect()
