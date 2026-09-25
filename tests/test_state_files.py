"""Session state files: tokens and KV cache round-trip through disk.

Real prompts contain token ids far above 255, which the sequence-file saver
used to reject: it built a uint8 vector and passed it to C as llama_token*.
"""

import pytest

import cyllama.llama.llama_cpp as cy

PROMPT = "The capital of France is"


@pytest.fixture(scope="module")
def model(model_path):
    cy.llama_backend_init()
    m = cy.LlamaModel(model_path, verbose=False)
    yield m
    m.close()


@pytest.fixture(scope="module")
def tokens(model):
    toks = model.get_vocab().tokenize(PROMPT, add_special=True, parse_special=False)
    assert max(toks) > 255
    return toks


def _ctx(model):
    p = cy.LlamaContextParams()
    p.n_ctx = 256
    return cy.LlamaContext(model, p)


def _greedy():
    s = cy.LlamaSampler()
    s.add_greedy()
    return s


def _next_token_after_restore(model, tokens, restore):
    """Greedy token for position len(tokens), decoding only tokens[-1] on a
    restored cache holding tokens[:-1]."""
    ctx = _ctx(model)
    restore(ctx)
    batch = cy.LlamaBatch(n_tokens=1, embd=0, n_seq_max=1)
    batch.set_batch([tokens[-1]], len(tokens) - 1, False)
    ctx.decode(batch)
    return _greedy().sample(ctx, -1)


@pytest.fixture(scope="module")
def expected_next(model, tokens):
    ctx = _ctx(model)
    ctx.decode(cy.llama_batch_get_one(tokens))
    return _greedy().sample(ctx, -1)


@pytest.fixture(scope="module")
def prefix_ctx(model, tokens):
    """Context whose cache holds tokens[:-1]."""
    ctx = _ctx(model)
    ctx.decode(cy.llama_batch_get_one(tokens[:-1]))
    return ctx


def test_seq_file_round_trip(model, tokens, prefix_ctx, expected_next, tmp_path):
    path = str(tmp_path / "seq.bin")
    prefix_ctx.save_state_seq_file(path, 0, tokens[:-1])

    loaded = {}

    def restore(ctx):
        loaded["tokens"] = ctx.load_state_seq_file(path, 0, max_n_tokens=64)

    assert _next_token_after_restore(model, tokens, restore) == expected_next
    assert loaded["tokens"] == tokens[:-1]


def test_state_file_round_trip(model, tokens, prefix_ctx, expected_next, tmp_path):
    path = str(tmp_path / "state.bin")
    assert prefix_ctx.save_state_file(path, tokens[:-1])

    loaded = {}

    def restore(ctx):
        loaded["tokens"] = ctx.load_state_file(path, max_n_tokens=64)

    assert _next_token_after_restore(model, tokens, restore) == expected_next
    assert loaded["tokens"] == tokens[:-1]
