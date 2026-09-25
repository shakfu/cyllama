"""LlamaSampler.sample(): equivalence with llama_sampler_sample, and the
inputs that made upstream abort the process now raise instead.

sample() mirrors upstream's function in Cython so it can check the result
before indexing. The equivalence tests compare it against the real upstream
function on identical clones, so a llama.cpp bump that changes upstream's
logic fails here rather than drifting silently.
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
    return model.get_vocab().tokenize(PROMPT, add_special=True, parse_special=False)


def _params():
    p = cy.LlamaContextParams()
    p.n_ctx = 256
    return p


def _decoded(model, tokens, samplers=None):
    ctx = cy.LlamaContext(model, _params(), samplers=samplers)
    ctx.decode(cy.llama_batch_get_one(tokens))
    return ctx


def _chain(*links):
    s = cy.LlamaSampler()
    for name, args in links:
        getattr(s, f"add_{name}")(*args)
    return s


CHAINS = {
    "greedy": [("greedy", ())],
    "top_k_temp_dist": [("top_k", (40,)), ("temp", (0.8,)), ("dist", (1234,))],
    "penalties_top_p_min_p_dist": [
        ("penalties", (128256, 64, 1.3, 0.1, 0.1)),
        ("top_p", (0.9, 1)),
        ("min_p", (0.05, 1)),
        ("dist", (7,)),
    ],
    "typical_dist": [("typical", (0.9, 1)), ("temp", (1.2,)), ("dist", (99,))],
    "mirostat_v2": [("temp", (0.9,)), ("mirostat_v2", (5, 5.0, 0.1))],
    "adaptive_p": [("top_k", (40,)), ("adaptive_p", (0.5, 0.9, 11))],
}


class TestMatchesUpstream:
    @pytest.mark.parametrize("links", CHAINS.values(), ids=CHAINS.keys())
    def test_cpu(self, model, tokens, links):
        ctx = _decoded(model, tokens)
        ours, ref = _chain(*links), _chain(*links)
        # repeated calls also compare the state each accept() leaves behind
        got = [ours.sample(ctx, -1) for _ in range(8)]
        want = [ref._sample_upstream(ctx, -1) for _ in range(8)]
        assert got == want

    @pytest.mark.parametrize(
        "links",
        [
            # fully offloaded: backend returns the token
            [("top_k", (40,)), ("temp", (0.8,)), ("dist", (5,))],
            # typical cannot be offloaded: backend returns candidates, CPU finishes
            [("top_k", (40,)), ("typical", (0.9, 1)), ("temp", (1.1,)), ("dist", (5,))],
        ],
        ids=["offloaded", "partial"],
    )
    def test_backend(self, model, tokens, links):
        ours, ref = _chain(*links), _chain(*links)
        ctx_ours = _decoded(model, tokens, samplers={0: ours})
        ctx_ref = _decoded(model, tokens, samplers={0: ref})
        assert ours.sample(ctx_ours, -1) == ref._sample_upstream(ctx_ref, -1)


class TestRaisesInsteadOfAborting:
    def test_chain_without_selector(self, model, tokens):
        ctx = _decoded(model, tokens)
        chain = _chain(("top_k", (5,)))
        with pytest.raises(ValueError, match="selected no token"):
            chain.sample(ctx, -1)
        # the process survived and the chain is still usable
        chain.add_greedy()
        assert chain.sample(ctx, -1) == _chain(("greedy", ()))._sample_upstream(ctx, -1)

    def test_empty_chain(self, model, tokens):
        ctx = _decoded(model, tokens)
        with pytest.raises(ValueError, match="selected no token"):
            cy.LlamaSampler().sample(ctx, -1)

    @pytest.mark.parametrize("idx", [0, 999, -999])
    def test_output_without_logits(self, model, tokens, idx):
        # llama_batch_get_one requests logits for the last token only
        ctx = _decoded(model, tokens)
        with pytest.raises(ValueError, match="no logits"):
            _chain(("greedy", ())).sample(ctx, idx)

    def test_before_any_decode(self, model):
        ctx = cy.LlamaContext(model, _params())
        with pytest.raises(ValueError, match="no logits"):
            _chain(("greedy", ())).sample(ctx, -1)

    def test_closed_sampler(self, model, tokens):
        ctx = _decoded(model, tokens)
        chain = _chain(("greedy", ()))
        chain.close()
        with pytest.raises(RuntimeError, match="closed"):
            chain.sample(ctx, -1)

    def test_closed_context(self, model, tokens):
        ctx = _decoded(model, tokens)
        ctx.close()
        with pytest.raises(RuntimeError, match="closed"):
            _chain(("greedy", ())).sample(ctx, -1)
