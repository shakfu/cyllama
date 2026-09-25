"""Backend sampling: sampler chains attached to a context run inside the
decode graph [EXPERIMENTAL upstream].

llama.cpp initialises an attached chain for that context's graph and never
resets it, so most tests here pin the binding rules that keep a misused
chain from aborting the process or sampling wrong.
"""

import gc
import math

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


def _params(n_seq_max=1):
    p = cy.LlamaContextParams()
    p.n_ctx = 256
    p.n_seq_max = n_seq_max
    return p


def _greedy():
    s = cy.LlamaSampler()
    s.add_greedy()
    return s


def _top_k_dist(k=10):
    s = cy.LlamaSampler()
    s.add_top_k(k)
    s.add_temp(0.8)
    s.add_dist(42)
    return s


def _decode(ctx, tokens):
    ctx.kv_cache_clear()
    ctx.decode(cy.llama_batch_get_one(tokens))


@pytest.fixture(scope="module")
def cpu_greedy_token(model, tokens):
    ctx = cy.LlamaContext(model, _params())
    _decode(ctx, tokens)
    tok = _greedy().sample(ctx, -1)
    ctx.close()
    return tok


class TestSampling:
    def test_greedy_via_set_sampler_matches_cpu(self, model, tokens, cpu_greedy_token):
        ctx = cy.LlamaContext(model, _params())
        chain = _greedy()
        assert ctx.set_sampler(0, chain) is True
        _decode(ctx, tokens)

        assert ctx.sampled_token_ith(-1) == cpu_greedy_token
        assert chain.sample(ctx, -1) == cpu_greedy_token

    def test_greedy_via_constructor_matches_cpu(self, model, tokens, cpu_greedy_token):
        chain = _greedy()
        ctx = cy.LlamaContext(model, _params(), samplers={0: chain})
        assert ctx.backend_samplers == {0: chain}
        _decode(ctx, tokens)

        assert ctx.sampled_token_ith(-1) == cpu_greedy_token

    def test_distribution_outputs_align(self, model, tokens):
        ctx = cy.LlamaContext(model, _params())
        chain = _top_k_dist(k=10)
        ctx.set_sampler(0, chain)
        _decode(ctx, tokens)

        probs = ctx.sampled_probs_ith(-1)
        cands = ctx.sampled_candidates_ith(-1)
        tok = ctx.sampled_token_ith(-1)
        assert len(probs) == len(cands) == 10
        assert math.isclose(sum(probs), 1.0, rel_tol=1e-4)
        assert tok in cands
        assert chain.sample(ctx, -1) == tok

    def test_no_backend_sampler_returns_none(self, model, tokens):
        ctx = cy.LlamaContext(model, _params())
        _decode(ctx, tokens)
        assert ctx.sampled_token_ith(-1) is None
        assert ctx.sampled_probs_ith(-1) is None
        assert ctx.sampled_logits_ith(-1) is None
        assert ctx.sampled_candidates_ith(-1) is None

    def test_context_retains_chain(self, model, tokens, cpu_greedy_token):
        ctx = cy.LlamaContext(model, _params())
        ctx.set_sampler(0, _greedy())  # no other reference
        gc.collect()
        _decode(ctx, tokens)
        assert ctx.backend_samplers[0].sample(ctx, -1) == cpu_greedy_token


class TestBinding:
    def test_reattach_same_seq_is_noop(self, model):
        ctx = cy.LlamaContext(model, _params())
        chain = _greedy()
        assert ctx.set_sampler(0, chain)
        assert ctx.set_sampler(0, chain)

    def test_attach_to_second_context_rejected(self, model):
        chain = _greedy()
        cy.LlamaContext(model, _params()).set_sampler(0, chain)
        with pytest.raises(ValueError, match="clone"):
            cy.LlamaContext(model, _params()).set_sampler(0, chain)

    def test_reattach_after_detach_rejected(self, model):
        ctx = cy.LlamaContext(model, _params(n_seq_max=2))
        chain = _greedy()
        ctx.set_sampler(0, chain)
        ctx.set_sampler(0, None)
        with pytest.raises(ValueError, match="clone"):
            ctx.set_sampler(1, chain)

    def test_sample_on_other_context_rejected(self, model, tokens):
        chain = _greedy()
        bound = cy.LlamaContext(model, _params())
        bound.set_sampler(0, chain)
        other = cy.LlamaContext(model, _params())
        _decode(other, tokens)
        with pytest.raises(ValueError, match="another context"):
            chain.sample(other, -1)

    def test_sample_after_bound_context_freed_rejected(self, model, tokens):
        chain = _greedy()
        bound = cy.LlamaContext(model, _params())
        bound.set_sampler(0, chain)
        del bound
        gc.collect()
        other = cy.LlamaContext(model, _params())
        _decode(other, tokens)
        with pytest.raises(ValueError, match="another context"):
            chain.sample(other, -1)

    def test_bound_chain_cannot_be_modified(self, model):
        chain = _greedy()
        cy.LlamaContext(model, _params()).set_sampler(0, chain)
        with pytest.raises(RuntimeError, match="cannot be modified"):
            chain.add_top_k(5)
        with pytest.raises(RuntimeError, match="cannot be modified"):
            chain.chain_remove(0)
        assert len(chain) == 1

    def test_clone_is_attachable(self, model, tokens, cpu_greedy_token):
        chain = _greedy()
        cy.LlamaContext(model, _params()).set_sampler(0, chain)
        other = cy.LlamaContext(model, _params())
        fresh = chain.clone()
        assert other.set_sampler(0, fresh)
        _decode(other, tokens)
        assert fresh.sample(other, -1) == cpu_greedy_token

    def test_duplicate_chain_in_constructor_rejected(self, model):
        chain = _greedy()
        with pytest.raises(ValueError, match="more than one seq_id"):
            cy.LlamaContext(model, _params(n_seq_max=2), samplers={0: chain, 1: chain})

    def test_empty_chain_not_attached_or_bound(self, model):
        ctx = cy.LlamaContext(model, _params())
        chain = cy.LlamaSampler()
        assert ctx.set_sampler(0, chain) is False
        assert ctx.backend_samplers == {}
        chain.add_greedy()  # still mutable
        assert ctx.set_sampler(0, chain) is True


class TestLifetime:
    def test_close_while_attached_rejected(self, model):
        ctx = cy.LlamaContext(model, _params())
        chain = _greedy()
        ctx.set_sampler(0, chain)
        with pytest.raises(RuntimeError, match="detach"):
            chain.close()

    def test_close_after_detach(self, model):
        ctx = cy.LlamaContext(model, _params())
        chain = _greedy()
        ctx.set_sampler(0, chain)
        assert ctx.set_sampler(0, None) is True
        assert ctx.backend_samplers == {}
        chain.close()

    def test_close_after_context_closed(self, model):
        ctx = cy.LlamaContext(model, _params())
        chain = _greedy()
        ctx.set_sampler(0, chain)
        ctx.close()
        chain.close()

    def test_replacing_chain_releases_previous(self, model):
        ctx = cy.LlamaContext(model, _params())
        first, second = _greedy(), _greedy()
        ctx.set_sampler(0, first)
        ctx.set_sampler(0, second)
        assert ctx.backend_samplers == {0: second}
        first.close()

    def test_set_sampler_on_closed_context(self, model):
        ctx = cy.LlamaContext(model, _params())
        ctx.close()
        with pytest.raises(RuntimeError, match="closed"):
            ctx.set_sampler(0, _greedy())
        with pytest.raises(RuntimeError, match="closed"):
            ctx.sampled_token_ith(0)


class TestValidation:
    def test_chain_link_rejected(self, model):
        chain = _top_k_dist()
        ctx = cy.LlamaContext(model, _params())
        with pytest.raises(TypeError, match="chain link"):
            ctx.set_sampler(0, chain.chain_get(0))

    def test_non_sampler_rejected(self, model):
        with pytest.raises(TypeError, match="LlamaSampler"):
            cy.LlamaContext(model, _params()).set_sampler(0, object())

    def test_closed_sampler_rejected(self, model):
        chain = _greedy()
        chain.close()
        with pytest.raises(ValueError, match="closed"):
            cy.LlamaContext(model, _params()).set_sampler(0, chain)

    @pytest.mark.parametrize("seq_id", [-1, 1])
    def test_seq_id_out_of_range(self, model, seq_id):
        with pytest.raises(ValueError, match="n_seq_max"):
            cy.LlamaContext(model, _params(n_seq_max=1)).set_sampler(seq_id, _greedy())

    def test_constructor_validates_before_binding(self, model):
        good = _greedy()
        with pytest.raises(ValueError, match="n_seq_max"):
            cy.LlamaContext(model, _params(), samplers={0: good, 7: _greedy()})
        # validation failed before any chain was bound
        assert cy.LlamaContext(model, _params()).set_sampler(0, good)
