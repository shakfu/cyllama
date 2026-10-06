"""Extended batch API (llama_batch_ext + llama_process) and the other llama.h
additions from llama.cpp v0.6.0: causal-attention getter, sampler state copy,
and the new context/model params."""

import gc
import math
import weakref

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


def _ctx(model, n_batch=64):
    p = cy.LlamaContextParams()
    p.n_ctx = 256
    p.n_batch = n_batch
    return cy.LlamaContext(model, p)


def _token_batch(ctx, tokens):
    batch = cy.LlamaBatchExt(ctx)
    for i, t in enumerate(tokens):
        batch.add_token(t, i, output=(i == len(tokens) - 1))
    return batch


class TestProcess:
    def test_matches_legacy_decode(self, model, tokens):
        legacy = _ctx(model)
        legacy.decode(cy.llama_batch_get_one(tokens))
        expected = legacy.get_logits_ith(-1)

        ctx = _ctx(model)
        batch = _token_batch(ctx, tokens)
        assert len(batch) == len(tokens)
        assert ctx.process(batch) == 0
        got = ctx.get_logits_ith(-1)

        assert len(got) == len(expected)
        assert max(abs(a - b) for a, b in zip(got, expected)) < 1e-3
        assert ctx.memory_seq_pos_max(0) == len(tokens) - 1

    def test_embedding_entry(self, model, tokens):
        ctx = _ctx(model)
        batch = cy.LlamaBatchExt(ctx)
        assert batch.n_pos_per_embd == 1  # not an M-RoPE model
        batch.add_embd([0.01] * model.n_embd_inp, 0, output=True)
        assert ctx.process(batch) == 0
        logits = ctx.get_logits_ith(-1)
        assert all(math.isfinite(x) for x in logits[:100])

    def test_add_seq_shares_entry(self, model, tokens):
        p = cy.LlamaContextParams()
        p.n_ctx = 256
        p.n_seq_max = 2
        p.kv_unified = True  # entries shared across sequences need a unified KV cache
        ctx = cy.LlamaContext(model, p)
        batch = cy.LlamaBatchExt(ctx)
        for i, t in enumerate(tokens):
            idx = batch.add_token(t, i)
            batch.add_seq(idx, 1)
        batch.set_output(len(tokens) - 1)
        ctx.process(batch)
        assert ctx.memory_seq_pos_max(0) == ctx.memory_seq_pos_max(1) == len(tokens) - 1

    def test_clear(self, model, tokens):
        ctx = _ctx(model)
        batch = _token_batch(ctx, tokens)
        batch.clear()
        assert len(batch) == 0
        assert batch.add_token(tokens[0], 0) == 0


class TestBatchErrors:
    def test_batch_full(self, model):
        ctx = _ctx(model, n_batch=2)
        batch = cy.LlamaBatchExt(ctx)
        batch.add_token(1, 0)
        batch.add_token(1, 1)
        with pytest.raises(ValueError, match="full"):
            batch.add_token(1, 2)
        assert len(batch) == 2

    def test_invalid_seq_id(self, model):
        batch = cy.LlamaBatchExt(_ctx(model))
        with pytest.raises(ValueError, match="seq_id"):
            batch.add_token(1, 0, seq_id=5)

    def test_invalid_token(self, model):
        batch = cy.LlamaBatchExt(_ctx(model))
        with pytest.raises(ValueError, match="token"):
            batch.add_token(model.get_vocab().n_vocab, 0)

    def test_rejected_entry_blocks_batch_until_clear(self, model):
        # upstream keeps a rejected entry, which would shift every later index
        ctx = _ctx(model)
        batch = cy.LlamaBatchExt(ctx)
        with pytest.raises(ValueError, match="embedding"):
            batch.add_embd([0.0] * 3, 0)
        with pytest.raises(ValueError, match="clear"):
            batch.add_token(1, 0)
        with pytest.raises(ValueError, match="clear"):
            ctx.process(batch)
        batch.clear()
        assert batch.add_token(1, 0) == 0

    def test_wrong_embedding_size(self, model):
        batch = cy.LlamaBatchExt(_ctx(model))
        with pytest.raises(ValueError, match="embedding"):
            batch.add_embd([0.0] * 3, 0)

    def test_wrong_position_count(self, model):
        batch = cy.LlamaBatchExt(_ctx(model))
        with pytest.raises(ValueError, match="position"):
            batch.add_token(1, [0, 0, 0, 0])
        assert len(batch) == 0  # rejected before the entry was added

    def test_bad_index(self, model):
        batch = cy.LlamaBatchExt(_ctx(model))
        with pytest.raises(IndexError):
            batch.set_output(0)
        with pytest.raises(IndexError):
            batch.set_pos(0, 1)
        with pytest.raises(ValueError):
            batch.add_seq(0, 0)

    def test_embd_state_unimplemented_upstream(self, model):
        batch = cy.LlamaBatchExt(_ctx(model))
        idx = batch.add_token(1, 0)
        with pytest.raises(RuntimeError, match="set_embd_state"):
            batch.set_embd_state(idx, [0.0] * 8)

    def test_other_context_rejected(self, model, tokens):
        batch = _token_batch(_ctx(model), tokens)
        with pytest.raises(ValueError, match="different context"):
            _ctx(model).process(batch)

    def test_batch_keeps_context_alive(self, model):
        ctx = _ctx(model)
        ref = weakref.ref(ctx)
        batch = cy.LlamaBatchExt(ctx)
        del ctx
        gc.collect()
        assert ref() is not None
        assert batch.ctx is ref()


def test_causal_attn_roundtrip(model):
    ctx = _ctx(model)
    assert ctx.get_causal_attn() is True
    ctx.set_causal_attn(False)
    assert ctx.get_causal_attn() is False
    ctx.set_causal_attn(True)
    assert ctx.get_causal_attn() is True


class TestSamplerCopy:
    @staticmethod
    def _dist(seed):
        s = cy.LlamaSampler()
        s.add_top_k(40)
        s.add_dist(seed)
        return s

    def test_copies_rng_state(self, model, tokens):
        ctx = _ctx(model)
        ctx.decode(cy.llama_batch_get_one(tokens))
        src, dst = self._dist(7), self._dist(7)
        for _ in range(5):
            src.sample(ctx, -1)
        src.copy_to(dst)
        assert [src.sample(ctx, -1) for _ in range(8)] == [dst.sample(ctx, -1) for _ in range(8)]

    def test_rejects_different_layout(self):
        greedy = cy.LlamaSampler()
        greedy.add_greedy()
        with pytest.raises(TypeError):
            self._dist(1).copy_to(greedy)  # chain lengths differ
        other = cy.LlamaSampler()
        other.add_top_k(40)
        other.add_greedy()
        with pytest.raises(TypeError):
            self._dist(1).copy_to(other)  # same length, different child types


class TestParams:
    def test_model_params_load_mtp(self):
        p = cy.LlamaModelParams()
        assert p.load_mtp is False
        p.load_mtp = True
        assert p.load_mtp is True

    @pytest.mark.parametrize("name", ["n_rs_seq", "n_outputs_max", "n_outputs_max_per_seq"])
    def test_context_params_counts(self, name):
        p = cy.LlamaContextParams()
        setattr(p, name, 3)
        assert getattr(p, name) == 3

    def test_ctx_type(self):
        p = cy.LlamaContextParams()
        assert p.ctx_type == cy.LLAMA_CONTEXT_TYPE_DEFAULT
        p.ctx_type = cy.LLAMA_CONTEXT_TYPE_MTP
        assert p.ctx_type == cy.LLAMA_CONTEXT_TYPE_MTP
        with pytest.raises(ValueError):
            p.ctx_type = 9

    def test_ctx_other_kept_alive_by_context(self, model):
        other = _ctx(model)
        ref = weakref.ref(other)
        p = cy.LlamaContextParams()
        p.n_ctx = 256
        p.ctx_other = other
        assert p.ctx_other is other
        ctx = cy.LlamaContext(model, p)
        p.ctx_other = None
        del other
        gc.collect()
        assert ref() is not None
        del ctx
        gc.collect()
        assert ref() is None
