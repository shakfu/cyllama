"""LlamaModel.from_user (llama_model_init_from_user), the GgmlTensor accessors
it hands to the callback, the GGUF array setters, and M-RoPE batches on a
synthetic qwen2vl model."""

import math

import pytest

import cyllama.llama.llama_cpp as cy
from synthetic_models import synthetic_metadata, synthetic_model

LLAMA_ROPE_TYPE_NORM = 0
LLAMA_ROPE_TYPE_MROPE = 8


@pytest.fixture(scope="module", autouse=True)
def backend():
    cy.llama_backend_init()


def _ctx(model, n_seq_max=1):
    p = cy.LlamaContextParams()
    p.n_ctx = 128
    p.n_batch = 32
    p.n_seq_max = n_seq_max
    return cy.LlamaContext(model, p, verbose=False)


def _last_logits(model, tokens):
    ctx = _ctx(model)
    ctx.decode(cy.llama_batch_get_one(tokens))
    return ctx.get_logits_ith(-1)


class TestFromUser:
    def test_hparams_follow_metadata(self):
        m = synthetic_model("llama", n_vocab=64, n_embd=128, n_layer=3)
        assert m.get_vocab().n_vocab == 64
        assert m.n_embd == 128
        assert m.n_layer == 3
        assert m.rope_type == LLAMA_ROPE_TYPE_NORM
        assert m.path_model is None

    def test_decodes_finite_and_deterministic(self):
        tokens = [1, 5, 9, 3]
        a = _last_logits(synthetic_model("llama", seed=1), tokens)
        b = _last_logits(synthetic_model("llama", seed=1), tokens)
        c = _last_logits(synthetic_model("llama", seed=2), tokens)
        assert len(a) == 128
        assert all(math.isfinite(x) for x in a)
        assert a == b
        assert a != c

    def test_callback_sees_every_tensor(self):
        seen = {}

        def record(t):
            seen[t.name] = (t.shape, t.type_name, t.n_elements, t.nbytes)
            t.set_data(bytes(t.nbytes))

        with synthetic_metadata("llama", n_vocab=128, n_embd=256, n_layer=2) as md:
            cy.LlamaModel.from_user(md, record, verbose=False)
        assert seen["token_embd.weight"] == ((256, 128), "f32", 256 * 128, 4 * 256 * 128)
        assert "output_norm.weight" in seen
        assert {f"blk.{i}.attn_q.weight" for i in range(2)} <= set(seen)

    def test_callback_error_propagates_and_skips_rest(self):
        calls = []

        def fail(t):
            calls.append(t.name)
            raise KeyError("boom")

        with synthetic_metadata() as md:
            with pytest.raises(KeyError, match="boom"):
                cy.LlamaModel.from_user(md, fail, verbose=False)
        assert len(calls) == 1

    def test_view_invalid_after_callback(self):
        kept = []

        def keep(t):
            kept.append(t)
            t.set_data(bytes(t.nbytes))

        with synthetic_metadata() as md:
            cy.LlamaModel.from_user(md, keep, verbose=False)
        with pytest.raises(ValueError, match="no longer valid"):
            kept[0].name

    def test_set_data_size_checked(self):
        with synthetic_metadata() as md:
            with pytest.raises(ValueError, match="needs"):
                cy.LlamaModel.from_user(md, lambda t: t.set_data(b"\0" * 4), verbose=False)

    def test_bad_metadata_raises(self):
        with cy.GGUFContext.empty() as md:
            md.set_val_str("general.architecture", "no-such-arch")
            with pytest.raises(ValueError, match="init_from_user"):
                cy.LlamaModel.from_user(md, lambda t: None, verbose=False)

    def test_callback_must_be_callable(self):
        with synthetic_metadata() as md:
            with pytest.raises(TypeError):
                cy.LlamaModel.from_user(md, 1, verbose=False)


class TestGgufArrays:
    def test_string_and_numeric_arrays(self):
        with cy.GGUFContext.empty() as g:
            g.set_arr_str("a.s", ["x", "yz", ""])
            g.set_arr_data("a.u", cy.GGUF_TYPE_UINT32, [1, 2, 3])
            assert g.get_value("a.s") == ["x", "yz", ""]
            assert g.get_value("a.u") == {"type": cy.GGUF_TYPE_UINT32, "length": 3}

    def test_rejects_non_numeric_type(self):
        with cy.GGUFContext.empty() as g:
            with pytest.raises(ValueError):
                g.set_arr_data("a", cy.GGUF_TYPE_STRING, [1])


class TestMRoPE:
    """qwen2vl uses M-RoPE: embedding entries carry 4 positions."""

    @pytest.fixture(scope="class")
    def model(self):
        return synthetic_model("qwen2vl")

    def test_rope_type(self, model):
        assert model.rope_type == LLAMA_ROPE_TYPE_MROPE

    def test_embedding_entries_take_four_positions(self, model):
        ctx = _ctx(model)
        batch = cy.LlamaBatchExt(ctx)
        assert batch.n_pos_per_embd == 4
        batch.add_token(1, 0)
        with pytest.raises(ValueError, match="4 position"):
            batch.add_embd([0.01] * model.n_embd_inp, 1)
        for i in range(3):
            batch.add_embd([0.01] * model.n_embd_inp, [1, 1, i, 0], output=(i == 2))
        assert ctx.process(batch) == 0
        assert all(math.isfinite(x) for x in ctx.get_logits_ith(-1))

    def test_token_entries_take_one_position(self, model):
        batch = cy.LlamaBatchExt(_ctx(model))
        with pytest.raises(ValueError, match="1 position"):
            batch.add_token(1, [0, 0, 0, 0])
