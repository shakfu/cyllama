"""Inputs that llama.cpp rejects with GGML_ASSERT / GGML_ABORT, which kill the
process. The wrapper must raise first. Each call here aborted before its guard."""

import pytest

import cyllama.llama.llama_cpp as cy

PROMPT = "The capital of France is"


@pytest.fixture(scope="module")
def model(model_path):
    return cy.LlamaModel(model_path, verbose=False)


def _ctx(model):
    p = cy.LlamaContextParams()
    p.n_ctx = 128
    return cy.LlamaContext(model, p)


class TestMemorySeqId:
    @pytest.mark.parametrize(
        "call",
        [
            lambda c: c.memory_seq_pos_max(300),
            lambda c: c.memory_seq_pos_min(-1),
            lambda c: c.memory_seq_keep(1),
            lambda c: c.memory_seq_add(5, 0, -1, 1),
            lambda c: c.memory_seq_div(5, 0, -1, 2),
            lambda c: c.memory_seq_cp(0, 9, 0, -1),
            lambda c: c.memory_seq_rm(-2, 0, -1),
        ],
        ids=["pos_max", "pos_min", "keep", "add", "div", "cp", "rm"],
    )
    def test_out_of_range_raises(self, model, call):
        with pytest.raises(IndexError, match="n_seq_max"):
            call(_ctx(model))

    def test_rm_all_sequences_allowed(self, model):
        ctx = _ctx(model)
        tokens = model.get_vocab().tokenize(PROMPT, add_special=True, parse_special=False)
        ctx.decode(cy.llama_batch_get_one(tokens))
        assert ctx.memory_seq_rm(-1, 0, -1)
        assert ctx.memory_seq_pos_max(0) < 0

    def test_closed_context_raises(self, model):
        ctx = _ctx(model)
        ctx.close()
        with pytest.raises(RuntimeError, match="closed"):
            ctx.memory_seq_pos_max(0)


class TestLoadMode:
    def test_invalid_value_rejected(self):
        p = cy.LlamaModelParams()
        with pytest.raises(ValueError, match="unknown load mode"):
            p.load_mode = 99
        assert p.load_mode_name  # still readable

    def test_valid_values_accepted(self):
        p = cy.LlamaModelParams()
        for mode in (cy.LLAMA_LOAD_MODE_AUTO, cy.LLAMA_LOAD_MODE_NONE, cy.LLAMA_LOAD_MODE_DIRECT_IO):
            p.load_mode = mode
            assert p.load_mode == mode


class TestChainLink:
    """chain_get() returns a link; the llama_sampler_chain_* functions cast any
    sampler to a chain without checking, so using a link as one corrupts memory."""

    def _link(self):
        s = cy.LlamaSampler()
        s.add_top_k(40)
        s.add_greedy()
        return s, s.chain_get(0)

    def test_len_of_link_is_zero(self):
        _, link = self._link()
        assert len(link) == 0

    def test_add_to_link_rejected(self):
        _, link = self._link()
        with pytest.raises(ValueError, match="not a chain"):
            link.add_greedy()

    @pytest.mark.parametrize("op", ["chain_get", "chain_remove"])
    def test_chain_ops_on_link_rejected(self, op):
        _, link = self._link()
        with pytest.raises(ValueError, match="not a chain"):
            getattr(link, op)(0)
