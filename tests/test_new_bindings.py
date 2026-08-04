"""Tests for bindings added alongside the llama.cpp / sd.cpp update.

Covers the newly wrapped llama.cpp surface (pooled embeddings, classifier
head accessors, suppress tokens, the DRY / top-n-sigma / adaptive-p
samplers, lazy grammars, sampler-chain introspection, memory shifting,
split-path helpers) plus the stable-diffusion.cpp version accessors.

Whisper VAD and mtmd video/batch coverage live in test_whisper_vad.py and
test_mtmd_video_batch.py respectively, since they need their own models.
"""

import gc
from pathlib import Path

import pytest

import cyllama.llama.llama_cpp as cy

ROOT = Path.cwd()
RERANKER_MODEL = ROOT / "models" / "bge-reranker-base-q8_0.gguf"


@pytest.fixture(scope="module")
def model(model_path):
    """Module-scoped model, reused across the read-only accessor tests."""
    params = cy.LlamaModelParams()
    params.n_gpu_layers = -1
    m = cy.LlamaModel(model_path, params)
    yield m
    m.close()
    gc.collect()


# =============================================================================
# Model accessors
# =============================================================================


class TestModelAccessors:
    def test_n_swa_is_zero_for_full_attention_model(self, model):
        """Llama 3.2 uses full attention, so the SWA window is 0."""
        assert model.n_swa == 0

    def test_n_embd_out(self, model):
        assert model.n_embd_out > 0

    def test_n_cls_out_defaults_to_one(self, model):
        """Generative models report a single (meaningless) classifier output."""
        assert model.n_cls_out == 1

    def test_cls_label_returns_none_without_labels(self, model):
        assert model.cls_label(0) is None

    def test_cls_label_rejects_out_of_range(self, model):
        with pytest.raises(IndexError):
            model.cls_label(model.n_cls_out)


class TestSuppressTokens:
    def test_returns_list(self, model):
        """Models without the gguf key report an empty list, not an error."""
        tokens = model.get_vocab().get_suppress_tokens()
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)


# =============================================================================
# Samplers
# =============================================================================


class TestNewSamplers:
    def test_add_top_n_sigma(self):
        with cy.LlamaSampler(cy.LlamaSamplerChainParams()) as s:
            s.add_top_n_sigma(2.0)
            assert len(s) == 1
            assert s.chain_get(0).name() == "top-n-sigma"

    def test_add_adaptive_p(self):
        with cy.LlamaSampler(cy.LlamaSamplerChainParams()) as s:
            s.add_adaptive_p(0.9, 0.1, 42)
            assert len(s) == 1

    def test_add_dry_with_breakers(self, model):
        vocab = model.get_vocab()
        with cy.LlamaSampler(cy.LlamaSamplerChainParams()) as s:
            s.add_dry(vocab, model.n_ctx_train, 0.8, 1.75, 2, -1, ["\n", ":", '"', "*"])
            assert len(s) == 1

    def test_add_dry_without_breakers(self, model):
        """seq_breakers is optional; None must not crash the char** marshalling."""
        vocab = model.get_vocab()
        with cy.LlamaSampler(cy.LlamaSamplerChainParams()) as s:
            s.add_dry(vocab, model.n_ctx_train, 0.8, 1.75, 2, -1)
            assert len(s) == 1

    def test_add_dry_disabled_multiplier_still_builds(self, model):
        """multiplier 0.0 is the disabled value; upstream returns a no-op link."""
        vocab = model.get_vocab()
        with cy.LlamaSampler(cy.LlamaSamplerChainParams()) as s:
            s.add_dry(vocab, model.n_ctx_train, 0.0, 1.75, 2, -1)
            assert len(s) == 1

    def test_add_grammar_lazy_patterns(self, model):
        vocab = model.get_vocab()
        with cy.LlamaSampler(cy.LlamaSamplerChainParams()) as s:
            s.add_grammar_lazy_patterns(
                vocab,
                'root ::= "yes" | "no"',
                "root",
                trigger_patterns=[".*?(<tool_call>)[\\s\\S]*"],
                trigger_tokens=[vocab.token_eos()],
            )
            assert len(s) == 1

    def test_add_grammar_lazy_patterns_no_triggers(self, model):
        """Both trigger lists optional; empty arrays must marshal cleanly."""
        vocab = model.get_vocab()
        with cy.LlamaSampler(cy.LlamaSamplerChainParams()) as s:
            s.add_grammar_lazy_patterns(vocab, 'root ::= "a"', "root")
            assert len(s) == 1


class TestSamplerChainIntrospection:
    def test_len_and_get(self):
        with cy.LlamaSampler(cy.LlamaSamplerChainParams()) as s:
            assert len(s) == 0
            s.add_top_k(40)
            s.add_temp(0.7)
            s.add_dist(1)
            assert len(s) == 3
            assert [s.chain_get(i).name() for i in range(3)] == ["top-k", "temp", "dist"]

    def test_negative_index(self):
        with cy.LlamaSampler(cy.LlamaSamplerChainParams()) as s:
            s.add_top_k(40)
            s.add_dist(1)
            assert s.chain_get(-1).name() == "dist"

    def test_get_out_of_range(self):
        with cy.LlamaSampler(cy.LlamaSamplerChainParams()) as s:
            s.add_top_k(40)
            with pytest.raises(IndexError):
                s.chain_get(5)

    def test_remove_detaches_link(self):
        with cy.LlamaSampler(cy.LlamaSamplerChainParams()) as s:
            s.add_top_k(40)
            s.add_temp(0.7)
            s.add_dist(1)
            detached = s.chain_remove(1)
            assert detached.name() == "temp"
            assert len(s) == 2
            assert [s.chain_get(i).name() for i in range(2)] == ["top-k", "dist"]

    def test_remove_out_of_range(self):
        with cy.LlamaSampler(cy.LlamaSamplerChainParams()) as s:
            with pytest.raises(IndexError):
                s.chain_remove(0)


# =============================================================================
# Context: memory + pooled embeddings
# =============================================================================


class TestContextMemory:
    def test_can_shift(self, model):
        ctx = cy.LlamaContext(model, cy.LlamaContextParams())
        try:
            assert ctx.memory_can_shift() is True
        finally:
            ctx.close()
            gc.collect()

    def test_seq_div_rejects_non_positive_divisor(self, model):
        ctx = cy.LlamaContext(model, cy.LlamaContextParams())
        try:
            with pytest.raises(ValueError):
                ctx.memory_seq_div(0, 0, 10, 0)
        finally:
            ctx.close()
            gc.collect()

    def test_get_embeddings_seq_none_without_pooling(self, model):
        """No pooled output configured -> None rather than an exception."""
        ctx = cy.LlamaContext(model, cy.LlamaContextParams())
        try:
            assert ctx.get_embeddings_seq(0) is None
        finally:
            ctx.close()
            gc.collect()


class TestGetLogits:
    def test_get_logits_length_matches_vocab(self, model):
        """Regression: get_logits() read n_vocab off LlamaModel, which has no
        such attribute, so every call raised AttributeError."""
        vocab = model.get_vocab()
        ctx = cy.LlamaContext(model, cy.LlamaContextParams())
        try:
            tokens = vocab.tokenize("Hello world", add_special=True, parse_special=False)
            batch = cy.LlamaBatch(n_tokens=len(tokens), embd=0, n_seq_max=1)
            batch.add_sequence(tokens, 0, False)
            ctx.decode(batch)

            logits = ctx.get_logits()
            assert logits is not None
            assert len(logits) == vocab.n_vocab
        finally:
            ctx.close()
            gc.collect()

    def test_get_logits_ith_length_matches_vocab(self, model):
        vocab = model.get_vocab()
        ctx = cy.LlamaContext(model, cy.LlamaContextParams())
        try:
            tokens = vocab.tokenize("Hello world", add_special=True, parse_special=False)
            batch = cy.LlamaBatch(n_tokens=len(tokens), embd=0, n_seq_max=1)
            batch.add_sequence(tokens, 0, False)
            ctx.decode(batch)

            assert len(ctx.get_logits_ith(-1)) == vocab.n_vocab
        finally:
            ctx.close()
            gc.collect()


# =============================================================================
# Pooling constants + RANK reranking
# =============================================================================


class TestPoolingConstants:
    def test_values_match_header(self):
        assert cy.LLAMA_POOLING_TYPE_UNSPECIFIED == -1
        assert cy.LLAMA_POOLING_TYPE_NONE == 0
        assert cy.LLAMA_POOLING_TYPE_MEAN == 1
        assert cy.LLAMA_POOLING_TYPE_CLS == 2
        assert cy.LLAMA_POOLING_TYPE_LAST == 3
        assert cy.LLAMA_POOLING_TYPE_RANK == 4


@pytest.mark.skipif(not RERANKER_MODEL.exists(), reason="reranker model not available")
class TestRankPooling:
    def test_rank_pooling_scores_relevant_document_higher(self):
        """RANK pooling attaches the classification head; get_embeddings_seq
        is the only way to read the resulting relevance score."""
        params = cy.LlamaModelParams()
        params.n_gpu_layers = -1
        model = cy.LlamaModel(str(RERANKER_MODEL), params)
        vocab = model.get_vocab()

        ctx_params = cy.LlamaContextParams()
        ctx_params.n_ctx = 512
        ctx_params.embeddings = True
        ctx_params.pooling_type = cy.LLAMA_POOLING_TYPE_RANK
        ctx = cy.LlamaContext(model, ctx_params)

        try:
            query = "What is the capital of France?"

            def score(document):
                tokens = list(vocab.tokenize(query, add_special=True, parse_special=False))
                tokens += list(vocab.tokenize(document, add_special=False, parse_special=False))
                batch = cy.LlamaBatch(n_tokens=len(tokens), embd=0, n_seq_max=1)
                batch.add_sequence(tokens, 0, False)
                ctx.memory_seq_rm(0, -1, -1)
                ctx.decode(batch)
                out = ctx.get_embeddings_seq(0)
                assert out is not None
                assert len(out) == model.n_cls_out
                return out[0]

            relevant = score("Paris is the capital of France.")
            irrelevant = score("Bananas are a rich source of potassium.")
            assert relevant > irrelevant
        finally:
            ctx.close()
            model.close()
            gc.collect()


# =============================================================================
# Module-level helpers
# =============================================================================


class TestSplitPathHelpers:
    def test_split_path_is_zero_based(self):
        """split_no is 0-based upstream; the name carries split_no + 1."""
        assert cy.llama_split_path("/models/ggml-model-q4_0", 1, 4) == "/models/ggml-model-q4_0-00002-of-00004.gguf"

    def test_split_prefix_roundtrip(self):
        path = cy.llama_split_path("/models/ggml-model-q4_0", 1, 4)
        assert cy.llama_split_prefix(path, 1, 4) == "/models/ggml-model-q4_0"

    def test_split_prefix_returns_none_on_mismatch(self):
        assert cy.llama_split_prefix("/models/plain.gguf", 1, 4) is None

    def test_split_path_rejects_bad_indices(self):
        with pytest.raises(ValueError):
            cy.llama_split_path("/models/m", 4, 4)
        with pytest.raises(ValueError):
            cy.llama_split_path("/models/m", 0, 0)


class TestSystemInfo:
    def test_print_system_info_returns_text(self):
        info = cy.llama_print_system_info()
        assert isinstance(info, str)
        assert len(info) > 0


# =============================================================================
# stable-diffusion.cpp
# =============================================================================


class TestSDVersion:
    def test_version_and_commit(self):
        import cyllama.sd.stable_diffusion as sd

        assert isinstance(sd.version(), str)
        assert isinstance(sd.commit(), str)
        assert sd.commit()

    def test_lms_sample_method_present(self):
        """LMS was added upstream; SampleMethod.COUNT must stay in sync."""
        import cyllama.sd.stable_diffusion as sd

        assert sd.SampleMethod.LMS < sd.SampleMethod.COUNT

    def test_adetailer_missing_model_raises(self):
        import cyllama.sd.stable_diffusion as sd

        with pytest.raises(FileNotFoundError):
            sd.Adetailer("definitely-not-a-real-detector.gguf")
