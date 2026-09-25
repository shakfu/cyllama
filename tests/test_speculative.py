"""
Tests for speculative decoding functionality.

This module tests the Cython wrappers for llama.cpp's speculative decoding API.
"""

import heapq
import math

import pytest
from cyllama.llama.llama_cpp import (
    LlamaBatch,
    LlamaModel,
    LlamaContext,
    LlamaContextParams,
    LlamaModelParams,
    Speculative,
    SpeculativeParams,
)


class TestSpeculativeParams:
    """Tests for SpeculativeParams class."""

    def test_default_initialization(self):
        """Test default parameter initialization."""
        params = SpeculativeParams()
        assert params.n_max == 3
        assert params.n_min == 0
        assert params.p_min == 0.0

    def test_custom_initialization(self):
        """Test custom parameter initialization."""
        params = SpeculativeParams(n_max=32, n_min=4, p_min=0.9)
        assert params.n_max == 32
        assert params.n_min == 4
        assert abs(params.p_min - 0.9) < 0.001

    def test_p_split_removed(self):
        with pytest.raises(TypeError):
            SpeculativeParams(p_split=0.2)

    def test_n_min_and_p_min_are_keyword_only(self):
        """A former (n_max, n_min, p_split) positional call must not silently set p_min."""
        with pytest.raises(TypeError):
            SpeculativeParams(8, 0, 0.1)

    def test_property_setters(self):
        """Test property setters."""
        params = SpeculativeParams()

        params.n_max = 24
        assert params.n_max == 24

        params.n_min = 2
        assert params.n_min == 2

        params.p_min = 0.85
        assert abs(params.p_min - 0.85) < 0.001

    def test_repr(self):
        """Test string representation."""
        params = SpeculativeParams(n_max=20, n_min=2, p_min=0.8)
        repr_str = repr(params)
        assert "SpeculativeParams" in repr_str
        assert "n_max=20" in repr_str
        assert "n_min=2" in repr_str
        assert "p_min=0.8" in repr_str

    def test_n_max_bounds(self):
        """Test n_max parameter with various values."""
        # Small values
        params = SpeculativeParams(n_max=1)
        assert params.n_max == 1

        # Large values
        params = SpeculativeParams(n_max=128)
        assert params.n_max == 128

    def test_p_min_bounds(self):
        """Test p_min parameter with various values."""
        # Minimum probability
        params = SpeculativeParams(p_min=0.0)
        assert params.p_min == 0.0

        # Maximum probability
        params = SpeculativeParams(p_min=1.0)
        assert params.p_min == 1.0

        # Mid-range
        params = SpeculativeParams(p_min=0.5)
        assert params.p_min == 0.5


class TestSpeculativeCompatibility:
    """Tests for speculative decoding compatibility checks."""

    @pytest.mark.slow
    def test_is_compat(self, model_path):
        """Test compatibility check with a model context."""
        # Load model
        model_params = LlamaModelParams()
        model_params.n_gpu_layers = 0  # CPU only for testing
        model = LlamaModel(model_path, model_params)

        # Create context
        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = 512
        ctx = LlamaContext(model, ctx_params)

        # Check compatibility (result depends on model, just ensure no crash)
        result = Speculative.is_compat(ctx)
        assert isinstance(result, bool)

    def test_is_compat_none_context(self):
        """Test compatibility check with None raises appropriate error."""
        # Skip this test as it can cause segfaults
        # The C API doesn't handle NULL pointers gracefully
        pytest.skip("Skipping None test to avoid segfaults in C API")


class TestSpeculativeInitialization:
    """Tests for Speculative class initialization."""

    @pytest.mark.slow
    def test_initialization_requires_speculative_type(self, model_path):
        """Test that initialization fails without a speculative type configured.

        The upstream API requires a valid speculative type (draft model, ngram, etc.)
        to be configured. Default params (type=NONE) will fail initialization.
        """
        model_params = LlamaModelParams()
        model_params.n_gpu_layers = 0
        model = LlamaModel(model_path, model_params)

        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = 512
        ctx_target = LlamaContext(model, ctx_params)

        params = SpeculativeParams()
        with pytest.raises(RuntimeError, match="Failed to initialize"):
            Speculative(params, ctx_target)


class TestSpeculativeEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_params_negative_values(self):
        """Test behavior with negative parameter values."""
        # Cython int allows negative values, but they may not be meaningful
        params = SpeculativeParams(n_max=-1, n_min=-1, p_min=-0.5)
        assert params.n_max == -1
        assert params.n_min == -1
        assert params.p_min == -0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestSpeculativeDraft:
    """draft() with the test model drafting for itself, checked against an
    independent implementation of upstream's rule: decode last_token_id after
    the prompt, then take the top candidate while its probability (softmax
    over the top 10 logits) is at least p_min."""

    N_CTX = 256

    @pytest.fixture(scope="class")
    def model(self, model_path):
        return LlamaModel(model_path, verbose=False)

    @pytest.fixture
    def spec(self, model):
        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = self.N_CTX
        return Speculative(SpeculativeParams(), LlamaContext(model, ctx_params), LlamaContext(model, ctx_params))

    @pytest.fixture(scope="class")
    def prompt(self, model):
        return model.get_vocab().tokenize("The capital of France is", add_special=True, parse_special=False)

    def _reference(self, model, prompt, last_token_id, n_max, p_min=0.0):
        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = self.N_CTX
        ctx = LlamaContext(model, ctx_params)
        batch = LlamaBatch(n_tokens=max(len(prompt), 1), embd=0, n_seq_max=1)

        def decode(tokens, n_past):
            batch.set_batch(tokens, n_past, False)
            ctx.decode(batch)

        if prompt:
            decode(prompt, 0)
        decode([last_token_id], len(prompt))
        result = []
        while len(result) < n_max:
            logits = ctx.get_logits_ith(-1)
            top = heapq.nlargest(10, logits)
            p = 1.0 / sum(math.exp(x - top[0]) for x in top)
            if p_min > 0 and p < p_min:
                break
            tok = max(range(len(logits)), key=logits.__getitem__)
            result.append(tok)
            if len(result) < n_max:
                decode([tok], len(prompt) + len(result))
        return result

    def test_draft_follows_last_token(self, spec, model, prompt):
        """The draft continues after last_token_id, not after prompt[-1]."""
        body, last = prompt[:-1], prompt[-1]
        draft = spec.draft(SpeculativeParams(n_max=5), body, last)
        assert draft == self._reference(model, body, last, 5)

    def test_kv_reuse_across_rounds(self, spec, model, prompt):
        """A second round reuses the cached prefix and drops stale draft tokens."""
        body, last = prompt[:-1], prompt[-1]
        first = spec.draft(SpeculativeParams(n_max=5), body, last)
        # target accepted two draft tokens and sampled the third itself
        body2 = body + [last] + first[:2]
        last2 = first[2]
        second = spec.draft(SpeculativeParams(n_max=5), body2, last2)
        assert second == self._reference(model, body2, last2, 5)

    @pytest.mark.parametrize("p_min", [0.3, 0.6, 0.9])
    def test_p_min_stops_at_low_confidence(self, spec, model, prompt, p_min):
        body, last = prompt[:-1], prompt[-1]
        draft = spec.draft(SpeculativeParams(n_max=8, p_min=p_min), body, last)
        assert draft == self._reference(model, body, last, 8, p_min)

    def test_n_min_discards_short_draft(self, spec, model, prompt):
        body, last = prompt[:-1], prompt[-1]
        n = len(self._reference(model, body, last, 8, p_min=0.6))
        assert n < 8, "prompt must produce a p_min-truncated draft for this test"
        assert spec.draft(SpeculativeParams(n_max=8, n_min=n + 1, p_min=0.6), body, last) == []

    def test_empty_prompt_drafts_from_last_token(self, spec, model, prompt):
        draft = spec.draft(SpeculativeParams(n_max=3), [], prompt[0])
        assert draft == self._reference(model, [], prompt[0], 3)

    @pytest.mark.parametrize("last", [-1, 10**9])
    def test_last_token_outside_vocab_raises(self, spec, prompt, last):
        with pytest.raises(ValueError, match="draft vocabulary"):
            spec.draft(SpeculativeParams(n_max=3), prompt, last)
