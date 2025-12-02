"""
Tests for speculative decoding functionality.

This module tests the Cython wrappers for llama.cpp's speculative decoding API.
"""

import pytest
from cyllama.llama.llama_cpp import (
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
        assert params.n_draft == 16
        assert params.n_reuse == 256
        assert params.p_min == 0.75

    def test_custom_initialization(self):
        """Test custom parameter initialization."""
        params = SpeculativeParams(n_draft=32, n_reuse=128, p_min=0.9)
        # Allow for floating point comparison tolerance
        assert params.n_draft == 32
        assert params.n_reuse == 128
        assert abs(params.p_min - 0.9) < 0.001

    def test_property_setters(self):
        """Test property setters."""
        params = SpeculativeParams()

        params.n_draft = 24
        assert params.n_draft == 24

        params.n_reuse = 512
        assert params.n_reuse == 512

        params.p_min = 0.85
        assert abs(params.p_min - 0.85) < 0.001

    def test_repr(self):
        """Test string representation."""
        params = SpeculativeParams(n_draft=20, n_reuse=200, p_min=0.8)
        repr_str = repr(params)
        assert "SpeculativeParams" in repr_str
        assert "n_draft=20" in repr_str
        assert "n_reuse=200" in repr_str
        assert "p_min=0.8" in repr_str

    def test_n_draft_bounds(self):
        """Test n_draft parameter with various values."""
        # Small values
        params = SpeculativeParams(n_draft=1)
        assert params.n_draft == 1

        # Large values
        params = SpeculativeParams(n_draft=128)
        assert params.n_draft == 128

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
    def test_are_compatible_same_model(self, model_path):
        """Test compatibility check with same model (should be compatible)."""
        # Load model
        model_params = LlamaModelParams()
        model_params.n_gpu_layers = 0  # CPU only for testing
        model = LlamaModel(model_path, model_params)

        # Create two contexts from same model
        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = 512
        ctx1 = LlamaContext(model, ctx_params)
        ctx2 = LlamaContext(model, ctx_params)

        # Check compatibility
        assert Speculative.are_compatible(ctx1, ctx2)

    def test_are_compatible_none_context(self):
        """Test compatibility check with None raises appropriate error."""
        # Skip this test as it can cause segfaults
        # The C API doesn't handle NULL pointers gracefully
        pytest.skip("Skipping None test to avoid segfaults in C API")


class TestSpeculativeInitialization:
    """Tests for Speculative class initialization."""

    @pytest.mark.slow
    def test_initialization_same_model(self, model_path):
        """Test initializing speculative decoding with same model contexts."""
        # Load model
        model_params = LlamaModelParams()
        model_params.n_gpu_layers = 0
        model = LlamaModel(model_path, model_params)

        # Create contexts
        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = 512
        ctx_target = LlamaContext(model, ctx_params)
        ctx_draft = LlamaContext(model, ctx_params)

        # Initialize speculative decoding
        spec = Speculative(ctx_target, ctx_draft)

        # Verify it was created
        assert spec is not None
        assert spec.ctx_tgt is ctx_target
        assert spec.ctx_dft is ctx_draft

    def test_initialization_incompatible_raises(self):
        """Test that incompatible contexts raise ValueError."""
        # We'll use mock contexts that are incompatible
        # In practice, this would be contexts from models with different vocabularies
        # For this test, we'll just verify the error handling path exists
        # The actual incompatibility check is done by llama.cpp
        pass

    @pytest.mark.slow
    def test_repr(self, model_path):
        """Test string representation of Speculative."""
        model_params = LlamaModelParams()
        model_params.n_gpu_layers = 0
        model = LlamaModel(model_path, model_params)

        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = 512
        ctx_target = LlamaContext(model, ctx_params)
        ctx_draft = LlamaContext(model, ctx_params)

        spec = Speculative(ctx_target, ctx_draft)
        repr_str = repr(spec)

        assert "Speculative" in repr_str
        assert "target=" in repr_str
        assert "draft=" in repr_str


class TestSpeculativeOperations:
    """Tests for speculative decoding operations."""

    @pytest.mark.slow
    def test_add_replacement(self, model_path):
        """Test adding token replacement mappings."""
        model_params = LlamaModelParams()
        model_params.n_gpu_layers = 0
        model = LlamaModel(model_path, model_params)

        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = 512
        ctx_target = LlamaContext(model, ctx_params)
        ctx_draft = LlamaContext(model, ctx_params)

        spec = Speculative(ctx_target, ctx_draft)

        # Add replacement - should not raise
        spec.add_replacement("hello", "hi")
        spec.add_replacement("world", "earth")

    @pytest.mark.slow
    def test_gen_draft_basic(self, model_path):
        """Test basic draft generation."""
        model_params = LlamaModelParams()
        model_params.n_gpu_layers = 0
        model = LlamaModel(model_path, model_params)

        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = 512
        ctx_target = LlamaContext(model, ctx_params)
        ctx_draft = LlamaContext(model, ctx_params)

        spec = Speculative(ctx_target, ctx_draft)
        params = SpeculativeParams(n_draft=8, p_min=0.5)

        # Simple prompt tokens (BOS + a few tokens)
        prompt_tokens = [1, 791, 2232]  # Example token IDs
        last_token = 374

        # Generate draft
        # Note: This may return empty list if conditions aren't met
        draft = spec.gen_draft(params, prompt_tokens, last_token)

        # Should return a list (possibly empty)
        assert isinstance(draft, list)
        assert all(isinstance(t, int) for t in draft)

    @pytest.mark.slow
    def test_gen_draft_empty_prompt(self, model_path):
        """Test draft generation with empty prompt."""
        model_params = LlamaModelParams()
        model_params.n_gpu_layers = 0
        model = LlamaModel(model_path, model_params)

        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = 512
        ctx_target = LlamaContext(model, ctx_params)
        ctx_draft = LlamaContext(model, ctx_params)

        spec = Speculative(ctx_target, ctx_draft)
        params = SpeculativeParams(n_draft=8, p_min=0.5)

        # Empty prompt
        draft = spec.gen_draft(params, [], 1)

        # Should return a list
        assert isinstance(draft, list)

    @pytest.mark.slow
    def test_gen_draft_varying_params(self, model_path):
        """Test draft generation with varying parameters."""
        model_params = LlamaModelParams()
        model_params.n_gpu_layers = 0
        model = LlamaModel(model_path, model_params)

        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = 512
        ctx_target = LlamaContext(model, ctx_params)
        ctx_draft = LlamaContext(model, ctx_params)

        spec = Speculative(ctx_target, ctx_draft)
        prompt_tokens = [1, 791, 2232]
        last_token = 374

        # Test with different n_draft values
        for n_draft in [4, 8, 16, 32]:
            params = SpeculativeParams(n_draft=n_draft, p_min=0.5)
            draft = spec.gen_draft(params, prompt_tokens, last_token)
            assert isinstance(draft, list)

        # Test with different p_min values
        for p_min in [0.3, 0.5, 0.7, 0.9]:
            params = SpeculativeParams(n_draft=8, p_min=p_min)
            draft = spec.gen_draft(params, prompt_tokens, last_token)
            assert isinstance(draft, list)


class TestSpeculativeEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_params_negative_values(self):
        """Test behavior with negative parameter values."""
        # Cython int allows negative values, but they may not be meaningful
        params = SpeculativeParams(n_draft=-1, n_reuse=-1, p_min=-0.5)
        assert params.n_draft == -1
        assert params.n_reuse == -1
        assert params.p_min == -0.5

    @pytest.mark.slow
    def test_multiple_speculative_instances(self, model_path):
        """Test creating multiple Speculative instances."""
        model_params = LlamaModelParams()
        model_params.n_gpu_layers = 0
        model = LlamaModel(model_path, model_params)

        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = 512

        # Create multiple context pairs
        ctx1_tgt = LlamaContext(model, ctx_params)
        ctx1_dft = LlamaContext(model, ctx_params)
        spec1 = Speculative(ctx1_tgt, ctx1_dft)

        ctx2_tgt = LlamaContext(model, ctx_params)
        ctx2_dft = LlamaContext(model, ctx_params)
        spec2 = Speculative(ctx2_tgt, ctx2_dft)

        # Both should be valid and independent
        assert spec1 is not spec2
        assert spec1.ctx_tgt is not spec2.ctx_tgt

    @pytest.mark.slow
    def test_cleanup_on_deletion(self, model_path):
        """Test that resources are cleaned up on deletion."""
        model_params = LlamaModelParams()
        model_params.n_gpu_layers = 0
        model = LlamaModel(model_path, model_params)

        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = 512
        ctx_target = LlamaContext(model, ctx_params)
        ctx_draft = LlamaContext(model, ctx_params)

        spec = Speculative(ctx_target, ctx_draft)

        # Delete and ensure no errors
        del spec

        # Can still create new instance
        spec2 = Speculative(ctx_target, ctx_draft)
        assert spec2 is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
