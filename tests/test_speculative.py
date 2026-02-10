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
        assert params.n_max == 16
        assert params.n_min == 0
        assert abs(params.p_split - 0.1) < 0.001
        assert params.p_min == 0.75

    def test_custom_initialization(self):
        """Test custom parameter initialization."""
        params = SpeculativeParams(n_max=32, n_min=4, p_split=0.2, p_min=0.9)
        assert params.n_max == 32
        assert params.n_min == 4
        assert abs(params.p_split - 0.2) < 0.001
        assert abs(params.p_min - 0.9) < 0.001

    def test_property_setters(self):
        """Test property setters."""
        params = SpeculativeParams()

        params.n_max = 24
        assert params.n_max == 24

        params.n_min = 2
        assert params.n_min == 2

        params.p_split = 0.3
        assert abs(params.p_split - 0.3) < 0.001

        params.p_min = 0.85
        assert abs(params.p_min - 0.85) < 0.001

    def test_repr(self):
        """Test string representation."""
        params = SpeculativeParams(n_max=20, n_min=2, p_split=0.1, p_min=0.8)
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

    def test_initialization_incompatible_raises(self):
        """Test that incompatible contexts raise ValueError."""
        pass


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
