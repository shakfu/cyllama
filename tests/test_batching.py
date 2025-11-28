"""
Tests for batch processing functionality.
"""

import pytest
from cyllama import (
    batch_generate,
    BatchGenerator,
    BatchRequest,
    BatchResponse,
    GenerationConfig,
)

# Skip all tests if model is not available
from pathlib import Path
from conftest import DEFAULT_MODEL
pytestmark = pytest.mark.skipif(
    not Path(DEFAULT_MODEL).exists(),
    reason="Model not found"
)


class TestBatchGenerate:
    """Test the convenience batch_generate function."""

    def test_basic_batch_generate(self, model_path):
        """Test basic batch generation with multiple prompts."""
        prompts = [
            "What is 2+2?",
            "What is 3+3?",
        ]

        config = GenerationConfig(
            max_tokens=20,
            temperature=0.0
        )

        responses = batch_generate(
            prompts,
            model_path=model_path,
            n_seq_max=2,
            config=config
        )

        assert len(responses) == len(prompts)
        for response in responses:
            assert isinstance(response, str)
            assert len(response) > 0

    def test_single_prompt(self, model_path):
        """Test batch generation with a single prompt."""
        prompts = ["Hello, how are you?"]

        config = GenerationConfig(max_tokens=10, temperature=0.0)

        responses = batch_generate(
            prompts,
            model_path=model_path,
            n_seq_max=1,
            config=config
        )

        assert len(responses) == 1
        assert isinstance(responses[0], str)
        assert len(responses[0]) > 0

    def test_empty_prompts(self, model_path):
        """Test batch generation with empty prompt list."""
        prompts = []

        responses = batch_generate(
            prompts,
            model_path=model_path,
            n_seq_max=1
        )

        assert len(responses) == 0

    def test_max_sequences_exceeded(self, model_path):
        """Test that error is raised when too many prompts for n_seq_max."""
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]

        config = GenerationConfig(max_tokens=5, temperature=0.0)

        with pytest.raises(ValueError, match="Too many prompts"):
            batch_generate(
                prompts,
                model_path=model_path,
                n_seq_max=2,  # Only 2 sequences allowed
                config=config
            )


class TestBatchGenerator:
    """Test the BatchGenerator class."""

    def test_initialization(self, model_path):
        """Test BatchGenerator initialization."""
        gen = BatchGenerator(
            model_path=model_path,
            batch_size=512,
            n_ctx=2048,
            n_seq_max=4,
            verbose=False
        )

        assert gen.model is not None
        assert gen.ctx is not None
        assert gen.vocab is not None
        assert gen.n_seq_max == 4

    def test_generate_batch_basic(self, model_path):
        """Test basic batch generation."""
        gen = BatchGenerator(
            model_path=model_path,
            n_seq_max=3,
            verbose=False
        )

        prompts = ["Hi", "Hello", "Hey"]
        config = GenerationConfig(max_tokens=5, temperature=0.0)

        responses = gen.generate_batch(prompts, config)

        assert len(responses) == 3
        for response in responses:
            assert isinstance(response, str)
            assert len(response) > 0

    def test_generate_batch_different_lengths(self, model_path):
        """Test batch generation with prompts of different lengths."""
        gen = BatchGenerator(
            model_path=model_path,
            n_seq_max=2,
            verbose=False
        )

        prompts = [
            "Hi",
            "What is the meaning of life?",
        ]
        config = GenerationConfig(max_tokens=10, temperature=0.0)

        responses = gen.generate_batch(prompts, config)

        assert len(responses) == 2
        for response in responses:
            assert isinstance(response, str)

    def test_generate_batch_detailed(self, model_path):
        """Test detailed batch generation with statistics."""
        gen = BatchGenerator(
            model_path=model_path,
            n_seq_max=2,
            verbose=False
        )

        requests = [
            BatchRequest(id=0, prompt="What is 1+1?", max_tokens=10),
            BatchRequest(id=1, prompt="What is 2+2?", max_tokens=10),
        ]

        config = GenerationConfig(temperature=0.0)

        results = gen.generate_batch_detailed(requests, config)

        assert len(results) == 2
        for result in results:
            assert isinstance(result, BatchResponse)
            assert isinstance(result.id, int)
            assert isinstance(result.prompt, str)
            assert isinstance(result.response, str)
            assert isinstance(result.tokens_generated, int)
            assert isinstance(result.time_taken, float)
            assert result.tokens_generated > 0
            assert result.time_taken > 0

    def test_temperature_zero_deterministic(self, model_path):
        """Test that temperature=0 gives deterministic results."""
        prompt = ["What is 2+2?"]
        config = GenerationConfig(max_tokens=10, temperature=0.0)

        # Generate twice with fresh generators
        response1 = batch_generate(
            prompt,
            model_path=model_path,
            n_seq_max=1,
            config=config
        )[0]

        response2 = batch_generate(
            prompt,
            model_path=model_path,
            n_seq_max=1,
            config=config
        )[0]

        # Should be identical
        assert response1 == response2

    def test_parallel_sequences(self, model_path):
        """Test that parallel sequences work correctly."""
        gen = BatchGenerator(
            model_path=model_path,
            n_seq_max=4,
            verbose=False
        )

        prompts = ["A", "B", "C", "D"]
        config = GenerationConfig(max_tokens=3, temperature=0.0)

        responses = gen.generate_batch(prompts, config)

        # All sequences should complete
        assert len(responses) == 4

        # Each should have content
        for response in responses:
            assert len(response) > 0


class TestBatchConfiguration:
    """Test various configuration options for batch processing."""

    def test_custom_config_parameters(self, model_path):
        """Test batch generation with custom config parameters."""
        prompts = ["Tell me a story"]

        responses = batch_generate(
            prompts,
            model_path=model_path,
            n_seq_max=1,
            config=GenerationConfig(
                max_tokens=15,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                min_p=0.05
            )
        )

        assert len(responses) == 1
        assert len(responses[0]) > 0

    def test_batch_size_parameter(self, model_path):
        """Test that batch_size parameter is respected."""
        gen = BatchGenerator(
            model_path=model_path,
            batch_size=256,  # Smaller than default
            n_seq_max=2,
            verbose=False
        )

        prompts = ["Hello", "Hi"]
        config = GenerationConfig(max_tokens=5, temperature=0.0)

        responses = gen.generate_batch(prompts, config)

        assert len(responses) == 2

    def test_context_size_parameter(self, model_path):
        """Test that n_ctx parameter is respected."""
        gen = BatchGenerator(
            model_path=model_path,
            n_ctx=1024,  # Smaller than default
            n_seq_max=1,
            verbose=False
        )

        prompts = ["Test"]
        config = GenerationConfig(max_tokens=5, temperature=0.0)

        responses = gen.generate_batch(prompts, config)

        assert len(responses) == 1


class TestBatchPooling:
    """Test batch memory pooling functionality."""

    def test_pooling_api_available(self):
        """Test that batch pooling API is available from llama_cpp module."""
        from cyllama.llama.llama_cpp import (
            get_pooled_batch,
            return_batch_to_pool,
            get_batch_pool_stats,
            reset_batch_pool,
        )

        # Verify functions are callable
        assert callable(get_pooled_batch)
        assert callable(return_batch_to_pool)
        assert callable(get_batch_pool_stats)
        assert callable(reset_batch_pool)

    def test_batch_pooling_basic(self):
        """Test basic batch pooling get/return cycle."""
        from cyllama.llama.llama_cpp import (
            get_pooled_batch,
            return_batch_to_pool,
            get_batch_pool_stats,
            reset_batch_pool,
        )

        # Reset pool to known state
        reset_batch_pool()
        stats = get_batch_pool_stats()
        assert stats["total_pools"] == 0
        assert stats["total_pooled_batches"] == 0

        # Get a batch from pool (creates new one)
        batch = get_pooled_batch(n_tokens=256, embd=0, n_seq_max=4)
        assert batch is not None

        # Return batch to pool
        return_batch_to_pool(batch)
        stats = get_batch_pool_stats()
        assert stats["total_pools"] == 1
        assert stats["total_pooled_batches"] == 1

        # Get batch again (should reuse from pool)
        batch2 = get_pooled_batch(n_tokens=256, embd=0, n_seq_max=4)
        stats = get_batch_pool_stats()
        # Pool now empty since we took the batch
        assert stats["total_pooled_batches"] == 0

        # Clean up
        return_batch_to_pool(batch2)
        reset_batch_pool()

    def test_batch_pooling_multiple_batches(self):
        """Test pooling with multiple batches of different sizes."""
        from cyllama.llama.llama_cpp import (
            get_pooled_batch,
            return_batch_to_pool,
            get_batch_pool_stats,
            reset_batch_pool,
        )

        reset_batch_pool()

        # Get batches of different sizes
        batch1 = get_pooled_batch(n_tokens=128, embd=0, n_seq_max=2)
        batch2 = get_pooled_batch(n_tokens=256, embd=0, n_seq_max=4)
        batch3 = get_pooled_batch(n_tokens=512, embd=0, n_seq_max=8)

        # Batches are in use, pool is empty
        stats = get_batch_pool_stats()
        assert stats["total_pooled_batches"] == 0

        # Return all
        return_batch_to_pool(batch1)
        return_batch_to_pool(batch2)
        return_batch_to_pool(batch3)

        stats = get_batch_pool_stats()
        assert stats["total_pools"] == 3  # 3 different configurations
        assert stats["total_pooled_batches"] == 3  # 3 batches available

        # Clean up
        reset_batch_pool()

    def test_batch_generator_with_pooling(self, model_path):
        """Test BatchGenerator with use_pooling=True."""
        gen = BatchGenerator(
            model_path=model_path,
            n_seq_max=2,
            verbose=False,
            use_pooling=True
        )

        assert gen.use_pooling is True

        prompts = ["Hello", "Hi"]
        config = GenerationConfig(max_tokens=5, temperature=0.0)

        responses = gen.generate_batch(prompts, config)

        assert len(responses) == 2
        for response in responses:
            assert isinstance(response, str)
            assert len(response) > 0

    def test_batch_generator_pooling_consistency(self, model_path):
        """Test that pooling produces same results as non-pooling."""
        prompts = ["What is 2+2?"]
        config = GenerationConfig(max_tokens=10, temperature=0.0)

        # Without pooling
        gen_no_pool = BatchGenerator(
            model_path=model_path,
            n_seq_max=1,
            verbose=False,
            use_pooling=False
        )
        response_no_pool = gen_no_pool.generate_batch(prompts, config)

        # With pooling
        gen_pool = BatchGenerator(
            model_path=model_path,
            n_seq_max=1,
            verbose=False,
            use_pooling=True
        )
        response_pool = gen_pool.generate_batch(prompts, config)

        # Results should be identical
        assert response_no_pool == response_pool

    def test_batch_generator_multiple_generations_with_pooling(self, model_path):
        """Test that pooling works correctly across multiple generations."""
        from cyllama.llama.llama_cpp import get_batch_pool_stats, reset_batch_pool

        reset_batch_pool()

        # Test that BatchGenerator with pooling can generate multiple times
        # Each generation should work correctly even when batches are reused
        gen = BatchGenerator(
            model_path=model_path,
            n_seq_max=2,
            verbose=False,
            use_pooling=True
        )

        config = GenerationConfig(max_tokens=5, temperature=0.0)

        # First generation
        prompts1 = ["Hello", "Hi"]
        responses1 = gen.generate_batch(prompts1, config)
        assert len(responses1) == 2

        # After first generation, batch should be in pool
        stats = get_batch_pool_stats()
        assert stats["total_pooled_batches"] >= 1

        # Create a new generator to use pooled batches
        gen2 = BatchGenerator(
            model_path=model_path,
            n_seq_max=2,
            verbose=False,
            use_pooling=True
        )

        # Second generation with same config should reuse batch
        prompts2 = ["Hey", "Howdy"]
        responses2 = gen2.generate_batch(prompts2, config)
        assert len(responses2) == 2

        reset_batch_pool()
