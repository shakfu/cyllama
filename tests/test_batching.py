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
DEFAULT_MODEL = "models/Llama-3.2-1B-Instruct-Q8_0.gguf"
pytestmark = pytest.mark.skipif(
    not pytest.importorskip("pathlib").Path(DEFAULT_MODEL).exists(),
    reason=f"Model not found at {DEFAULT_MODEL}"
)


class TestBatchGenerate:
    """Test the convenience batch_generate function."""

    def test_basic_batch_generate(self):
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
            model_path=DEFAULT_MODEL,
            n_seq_max=2,
            config=config
        )

        assert len(responses) == len(prompts)
        for response in responses:
            assert isinstance(response, str)
            assert len(response) > 0

    def test_single_prompt(self):
        """Test batch generation with a single prompt."""
        prompts = ["Hello, how are you?"]

        config = GenerationConfig(max_tokens=10, temperature=0.0)

        responses = batch_generate(
            prompts,
            model_path=DEFAULT_MODEL,
            n_seq_max=1,
            config=config
        )

        assert len(responses) == 1
        assert isinstance(responses[0], str)
        assert len(responses[0]) > 0

    def test_empty_prompts(self):
        """Test batch generation with empty prompt list."""
        prompts = []

        responses = batch_generate(
            prompts,
            model_path=DEFAULT_MODEL,
            n_seq_max=1
        )

        assert len(responses) == 0

    def test_max_sequences_exceeded(self):
        """Test that error is raised when too many prompts for n_seq_max."""
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]

        config = GenerationConfig(max_tokens=5, temperature=0.0)

        with pytest.raises(ValueError, match="Too many prompts"):
            batch_generate(
                prompts,
                model_path=DEFAULT_MODEL,
                n_seq_max=2,  # Only 2 sequences allowed
                config=config
            )


class TestBatchGenerator:
    """Test the BatchGenerator class."""

    def test_initialization(self):
        """Test BatchGenerator initialization."""
        gen = BatchGenerator(
            model_path=DEFAULT_MODEL,
            batch_size=512,
            n_ctx=2048,
            n_seq_max=4,
            verbose=False
        )

        assert gen.model is not None
        assert gen.ctx is not None
        assert gen.vocab is not None
        assert gen.n_seq_max == 4

    def test_generate_batch_basic(self):
        """Test basic batch generation."""
        gen = BatchGenerator(
            model_path=DEFAULT_MODEL,
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

    def test_generate_batch_different_lengths(self):
        """Test batch generation with prompts of different lengths."""
        gen = BatchGenerator(
            model_path=DEFAULT_MODEL,
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

    def test_generate_batch_detailed(self):
        """Test detailed batch generation with statistics."""
        gen = BatchGenerator(
            model_path=DEFAULT_MODEL,
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

    def test_temperature_zero_deterministic(self):
        """Test that temperature=0 gives deterministic results."""
        prompt = ["What is 2+2?"]
        config = GenerationConfig(max_tokens=10, temperature=0.0)

        # Generate twice with fresh generators
        response1 = batch_generate(
            prompt,
            model_path=DEFAULT_MODEL,
            n_seq_max=1,
            config=config
        )[0]

        response2 = batch_generate(
            prompt,
            model_path=DEFAULT_MODEL,
            n_seq_max=1,
            config=config
        )[0]

        # Should be identical
        assert response1 == response2

    def test_parallel_sequences(self):
        """Test that parallel sequences work correctly."""
        gen = BatchGenerator(
            model_path=DEFAULT_MODEL,
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

    def test_custom_config_parameters(self):
        """Test batch generation with custom config parameters."""
        prompts = ["Tell me a story"]

        responses = batch_generate(
            prompts,
            model_path=DEFAULT_MODEL,
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

    def test_batch_size_parameter(self):
        """Test that batch_size parameter is respected."""
        gen = BatchGenerator(
            model_path=DEFAULT_MODEL,
            batch_size=256,  # Smaller than default
            n_seq_max=2,
            verbose=False
        )

        prompts = ["Hello", "Hi"]
        config = GenerationConfig(max_tokens=5, temperature=0.0)

        responses = gen.generate_batch(prompts, config)

        assert len(responses) == 2

    def test_context_size_parameter(self):
        """Test that n_ctx parameter is respected."""
        gen = BatchGenerator(
            model_path=DEFAULT_MODEL,
            n_ctx=1024,  # Smaller than default
            n_seq_max=1,
            verbose=False
        )

        prompts = ["Test"]
        config = GenerationConfig(max_tokens=5, temperature=0.0)

        responses = gen.generate_batch(prompts, config)

        assert len(responses) == 1
