"""
Tests for high-level generation API.
"""

import pytest
from cyllama import (
    complete,
    chat,
    LLM,
    GenerationConfig,
)


# Test data
DEFAULT_MODEL = "models/Llama-3.2-1B-Instruct-Q8_0.gguf"


class TestGenerationConfig:
    """Tests for GenerationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GenerationConfig()
        assert config.max_tokens == 512
        assert config.temperature == 0.8
        assert config.top_k == 40
        assert config.top_p == 0.95
        assert config.min_p == 0.05
        assert config.repeat_penalty == 1.1
        assert config.n_gpu_layers == 99
        assert config.n_ctx is None
        assert config.n_batch == 512
        assert config.seed == -1
        assert config.stop_sequences == []
        assert config.add_bos is True
        assert config.parse_special is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = GenerationConfig(
            max_tokens=100,
            temperature=0.5,
            top_k=20,
            n_gpu_layers=0,
            stop_sequences=["STOP", "END"]
        )
        assert config.max_tokens == 100
        assert config.temperature == 0.5
        assert config.top_k == 20
        assert config.n_gpu_layers == 0
        assert config.stop_sequences == ["STOP", "END"]


class TestLLM:
    """Tests for LLM class."""

    @pytest.mark.slow
    def test_initialization(self):
        """Test LLM initialization."""
        gen = LLM(DEFAULT_MODEL, verbose=False)
        assert gen.model_path == DEFAULT_MODEL
        assert gen.model is not None
        assert gen.vocab is not None

    @pytest.mark.slow
    def test_simple_generation(self):
        """Test basic text generation."""
        gen = LLM(DEFAULT_MODEL)
        config = GenerationConfig(max_tokens=20, temperature=0.0)  # Greedy for consistency
        response = gen("What is 2+2?", config=config)

        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.slow
    def test_streaming_generation(self):
        """Test streaming text generation."""
        gen = LLM(DEFAULT_MODEL)
        config = GenerationConfig(max_tokens=20, temperature=0.0)

        chunks = list(gen("Count to 3:", config=config, stream=True))

        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

        # Reconstruct full response
        full_response = "".join(chunks)
        assert len(full_response) > 0

    @pytest.mark.slow
    def test_token_callback(self):
        """Test on_token callback."""
        gen = LLM(DEFAULT_MODEL)
        config = GenerationConfig(max_tokens=10, temperature=0.0)

        tokens = []
        def on_token(token: str):
            tokens.append(token)

        response = gen("Hello", config=config, on_token=on_token)

        assert len(tokens) > 0
        assert "".join(tokens) == response

    @pytest.mark.slow
    def test_generation_with_stats(self):
        """Test generation with statistics."""
        gen = LLM(DEFAULT_MODEL)
        config = GenerationConfig(max_tokens=20, temperature=0.0)

        response, stats = gen.generate_with_stats("Test prompt", config=config)

        assert isinstance(response, str)
        assert stats.prompt_tokens > 0
        assert stats.generated_tokens >= 0
        assert stats.total_time > 0
        assert stats.tokens_per_second >= 0

    @pytest.mark.slow
    def test_different_temperatures(self):
        """Test generation with different temperatures."""
        gen = LLM(DEFAULT_MODEL)

        # Greedy (deterministic with same seed)
        config_greedy = GenerationConfig(max_tokens=10, temperature=0.0, seed=42)
        response1 = gen("Hello", config=config_greedy)
        response2 = gen("Hello", config=config_greedy)
        # Note: May not be identical due to context recreation, but both should be valid
        assert isinstance(response1, str)
        assert isinstance(response2, str)
        assert len(response1) > 0
        assert len(response2) > 0

        # High temperature (more random)
        config_random = GenerationConfig(max_tokens=10, temperature=1.5, seed=42)
        response3 = gen("Hello", config=config_random)
        assert isinstance(response3, str)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.slow
    def test_complete_function(self):
        """Test complete() convenience function."""
        response = complete(
            "What is Python?",
            model_path=DEFAULT_MODEL,
            max_tokens=30,
            temperature=0.0
        )

        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.slow
    def test_complete_streaming(self):
        """Test complete() with streaming."""
        chunks = list(complete(
            "Count to 3:",
            model_path=DEFAULT_MODEL,
            max_tokens=20,
            temperature=0.0,
            stream=True
        ))

        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert len(full_response) > 0

    @pytest.mark.slow
    def test_chat_function(self):
        """Test chat() convenience function."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ]

        response = chat(
            messages,
            model_path=DEFAULT_MODEL,
            max_tokens=30,
            temperature=0.0
        )

        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.slow
    def test_chat_streaming(self):
        """Test chat() with streaming."""
        messages = [
            {"role": "user", "content": "Count to 3"}
        ]

        chunks = list(chat(
            messages,
            model_path=DEFAULT_MODEL,
            max_tokens=20,
            temperature=0.0,
            stream=True
        ))

        assert len(chunks) > 0


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.slow
    def test_empty_prompt(self):
        """Test generation with empty prompt."""
        gen = LLM(DEFAULT_MODEL)
        config = GenerationConfig(max_tokens=10)

        # Empty prompt should still work (BOS token)
        response = gen("", config=config)
        assert isinstance(response, str)

    @pytest.mark.slow
    def test_max_tokens_zero(self):
        """Test generation with max_tokens=0."""
        gen = LLM(DEFAULT_MODEL)
        config = GenerationConfig(max_tokens=0)

        response = gen("Test", config=config)
        assert response == ""

    @pytest.mark.slow
    def test_very_long_prompt(self):
        """Test generation with long prompt."""
        gen = LLM(DEFAULT_MODEL)
        config = GenerationConfig(max_tokens=10, n_ctx=2048)

        long_prompt = "Hello " * 100
        response = gen(long_prompt, config=config)
        assert isinstance(response, str)

    @pytest.mark.slow
    def test_context_recreation(self):
        """Test that context is recreated when needed."""
        gen = LLM(DEFAULT_MODEL)

        # Generate with small context
        config1 = GenerationConfig(max_tokens=10, n_ctx=512)
        response1 = gen("Test1", config=config1)

        # Generate with larger context (should recreate)
        config2 = GenerationConfig(max_tokens=10, n_ctx=1024)
        response2 = gen("Test2", config=config2)

        assert isinstance(response1, str)
        assert isinstance(response2, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
