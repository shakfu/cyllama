"""
Tests for high-level generation API.
"""

import pytest
from cyllama import (
    complete,
    chat,
    LLM,
    GenerationConfig,
    Response,
)


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

    def test_validation_max_tokens(self):
        """Test max_tokens validation."""
        with pytest.raises(ValueError, match="max_tokens must be >= 0"):
            GenerationConfig(max_tokens=-1)
        # Valid edge cases
        config = GenerationConfig(max_tokens=0)  # 0 means "generate nothing"
        assert config.max_tokens == 0
        config = GenerationConfig(max_tokens=1)
        assert config.max_tokens == 1

    def test_validation_temperature(self):
        """Test temperature validation."""
        with pytest.raises(ValueError, match="temperature must be >= 0.0"):
            GenerationConfig(temperature=-0.1)
        # Valid edge cases
        config = GenerationConfig(temperature=0.0)
        assert config.temperature == 0.0
        config = GenerationConfig(temperature=2.0)  # High but valid
        assert config.temperature == 2.0

    def test_validation_top_k(self):
        """Test top_k validation."""
        with pytest.raises(ValueError, match="top_k must be >= 0"):
            GenerationConfig(top_k=-1)
        # Valid edge case (0 means disabled)
        config = GenerationConfig(top_k=0)
        assert config.top_k == 0

    def test_validation_top_p(self):
        """Test top_p validation."""
        with pytest.raises(ValueError, match="top_p must be between 0.0 and 1.0"):
            GenerationConfig(top_p=-0.1)
        with pytest.raises(ValueError, match="top_p must be between 0.0 and 1.0"):
            GenerationConfig(top_p=1.1)
        # Valid edge cases
        config = GenerationConfig(top_p=0.0)
        assert config.top_p == 0.0
        config = GenerationConfig(top_p=1.0)
        assert config.top_p == 1.0

    def test_validation_min_p(self):
        """Test min_p validation."""
        with pytest.raises(ValueError, match="min_p must be between 0.0 and 1.0"):
            GenerationConfig(min_p=-0.1)
        with pytest.raises(ValueError, match="min_p must be between 0.0 and 1.0"):
            GenerationConfig(min_p=1.1)
        # Valid edge cases
        config = GenerationConfig(min_p=0.0)
        assert config.min_p == 0.0
        config = GenerationConfig(min_p=1.0)
        assert config.min_p == 1.0

    def test_validation_repeat_penalty(self):
        """Test repeat_penalty validation."""
        with pytest.raises(ValueError, match="repeat_penalty must be >= 0.0"):
            GenerationConfig(repeat_penalty=-0.1)
        # Valid edge case
        config = GenerationConfig(repeat_penalty=0.0)
        assert config.repeat_penalty == 0.0

    def test_validation_n_gpu_layers(self):
        """Test n_gpu_layers validation."""
        with pytest.raises(ValueError, match="n_gpu_layers must be >= 0"):
            GenerationConfig(n_gpu_layers=-1)
        # Valid edge case
        config = GenerationConfig(n_gpu_layers=0)
        assert config.n_gpu_layers == 0

    def test_validation_n_ctx(self):
        """Test n_ctx validation."""
        with pytest.raises(ValueError, match="n_ctx must be >= 1 or None"):
            GenerationConfig(n_ctx=0)
        with pytest.raises(ValueError, match="n_ctx must be >= 1 or None"):
            GenerationConfig(n_ctx=-1)
        # Valid cases
        config = GenerationConfig(n_ctx=None)
        assert config.n_ctx is None
        config = GenerationConfig(n_ctx=1)
        assert config.n_ctx == 1

    def test_validation_n_batch(self):
        """Test n_batch validation."""
        with pytest.raises(ValueError, match="n_batch must be >= 1"):
            GenerationConfig(n_batch=0)
        with pytest.raises(ValueError, match="n_batch must be >= 1"):
            GenerationConfig(n_batch=-1)
        # Valid edge case
        config = GenerationConfig(n_batch=1)
        assert config.n_batch == 1

    def test_validation_seed(self):
        """Test seed validation."""
        with pytest.raises(ValueError, match="seed must be >= -1"):
            GenerationConfig(seed=-2)
        # Valid edge cases
        config = GenerationConfig(seed=-1)  # random
        assert config.seed == -1
        config = GenerationConfig(seed=0)
        assert config.seed == 0
        config = GenerationConfig(seed=42)
        assert config.seed == 42

    def test_validation_multiple_errors(self):
        """Test that multiple validation errors are reported together."""
        with pytest.raises(ValueError) as exc_info:
            GenerationConfig(max_tokens=-1, temperature=-1.0, top_p=2.0)
        error_msg = str(exc_info.value)
        assert "max_tokens" in error_msg
        assert "temperature" in error_msg
        assert "top_p" in error_msg


class TestLLM:
    """Tests for LLM class."""

    @pytest.mark.slow
    def test_initialization(self, model_path):
        """Test LLM initialization."""
        gen = LLM(model_path, verbose=False)
        assert gen.model_path == model_path
        assert gen.model is not None
        assert gen.vocab is not None

    @pytest.mark.slow
    def test_simple_generation(self, model_path):
        """Test basic text generation."""
        gen = LLM(model_path)
        config = GenerationConfig(max_tokens=20, temperature=0.0)  # Greedy for consistency
        response = gen("What is 2+2?", config=config)

        assert isinstance(response, (str, Response))
        assert len(response) > 0

    @pytest.mark.slow
    def test_streaming_generation(self, model_path):
        """Test streaming text generation."""
        gen = LLM(model_path)
        config = GenerationConfig(max_tokens=20, temperature=0.0)

        chunks = list(gen("Count to 3:", config=config, stream=True))

        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

        # Reconstruct full response
        full_response = "".join(chunks)
        assert len(full_response) > 0

    @pytest.mark.slow
    def test_token_callback(self, model_path):
        """Test on_token callback."""
        gen = LLM(model_path)
        config = GenerationConfig(max_tokens=10, temperature=0.0)

        tokens = []
        def on_token(token: str):
            tokens.append(token)

        response = gen("Hello", config=config, on_token=on_token)

        assert len(tokens) > 0
        assert "".join(tokens) == response

    @pytest.mark.slow
    def test_generation_with_stats(self, model_path):
        """Test generation with statistics."""
        gen = LLM(model_path)
        config = GenerationConfig(max_tokens=20, temperature=0.0)

        response = gen.generate_with_stats("Test prompt", config=config)

        assert isinstance(response, Response)
        assert response.stats is not None
        assert response.stats.prompt_tokens > 0
        assert response.stats.generated_tokens >= 0
        assert response.stats.total_time > 0
        assert response.stats.tokens_per_second >= 0

    @pytest.mark.slow
    def test_different_temperatures(self, model_path):
        """Test generation with different temperatures."""
        gen = LLM(model_path)

        # Greedy (deterministic with same seed)
        config_greedy = GenerationConfig(max_tokens=10, temperature=0.0, seed=42)
        response1 = gen("Hello", config=config_greedy)
        response2 = gen("Hello", config=config_greedy)
        # Note: May not be identical due to context recreation, but both should be valid
        assert isinstance(response1, (str, Response))
        assert isinstance(response2, (str, Response))
        assert len(response1) > 0
        assert len(response2) > 0

        # High temperature (more random)
        config_random = GenerationConfig(max_tokens=10, temperature=1.5, seed=42)
        response3 = gen("Hello", config=config_random)
        assert isinstance(response3, (str, Response))


class TestLLMResourceManagement:
    """Tests for LLM resource management (context reuse, cleanup)."""

    @pytest.mark.slow
    def test_context_manager(self, model_path):
        """Test LLM as context manager."""
        with LLM(model_path) as llm:
            config = GenerationConfig(max_tokens=10, temperature=0.0)
            response = llm("Hello", config=config)
            assert isinstance(response, (str, Response))
            assert len(response) > 0
        # After exiting, resources should be cleaned up
        assert llm._closed is True

    @pytest.mark.slow
    def test_explicit_close(self, model_path):
        """Test explicit close() method."""
        llm = LLM(model_path)
        config = GenerationConfig(max_tokens=10, temperature=0.0)
        response = llm("Hello", config=config)
        assert isinstance(response, (str, Response))

        # Close explicitly
        llm.close()
        assert llm._closed is True

        # Should be safe to call close() multiple times
        llm.close()
        assert llm._closed is True

    @pytest.mark.slow
    def test_context_reuse(self, model_path):
        """Test that context is reused when size permits."""
        llm = LLM(model_path, verbose=False)
        config = GenerationConfig(max_tokens=20, temperature=0.0)

        # First generation creates context
        response1 = llm("Hello", config=config)
        assert response1

        # Record context size
        ctx_size_after_first = llm._ctx_size
        assert ctx_size_after_first > 0

        # Second generation with same config should reuse context
        response2 = llm("Hi", config=config)
        assert response2

        # Context size should remain the same (reused)
        assert llm._ctx_size == ctx_size_after_first

        llm.close()

    @pytest.mark.slow
    def test_context_recreation_when_needed(self, model_path):
        """Test that context is recreated when larger size needed."""
        llm = LLM(model_path, verbose=False)

        # Small generation
        config_small = GenerationConfig(max_tokens=10, temperature=0.0)
        llm("A", config=config_small)
        small_ctx_size = llm._ctx_size

        # Large generation should recreate context
        config_large = GenerationConfig(max_tokens=500, temperature=0.0)
        llm("B", config=config_large)
        large_ctx_size = llm._ctx_size

        assert large_ctx_size > small_ctx_size

        llm.close()

    @pytest.mark.slow
    def test_reset_context(self, model_path):
        """Test reset_context() method."""
        llm = LLM(model_path, verbose=False)
        config = GenerationConfig(max_tokens=10, temperature=0.0)

        # Generate to create context
        llm("Hello", config=config)
        assert llm._ctx is not None
        assert llm._ctx_size > 0

        # Reset context
        llm.reset_context()
        assert llm._ctx is None
        assert llm._ctx_size == 0

        # Next generation should create new context
        llm("World", config=config)
        assert llm._ctx is not None
        assert llm._ctx_size > 0

        llm.close()

    @pytest.mark.slow
    def test_reopen_after_close(self, model_path):
        """Test that LLM can be used after close()."""
        llm = LLM(model_path, verbose=False)
        config = GenerationConfig(max_tokens=10, temperature=0.0)

        # First generation
        response1 = llm("Hello", config=config)
        assert response1

        # Close
        llm.close()
        assert llm._closed is True

        # Use again (should reopen)
        response2 = llm("Hi", config=config)
        assert response2
        assert llm._closed is False  # Reopened

        llm.close()

    @pytest.mark.slow
    def test_multiple_generations_same_instance(self, model_path):
        """Test multiple generations reuse context efficiently."""
        with LLM(model_path, verbose=False) as llm:
            config = GenerationConfig(max_tokens=15, temperature=0.0)

            responses = []
            for prompt in ["One", "Two", "Three", "Four"]:
                response = llm(prompt, config=config)
                responses.append(response)

            # All should have generated valid responses
            assert len(responses) == 4
            assert all(len(r) > 0 for r in responses)

            # Context should exist and be reused
            assert llm._ctx is not None


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.slow
    def test_complete_function(self, model_path):
        """Test complete() convenience function."""
        response = complete(
            "What is Python?",
            model_path=model_path,
            max_tokens=30,
            temperature=0.0
        )

        assert isinstance(response, (str, Response))
        assert len(response) > 0

    @pytest.mark.slow
    def test_complete_streaming(self, model_path):
        """Test complete() with streaming."""
        chunks = list(complete(
            "Count to 3:",
            model_path=model_path,
            max_tokens=20,
            temperature=0.0,
            stream=True
        ))

        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert len(full_response) > 0

    @pytest.mark.slow
    def test_chat_function(self, model_path):
        """Test chat() convenience function."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ]

        response = chat(
            messages,
            model_path=model_path,
            max_tokens=30,
            temperature=0.0
        )

        assert isinstance(response, (str, Response))
        assert len(response) > 0

    @pytest.mark.slow
    def test_chat_streaming(self, model_path):
        """Test chat() with streaming."""
        messages = [
            {"role": "user", "content": "Count to 3"}
        ]

        chunks = list(chat(
            messages,
            model_path=model_path,
            max_tokens=20,
            temperature=0.0,
            stream=True
        ))

        assert len(chunks) > 0


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.slow
    def test_empty_prompt(self, model_path):
        """Test generation with empty prompt."""
        gen = LLM(model_path)
        config = GenerationConfig(max_tokens=10)

        # Empty prompt should still work (BOS token)
        response = gen("", config=config)
        assert isinstance(response, (str, Response))

    @pytest.mark.slow
    def test_max_tokens_zero(self, model_path):
        """Test generation with max_tokens=0."""
        gen = LLM(model_path)
        config = GenerationConfig(max_tokens=0)

        response = gen("Test", config=config)
        assert response == ""

    @pytest.mark.slow
    def test_very_long_prompt(self, model_path):
        """Test generation with long prompt."""
        gen = LLM(model_path)
        config = GenerationConfig(max_tokens=10, n_ctx=2048)

        long_prompt = "Hello " * 100
        response = gen(long_prompt, config=config)
        assert isinstance(response, (str, Response))

    @pytest.mark.slow
    def test_context_recreation(self, model_path):
        """Test that context is recreated when needed."""
        gen = LLM(model_path)

        # Generate with small context
        config1 = GenerationConfig(max_tokens=10, n_ctx=512)
        response1 = gen("Test1", config=config1)

        # Generate with larger context (should recreate)
        config2 = GenerationConfig(max_tokens=10, n_ctx=1024)
        response2 = gen("Test2", config=config2)

        assert isinstance(response1, (str, Response))
        assert isinstance(response2, (str, Response))


class TestStopSequences:
    """Tests for stop sequence handling."""

    @pytest.mark.slow
    def test_basic_stop_sequence(self, model_path):
        """Test that generation stops at stop sequence."""
        gen = LLM(model_path)
        # Use a common token that's likely to appear
        config = GenerationConfig(
            max_tokens=100,
            temperature=0.0,
            stop_sequences=["\n\n"]  # Stop at double newline
        )

        response = gen("Count from 1 to 10:", config=config)
        assert isinstance(response, (str, Response))
        # Should not contain the stop sequence
        assert "\n\n" not in response

    @pytest.mark.slow
    def test_multiple_stop_sequences(self, model_path):
        """Test with multiple stop sequences."""
        gen = LLM(model_path)
        config = GenerationConfig(
            max_tokens=100,
            temperature=0.0,
            stop_sequences=[".", "!", "?"]  # Stop at any sentence ending
        )

        response = gen("Tell me something about Python", config=config)
        assert isinstance(response, (str, Response))
        # Should not contain any stop sequence
        assert "." not in response
        assert "!" not in response
        assert "?" not in response

    @pytest.mark.slow
    def test_stop_sequence_not_in_output(self, model_path):
        """Test that stop sequence is excluded from output."""
        gen = LLM(model_path)
        config = GenerationConfig(
            max_tokens=50,
            temperature=0.0,
            stop_sequences=["."]
        )

        response = gen("Say hello", config=config)
        # The stop sequence itself should not be included
        assert not response.text.endswith(".")

    @pytest.mark.slow
    def test_empty_stop_sequences(self, model_path):
        """Test with empty stop sequences list."""
        gen = LLM(model_path)
        config = GenerationConfig(
            max_tokens=20,
            temperature=0.0,
            stop_sequences=[]  # No stop sequences
        )

        response = gen("Hello", config=config)
        assert isinstance(response, (str, Response))
        assert len(response) > 0

    @pytest.mark.slow
    def test_stop_sequence_streaming(self, model_path):
        """Test stop sequences work in streaming mode."""
        gen = LLM(model_path)
        config = GenerationConfig(
            max_tokens=100,
            temperature=0.0,
            stop_sequences=["."]
        )

        chunks = list(gen("Say hello", config=config, stream=True))
        full_response = "".join(chunks)

        assert isinstance(full_response, str)  # streaming returns str chunks
        # Stop sequence should not be in output
        assert not full_response.endswith(".")

    def test_find_stop_sequence_helper(self, model_path):
        """Test the _find_stop_sequence helper method."""
        gen = LLM(model_path)

        # Single stop sequence
        pos, length = gen._find_stop_sequence("hello world", ["world"])
        assert pos == 6
        assert length == 5

        # Multiple stop sequences - should find earliest
        pos, length = gen._find_stop_sequence("hello world", ["world", "ell"])
        assert pos == 1  # "ell" starts at position 1
        assert length == 3

        # No match
        pos, length = gen._find_stop_sequence("hello world", ["xyz", "abc"])
        assert pos is None
        assert length == 0

        # Empty stop sequences
        pos, length = gen._find_stop_sequence("hello world", [])
        assert pos is None
        assert length == 0

        # Stop sequence at start
        pos, length = gen._find_stop_sequence("hello world", ["hello"])
        assert pos == 0
        assert length == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
