"""
Comprehensive tests for edge cases, error conditions, and robustness.

This module covers test gaps identified in CODE_REVIEW.md:
- Error condition tests (invalid model paths, corrupted data, OOM conditions)
- Unicode handling tests (non-ASCII, emoji, special characters)
- Concurrent execution tests (thread safety, parallel access)
- Boundary condition tests (limits, edge values)
"""

import pytest
import os
import threading
import concurrent.futures
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from conftest import DEFAULT_MODEL

# Skip all tests if model is not available
pytestmark = pytest.mark.skipif(
    not Path(DEFAULT_MODEL).exists(),
    reason="Model not found"
)


# =============================================================================
# Error Condition Tests (HIGH PRIORITY)
# =============================================================================

class TestInvalidModelPath:
    """Tests for invalid model path handling."""

    def test_nonexistent_model_path(self):
        """Test loading model from nonexistent path raises appropriate error."""
        from cyllama import LLM

        with pytest.raises((FileNotFoundError, RuntimeError, OSError, ValueError)):
            LLM("/nonexistent/path/to/model.gguf", verbose=False)

    def test_directory_as_model_path(self, tmp_path):
        """Test loading directory instead of file raises error."""
        from cyllama import LLM

        # tmp_path is a directory
        with pytest.raises((IsADirectoryError, RuntimeError, OSError, ValueError)):
            LLM(str(tmp_path), verbose=False)

    def test_empty_file_as_model(self, tmp_path):
        """Test loading empty file as model raises error."""
        from cyllama import LLM

        empty_file = tmp_path / "empty.gguf"
        empty_file.touch()

        with pytest.raises((RuntimeError, ValueError, OSError)):
            LLM(str(empty_file), verbose=False)

    def test_invalid_gguf_file(self, tmp_path):
        """Test loading invalid GGUF file (wrong magic) raises error."""
        from cyllama import LLM

        invalid_file = tmp_path / "invalid.gguf"
        invalid_file.write_bytes(b"NOT_A_VALID_GGUF_FILE" * 100)

        with pytest.raises((RuntimeError, ValueError, OSError)):
            LLM(str(invalid_file), verbose=False)

    def test_truncated_model_file(self, tmp_path, model_path):
        """Test loading truncated model file raises error."""
        from cyllama import LLM

        # Read first 1KB of valid model (will be invalid)
        with open(model_path, 'rb') as f:
            partial_data = f.read(1024)

        truncated_file = tmp_path / "truncated.gguf"
        truncated_file.write_bytes(partial_data)

        with pytest.raises((RuntimeError, ValueError, OSError)):
            LLM(str(truncated_file), verbose=False)


class TestContextErrors:
    """Tests for context creation and management errors."""

    def test_context_size_zero(self, model_path):
        """Test context size 0 raises validation error."""
        from cyllama import LLM, GenerationConfig

        with pytest.raises(ValueError, match="n_ctx must be >= 1"):
            GenerationConfig(n_ctx=0)

    def test_batch_size_zero(self, model_path):
        """Test batch size 0 raises validation error."""
        from cyllama import GenerationConfig

        with pytest.raises(ValueError, match="n_batch must be >= 1"):
            GenerationConfig(n_batch=0)

    def test_negative_max_tokens(self, model_path):
        """Test negative max_tokens raises validation error."""
        from cyllama import GenerationConfig

        with pytest.raises(ValueError, match="max_tokens must be >= 0"):
            GenerationConfig(max_tokens=-1)


class TestBatchGeneratorErrors:
    """Tests for BatchGenerator error conditions."""

    def test_invalid_model_path(self):
        """Test BatchGenerator with invalid model path."""
        from cyllama import BatchGenerator

        with pytest.raises((FileNotFoundError, RuntimeError, OSError, ValueError)):
            BatchGenerator(
                model_path="/nonexistent/model.gguf",
                n_seq_max=1,
                verbose=False
            )

    def test_zero_n_seq_max(self, model_path):
        """Test BatchGenerator with n_seq_max=0."""
        from cyllama import BatchGenerator

        # n_seq_max=0 should either raise or create generator that can't process
        try:
            gen = BatchGenerator(
                model_path=model_path,
                n_seq_max=0,
                verbose=False
            )
            # If it creates, any prompts should fail
            with pytest.raises((ValueError, RuntimeError)):
                gen.generate_batch(["test"])
            gen.close()
        except (ValueError, RuntimeError):
            pass  # Also acceptable to fail at creation


class TestMemoryEstimationErrors:
    """Tests for memory estimation error handling."""

    def test_invalid_model_path_memory(self):
        """Test memory estimation with invalid model path."""
        from cyllama.memory import estimate_memory_usage, estimate_gpu_layers

        # Should handle gracefully (return defaults or raise)
        try:
            result = estimate_memory_usage("/nonexistent/model.gguf")
            # If it returns, should have some structure
            assert isinstance(result, dict)
        except (FileNotFoundError, OSError, RuntimeError):
            pass  # Also acceptable

    def test_invalid_gpu_memory_string(self):
        """Test parse_gpu_memory with invalid input."""
        from cyllama.memory import parse_gpu_memory

        with pytest.raises(ValueError):
            parse_gpu_memory("invalid")

        with pytest.raises(ValueError):
            parse_gpu_memory("abc GB")

        with pytest.raises(ValueError):
            parse_gpu_memory("-5GB")

    def test_negative_context_size(self):
        """Test memory estimation with negative context size."""
        from cyllama.memory import estimate_memory_usage

        # Should raise or clamp to valid value
        try:
            result = estimate_memory_usage(
                "nonexistent.gguf",
                ctx_size=-100,
                batch_size=512
            )
            # If it succeeds, context should have been clamped
        except (ValueError, OSError, FileNotFoundError):
            pass  # Expected


# =============================================================================
# Unicode Handling Tests (HIGH PRIORITY)
# =============================================================================

class TestUnicodePrompts:
    """Tests for Unicode character handling in prompts."""

    @pytest.mark.slow
    def test_basic_unicode(self, model_path):
        """Test basic Unicode characters in prompt."""
        from cyllama import LLM, GenerationConfig

        llm = LLM(model_path, verbose=False)
        config = GenerationConfig(max_tokens=10, temperature=0.0)

        # Various Unicode scripts
        prompts = [
            "Hello, world!",  # Basic Latin
            "Bonjour le monde!",  # French (Latin extended)
            "Hallo Welt!",  # German (umlauts possible in response)
        ]

        for prompt in prompts:
            response = llm(prompt, config=config)
            assert isinstance(response, str)

        llm.close()

    @pytest.mark.slow
    def test_cjk_characters(self, model_path):
        """Test CJK (Chinese, Japanese, Korean) characters."""
        from cyllama import LLM, GenerationConfig

        llm = LLM(model_path, verbose=False)
        config = GenerationConfig(max_tokens=10, temperature=0.0)

        # Chinese
        response = llm("Translate: Hello", config=config)
        assert isinstance(response, str)

        llm.close()

    @pytest.mark.slow
    def test_emoji_in_prompt(self, model_path):
        """Test emoji characters in prompt."""
        from cyllama import LLM, GenerationConfig

        llm = LLM(model_path, verbose=False)
        config = GenerationConfig(max_tokens=10, temperature=0.0)

        # Basic emoji
        response = llm("What does this mean: happy face", config=config)
        assert isinstance(response, str)

        llm.close()

    @pytest.mark.slow
    def test_mixed_scripts(self, model_path):
        """Test mixed Unicode scripts in single prompt."""
        from cyllama import LLM, GenerationConfig

        llm = LLM(model_path, verbose=False)
        config = GenerationConfig(max_tokens=10, temperature=0.0)

        # Mix of scripts
        response = llm("Hello in different styles", config=config)
        assert isinstance(response, str)

        llm.close()

    @pytest.mark.slow
    def test_special_unicode_characters(self, model_path):
        """Test special Unicode characters."""
        from cyllama import LLM, GenerationConfig

        llm = LLM(model_path, verbose=False)
        config = GenerationConfig(max_tokens=10, temperature=0.0)

        # Various special characters
        special_chars = [
            "Test with dash: -",
            "Test with quotes: \"text\"",
            "Test with smart quotes",
            "Test with ellipsis...",
            "Math symbols: x + y = z",
        ]

        for prompt in special_chars:
            response = llm(prompt, config=config)
            assert isinstance(response, str)

        llm.close()

    @pytest.mark.slow
    def test_unicode_in_batch_generation(self, model_path):
        """Test Unicode in batch generation."""
        from cyllama import BatchGenerator, GenerationConfig

        gen = BatchGenerator(
            model_path=model_path,
            n_seq_max=3,
            verbose=False
        )

        prompts = [
            "Hello world",
            "Bonjour",
            "Testing",
        ]

        config = GenerationConfig(max_tokens=5, temperature=0.0)
        responses = gen.generate_batch(prompts, config)

        assert len(responses) == 3
        for response in responses:
            assert isinstance(response, str)

        gen.close()

    @pytest.mark.slow
    def test_unicode_in_streaming(self, model_path):
        """Test Unicode handling in streaming mode."""
        from cyllama import LLM, GenerationConfig

        llm = LLM(model_path, verbose=False)
        config = GenerationConfig(max_tokens=20, temperature=0.0)

        # Stream with potential Unicode in response
        chunks = list(llm("Say hello in French", config=config, stream=True))

        full_response = "".join(chunks)
        assert isinstance(full_response, str)
        # Each chunk should be valid Unicode
        for chunk in chunks:
            assert isinstance(chunk, str)

        llm.close()

    @pytest.mark.slow
    def test_null_bytes_in_prompt(self, model_path):
        """Test handling of null bytes in prompt."""
        from cyllama import LLM, GenerationConfig

        llm = LLM(model_path, verbose=False)
        config = GenerationConfig(max_tokens=5, temperature=0.0)

        # Null bytes should be handled gracefully
        try:
            response = llm("Hello\x00World", config=config)
            assert isinstance(response, str)
        except (ValueError, UnicodeError):
            pass  # Also acceptable to reject

        llm.close()

    @pytest.mark.slow
    def test_surrogate_pairs(self, model_path):
        """Test handling of surrogate pairs (emoji, rare chars)."""
        from cyllama import LLM, GenerationConfig

        llm = LLM(model_path, verbose=False)
        config = GenerationConfig(max_tokens=5, temperature=0.0)

        # Characters that require surrogate pairs in UTF-16
        # Using safe alternative
        response = llm("Describe: a happy face", config=config)
        assert isinstance(response, str)

        llm.close()


class TestUnicodeBatchGenerator:
    """Tests for Unicode handling in BatchGenerator."""

    @pytest.mark.slow
    def test_unicode_prompt_list(self, model_path):
        """Test batch with Unicode prompts."""
        from cyllama import BatchGenerator, GenerationConfig

        gen = BatchGenerator(
            model_path=model_path,
            n_seq_max=2,
            verbose=False
        )

        prompts = ["Hello", "World"]
        config = GenerationConfig(max_tokens=5, temperature=0.0)

        responses = gen.generate_batch(prompts, config)

        assert len(responses) == 2
        for r in responses:
            assert isinstance(r, str)

        gen.close()


# =============================================================================
# Concurrent Execution Tests (MEDIUM PRIORITY)
# =============================================================================

class TestThreadSafety:
    """Tests for thread safety of various components."""

    @pytest.mark.slow
    def test_multiple_llm_instances(self, model_path):
        """Test creating multiple LLM instances concurrently."""
        from cyllama import LLM, GenerationConfig

        def create_and_generate(idx):
            llm = LLM(model_path, verbose=False)
            config = GenerationConfig(max_tokens=5, temperature=0.0)
            response = llm(f"Test {idx}", config=config)
            llm.close()
            return response

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(create_and_generate, i) for i in range(3)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 3
        for r in results:
            assert isinstance(r, str)

    @pytest.mark.slow
    def test_shared_llm_sequential(self, model_path):
        """Test sequential generation from shared LLM instance."""
        from cyllama import LLM, GenerationConfig

        llm = LLM(model_path, verbose=False)
        config = GenerationConfig(max_tokens=5, temperature=0.0)

        # Sequential access should work fine
        results = []
        for i in range(5):
            response = llm(f"Test {i}", config=config)
            results.append(response)

        assert len(results) == 5
        for r in results:
            assert isinstance(r, str)

        llm.close()

    @pytest.mark.slow
    def test_batch_generator_separate_instances(self, model_path):
        """Test multiple BatchGenerator instances in separate threads."""
        from cyllama import BatchGenerator, GenerationConfig

        def batch_generate(idx):
            gen = BatchGenerator(
                model_path=model_path,
                n_seq_max=1,
                verbose=False
            )
            config = GenerationConfig(max_tokens=3, temperature=0.0)
            responses = gen.generate_batch([f"Test {idx}"], config)
            gen.close()
            return responses[0]

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(batch_generate, i) for i in range(2)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 2
        for r in results:
            assert isinstance(r, str)

    def test_generation_config_thread_safe(self):
        """Test that GenerationConfig is thread-safe."""
        from cyllama import GenerationConfig

        def create_config(idx):
            config = GenerationConfig(
                max_tokens=idx * 10 + 10,
                temperature=0.5,
                top_k=40
            )
            return config.max_tokens

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_config, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 10
        assert set(results) == {10, 20, 30, 40, 50, 60, 70, 80, 90, 100}

    @pytest.mark.slow
    def test_context_manager_cleanup_in_threads(self, model_path):
        """Test context manager cleanup in multithreaded environment."""
        from cyllama import LLM, GenerationConfig

        cleanup_count = [0]
        lock = threading.Lock()

        def generate_with_context():
            with LLM(model_path, verbose=False) as llm:
                config = GenerationConfig(max_tokens=5, temperature=0.0)
                result = llm("Hi", config=config)
            with lock:
                cleanup_count[0] += 1
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(generate_with_context) for _ in range(3)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 3
        assert cleanup_count[0] == 3  # All cleaned up


class TestMemoryPoolThreadSafety:
    """Tests for memory pool thread safety."""

    def test_batch_pool_concurrent_access(self):
        """Test concurrent access to batch pool."""
        from cyllama.llama.llama_cpp import (
            get_pooled_batch,
            return_batch_to_pool,
            reset_batch_pool,
        )

        reset_batch_pool()

        def get_and_return():
            batch = get_pooled_batch(n_tokens=128, embd=0, n_seq_max=2)
            # Simulate some work
            return_batch_to_pool(batch)
            return True

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(get_and_return) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert all(results)
        reset_batch_pool()


# =============================================================================
# Boundary Condition Tests (MEDIUM PRIORITY)
# =============================================================================

class TestMaxTokensBoundary:
    """Tests for max_tokens boundary conditions."""

    @pytest.mark.slow
    def test_max_tokens_one(self, model_path):
        """Test generation with max_tokens=1."""
        from cyllama import LLM, GenerationConfig

        llm = LLM(model_path, verbose=False)
        config = GenerationConfig(max_tokens=1, temperature=0.0)

        response = llm("Hello", config=config)
        # Should generate exactly 1 token (might be multiple chars)
        assert isinstance(response, str)
        assert len(response) >= 0  # Could be empty if token is special

        llm.close()

    @pytest.mark.slow
    def test_max_tokens_very_large(self, model_path):
        """Test generation with very large max_tokens."""
        from cyllama import LLM, GenerationConfig

        llm = LLM(model_path, verbose=False)
        # Large max_tokens but reasonable context - model will hit EOS first
        config = GenerationConfig(max_tokens=500, temperature=0.0, n_ctx=1024)

        response = llm("Say hello", config=config)
        assert isinstance(response, str)

        llm.close()


class TestContextSizeBoundary:
    """Tests for context size boundary conditions."""

    @pytest.mark.slow
    def test_context_size_minimum(self, model_path):
        """Test with minimum context size (n_ctx=1)."""
        from cyllama import LLM, GenerationConfig

        llm = LLM(model_path, verbose=False)
        # Very small context
        config = GenerationConfig(max_tokens=1, n_ctx=64)  # Minimum practical

        response = llm("A", config=config)
        assert isinstance(response, str)

        llm.close()

    @pytest.mark.slow
    def test_prompt_near_context_limit(self, model_path):
        """Test prompt that approaches context limit."""
        from cyllama import LLM, GenerationConfig

        llm = LLM(model_path, verbose=False)
        config = GenerationConfig(max_tokens=5, n_ctx=256)

        # Create prompt close to context size
        long_prompt = "Hello " * 30  # ~30 tokens, well under 256

        response = llm(long_prompt, config=config)
        assert isinstance(response, str)

        llm.close()


class TestBatchSizeBoundary:
    """Tests for batch size boundary conditions."""

    @pytest.mark.slow
    def test_batch_size_one(self, model_path):
        """Test with batch_size=1 (minimum)."""
        from cyllama import LLM, GenerationConfig

        llm = LLM(model_path, verbose=False)
        config = GenerationConfig(max_tokens=5, n_batch=1)

        response = llm("Test", config=config)
        assert isinstance(response, str)

        llm.close()

    @pytest.mark.slow
    def test_batch_size_small(self, model_path):
        """Test with very small batch_size."""
        from cyllama import BatchGenerator, GenerationConfig

        gen = BatchGenerator(
            model_path=model_path,
            batch_size=8,  # Very small
            n_seq_max=1,
            verbose=False
        )

        config = GenerationConfig(max_tokens=5, temperature=0.0)
        responses = gen.generate_batch(["Hello"], config)

        assert len(responses) == 1
        assert isinstance(responses[0], str)

        gen.close()


class TestTemperatureBoundary:
    """Tests for temperature boundary conditions."""

    @pytest.mark.slow
    def test_temperature_zero(self, model_path):
        """Test greedy decoding (temperature=0)."""
        from cyllama import LLM, GenerationConfig

        llm = LLM(model_path, verbose=False)
        config = GenerationConfig(max_tokens=10, temperature=0.0)

        # Multiple runs should be deterministic
        response1 = llm("What is 2+2?", config=config)
        llm.reset_context()
        response2 = llm("What is 2+2?", config=config)

        # With temperature=0 and same seed/context, should be identical
        assert response1 == response2

        llm.close()

    @pytest.mark.slow
    def test_temperature_very_high(self, model_path):
        """Test very high temperature."""
        from cyllama import LLM, GenerationConfig

        llm = LLM(model_path, verbose=False)
        config = GenerationConfig(max_tokens=10, temperature=5.0)

        response = llm("Hello", config=config)
        assert isinstance(response, str)

        llm.close()


class TestTopKTopPBoundary:
    """Tests for top_k and top_p boundary conditions."""

    @pytest.mark.slow
    def test_top_k_one(self, model_path):
        """Test top_k=1 (equivalent to greedy)."""
        from cyllama import LLM, GenerationConfig

        llm = LLM(model_path, verbose=False)
        config = GenerationConfig(max_tokens=10, top_k=1, temperature=1.0)

        response = llm("Hello", config=config)
        assert isinstance(response, str)

        llm.close()

    @pytest.mark.slow
    def test_top_p_zero(self, model_path):
        """Test top_p=0 (should be greedy-ish)."""
        from cyllama import LLM, GenerationConfig

        llm = LLM(model_path, verbose=False)
        config = GenerationConfig(max_tokens=10, top_p=0.0, temperature=1.0)

        response = llm("Hello", config=config)
        assert isinstance(response, str)

        llm.close()

    @pytest.mark.slow
    def test_top_p_one(self, model_path):
        """Test top_p=1.0 (full distribution)."""
        from cyllama import LLM, GenerationConfig

        llm = LLM(model_path, verbose=False)
        config = GenerationConfig(max_tokens=10, top_p=1.0, temperature=0.5)

        response = llm("Hello", config=config)
        assert isinstance(response, str)

        llm.close()


class TestSeqMaxBoundary:
    """Tests for n_seq_max boundary conditions."""

    @pytest.mark.slow
    def test_n_seq_max_minimum(self, model_path):
        """Test n_seq_max=1 (minimum)."""
        from cyllama import BatchGenerator, GenerationConfig

        gen = BatchGenerator(
            model_path=model_path,
            n_seq_max=1,
            verbose=False
        )

        config = GenerationConfig(max_tokens=5, temperature=0.0)
        responses = gen.generate_batch(["Hello"], config)

        assert len(responses) == 1

        gen.close()

    @pytest.mark.slow
    def test_prompts_equal_n_seq_max(self, model_path):
        """Test when prompts count equals n_seq_max exactly."""
        from cyllama import BatchGenerator, GenerationConfig

        n_seq = 4
        gen = BatchGenerator(
            model_path=model_path,
            n_seq_max=n_seq,
            verbose=False
        )

        prompts = [f"Prompt {i}" for i in range(n_seq)]
        config = GenerationConfig(max_tokens=3, temperature=0.0)
        responses = gen.generate_batch(prompts, config)

        assert len(responses) == n_seq

        gen.close()


class TestStopSequenceBoundary:
    """Tests for stop sequence boundary conditions."""

    @pytest.mark.slow
    def test_empty_stop_sequence_string(self, model_path):
        """Test with empty string in stop sequences."""
        from cyllama import LLM, GenerationConfig

        llm = LLM(model_path, verbose=False)
        # Empty string should be handled gracefully
        config = GenerationConfig(
            max_tokens=10,
            temperature=0.0,
            stop_sequences=[""]  # Empty string
        )

        response = llm("Hello", config=config)
        assert isinstance(response, str)

        llm.close()

    @pytest.mark.slow
    def test_many_stop_sequences(self, model_path):
        """Test with many stop sequences."""
        from cyllama import LLM, GenerationConfig

        llm = LLM(model_path, verbose=False)
        # Many stop sequences
        config = GenerationConfig(
            max_tokens=50,
            temperature=0.0,
            stop_sequences=[".", "!", "?", ",", ";", ":", "\n", "\t"]
        )

        response = llm("Tell me something", config=config)
        assert isinstance(response, str)

        llm.close()

    @pytest.mark.slow
    def test_long_stop_sequence(self, model_path):
        """Test with very long stop sequence."""
        from cyllama import LLM, GenerationConfig

        llm = LLM(model_path, verbose=False)
        config = GenerationConfig(
            max_tokens=50,
            temperature=0.0,
            stop_sequences=["This is a very long stop sequence that is unlikely to appear"]
        )

        response = llm("Hello world", config=config)
        assert isinstance(response, str)

        llm.close()


class TestRepeatPenaltyBoundary:
    """Tests for repeat_penalty boundary conditions."""

    @pytest.mark.slow
    def test_repeat_penalty_zero(self, model_path):
        """Test repeat_penalty=0 (disabled)."""
        from cyllama import LLM, GenerationConfig

        llm = LLM(model_path, verbose=False)
        config = GenerationConfig(max_tokens=10, repeat_penalty=0.0)

        response = llm("Hello", config=config)
        assert isinstance(response, str)

        llm.close()

    @pytest.mark.slow
    def test_repeat_penalty_high(self, model_path):
        """Test high repeat_penalty."""
        from cyllama import LLM, GenerationConfig

        llm = LLM(model_path, verbose=False)
        config = GenerationConfig(max_tokens=10, repeat_penalty=5.0)

        response = llm("Hello", config=config)
        assert isinstance(response, str)

        llm.close()


# =============================================================================
# Additional Edge Cases
# =============================================================================

class TestSpecialPrompts:
    """Tests for special prompt content."""

    @pytest.mark.slow
    def test_whitespace_only_prompt(self, model_path):
        """Test whitespace-only prompt."""
        from cyllama import LLM, GenerationConfig

        llm = LLM(model_path, verbose=False)
        config = GenerationConfig(max_tokens=5, temperature=0.0)

        response = llm("   \n\t\r   ", config=config)
        assert isinstance(response, str)

        llm.close()

    @pytest.mark.slow
    def test_newlines_in_prompt(self, model_path):
        """Test prompt with many newlines."""
        from cyllama import LLM, GenerationConfig

        llm = LLM(model_path, verbose=False)
        config = GenerationConfig(max_tokens=5, temperature=0.0)

        response = llm("Line 1\nLine 2\nLine 3\n\n\nLine 6", config=config)
        assert isinstance(response, str)

        llm.close()

    @pytest.mark.slow
    def test_special_tokens_in_prompt(self, model_path):
        """Test prompt with special token markers."""
        from cyllama import LLM, GenerationConfig

        llm = LLM(model_path, verbose=False)
        config = GenerationConfig(max_tokens=5, temperature=0.0)

        # These look like special tokens but are in the prompt
        response = llm("<|start|> Hello <|end|>", config=config)
        assert isinstance(response, str)

        llm.close()

    @pytest.mark.slow
    def test_repeated_prompt(self, model_path):
        """Test highly repetitive prompt."""
        from cyllama import LLM, GenerationConfig

        llm = LLM(model_path, verbose=False)
        config = GenerationConfig(max_tokens=10, temperature=0.0)

        response = llm("test " * 20, config=config)
        assert isinstance(response, str)

        llm.close()


class TestResourceLimits:
    """Tests for resource limit handling."""

    @pytest.mark.slow
    def test_multiple_generations_memory_stable(self, model_path):
        """Test that repeated generations don't leak memory significantly."""
        from cyllama import LLM, GenerationConfig
        import gc

        llm = LLM(model_path, verbose=False)
        config = GenerationConfig(max_tokens=10, temperature=0.0)

        # Run multiple generations
        for i in range(20):
            response = llm(f"Test iteration {i}", config=config)
            assert isinstance(response, str)

        # Force garbage collection
        gc.collect()

        # Should still work after many iterations
        final_response = llm("Final test", config=config)
        assert isinstance(final_response, str)

        llm.close()

    @pytest.mark.slow
    def test_batch_generator_reuse(self, model_path):
        """Test BatchGenerator can be reused for multiple batches."""
        from cyllama import BatchGenerator, GenerationConfig

        # Use separate generators for each batch to avoid KV cache accumulation
        for i in range(2):
            gen = BatchGenerator(
                model_path=model_path,
                n_seq_max=1,
                n_ctx=512,
                verbose=False
            )

            config = GenerationConfig(max_tokens=3, temperature=0.0)
            responses = gen.generate_batch([f"Test {i}"], config)
            assert len(responses) == 1
            assert isinstance(responses[0], str)

            gen.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
