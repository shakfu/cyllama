"""
Tests for async API module.

Tests the async/await support for cyllama text generation.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock

# Import the API (sync and async are now in the same module)
from cyllama.api import (
    AsyncLLM,
    complete_async,
    chat_async,
    stream_complete_async,
    GenerationConfig,
    LLM,
)


class TestAsyncLLM:
    """Tests for AsyncLLM class."""

    def test_init_with_defaults(self):
        """Test AsyncLLM initialization with default config."""
        with patch.object(LLM, '__init__', return_value=None):
            llm = AsyncLLM("model.gguf")
            assert llm._llm is not None
            assert llm._lock is not None

    def test_init_with_kwargs(self):
        """Test AsyncLLM initialization with kwargs."""
        with patch.object(LLM, '__init__', return_value=None) as mock_init:
            llm = AsyncLLM("model.gguf", temperature=0.5, max_tokens=100)
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs.get('temperature') == 0.5
            assert call_kwargs.get('max_tokens') == 100

    def test_init_with_config(self):
        """Test AsyncLLM initialization with GenerationConfig."""
        config = GenerationConfig(temperature=0.7)
        with patch.object(LLM, '__init__', return_value=None) as mock_init:
            llm = AsyncLLM("model.gguf", config=config)
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs.get('config') == config

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        with patch.object(LLM, '__init__', return_value=None):
            with patch.object(LLM, 'close', return_value=None) as mock_close:
                async with AsyncLLM("model.gguf") as llm:
                    assert llm is not None
                mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close(self):
        """Test explicit close."""
        with patch.object(LLM, '__init__', return_value=None):
            with patch.object(LLM, 'close', return_value=None) as mock_close:
                llm = AsyncLLM("model.gguf")
                await llm.close()
                mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate(self):
        """Test async generation."""
        with patch.object(LLM, '__init__', return_value=None):
            with patch.object(LLM, '_generate', return_value="Hello world") as mock_gen:
                llm = AsyncLLM("model.gguf")
                result = await llm("Test prompt")
                assert result == "Hello world"
                mock_gen.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_method(self):
        """Test explicit generate method."""
        with patch.object(LLM, '__init__', return_value=None):
            with patch.object(LLM, '_generate', return_value="Response") as mock_gen:
                llm = AsyncLLM("model.gguf")
                result = await llm.generate("Test prompt")
                assert result == "Response"

    @pytest.mark.asyncio
    async def test_generate_with_kwargs(self):
        """Test generation with override kwargs."""
        with patch.object(LLM, '__init__', return_value=None):
            llm = AsyncLLM("model.gguf")
            # Mock config
            llm._llm.config = GenerationConfig()

            with patch.object(LLM, '_generate', return_value="Result") as mock_gen:
                result = await llm("Prompt", temperature=0.5)
                assert result == "Result"
                # Check that config was built with override
                call_args = mock_gen.call_args
                config = call_args[0][1]  # Second positional arg
                assert config.temperature == 0.5

    @pytest.mark.asyncio
    async def test_reset_context(self):
        """Test async reset_context."""
        with patch.object(LLM, '__init__', return_value=None):
            with patch.object(LLM, 'reset_context', return_value=None) as mock_reset:
                llm = AsyncLLM("model.gguf")
                await llm.reset_context()
                mock_reset.assert_called_once()

    def test_config_property(self):
        """Test config property access."""
        config = GenerationConfig(temperature=0.8)
        with patch.object(LLM, '__init__', return_value=None):
            llm = AsyncLLM("model.gguf")
            llm._llm.config = config
            assert llm.config == config

    def test_model_path_property(self):
        """Test model_path property access."""
        with patch.object(LLM, '__init__', return_value=None):
            llm = AsyncLLM("model.gguf")
            llm._llm.model_path = "test/path.gguf"
            assert llm.model_path == "test/path.gguf"

    @pytest.mark.asyncio
    async def test_generate_with_stats(self):
        """Test generate_with_stats method."""
        from cyllama.api import GenerationStats

        stats = GenerationStats(
            prompt_tokens=10,
            generated_tokens=20,
            total_time=1.0,
            tokens_per_second=20.0
        )

        with patch.object(LLM, '__init__', return_value=None):
            with patch.object(LLM, 'generate_with_stats', return_value=("Result", stats)):
                llm = AsyncLLM("model.gguf")
                text, returned_stats = await llm.generate_with_stats("Prompt")
                assert text == "Result"
                assert returned_stats.generated_tokens == 20

    @pytest.mark.asyncio
    async def test_build_config(self):
        """Test _build_config helper."""
        with patch.object(LLM, '__init__', return_value=None):
            llm = AsyncLLM("model.gguf")
            llm._llm.config = GenerationConfig(temperature=0.8, max_tokens=100)

            new_config = llm._build_config(None, {'temperature': 0.5})
            assert new_config.temperature == 0.5
            assert new_config.max_tokens == 100  # Inherited from base

    @pytest.mark.asyncio
    async def test_concurrent_access_blocked(self):
        """Test that concurrent access is serialized by lock."""
        call_count = 0
        call_order = []

        async def slow_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            current = call_count
            call_order.append(f"start_{current}")
            await asyncio.sleep(0.1)
            call_order.append(f"end_{current}")
            return f"Result {current}"

        with patch.object(LLM, '__init__', return_value=None):
            llm = AsyncLLM("model.gguf")

            with patch('cyllama.api.asyncio.to_thread', side_effect=slow_generate):
                # Start two concurrent calls
                task1 = asyncio.create_task(llm("Prompt 1"))
                task2 = asyncio.create_task(llm("Prompt 2"))

                results = await asyncio.gather(task1, task2)

                # Due to lock, calls should be serialized
                assert call_order == ["start_1", "end_1", "start_2", "end_2"]


class TestCompleteAsync:
    """Tests for complete_async function."""

    @pytest.mark.asyncio
    async def test_complete_async_basic(self):
        """Test basic async completion."""
        with patch('cyllama.api.complete', return_value="Response") as mock_complete:
            result = await complete_async("Test", model_path="model.gguf")
            assert result == "Response"
            mock_complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_async_with_kwargs(self):
        """Test async completion with kwargs."""
        with patch('cyllama.api.complete', return_value="Response") as mock_complete:
            result = await complete_async(
                "Test",
                model_path="model.gguf",
                temperature=0.5,
                max_tokens=100
            )
            assert result == "Response"
            call_kwargs = mock_complete.call_args[1]
            assert call_kwargs.get('temperature') == 0.5
            assert call_kwargs.get('max_tokens') == 100


class TestChatAsync:
    """Tests for chat_async function."""

    @pytest.mark.asyncio
    async def test_chat_async_basic(self):
        """Test basic async chat."""
        messages = [
            {"role": "user", "content": "Hello"}
        ]

        with patch('cyllama.api.chat', return_value="Hi there!") as mock_chat:
            result = await chat_async(messages, model_path="model.gguf")
            assert result == "Hi there!"
            mock_chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_async_with_system(self):
        """Test async chat with system message."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"}
        ]

        with patch('cyllama.api.chat', return_value="Response") as mock_chat:
            result = await chat_async(messages, model_path="model.gguf")
            assert result == "Response"


class TestStreamCompleteAsync:
    """Tests for stream_complete_async function."""

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self):
        """Test that streaming yields chunks."""
        chunks = ["Hello", " ", "World"]

        async def mock_stream(*args, **kwargs):
            for chunk in chunks:
                yield chunk

        with patch.object(AsyncLLM, '__init__', return_value=None):
            with patch.object(AsyncLLM, 'stream', side_effect=mock_stream):
                with patch.object(AsyncLLM, 'close', return_value=None):
                    result_chunks = []
                    async for chunk in stream_complete_async("Test", "model.gguf"):
                        result_chunks.append(chunk)

                    assert result_chunks == chunks


class TestAsyncLLMStream:
    """Tests for AsyncLLM streaming."""

    @pytest.mark.asyncio
    async def test_stream_basic(self):
        """Test basic streaming."""

        def sync_generator(prompt, config):
            yield "Hello"
            yield " "
            yield "World"

        with patch.object(LLM, '__init__', return_value=None):
            llm = AsyncLLM("model.gguf")
            llm._llm._generate_stream = sync_generator

            chunks = []
            async for chunk in llm.stream("Test"):
                chunks.append(chunk)

            assert chunks == ["Hello", " ", "World"]


# Integration tests (require actual model)
class TestAsyncIntegration:
    """Integration tests with real model."""

    @pytest.fixture
    def model_path(self):
        """Get test model path."""
        import os
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "models",
            "Llama-3.2-1B-Instruct-Q8_0.gguf"
        )
        if not os.path.exists(path):
            pytest.skip("Test model not available")
        return path

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_real_generation(self, model_path):
        """Test real async generation."""
        async with AsyncLLM(model_path, max_tokens=10) as llm:
            result = await llm("Say hello")
            assert len(result) > 0

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_real_streaming(self, model_path):
        """Test real async streaming."""
        async with AsyncLLM(model_path, max_tokens=10) as llm:
            chunks = []
            async for chunk in llm.stream("Say hello"):
                chunks.append(chunk)
            assert len(chunks) > 0
            assert "".join(chunks)  # Should produce some output

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_complete_async_real(self, model_path):
        """Test real complete_async."""
        result = await complete_async(
            "What is 2+2?",
            model_path=model_path,
            max_tokens=20
        )
        assert len(result) > 0
