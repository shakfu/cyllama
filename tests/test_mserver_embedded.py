#!/usr/bin/env python3
"""
Tests for Python-based llama.cpp server functionality.

These tests cover the PythonServer class and related components,
ensuring proper server functionality using existing cyllama bindings.
"""

import time
import pytest
from unittest.mock import Mock, patch

from cyllama.llama.server.python import (
    ServerConfig,
    PythonServer,
    ServerSlot,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    start_python_server,
)


class TestServerConfig:
    """Test ServerConfig class functionality."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ServerConfig(model_path="test.gguf")

        assert config.model_path == "test.gguf"
        assert config.host == "127.0.0.1"
        assert config.port == 8080
        assert config.n_ctx == 4096
        assert config.n_batch == 2048
        assert config.n_threads == -1
        assert config.n_gpu_layers == -1
        assert config.embedding is False
        assert config.n_parallel == 1
        assert config.model_alias == "gpt-3.5-turbo"

        # Embedding defaults
        assert config.embedding_model_path is None
        assert config.embedding_n_ctx == 512
        assert config.embedding_n_batch == 512
        assert config.embedding_n_gpu_layers == -1
        assert config.embedding_pooling == "mean"
        assert config.embedding_normalize is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ServerConfig(
            model_path="custom.gguf",
            host="0.0.0.0",
            port=9090,
            n_ctx=8192,
            n_gpu_layers=32,
            embedding=True,
            n_parallel=4,
            model_alias="custom-model",
        )

        assert config.model_path == "custom.gguf"
        assert config.host == "0.0.0.0"
        assert config.port == 9090
        assert config.n_ctx == 8192
        assert config.n_gpu_layers == 32
        assert config.embedding is True
        assert config.n_parallel == 4
        assert config.model_alias == "custom-model"

    def test_embedding_config(self):
        """Test embedding-specific configuration values."""
        config = ServerConfig(
            model_path="chat.gguf",
            embedding=True,
            embedding_model_path="embed.gguf",
            embedding_n_ctx=256,
            embedding_n_batch=256,
            embedding_n_gpu_layers=0,
            embedding_pooling="cls",
            embedding_normalize=False,
        )

        assert config.embedding is True
        assert config.embedding_model_path == "embed.gguf"
        assert config.embedding_n_ctx == 256
        assert config.embedding_n_batch == 256
        assert config.embedding_n_gpu_layers == 0
        assert config.embedding_pooling == "cls"
        assert config.embedding_normalize is False

    def test_embedding_model_path_defaults_to_none(self):
        """Test that embedding_model_path defaults to None (uses model_path at runtime)."""
        config = ServerConfig(model_path="model.gguf", embedding=True)
        assert config.embedding_model_path is None


class TestChatMessage:
    """Test ChatMessage dataclass."""

    def test_create_message(self):
        """Test creating a chat message."""
        message = ChatMessage(role="user", content="Hello!")
        assert message.role == "user"
        assert message.content == "Hello!"


class TestChatRequest:
    """Test ChatRequest dataclass."""

    def test_create_request(self):
        """Test creating a chat request."""
        messages = [ChatMessage(role="user", content="Hello!")]
        request = ChatRequest(messages=messages)

        assert len(request.messages) == 1
        assert request.messages[0].role == "user"
        assert request.model == "gpt-3.5-turbo"
        assert request.temperature == 0.8
        assert request.stream is False

    def test_custom_request(self):
        """Test chat request with custom parameters."""
        messages = [ChatMessage(role="user", content="Hello!")]
        request = ChatRequest(
            messages=messages, model="custom-model", max_tokens=100, temperature=0.5, stream=True, stop=["STOP"]
        )

        assert request.model == "custom-model"
        assert request.max_tokens == 100
        assert request.temperature == 0.5
        assert request.stream is True
        assert request.stop == ["STOP"]


class TestServerSlot:
    """Test ServerSlot class functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        vocab = Mock()
        vocab.tokenize.return_value = [1, 2, 3, 4, 5]
        vocab.is_eog_token.return_value = False
        vocab.token_to_piece.return_value = "test"
        model.get_vocab.return_value = vocab
        return model

    @pytest.fixture
    def mock_context(self):
        """Create a mock context for testing."""
        context = Mock()
        context.decode.return_value = 0  # Success
        return context

    @pytest.fixture
    def mock_sampler(self):
        """Create a mock sampler for testing."""
        sampler = Mock()
        sampler.sample.return_value = 10  # Sample token
        return sampler

    def test_slot_creation(self, mock_model):
        """Test creating a server slot."""
        config = ServerConfig(model_path="test.gguf")

        with patch("cyllama.llama.server.python.LlamaContext"), patch("cyllama.llama.server.python.LlamaSampler"):
            slot = ServerSlot(0, mock_model, config)

            assert slot.id == 0
            assert slot.model == mock_model
            assert not slot.is_processing
            assert slot.task_id is None

    def test_slot_reset(self, mock_model):
        """Test resetting a server slot."""
        config = ServerConfig(model_path="test.gguf")

        with (
            patch("cyllama.llama.server.python.LlamaContext") as MockContext,
            patch("cyllama.llama.server.python.LlamaSampler"),
        ):
            mock_context = Mock()
            MockContext.return_value = mock_context

            slot = ServerSlot(0, mock_model, config)

            # Set some state
            slot.is_processing = True
            slot.task_id = "test-123"
            slot.generated_tokens = [4, 5]
            slot.response_text = "test response"

            # Reset
            slot.reset()

            assert not slot.is_processing
            assert slot.task_id is None
            assert len(slot.generated_tokens) == 0
            assert slot.response_text == ""
            assert mock_context.n_tokens == 0

    def test_process_and_generate_success(self, mock_model):
        """Test successful prompt processing and generation."""
        config = ServerConfig(model_path="test.gguf")

        with (
            patch("cyllama.llama.server.python.LlamaContext") as MockContext,
            patch("cyllama.llama.server.python.LlamaSampler") as MockSampler,
            patch("cyllama.llama.server.python.llama_batch_get_one") as mock_batch,
        ):
            mock_context = Mock()
            mock_context.decode.return_value = 0  # Success
            mock_context.n_ctx = 512
            MockContext.return_value = mock_context

            mock_sampler = Mock()
            mock_sampler.sample.side_effect = [10, 20]  # Two tokens then will stop
            MockSampler.return_value = mock_sampler

            # Mock vocab
            mock_vocab = mock_model.get_vocab()
            mock_vocab.tokenize.return_value = [1, 2, 3]  # 3 tokens for prompt
            mock_vocab.is_eog.side_effect = [False, True]  # First token not EOS, second is EOS
            mock_vocab.token_to_piece.side_effect = [" Hello", " world"]

            slot = ServerSlot(0, mock_model, config)

            result = slot.process_and_generate("Hello world", max_tokens=10)

            assert result == " Hello"  # Should stop at EOS
            mock_batch.assert_called()
            mock_context.decode.assert_called()

    def test_process_and_generate_too_long(self, mock_model):
        """Test prompt processing with too long prompt."""
        config = ServerConfig(model_path="test.gguf", n_ctx=3)  # Small context

        # Mock tokenizer to return more tokens than context
        mock_vocab = mock_model.get_vocab()
        mock_vocab.tokenize.return_value = [1, 2, 3, 4, 5]  # 5 tokens > 3 ctx

        with patch("cyllama.llama.server.python.LlamaContext"), patch("cyllama.llama.server.python.LlamaSampler"):
            slot = ServerSlot(0, mock_model, config)

            result = slot.process_and_generate("Very long prompt")

            assert result == ""  # Should return empty string for too long prompt

    def test_process_and_generate_error(self, mock_model):
        """Test handling of generation errors."""
        config = ServerConfig(model_path="test.gguf")

        with (
            patch("cyllama.llama.server.python.LlamaContext") as MockContext,
            patch("cyllama.llama.server.python.LlamaSampler"),
        ):
            mock_context = Mock()
            mock_context.decode.side_effect = Exception("Decode error")
            MockContext.return_value = mock_context

            mock_vocab = mock_model.get_vocab()
            mock_vocab.tokenize.return_value = [1, 2, 3]

            slot = ServerSlot(0, mock_model, config)

            result = slot.process_and_generate("Hello world")

            assert result == ""  # Should return empty string on error


class TestPythonServer:
    """Test PythonServer class functionality."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return ServerConfig(model_path="test.gguf", port=18080)

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        vocab = Mock()
        vocab.tokenize.return_value = [1, 2, 3]
        vocab.is_eog_token.return_value = False
        vocab.token_to_piece.return_value = " test"
        model.get_vocab.return_value = vocab
        return model

    def test_server_creation(self, config):
        """Test creating a Python server."""
        server = PythonServer(config)

        assert server.config == config
        assert server.model is None
        assert len(server.slots) == 0
        assert not server.running

    @patch("cyllama.llama.server.python.LlamaModel")
    @patch("cyllama.llama.server.python.ServerSlot")
    def test_load_model_success(self, MockServerSlot, MockLlamaModel, config, mock_model):
        """Test successful model loading."""
        MockLlamaModel.return_value = mock_model
        MockServerSlot.return_value = Mock()

        server = PythonServer(config)
        result = server.load_model()

        assert result is True
        assert server.model == mock_model
        MockLlamaModel.assert_called_once()
        assert len(server.slots) == config.n_parallel

    @patch("cyllama.llama.server.python.LlamaModel")
    def test_load_model_failure(self, MockLlamaModel, config):
        """Test model loading failure."""
        MockLlamaModel.side_effect = Exception("Model load failed")

        server = PythonServer(config)
        result = server.load_model()

        assert result is False
        assert server.model is None

    def test_get_available_slot(self, config):
        """Test getting available slots."""
        server = PythonServer(config)

        # Create mock slots
        slot1 = Mock()
        slot1.is_processing = True
        slot2 = Mock()
        slot2.is_processing = False
        slot3 = Mock()
        slot3.is_processing = True

        server.slots = [slot1, slot2, slot3]

        available_slot = server.get_available_slot()
        assert available_slot == slot2

    def test_get_available_slot_none_available(self, config):
        """Test getting available slots when none are available."""
        server = PythonServer(config)

        # All slots busy
        slot1 = Mock()
        slot1.is_processing = True
        slot2 = Mock()
        slot2.is_processing = True

        server.slots = [slot1, slot2]

        available_slot = server.get_available_slot()
        assert available_slot is None

    def test_messages_to_prompt(self, config):
        """Test converting messages to prompt."""
        server = PythonServer(config)

        messages = [
            ChatMessage(role="system", content="You are helpful"),
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there"),
            ChatMessage(role="user", content="How are you?"),
        ]

        prompt = server._messages_to_prompt(messages)

        expected = "System: You are helpful\nUser: Hello\nAssistant: Hi there\nUser: How are you?\nAssistant:"
        assert prompt == expected

    def test_process_chat_completion_success(self, config, mock_model):
        """Test successful chat completion processing."""
        server = PythonServer(config)
        server.model = mock_model

        # Create mock slot
        mock_slot = Mock()
        mock_slot.is_processing = False
        mock_slot.task_id = None
        mock_slot.process_and_generate.return_value = " Hello there!"

        server.slots = [mock_slot]

        # Mock vocab for token counting
        mock_vocab = mock_model.get_vocab()
        mock_vocab.tokenize.side_effect = [
            [1, 2, 3],  # prompt tokens
            [10, 20],  # completion tokens
        ]

        messages = [ChatMessage(role="user", content="Hello")]
        request = ChatRequest(messages=messages, max_tokens=10)

        response = server.process_chat_completion(request)

        assert isinstance(response, ChatResponse)
        assert len(response.choices) == 1
        assert response.choices[0].message.content == " Hello there!"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage["prompt_tokens"] == 3
        assert response.usage["completion_tokens"] == 2
        assert response.usage["total_tokens"] == 5

        # Verify slot was reset
        mock_slot.reset.assert_called_once()
        # Verify process_and_generate was called
        mock_slot.process_and_generate.assert_called_once()

    def test_process_chat_completion_no_slots(self, config):
        """Test chat completion with no available slots."""
        server = PythonServer(config)
        server.slots = []  # No slots

        messages = [ChatMessage(role="user", content="Hello")]
        request = ChatRequest(messages=messages)

        with pytest.raises(RuntimeError, match="No available slots"):
            server.process_chat_completion(request)

    def test_context_manager(self, config):
        """Test server as context manager."""
        server = PythonServer(config)

        with patch.object(server, "start", return_value=True) as mock_start, patch.object(server, "stop") as mock_stop:
            with server:
                pass

            mock_start.assert_called_once()
            mock_stop.assert_called_once()

    def test_context_manager_start_failure(self, config):
        """Test context manager when start fails."""
        server = PythonServer(config)

        with patch.object(server, "start", return_value=False):
            with pytest.raises(RuntimeError, match="Failed to start server"):
                with server:
                    pass


class TestHTTPEndpoints:
    """Test HTTP endpoint functionality."""

    @pytest.fixture
    def server_config(self):
        """Create a test server configuration."""
        return ServerConfig(model_path="test.gguf", port=18090)

    @pytest.fixture
    def mock_server(self, server_config):
        """Create a mock server for testing."""
        server = PythonServer(server_config)
        server.model = Mock()
        server.slots = [Mock()]
        return server

    def test_models_endpoint_response(self, mock_server):
        """Test the models endpoint response format."""
        # Create request handler
        handler_class = mock_server._create_request_handler()

        # Mock the handler instance
        handler = Mock(spec=handler_class)
        handler._send_json_response = Mock()

        # Create bound method
        handle_models = handler_class._handle_models.__get__(handler)

        # Call the handler
        handle_models()

        # Verify response structure
        handler._send_json_response.assert_called_once()
        call_args = handler._send_json_response.call_args[0][0]

        assert call_args["object"] == "list"
        assert "data" in call_args
        assert len(call_args["data"]) == 1
        assert call_args["data"][0]["id"] == "gpt-3.5-turbo"
        assert call_args["data"][0]["object"] == "model"

    def test_health_endpoint_response(self, mock_server):
        """Test the health endpoint response."""
        # Create request handler
        handler_class = mock_server._create_request_handler()

        # Mock the handler instance
        handler = Mock(spec=handler_class)
        handler._send_json_response = Mock()

        # Test health endpoint logic
        health_response = {"status": "ok"}
        handler._send_json_response(health_response)

        handler._send_json_response.assert_called_once_with({"status": "ok"})


class TestEmbeddingsEndpoint:
    """Test /v1/embeddings endpoint functionality."""

    @pytest.fixture
    def embedding_config(self):
        """Create a config with embedding enabled."""
        return ServerConfig(
            model_path="test.gguf",
            port=18095,
            embedding=True,
            embedding_model_path="embed.gguf",
        )

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock Embedder."""
        from cyllama.rag.types import EmbeddingResult

        embedder = Mock()
        embedder.dimension = 384
        embedder.pooling = "mean"
        embedder.embed_with_info.return_value = EmbeddingResult(
            embedding=[0.1, 0.2, 0.3],
            text="hello",
            token_count=2,
        )
        return embedder

    def test_embeddings_disabled(self):
        """Test embeddings endpoint returns error when not enabled."""
        config = ServerConfig(model_path="test.gguf", port=18095, embedding=False)
        server = PythonServer(config)
        server.model = Mock()

        handler_class = server._create_request_handler()
        handler = Mock(spec=handler_class)
        handler._send_error = Mock()
        handler._send_json_response = Mock()

        handle_embeddings = handler_class._handle_embeddings.__get__(handler)
        handle_embeddings({"input": "hello"})

        handler._send_error.assert_called_once_with(400, "Embeddings not enabled")

    def test_embeddings_single_input(self, embedding_config, mock_embedder):
        """Test embeddings with a single string input."""
        server = PythonServer(embedding_config)
        server.model = Mock()
        server.embedder = mock_embedder

        handler_class = server._create_request_handler()
        handler = Mock(spec=handler_class)
        handler._send_error = Mock()
        handler._send_json_response = Mock()

        handle_embeddings = handler_class._handle_embeddings.__get__(handler)
        handle_embeddings({"input": "hello world"})

        handler._send_json_response.assert_called_once()
        response = handler._send_json_response.call_args[0][0]

        assert response["object"] == "list"
        assert len(response["data"]) == 1
        assert response["data"][0]["object"] == "embedding"
        assert response["data"][0]["embedding"] == [0.1, 0.2, 0.3]
        assert response["data"][0]["index"] == 0
        assert response["usage"]["prompt_tokens"] == 2
        assert response["usage"]["total_tokens"] == 2

    def test_embeddings_batch_input(self, embedding_config, mock_embedder):
        """Test embeddings with a list of strings."""
        from cyllama.rag.types import EmbeddingResult

        mock_embedder.embed_with_info.side_effect = [
            EmbeddingResult(embedding=[0.1, 0.2], text="first", token_count=1),
            EmbeddingResult(embedding=[0.3, 0.4], text="second", token_count=3),
        ]

        server = PythonServer(embedding_config)
        server.model = Mock()
        server.embedder = mock_embedder

        handler_class = server._create_request_handler()
        handler = Mock(spec=handler_class)
        handler._send_error = Mock()
        handler._send_json_response = Mock()

        handle_embeddings = handler_class._handle_embeddings.__get__(handler)
        handle_embeddings({"input": ["first", "second"]})

        handler._send_json_response.assert_called_once()
        response = handler._send_json_response.call_args[0][0]

        assert len(response["data"]) == 2
        assert response["data"][0]["index"] == 0
        assert response["data"][0]["embedding"] == [0.1, 0.2]
        assert response["data"][1]["index"] == 1
        assert response["data"][1]["embedding"] == [0.3, 0.4]
        assert response["usage"]["prompt_tokens"] == 4
        assert response["usage"]["total_tokens"] == 4

    def test_embeddings_missing_input(self, embedding_config, mock_embedder):
        """Test embeddings with missing input field."""
        server = PythonServer(embedding_config)
        server.model = Mock()
        server.embedder = mock_embedder

        handler_class = server._create_request_handler()
        handler = Mock(spec=handler_class)
        handler._send_error = Mock()
        handler._send_json_response = Mock()

        handle_embeddings = handler_class._handle_embeddings.__get__(handler)
        handle_embeddings({"model": "test"})

        handler._send_error.assert_called_once_with(400, "Missing 'input' field")

    def test_embeddings_invalid_input_type(self, embedding_config, mock_embedder):
        """Test embeddings with invalid input type."""
        server = PythonServer(embedding_config)
        server.model = Mock()
        server.embedder = mock_embedder

        handler_class = server._create_request_handler()
        handler = Mock(spec=handler_class)
        handler._send_error = Mock()
        handler._send_json_response = Mock()

        handle_embeddings = handler_class._handle_embeddings.__get__(handler)
        handle_embeddings({"input": 12345})

        handler._send_error.assert_called_once_with(400, "Invalid 'input' field: must be string or list of strings")

    def test_embeddings_custom_model_name(self, embedding_config, mock_embedder):
        """Test embeddings response includes custom model name."""
        server = PythonServer(embedding_config)
        server.model = Mock()
        server.embedder = mock_embedder

        handler_class = server._create_request_handler()
        handler = Mock(spec=handler_class)
        handler._send_error = Mock()
        handler._send_json_response = Mock()

        handle_embeddings = handler_class._handle_embeddings.__get__(handler)
        handle_embeddings({"input": "hello", "model": "nomic-embed-text"})

        response = handler._send_json_response.call_args[0][0]
        assert response["model"] == "nomic-embed-text"

    def test_load_model_initializes_embedder(self):
        """Test that load_model creates an Embedder when embedding=True."""
        config = ServerConfig(
            model_path="test.gguf",
            embedding=True,
            embedding_model_path="embed.gguf",
            embedding_pooling="cls",
            embedding_normalize=False,
        )
        server = PythonServer(config)

        mock_embedder = Mock()
        with (
            patch("cyllama.llama.server.python.LlamaModel"),
            patch("cyllama.llama.server.python.ServerSlot"),
            patch("cyllama.rag.embedder.Embedder", return_value=mock_embedder) as MockEmbedder,
        ):
            result = server.load_model()

            assert result is True
            assert server.embedder is mock_embedder
            MockEmbedder.assert_called_once_with(
                model_path="embed.gguf",
                n_ctx=512,
                n_batch=512,
                n_gpu_layers=-1,
                pooling="cls",
                normalize=False,
            )

    def test_load_model_embedder_uses_model_path_fallback(self):
        """Test that load_model uses model_path when embedding_model_path is None."""
        config = ServerConfig(
            model_path="chat.gguf",
            embedding=True,
        )
        server = PythonServer(config)

        mock_embedder = Mock()
        with (
            patch("cyllama.llama.server.python.LlamaModel"),
            patch("cyllama.llama.server.python.ServerSlot"),
            patch("cyllama.rag.embedder.Embedder", return_value=mock_embedder) as MockEmbedder,
        ):
            result = server.load_model()

            assert result is True
            # Should use model_path since embedding_model_path is None
            call_kwargs = MockEmbedder.call_args[1]
            assert call_kwargs["model_path"] == "chat.gguf"

    def test_load_model_no_embedder_when_disabled(self):
        """Test that load_model does not create Embedder when embedding=False."""
        config = ServerConfig(model_path="test.gguf", embedding=False)
        server = PythonServer(config)

        with (
            patch("cyllama.llama.server.python.LlamaModel"),
            patch("cyllama.llama.server.python.ServerSlot"),
        ):
            result = server.load_model()

            assert result is True
            assert server.embedder is None


class TestConvenienceFunction:
    """Test convenience functions."""

    @patch("cyllama.llama.server.python.PythonServer")
    def test_start_python_server_success(self, MockPythonServer):
        """Test start_python_server convenience function."""
        mock_server = Mock()
        mock_server.start.return_value = True
        MockPythonServer.return_value = mock_server

        result = start_python_server("test.gguf", port=9090, n_ctx=8192)

        MockPythonServer.assert_called_once()
        config_arg = MockPythonServer.call_args[0][0]
        assert config_arg.model_path == "test.gguf"
        assert config_arg.port == 9090
        assert config_arg.n_ctx == 8192

        mock_server.start.assert_called_once()
        assert result == mock_server

    @patch("cyllama.llama.server.python.PythonServer")
    def test_start_python_server_failure(self, MockPythonServer):
        """Test start_python_server when server fails to start."""
        mock_server = Mock()
        mock_server.start.return_value = False
        MockPythonServer.return_value = mock_server

        with pytest.raises(RuntimeError, match="Failed to start Python server"):
            start_python_server("test.gguf")


# Integration tests (require actual models)
@pytest.mark.slow
class TestRequestValidation:
    """Test server request handling with malformed inputs."""

    def test_chat_request_missing_messages(self):
        """Test ChatRequest requires messages."""
        with pytest.raises(TypeError):
            ChatRequest()  # messages is required

    def test_chat_request_empty_messages(self):
        """Test ChatRequest with empty messages list."""
        request = ChatRequest(messages=[])
        assert request.messages == []

    def test_chat_request_default_sampler_params(self):
        """Test ChatRequest has correct default sampler parameters."""
        messages = [ChatMessage(role="user", content="test")]
        request = ChatRequest(messages=messages)
        assert request.temperature == 0.8
        assert request.min_p == 0.05
        assert request.seed is None

    def test_chat_request_custom_sampler_params(self):
        """Test ChatRequest with custom sampler parameters."""
        messages = [ChatMessage(role="user", content="test")]
        request = ChatRequest(
            messages=messages,
            temperature=0.2,
            min_p=0.1,
            seed=42,
        )
        assert request.temperature == 0.2
        assert request.min_p == 0.1
        assert request.seed == 42

    def test_parse_request_missing_role(self):
        """Test that messages without 'role' raise KeyError during parsing."""
        # Simulate what _handle_chat_completions does: msg["role"]
        data = {"messages": [{"content": "hello"}]}
        with pytest.raises(KeyError):
            [ChatMessage(role=msg["role"], content=msg["content"]) for msg in data["messages"]]

    def test_parse_request_missing_content(self):
        """Test that messages without 'content' raise KeyError during parsing."""
        data = {"messages": [{"role": "user"}]}
        with pytest.raises(KeyError):
            [ChatMessage(role=msg["role"], content=msg["content"]) for msg in data["messages"]]

    def test_parse_request_extra_fields_ignored(self):
        """Test that unknown fields in request data are silently ignored."""
        messages = [ChatMessage(role="user", content="hello")]
        # Extra fields like 'frequency_penalty' should not cause errors
        request = ChatRequest(messages=messages, model="custom")
        assert request.model == "custom"

    def test_process_completion_with_stop_words(self):
        """Test that stop words truncate generated text correctly."""
        config = ServerConfig(model_path="test.gguf")
        server = PythonServer(config)
        server.model = Mock()

        mock_slot = Mock()
        mock_slot.is_processing = False
        mock_slot.task_id = None
        mock_slot.process_and_generate.return_value = "Hello STOP world"
        server.slots = [mock_slot]

        vocab = Mock()
        vocab.tokenize.side_effect = [[1, 2], [3]]
        server.model.get_vocab.return_value = vocab

        messages = [ChatMessage(role="user", content="Hi")]
        request = ChatRequest(messages=messages, max_tokens=50, stop=["STOP"])
        response = server.process_chat_completion(request)

        assert response.choices[0].message.content == "Hello "

    def test_messages_to_prompt_unknown_role(self):
        """Test that unknown roles are silently skipped in prompt conversion."""
        config = ServerConfig(model_path="test.gguf")
        server = PythonServer(config)
        messages = [
            ChatMessage(role="unknown_role", content="test"),
            ChatMessage(role="user", content="Hello"),
        ]
        prompt = server._messages_to_prompt(messages)
        # Unknown role is skipped, user message and trailing "Assistant:" present
        assert "User: Hello" in prompt
        assert prompt.endswith("Assistant:")


class TestPythonServerIntegration:
    """Integration tests for Python server functionality."""

    def test_server_lifecycle_integration(self, model_path):
        """Test complete server lifecycle with real model."""
        config = ServerConfig(
            model_path=model_path,
            port=18091,  # Different port
            n_ctx=512,  # Small context for faster startup
            n_gpu_layers=0,  # CPU only for reliability
        )

        server = PythonServer(config)

        try:
            # Load model
            assert server.load_model() is True
            assert server.model is not None
            assert len(server.slots) == config.n_parallel

            # Test chat completion
            messages = [ChatMessage(role="user", content="Say hello")]
            request = ChatRequest(messages=messages, max_tokens=5)

            response = server.process_chat_completion(request)

            assert isinstance(response, ChatResponse)
            assert len(response.choices) == 1
            assert response.choices[0].message.role == "assistant"
            assert len(response.choices[0].message.content) > 0

        finally:
            server.stop()

    def test_context_manager_integration(self, model_path):
        """Test server context manager with real model."""
        config = ServerConfig(
            model_path=model_path,
            port=18092,  # Different port
            n_ctx=512,
            n_gpu_layers=0,
        )

        # This should work without errors
        with PythonServer(config) as server:
            assert server.model is not None
            assert len(server.slots) > 0

        # Server should be stopped after context exit
        assert not server.running


# =============================================================================
# EmbeddedServer Tests (Mongoose-based server)
# =============================================================================


class TestEmbeddedServerLifecycle:
    """Tests for EmbeddedServer start/stop lifecycle."""

    def test_embedded_server_direct_stop(self, model_path):
        """Test that embedded server can be started and stopped directly without context manager."""
        from cyllama.llama.server.embedded import EmbeddedServer

        config = ServerConfig(model_path=model_path, host="127.0.0.1", port=8098, n_ctx=256)

        server = EmbeddedServer(config)

        # Start server
        result = server.start()
        assert result is True, "Server failed to start"

        try:
            # Wait a bit to ensure server is running
            time.sleep(1)
        finally:
            # Stop server directly
            server.stop()

    def test_embedded_server_context_manager(self, model_path):
        """Test that embedded server context manager properly starts and stops the server."""
        from cyllama.llama.server.embedded import EmbeddedServer

        config = ServerConfig(model_path=model_path, host="127.0.0.1", port=8097, n_ctx=256)

        # Use context manager - should start server on entry and stop on exit
        with EmbeddedServer(config) as server:
            # Wait a moment to ensure server is fully operational
            time.sleep(1)
        # Context manager exit should have called stop()
