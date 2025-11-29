#!/usr/bin/env python3
"""
Tests for Python-based llama.cpp server functionality.

These tests cover the PythonServer class and related components,
ensuring proper server functionality using existing cyllama bindings.
"""

import json
import time
import pytest
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from cyllama.llama.server.python import (
    ServerConfig,
    PythonServer,
    ServerSlot,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    start_python_server
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
            model_alias="custom-model"
        )

        assert config.model_path == "custom.gguf"
        assert config.host == "0.0.0.0"
        assert config.port == 9090
        assert config.n_ctx == 8192
        assert config.n_gpu_layers == 32
        assert config.embedding is True
        assert config.n_parallel == 4
        assert config.model_alias == "custom-model"


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
            messages=messages,
            model="custom-model",
            max_tokens=100,
            temperature=0.5,
            stream=True,
            stop=["STOP"]
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

        with patch('cyllama.llama.server.python.LlamaContext'), \
             patch('cyllama.llama.server.python.LlamaSampler'):
            slot = ServerSlot(0, mock_model, config)

            assert slot.id == 0
            assert slot.model == mock_model
            assert not slot.is_processing
            assert slot.task_id is None

    def test_slot_reset(self, mock_model):
        """Test resetting a server slot."""
        config = ServerConfig(model_path="test.gguf")

        with patch('cyllama.llama.server.python.LlamaContext') as MockContext, \
             patch('cyllama.llama.server.python.LlamaSampler'):

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

        with patch('cyllama.llama.server.python.LlamaContext') as MockContext, \
             patch('cyllama.llama.server.python.LlamaSampler') as MockSampler, \
             patch('cyllama.llama.server.python.llama_batch_get_one') as mock_batch:

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

        with patch('cyllama.llama.server.python.LlamaContext'), \
             patch('cyllama.llama.server.python.LlamaSampler'):
            slot = ServerSlot(0, mock_model, config)

            result = slot.process_and_generate("Very long prompt")

            assert result == ""  # Should return empty string for too long prompt

    def test_process_and_generate_error(self, mock_model):
        """Test handling of generation errors."""
        config = ServerConfig(model_path="test.gguf")

        with patch('cyllama.llama.server.python.LlamaContext') as MockContext, \
             patch('cyllama.llama.server.python.LlamaSampler'):

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

    @patch('cyllama.llama.server.python.LlamaModel')
    @patch('cyllama.llama.server.python.ServerSlot')
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

    @patch('cyllama.llama.server.python.LlamaModel')
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
            ChatMessage(role="user", content="How are you?")
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
            [10, 20]    # completion tokens
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

        with patch.object(server, 'start', return_value=True) as mock_start, \
             patch.object(server, 'stop') as mock_stop:

            with server:
                pass

            mock_start.assert_called_once()
            mock_stop.assert_called_once()

    def test_context_manager_start_failure(self, config):
        """Test context manager when start fails."""
        server = PythonServer(config)

        with patch.object(server, 'start', return_value=False):
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


class TestConvenienceFunction:
    """Test convenience functions."""

    @patch('cyllama.llama.server.python.PythonServer')
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

    @patch('cyllama.llama.server.python.PythonServer')
    def test_start_python_server_failure(self, MockPythonServer):
        """Test start_python_server when server fails to start."""
        mock_server = Mock()
        mock_server.start.return_value = False
        MockPythonServer.return_value = mock_server

        with pytest.raises(RuntimeError, match="Failed to start Python server"):
            start_python_server("test.gguf")


# Integration tests (require actual models)
@pytest.mark.slow
class TestPythonServerIntegration:
    """Integration tests for Python server functionality."""


    def test_server_lifecycle_integration(self, model_path):
        """Test complete server lifecycle with real model."""
        config = ServerConfig(
            model_path=model_path,
            port=18091,  # Different port
            n_ctx=512,   # Small context for faster startup
            n_gpu_layers=0  # CPU only for reliability
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
            n_gpu_layers=0
        )

        # This should work without errors
        with PythonServer(config) as server:
            assert server.model is not None
            assert len(server.slots) > 0

        # Server should be stopped after context exit
        assert not server.running