"""Tests for the Mongoose-based HTTP server."""

import pytest
import time
import json
import threading
from unittest.mock import patch, MagicMock

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Skip tests if Mongoose server is not available
try:
    from cyllama.llama.server.mongoose_server import MongooseServer, start_mongoose_server
    from cyllama.llama.server.embedded import ServerConfig
    MONGOOSE_AVAILABLE = True
except ImportError:
    MONGOOSE_AVAILABLE = False


@pytest.mark.skipif(not MONGOOSE_AVAILABLE, reason="Mongoose server not available")
class TestMongooseServer:
    """Test cases for MongooseServer class."""

    def test_server_creation(self):
        """Test creating a MongooseServer instance."""
        config = ServerConfig(model_path="test_model.gguf")
        server = MongooseServer(config)

        assert server._config == config
        assert server._model is None
        assert server._slots == []
        assert not server._running

    @patch('cyllama.llama.server.mongoose_server.LlamaModel')
    def test_load_model_success(self, mock_model_class):
        """Test successful model loading."""
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        config = ServerConfig(model_path="test_model.gguf", n_parallel=2)
        server = MongooseServer(config)

        result = server.load_model()

        assert result is True
        assert server._model == mock_model
        assert len(server._slots) == 2
        mock_model_class.assert_called_once_with(path_model="test_model.gguf")

    @patch('cyllama.llama.server.mongoose_server.LlamaModel')
    def test_load_model_failure(self, mock_model_class):
        """Test model loading failure."""
        mock_model_class.side_effect = Exception("Model not found")

        config = ServerConfig(model_path="nonexistent_model.gguf")
        server = MongooseServer(config)

        result = server.load_model()

        assert result is False
        assert server._model is None
        assert len(server._slots) == 0

    @patch('cyllama.llama.server.mongoose_server.LlamaModel')
    def test_get_available_slot(self, mock_model_class):
        """Test getting available slots."""
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        config = ServerConfig(model_path="test_model.gguf", n_parallel=2)
        server = MongooseServer(config)
        server.load_model()

        # Initially all slots should be available
        slot = server.get_available_slot()
        assert slot is not None
        assert not slot.is_processing

        # Mark slot as processing
        slot.is_processing = True

        # Should return the other slot
        slot2 = server.get_available_slot()
        assert slot2 is not None
        assert slot2 != slot

        # Mark second slot as processing
        slot2.is_processing = True

        # No slots available
        slot3 = server.get_available_slot()
        assert slot3 is None

    def test_messages_to_prompt(self):
        """Test converting OpenAI messages to prompt string."""
        config = ServerConfig(model_path="test_model.gguf")
        server = MongooseServer(config)

        from cyllama.llama.server.embedded import ChatMessage

        messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="What is 2+2?"),
        ]

        prompt = server._messages_to_prompt(messages)
        expected = "System: You are a helpful assistant.\nUser: What is 2+2?\nAssistant:"

        assert prompt == expected

    @patch('cyllama.llama.server.mongoose_server.LlamaModel')
    def test_handle_http_request_health(self, mock_model_class):
        """Test health endpoint."""
        config = ServerConfig(model_path="test_model.gguf")
        server = MongooseServer(config)

        # Mock connection
        mock_conn = MagicMock()

        server.handle_http_request(mock_conn, "GET", "/health", {}, "")

        mock_conn.send_json.assert_called_once_with({"status": "ok"})

    @patch('cyllama.llama.server.mongoose_server.LlamaModel')
    def test_handle_http_request_models(self, mock_model_class):
        """Test models endpoint."""
        config = ServerConfig(model_path="test_model.gguf", model_alias="test-model")
        server = MongooseServer(config)

        # Mock connection
        mock_conn = MagicMock()

        server.handle_http_request(mock_conn, "GET", "/v1/models", {}, "")

        # Verify models response structure
        args, kwargs = mock_conn.send_json.call_args
        response_data = args[0]

        assert response_data["object"] == "list"
        assert len(response_data["data"]) == 1
        assert response_data["data"][0]["id"] == "test-model"
        assert response_data["data"][0]["object"] == "model"

    @patch('cyllama.llama.server.mongoose_server.LlamaModel')
    def test_handle_http_request_not_found(self, mock_model_class):
        """Test 404 handling."""
        config = ServerConfig(model_path="test_model.gguf")
        server = MongooseServer(config)

        # Mock connection
        mock_conn = MagicMock()

        server.handle_http_request(mock_conn, "GET", "/unknown", {}, "")

        mock_conn.send_error.assert_called_once_with(404, "Not Found")

    def test_start_mongoose_server_function(self):
        """Test the convenience function."""
        with patch('cyllama.llama.server.mongoose_server.MongooseServer') as mock_server_class:
            mock_server = MagicMock()
            mock_server.start.return_value = True
            mock_server_class.return_value = mock_server

            result = start_mongoose_server("test_model.gguf", host="0.0.0.0", port=8081)

            assert result == mock_server
            mock_server.start.assert_called_once()

    def test_start_mongoose_server_function_failure(self):
        """Test the convenience function with failure."""
        with patch('cyllama.llama.server.mongoose_server.MongooseServer') as mock_server_class:
            mock_server = MagicMock()
            mock_server.start.return_value = False
            mock_server_class.return_value = mock_server

            with pytest.raises(RuntimeError, match="Failed to start Mongoose server"):
                start_mongoose_server("test_model.gguf")


@pytest.mark.skipif(not MONGOOSE_AVAILABLE, reason="Mongoose server not available")
@pytest.mark.skipif(not REQUESTS_AVAILABLE, reason="requests library not available")
class TestMongooseServerIntegration:
    """Integration tests for MongooseServer with real HTTP requests."""

    @pytest.fixture
    def mock_server(self):
        """Create a mocked server for integration tests."""
        with patch('cyllama.llama.server.mongoose_server.LlamaModel'):
            config = ServerConfig(
                model_path="test_model.gguf",
                host="127.0.0.1",
                port=8089,  # Use different port to avoid conflicts
                n_parallel=1
            )
            server = MongooseServer(config)
            yield server

    def test_server_lifecycle(self, mock_server):
        """Test server start and stop lifecycle."""
        # Server should start successfully (mocked)
        with patch.object(mock_server, 'load_model', return_value=True):
            with patch.object(mock_server, '_server_loop'):
                assert mock_server.start() is True
                assert mock_server._running is True

                # Stop the server
                mock_server.stop()
                assert mock_server._running is False


@pytest.mark.skipif(not MONGOOSE_AVAILABLE, reason="Mongoose server not available")
class TestMongooseConnection:
    """Test cases for MongooseConnection wrapper."""

    def test_connection_creation(self):
        """Test creating a MongooseConnection."""
        from cyllama.llama.server.mongoose_server import MongooseConnection

        conn = MongooseConnection()
        assert not conn.is_valid
        assert conn._conn is None

    def test_send_json(self):
        """Test sending JSON response."""
        from cyllama.llama.server.mongoose_server import MongooseConnection

        conn = MongooseConnection()
        # Without a valid connection, should return False
        result = conn.send_json({"test": "data"})
        assert result is False

    def test_send_error(self):
        """Test sending error response."""
        from cyllama.llama.server.mongoose_server import MongooseConnection

        conn = MongooseConnection()
        with patch.object(conn, 'send_json') as mock_send_json:
            conn.send_error(400, "Bad Request")

            expected_data = {
                "error": {
                    "type": "invalid_request_error",
                    "message": "Bad Request"
                }
            }
            mock_send_json.assert_called_once_with(expected_data, 400)


@pytest.mark.skipif(not MONGOOSE_AVAILABLE, reason="Mongoose server not available")
class TestMongooseServerConfigIntegration:
    """Test MongooseServer with different configurations."""

    def test_different_configs(self):
        """Test server with different configuration options."""
        configs = [
            ServerConfig(model_path="test1.gguf", port=8090, n_parallel=1),
            ServerConfig(model_path="test2.gguf", port=8091, n_parallel=2, model_alias="custom-model"),
            ServerConfig(model_path="test3.gguf", port=8092, host="0.0.0.0"),
        ]

        for config in configs:
            server = MongooseServer(config)
            assert server._config == config
            assert server._config.model_path == config.model_path
            assert server._config.port == config.port
            assert server._config.n_parallel == config.n_parallel


if __name__ == "__main__":
    pytest.main([__file__])