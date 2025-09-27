#!/usr/bin/env python3
"""
Tests for llama.cpp server wrapper functionality.

These tests cover the LlamaServer class, ServerConfig, and LlamaServerClient,
ensuring proper server lifecycle management and API functionality.
"""

import time
import pytest
import subprocess
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from cyllama.llama.server.launcher import (
    ServerConfig,
    LlamaServer,
    LlamaServerClient,
    start_server
)


class TestServerConfig:
    """Test ServerConfig class functionality."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ServerConfig(model_path="test.gguf")

        assert config.model_path == "test.gguf"
        assert config.host == "127.0.0.1"
        assert config.port == 8080
        assert config.ctx_size == 4096
        assert config.batch_size == 2048
        assert config.threads == -1
        assert config.n_gpu_layers == -1
        assert config.embedding is False
        assert config.webui is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ServerConfig(
            model_path="custom.gguf",
            host="0.0.0.0",
            port=9090,
            ctx_size=8192,
            n_gpu_layers=32,
            embedding=True,
            webui=False
        )

        assert config.model_path == "custom.gguf"
        assert config.host == "0.0.0.0"
        assert config.port == 9090
        assert config.ctx_size == 8192
        assert config.n_gpu_layers == 32
        assert config.embedding is True
        assert config.webui is False

    def test_config_to_args(self):
        """Test conversion of config to command line arguments."""
        config = ServerConfig(
            model_path="test.gguf",
            host="127.0.0.1",
            port=8080,
            ctx_size=4096,
            n_gpu_layers=10,
            embedding=True,
            webui=False,
            extra_args=["--verbose"]
        )

        args = config.to_args()

        # Check required arguments
        assert "-m" in args
        assert "test.gguf" in args
        assert "--host" in args
        assert "127.0.0.1" in args
        assert "--port" in args
        assert "8080" in args
        assert "-c" in args
        assert "4096" in args
        assert "-ngl" in args
        assert "10" in args
        assert "--embedding" in args
        assert "--no-webui" in args
        assert "--verbose" in args

    def test_tensor_split_args(self):
        """Test tensor split configuration."""
        config = ServerConfig(
            model_path="test.gguf",
            tensor_split=[3.0, 1.0]
        )

        args = config.to_args()

        assert "--tensor-split" in args
        tensor_split_idx = args.index("--tensor-split")
        assert args[tensor_split_idx + 1] == "3.0,1.0"

    def test_security_args(self):
        """Test security-related configuration."""
        config = ServerConfig(
            model_path="test.gguf",
            api_key="secret",
            ssl_cert_file="cert.pem",
            ssl_key_file="key.pem"
        )

        args = config.to_args()

        assert "--api-key" in args
        assert "secret" in args
        assert "--ssl-cert-file" in args
        assert "cert.pem" in args
        assert "--ssl-key-file" in args
        assert "key.pem" in args


class TestLlamaServer:
    """Test LlamaServer class functionality."""

    def test_init_with_auto_binary_detection(self):
        """Test server initialization with automatic binary detection."""
        config = ServerConfig(model_path="test.gguf")

        # Mock the binary detection
        with patch.object(LlamaServer, '_find_server_binary') as mock_find:
            mock_find.return_value = Path("/fake/llama-server")

            with patch.object(Path, 'exists', return_value=True):
                server = LlamaServer(config)
                assert server.server_binary == Path("/fake/llama-server")

    def test_init_with_custom_binary(self):
        """Test server initialization with custom binary path."""
        config = ServerConfig(model_path="test.gguf")

        with patch.object(Path, 'exists', return_value=True):
            server = LlamaServer(config, server_binary="/custom/llama-server")
            assert server.server_binary == Path("/custom/llama-server")

    def test_init_binary_not_found(self):
        """Test server initialization when binary is not found."""
        config = ServerConfig(model_path="test.gguf")

        with patch.object(Path, 'exists', return_value=False):
            with pytest.raises(FileNotFoundError, match="Server binary not found"):
                LlamaServer(config, server_binary="/nonexistent/llama-server")

    def test_find_server_binary_candidates(self):
        """Test server binary discovery logic."""
        config = ServerConfig(model_path="test.gguf")

        with patch.object(LlamaServer, '_find_server_binary') as mock_find:
            mock_find.return_value = Path("/found/llama-server")

            with patch.object(Path, 'exists', return_value=True):
                server = LlamaServer(config)
                assert mock_find.called

    @patch('subprocess.Popen')
    @patch.object(Path, 'exists', return_value=True)
    def test_start_server_success(self, mock_exists, mock_popen):
        """Test successful server start."""
        config = ServerConfig(model_path="test.gguf")
        mock_process = Mock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        with patch.object(LlamaServer, '_find_server_binary') as mock_find:
            mock_find.return_value = Path("/fake/llama-server")
            server = LlamaServer(config)

            with patch.object(server, 'wait_for_ready', return_value=True):
                server.start()

                assert server.process == mock_process
                mock_popen.assert_called_once()

    @patch('subprocess.Popen')
    @patch.object(Path, 'exists')
    def test_start_server_model_not_found(self, mock_exists, mock_popen):
        """Test server start when model file doesn't exist."""
        config = ServerConfig(model_path="nonexistent.gguf")

        # First call (binary check) returns True, second call (model check) returns False
        mock_exists.side_effect = [True, False]

        with patch.object(LlamaServer, '_find_server_binary') as mock_find:
            mock_find.return_value = Path("/fake/llama-server")

            server = LlamaServer(config)

            with pytest.raises(FileNotFoundError, match="Model not found"):
                server.start()

    @patch('subprocess.Popen')
    @patch.object(Path, 'exists', return_value=True)
    def test_start_server_already_running(self, mock_exists, mock_popen):
        """Test starting server when already running."""
        config = ServerConfig(model_path="test.gguf")

        with patch.object(LlamaServer, '_find_server_binary') as mock_find:
            mock_find.return_value = Path("/fake/llama-server")
            server = LlamaServer(config)
            server.process = Mock()  # Simulate running process

            with patch.object(server, 'is_running', return_value=True):
                with pytest.raises(RuntimeError, match="already running"):
                    server.start()

    def test_stop_server_graceful(self):
        """Test graceful server stop."""
        config = ServerConfig(model_path="test.gguf")

        with patch.object(LlamaServer, '_find_server_binary') as mock_find:
            mock_find.return_value = Path("/fake/llama-server")
            with patch.object(Path, 'exists', return_value=True):
                server = LlamaServer(config)

                mock_process = Mock()
                mock_process.poll.return_value = None  # Running
                server.process = mock_process

                server.stop()

                mock_process.terminate.assert_called_once()
                mock_process.wait.assert_called_once()
                assert server.process is None

    def test_stop_server_force(self):
        """Test forced server stop when graceful fails."""
        config = ServerConfig(model_path="test.gguf")

        with patch.object(LlamaServer, '_find_server_binary') as mock_find:
            mock_find.return_value = Path("/fake/llama-server")
            with patch.object(Path, 'exists', return_value=True):
                server = LlamaServer(config)

                mock_process = Mock()
                mock_process.poll.return_value = None  # Running
                mock_process.wait.side_effect = [subprocess.TimeoutExpired("cmd", 10), None]
                server.process = mock_process

                server.stop()

                mock_process.terminate.assert_called_once()
                mock_process.kill.assert_called_once()
                assert server.process is None

    def test_is_running_true(self):
        """Test is_running when server is running."""
        config = ServerConfig(model_path="test.gguf")

        with patch.object(LlamaServer, '_find_server_binary') as mock_find:
            mock_find.return_value = Path("/fake/llama-server")
            with patch.object(Path, 'exists', return_value=True):
                server = LlamaServer(config)

                mock_process = Mock()
                mock_process.poll.return_value = None  # Running
                server.process = mock_process

                assert server.is_running() is True

    def test_is_running_false(self):
        """Test is_running when server is not running."""
        config = ServerConfig(model_path="test.gguf")

        with patch.object(LlamaServer, '_find_server_binary') as mock_find:
            mock_find.return_value = Path("/fake/llama-server")
            with patch.object(Path, 'exists', return_value=True):
                server = LlamaServer(config)

                # No process
                assert server.is_running() is False

                # Stopped process
                mock_process = Mock()
                mock_process.poll.return_value = 0  # Stopped
                server.process = mock_process

                assert server.is_running() is False

    def test_wait_for_ready_success(self):
        """Test waiting for server to be ready - success case."""
        config = ServerConfig(model_path="test.gguf")

        with patch.object(LlamaServer, '_find_server_binary') as mock_find:
            mock_find.return_value = Path("/fake/llama-server")
            with patch.object(Path, 'exists', return_value=True):
                server = LlamaServer(config)

                try:
                    import requests
                    # Mock successful health check
                    with patch('requests.get') as mock_get:
                        mock_response = Mock()
                        mock_response.status_code = 200
                        mock_get.return_value = mock_response

                        with patch.object(server, 'is_running', return_value=True):
                            result = server.wait_for_ready(timeout=1.0)
                            assert result is True
                except ImportError:
                    pytest.skip("requests not available")

    def test_wait_for_ready_timeout(self):
        """Test waiting for server to be ready - timeout case."""
        config = ServerConfig(model_path="test.gguf")

        with patch.object(LlamaServer, '_find_server_binary') as mock_find:
            mock_find.return_value = Path("/fake/llama-server")
            with patch.object(Path, 'exists', return_value=True):
                server = LlamaServer(config)

                try:
                    import requests
                    # Mock failed health check
                    with patch('requests.get') as mock_get:
                        mock_get.side_effect = requests.RequestException("Connection failed")

                        with patch.object(server, 'is_running', return_value=True):
                            result = server.wait_for_ready(timeout=0.1)
                            assert result is False
                except ImportError:
                    pytest.skip("requests not available")

    def test_context_manager(self):
        """Test server as context manager."""
        config = ServerConfig(model_path="test.gguf")

        with patch.object(LlamaServer, '_find_server_binary') as mock_find:
            mock_find.return_value = Path("/fake/llama-server")
            with patch.object(Path, 'exists', return_value=True):
                server = LlamaServer(config)

                with patch.object(server, 'start') as mock_start:
                    with patch.object(server, 'stop') as mock_stop:
                        with server:
                            pass

                        mock_start.assert_called_once()
                        mock_stop.assert_called_once()


class TestLlamaServerClient:
    """Test LlamaServerClient class functionality."""

    @pytest.fixture(autouse=True)
    def check_requests(self):
        """Fixture to skip tests if requests is not available."""
        try:
            import requests
        except ImportError:
            pytest.skip("requests library required for client tests")

    @patch('requests.Session')
    def test_client_init_no_auth(self, mock_session_class):
        """Test client initialization without authentication."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        client = LlamaServerClient("http://localhost:8080")

        assert client.base_url == "http://localhost:8080"
        assert client.api_key is None
        mock_session.headers.update.assert_called()

    @patch('requests.Session')
    def test_client_init_with_auth(self, mock_session_class):
        """Test client initialization with authentication."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        client = LlamaServerClient("http://localhost:8080", api_key="secret")

        assert client.api_key == "secret"
        # Should be called twice - once for auth, once for content-type
        assert mock_session.headers.update.call_count == 2

    @patch('requests.Session')
    def test_chat_completion(self, mock_session_class):
        """Test chat completion request."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hello!"}}]}
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session

        client = LlamaServerClient()
        messages = [{"role": "user", "content": "Hello"}]

        result = client.chat_completion(messages, temperature=0.7)

        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert call_args[0][0].endswith("/v1/chat/completions")
        assert call_args[1]["json"]["messages"] == messages
        assert call_args[1]["json"]["temperature"] == 0.7

    @patch('requests.Session')
    def test_embedding(self, mock_session_class):
        """Test embedding request."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session

        client = LlamaServerClient()

        result = client.embedding("Hello world")

        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert call_args[0][0].endswith("/v1/embeddings")
        assert call_args[1]["json"]["input"] == "Hello world"

    @patch('requests.Session')
    def test_models(self, mock_session_class):
        """Test models request."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"data": [{"id": "model1"}]}
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        client = LlamaServerClient()

        result = client.models()

        mock_session.get.assert_called_once()
        call_args = mock_session.get.call_args
        assert call_args[0][0].endswith("/v1/models")

    @patch('requests.Session')
    def test_health(self, mock_session_class):
        """Test health check request."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"status": "ok"}
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        client = LlamaServerClient()

        result = client.health()

        mock_session.get.assert_called_once()
        call_args = mock_session.get.call_args
        assert call_args[0][0].endswith("/health")


class TestConvenienceFunctions:
    """Test convenience functions."""

    @patch('cyllama.llama.server.launcher.LlamaServer')
    def test_start_server_function(self, mock_server_class):
        """Test start_server convenience function."""
        mock_server = Mock()
        mock_server_class.return_value = mock_server

        result = start_server("test.gguf", port=9090, ctx_size=8192)

        mock_server_class.assert_called_once()
        config_arg = mock_server_class.call_args[0][0]
        assert config_arg.model_path == "test.gguf"
        assert config_arg.port == 9090
        assert config_arg.ctx_size == 8192

        mock_server.start.assert_called_once()
        assert result == mock_server


# Integration tests (these require actual models and should be marked as slow)
@pytest.mark.slow
class TestServerIntegration:
    """Integration tests for server functionality."""

    @pytest.fixture
    def model_path(self):
        """Fixture providing path to test model."""
        # This would use the same model fixture as other tests
        model_file = Path("models/Llama-3.2-1B-Instruct-Q8_0.gguf")
        if not model_file.exists():
            pytest.skip("Test model not available")
        return str(model_file)

    @pytest.fixture(autouse=True)
    def check_requests(self):
        """Fixture to skip tests if requests is not available."""
        try:
            import requests
        except ImportError:
            pytest.skip("requests library required for integration tests")

    def test_server_lifecycle_integration(self, model_path):
        """Test complete server lifecycle with real binary."""
        config = ServerConfig(
            model_path=model_path,
            port=18080,  # Use different port to avoid conflicts
            ctx_size=512,  # Small context for faster startup
            n_gpu_layers=0  # CPU only for reliability
        )

        server = LlamaServer(config)

        try:
            # Start server
            server.start(timeout=60.0)  # Longer timeout for model loading
            assert server.is_running()

            # Check status
            status = server.get_status()
            assert status["running"] is True
            assert status["pid"] is not None

            # Test client connection
            client = LlamaServerClient(f"http://127.0.0.1:18080")

            # Simple health check
            health = client.health()
            assert "status" in health

            # Test models endpoint
            models = client.models()
            assert "data" in models

        finally:
            # Always stop server
            server.stop()
            assert not server.is_running()

    def test_server_context_manager_integration(self, model_path):
        """Test server context manager with real binary."""
        config = ServerConfig(
            model_path=model_path,
            port=18081,  # Different port
            ctx_size=512,
            n_gpu_layers=0
        )

        with LlamaServer(config) as server:
            assert server.is_running()

            # Quick API test
            client = LlamaServerClient(f"http://127.0.0.1:18081")
            health = client.health()
            assert "status" in health

        # Server should be stopped after context exit
        assert not server.is_running()