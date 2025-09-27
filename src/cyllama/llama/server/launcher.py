#!/usr/bin/env python3
"""
Llama.cpp Server Wrapper

This module provides a Python wrapper around the llama.cpp server binary,
enabling easy management of OpenAI-compatible API endpoints for LLM inference.

The wrapper manages the llama-server subprocess and provides a convenient
interface for starting, stopping, and monitoring the server.
"""

import os
import time
import json
import signal
import logging
import subprocess
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field


@dataclass
class ServerConfig:
    """Configuration for the Llama server."""

    # Model configuration
    model_path: str

    # Server configuration
    host: str = "127.0.0.1"
    port: int = 8080
    api_prefix: str = ""

    # Performance configuration
    ctx_size: int = 4096
    batch_size: int = 2048
    threads: int = -1
    threads_batch: Optional[int] = None
    threads_http: int = -1

    # GPU configuration
    n_gpu_layers: int = -1
    main_gpu: int = 0
    tensor_split: Optional[List[float]] = None

    # Server features
    embedding: bool = False
    reranking: bool = False
    metrics: bool = False
    webui: bool = True
    slots: bool = True

    # Security
    api_key: Optional[str] = None
    api_key_file: Optional[str] = None
    ssl_key_file: Optional[str] = None
    ssl_cert_file: Optional[str] = None

    # Timeouts and limits
    timeout: int = 600
    cache_reuse: int = 0

    # Additional arguments
    extra_args: List[str] = field(default_factory=list)

    def to_args(self) -> List[str]:
        """Convert configuration to command line arguments."""
        args = []

        # Model
        args.extend(["-m", str(self.model_path)])

        # Server
        args.extend(["--host", self.host])
        args.extend(["--port", str(self.port)])
        if self.api_prefix:
            args.extend(["--api-prefix", self.api_prefix])

        # Performance
        args.extend(["-c", str(self.ctx_size)])
        args.extend(["-b", str(self.batch_size)])
        args.extend(["-t", str(self.threads)])
        if self.threads_batch is not None:
            args.extend(["-tb", str(self.threads_batch)])
        args.extend(["--threads-http", str(self.threads_http)])

        # GPU
        if self.n_gpu_layers >= 0:
            args.extend(["-ngl", str(self.n_gpu_layers)])
        args.extend(["--main-gpu", str(self.main_gpu)])
        if self.tensor_split:
            args.extend(["--tensor-split", ",".join(map(str, self.tensor_split))])

        # Features
        if self.embedding:
            args.append("--embedding")
        if self.reranking:
            args.append("--reranking")
        if self.metrics:
            args.append("--metrics")
        if not self.webui:
            args.append("--no-webui")
        if not self.slots:
            args.append("--no-slots")

        # Security
        if self.api_key:
            args.extend(["--api-key", self.api_key])
        if self.api_key_file:
            args.extend(["--api-key-file", str(self.api_key_file)])
        if self.ssl_key_file:
            args.extend(["--ssl-key-file", str(self.ssl_key_file)])
        if self.ssl_cert_file:
            args.extend(["--ssl-cert-file", str(self.ssl_cert_file)])

        # Timeouts
        args.extend(["--timeout", str(self.timeout)])
        args.extend(["--cache-reuse", str(self.cache_reuse)])

        # Extra arguments
        args.extend(self.extra_args)

        return args


class LlamaServer:
    """
    Wrapper for the llama.cpp server binary.

    Provides easy management of the llama-server subprocess with OpenAI-compatible
    API endpoints for LLM inference, embeddings, and other features.

    Example:
        config = ServerConfig(model_path="models/model.gguf", port=8080)
        server = LlamaServer(config)
        server.start()
        # Server is now running at http://127.0.0.1:8080
        server.stop()
    """

    def __init__(self, config: ServerConfig, server_binary: Optional[str] = None):
        """
        Initialize the server wrapper.

        Args:
            config: Server configuration
            server_binary: Path to llama-server binary (auto-detected if None)
        """
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self._shutdown_event = threading.Event()

        # Find server binary
        if server_binary:
            self.server_binary = Path(server_binary)
        else:
            self.server_binary = self._find_server_binary()

        if not self.server_binary.exists():
            raise FileNotFoundError(f"Server binary not found: {self.server_binary}")

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def _find_server_binary(self) -> Path:
        """Find the llama-server binary."""
        # Try relative to this file
        cyllama_root = Path(__file__).parent.parent.parent.parent
        candidates = [
            cyllama_root / "thirdparty" / "llama.cpp" / "bin" / "llama-server",
            cyllama_root / "build" / "llama.cpp" / "bin" / "llama-server",
            Path("llama-server"),  # In PATH
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError("Could not find llama-server binary")

    def start(self, wait_for_ready: bool = True, timeout: float = 30.0) -> None:
        """
        Start the server.

        Args:
            wait_for_ready: Wait for server to be ready before returning
            timeout: Maximum time to wait for server to be ready
        """
        if self.is_running():
            raise RuntimeError("Server is already running")

        # Validate model exists
        if not Path(self.config.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.config.model_path}")

        # Build command
        cmd = [str(self.server_binary)] + self.config.to_args()

        self.logger.info(f"Starting server: {' '.join(cmd)}")

        # Start process
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start server: {e}")

        # Wait for ready if requested
        if wait_for_ready:
            if not self.wait_for_ready(timeout):
                self.stop()
                raise RuntimeError(f"Server failed to start within {timeout} seconds")

        self.logger.info(f"Server started successfully (PID: {self.process.pid})")

    def stop(self, timeout: float = 10.0) -> None:
        """
        Stop the server.

        Args:
            timeout: Maximum time to wait for graceful shutdown
        """
        if not self.process:
            return

        self.logger.info("Stopping server...")

        # Signal shutdown
        self._shutdown_event.set()

        # Try graceful shutdown first
        if self.process.poll() is None:
            self.process.terminate()

            try:
                self.process.wait(timeout=timeout)
                self.logger.info("Server stopped gracefully")
            except subprocess.TimeoutExpired:
                self.logger.warning("Server did not stop gracefully, forcing shutdown...")
                self.process.kill()
                self.process.wait()
                self.logger.info("Server forcefully stopped")

        self.process = None

    def restart(self, timeout: float = 30.0) -> None:
        """
        Restart the server.

        Args:
            timeout: Maximum time to wait for restart
        """
        self.stop()
        time.sleep(1)  # Brief pause
        self.start(wait_for_ready=True, timeout=timeout)

    def is_running(self) -> bool:
        """Check if the server is running."""
        return self.process is not None and self.process.poll() is None

    def wait_for_ready(self, timeout: float = 30.0) -> bool:
        """
        Wait for the server to be ready to accept requests.

        Args:
            timeout: Maximum time to wait

        Returns:
            True if server is ready, False if timeout
        """
        try:
            import requests
        except ImportError:
            self.logger.warning("requests not available, cannot check server readiness")
            time.sleep(2)  # Brief wait instead
            return self.is_running()

        start_time = time.time()
        url = f"http://{self.config.host}:{self.config.port}{self.config.api_prefix}/health"

        while time.time() - start_time < timeout:
            if not self.is_running():
                return False

            try:
                response = requests.get(url, timeout=1.0)
                if response.status_code == 200:
                    return True
            except requests.RequestException:
                pass

            time.sleep(0.5)

        return False

    def get_status(self) -> Dict[str, Any]:
        """Get server status information."""
        status = {
            "running": self.is_running(),
            "pid": self.process.pid if self.process else None,
            "config": {
                "host": self.config.host,
                "port": self.config.port,
                "model_path": self.config.model_path,
            }
        }

        if self.is_running():
            try:
                import requests
                url = f"http://{self.config.host}:{self.config.port}{self.config.api_prefix}/v1/models"
                response = requests.get(url, timeout=2.0)
                if response.status_code == 200:
                    status["api_ready"] = True
                    status["models"] = response.json()
                else:
                    status["api_ready"] = False
            except ImportError:
                status["api_ready"] = None  # Cannot check without requests
            except Exception:
                status["api_ready"] = False

        return status

    def get_logs(self, lines: int = 50) -> Dict[str, List[str]]:
        """
        Get recent server logs.

        Args:
            lines: Number of recent lines to return

        Returns:
            Dictionary with 'stdout' and 'stderr' logs
        """
        if not self.process:
            return {"stdout": [], "stderr": []}

        # Note: This is a simplified implementation
        # In practice, you might want to capture logs to files
        return {
            "stdout": ["Logs would be captured here"],
            "stderr": ["Error logs would be captured here"]
        }

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class LlamaServerClient:
    """
    Simple client for making requests to a local LlamaServer instance.

    Provides convenient methods for common operations like chat completions
    and embeddings that work with the OpenAI-compatible API.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8080", api_key: Optional[str] = None):
        """
        Initialize the client.

        Args:
            base_url: Base URL of the server
            api_key: API key for authentication (if required)
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key

        # Setup session
        try:
            import requests
            self.session = requests.Session()
            if api_key:
                self.session.headers.update({"Authorization": f"Bearer {api_key}"})
            self.session.headers.update({"Content-Type": "application/json"})
        except ImportError:
            raise ImportError("requests library is required for LlamaServerClient. Install with: pip install requests")

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Create a chat completion.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Chat completion response
        """
        url = f"{self.base_url}/v1/chat/completions"
        data = {
            "messages": messages,
            **kwargs
        }

        response = self.session.post(url, json=data)
        response.raise_for_status()
        return response.json()

    def embedding(self, input_text: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """
        Create embeddings.

        Args:
            input_text: Text or list of texts to embed
            **kwargs: Additional parameters

        Returns:
            Embedding response
        """
        url = f"{self.base_url}/v1/embeddings"
        data = {
            "input": input_text,
            **kwargs
        }

        response = self.session.post(url, json=data)
        response.raise_for_status()
        return response.json()

    def models(self) -> Dict[str, Any]:
        """Get available models."""
        url = f"{self.base_url}/v1/models"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def health(self) -> Dict[str, Any]:
        """Check server health."""
        url = f"{self.base_url}/health"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()


# Convenience functions
def start_server(model_path: str, **kwargs) -> LlamaServer:
    """
    Start a server with simple configuration.

    Args:
        model_path: Path to the model file
        **kwargs: Additional configuration parameters

    Returns:
        Started LlamaServer instance
    """
    config = ServerConfig(model_path=model_path, **kwargs)
    server = LlamaServer(config)
    server.start()
    return server


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Llama.cpp Server Wrapper")
    parser.add_argument("-m", "--model", required=True, help="Path to model file")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--ctx-size", type=int, default=4096, help="Context size")
    parser.add_argument("--gpu-layers", type=int, default=-1, help="GPU layers")

    args = parser.parse_args()

    config = ServerConfig(
        model_path=args.model,
        host=args.host,
        port=args.port,
        ctx_size=args.ctx_size,
        n_gpu_layers=args.gpu_layers
    )

    logging.basicConfig(level=logging.INFO)

    with LlamaServer(config) as server:
        print(f"Server running at http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down server...")