#!/usr/bin/env python3
"""
Example usage of the EmbeddedLlamaServer.

This script demonstrates how to use the embedded server that directly uses
the cyllama bindings without requiring external llama-server binary.

Usage:
    python3 examples/embedded_server_example.py -m models/Llama-3.2-1B-Instruct-Q8_0.gguf
"""

import sys
import os
import argparse
import time
import json
import requests
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cyllama.llama.embedded_server import (
    EmbeddedLlamaServer,
    ServerConfig,
    ChatMessage,
    ChatRequest
)


def test_api_endpoints(base_url: str):
    """Test the various API endpoints."""
    print("\n--- Testing API Endpoints ---")

    try:
        # Test health endpoint
        print("Testing /health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"✓ Health: {response.json()}")

        # Test models endpoint
        print("Testing /v1/models endpoint...")
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        models_data = response.json()
        print(f"✓ Models: {len(models_data['data'])} model(s) available")

        # Test chat completion
        print("Testing /v1/chat/completions endpoint...")
        chat_data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": "Hello! Please respond with just 'Hello there!'"}
            ],
            "max_tokens": 10,
            "temperature": 0.7
        }

        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=chat_data,
            headers={"Content-Type": "application/json"},
            timeout=30  # Generation can take time
        )

        if response.status_code == 200:
            completion_data = response.json()
            content = completion_data["choices"][0]["message"]["content"]
            print(f"✓ Chat completion: '{content.strip()}'")
            print(f"  Usage: {completion_data['usage']}")
        else:
            print(f"✗ Chat completion failed: {response.status_code} - {response.text}")

    except requests.RequestException as e:
        print(f"✗ API test failed: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Embedded Llama.cpp Server Example")
    parser.add_argument("-m", "--model", required=True, help="Path to model file")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--ctx-size", type=int, default=1024, help="Context size")
    parser.add_argument("--gpu-layers", type=int, default=-1, help="GPU layers (-1 for auto)")
    parser.add_argument("--test-api", action="store_true", help="Test API endpoints after starting")

    args = parser.parse_args()

    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1

    print(f"Starting embedded server with model: {model_path}")
    print(f"Server will be available at: http://{args.host}:{args.port}")

    # Create server configuration
    config = ServerConfig(
        model_path=str(model_path),
        host=args.host,
        port=args.port,
        n_ctx=args.ctx_size,
        n_gpu_layers=args.gpu_layers,
        n_parallel=1  # Single slot for simplicity
    )

    # Start server using context manager for automatic cleanup
    try:
        with EmbeddedLlamaServer(config) as server:
            print("✓ Embedded server started successfully!")
            print(f"✓ Model loaded: {server.model is not None}")
            print(f"✓ Available slots: {len(server.slots)}")

            base_url = f"http://{args.host}:{args.port}"

            if args.test_api:
                # Wait a moment for server to be fully ready
                time.sleep(1)
                test_api_endpoints(base_url)

            print(f"\n--- Server Running ---")
            print(f"Base URL: {base_url}")
            print("Available endpoints:")
            print(f"  GET  {base_url}/health")
            print(f"  GET  {base_url}/v1/models")
            print(f"  POST {base_url}/v1/chat/completions")
            print()
            print("Example curl command:")
            print(f"curl -X POST {base_url}/v1/chat/completions \\")
            print('  -H "Content-Type: application/json" \\')
            print('  -d \'{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"Hello!"}],"max_tokens":20}\'')
            print()
            print("Press Ctrl+C to stop the server...")

            # Keep server running until interrupted
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n\nReceived Ctrl+C, shutting down server...")

    except Exception as e:
        print(f"✗ Server failed: {e}")
        return 1

    print("✓ Server stopped successfully!")
    return 0


if __name__ == "__main__":
    exit(main())