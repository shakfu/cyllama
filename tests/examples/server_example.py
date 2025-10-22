#!/usr/bin/env python3
"""
Example usage of the LlamaServer wrapper.

This script demonstrates how to start a llama.cpp server using the Python wrapper
and interact with it via the OpenAI-compatible API.

Requirements:
    pip install requests

Usage:
    python3 examples/server_example.py -m models/Llama-3.2-1B-Instruct-Q8_0.gguf
"""

import argparse
import time
from pathlib import Path

from cyllama.llama.server import ServerConfig, LlamaServer, LlamaServerClient


def main():
    parser = argparse.ArgumentParser(description="Llama.cpp Server Example")
    parser.add_argument("-m", "--model", required=True, help="Path to model file")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--ctx-size", type=int, default=2048, help="Context size")
    parser.add_argument("--gpu-layers", type=int, default=-1, help="GPU layers (-1 for auto)")

    args = parser.parse_args()

    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1

    # Create server configuration
    config = ServerConfig(
        model_path=str(model_path),
        host=args.host,
        port=args.port,
        ctx_size=args.ctx_size,
        n_gpu_layers=args.gpu_layers,
        webui=True,  # Enable web UI
        metrics=True,  # Enable metrics endpoint
    )

    print(f"Starting server with model: {model_path}")
    print(f"Server will be available at: http://{args.host}:{args.port}")
    print(f"Web UI: http://{args.host}:{args.port}")
    print(f"API Base URL: http://{args.host}:{args.port}/v1")

    # Start server using context manager for automatic cleanup
    try:
        with LlamaServer(config) as server:
            print("✓ Server started successfully!")

            # Get server status
            status = server.get_status()
            print(f"✓ Server PID: {status['pid']}")
            print(f"✓ API Ready: {status.get('api_ready', 'unknown')}")

            # Test the API if requests is available
            try:
                client = LlamaServerClient(f"http://{args.host}:{args.port}")

                print("\n--- Testing API ---")

                # Test models endpoint
                print("Testing /v1/models endpoint...")
                models = client.models()
                print(f"✓ Available models: {len(models.get('data', []))}")

                # Test health endpoint
                print("Testing /health endpoint...")
                health = client.health()
                print(f"✓ Health status: {health.get('status', 'unknown')}")

                # Test chat completion
                print("Testing /v1/chat/completions endpoint...")
                messages = [
                    {"role": "user", "content": "Hello! Please respond with just 'Hello there!'"}
                ]

                try:
                    response = client.chat_completion(
                        messages=messages,
                        max_tokens=10,
                        temperature=0.7
                    )

                    if "choices" in response and len(response["choices"]) > 0:
                        content = response["choices"][0]["message"]["content"]
                        print(f"✓ Chat response: {content.strip()}")
                    else:
                        print("✗ Unexpected response format")

                except Exception as e:
                    print(f"✗ Chat completion failed: {e}")

                print("\n--- Server is running ---")
                print("You can now:")
                print(f"1. Open the web UI: http://{args.host}:{args.port}")
                print(f"2. Make API requests to: http://{args.host}:{args.port}/v1/")
                print("3. Press Ctrl+C to stop the server")

                # Keep server running until interrupted
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n\nShutting down server...")

            except ImportError:
                print("\n--- API Testing Skipped ---")
                print("Install 'requests' library to test API endpoints:")
                print("  pip install requests")
                print("\nServer is still running...")
                print(f"Web UI: http://{args.host}:{args.port}")
                print("Press Ctrl+C to stop")

                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n\nShutting down server...")

    except Exception as e:
        print(f"✗ Server failed to start: {e}")
        return 1

    print("✓ Server stopped successfully!")
    return 0


if __name__ == "__main__":
    exit(main())