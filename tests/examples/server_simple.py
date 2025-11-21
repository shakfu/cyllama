#!/usr/bin/env python3
"""
Simple server example using relative imports.

Run from the project root directory:
    python3 examples/server_simple.py -m models/Llama-3.2-1B-Instruct-Q8_0.gguf
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cyllama.llama.server import ServerConfig, start_server


def main():
    parser = argparse.ArgumentParser(description="Simple Llama.cpp Server")
    parser.add_argument("-m", "--model", required=True, help="Path to model file")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")

    args = parser.parse_args()

    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1

    print(f"Starting server with model: {model_path}")
    print(f"Server URL: http://127.0.0.1:{args.port}")

    try:
        # Start server (convenience function)
        server = start_server(
            model_path=str(model_path),
            port=args.port,
            ctx_size=1024,
            n_gpu_layers=-1  # Auto-detect
        )

        print("✓ Server started! Press Ctrl+C to stop...")

        # Keep running
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping server...")
            server.stop()
            print("✓ Server stopped")

    except Exception as e:
        print(f"✗ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())