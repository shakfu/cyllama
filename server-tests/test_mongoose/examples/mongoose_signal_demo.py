#!/usr/bin/env python3
"""
Example demonstrating Ctrl+C signal handling in Mongoose server like the C example.

This example shows how the embedded mongoose server handles SIGINT/SIGTERM signals
for graceful shutdown, following the same pattern as tests/web/http-server/main.c.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from cyllama.llama.server.mongoose_server import MongooseServer
from cyllama.llama.server.embedded import ServerConfig


def main():
    parser = argparse.ArgumentParser(description="Mongoose Signal Handling Demo")
    parser.add_argument("-m", "--model", default="models/Llama-3.2-1B-Instruct-Q8_0.gguf",
                       help="Path to model file")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8090, help="Port to listen on")
    parser.add_argument("--ctx-size", type=int, default=512, help="Context size")
    parser.add_argument("--gpu-layers", type=int, default=-1, help="GPU layers (-1 for auto)")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1

    print("Mongoose Signal Handling Demo")
    print("=============================")
    print(f"Model: {model_path}")
    print(f"Server: http://{args.host}:{args.port}")
    print()

    # Create server configuration
    config = ServerConfig(
        model_path=str(model_path),
        host=args.host,
        port=args.port,
        n_ctx=args.ctx_size,
        n_gpu_layers=args.gpu_layers,
        n_parallel=1  # Single slot for simplicity
    )

    # Create and start server
    try:
        server = MongooseServer(config)

        if not server.start():
            print("✗ Failed to start server")
            return 1

        print("✓ Mongoose server started successfully!")
        print(f"✓ Listening on: http://{args.host}:{args.port}")
        print("✓ Model loaded and slots initialized")
        print()
        print("Available endpoints:")
        print(f"  GET  http://{args.host}:{args.port}/health")
        print(f"  GET  http://{args.host}:{args.port}/v1/models")
        print(f"  POST http://{args.host}:{args.port}/v1/chat/completions")
        print()
        print("Example curl command:")
        print(f"curl -X POST http://{args.host}:{args.port}/v1/chat/completions \\")
        print('  -H "Content-Type: application/json" \\')
        print('  -d \'{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"Hello!"}],"max_tokens":20}\'')
        print()
        print("Press Ctrl+C to stop the server...")
        print()

        # Use the mongoose server's signal-based wait pattern (like the C example)
        # This will block until SIGINT/SIGTERM is received
        server.wait_for_shutdown()

    except Exception as e:
        print(f"✗ Server error: {e}")
        return 1
    finally:
        # Ensure cleanup
        try:
            server.stop()
            print("✓ Server stopped successfully!")
        except:
            pass

    return 0


if __name__ == "__main__":
    exit(main())