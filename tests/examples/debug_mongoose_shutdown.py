#!/usr/bin/env python3
"""
Debug script to test Mongoose server shutdown behavior.

Usage:
    python debug_mongoose_shutdown.py -m models/Llama-3.2-1B-Instruct-Q8_0.gguf
"""

import sys
import time
import logging
import argparse
from pathlib import Path

# Add the src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cyllama.llama.server.python import ServerConfig

def test_mongoose_shutdown(model_path):
    print("Testing Mongoose server shutdown behavior...")

    # Enable debug logging
    logging.basicConfig(level=logging.DEBUG)

    try:
        from cyllama.llama.server.embedded import EmbeddedServer

        config = ServerConfig(
            model_path=model_path,
            host="127.0.0.1",
            port=8099,
            n_ctx=256
        )

        print("Creating MongooseServer...")
        server = EmbeddedServer(config)

        print("Starting server...")
        if not server.start():
            print("Failed to start server")
            return

        print("Server started, waiting 2 seconds...")
        time.sleep(2)

        print("Calling stop()...")
        server.stop()
        print("Stop() completed")

    except ImportError as e:
        print(f"Mongoose server not available: {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug Mongoose shutdown")
    parser.add_argument("-m", "--model", required=True, help="Path to model file")
    args = parser.parse_args()
    test_mongoose_shutdown(args.model)