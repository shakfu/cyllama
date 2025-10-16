#!/usr/bin/env python3
"""
Debug script to test Mongoose server shutdown behavior.
"""

import sys
import time
import logging
from pathlib import Path

# Add the src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cyllama.llama.server.embedded import ServerConfig

def test_mongoose_shutdown():
    print("Testing Mongoose server shutdown behavior...")

    # Enable debug logging
    logging.basicConfig(level=logging.DEBUG)

    try:
        from cyllama.llama.server.mongoose_server import MongooseServer

        config = ServerConfig(
            model_path="models/Llama-3.2-1B-Instruct-Q8_0.gguf",
            host="127.0.0.1",
            port=8099,
            n_ctx=256
        )

        print("Creating MongooseServer...")
        server = MongooseServer(config)

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
    test_mongoose_shutdown()