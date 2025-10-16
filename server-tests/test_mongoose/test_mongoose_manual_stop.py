#!/usr/bin/env python3
"""
Manual test of Mongoose server start/stop with debug logging.
"""

import sys
import time
import logging
from pathlib import Path

# Add the src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cyllama.llama.server.embedded import ServerConfig

def test_manual_stop():
    # Enable info level logging to see our debug messages
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

    print("Testing Mongoose server manual start/stop...")

    try:
        from cyllama.llama.server.mongoose_server import MongooseServer

        config = ServerConfig(
            model_path="models/Llama-3.2-1B-Instruct-Q8_0.gguf",
            host="127.0.0.1",
            port=8099,
            n_ctx=256
        )

        print("Creating server with context manager...")

        with MongooseServer(config) as server:
            print("✓ Server started successfully")
            print("Waiting 1 second...")
            time.sleep(1)
            print("Exiting context manager (should trigger graceful shutdown)...")
            # Context manager __exit__ will be called here

        print("✓ Context manager completed")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_manual_stop()
    if success:
        print("\n✓ Manual start/stop works correctly")
    else:
        print("\n✗ Manual start/stop has issues")