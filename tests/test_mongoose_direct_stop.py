#!/usr/bin/env python3
"""
Test Mongoose server direct start/stop without context manager.
"""

import sys
import time
import signal
from pathlib import Path

# Add the src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cyllama.llama.server.embedded import ServerConfig

def test_direct_stop():
    print("Testing Mongoose server direct start/stop...")

    try:
        from cyllama.llama.server.mongoose_server import MongooseServer

        config = ServerConfig(
            model_path="models/Llama-3.2-1B-Instruct-Q8_0.gguf",
            host="127.0.0.1",
            port=8099,
            n_ctx=256
        )

        server = MongooseServer(config)
        print("✓ Server created")

        # Start server
        if not server.start():
            print("✗ Failed to start server")
            return False
        print("✓ Server started")

        # Wait a bit
        time.sleep(1)
        print("✓ Server running for 1 second")

        # Stop server directly
        print("Stopping server...")
        server.stop()
        print("✓ Server stopped")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_direct_stop()
    if success:
        print("\n✓ Direct start/stop works correctly")
    else:
        print("\n✗ Direct start/stop has issues")