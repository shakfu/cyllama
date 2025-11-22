#!/usr/bin/env python3
"""
Minimal test to isolate Mongoose server issue.
"""

import sys
import time
import logging
from pathlib import Path

# Add the src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_minimal_mongoose():
    print("Testing minimal Mongoose functionality...")

    try:
        from cyllama.llama.server.embedded import EmbeddedServer
        from cyllama.llama.server.python import ServerConfig

        print("✓ Import successful")

        config = ServerConfig(
            model_path="models/Llama-3.2-1B-Instruct-Q8_0.gguf",
            host="127.0.0.1",
            port=8099,
            n_ctx=256
        )

        print("✓ Config created")

        server = EmbeddedServer(config)
        print("✓ Server instance created")

        # Test model loading separately
        print("Testing model loading...")
        if server.load_model():
            print("✓ Model loaded successfully")
        else:
            print("✗ Model loading failed")
            return

        # Test server start with minimal operations
        print("Testing server start (without full server loop)...")

        # Let's try to avoid the start() method and test individual components
        listen_addr = f"http://{config.host}:{config.port}"
        print(f"✓ Listen address: {listen_addr}")

        print("All basic operations successful - issue is likely in server.start() or event loop")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_minimal_mongoose()