#!/usr/bin/env python3
"""
Debug script to isolate where Mongoose server.start() hangs.
"""

import sys
import time
import logging
from pathlib import Path

# Add the src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cyllama.llama.server.embedded import ServerConfig

def test_mongoose_start_step_by_step():
    print("Testing Mongoose server.start() step by step...")

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

        print("1. Creating MongooseServer...")
        server = MongooseServer(config)
        print("✓ Server instance created")

        print("2. Testing model loading...")
        if not server.load_model():
            print("✗ Model loading failed")
            return
        print("✓ Model loaded successfully")

        print("3. Testing listen address creation...")
        listen_addr = f"http://{config.host}:{config.port}"
        addr_bytes = listen_addr.encode('utf-8')
        print(f"✓ Listen address: {listen_addr}")

        print("4. Testing manager userdata assignment...")
        # Let's do what start() does step by step
        server._mgr.userdata = <void*>server
        print("✓ Manager userdata assigned")

        print("5. Testing HTTP listener creation...")
        print("   This is where it might hang...")

        # Import the Mongoose C functions
        from cyllama.llama.server.mongoose_server import cyllama_mg_http_listen, _http_event_handler

        # This is the line that likely hangs
        print("   Calling cyllama_mg_http_listen...")
        server._listener = cyllama_mg_http_listen(&server._mgr, addr_bytes,
                                                 <mg_event_handler_t>_http_event_handler,
                                                 NULL)

        if server._listener == NULL:
            print("✗ HTTP listener creation failed")
            return
        print("✓ HTTP listener created successfully")

        print("6. All steps completed successfully - issue is elsewhere")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mongoose_start_step_by_step()