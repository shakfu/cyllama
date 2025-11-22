#!/usr/bin/env python3
"""
Test embedded server direct start/stop without context manager.
"""

import time
from cyllama.llama.server.python import ServerConfig
from cyllama.llama.server.embedded import EmbeddedServer


def test_direct_stop():
    """Test that embedded server can be started and stopped directly without context manager."""
    config = ServerConfig(
        model_path="models/Llama-3.2-1B-Instruct-Q8_0.gguf",
        host="127.0.0.1",
        port=8099,
        n_ctx=256
    )

    server = EmbeddedServer(config)

    # Start server
    result = server.start()
    assert result is True, "Server failed to start"

    try:
        # Wait a bit to ensure server is running
        time.sleep(1)

        # Server should be running at this point
        # (No direct way to check _running since it's a cdef attribute,
        # but if start() returned True, it should be running)

    finally:
        # Stop server directly
        server.stop()
        # If stop() completes without exception, the test passes