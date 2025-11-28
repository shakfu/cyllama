#!/usr/bin/env python3
"""
Test of embedded server start/stop using context manager.
"""

import time
import logging

from cyllama.llama.server.python import ServerConfig
from cyllama.llama.server.embedded import EmbeddedServer


def test_manual_stop(model_path):
    """Test that embedded server context manager properly starts and stops the server."""
    # Enable info level logging to see debug messages
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

    config = ServerConfig(
        model_path=model_path,
        host="127.0.0.1",
        port=8099,
        n_ctx=256
    )

    # Use context manager - should start server on entry and stop on exit
    with EmbeddedServer(config) as server:
        # Server should be started (no direct way to verify since _running is cdef)
        # But if we reach here without exception, start was successful

        # Wait a moment to ensure server is fully operational
        time.sleep(1)

    # Context manager exit should have called stop()
    # If we reach here without exception, stop was successful
    # No assertion needed - any exception will fail the test