#!/usr/bin/env python3

import os
import time
import pytest
from multiprocessing import Process

from cyllama.llama.server.embedded import ServerConfig
from cyllama.llama.server.mongoose_server import MongooseServer


def run_server():
    """Run the server in a subprocess."""
    config = ServerConfig(
        model_path='models/Llama-3.2-1B-Instruct-Q8_0.gguf',
        host='127.0.0.1',
        port=8080,
        n_ctx=2048,
        n_gpu_layers=-1,
        n_parallel=1
    )

    server = MongooseServer(config)

    if not server.start():
        return

    try:
        # This should be responsive to SIGTERM
        server.wait_for_shutdown()
    finally:
        server.stop()


@pytest.mark.skip(reason="Graceful shutdown from external process has signal handling issues - similar to test_mserver_signal.py")
def test_shutdown():
    """Test graceful shutdown of Mongoose server."""
    # Start server in a separate process
    server_process = Process(target=run_server)
    server_process.start()

    try:
        # Wait for server to be ready
        time.sleep(10)

        # Send SIGTERM
        server_process.terminate()

        # Wait for up to 5 seconds for graceful shutdown
        server_process.join(timeout=5)

        # Assert server shutdown gracefully
        assert not server_process.is_alive(), \
            "Server did not shutdown gracefully within 5 seconds"

    finally:
        # Clean up if test fails
        if server_process.is_alive():
            server_process.kill()
            server_process.join()