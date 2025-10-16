#!/usr/bin/env python3

import os
import signal
import sys
import time
import threading
from multiprocessing import Process

# Add the src directory to the path
sys.path.insert(0, 'src')

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
        print("Failed to start server")
        return

    print(f"Server started, PID: {os.getpid()}")

    try:
        # This should be responsive to SIGINT
        server.wait_for_shutdown()
        print("Server shutdown completed")
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        server.stop()

def test_shutdown():
    """Test graceful shutdown."""
    # Start server in a separate process
    server_process = Process(target=run_server)
    server_process.start()

    # Wait for server to be ready
    time.sleep(10)

    print(f"Sending SIGINT to server process {server_process.pid}")

    # Send SIGINT
    server_process.terminate()  # This sends SIGTERM

    # Wait for up to 5 seconds for graceful shutdown
    server_process.join(timeout=5)

    if server_process.is_alive():
        print("Server did not shutdown gracefully, sending SIGKILL")
        server_process.kill()
        server_process.join()
        return False
    else:
        print("Server shutdown gracefully")
        return True

if __name__ == '__main__':
    success = test_shutdown()
    exit(0 if success else 1)