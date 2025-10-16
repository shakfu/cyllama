#!/usr/bin/env python3

import os
import signal
import sys
import time
import threading

# Add the src directory to the path
sys.path.insert(0, 'src')

from cyllama.llama.server.embedded import ServerConfig
from cyllama.llama.server.mongoose_server import MongooseServer

def test_signal_handling():
    """Test that signal handling works correctly."""

    config = ServerConfig(
        model_path='models/Llama-3.2-1B-Instruct-Q8_0.gguf',
        host='127.0.0.1',
        port=8080,
        n_ctx=2048,
        n_gpu_layers=-1,
        n_parallel=1
    )

    server = MongooseServer(config)

    print("Loading model...")
    if not server.start():
        print("Failed to start server")
        return False

    print(f"Server started successfully!")
    print(f"Process PID: {os.getpid()}")

    # Set up a timer to automatically send SIGINT after 1 second
    def send_signal():
        time.sleep(1)
        print("Sending SIGINT to self...")
        os.kill(os.getpid(), signal.SIGINT)

    timer_thread = threading.Thread(target=send_signal)
    timer_thread.daemon = True
    timer_thread.start()

    print("Starting event loop (will auto-shutdown in 1 second)...")
    try:
        # This should exit when SIGINT is received
        server.wait_for_shutdown()
        print("Event loop exited successfully - GIL release fixed!")
        return True
    except Exception as e:
        print(f"Error in event loop: {e}")
        return False
    finally:
        print("Cleaning up...")
        server.stop()

if __name__ == '__main__':
    success = test_signal_handling()
    print(f"Test {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)