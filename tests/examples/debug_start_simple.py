#!/usr/bin/env python3
"""
Simple debug to test if server.start() hangs.
"""

import sys
import time
import threading
from pathlib import Path

# Add the src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cyllama.llama.server.embedded import ServerConfig

def test_start_with_timeout():
    print("Testing server.start() with timeout...")

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

        # Use threading to test if start() hangs
        start_result = [None]
        exception_result = [None]

        def start_thread():
            try:
                print("   Calling server.start()...")
                result = server.start()
                start_result[0] = result
                print(f"   server.start() returned: {result}")
            except Exception as e:
                exception_result[0] = e
                print(f"   server.start() raised exception: {e}")

        # Start the thread
        thread = threading.Thread(target=start_thread)
        thread.start()

        # Wait up to 10 seconds
        thread.join(timeout=10)

        if thread.is_alive():
            print("✗ server.start() hangs - thread is still running after 10 seconds")

            # Try to stop gracefully
            try:
                server.stop()
                print("   Called server.stop()")
            except:
                pass

            return False
        else:
            if exception_result[0]:
                print(f"✗ server.start() failed with exception: {exception_result[0]}")
                return False
            elif start_result[0]:
                print("✓ server.start() completed successfully")

                # Stop the server
                server.stop()
                print("✓ server.stop() completed")
                return True
            else:
                print("✗ server.start() returned False")
                return False

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_start_with_timeout()
    if success:
        print("\n✓ Start/stop cycle works correctly")
    else:
        print("\n✗ Start/stop cycle has issues")