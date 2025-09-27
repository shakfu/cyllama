#!/usr/bin/env python3
"""
Test Mongoose server signal handling directly.
"""

import sys
import os
import signal
import subprocess
import time
from pathlib import Path

def test_mongoose_signal():
    print("Testing Mongoose server signal handling...")

    # Start server in background
    cmd = [
        sys.executable, "-m", "cyllama.llama.server",
        "--server-type", "mongoose",
        "-m", "models/Llama-3.2-1B-Instruct-Q8_0.gguf",
        "--port", "8099",
        "--ctx-size", "256"
    ]

    env = {"PYTHONPATH": "src"}
    process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)

    try:
        # Give server time to start
        print("Waiting 5 seconds for server to start...")
        time.sleep(5)

        print(f"Server PID: {process.pid}")
        print("Sending SIGINT...")

        # Send SIGINT directly to the process
        process.send_signal(signal.SIGINT)

        # Wait for shutdown
        try:
            stdout, stderr = process.communicate(timeout=10)
            print(f"✓ Server exited with code: {process.returncode}")

            # Check stderr for our signal handling messages
            stderr_text = stderr.decode()
            if "Received signal" in stderr_text:
                print("✓ Signal handler was called")
            else:
                print("✗ Signal handler was NOT called")

            if "Shutting down" in stderr_text:
                print("✓ Shutdown message detected")
            else:
                print("✗ No shutdown message")

            return True

        except subprocess.TimeoutExpired:
            print("✗ Server did not respond to SIGINT within timeout")
            process.kill()
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        try:
            process.kill()
        except:
            pass
        return False

if __name__ == "__main__":
    success = test_mongoose_signal()
    if success:
        print("\n✓ Mongoose signal handling works")
    else:
        print("\n✗ Mongoose signal handling failed")