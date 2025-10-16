#!/usr/bin/env python3
"""
Test if Mongoose server can actually serve HTTP requests.
"""

import sys
import time
import threading
import subprocess
import requests
from pathlib import Path

# Add the src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_mongoose_http():
    print("Testing Mongoose server HTTP functionality...")

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
        print("Waiting for server to start...")
        time.sleep(10)

        # Test if server is responding
        print("Testing HTTP requests...")

        try:
            # Test health endpoint
            response = requests.get("http://127.0.0.1:8099/health", timeout=5)
            print(f"✓ Health check: {response.status_code} - {response.text}")

            # Test models endpoint
            response = requests.get("http://127.0.0.1:8099/v1/models", timeout=5)
            print(f"✓ Models endpoint: {response.status_code}")

            return True

        except requests.exceptions.RequestException as e:
            print(f"✗ HTTP request failed: {e}")
            return False

    finally:
        # Clean up
        print("Terminating server...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

if __name__ == "__main__":
    success = test_mongoose_http()
    if success:
        print("\n✓ Mongoose server HTTP functionality works")
    else:
        print("\n✗ Mongoose server HTTP functionality failed")