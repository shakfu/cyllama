#!/usr/bin/env python3
"""
Test if Mongoose server can actually serve HTTP requests.
"""

import sys
import time
import subprocess
import requests
import pytest
from conftest import DEFAULT_MODEL


@pytest.mark.skip(reason="Hanging if run in the test suite")
def test_mongoose_http():
    """Test that Mongoose server can serve HTTP requests."""
    # Start server in background
    cmd = [
        sys.executable, "-m", "cyllama.llama.server",
        "--server-type", "mongoose",
        "-m", DEFAULT_MODEL,
        "--port", "8099",
        "--ctx-size", "256"
    ]

    env = {"PYTHONPATH": "src"}
    process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)

    try:
        # Wait for server to be ready (retry for up to 30 seconds)
        max_retries = 30
        server_ready = False

        for i in range(max_retries):
            time.sleep(1)
            try:
                response = requests.get("http://127.0.0.1:8099/health", timeout=2)
                if response.status_code == 200:
                    server_ready = True
                    break
            except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
                continue

        # Get server output for debugging if it didn't start
        if not server_ready:
            stdout, stderr = process.communicate(timeout=1) if process.poll() is not None else (b"", b"")
            assert server_ready, \
                f"Server did not become ready within 30 seconds. stderr: {stderr.decode() if stderr else 'N/A'}"

        # Test health endpoint
        response = requests.get("http://127.0.0.1:8099/health", timeout=5)
        assert response.status_code == 200, \
            f"Health check failed with status {response.status_code}: {response.text}"

        # Test models endpoint
        response = requests.get("http://127.0.0.1:8099/v1/models", timeout=5)
        assert response.status_code == 200, \
            f"Models endpoint failed with status {response.status_code}: {response.text}"

    finally:
        # Clean up
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()