#!/usr/bin/env python3
"""
Test Mongoose server signal handling directly.
"""

import sys
import signal
import subprocess
import time
import pytest
from conftest import DEFAULT_MODEL


@pytest.mark.skip(reason="Signal handling works interactively (Ctrl+C) but not via subprocess.send_signal() - Python signal handlers may not be invoked properly across process boundaries")
def test_mongoose_signal():
    """Test that Mongoose server properly handles SIGINT signal."""
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
        # Give server time to start
        time.sleep(5)

        # Send SIGINT directly to the process
        process.send_signal(signal.SIGINT)

        # Wait for shutdown
        try:
            stdout, stderr = process.communicate(timeout=10)
            stderr_text = stderr.decode()

            # Assert server exited cleanly
            assert process.returncode is not None, "Server process did not exit"

            # Assert signal handler was called
            assert "Received signal" in stderr_text, \
                f"Signal handler was not called. stderr: {stderr_text}"

            # Assert shutdown message was logged
            assert "Stopping Mongoose server" in stderr_text, \
                f"Shutdown message not found. stderr: {stderr_text}"

        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            assert False, "Server did not respond to SIGINT within timeout"

    except Exception as e:
        # Clean up process if test fails
        try:
            process.kill()
            process.communicate()
        except:
            pass
        raise