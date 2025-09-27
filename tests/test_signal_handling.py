#!/usr/bin/env python3
"""
Test signal handling directly.
"""

import sys
import time
import signal
from pathlib import Path

# Add the src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    print(f"Signal {signum} received!")
    shutdown_requested = True

def test_signal_handling():
    global shutdown_requested

    print("Testing signal handling...")

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    print("Signal handlers registered")

    print("Waiting for signal (send Ctrl+C or SIGINT)...")
    try:
        while not shutdown_requested:
            time.sleep(0.1)
        print("Shutdown requested via signal - exiting gracefully")
        return True
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught directly")
        return True

if __name__ == "__main__":
    success = test_signal_handling()
    if success:
        print("✓ Signal handling works")
    else:
        print("✗ Signal handling failed")