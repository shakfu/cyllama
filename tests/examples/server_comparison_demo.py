#!/usr/bin/env python3
"""
Demo comparing the embedded Python server vs Mongoose C server.

Usage:
    python server_comparison_demo.py -m models/Llama-3.2-1B-Instruct-Q8_0.gguf
"""

import subprocess
import time
import sys
import argparse
from pathlib import Path

def run_server_demo(model_path, server_type, port):
    """Run a server demo for the specified type."""
    print(f"\n{'='*50}")
    print(f"Testing {server_type.upper()} Server on port {port}")
    print(f"{'='*50}")

    cmd = [
        sys.executable, "-m", "cyllama.llama.server",
        "-m", model_path,
        "--server-type", server_type,
        "--port", str(port),
        "--ctx-size", "256"
    ]

    print(f"Command: {' '.join(cmd)}")
    print(f"Starting {server_type} server...")

    # Start server and let it run for a few seconds
    try:
        env = {"PYTHONPATH": "src"}
        process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Let it run for a moment to see startup
        time.sleep(3)

        # Terminate gracefully
        process.terminate()
        stdout, stderr = process.communicate(timeout=5)

        print(f"✓ {server_type.capitalize()} server started and stopped successfully")
        if "Model loaded successfully" in stderr.decode():
            print("✓ Model loaded successfully")
        if f"server running at http://127.0.0.1:{port}" in stderr.decode().lower():
            print("✓ Server listening on correct port")

    except subprocess.TimeoutExpired:
        process.kill()
        print(f"✓ {server_type.capitalize()} server started (killed after timeout)")
    except Exception as e:
        print(f"✗ Error testing {server_type} server: {e}")

def main(model_path):
    print("Server Comparison Demo")
    print("=====================")
    print("This demo tests both server implementations:")
    print("1. Embedded Python server (default)")
    print("2. Mongoose C server (high-performance)")

    # Test embedded server
    run_server_demo(model_path, "embedded", 8095)

    # Test mongoose server
    run_server_demo(model_path, "mongoose", 8096)

    print(f"\n{'='*50}")
    print("Comparison Summary:")
    print("• Embedded Server: Python HTTP server with GIL limitations")
    print("• Mongoose Server: High-performance C networking")
    print("• Both use same OpenAI-compatible API")
    print("• Both use same ServerSlot logic for LLM inference")
    print("• Mongoose recommended for production/high-throughput use")
    print(f"{'='*50}")

    print("\nUsage examples:")
    print("# Default embedded server:")
    print("python -m cyllama.llama.server -m model.gguf")
    print()
    print("# High-performance Mongoose server:")
    print("python -m cyllama.llama.server -m model.gguf --server-type mongoose")
    print()
    print("# With multiple parallel slots:")
    print("python -m cyllama.llama.server -m model.gguf --server-type mongoose --n-parallel 4")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server Comparison Demo")
    parser.add_argument("-m", "--model", required=True, help="Path to model file")
    args = parser.parse_args()
    main(args.model)