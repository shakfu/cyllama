"""Example: Using Progress Callback to monitor model loading.

This example demonstrates:
1. Basic progress callback for loading feedback
2. Progress bar display during model loading
3. Aborting model loading based on conditions
4. Timeout-based loading cancellation
"""

import io
import os
import sys
import time
import threading
from pathlib import Path

import cyllama.llama.llama_cpp as cy

# Force unbuffered stdout for progress bar
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=False)
elif not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(
        open(sys.stdout.fileno(), 'wb', 0),
        write_through=True
    )


def basic_progress_example(model_path: str):
    """Basic example: Print loading progress."""
    print("\n=== Basic Progress Callback ===\n")

    def on_progress(progress: float) -> bool:
        print(f"Loading: {progress * 100:.1f}%")
        return True  # continue loading

    params = cy.LlamaModelParams()
    params.progress_callback = on_progress

    print(f"Loading model: {model_path}")
    model = cy.LlamaModel(model_path, params)
    print("Model loaded successfully!")

    return model


def progress_bar_example(model_path: str):
    """Display a visual progress bar during loading.

    Uses ANSI escape codes for reliable in-place updates.
    """
    print("\n=== Progress Bar Example ===\n")

    last_percent = [-1]  # use list to allow modification in nested function

    def on_progress(progress: float) -> bool:
        # Only update display when percentage changes by at least 1%
        percent = int(progress * 100)
        if percent == last_percent[0]:
            return True
        last_percent[0] = percent

        bar_width = 40
        filled = int(bar_width * progress)
        bar = "=" * filled + "-" * (bar_width - filled)
        # ANSI: \033[2K clears line, \r returns to start
        sys.stdout.write(f"\033[2K\r[{bar}] {percent:3d}%")
        sys.stdout.flush()
        return True

    params = cy.LlamaModelParams()
    params.progress_callback = on_progress
    params.use_mmap = False  # Disable mmap to slow loading and show progress

    print(f"Loading: {Path(model_path).name}", flush=True)
    model = cy.LlamaModel(model_path, params)
    print()  # newline after progress bar
    print("Model loaded successfully!")

    return model


def abort_loading_example(model_path: str, abort_at: float = 0.5):
    """Demonstrate aborting model loading at a certain percentage."""
    print(f"\n=== Abort Loading Example (at {abort_at * 100:.0f}%) ===\n")

    def on_progress(progress: float) -> bool:
        print(f"Loading: {progress * 100:.1f}%")
        if progress >= abort_at:
            print(f"Aborting at {progress * 100:.1f}%!")
            return False  # abort loading
        return True

    params = cy.LlamaModelParams()
    params.progress_callback = on_progress

    print(f"Loading model: {model_path}")
    try:
        model = cy.LlamaModel(model_path, params)
        print("Model loaded successfully!")
        return model
    except ValueError as e:
        print(f"Loading aborted: {e}")
        return None


def timeout_loading_example(model_path: str, timeout_seconds: float = 2.0):
    """Demonstrate timeout-based loading cancellation."""
    print(f"\n=== Timeout Loading Example ({timeout_seconds}s timeout) ===\n")

    start_time = time.time()
    timed_out = False

    def on_progress(progress: float) -> bool:
        nonlocal timed_out
        elapsed = time.time() - start_time
        print(f"Loading: {progress * 100:.1f}% (elapsed: {elapsed:.2f}s)")

        if elapsed > timeout_seconds:
            timed_out = True
            print(f"Timeout after {elapsed:.2f}s!")
            return False  # abort loading
        return True

    params = cy.LlamaModelParams()
    params.use_mmap = False  # disable mmap to make loading slower for demo
    params.progress_callback = on_progress

    print(f"Loading model with {timeout_seconds}s timeout...")
    try:
        model = cy.LlamaModel(model_path, params)
        print(f"Model loaded in {time.time() - start_time:.2f}s")
        return model
    except ValueError as e:
        if timed_out:
            print(f"Loading timed out after {time.time() - start_time:.2f}s")
        else:
            print(f"Loading failed: {e}")
        return None


def cancellable_loading_example(model_path: str):
    """Demonstrate cancellable loading from another thread."""
    print("\n=== Cancellable Loading Example ===\n")

    cancelled = False
    load_complete = threading.Event()

    def on_progress(progress: float) -> bool:
        print(f"Loading: {progress * 100:.1f}%")
        if cancelled:
            print("Cancelled by user!")
            return False
        return True

    def cancel_after_delay(delay: float):
        nonlocal cancelled
        time.sleep(delay)
        if not load_complete.is_set():
            print("\n[Background thread] Sending cancel signal...")
            cancelled = True

    params = cy.LlamaModelParams()
    params.use_mmap = False  # disable mmap to make loading slower
    params.progress_callback = on_progress

    # Start a thread that will cancel loading after 0.5 seconds
    cancel_thread = threading.Thread(target=cancel_after_delay, args=(0.5,))
    cancel_thread.start()

    print(f"Loading model (will be cancelled after 0.5s)...")
    try:
        model = cy.LlamaModel(model_path, params)
        load_complete.set()
        print("Model loaded successfully!")
        cancel_thread.join()
        return model
    except ValueError as e:
        load_complete.set()
        cancel_thread.join()
        print(f"Loading cancelled: {e}")
        return None


def statistics_callback_example(model_path: str):
    """Collect loading statistics using progress callback."""
    print("\n=== Statistics Callback Example ===\n")

    stats = {
        "start_time": None,
        "end_time": None,
        "progress_updates": [],
        "intervals": [],
    }

    def on_progress(progress: float) -> bool:
        now = time.time()

        if stats["start_time"] is None:
            stats["start_time"] = now

        stats["progress_updates"].append((now, progress))

        if len(stats["progress_updates"]) > 1:
            prev_time, prev_progress = stats["progress_updates"][-2]
            interval = now - prev_time
            delta = progress - prev_progress
            stats["intervals"].append((interval, delta))

        if progress >= 1.0:
            stats["end_time"] = now

        return True

    params = cy.LlamaModelParams()
    params.progress_callback = on_progress

    print(f"Loading model: {model_path}")
    model = cy.LlamaModel(model_path, params)

    # Print statistics
    total_time = stats["end_time"] - stats["start_time"]
    num_updates = len(stats["progress_updates"])

    print(f"\nLoading Statistics:")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Progress updates: {num_updates}")
    print(f"  Average update interval: {total_time / num_updates * 1000:.1f}ms")

    if stats["intervals"]:
        avg_interval = sum(i[0] for i in stats["intervals"]) / len(stats["intervals"])
        print(f"  Average interval between updates: {avg_interval * 1000:.1f}ms")

    return model


def main(model_path):
    """Run all progress callback examples."""
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("Please run 'make download' or provide a valid model path.")
        print("\nUsage: python progress_callback_example.py -m <model_path>")
        return

    # Initialize backend and disable verbose logging
    cy.llama_backend_init()
    cy.disable_logging()

    try:
        # Run examples
        basic_progress_example(model_path)
        progress_bar_example(model_path)
        statistics_callback_example(model_path)

        # These examples abort loading intentionally
        abort_loading_example(model_path, abort_at=0.3)
        # timeout_loading_example(model_path, timeout_seconds=0.1)  # May succeed if fast
        # cancellable_loading_example(model_path)  # May succeed if fast

    finally:
        cy.llama_backend_free()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Progress Callback Examples")
    parser.add_argument("-m", "--model", required=True, help="Path to model file")
    args = parser.parse_args()

    main(args.model)
