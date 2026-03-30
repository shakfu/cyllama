#!/usr/bin/env python3
"""RSS-growth leak detector for cyllama Cython wrappers.

Exercises model load/unload, inference, and context creation in a loop,
measuring RSS after each cycle.  If RSS grows beyond a threshold over
multiple cycles the test fails -- indicating a likely native memory leak.

Usage:
    python scripts/leak_check.py [--cycles N] [--threshold PCT] [--model PATH]

Requires: the default test model (make download).
"""

import argparse
import gc
import os
import resource
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "Llama-3.2-1B-Instruct-Q8_0.gguf"


def get_rss_mb() -> float:
    """Return current RSS in megabytes (macOS + Linux)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS reports in bytes, Linux in kilobytes
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


def run_inference_cycle(model_path: str) -> None:
    """Load model, run inference, tear down -- one full cycle."""
    from cyllama import GenerationConfig, LLM

    config = GenerationConfig(max_tokens=16)
    with LLM(model_path, n_gpu_layers=0, n_ctx=256, n_batch=256) as llm:
        # Generate a short completion to exercise the full pipeline
        llm("Hello", config=config)


def run_context_cycle(model_path: str) -> None:
    """Load model, create/destroy multiple contexts."""
    from cyllama.llama.llama_cpp import (
        LlamaContext,
        LlamaContextParams,
        LlamaModel,
        LlamaModelParams,
    )

    params = LlamaModelParams()
    params.n_gpu_layers = 0
    model = LlamaModel(model_path, params)

    for _ in range(3):
        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = 256
        ctx = LlamaContext(model, ctx_params, verbose=False)
        del ctx

    del model


def main():
    parser = argparse.ArgumentParser(description="RSS-growth leak detector")
    parser.add_argument("--cycles", type=int, default=10, help="Number of load/unload cycles (default: 10)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=20.0,
        help="Max allowed RSS growth as %% of baseline (default: 20)",
    )
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to GGUF model")
    args = parser.parse_args()

    model_path = args.model
    if not Path(model_path).exists():
        print(f"ERROR: Model not found: {model_path}")
        print("Run 'make download' to fetch the default test model.")
        sys.exit(1)

    # Suppress llama.cpp log noise
    os.environ.setdefault("LLAMA_LOG_LEVEL", "0")

    # Warm-up cycle (first load has one-time allocations: Metal shaders, etc.)
    print(f"Warm-up cycle (model: {Path(model_path).name})...")
    run_inference_cycle(model_path)
    run_context_cycle(model_path)
    gc.collect()

    baseline_rss = get_rss_mb()
    print(f"Baseline RSS after warm-up: {baseline_rss:.1f} MB")
    print(f"Running {args.cycles} cycles (threshold: {args.threshold:.0f}% growth)...\n")

    rss_values = [baseline_rss]

    for i in range(1, args.cycles + 1):
        run_inference_cycle(model_path)
        run_context_cycle(model_path)
        gc.collect()

        rss = get_rss_mb()
        rss_values.append(rss)
        delta = rss - baseline_rss
        pct = (delta / baseline_rss) * 100 if baseline_rss > 0 else 0
        status = "OK" if pct <= args.threshold else "WARN"
        print(f"  Cycle {i:2d}/{args.cycles}: RSS={rss:.1f} MB  delta={delta:+.1f} MB ({pct:+.1f}%)  [{status}]")

    final_rss = rss_values[-1]
    total_growth = final_rss - baseline_rss
    total_pct = (total_growth / baseline_rss) * 100 if baseline_rss > 0 else 0

    print(f"\nFinal RSS: {final_rss:.1f} MB (baseline: {baseline_rss:.1f} MB)")
    print(f"Total growth: {total_growth:+.1f} MB ({total_pct:+.1f}%)")

    # Check for monotonic growth pattern (stronger signal than just final value)
    # If RSS grew in >70% of cycles, that's a leak pattern even if under threshold
    growing_cycles = sum(1 for i in range(1, len(rss_values)) if rss_values[i] > rss_values[i - 1] + 0.5)
    monotonic_pct = (growing_cycles / args.cycles) * 100

    if total_pct > args.threshold:
        print(f"\nFAIL: RSS grew {total_pct:.1f}% (threshold: {args.threshold:.0f}%)")
        sys.exit(1)
    elif monotonic_pct > 70 and total_pct > 5:
        print(f"\nWARN: RSS grew in {growing_cycles}/{args.cycles} cycles ({monotonic_pct:.0f}%) -- possible slow leak")
        sys.exit(1)
    else:
        print("\nPASS: No significant memory leak detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()
