#!/usr/bin/env python3
"""
Benchmark property access performance with caching optimization.

Usage:
    python property_access_benchmark.py -m models/Llama-3.2-1B-Instruct-Q8_0.gguf
"""

import time
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cyllama

def benchmark_property_access(model_path):
    """Benchmark model property access performance"""
    print("Property Access Performance Benchmark")
    print("=" * 50)

    # Load model
    model_params = cyllama.LlamaModelParams()
    model = cyllama.LlamaModel(model_path, model_params)

    # Test different property access patterns
    test_cases = [
        ("Single property (n_embd)", lambda: model.n_embd),
        ("Single property (n_layer)", lambda: model.n_layer),
        ("Single property (n_head)", lambda: model.n_head),
        ("Single property (n_head_kv)", lambda: model.n_head_kv),
        ("Single property (n_ctx_train)", lambda: model.n_ctx_train),
        ("Single property (n_params)", lambda: model.n_params),
        ("Single property (size)", lambda: model.size),
        ("Multiple properties", lambda: (model.n_embd, model.n_layer, model.n_head, model.n_head_kv)),
        ("Property-heavy loop", lambda: [model.n_embd + model.n_layer + model.n_head for _ in range(10)]),
    ]

    print(f"Model: {model.path_model}")
    print(f"Properties to test: n_embd={model.n_embd}, n_layer={model.n_layer}, n_head={model.n_head}")
    print()

    for name, access_func in test_cases:
        print(f"Testing {name}:")

        # Warm-up runs
        for _ in range(10):
            access_func()

        # Benchmark runs
        iterations = 100000  # Many iterations to see caching benefit
        start_time = time.time()

        for _ in range(iterations):
            result = access_func()

        end_time = time.time()

        elapsed = end_time - start_time
        accesses_per_second = iterations / elapsed

        print(f"  {iterations:,} accesses in {elapsed:.4f}s")
        print(f"  Speed: {accesses_per_second:,.0f} accesses/s")
        print(f"  Avg time per access: {elapsed/iterations*1000000:.2f} μs")
        print()

    # Test memory estimation workload (property-heavy)
    print("Memory estimation workload simulation:")
    iterations = 10000

    def memory_estimation_workload():
        # Simulates memory.py estimation calculations that access properties frequently
        n_embd = model.n_embd
        n_layer = model.n_layer
        n_head = model.n_head
        n_head_kv = model.n_head_kv
        n_ctx_train = model.n_ctx_train
        n_params = model.n_params
        size = model.size

        # Simulate memory calculations
        layer_params = n_embd * n_embd * 4 + n_embd * 8192 * 2  # Typical calculation
        total_params = n_layer * layer_params
        memory_estimate = size + (n_ctx_train * n_embd * 2)

        return memory_estimate

    # Warm-up
    for _ in range(100):
        memory_estimation_workload()

    start_time = time.time()
    for _ in range(iterations):
        result = memory_estimation_workload()
    end_time = time.time()

    elapsed = end_time - start_time
    workloads_per_second = iterations / elapsed

    print(f"  {iterations:,} memory estimation workloads in {elapsed:.4f}s")
    print(f"  Speed: {workloads_per_second:,.0f} workloads/s")
    print(f"  Avg time per workload: {elapsed/iterations*1000:.2f} ms")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Property Access Benchmark")
    parser.add_argument("-m", "--model", required=True, help="Path to model file")
    args = parser.parse_args()
    benchmark_property_access(args.model)