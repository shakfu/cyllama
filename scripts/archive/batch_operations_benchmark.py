#!/usr/bin/env python3
"""
Benchmark batch operations performance with nogil optimizations.
"""

import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cyllama

def benchmark_batch_operations():
    """Benchmark batch operations performance"""
    print("Batch Operations Performance Benchmark")
    print("=" * 50)

    # Test tokens of varying lengths
    test_cases = [
        ("Small batch", list(range(10))),
        ("Medium batch", list(range(50))),
        ("Large batch", list(range(100))),
        ("Very large batch", list(range(500))),
    ]

    print("Testing batch operations with nogil optimizations")
    print()

    for name, tokens in test_cases:
        print(f"Testing {name} ({len(tokens)} tokens):")

        # Test llama_batch_get_one function
        # Warm-up runs
        for _ in range(10):
            batch = cyllama.llama_batch_get_one(tokens, n_past=0)

        # Benchmark runs
        iterations = 1000
        start_time = time.time()

        for _ in range(iterations):
            batch = cyllama.llama_batch_get_one(tokens, n_past=0)

        end_time = time.time()

        elapsed = end_time - start_time
        batches_per_second = iterations / elapsed

        print(f"  llama_batch_get_one: {iterations} calls in {elapsed:.4f}s")
        print(f"  Speed: {batches_per_second:,.0f} batch creations/s")
        print(f"  Avg time per batch: {elapsed/iterations*1000:.3f} ms")

        # Test batch.set_batch method
        batch = cyllama.LlamaBatch(n_tokens=len(tokens), embd=0, n_seq_max=1)

        # Warm-up runs
        for _ in range(10):
            batch.set_batch(tokens, n_past=0, logits_all=False)

        # Benchmark runs
        start_time = time.time()

        for _ in range(iterations):
            batch.set_batch(tokens, n_past=0, logits_all=False)

        end_time = time.time()

        elapsed = end_time - start_time
        operations_per_second = iterations / elapsed

        print(f"  set_batch: {iterations} calls in {elapsed:.4f}s")
        print(f"  Speed: {operations_per_second:,.0f} operations/s")
        print(f"  Avg time per operation: {elapsed/iterations*1000:.3f} ms")

        # Test batch.add_sequence method
        # Create initial batch with fewer tokens to add sequences to
        base_tokens = tokens[:len(tokens)//2] if len(tokens) > 2 else tokens[:1]
        add_tokens = tokens[len(tokens)//2:] if len(tokens) > 2 else tokens[1:2]

        if add_tokens:  # Only test if we have tokens to add
            batch = cyllama.LlamaBatch(n_tokens=len(base_tokens) + len(add_tokens), embd=0, n_seq_max=2)

            # Warm-up runs
            for _ in range(10):
                batch.set_batch(base_tokens, n_past=0, logits_all=False)
                batch.add_sequence(add_tokens, seq_id=1, logits_all=False)

            # Benchmark runs
            start_time = time.time()

            for _ in range(iterations):
                batch.set_batch(base_tokens, n_past=0, logits_all=False)
                batch.add_sequence(add_tokens, seq_id=1, logits_all=False)

            end_time = time.time()

            elapsed = end_time - start_time
            operations_per_second = iterations / elapsed

            print(f"  add_sequence: {iterations} calls in {elapsed:.4f}s")
            print(f"  Speed: {operations_per_second:,.0f} operations/s")
            print(f"  Avg time per operation: {elapsed/iterations*1000:.3f} ms")

        print()

    # Test performance-critical workload simulation
    print("Batch-heavy workload simulation:")
    iterations = 5000
    tokens = list(range(32))  # Typical prompt size

    def batch_workload():
        # Simulates typical batch processing workflow
        batch = cyllama.llama_batch_get_one(tokens, n_past=0)
        batch.set_last_logits_to_true()
        return batch

    # Warm-up
    for _ in range(100):
        batch_workload()

    start_time = time.time()
    for _ in range(iterations):
        result = batch_workload()
    end_time = time.time()

    elapsed = end_time - start_time
    workloads_per_second = iterations / elapsed

    print(f"  {iterations:,} batch workloads in {elapsed:.4f}s")
    print(f"  Speed: {workloads_per_second:,.0f} workloads/s")
    print(f"  Avg time per workload: {elapsed/iterations*1000:.3f} ms")
    print()

if __name__ == "__main__":
    benchmark_batch_operations()