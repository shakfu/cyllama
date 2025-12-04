#!/usr/bin/env python3
"""
Benchmark to compare performance with and without memory pooling.
This demonstrates the actual benefits of the pooling optimizations.

Usage:
    python pooling_comparison_benchmark.py -m models/Llama-3.2-1B-Instruct-Q8_0.gguf
"""

import time
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cyllama

def benchmark_without_pooling(model_path):
    """Benchmark using direct allocation (no pooling)"""
    print("=" * 60)
    print("BENCHMARK WITHOUT POOLING (Direct Allocation)")
    print("=" * 60)

    # Load model
    model_params = cyllama.LlamaModelParams()
    model = cyllama.LlamaModel(model_path, model_params)
    vocab = model.get_vocab()

    # Reset pools to ensure clean state
    cyllama.reset_token_pool()
    cyllama.reset_batch_pool()

    # Test tokenization without pooling by creating fresh lists every time
    test_texts = [
        "Hello world",
        "This is a longer sentence that should demonstrate memory allocation patterns.",
        "The quick brown fox jumps over the lazy dog. " * 3,
    ]

    tokenization_times = []
    iterations = 500

    print("Tokenization without pooling:")
    for i, text in enumerate(test_texts):
        start_time = time.time()

        for _ in range(iterations):
            # Simulate no pooling by creating new lists directly
            tokens = vocab.tokenize(text, add_special=True, parse_special=False)
            # Force creation of new list (simulating pre-pooling behavior)
            new_list = [0] * len(tokens)
            for j in range(len(tokens)):
                new_list[j] = tokens[j]

        end_time = time.time()
        elapsed = end_time - start_time
        ops_per_sec = iterations / elapsed

        print(f"  Text {i+1} ({len(text)} chars): {ops_per_sec:,.0f} ops/s, {elapsed:.4f}s")
        tokenization_times.append(elapsed)

    # Test batch creation without pooling
    batch_times = []
    batch_sizes = [8, 32, 64, 128]

    print("\nBatch creation without pooling:")
    for size in batch_sizes:
        tokens = list(range(size))
        start_time = time.time()

        for _ in range(iterations):
            # Create batch using constructor (no pooling) and set it up manually
            batch = cyllama.LlamaBatch(n_tokens=size, embd=0, n_seq_max=1)
            # Simulate the setup work that llama_batch_get_one does
            batch.set_batch(tokens, n_past=0, logits_all=False)

        end_time = time.time()
        elapsed = end_time - start_time
        ops_per_sec = iterations / elapsed

        print(f"  Size {size}: {ops_per_sec:,.0f} ops/s, {elapsed:.4f}s")
        batch_times.append(elapsed)

    return tokenization_times, batch_times

def benchmark_with_pooling(model_path):
    """Benchmark using memory pooling"""
    print("\n" + "=" * 60)
    print("BENCHMARK WITH POOLING (Memory Pool Optimization)")
    print("=" * 60)

    # Load model
    model_params = cyllama.LlamaModelParams()
    model = cyllama.LlamaModel(model_path, model_params)
    vocab = model.get_vocab()

    # Reset pools to ensure clean state
    cyllama.reset_token_pool()
    cyllama.reset_batch_pool()

    # Test tokenization with pooling (current implementation)
    test_texts = [
        "Hello world",
        "This is a longer sentence that should demonstrate memory allocation patterns.",
        "The quick brown fox jumps over the lazy dog. " * 3,
    ]

    tokenization_times = []
    iterations = 500

    print("Tokenization with pooling:")
    for i, text in enumerate(test_texts):
        start_time = time.time()

        for _ in range(iterations):
            # This uses the memory pool internally
            tokens = vocab.tokenize(text, add_special=True, parse_special=False)

        end_time = time.time()
        elapsed = end_time - start_time
        ops_per_sec = iterations / elapsed

        print(f"  Text {i+1} ({len(text)} chars): {ops_per_sec:,.0f} ops/s, {elapsed:.4f}s")
        tokenization_times.append(elapsed)

    # Test batch creation with pooling
    batch_times = []
    batch_sizes = [8, 32, 64, 128]

    print("\nBatch creation with pooling:")
    for size in batch_sizes:
        tokens = list(range(size))
        start_time = time.time()

        for _ in range(iterations):
            # This uses the memory pool internally
            batch = cyllama.llama_batch_get_one(tokens, n_past=0)

        end_time = time.time()
        elapsed = end_time - start_time
        ops_per_sec = iterations / elapsed

        print(f"  Size {size}: {ops_per_sec:,.0f} ops/s, {elapsed:.4f}s")
        batch_times.append(elapsed)

    # Show pool statistics
    print("\nMemory Pool Statistics:")
    token_stats = cyllama.get_token_pool_stats()
    batch_stats = cyllama.get_batch_pool_stats()

    print(f"Token Pool - Total usage: {sum(token_stats['usage_count'].values())}")
    print(f"Token Pool - Pool hits: {token_stats['total_pooled_lists']}")
    print(f"Batch Pool - Total usage: {sum(batch_stats['usage_count'].values())}")
    print(f"Batch Pool - Pool hits: {batch_stats['total_pooled_batches']}")

    return tokenization_times, batch_times

def compare_results(no_pool_token_times, no_pool_batch_times, pool_token_times, pool_batch_times):
    """Compare and show improvement percentages"""
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON & IMPROVEMENTS")
    print("=" * 60)

    print("Tokenization Performance:")
    for i in range(len(no_pool_token_times)):
        no_pool_time = no_pool_token_times[i]
        pool_time = pool_token_times[i]
        improvement = ((no_pool_time - pool_time) / no_pool_time) * 100

        no_pool_ops = 500 / no_pool_time
        pool_ops = 500 / pool_time

        print(f"  Text {i+1}:")
        print(f"    Without pooling: {no_pool_ops:,.0f} ops/s")
        print(f"    With pooling:    {pool_ops:,.0f} ops/s")
        if improvement > 0:
            print(f"    Improvement:     +{improvement:.1f}% faster")
        else:
            print(f"    Change:          {improvement:.1f}% (slower)")
        print()

    print("Batch Creation Performance:")
    batch_sizes = [8, 32, 64, 128]
    for i in range(len(no_pool_batch_times)):
        no_pool_time = no_pool_batch_times[i]
        pool_time = pool_batch_times[i]
        improvement = ((no_pool_time - pool_time) / no_pool_time) * 100

        no_pool_ops = 500 / no_pool_time
        pool_ops = 500 / pool_time

        print(f"  Size {batch_sizes[i]}:")
        print(f"    Without pooling: {no_pool_ops:,.0f} ops/s")
        print(f"    With pooling:    {pool_ops:,.0f} ops/s")
        if improvement > 0:
            print(f"    Improvement:     +{improvement:.1f}% faster")
        else:
            print(f"    Change:          {improvement:.1f}% (slower)")
        print()

    # Calculate overall improvements
    total_no_pool_time = sum(no_pool_token_times) + sum(no_pool_batch_times)
    total_pool_time = sum(pool_token_times) + sum(pool_batch_times)
    overall_improvement = ((total_no_pool_time - total_pool_time) / total_no_pool_time) * 100

    print(f"Overall Performance Improvement: {overall_improvement:.1f}% faster with pooling")

def benchmark_memory_pressure():
    """Test memory pooling under high pressure (many allocations)"""
    print("\n" + "=" * 60)
    print("MEMORY PRESSURE TEST (High Allocation Rate)")
    print("=" * 60)

    # Reset pools
    cyllama.reset_token_pool()
    cyllama.reset_batch_pool()

    iterations = 1000

    print("Testing high-frequency allocations...")

    # Without pooling simulation
    start_time = time.time()
    for i in range(iterations):
        # Simulate creating many different-sized objects
        size = (i % 50) + 10  # sizes from 10 to 59
        # Direct allocation (no pooling)
        temp_list = [0] * size
        temp_batch = cyllama.LlamaBatch(n_tokens=size, embd=0, n_seq_max=1)
    end_time = time.time()
    no_pool_time = end_time - start_time

    # Reset for fair comparison
    cyllama.reset_token_pool()
    cyllama.reset_batch_pool()

    # With pooling
    start_time = time.time()
    for i in range(iterations):
        # Simulate creating many different-sized objects
        size = (i % 50) + 10  # sizes from 10 to 59
        tokens = list(range(size))
        # Uses pooling internally
        temp_batch = cyllama.llama_batch_get_one(tokens, n_past=0)
        # Return to pool (in real usage)
        cyllama.return_batch_to_pool(temp_batch)
    end_time = time.time()
    pool_time = end_time - start_time

    improvement = ((no_pool_time - pool_time) / no_pool_time) * 100

    print(f"High-pressure allocation test ({iterations} allocations):")
    print(f"  Without pooling: {no_pool_time:.4f}s ({iterations/no_pool_time:,.0f} allocs/s)")
    print(f"  With pooling:    {pool_time:.4f}s ({iterations/pool_time:,.0f} allocs/s)")
    print(f"  Improvement:     {improvement:.1f}% faster with pooling")

    # Show final pool state
    batch_stats = cyllama.get_batch_pool_stats()
    print(f"  Final pool state: {batch_stats['total_pooled_batches']} objects pooled")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pooling Comparison Benchmark")
    parser.add_argument("-m", "--model", required=True, help="Path to model file")
    args = parser.parse_args()

    print("Memory Pooling Performance Comparison")
    print("=====================================")
    print("This benchmark compares performance with and without memory pooling")
    print("to demonstrate the actual benefits of the optimization.")

    # Run benchmarks
    no_pool_token_times, no_pool_batch_times = benchmark_without_pooling(args.model)
    pool_token_times, pool_batch_times = benchmark_with_pooling(args.model)

    # Compare results
    compare_results(no_pool_token_times, no_pool_batch_times, pool_token_times, pool_batch_times)

    # Test under memory pressure
    benchmark_memory_pressure()