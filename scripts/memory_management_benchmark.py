#!/usr/bin/env python3
"""
Benchmark memory management performance with pooling optimizations.
"""

import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cyllama

def benchmark_memory_management():
    """Benchmark memory management performance"""
    print("Memory Management Performance Benchmark")
    print("=" * 50)

    # Load model for realistic benchmarking
    model_params = cyllama.LlamaModelParams()
    model = cyllama.LlamaModel("models/Llama-3.2-1B-Instruct-Q8_0.gguf", model_params)
    vocab = model.get_vocab()

    print(f"Model: {model.path_model}")
    print(f"Vocabulary size: {vocab.n_vocab}")
    print()

    # Test different tokenization workloads
    test_texts = [
        ("Short text", "Hello world"),
        ("Medium text", "This is a longer sentence that should demonstrate tokenization performance."),
        ("Long text", "The quick brown fox jumps over the lazy dog. " * 5),
        ("Very long text", "Machine learning and artificial intelligence are transforming the world. " * 10),
    ]

    for name, text in test_texts:
        print(f"Testing {name} tokenization ({len(text)} chars):")

        # Test tokenization with memory pool
        iterations = 1000

        # Warm-up
        for _ in range(10):
            tokens = vocab.tokenize(text, add_special=True, parse_special=False)

        # Benchmark tokenization (uses memory pool internally)
        start_time = time.time()
        total_tokens = 0

        for _ in range(iterations):
            tokens = vocab.tokenize(text, add_special=True, parse_special=False)
            total_tokens += len(tokens)

        end_time = time.time()

        elapsed = end_time - start_time
        tokens_per_second = total_tokens / elapsed
        calls_per_second = iterations / elapsed

        print(f"  {iterations} tokenization calls, {total_tokens} total tokens")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Speed: {tokens_per_second:,.0f} tokens/s")
        print(f"  Calls: {calls_per_second:,.0f} calls/s")
        print(f"  Avg tokens per call: {total_tokens/iterations:.1f}")
        print()

    # Test batch creation workloads
    print("Batch creation performance:")

    # Test different batch sizes
    batch_sizes = [8, 16, 32, 64, 128]

    for batch_size in batch_sizes:
        print(f"Testing batch size {batch_size}:")

        tokens = list(range(batch_size))
        iterations = 500

        # Test llama_batch_get_one (uses batch memory pool internally)
        # Warm-up
        for _ in range(10):
            batch = cyllama.llama_batch_get_one(tokens, n_past=0)

        # Benchmark batch creation
        start_time = time.time()

        for _ in range(iterations):
            batch = cyllama.llama_batch_get_one(tokens, n_past=0)

        end_time = time.time()

        elapsed = end_time - start_time
        batches_per_second = iterations / elapsed

        print(f"  llama_batch_get_one: {iterations} calls in {elapsed:.4f}s")
        print(f"  Speed: {batches_per_second:,.0f} batch creations/s")
        print(f"  Avg time per batch: {elapsed/iterations*1000:.3f} ms")

        # Test pooled batch creation
        # Warm-up
        for _ in range(10):
            batch = cyllama.get_pooled_batch(batch_size, 0, 1)

        # Benchmark pooled batch creation
        start_time = time.time()

        for _ in range(iterations):
            batch = cyllama.get_pooled_batch(batch_size, 0, 1)

        end_time = time.time()

        elapsed = end_time - start_time
        batches_per_second = iterations / elapsed

        print(f"  get_pooled_batch: {iterations} calls in {elapsed:.4f}s")
        print(f"  Speed: {batches_per_second:,.0f} batch creations/s")
        print(f"  Avg time per batch: {elapsed/iterations*1000:.3f} ms")
        print()

    # Test memory pool statistics
    print("Memory Pool Statistics:")

    # Get token pool stats
    token_stats = cyllama.get_token_pool_stats()
    print("Token Pool:")
    print(f"  Total pools: {token_stats['total_pools']}")
    print(f"  Total pooled lists: {token_stats['total_pooled_lists']}")
    print(f"  Pool sizes by token count: {token_stats['pool_sizes']}")
    print(f"  Usage count by token count: {token_stats['usage_count']}")
    print()

    # Get batch pool stats
    batch_stats = cyllama.get_batch_pool_stats()
    print("Batch Pool:")
    print(f"  Total pools: {batch_stats['total_pools']}")
    print(f"  Total pooled batches: {batch_stats['total_pooled_batches']}")
    print(f"  Pool configs: {batch_stats['pool_configs']}")
    print(f"  Usage count by config: {batch_stats['usage_count']}")
    print()

    # Memory-intensive workload simulation
    print("Memory-intensive workload simulation:")
    iterations = 200
    mixed_workload_times = []

    for i in range(iterations):
        start_time = time.time()

        # Mixed workload: tokenization + batch creation
        text = f"This is test iteration {i} with some variable content."
        tokens = vocab.tokenize(text, add_special=True, parse_special=False)
        batch = cyllama.llama_batch_get_one(tokens, n_past=0)

        # Simulate using the batch (return to pool would happen here in real usage)
        # In a real scenario, batches would be returned to pool after use
        cyllama.return_batch_to_pool(batch)

        end_time = time.time()
        mixed_workload_times.append(end_time - start_time)

    avg_workload_time = sum(mixed_workload_times) / len(mixed_workload_times)
    workloads_per_second = 1.0 / avg_workload_time

    print(f"  {iterations} mixed workloads (tokenize + batch + return)")
    print(f"  Speed: {workloads_per_second:,.0f} workloads/s")
    print(f"  Avg time per workload: {avg_workload_time*1000:.3f} ms")
    print()

    # Final pool statistics after workload
    print("Final Memory Pool Statistics:")

    token_stats = cyllama.get_token_pool_stats()
    print("Token Pool:")
    print(f"  Total pooled lists: {token_stats['total_pooled_lists']}")
    print(f"  Usage count: {sum(token_stats['usage_count'].values())}")

    batch_stats = cyllama.get_batch_pool_stats()
    print("Batch Pool:")
    print(f"  Total pooled batches: {batch_stats['total_pooled_batches']}")
    print(f"  Usage count: {sum(batch_stats['usage_count'].values())}")

if __name__ == "__main__":
    benchmark_memory_management()