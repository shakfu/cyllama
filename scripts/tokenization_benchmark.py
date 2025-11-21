#!/usr/bin/env python3
"""
Benchmark tokenization performance with optimizations.
"""

import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cyllama

def benchmark_tokenization():
    """Benchmark tokenization performance"""
    print("Tokenization Performance Benchmark")
    print("=" * 50)

    # Load model
    model_params = cyllama.LlamaModelParams()
    model = cyllama.LlamaModel("models/Llama-3.2-1B-Instruct-Q8_0.gguf", model_params)
    vocab = model.get_vocab()

    # Test texts of varying lengths
    test_cases = [
        ("Short", "Hello world"),
        ("Medium", "This is a longer sentence that should take more time to tokenize and process."),
        ("Long", "The quick brown fox jumps over the lazy dog. " * 10),
        ("Very Long", "Machine learning and artificial intelligence are transforming the world. " * 20),
    ]

    print(f"Vocabulary size: {vocab.n_vocab}")
    print()

    for name, text in test_cases:
        print(f"Testing {name} text ({len(text)} chars):")

        # Warm-up run
        for _ in range(5):
            tokens = vocab.tokenize(text, add_special=True, parse_special=False)

        # Benchmark runs
        iterations = 1000
        start_time = time.time()

        total_tokens = 0
        for _ in range(iterations):
            tokens = vocab.tokenize(text, add_special=True, parse_special=False)
            total_tokens += len(tokens)

        end_time = time.time()

        elapsed = end_time - start_time
        tokens_per_second = total_tokens / elapsed
        calls_per_second = iterations / elapsed

        print(f"  {iterations} calls, {total_tokens} total tokens")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Speed: {tokens_per_second:,.0f} tokens/s")
        print(f"  Calls: {calls_per_second:,.0f} calls/s")
        print(f"  Avg tokens per call: {total_tokens/iterations:.1f}")
        print()

if __name__ == "__main__":
    benchmark_tokenization()