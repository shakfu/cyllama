#!/usr/bin/env python3
"""
Simple Performance Profiling for cyllama operations

Focus on the most time-consuming operations to identify bottlenecks.
"""

import cProfile
import pstats
import io
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cyllama


def profile_tokenization():
    """Profile tokenization operations."""
    print("=== Profiling Tokenization ===")

    # Load model
    model_params = cyllama.LlamaModelParams()
    model = cyllama.LlamaModel("models/Llama-3.2-1B-Instruct-Q8_0.gguf", model_params)

    test_texts = [
        "Hello world",
        "This is a longer sentence that should take more time to tokenize.",
        "The quick brown fox jumps over the lazy dog. " * 5,
    ]

    def tokenize_benchmark():
        for text in test_texts:
            for _ in range(20):  # 20 iterations per text
                tokens = model.tokenize(text, add_special=True, parse_special=False)
        return "done"

    # Profile
    pr = cProfile.Profile()
    pr.enable()
    start_time = time.time()

    result = tokenize_benchmark()

    end_time = time.time()
    pr.disable()

    print(f"Tokenization took: {end_time - start_time:.3f} seconds")

    # Show results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(10)  # Top 10

    print("Top 10 functions by cumulative time:")
    print(s.getvalue())

    return pr


def profile_inference():
    """Profile basic inference operations."""
    print("\n=== Profiling Inference ===")

    # Setup
    model_params = cyllama.LlamaModelParams()
    model = cyllama.LlamaModel("models/Llama-3.2-1B-Instruct-Q8_0.gguf", model_params)

    ctx_params = cyllama.LlamaContextParams()
    ctx_params.n_ctx = 512
    ctx_params.n_batch = 1
    context = cyllama.LlamaContext(model, ctx_params)

    sampler_params = cyllama.LlamaSamplerChainParams()
    sampler = cyllama.LlamaSampler(sampler_params)

    # Tokenize a simple prompt
    tokens = model.tokenize("Hello", add_special=True, parse_special=False)

    def inference_benchmark():
        # Create batch
        batch = cyllama.LlamaBatch(n_tokens=len(tokens), embd=0, n_seq_max=1)
        for i, token in enumerate(tokens):
            batch.add(token, i, [0], i == len(tokens) - 1)

        # Decode
        context.decode(batch)

        # Sample multiple times
        for _ in range(10):
            token = sampler.sample(context, -1)

            # Create next batch
            next_batch = cyllama.LlamaBatch(n_tokens=1, embd=0, n_seq_max=1)
            next_batch.add(token, len(tokens), [0], True)
            context.decode(next_batch)

        return "done"

    # Profile
    pr = cProfile.Profile()
    pr.enable()
    start_time = time.time()

    result = inference_benchmark()

    end_time = time.time()
    pr.disable()

    print(f"Inference took: {end_time - start_time:.3f} seconds")

    # Show results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(10)  # Top 10

    print("Top 10 functions by cumulative time:")
    print(s.getvalue())

    return pr


def profile_logits():
    """Profile logits operations."""
    print("\n=== Profiling Logits ===")

    # Setup
    model_params = cyllama.LlamaModelParams()
    model = cyllama.LlamaModel("models/Llama-3.2-1B-Instruct-Q8_0.gguf", model_params)

    ctx_params = cyllama.LlamaContextParams()
    ctx_params.n_ctx = 512
    context = cyllama.LlamaContext(model, ctx_params)

    # Tokenize and decode once to get logits
    tokens = model.tokenize("Test", add_special=True, parse_special=False)
    batch = cyllama.LlamaBatch(n_tokens=len(tokens), embd=0, n_seq_max=1)
    for i, token in enumerate(tokens):
        batch.add(token, i, [0], i == len(tokens) - 1)
    context.decode(batch)

    def logits_benchmark():
        for _ in range(50):  # Get logits 50 times
            logits = context.get_logits()
        return "done"

    # Profile
    pr = cProfile.Profile()
    pr.enable()
    start_time = time.time()

    result = logits_benchmark()

    end_time = time.time()
    pr.disable()

    print(f"Logits operations took: {end_time - start_time:.3f} seconds")

    # Show results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(10)

    print("Top 10 functions by cumulative time:")
    print(s.getvalue())

    return pr


def main():
    """Main profiling function."""
    print("cyllama Simple Performance Profiling")
    print("=" * 50)

    try:
        # Profile key operations
        profiles = {}

        profiles['tokenization'] = profile_tokenization()
        profiles['inference'] = profile_inference()
        profiles['logits'] = profile_logits()

        print("\n" + "=" * 50)
        print("Profiling Complete!")
        print("\nLook for:")
        print("- Functions taking the most cumulative time")
        print("- Functions called many times (high ncalls)")
        print("- Python vs C/Cython time distribution")

    except Exception as e:
        print(f"Error during profiling: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()