#!/usr/bin/env python3
"""
Quick Performance Profile using existing working code patterns.
"""

import cProfile
import pstats
import io
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cyllama


def profile_tokenization():
    """Profile tokenization using vocab."""
    print("=== Tokenization Performance ===")

    # Load model
    model_params = cyllama.LlamaModelParams()
    model = cyllama.LlamaModel("models/Llama-3.2-1B-Instruct-Q8_0.gguf", model_params)
    vocab = model.get_vocab()

    texts = ["Hello world", "This is a test sentence", "AI and machine learning"]

    def tokenize_loop():
        total_tokens = 0
        for text in texts:
            for _ in range(50):  # 50 iterations
                tokens = vocab.tokenize(text, add_special=True, parse_special=False)
                total_tokens += len(tokens)
        return total_tokens

    # Profile
    pr = cProfile.Profile()
    pr.enable()

    start = time.time()
    total = tokenize_loop()
    end = time.time()

    pr.disable()

    print(f"Tokenized {total} tokens in {end-start:.3f}s ({total/(end-start):.0f} tokens/s)")

    # Show profile
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(10)

    print("Profile results:")
    print(s.getvalue())


def profile_memory_operations():
    """Profile memory-intensive operations like getting logits."""
    print("\n=== Memory Operations Performance ===")

    # Setup
    model_params = cyllama.LlamaModelParams()
    model = cyllama.LlamaModel("models/Llama-3.2-1B-Instruct-Q8_0.gguf", model_params)
    vocab = model.get_vocab()

    ctx_params = cyllama.LlamaContextParams()
    ctx_params.n_ctx = 64  # Small context for speed
    context = cyllama.LlamaContext(model, ctx_params)

    # Create a simple batch to generate logits
    tokens = vocab.tokenize("Test", add_special=True, parse_special=False)
    batch = cyllama.LlamaBatch(n_tokens=len(tokens), embd=0, n_seq_max=1)

    # Add tokens to batch using add_sequence
    batch.add_sequence(tokens, 0, True)

    # Decode to get logits
    context.decode(batch)

    def memory_loop():
        logits_size = 0
        for _ in range(20):  # 20 iterations
            logits = context.get_logits()
            logits_size += len(logits)
        return logits_size

    # Profile
    pr = cProfile.Profile()
    pr.enable()

    start = time.time()
    total_size = memory_loop()
    end = time.time()

    pr.disable()

    print(f"Retrieved {total_size:,} logit values in {end-start:.3f}s")

    # Show profile
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(10)

    print("Profile results:")
    print(s.getvalue())


def profile_model_properties():
    """Profile accessing model properties and metadata."""
    print("\n=== Model Properties Performance ===")

    model_params = cyllama.LlamaModelParams()
    model = cyllama.LlamaModel("models/Llama-3.2-1B-Instruct-Q8_0.gguf", model_params)

    def properties_loop():
        total = 0
        for _ in range(1000):  # 1000 iterations
            total += model.n_embd
            total += model.n_layer
            total += model.n_head
        return total

    # Profile
    pr = cProfile.Profile()
    pr.enable()

    start = time.time()
    result = properties_loop()
    end = time.time()

    pr.disable()

    print(f"Property access took {end-start:.3f}s for 3000 accesses")

    # Show profile
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(10)

    print("Profile results:")
    print(s.getvalue())


def main():
    """Run all profiling tests."""
    print("cyllama Quick Performance Profile")
    print("=" * 50)

    try:
        profile_tokenization()
        profile_memory_operations()
        profile_model_properties()

        print("\n" + "=" * 50)
        print("ANALYSIS GUIDELINES:")
        print("=" * 50)
        print("Look for:")
        print("1. High 'cumtime' (cumulative time) - these are expensive operations")
        print("2. High 'ncalls' with significant 'tottime' - these are hot paths")
        print("3. Python vs Cython function names - Python overhead")
        print("4. Memory allocation/deallocation patterns")
        print("\nOptimization priorities:")
        print("- Add 'nogil' to functions spending most time in C code")
        print("- Cache frequently accessed properties")
        print("- Optimize hot loops with high call counts")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()