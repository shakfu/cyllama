#!/usr/bin/env python3
"""
Focused Performance Profiling for cyllama

Profile the most important operations to find bottlenecks.

Usage:
    python focused_profile.py -m models/Llama-3.2-1B-Instruct-Q8_0.gguf
"""

import cProfile
import pstats
import io
import sys
import time
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cyllama
from cyllama import llama_batch_get_one


def profile_tokenization(model_path):
    """Profile tokenization operations."""
    print("=== Profiling Tokenization ===")

    # Load model and get vocab
    model_params = cyllama.LlamaModelParams()
    model = cyllama.LlamaModel(model_path, model_params)
    vocab = model.get_vocab()

    test_texts = [
        "Hello world",
        "This is a longer sentence to tokenize",
        "Machine learning and AI " * 3,
    ]

    def tokenize_benchmark():
        all_tokens = []
        for text in test_texts:
            for _ in range(30):  # 30 iterations
                tokens = vocab.tokenize(text, add_special=True, parse_special=False)
                all_tokens.extend(tokens)
        return len(all_tokens)

    # Profile
    pr = cProfile.Profile()
    pr.enable()
    start_time = time.time()

    total_tokens = tokenize_benchmark()

    end_time = time.time()
    pr.disable()

    print(f"Tokenized {total_tokens} tokens in {end_time - start_time:.3f} seconds")
    print(f"Tokens per second: {total_tokens / (end_time - start_time):.0f}")

    # Show results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(15)

    print("Top functions by cumulative time:")
    lines = s.getvalue().split('\n')
    for line in lines[:25]:  # First 25 lines
        print(line)

    return pr


def profile_inference_basic(model_path):
    """Profile basic inference cycle."""
    print("\n=== Profiling Basic Inference ===")

    # Setup
    model_params = cyllama.LlamaModelParams()
    model = cyllama.LlamaModel(model_path, model_params)
    vocab = model.get_vocab()

    ctx_params = cyllama.LlamaContextParams()
    ctx_params.n_ctx = 256  # Small context for speed
    ctx_params.n_batch = 1
    context = cyllama.LlamaContext(model, ctx_params)

    sampler_params = cyllama.LlamaSamplerChainParams()
    sampler = cyllama.LlamaSampler(sampler_params)

    # Tokenize prompt
    prompt = "The future"
    tokens = vocab.tokenize(prompt, add_special=True, parse_special=False)

    def inference_cycle():
        # Create and process batch
        batch = llama_batch_get_one(tokens, 0)

        # Decode
        context.decode(batch)

        # Generate a few tokens
        pos = len(tokens)
        for _ in range(5):  # Generate 5 tokens
            # Sample
            token = sampler.sample(context, -1)

            # Create next batch
            batch = llama_batch_get_one([token], pos)
            context.decode(batch)
            pos += 1

        return pos

    # Profile
    pr = cProfile.Profile()
    pr.enable()
    start_time = time.time()

    final_pos = inference_cycle()

    end_time = time.time()
    pr.disable()

    print(f"Generated {final_pos - len(tokens)} tokens in {end_time - start_time:.3f} seconds")
    print(f"Tokens per second: {(final_pos - len(tokens)) / (end_time - start_time):.1f}")

    # Show results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(15)

    print("Top functions by cumulative time:")
    lines = s.getvalue().split('\n')
    for line in lines[:25]:
        print(line)

    return pr


def profile_logits_retrieval(model_path):
    """Profile logits retrieval operations."""
    print("\n=== Profiling Logits Retrieval ===")

    # Setup minimal context for logits
    model_params = cyllama.LlamaModelParams()
    model = cyllama.LlamaModel(model_path, model_params)
    vocab = model.get_vocab()

    ctx_params = cyllama.LlamaContextParams()
    ctx_params.n_ctx = 128
    context = cyllama.LlamaContext(model, ctx_params)

    # Prepare context with one token
    tokens = vocab.tokenize("Test", add_special=True, parse_special=False)
    batch = llama_batch_get_one(tokens, 0)
    context.decode(batch)

    def logits_benchmark():
        logits_count = 0
        for _ in range(100):  # Get logits 100 times
            logits = context.get_logits()
            logits_count += len(logits)
        return logits_count

    # Profile
    pr = cProfile.Profile()
    pr.enable()
    start_time = time.time()

    total_logits = logits_benchmark()

    end_time = time.time()
    pr.disable()

    print(f"Retrieved {total_logits:,} logit values in {end_time - start_time:.3f} seconds")
    print(f"Logits per second: {total_logits / (end_time - start_time):,.0f}")

    # Show results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(15)

    print("Top functions by cumulative time:")
    lines = s.getvalue().split('\n')
    for line in lines[:20]:
        print(line)

    return pr


def main(model_path):
    """Main profiling function."""
    print("cyllama Focused Performance Profiling")
    print("=" * 60)

    try:
        # Profile each operation separately
        tokenization_profile = profile_tokenization(model_path)
        inference_profile = profile_inference_basic(model_path)
        logits_profile = profile_logits_retrieval(model_path)

        print("\n" + "=" * 60)
        print("PERFORMANCE BOTTLENECK ANALYSIS")
        print("=" * 60)
        print("\nKey metrics to look for:")
        print("- High 'cumtime' values indicate expensive operations")
        print("- High 'ncalls' with significant 'tottime' indicate hot loops")
        print("- Python function names vs C/Cython function names")
        print("- Memory allocation patterns")

        print("\nBased on the profiles above, focus optimization on:")
        print("1. Functions with highest cumulative time")
        print("2. Functions called many times (hot paths)")
        print("3. Python overhead vs C/Cython performance")

    except Exception as e:
        print(f"Error during profiling: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Focused Performance Profiling")
    parser.add_argument("-m", "--model", required=True, help="Path to model file")
    args = parser.parse_args()
    main(args.model)