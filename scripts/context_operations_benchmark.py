#!/usr/bin/env python3
"""
Benchmark context operations performance with optimizations.
"""

import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cyllama

def benchmark_context_operations():
    """Benchmark context operations performance"""
    print("Context Operations Performance Benchmark")
    print("=" * 50)

    # Load model and create context
    model_params = cyllama.LlamaModelParams()
    model = cyllama.LlamaModel("models/Llama-3.2-1B-Instruct-Q8_0.gguf", model_params)
    vocab = model.get_vocab()

    ctx_params = cyllama.LlamaContextParams()
    ctx_params.n_ctx = 512
    ctx_params.n_batch = 32
    ctx = cyllama.LlamaContext(model, ctx_params)

    # Create sampler
    sampler_params = cyllama.LlamaSamplerChainParams()
    sampler = cyllama.LlamaSampler(sampler_params)
    sampler.add_greedy()

    print(f"Model: {model.path_model}")
    print(f"Context size: {ctx.n_ctx}")
    print()

    # Test different batch sizes
    test_cases = [
        ("Small batch", list(range(10))),
        ("Medium batch", list(range(32))),
        ("Large batch", list(range(64))),
    ]

    for name, tokens in test_cases:
        print(f"Testing {name} ({len(tokens)} tokens):")

        # Test decode operation
        batch = cyllama.llama_batch_get_one(tokens, n_past=0)

        # Warm-up runs
        for _ in range(5):
            try:
                ctx.decode(batch)
            except Exception:
                pass  # Context may be full, ignore for benchmark

        # Benchmark decode operations
        iterations = 100
        start_time = time.time()

        successful_decodes = 0
        for _ in range(iterations):
            try:
                result = ctx.decode(batch)
                successful_decodes += 1
            except Exception:
                pass  # Context may be full, ignore for benchmark

        end_time = time.time()

        elapsed = end_time - start_time
        if successful_decodes > 0:
            decodes_per_second = successful_decodes / elapsed
            print(f"  decode: {successful_decodes}/{iterations} successful calls in {elapsed:.4f}s")
            print(f"  Speed: {decodes_per_second:,.0f} decode ops/s")
            print(f"  Avg time per decode: {elapsed/successful_decodes*1000:.3f} ms")
        else:
            print(f"  decode: No successful calls (context full)")

        # Test sample operation (only if decode succeeded)
        if successful_decodes > 0:
            # Warm-up runs
            for _ in range(10):
                try:
                    sampler.sample(ctx, -1)
                except Exception:
                    pass

            # Benchmark sample operations
            start_time = time.time()

            successful_samples = 0
            for _ in range(iterations):
                try:
                    result = sampler.sample(ctx, -1)
                    successful_samples += 1
                except Exception:
                    pass

            end_time = time.time()

            elapsed = end_time - start_time
            if successful_samples > 0:
                samples_per_second = successful_samples / elapsed
                print(f"  sample: {successful_samples}/{iterations} successful calls in {elapsed:.4f}s")
                print(f"  Speed: {samples_per_second:,.0f} sample ops/s")
                print(f"  Avg time per sample: {elapsed/successful_samples*1000:.3f} ms")
            else:
                print(f"  sample: No successful calls")

        print()

    # Test inference simulation (decode + sample chain)
    print("Inference simulation (decode + sample chain):")
    prompt_tokens = list(range(20))
    iterations = 50

    # Create fresh context for each iteration to avoid state issues
    inference_times = []
    successful_inferences = 0

    for i in range(iterations):
        # Create a fresh context for each inference
        fresh_ctx = cyllama.LlamaContext(model, ctx_params)

        try:
            start_time = time.time()

            # Decode prompt
            batch = cyllama.llama_batch_get_one(prompt_tokens, n_past=0)
            fresh_ctx.decode(batch)

            # Sample next token
            next_token = sampler.sample(fresh_ctx, -1)

            end_time = time.time()
            inference_times.append(end_time - start_time)
            successful_inferences += 1

        except Exception as e:
            pass  # Skip failed inferences

    if successful_inferences > 0:
        avg_inference_time = sum(inference_times) / len(inference_times)
        inferences_per_second = successful_inferences / sum(inference_times)

        print(f"  {successful_inferences}/{iterations} successful inference cycles")
        print(f"  Speed: {inferences_per_second:,.0f} inference cycles/s")
        print(f"  Avg time per inference: {avg_inference_time*1000:.3f} ms")
        print(f"  Total inference time: {sum(inference_times):.4f}s")
    else:
        print(f"  No successful inference cycles")

    print()

if __name__ == "__main__":
    benchmark_context_operations()