#!/usr/bin/env python3
"""
Performance Profiling Script for cyllama

This script profiles various cyllama operations to identify performance bottlenecks
using cProfile and provides detailed analysis of where time is spent.
"""

import cProfile
import pstats
import io
import sys
import time
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import cyllama
    from cyllama import (
        LlamaModel, LlamaModelParams, LlamaContext, LlamaContextParams,
        LlamaBatch, LlamaSampler, LlamaSamplerChainParams
    )
except ImportError as e:
    print(f"Error importing cyllama: {e}")
    print("Make sure cyllama is built: make build")
    sys.exit(1)


class CyllamaProfiler:
    """Profiler for cyllama operations."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.context = None
        self.sampler = None

    def setup_model(self):
        """Setup model, context and sampler for profiling."""
        print("Setting up model for profiling...")

        # Model parameters
        model_params = LlamaModelParams()

        # Load model
        self.model = LlamaModel(self.model_path, model_params)

        # Context parameters
        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = 512  # Smaller context for faster profiling
        ctx_params.n_batch = 1
        ctx_params.n_threads = 4

        # Create context
        self.context = LlamaContext(self.model, ctx_params)

        # Sampler parameters
        sampler_params = LlamaSamplerChainParams()
        self.sampler = LlamaSampler(sampler_params)

        print(f"Model loaded: {self.model_path}")
        print(f"Context size: {ctx_params.n_ctx}")
        print(f"Vocab size: {self.model.n_vocab}")

    def profile_model_loading(self):
        """Profile model loading operations."""
        print("\n=== Profiling Model Loading ===")

        def load_model():
            model_params = LlamaModelParams()
            model = LlamaModel(self.model_path, model_params)

            ctx_params = LlamaContextParams()
            ctx_params.n_ctx = 512
            ctx_params.n_batch = 1
            context = LlamaContext(model, ctx_params)

            sampler_params = LlamaSamplerChainParams()
            sampler = LlamaSampler(sampler_params)

            return model, context, sampler

        # Profile model loading
        pr = cProfile.Profile()
        pr.enable()

        start_time = time.time()
        model, context, sampler = load_model()
        end_time = time.time()

        pr.disable()

        print(f"Model loading took: {end_time - start_time:.2f} seconds")

        # Analyze results
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions

        print("Top functions by cumulative time:")
        print(s.getvalue())

        return pr

    def profile_tokenization(self):
        """Profile tokenization operations."""
        print("\n=== Profiling Tokenization ===")

        test_texts = [
            "Hello world",
            "This is a longer sentence that should take more time to tokenize.",
            "The quick brown fox jumps over the lazy dog. " * 10,  # Longer text
            "A very short text.",
            "Machine learning and artificial intelligence are transforming the world.",
        ]

        def tokenize_texts():
            all_tokens = []
            for text in test_texts:
                # Test multiple tokenization calls
                for _ in range(10):
                    tokens = self.model.tokenize(text, add_special=True, parse_special=False)
                    all_tokens.extend(tokens)
            return all_tokens

        # Profile tokenization
        pr = cProfile.Profile()
        pr.enable()

        start_time = time.time()
        tokens = tokenize_texts()
        end_time = time.time()

        pr.disable()

        print(f"Tokenization of {len(test_texts)} texts (10x each) took: {end_time - start_time:.3f} seconds")
        print(f"Total tokens generated: {len(tokens)}")

        # Analyze results
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(15)

        print("Top functions by cumulative time:")
        print(s.getvalue())

        return pr

    def profile_inference(self):
        """Profile inference operations (decode + sample)."""
        print("\n=== Profiling Inference Operations ===")

        # Tokenize a prompt
        prompt = "The future of artificial intelligence is"
        tokens = self.model.tokenize(prompt, add_special=True, parse_special=False)

        def run_inference():
            # Create batch
            batch = LlamaBatch(n_tokens=len(tokens), embd=0, n_seq_max=1)

            # Add tokens to batch
            for i, token in enumerate(tokens):
                batch.add(token, i, [0], False)

            # Last token should generate logits
            batch.set_last_logits_to_true()

            # Decode
            self.context.decode(batch)

            # Get logits and sample multiple times
            generated_tokens = []
            for _ in range(20):  # Generate 20 tokens
                # Sample
                token = self.sampler.sample(self.context, -1)
                generated_tokens.append(token)

                # Prepare next batch with sampled token
                batch = LlamaBatch(n_tokens=1, embd=0, n_seq_max=1)
                batch.add(token, len(tokens) + len(generated_tokens) - 1, [0], True)

                # Decode
                self.context.decode(batch)

            return generated_tokens

        # Profile inference
        pr = cProfile.Profile()
        pr.enable()

        start_time = time.time()
        generated = run_inference()
        end_time = time.time()

        pr.disable()

        print(f"Inference (20 tokens) took: {end_time - start_time:.3f} seconds")
        print(f"Tokens per second: {len(generated) / (end_time - start_time):.1f}")

        # Analyze results
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(15)

        print("Top functions by cumulative time:")
        print(s.getvalue())

        return pr

    def profile_logits_operations(self):
        """Profile logits retrieval operations."""
        print("\n=== Profiling Logits Operations ===")

        # Tokenize and decode once
        tokens = self.model.tokenize("Test prompt", add_special=True, parse_special=False)
        batch = LlamaBatch(n_tokens=len(tokens), embd=0, n_seq_max=1)

        for i, token in enumerate(tokens):
            batch.add(token, i, [0], i == len(tokens) - 1)

        self.context.decode(batch)

        def get_logits_multiple():
            logits_list = []
            for _ in range(100):  # Get logits 100 times
                logits = self.context.get_logits()
                logits_list.append(logits[:10])  # Just first 10 for memory efficiency
            return logits_list

        # Profile logits operations
        pr = cProfile.Profile()
        pr.enable()

        start_time = time.time()
        logits_data = get_logits_multiple()
        end_time = time.time()

        pr.disable()

        print(f"Getting logits 100 times took: {end_time - start_time:.3f} seconds")
        print(f"Logits operations per second: {100 / (end_time - start_time):.1f}")

        # Analyze results
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(15)

        print("Top functions by cumulative time:")
        print(s.getvalue())

        return pr

    def profile_batch_operations(self):
        """Profile batch creation and manipulation."""
        print("\n=== Profiling Batch Operations ===")

        def batch_operations():
            batches = []
            for _ in range(1000):  # Create 1000 batches
                batch = LlamaBatch(n_tokens=10, embd=0, n_seq_max=1)

                # Add tokens
                for i in range(10):
                    batch.add(i + 1000, i, [0], i == 9)

                batches.append(batch)

            return batches

        # Profile batch operations
        pr = cProfile.Profile()
        pr.enable()

        start_time = time.time()
        batches = batch_operations()
        end_time = time.time()

        pr.disable()

        print(f"Creating and populating 1000 batches took: {end_time - start_time:.3f} seconds")
        print(f"Batches per second: {1000 / (end_time - start_time):.1f}")

        # Analyze results
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(15)

        print("Top functions by cumulative time:")
        print(s.getvalue())

        return pr

    def save_profile_data(self, profiles, output_dir="profile_results"):
        """Save detailed profile data to files."""
        os.makedirs(output_dir, exist_ok=True)

        for name, pr in profiles.items():
            filename = os.path.join(output_dir, f"{name}_profile.prof")
            pr.dump_stats(filename)
            print(f"Saved profile data: {filename}")

            # Also save readable report
            with open(os.path.join(output_dir, f"{name}_report.txt"), 'w') as f:
                s = io.StringIO()
                ps = pstats.Stats(pr, stream=s)
                ps.sort_stats('cumulative')
                ps.print_stats()
                f.write(s.getvalue())


def main():
    """Main profiling function."""
    model_path = "models/Llama-3.2-1B-Instruct-Q8_0.gguf"

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please download a model or update the path")
        return

    print("cyllama Performance Profiling")
    print("=" * 50)

    profiler = CyllamaProfiler(model_path)

    profiles = {}

    # 1. Profile model loading
    try:
        profiles['model_loading'] = profiler.profile_model_loading()
    except Exception as e:
        print(f"Error profiling model loading: {e}")

    # Setup model for other operations
    try:
        profiler.setup_model()
    except Exception as e:
        print(f"Error setting up model: {e}")
        return

    # 2. Profile tokenization
    try:
        profiles['tokenization'] = profiler.profile_tokenization()
    except Exception as e:
        print(f"Error profiling tokenization: {e}")

    # 3. Profile inference
    try:
        profiles['inference'] = profiler.profile_inference()
    except Exception as e:
        print(f"Error profiling inference: {e}")

    # 4. Profile logits operations
    try:
        profiles['logits'] = profiler.profile_logits_operations()
    except Exception as e:
        print(f"Error profiling logits: {e}")

    # 5. Profile batch operations
    try:
        profiles['batch_ops'] = profiler.profile_batch_operations()
    except Exception as e:
        print(f"Error profiling batch operations: {e}")

    # Save detailed profile data
    profiler.save_profile_data(profiles)

    print("\n" + "=" * 50)
    print("Profiling Complete!")
    print("Check profile_results/ directory for detailed reports")
    print("\nKey areas to investigate:")
    print("- Functions with high cumulative time")
    print("- Functions called many times")
    print("- Memory allocation patterns")
    print("- C/Cython vs Python time split")


if __name__ == "__main__":
    main()