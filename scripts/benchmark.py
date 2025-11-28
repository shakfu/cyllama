#!/usr/bin/env python3
"""
Benchmark script for cyllama text generation performance.

Measures prefill speed, decode speed, and time-to-first-token across multiple runs.

Usage:
    python benchmark.py -m models/Llama-3.2-1B-Instruct-Q8_0.gguf
    python benchmark.py -m models/model.gguf -n 20 -p "Explain quantum computing"
    python benchmark.py -m models/model.gguf --no-warmup -v
"""

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Disable logging before importing cyllama
import logging
logging.disable(logging.CRITICAL)

import cyllama
from cyllama.llama.llama_cpp import (
    LlamaModel,
    LlamaContext,
    LlamaModelParams,
    LlamaContextParams,
    LlamaSampler,
    LlamaSamplerChainParams,
    llama_batch_get_one,
)


@dataclass
class RunMetrics:
    """Metrics from a single benchmark run."""
    prompt_tokens: int = 0
    generated_tokens: int = 0
    prefill_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    total_time_ms: float = 0.0

    @property
    def ttft_ms(self) -> float:
        """Time to first token (same as prefill time)."""
        return self.prefill_time_ms

    @property
    def prefill_tokens_per_sec(self) -> float:
        """Prefill speed in tokens/second."""
        if self.prefill_time_ms > 0:
            return self.prompt_tokens / (self.prefill_time_ms / 1000)
        return 0.0

    @property
    def decode_tokens_per_sec(self) -> float:
        """Decode speed in tokens/second."""
        if self.decode_time_ms > 0:
            return self.generated_tokens / (self.decode_time_ms / 1000)
        return 0.0

    @property
    def overall_tokens_per_sec(self) -> float:
        """Overall speed (generated tokens / total time)."""
        if self.total_time_ms > 0:
            return self.generated_tokens / (self.total_time_ms / 1000)
        return 0.0


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""
    runs: List[RunMetrics] = field(default_factory=list)
    warmup_run: Optional[RunMetrics] = None

    @property
    def num_runs(self) -> int:
        return len(self.runs)

    def _calc_stats(self, values: List[float]) -> dict:
        """Calculate statistics for a list of values."""
        if not values:
            return {"avg": 0, "median": 0, "min": 0, "max": 0, "stdev": 0}
        return {
            "avg": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0,
        }

    @property
    def prefill_stats(self) -> dict:
        """Statistics for prefill speed (tokens/sec)."""
        return self._calc_stats([r.prefill_tokens_per_sec for r in self.runs])

    @property
    def decode_stats(self) -> dict:
        """Statistics for decode speed (tokens/sec)."""
        return self._calc_stats([r.decode_tokens_per_sec for r in self.runs])

    @property
    def ttft_stats(self) -> dict:
        """Statistics for time-to-first-token (ms)."""
        return self._calc_stats([r.ttft_ms for r in self.runs])

    @property
    def overall_stats(self) -> dict:
        """Statistics for overall speed (tokens/sec)."""
        return self._calc_stats([r.overall_tokens_per_sec for r in self.runs])


def run_single_benchmark(
    model: LlamaModel,
    vocab,
    ctx_params: LlamaContextParams,
    prompt: str,
    max_tokens: int,
) -> RunMetrics:
    """
    Run a single benchmark iteration with detailed timing.

    Args:
        model: Loaded LlamaModel
        vocab: Model vocabulary
        ctx_params: Context parameters
        prompt: Input prompt
        max_tokens: Maximum tokens to generate

    Returns:
        RunMetrics with timing breakdown
    """
    metrics = RunMetrics()

    # Create fresh context
    ctx = LlamaContext(model, ctx_params)

    # Create sampler (greedy for determinism)
    sampler_params = LlamaSamplerChainParams()
    sampler_params.no_perf = True
    sampler = LlamaSampler(sampler_params)
    sampler.add_greedy()

    # Tokenize prompt
    prompt_tokens = vocab.tokenize(prompt, add_special=True, parse_special=True)
    metrics.prompt_tokens = len(prompt_tokens)

    # === PREFILL PHASE ===
    prefill_start = time.perf_counter()

    batch = llama_batch_get_one(prompt_tokens)
    ctx.decode(batch)

    prefill_end = time.perf_counter()
    metrics.prefill_time_ms = (prefill_end - prefill_start) * 1000

    # === DECODE PHASE ===
    decode_start = time.perf_counter()

    n_pos = len(prompt_tokens)
    generated_tokens = 0

    for _ in range(max_tokens):
        # Sample next token
        token_id = sampler.sample(ctx, -1)

        # Check for end of generation
        if vocab.is_eog(token_id):
            break

        generated_tokens += 1

        # Decode next token
        batch = llama_batch_get_one([token_id], n_pos)
        ctx.decode(batch)
        n_pos += 1

    decode_end = time.perf_counter()
    metrics.decode_time_ms = (decode_end - decode_start) * 1000
    metrics.generated_tokens = generated_tokens
    metrics.total_time_ms = (decode_end - prefill_start) * 1000

    return metrics


def run_benchmark(
    model_path: str,
    prompt: str,
    num_runs: int,
    max_tokens: int,
    n_ctx: int,
    warmup: bool = True,
    verbose: bool = False,
) -> BenchmarkResults:
    """
    Run complete benchmark with optional warmup.

    Args:
        model_path: Path to model file
        prompt: Input prompt
        num_runs: Number of benchmark runs
        max_tokens: Maximum tokens to generate
        n_ctx: Context size
        warmup: Whether to run a warmup iteration
        verbose: Print progress

    Returns:
        BenchmarkResults with all metrics
    """
    # Suppress llama.cpp output
    cyllama.disable_logging()

    results = BenchmarkResults()

    if verbose:
        print(f"Loading model: {model_path}")

    # Load model
    model_params = LlamaModelParams()
    model_params.n_gpu_layers = 99
    model = LlamaModel(model_path, model_params)
    vocab = model.get_vocab()

    # Context parameters
    ctx_params = LlamaContextParams()
    ctx_params.n_ctx = n_ctx
    ctx_params.n_batch = 512

    # Tokenize once to get prompt length
    prompt_tokens = vocab.tokenize(prompt, add_special=True, parse_special=True)

    if verbose:
        print(f"Prompt tokens: {len(prompt_tokens)}")
        print(f"Max generation: {max_tokens} tokens")
        print(f"Context size: {n_ctx}")

    # Warmup run (excluded from stats)
    if warmup:
        if verbose:
            print("\nWarmup run...")
        results.warmup_run = run_single_benchmark(
            model, vocab, ctx_params, prompt, max_tokens
        )
        if verbose:
            print(f"  Warmup: prefill={results.warmup_run.prefill_time_ms:.1f}ms, "
                  f"decode={results.warmup_run.decode_time_ms:.1f}ms, "
                  f"ttft={results.warmup_run.ttft_ms:.1f}ms")

    # Benchmark runs
    if verbose:
        print(f"\nRunning {num_runs} benchmark iterations...")
        print("-" * 60)

    for i in range(num_runs):
        metrics = run_single_benchmark(
            model, vocab, ctx_params, prompt, max_tokens
        )
        results.runs.append(metrics)

        if verbose:
            print(f"  Run {i + 1:3d}: "
                  f"prefill={metrics.prefill_time_ms:6.1f}ms ({metrics.prefill_tokens_per_sec:7.1f} tok/s), "
                  f"decode={metrics.decode_time_ms:6.1f}ms ({metrics.decode_tokens_per_sec:5.1f} tok/s), "
                  f"ttft={metrics.ttft_ms:6.1f}ms")

    return results


def format_stats(stats: dict, unit: str = "tok/s") -> str:
    """Format statistics for display."""
    return (f"avg={stats['avg']:7.2f} {unit}, "
            f"median={stats['median']:7.2f}, "
            f"min={stats['min']:7.2f}, "
            f"max={stats['max']:7.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark cyllama text generation performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmark.py -m models/Llama-3.2-1B-Instruct-Q8_0.gguf
    python benchmark.py -m models/model.gguf -n 20
    python benchmark.py -m models/model.gguf -p "Explain quantum computing" -t 100
    python benchmark.py -m models/model.gguf --no-warmup -v

Metrics explained:
    Prefill:  Processing input prompt tokens (batch inference)
    Decode:   Generating output tokens one-by-one (autoregressive)
    TTFT:     Time to first token (same as prefill time)
        """
    )
    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to model file"
    )
    parser.add_argument(
        "-n", "--num-runs",
        type=int,
        default=10,
        help="Number of benchmark runs (default: 10)"
    )
    parser.add_argument(
        "-p", "--prompt",
        default="What is the age of the universe?",
        help="Prompt to use for generation (default: 'What is the age of the universe?')"
    )
    parser.add_argument(
        "-t", "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate per run (default: 128)"
    )
    parser.add_argument(
        "-c", "--context-size",
        type=int,
        default=2048,
        help="Context size (default: 2048)"
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warmup run"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show progress during benchmark"
    )

    args = parser.parse_args()

    # Validate model path
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}", file=sys.stderr)
        return 1

    print("=" * 65)
    print("cyllama Benchmark")
    print("=" * 65)
    print(f"Model:        {args.model}")
    print(f"Prompt:       {args.prompt[:50]}{'...' if len(args.prompt) > 50 else ''}")
    print(f"Runs:         {args.num_runs}")
    print(f"Max tokens:   {args.max_tokens}")
    print(f"Context size: {args.context_size}")
    print(f"Warmup:       {'yes' if not args.no_warmup else 'no'}")
    print("=" * 65)

    try:
        results = run_benchmark(
            model_path=args.model,
            prompt=args.prompt,
            num_runs=args.num_runs,
            max_tokens=args.max_tokens,
            n_ctx=args.context_size,
            warmup=not args.no_warmup,
            verbose=args.verbose,
        )

        if results.num_runs == 0:
            print("\nNo successful runs completed.")
            return 1

        # Get first run for token counts
        first_run = results.runs[0]

        print("\n" + "=" * 65)
        print("RESULTS")
        print("=" * 65)
        print(f"Successful runs: {results.num_runs}")
        print(f"Prompt tokens:   {first_run.prompt_tokens}")
        print(f"Output tokens:   {first_run.generated_tokens}")
        print("-" * 65)

        # Prefill stats
        prefill = results.prefill_stats
        print(f"\nPrefill (prompt processing):")
        print(f"  {format_stats(prefill, 'tok/s')}")

        # Decode stats
        decode = results.decode_stats
        print(f"\nDecode (token generation):")
        print(f"  {format_stats(decode, 'tok/s')}")

        # TTFT stats
        ttft = results.ttft_stats
        print(f"\nTime to First Token (TTFT):")
        print(f"  {format_stats(ttft, 'ms')}")

        # Overall stats
        overall = results.overall_stats
        print(f"\nOverall (generated tokens / total time):")
        print(f"  {format_stats(overall, 'tok/s')}")

        print("-" * 65)

        # Summary
        print(f"\nSummary:")
        print(f"  Prefill:  {prefill['avg']:7.1f} tokens/sec")
        print(f"  Decode:   {decode['avg']:7.1f} tokens/sec")
        print(f"  TTFT:     {ttft['avg']:7.1f} ms")

        return 0

    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")
        return 130
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
