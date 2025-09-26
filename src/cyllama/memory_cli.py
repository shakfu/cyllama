#!/usr/bin/env python3
"""Command-line interface for GPU memory estimation.

This utility helps users estimate optimal GPU layer allocation for their models
and hardware configurations.
"""

import argparse
import sys
from pathlib import Path

from .memory import estimate_gpu_layers, estimate_memory_usage


def parse_gpu_memory(gpu_memory_str: str):
    """Parse GPU memory specification.

    Args:
        gpu_memory_str: Memory specification like "8192" or "4096,4096"

    Returns:
        int or list of ints representing memory in MB
    """
    if ',' in gpu_memory_str:
        # Multi-GPU setup
        return [int(x.strip()) for x in gpu_memory_str.split(',')]
    else:
        # Single GPU setup
        return int(gpu_memory_str)


def format_bytes(bytes_val):
    """Format bytes value in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} TB"


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Estimate GPU memory requirements for cyllama models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic memory estimation
  python -m cyllama.memory_cli models/model.gguf

  # With GPU memory constraint (8GB)
  python -m cyllama.memory_cli models/model.gguf --gpu-memory 8192

  # Multi-GPU setup (2x 4GB GPUs)
  python -m cyllama.memory_cli models/model.gguf --gpu-memory 4096,4096

  # Custom context and batch size
  python -m cyllama.memory_cli models/model.gguf --gpu-memory 8192 --ctx-size 4096 --batch-size 2

  # Quick memory overview only
  python -m cyllama.memory_cli models/model.gguf --overview-only
        """
    )

    parser.add_argument(
        'model_path',
        type=str,
        help='Path to the GGUF model file'
    )

    parser.add_argument(
        '--gpu-memory',
        type=str,
        help='Available GPU memory in MB (single: "8192", multi: "4096,4096")'
    )

    parser.add_argument(
        '--ctx-size',
        type=int,
        default=2048,
        help='Context size for inference (default: 2048)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for inference (default: 1)'
    )

    parser.add_argument(
        '--n-parallel',
        type=int,
        default=1,
        help='Number of parallel sequences (default: 1)'
    )

    parser.add_argument(
        '--kv-cache-type',
        choices=['f16', 'f32'],
        default='f16',
        help='KV cache precision (default: f16)'
    )

    parser.add_argument(
        '--overview-only',
        action='store_true',
        help='Show only memory overview without GPU allocation'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}", file=sys.stderr)
        return 1

    print(f"Analyzing model: {model_path}")
    print()

    try:
        # Always show memory overview
        overview = estimate_memory_usage(
            model_path=model_path,
            ctx_size=args.ctx_size,
            batch_size=args.batch_size,
            verbose=args.verbose
        )

        print("=== Memory Overview ===")
        print(f"Model parameters: {overview['parameters']['total_params']:,}")
        print(f"Architecture: {overview['parameters']['n_embd']}d x {overview['parameters']['n_layer']} layers")
        print(f"Vocabulary size: {overview['parameters']['n_vocab']:,}")
        print()

        print("Model size estimates:")
        for precision, size_mb in overview['model_size_mb'].items():
            print(f"  {precision.upper()}: {size_mb:,} MB ({format_bytes(size_mb * 1024 * 1024)})")
        print()

        print(f"KV cache (ctx={args.ctx_size}, batch={args.batch_size}):")
        for precision, size_mb in overview['kv_cache_mb'].items():
            print(f"  {precision.upper()}: {size_mb:,} MB ({format_bytes(size_mb * 1024 * 1024)})")
        print()

        print(f"Graph memory: {overview['graph_mb']:,} MB ({format_bytes(overview['graph_mb'] * 1024 * 1024)})")
        print()

        # GPU allocation estimation if requested
        if not args.overview_only and args.gpu_memory:
            gpu_memory = parse_gpu_memory(args.gpu_memory)

            print("=== GPU Memory Allocation ===")

            estimate = estimate_gpu_layers(
                model_path=model_path,
                gpu_memory_mb=gpu_memory,
                ctx_size=args.ctx_size,
                batch_size=args.batch_size,
                n_parallel=args.n_parallel,
                kv_cache_type=args.kv_cache_type,
                verbose=args.verbose
            )

            if isinstance(gpu_memory, list):
                print(f"Multi-GPU setup: {len(gpu_memory)} GPUs")
                for i, mem in enumerate(gpu_memory):
                    print(f"  GPU {i}: {mem:,} MB")
                print(f"Total GPU memory: {sum(gpu_memory):,} MB")
            else:
                print(f"Single GPU: {gpu_memory:,} MB")
            print()

            print(f"Recommended GPU layers: {estimate.layers}/{overview['parameters']['n_layer']}")
            print(f"GPU layers: {estimate.layers * 100 / overview['parameters']['n_layer']:.1f}% of model")
            print()

            print(f"Memory allocation:")
            print(f"  Graph memory: {format_bytes(estimate.graph_size)}")
            print(f"  KV cache: {format_bytes(estimate.vram_kv)}")
            print(f"  Total VRAM: {format_bytes(estimate.vram)}")
            print()

            if estimate.tensor_split:
                print("Tensor split across GPUs:")
                for i, layers in enumerate(estimate.tensor_split):
                    print(f"  GPU {i}: {layers} layers")
                print()

            # Performance estimates
            cpu_layers = overview['parameters']['n_layer'] - estimate.layers
            if cpu_layers > 0:
                print(f"CPU fallback: {cpu_layers} layers ({cpu_layers * 100 / overview['parameters']['n_layer']:.1f}% of model)")
                print("Note: CPU layers will significantly impact inference speed")
            else:
                print("All layers fit in GPU memory - optimal performance expected")

        elif not args.overview_only:
            print("Use --gpu-memory to estimate GPU layer allocation")
            print("Example: --gpu-memory 8192 (for 8GB GPU)")

    except Exception as e:
        print(f"Error during estimation: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())