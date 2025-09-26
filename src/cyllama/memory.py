"""GPU Memory estimation for cyllama models.

This module provides functionality to estimate GPU memory requirements
for different model architectures and configurations, helping users
optimize model loading for their hardware.

Adapted from xllamacpp's memory estimation functionality.
"""

import argparse
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class MemoryEstimate:
    """Memory estimation results for model loading."""
    layers: int
    graph_size: int
    vram: int
    vram_kv: int
    total_size: int
    tensor_split: Optional[List[int]] = None


def get_file_host_endian(file_path: Union[str, Path]) -> Tuple[str, str]:
    """Determine file and host endianness."""

    # Host endianness
    host_endian = 'little' if sys.byteorder == 'little' else 'big'

    # File endianness (check GGUF magic)
    try:
        with open(file_path, 'rb') as f:
            magic = f.read(4)
            if magic == b'GGUF':
                file_endian = 'little'
            elif magic == b'FUGG':
                file_endian = 'big'
            else:
                file_endian = 'little'  # default
    except:
        file_endian = 'little'  # default

    return file_endian, host_endian


def dump_metadata_json(model_path: Union[str, Path]) -> Dict:
    """Extract metadata from GGUF model file."""
    try:
        from .llama.llama_cpp import LlamaModel, LlamaModelParams

        # Load model to extract metadata
        params = LlamaModelParams()
        model = LlamaModel(str(model_path), params)

        # Get basic model info
        vocab = model.get_vocab()

        # Extract key metadata
        metadata = {
            'general.architecture': 'llama',  # default
            'llama.context_length': 2048,    # default
            'llama.embedding_length': 4096,  # default
            'llama.block_count': 32,         # default
            'llama.feed_forward_length': 11008,  # default
            'llama.attention.head_count': 32,    # default
            'llama.attention.head_count_kv': 32, # default
            'general.file_type': 1,              # default to Q4_0
        }

        # Try to get actual vocab size
        try:
            metadata['tokenizer.ggml.tokens'] = [f"token_{i}" for i in range(vocab.n_vocab)]
        except:
            metadata['tokenizer.ggml.tokens'] = [f"token_{i}" for i in range(32000)]

        return metadata

    except Exception as e:
        # Fallback metadata for when model can't be loaded
        return {
            'general.architecture': 'llama',
            'llama.context_length': 2048,
            'llama.embedding_length': 4096,
            'llama.block_count': 32,
            'llama.feed_forward_length': 11008,
            'llama.attention.head_count': 32,
            'llama.attention.head_count_kv': 32,
            'general.file_type': 1,
            'tokenizer.ggml.tokens': [f"token_{i}" for i in range(32000)],
        }


def graph_size(
    architecture: str,
    n_layers: int,
    n_embd: int,
    n_ff: int,
    n_head: int,
    n_head_kv: int,
    n_vocab: int,
    n_ctx: int,
    n_batch: int,
    f16_kv: bool = True,
    mul_mat_q: bool = True,
    offload_kqv: bool = True,
    flash_attn: bool = False,
) -> int:
    """Calculate graph memory requirements for different architectures."""

    # Base graph size calculation
    if architecture in ['llama', 'yi', 'deepseek', 'deepseek2']:
        # Standard transformer architecture
        graph_size = (
            n_ctx * n_batch * (n_embd + n_ff) * 4 +  # activations
            n_layers * n_embd * n_embd * 4 +          # attention weights
            n_vocab * n_embd * 4                       # output layer
        )
    elif architecture == 'gemma':
        # Gemma specific calculations
        graph_size = (
            n_ctx * n_batch * (n_embd + n_ff) * 4 +
            n_layers * n_embd * n_embd * 4 +
            n_vocab * n_embd * 4
        )
    elif architecture in ['qwen2', 'qwen2moe']:
        # Qwen2 specific calculations
        graph_size = (
            n_ctx * n_batch * (n_embd + n_ff) * 4 +
            n_layers * n_embd * n_embd * 4 +
            n_vocab * n_embd * 4
        )
    elif architecture == 'stablelm':
        # StableLM specific calculations
        graph_size = (
            n_ctx * n_batch * (n_embd + n_ff) * 4 +
            n_layers * n_embd * n_embd * 4 +
            n_vocab * n_embd * 4
        )
    else:
        # Default calculation for unknown architectures
        graph_size = (
            n_ctx * n_batch * (n_embd + n_ff) * 4 +
            n_layers * n_embd * n_embd * 4 +
            n_vocab * n_embd * 4
        )

    # Apply modifiers
    if flash_attn:
        graph_size = int(graph_size * 0.8)  # Flash attention reduces memory

    if not offload_kqv:
        graph_size = int(graph_size * 1.2)  # No KQV offload increases memory

    # Add some safety margin
    graph_size = int(graph_size * 1.1)

    return graph_size


def projector_memory_requirements(metadata: Dict) -> int:
    """Calculate memory requirements for projector tensors (multimodal models)."""
    # Check if this is a multimodal model
    if 'clip' in metadata.get('general.architecture', '').lower():
        # Estimate CLIP projector size
        return 1024 * 1024 * 100  # ~100MB estimate

    # No projector
    return 0


def estimate_gpu_layers(
    model_path: Union[str, Path],
    gpu_memory_mb: Union[int, List[int]],
    ctx_size: int = 2048,
    batch_size: int = 1,
    n_parallel: int = 1,
    kv_cache_type: str = 'f16',
    use_mmap: bool = True,
    verbose: bool = False
) -> MemoryEstimate:
    """Estimate optimal GPU layer allocation for given memory constraints.

    Args:
        model_path: Path to the GGUF model file
        gpu_memory_mb: Available GPU memory in MB (int for single GPU, list for multi-GPU)
        ctx_size: Context size for inference
        batch_size: Batch size for inference
        n_parallel: Number of parallel sequences
        kv_cache_type: KV cache precision ('f16' or 'f32')
        use_mmap: Whether to use memory mapping
        verbose: Enable verbose output

    Returns:
        MemoryEstimate with allocation details
    """

    # Load model metadata
    metadata = dump_metadata_json(model_path)

    # Extract model parameters
    architecture = metadata.get('general.architecture', 'llama')
    n_ctx_train = metadata.get('llama.context_length', 2048)
    n_embd = metadata.get('llama.embedding_length', 4096)
    n_layer = metadata.get('llama.block_count', 32)
    n_ff = metadata.get('llama.feed_forward_length', 11008)
    n_head = metadata.get('llama.attention.head_count', 32)
    n_head_kv = metadata.get('llama.attention.head_count_kv', 32)
    n_vocab = len(metadata.get('tokenizer.ggml.tokens', [32000]))
    file_type = metadata.get('general.file_type', 1)

    # Adjust context size
    n_ctx = min(ctx_size, n_ctx_train)

    # Calculate KV cache size per layer
    kv_cache_multiplier = 2 if kv_cache_type == 'f32' else 1  # f16 is default
    kv_cache_size_per_layer = (
        n_ctx * batch_size * n_parallel * n_embd * kv_cache_multiplier * 2  # K and V
    )

    # Calculate graph memory requirements
    graph_mem = graph_size(
        architecture=architecture,
        n_layers=n_layer,
        n_embd=n_embd,
        n_ff=n_ff,
        n_head=n_head,
        n_head_kv=n_head_kv,
        n_vocab=n_vocab,
        n_ctx=n_ctx,
        n_batch=batch_size,
        f16_kv=(kv_cache_type == 'f16'),
        offload_kqv=True,
        flash_attn=False,
    )

    # Calculate projector memory
    projector_mem = projector_memory_requirements(metadata)

    # Estimate layer size based on model parameters and quantization
    # This is a rough estimate - actual sizes vary by quantization scheme
    layer_size_mb = (n_embd * n_embd * 4 + n_embd * n_ff * 2) // (1024 * 1024)

    # Adjust for quantization (file_type affects size)
    quantization_factors = {
        0: 1.0,    # F32
        1: 0.5,    # F16
        2: 0.3,    # Q4_0
        3: 0.3,    # Q4_1
        6: 0.2,    # Q5_0
        7: 0.2,    # Q5_1
        8: 0.15,   # Q8_0
    }
    quant_factor = quantization_factors.get(file_type, 0.3)  # default Q4
    layer_size_mb = int(layer_size_mb * quant_factor)

    # Handle multi-GPU scenario
    if isinstance(gpu_memory_mb, list):
        # Multi-GPU setup
        total_gpu_memory = sum(gpu_memory_mb)
        num_gpus = len(gpu_memory_mb)

        # Reserve memory for graph and projector on first GPU
        available_memory = total_gpu_memory - (graph_mem // (1024 * 1024)) - (projector_mem // (1024 * 1024))

        # Calculate how many layers can fit (ensure non-negative)
        if available_memory <= 0 or layer_size_mb + kv_cache_size_per_layer // (1024 * 1024) <= 0:
            max_layers = 0
        else:
            max_layers = min(n_layer, max(0, available_memory // (layer_size_mb + kv_cache_size_per_layer // (1024 * 1024))))

        # Distribute layers across GPUs
        tensor_split = []
        remaining_layers = max_layers
        for i, gpu_mem in enumerate(gpu_memory_mb):
            gpu_layers = min(remaining_layers, remaining_layers // (num_gpus - i))
            tensor_split.append(gpu_layers)
            remaining_layers -= gpu_layers

        vram_total = sum(gpu_memory_mb[i] for i, layers in enumerate(tensor_split) if layers > 0)

    else:
        # Single GPU setup
        available_memory = gpu_memory_mb - (graph_mem // (1024 * 1024)) - (projector_mem // (1024 * 1024))

        # Calculate how many layers can fit (ensure non-negative)
        if available_memory <= 0 or layer_size_mb + kv_cache_size_per_layer // (1024 * 1024) <= 0:
            max_layers = 0
        else:
            max_layers = min(n_layer, max(0, available_memory // (layer_size_mb + kv_cache_size_per_layer // (1024 * 1024))))
        tensor_split = None
        vram_total = gpu_memory_mb if max_layers > 0 else 0

    # Calculate total KV cache size (ensure non-negative)
    vram_kv = max(0, max_layers * kv_cache_size_per_layer)

    # Calculate total model size estimate
    total_size = n_layer * layer_size_mb * 1024 * 1024  # Convert back to bytes

    if verbose:
        print(f"Model: {model_path}")
        print(f"Architecture: {architecture}")
        print(f"Layers: {n_layer}, Embedding: {n_embd}, Vocab: {n_vocab}")
        print(f"Estimated layer size: {layer_size_mb} MB")
        print(f"Graph memory: {graph_mem // (1024 * 1024)} MB")
        print(f"GPU layers: {max_layers}/{n_layer}")
        if tensor_split:
            print(f"Tensor split: {tensor_split}")

    return MemoryEstimate(
        layers=max_layers,
        graph_size=graph_mem,
        vram=vram_total * 1024 * 1024,  # Convert to bytes
        vram_kv=vram_kv,
        total_size=total_size,
        tensor_split=tensor_split
    )


def estimate_memory_usage(
    model_path: Union[str, Path],
    ctx_size: int = 2048,
    batch_size: int = 1,
    verbose: bool = False
) -> Dict:
    """Quick memory usage estimation without GPU constraints.

    Args:
        model_path: Path to the GGUF model file
        ctx_size: Context size for inference
        batch_size: Batch size for inference
        verbose: Enable verbose output

    Returns:
        Dictionary with memory usage estimates
    """

    metadata = dump_metadata_json(model_path)

    n_embd = metadata.get('llama.embedding_length', 4096)
    n_layer = metadata.get('llama.block_count', 32)
    n_ff = metadata.get('llama.feed_forward_length', 11008)
    n_head = metadata.get('llama.attention.head_count', 32)
    n_head_kv = metadata.get('llama.attention.head_count_kv', 32)
    n_vocab = len(metadata.get('tokenizer.ggml.tokens', [32000]))
    file_type = metadata.get('general.file_type', 1)
    architecture = metadata.get('general.architecture', 'llama')

    # Calculate various memory components
    kv_cache_f16 = n_layer * ctx_size * batch_size * n_embd * 2 * 2  # K and V, f16
    kv_cache_f32 = kv_cache_f16 * 2  # f32 is double f16

    graph_mem = graph_size(
        architecture=architecture,
        n_layers=n_layer,
        n_embd=n_embd,
        n_ff=n_ff,
        n_head=n_head,
        n_head_kv=n_head_kv,
        n_vocab=n_vocab,
        n_ctx=ctx_size,
        n_batch=batch_size,
    )

    # Estimate model size
    layer_params = n_embd * n_embd * 4 + n_embd * n_ff * 2  # Rough estimate
    total_params = n_layer * layer_params + n_vocab * n_embd

    # Size in different precisions
    model_size_f32 = total_params * 4
    model_size_f16 = total_params * 2
    model_size_q4 = total_params // 2  # Rough Q4 estimate
    model_size_q8 = total_params        # Rough Q8 estimate

    result = {
        'model_size_mb': {
            'f32': model_size_f32 // (1024 * 1024),
            'f16': model_size_f16 // (1024 * 1024),
            'q4_0': model_size_q4 // (1024 * 1024),
            'q8_0': model_size_q8 // (1024 * 1024),
        },
        'kv_cache_mb': {
            'f16': kv_cache_f16 // (1024 * 1024),
            'f32': kv_cache_f32 // (1024 * 1024),
        },
        'graph_mb': graph_mem // (1024 * 1024),
        'parameters': {
            'n_embd': n_embd,
            'n_layer': n_layer,
            'n_ff': n_ff,
            'n_vocab': n_vocab,
            'total_params': total_params,
        }
    }

    if verbose:
        print(f"Model: {model_path}")
        print(f"Architecture: {architecture}")
        print(f"Parameters: {total_params:,}")
        print(f"Model size estimates:")
        for precision, size in result['model_size_mb'].items():
            print(f"  {precision}: {size} MB")
        print(f"KV cache (ctx={ctx_size}, batch={batch_size}):")
        for precision, size in result['kv_cache_mb'].items():
            print(f"  {precision}: {size} MB")
        print(f"Graph memory: {result['graph_mb']} MB")

    return result


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
    """Command-line interface for GPU memory estimation.

    This utility helps users estimate optimal GPU layer allocation for their models
    and hardware configurations.
    """
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
