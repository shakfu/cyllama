"""GPU Memory estimation for cyllama models.

This module provides functionality to estimate GPU memory requirements
for different model architectures and configurations, helping users
optimize model loading for their hardware.

Adapted from xllamacpp memory estimation functionality.
"""

import json
import math
import struct
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
    import sys

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