"""Tiny random-weight models built in memory with ``LlamaModel.from_user``.

Follows llama.cpp's tests/test-llama-archs.cpp: GGUF metadata for one
architecture plus the "test" tokenizer (token i is "tok_i"). Weights are
seeded per tensor name, so a given (arch, sizes, seed) always builds the same
model. Covers architectures the downloaded test models do not, e.g. M-RoPE.
"""

import random
import struct
import zlib
from typing import Optional

import cyllama.llama.llama_cpp as cy

# ggml_type values
_F32, _F16 = 0, 1


def _fill(seed: int, stdev: float):
    def set_tensor_data(tensor):
        rng = random.Random(seed ^ zlib.crc32(tensor.name.encode()))
        values = [rng.gauss(0.0, stdev) for _ in range(tensor.n_elements)]
        if tensor.type == _F32:
            tensor.set_data(struct.pack(f"<{len(values)}f", *values))
        elif tensor.type == _F16:
            tensor.set_data(struct.pack(f"<{len(values)}e", *values))
        else:
            raise TypeError(f"cannot fill {tensor.name} of type {tensor.type_name}")

    return set_tensor_data


def synthetic_metadata(
    arch: str = "llama",
    *,
    n_vocab: int = 128,
    n_embd: int = 256,
    n_head: int = 2,
    n_head_kv: int = 2,
    n_ff: int = 384,
    n_layer: int = 2,
    n_ctx: int = 256,
) -> cy.GGUFContext:
    """Return GGUF metadata for a tiny model of ``arch``."""
    g = cy.GGUFContext.empty()
    g.set_val_str("general.architecture", arch)
    for key, value in [
        ("vocab_size", n_vocab),
        ("context_length", n_ctx),
        ("embedding_length", n_embd),
        ("feed_forward_length", n_ff),
        ("block_count", n_layer),
        ("attention.head_count", n_head),
        ("attention.head_count_kv", n_head_kv),
    ]:
        g.set_val_u32(f"{arch}.{key}", value)
    g.set_val_f32(f"{arch}.attention.layer_norm_rms_epsilon", 1e-5)
    # M-RoPE sections count rope pairs; read only by M-RoPE architectures
    quarter = n_embd // n_head // 4
    g.set_arr_data(f"{arch}.rope.dimension_sections", cy.GGUF_TYPE_UINT32, [quarter] * 4)
    g.set_val_str("tokenizer.ggml.model", "test")
    g.set_arr_str("tokenizer.ggml.tokens", [f"tok_{i}" for i in range(n_vocab)])
    g.set_arr_data("tokenizer.ggml.scores", cy.GGUF_TYPE_FLOAT32, [0.0] * n_vocab)
    return g


def _set_extra(g: cy.GGUFContext, extra: dict) -> None:
    for key, value in extra.items():
        if isinstance(value, str):
            g.set_val_str(key, value)
        else:
            g.set_val_f32(key, value)


def synthetic_model(
    arch: str = "llama", *, seed: int = 0, stdev: float = 0.1, extra: Optional[dict] = None, **sizes
) -> cy.LlamaModel:
    """Build a tiny random-weight model of ``arch``.

    Args:
        extra: more metadata, str values as strings, others as f32.
        sizes: go to :func:`synthetic_metadata`.
    """
    with synthetic_metadata(arch, **sizes) as metadata:
        _set_extra(metadata, extra or {})
        return cy.LlamaModel.from_user(metadata, _fill(seed, stdev), verbose=False)


def save_synthetic_model(path, arch: str = "llama", *, extra: Optional[dict] = None, **kwargs) -> None:
    """Write :func:`synthetic_model` to a GGUF file, ``extra`` metadata included."""
    path = str(path)
    cy.model_save_to_file(synthetic_model(arch, **kwargs), path)
    # the file has only the keys libllama reads: rewrite its header with the extra keys
    with cy.GGUFContext.from_file(path) as g:
        with open(path, "rb") as f:
            f.seek(g.data_offset)
            data = f.read()
        _set_extra(g, extra or {})
        g.write_to_file(path, only_meta=True)
    with open(path, "ab") as f:
        f.write(data)
