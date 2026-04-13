"""
Central defaults and constants for the cyllama high-level API.

All generation-related default values live here so that ``api.py``,
``__main__.py``, ``batching.py``, agent configs, and integration layers
share a single source of truth.  When llama.cpp upstream changes a
default, only this file needs updating.

The values are chosen to match the llama.cpp C library defaults unless
there is a documented reason to diverge.
"""

# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------
# llama.cpp uses 0xFFFFFFFF as a sentinel meaning "use a random seed
# internally".  Previous cyllama releases used -1 and translated it to
# ``int(time.time())``, which produced low-entropy, second-granularity
# seeds.  Using the C sentinel lets the library handle randomisation
# properly.
LLAMA_DEFAULT_SEED: int = 0xFFFFFFFF

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
DEFAULT_TEMPERATURE: float = 0.8
DEFAULT_TOP_K: int = 40
DEFAULT_TOP_P: float = 0.95
DEFAULT_MIN_P: float = 0.05
DEFAULT_REPEAT_PENALTY: float = 1.0   # 1.0 = disabled (C library default)
DEFAULT_PENALTY_LAST_N: int = 64      # last n tokens to penalize (common.h default)
DEFAULT_PENALTY_FREQ: float = 0.0     # frequency penalty, 0.0 = disabled
DEFAULT_PENALTY_PRESENT: float = 0.0  # presence penalty, 0.0 = disabled

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
DEFAULT_MAX_TOKENS: int = 512

# ---------------------------------------------------------------------------
# Model / context
# ---------------------------------------------------------------------------
DEFAULT_N_GPU_LAYERS: int = -1   # -1 = offload all layers (C library default)
DEFAULT_N_BATCH: int = 2048      # C library default (llama_context_default_params)
DEFAULT_MAIN_GPU: int = 0
DEFAULT_SPLIT_MODE: int = 1      # LLAMA_SPLIT_MODE_LAYER

# n_ctx: intentional divergence from the C library default of 512.
#
# The high-level API defaults to None (auto-sized as prompt_length +
# max_tokens) because the C default of 512 silently truncates any
# generation that exceeds it.  The C API must pick a fixed value at
# struct-init time before the prompt is known; the Python LLM class
# sees the prompt first and can size the context exactly.  The
# tradeoff is higher memory usage per call vs. silent truncation.
