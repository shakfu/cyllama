from .llama.llama_cpp import *

# High-level API (sync and async)
from .api import (
    LLM,
    complete,
    chat,
    simple,
    GenerationConfig,
    GenerationStats,
    Response,
    # Chat template utilities
    apply_chat_template,
    get_chat_template,
    # Async API
    AsyncLLM,
    complete_async,
    chat_async,
    stream_complete_async,
)

# Batching utilities
from .batching import (
    batch_generate,
    BatchGenerator,
    BatchRequest,
    BatchResponse,
)

# Memory utilities
from .memory import (
    estimate_gpu_layers,
    estimate_memory_usage,
    MemoryEstimate,
)

# Agent support
from . import agents

# Utilities (logging, colors)
from . import utils

# Multimodal support (optional import)
try:
    from .llama import mtmd
    HAS_MULTIMODAL = True
except ImportError as e:
    mtmd = None
    HAS_MULTIMODAL = False
    import warnings
    warnings.warn(f"Multimodal support not available: {e}", ImportWarning)

__version__ = "0.1.16"
