from .api import simple
from .llama.llama_cpp import *

# High-level generation APIs
from .generate import (
    generate,
    chat,
    Generator,
    GenerationConfig,
    GenerationStats,
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

# Multimodal support (optional import)
try:
    from .llama import mtmd
    HAS_MULTIMODAL = True
except ImportError as e:
    mtmd = None
    HAS_MULTIMODAL = False
    import warnings
    warnings.warn(f"Multimodal support not available: {e}", ImportWarning)
