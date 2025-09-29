from .api import simple
from .llama.llama_cpp import *

# Multimodal support (optional import)
try:
    from .llama import mtmd
    HAS_MULTIMODAL = True
except ImportError as e:
    mtmd = None
    HAS_MULTIMODAL = False
    import warnings
    warnings.warn(f"Multimodal support not available: {e}", ImportWarning)
