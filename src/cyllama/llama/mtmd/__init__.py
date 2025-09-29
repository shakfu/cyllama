"""
Multimodal support for cyllama using libmtmd.

This module provides Python wrappers for llama.cpp's multimodal capabilities,
including vision and audio processing through the mtmd library.

Example usage:
    >>> import cyllama
    >>> from cyllama.llama.mtmd import MultimodalProcessor
    >>>
    >>> # Load model and create processor
    >>> model = cyllama.LlamaModel("model.gguf")
    >>> processor = MultimodalProcessor("vision.mmproj", model)
    >>>
    >>> # Process image with text
    >>> response = processor.process_image("What's in this image?", "image.jpg")
    >>> print(response)
"""

from .multimodal import (
    MultimodalProcessor,
    VisionLanguageChat,
    AudioProcessor,
    ImageAnalyzer,
    MultimodalError,
    UnsupportedModalityError,
)

from ..mtmd import (
    MtmdContext,
    MtmdContextParams,
    MtmdBitmap,
    MtmdInputChunk,
    MtmdInputChunks,
    MtmdInputChunkType,
    get_default_media_marker,
)

__all__ = [
    # High-level convenience classes
    "MultimodalProcessor",
    "VisionLanguageChat",
    "AudioProcessor",
    "ImageAnalyzer",

    # Low-level Cython wrappers
    "MtmdContext",
    "MtmdContextParams",
    "MtmdBitmap",
    "MtmdInputChunk",
    "MtmdInputChunks",
    "MtmdInputChunkType",

    # Utilities
    "get_default_media_marker",

    # Exceptions
    "MultimodalError",
    "UnsupportedModalityError",
]