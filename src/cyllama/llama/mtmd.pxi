# mtmd.pyx - Cython implementation of libmtmd multimodal support
#
# This file provides Python/Cython wrappers for the mtmd C API from llama.cpp
# Includes support for vision and audio multimodal capabilities

# distutils: language = c++
# cython: language_level = 3

import os
from typing import List, Optional, Union, Tuple, Any
from enum import IntEnum

cimport cython
from cython cimport view
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.stdint cimport uint32_t, int32_t, uintptr_t
from libc.stddef cimport size_t

from .mtmd cimport *
# from .llama cimport llama_model, llama_context, llama_token, llama_pos, llama_seq_id
# from .ggml cimport ggml_log_level


class MtmdInputChunkType(IntEnum):
    """Enum for mtmd input chunk types."""
    TEXT = MTMD_INPUT_CHUNK_TYPE_TEXT
    IMAGE = MTMD_INPUT_CHUNK_TYPE_IMAGE
    AUDIO = MTMD_INPUT_CHUNK_TYPE_AUDIO


cdef class MtmdContextParams:
    """Parameters for creating an mtmd context."""

    cdef mtmd_context_params _params

    def __init__(self, use_gpu: bool = True, print_timings: bool = False,
                 n_threads: int = 1, media_marker: str = None):
        """Initialize mtmd context parameters.

        Args:
            use_gpu: Whether to use GPU acceleration
            print_timings: Whether to print timing information
            n_threads: Number of threads for processing
            media_marker: Custom media marker (defaults to mtmd default)
        """
        self._params = mtmd_context_params_default()
        self._params.use_gpu = use_gpu
        self._params.print_timings = print_timings
        self._params.n_threads = n_threads

        if media_marker is not None:
            # Store the marker (Note: this requires careful memory management)
            marker_bytes = media_marker.encode('utf-8')
            self._params.media_marker = marker_bytes

    @property
    def use_gpu(self) -> bool:
        return self._params.use_gpu

    @use_gpu.setter
    def use_gpu(self, value: bool):
        self._params.use_gpu = value

    @property
    def print_timings(self) -> bool:
        return self._params.print_timings

    @print_timings.setter
    def print_timings(self, value: bool):
        self._params.print_timings = value

    @property
    def n_threads(self) -> int:
        return self._params.n_threads

    @n_threads.setter
    def n_threads(self, value: int):
        self._params.n_threads = value


cdef class MtmdBitmap:
    """Wrapper for mtmd_bitmap structure."""

    cdef mtmd_bitmap * _bitmap
    cdef bint _owner

    def __init__(self):
        self._bitmap = NULL
        self._owner = False

    def __dealloc__(self):
        if self._bitmap is not NULL and self._owner:
            mtmd_bitmap_free(self._bitmap)

    @staticmethod
    def create_image(width: int, height: int, data: bytes) -> MtmdBitmap:
        """Create a bitmap from image data.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            data: RGB image data (width * height * 3 bytes)

        Returns:
            MtmdBitmap instance
        """
        cdef MtmdBitmap bitmap = MtmdBitmap()
        cdef bytes _data = <bytes>data
        cdef const unsigned char* data_ptr = <const unsigned char*>_data
        bitmap._bitmap = mtmd_bitmap_init(<uint32_t>width, <uint32_t>height, data_ptr)
        bitmap._owner = True

        if bitmap._bitmap is NULL:
            raise RuntimeError("Failed to create image bitmap")

        return bitmap

    @staticmethod
    def create_audio(samples: List[float]) -> MtmdBitmap:
        """Create a bitmap from audio data.

        Args:
            samples: List of float audio samples (PCM F32 format)

        Returns:
            MtmdBitmap instance
        """
        cdef MtmdBitmap bitmap = MtmdBitmap()
        cdef size_t n_samples = len(samples)
        cdef float* data_ptr = <float*>malloc(n_samples * sizeof(float))

        if data_ptr is NULL:
            raise MemoryError("Failed to allocate memory for audio data")

        try:
            # Copy samples to C array
            for i in range(n_samples):
                data_ptr[i] = samples[i]

            bitmap._bitmap = mtmd_bitmap_init_from_audio(n_samples, data_ptr)
            bitmap._owner = True

            if bitmap._bitmap is NULL:
                raise RuntimeError("Failed to create audio bitmap")

        finally:
            free(data_ptr)

        return bitmap

    @staticmethod
    def from_file(mtmd_ctx, file_path: str) -> MtmdBitmap:
        """Load bitmap from file.

        Args:
            mtmd_ctx: MtmdContext instance
            file_path: Path to image or audio file

        Returns:
            MtmdBitmap instance
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        cdef MtmdBitmap bitmap = MtmdBitmap()
        cdef MtmdContext ctx = <MtmdContext>mtmd_ctx
        cdef bytes path_bytes = file_path.encode('utf-8')

        bitmap._bitmap = mtmd_helper_bitmap_init_from_file(ctx._ctx, path_bytes)
        bitmap._owner = True

        if bitmap._bitmap is NULL:
            raise RuntimeError(f"Failed to load bitmap from file: {file_path}")

        return bitmap

    @staticmethod
    def from_buffer(mtmd_ctx, data: bytes) -> MtmdBitmap:
        """Load bitmap from buffer.

        Args:
            mtmd_ctx: MtmdContext instance
            data: File data buffer

        Returns:
            MtmdBitmap instance
        """
        cdef MtmdBitmap bitmap = MtmdBitmap()
        cdef MtmdContext ctx = <MtmdContext>mtmd_ctx
        cdef const unsigned char* buf_ptr = <const unsigned char*>data
        cdef size_t buf_len = len(data)

        bitmap._bitmap = mtmd_helper_bitmap_init_from_buf(ctx._ctx, buf_ptr, buf_len)
        bitmap._owner = True

        if bitmap._bitmap is NULL:
            raise RuntimeError("Failed to load bitmap from buffer")

        return bitmap

    @property
    def width(self) -> int:
        """Get bitmap width."""
        if self._bitmap is NULL:
            raise RuntimeError("Bitmap not initialized")
        return mtmd_bitmap_get_nx(self._bitmap)

    @property
    def height(self) -> int:
        """Get bitmap height."""
        if self._bitmap is NULL:
            raise RuntimeError("Bitmap not initialized")
        return mtmd_bitmap_get_ny(self._bitmap)

    @property
    def data(self) -> bytes:
        """Get bitmap data as bytes."""
        if self._bitmap is NULL:
            raise RuntimeError("Bitmap not initialized")

        cdef const unsigned char* data_ptr = mtmd_bitmap_get_data(self._bitmap)
        cdef size_t n_bytes = mtmd_bitmap_get_n_bytes(self._bitmap)

        return (<char*>data_ptr)[:n_bytes]

    @property
    def is_audio(self) -> bool:
        """Check if this is an audio bitmap."""
        if self._bitmap is NULL:
            raise RuntimeError("Bitmap not initialized")
        return mtmd_bitmap_is_audio(self._bitmap)

    @property
    def id(self) -> Optional[str]:
        """Get bitmap ID."""
        if self._bitmap is NULL:
            raise RuntimeError("Bitmap not initialized")

        cdef const char* id_ptr = mtmd_bitmap_get_id(self._bitmap)
        if id_ptr is NULL:
            return None
        return id_ptr.decode('utf-8')

    @id.setter
    def id(self, value: str):
        """Set bitmap ID."""
        if self._bitmap is NULL:
            raise RuntimeError("Bitmap not initialized")

        cdef bytes id_bytes = value.encode('utf-8')
        mtmd_bitmap_set_id(self._bitmap, id_bytes)


cdef class MtmdInputChunk:
    """Wrapper for mtmd_input_chunk structure."""

    cdef const mtmd_input_chunk * _chunk
    cdef bint _owner

    def __init__(self):
        self._chunk = NULL
        self._owner = False

    def __dealloc__(self):
        if self._chunk is not NULL and self._owner:
            mtmd_input_chunk_free(<mtmd_input_chunk*>self._chunk)

    @property
    def type(self) -> MtmdInputChunkType:
        """Get the chunk type."""
        if self._chunk is NULL:
            raise RuntimeError("Chunk not initialized")
        return MtmdInputChunkType(mtmd_input_chunk_get_type(self._chunk))

    @property
    def n_tokens(self) -> int:
        """Get number of tokens in this chunk."""
        if self._chunk is NULL:
            raise RuntimeError("Chunk not initialized")
        return mtmd_input_chunk_get_n_tokens(self._chunk)

    @property
    def n_pos(self) -> int:
        """Get number of positions in this chunk."""
        if self._chunk is NULL:
            raise RuntimeError("Chunk not initialized")
        return mtmd_input_chunk_get_n_pos(self._chunk)

    @property
    def id(self) -> Optional[str]:
        """Get chunk ID (None for text chunks)."""
        if self._chunk is NULL:
            raise RuntimeError("Chunk not initialized")

        cdef const char* id_ptr = mtmd_input_chunk_get_id(self._chunk)
        if id_ptr is NULL:
            return None
        return id_ptr.decode('utf-8')

    def get_text_tokens(self) -> List[int]:
        """Get text tokens from this chunk."""
        if self._chunk is NULL:
            raise RuntimeError("Chunk not initialized")

        if self.type != MtmdInputChunkType.TEXT:
            raise ValueError("This is not a text chunk")

        cdef size_t n_tokens_out
        cdef const llama_token* tokens = mtmd_input_chunk_get_tokens_text(self._chunk, &n_tokens_out)

        if tokens is NULL:
            return []

        return [tokens[i] for i in range(n_tokens_out)]


cdef class MtmdInputChunks:
    """Wrapper for mtmd_input_chunks structure."""

    cdef mtmd_input_chunks * _chunks
    cdef bint _owner

    def __init__(self):
        self._chunks = mtmd_input_chunks_init()
        self._owner = True

        if self._chunks is NULL:
            raise RuntimeError("Failed to initialize input chunks")

    def __dealloc__(self):
        if self._chunks is not NULL and self._owner:
            mtmd_input_chunks_free(self._chunks)

    def __len__(self) -> int:
        """Get number of chunks."""
        if self._chunks is NULL:
            return 0
        return mtmd_input_chunks_size(self._chunks)

    def __getitem__(self, idx: int) -> MtmdInputChunk:
        """Get chunk by index."""
        if self._chunks is NULL:
            raise RuntimeError("Chunks not initialized")

        cdef size_t size = mtmd_input_chunks_size(self._chunks)
        if idx < 0 or idx >= size:
            raise IndexError(f"Index {idx} out of range [0, {size})")

        cdef MtmdInputChunk chunk = MtmdInputChunk()
        chunk._chunk = mtmd_input_chunks_get(self._chunks, idx)
        chunk._owner = False  # Managed by chunks container

        return chunk

    @property
    def total_tokens(self) -> int:
        """Get total number of tokens across all chunks."""
        if self._chunks is NULL:
            return 0
        return mtmd_helper_get_n_tokens(self._chunks)

    @property
    def total_positions(self) -> int:
        """Get total number of positions across all chunks."""
        if self._chunks is NULL:
            return 0
        return mtmd_helper_get_n_pos(self._chunks)


cdef class MtmdContext:
    """Main multimodal context for libmtmd."""

    cdef mtmd_context * _ctx
    cdef object _model_ref  # Keep reference to prevent GC

    def __init__(self, mmproj_path: str, llama_model: LlamaModel, params: MtmdContextParams = None):
        """Initialize mtmd context.

        Args:
            mmproj_path: Path to multimodal projector file (.mmproj)
            llama_model: LlamaModel instance
            params: Optional context parameters
        """
        if not os.path.exists(mmproj_path):
            raise FileNotFoundError(f"Multimodal projector file not found: {mmproj_path}")

        if params is None:
            params = MtmdContextParams()

        self._model_ref = llama_model
        cdef bytes path_bytes = mmproj_path.encode('utf-8')

        # Get the underlying llama_model pointer
        # Use Python object attribute access to get the pointer value
        cdef uintptr_t model_ptr_value = <uintptr_t>llama_model.ptr
        cdef llama.llama_model* model_ptr = <llama.llama_model*>model_ptr_value

        self._ctx = mtmd_init_from_file(path_bytes, model_ptr, params._params)

        if self._ctx is NULL:
            raise RuntimeError(f"Failed to initialize mtmd context from: {mmproj_path}")

    def __dealloc__(self):
        if self._ctx is not NULL:
            mtmd_free(self._ctx)

    @property
    def supports_vision(self) -> bool:
        """Check if the model supports vision input."""
        if self._ctx is NULL:
            return False
        return mtmd_support_vision(self._ctx)

    @property
    def supports_audio(self) -> bool:
        """Check if the model supports audio input."""
        if self._ctx is NULL:
            return False
        return mtmd_support_audio(self._ctx)

    @property
    def audio_bitrate(self) -> int:
        """Get audio bitrate in Hz (-1 if audio not supported)."""
        if self._ctx is NULL:
            return -1
        return mtmd_get_audio_bitrate(self._ctx)

    @property
    def uses_non_causal(self) -> bool:
        """Check if model requires non-causal attention for decode."""
        if self._ctx is NULL:
            return False
        return mtmd_decode_use_non_causal(self._ctx)

    @property
    def uses_mrope(self) -> bool:
        """Check if model uses M-RoPE for decode."""
        if self._ctx is NULL:
            return False
        return mtmd_decode_use_mrope(self._ctx)

    def tokenize(self, text: str, bitmaps: List[MtmdBitmap],
                 add_special: bool = True, parse_special: bool = True) -> MtmdInputChunks:
        """Tokenize text with multimodal content.

        Args:
            text: Input text with media markers
            bitmaps: List of MtmdBitmap objects (images/audio)
            add_special: Whether to add special tokens
            parse_special: Whether to parse special tokens

        Returns:
            MtmdInputChunks containing the tokenized input
        """
        cdef MtmdBitmap bitmap_obj
        if self._ctx is NULL:
            raise RuntimeError("Context not initialized")

        # Prepare input text structure
        cdef mtmd_input_text input_text
        cdef bytes text_bytes = text.encode('utf-8')
        input_text.text = text_bytes
        input_text.add_special = add_special
        input_text.parse_special = parse_special

        # Prepare bitmap pointers
        cdef size_t n_bitmaps = len(bitmaps)
        cdef mtmd_bitmap** bitmap_ptrs = NULL

        if n_bitmaps > 0:
            bitmap_ptrs = <mtmd_bitmap**>malloc(n_bitmaps * sizeof(mtmd_bitmap*))
            if bitmap_ptrs is NULL:
                raise MemoryError("Failed to allocate bitmap pointers")

            # Pre-declare the variable outside the loop
            for i in range(n_bitmaps):
                bitmap_obj = bitmaps[i]
                bitmap_ptrs[i] = bitmap_obj._bitmap

        # Create output chunks
        cdef MtmdInputChunks chunks = MtmdInputChunks()
        cdef int32_t result

        try:
            # Perform tokenization
            result = mtmd_tokenize(self._ctx, chunks._chunks, &input_text,
                                   <const mtmd_bitmap**>bitmap_ptrs, n_bitmaps)

            if result != 0:
                if result == 1:
                    raise ValueError("Number of bitmaps does not match number of markers in text")
                elif result == 2:
                    raise RuntimeError("Image preprocessing error")
                else:
                    raise RuntimeError(f"Tokenization failed with error code: {result}")

            return chunks

        finally:
            if bitmap_ptrs is not NULL:
                free(bitmap_ptrs)

    def encode_chunk(self, chunk: MtmdInputChunk) -> int:
        """Encode a single input chunk.

        Args:
            chunk: Input chunk to encode

        Returns:
            0 on success, non-zero on error
        """
        if self._ctx is NULL:
            raise RuntimeError("Context not initialized")

        return mtmd_encode_chunk(self._ctx, chunk._chunk)

    def get_output_embeddings(self, n_tokens: int, n_embd: int) -> List[List[float]]:
        """Get output embeddings from the last encode pass.

        Args:
            n_tokens: Number of tokens
            n_embd: Embedding dimension

        Returns:
            List of embedding vectors
        """
        if self._ctx is NULL:
            raise RuntimeError("Context not initialized")

        cdef float* embd_ptr = mtmd_get_output_embd(self._ctx)
        if embd_ptr is NULL:
            raise RuntimeError("No embeddings available")

        # Convert to Python list of lists
        embeddings = []
        for i in range(n_tokens):
            token_embd = []
            for j in range(n_embd):
                token_embd.append(embd_ptr[i * n_embd + j])
            embeddings.append(token_embd)

        return embeddings

    def eval_chunks(self, llama_ctx, chunks: MtmdInputChunks, n_past: int = 0,
                    seq_id: int = 0, n_batch: int = 32, logits_last: bool = True) -> int:
        """Evaluate chunks using helper function.

        Args:
            llama_ctx: LlamaContext instance
            chunks: Input chunks to evaluate
            n_past: Number of past tokens
            seq_id: Sequence ID
            n_batch: Batch size
            logits_last: Whether to compute logits only for last token

        Returns:
            New n_past value after evaluation
        """
        if self._ctx is NULL:
            raise RuntimeError("Context not initialized")

        # Get the underlying llama_context pointer
        # Use Python object attribute access to get the pointer value
        cdef uintptr_t ctx_ptr_value = <uintptr_t>llama_ctx.ptr
        cdef llama_context* ctx_ptr = <llama_context*>ctx_ptr_value

        cdef llama_pos new_n_past
        cdef int32_t result = mtmd_helper_eval_chunks(
            self._ctx, ctx_ptr, chunks._chunks, <llama_pos>n_past,
            <llama_seq_id>seq_id, <int32_t>n_batch, logits_last, &new_n_past
        )

        if result != 0:
            raise RuntimeError(f"Chunk evaluation failed with error code: {result}")

        return new_n_past


def get_default_media_marker() -> str:
    """Get the default media marker string."""
    cdef const char* marker = mtmd_default_marker()
    return marker.decode('utf-8')