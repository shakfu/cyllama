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
from libc.stdint cimport uint32_t, int32_t
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
    # Retains the encoded media_marker so the const char* stored in
    # _params.media_marker stays valid for the lifetime of the params
    # object. Without this, the bytes object is collected as soon as the
    # __init__ frame returns and the C pointer dangles.
    cdef bytes _media_marker_bytes

    def __init__(self, use_gpu: bool = True, print_timings: bool = False,
                 n_threads: int = 1, media_marker: str = None,
                 flash_attn_type: int = 0, image_min_tokens: int = -1,
                 image_max_tokens: int = -1, warmup: bool = True,
                 batch_max_tokens: int = None):
        """Initialize mtmd context parameters.

        Args:
            use_gpu: Whether to use GPU acceleration
            print_timings: Whether to print timing information
            n_threads: Number of threads for processing
            media_marker: Custom media marker (defaults to mtmd default)
            flash_attn_type: Flash attention type (0=auto, see llama_flash_attn_type enum)
            image_min_tokens: Minimum number of tokens for image input (-1=from metadata)
            image_max_tokens: Maximum number of tokens for image input (-1=from metadata)
            warmup: Whether to run a warmup encode pass after initialization
            batch_max_tokens: Soft cap on output tokens per encode batch
                (None=keep mtmd default of 1024; the first image is always
                added even if it exceeds this limit)
        """
        self._params = mtmd_context_params_default()
        self._params.use_gpu = use_gpu
        self._params.print_timings = print_timings
        self._params.n_threads = n_threads
        self._params.flash_attn_type = <llama_flash_attn_type>flash_attn_type
        self._params.image_min_tokens = image_min_tokens
        self._params.image_max_tokens = image_max_tokens
        self._params.warmup = warmup

        if batch_max_tokens is not None:
            self._params.batch_max_tokens = batch_max_tokens

        if media_marker is not None:
            self._media_marker_bytes = media_marker.encode('utf-8')
            self._params.media_marker = self._media_marker_bytes

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

    @property
    def flash_attn_type(self) -> int:
        return self._params.flash_attn_type

    @flash_attn_type.setter
    def flash_attn_type(self, value: int):
        self._params.flash_attn_type = <llama_flash_attn_type>value

    @property
    def image_min_tokens(self) -> int:
        return self._params.image_min_tokens

    @image_min_tokens.setter
    def image_min_tokens(self, value: int):
        self._params.image_min_tokens = value

    @property
    def image_max_tokens(self) -> int:
        return self._params.image_max_tokens

    @image_max_tokens.setter
    def image_max_tokens(self, value: int):
        self._params.image_max_tokens = value

    @property
    def warmup(self) -> bool:
        """Whether to run a warmup encode pass after initialization."""
        return self._params.warmup

    @warmup.setter
    def warmup(self, value: bool):
        self._params.warmup = value

    @property
    def batch_max_tokens(self) -> int:
        """Soft cap on output tokens per encode batch (default: 1024)."""
        return self._params.batch_max_tokens

    @batch_max_tokens.setter
    def batch_max_tokens(self, value: int):
        self._params.batch_max_tokens = value


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
        if width <= 0 or height <= 0:
            raise ValueError(
                f"width and height must be positive (got width={width}, height={height})"
            )
        # mtmd_bitmap_init takes a raw unsigned char* with no length and reads
        # width * height * 3 bytes (RGB, 1 byte/channel). Validate up front so
        # short buffers raise a clean Python error instead of letting the C
        # side read past the bytes object.
        cdef Py_ssize_t expected = <Py_ssize_t>width * <Py_ssize_t>height * 3
        if len(data) < expected:
            raise ValueError(
                f"data buffer too small for {width}x{height} RGB image: "
                f"need {expected} bytes, got {len(data)}"
            )
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

        bitmap._bitmap = mtmd_helper_bitmap_init_from_file(ctx._ctx, path_bytes, False).bitmap
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

        bitmap._bitmap = mtmd_helper_bitmap_init_from_buf(ctx._ctx, buf_ptr, buf_len, False).bitmap
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
        cdef llama.llama_model* model_ptr = llama_model.ptr

        self._ctx = mtmd_init_from_file(path_bytes, model_ptr, params._params)

        if self._ctx is NULL:
            raise RuntimeError(f"Failed to initialize mtmd context from: {mmproj_path}")

    def __dealloc__(self):
        if self._ctx is not NULL:
            mtmd_free(self._ctx)
            self._ctx = NULL

    def close(self):
        """Release the underlying mtmd context immediately. Idempotent."""
        if self._ctx is not NULL:
            mtmd_free(self._ctx)
            self._ctx = NULL

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

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
    def audio_sample_rate(self) -> int:
        """Get audio sample rate in Hz (-1 if audio not supported)."""
        if self._ctx is NULL:
            return -1
        return mtmd_get_audio_sample_rate(self._ctx)

    @property
    def marker(self) -> str:
        """Get the media marker string used by this context.

        This is the marker (e.g. the default from ``get_default_media_marker``
        or a custom one set via ``MtmdContextParams``) that must appear in the
        prompt text wherever media is to be inserted.
        """
        if self._ctx is NULL:
            raise RuntimeError("Context not initialized")
        cdef const char* marker = mtmd_get_marker(self._ctx)
        if marker is NULL:
            raise RuntimeError("Failed to get media marker")
        return marker.decode('utf-8')

    @property
    def uses_non_causal(self) -> bool:
        """Check if model requires non-causal attention for decode."""
        if self._ctx is NULL:
            return False
        return mtmd_decode_use_non_causal(self._ctx, NULL)

    @property
    def uses_mrope(self) -> bool:
        """Check if model uses M-RoPE for decode."""
        if self._ctx is NULL:
            return False
        return mtmd_decode_use_mrope(self._ctx)

    def model_can_chat(self, LlamaContext llama_ctx) -> bool:
        """Whether the paired text model can be used for chat.

        Some mmproj/model pairings are encode-only (embedding or
        classification projectors) and have no usable chat path.
        """
        if self._ctx is NULL:
            raise RuntimeError("Context not initialized")
        return mtmd_helper_model_can_chat(llama_ctx.ptr, self._ctx)

    def batch(self) -> MtmdBatch:
        """Create a batch for encoding several media chunks in one pass.

        See :class:`MtmdBatch`.
        """
        if self._ctx is NULL:
            raise RuntimeError("Context not initialized")
        cdef mtmd_batch * b = mtmd_batch_init(self._ctx)
        if b is NULL:
            raise RuntimeError("Failed to create mtmd batch")
        cdef MtmdBatch batch = MtmdBatch.__new__(MtmdBatch)
        batch._batch = b
        batch._ctx_ref = self
        return batch

    def open_video(self, path: str, fps_target: float = 0.0,
                   ffmpeg_bin_dir: Optional[str] = None,
                   timestamp_interval_ms: int = 0) -> MtmdVideo:
        """Open a video file and iterate it as frames plus timestamp text.

        Requires ``ffmpeg`` and ``ffprobe`` on the system. See
        :class:`MtmdVideo` for the read loop.

        Args:
            path: Path to the video file.
            fps_target: Sampling rate in frames per second; <= 0 uses the
                helper default (4 fps).
            ffmpeg_bin_dir: Directory holding the ffmpeg/ffprobe binaries;
                None searches PATH.
            timestamp_interval_ms: Emit a text chunk like ``"[10m50.5s]"``
                at this interval; <= 0 uses the helper default (5000 ms).
        """
        if self._ctx is NULL:
            raise RuntimeError("Context not initialized")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")

        cdef mtmd_helper_video_init_params params = mtmd_helper_video_init_params_default()
        if fps_target > 0:
            params.fps_target = fps_target
        if timestamp_interval_ms > 0:
            params.timestamp_interval_ms = <int64_t>timestamp_interval_ms

        cdef bytes dir_bytes
        if ffmpeg_bin_dir is not None:
            dir_bytes = ffmpeg_bin_dir.encode('utf-8')
            params.ffmpeg_bin_dir = <const char*>dir_bytes

        cdef bytes path_bytes = path.encode('utf-8')
        cdef const char * c_path = <const char*>path_bytes
        cdef mtmd_context * ctx = self._ctx
        cdef mtmd_helper_video * vid
        with nogil:
            vid = mtmd_helper_video_init(ctx, c_path, params)
        if vid is NULL:
            raise RuntimeError(
                f"Failed to open video: {path}. Check that the file is readable "
                "and that ffmpeg/ffprobe are installed and on PATH "
                "(or pass ffmpeg_bin_dir)."
            )

        cdef MtmdVideo video = MtmdVideo.__new__(MtmdVideo)
        video._video = vid
        video._ctx_ref = self
        return video

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
        input_text.text_len = len(text_bytes)
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

    def eval_chunks(self, LlamaContext llama_ctx, MtmdInputChunks chunks, n_past: int = 0,
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

        cdef llama_context* ctx_ptr = llama_ctx.ptr

        cdef llama_pos new_n_past
        cdef int32_t result = mtmd_helper_eval_chunks(
            self._ctx, ctx_ptr, chunks._chunks, <llama_pos>n_past,
            <llama_seq_id>seq_id, <int32_t>n_batch, logits_last, &new_n_past
        )

        if result != 0:
            raise RuntimeError(f"Chunk evaluation failed with error code: {result}")

        return new_n_past


cdef class MtmdBatch:
    """Encode several media chunks in a single mmproj pass.

    The per-chunk :meth:`MtmdContext.encode` path runs the projector once
    per image; batching amortises that over several chunks, which matters
    for multi-image prompts and for video, where every sampled frame is
    its own chunk.

    Chunks are borrowed, not owned: the batch never frees them, so the
    :class:`MtmdInputChunks` they came from must outlive the batch.

    Create via :meth:`MtmdContext.batch`; a batch belongs to the context
    that made it and cannot be shared across contexts.

    Example:
        >>> with ctx.batch() as batch:
        ...     for chunk in chunks:
        ...         if chunk.type != MtmdInputChunkType.TEXT:
        ...             batch.add_chunk(chunk)
        ...     batch.encode()
        ...     embd = batch.get_output_embd(chunks[1], model.n_embd)
    """

    cdef mtmd_batch * _batch
    cdef object _ctx_ref  # keeps the owning MtmdContext alive

    def __cinit__(self):
        self._batch = NULL
        self._ctx_ref = None

    def __dealloc__(self):
        if self._batch is not NULL:
            mtmd_batch_free(self._batch)
            self._batch = NULL

    def add_chunk(self, MtmdInputChunk chunk) -> None:
        """Add a media chunk to the batch.

        Text chunks are rejected -- they need no projector pass.

        Raises:
            ValueError: The chunk is a text chunk, or does not fit.
            RuntimeError: The batch is closed or the chunk is uninitialized.
        """
        if self._batch is NULL:
            raise RuntimeError("Batch is closed")
        if chunk._chunk is NULL:
            raise RuntimeError("Chunk not initialized")

        cdef int32_t result = mtmd_batch_add_chunk(self._batch, chunk._chunk)
        if result == 0:
            return
        if result == 2:
            raise ValueError(
                "Chunk does not fit in the batch; encode the batch and start a new one"
            )
        if result == 3:
            raise ValueError(
                "Chunk cannot be batched with the chunks already added "
                "(incompatible media type or geometry); use a separate batch"
            )
        raise ValueError(
            f"Failed to add chunk to batch (error {result}); "
            "note that text chunks are not accepted"
        )

    def encode(self) -> None:
        """Run the projector over every chunk added so far.

        Raises:
            RuntimeError: Encoding failed.
        """
        if self._batch is NULL:
            raise RuntimeError("Batch is closed")

        cdef mtmd_batch * b = self._batch
        cdef int32_t result
        with nogil:
            result = mtmd_batch_encode(b)
        if result != 0:
            raise RuntimeError(f"Batch encoding failed with error code: {result}")

    def get_output_embd(self, MtmdInputChunk chunk, n_embd: int) -> List[List[float]]:
        """Embeddings produced for ``chunk`` by the last :meth:`encode`.

        Args:
            chunk: A chunk previously passed to :meth:`add_chunk`.
            n_embd: Embedding dimension of the text model, i.e.
                ``LlamaModel.n_embd``. mtmd does not expose it on the batch,
                so it must be supplied -- same convention as
                :meth:`MtmdContext.get_output_embd`.

        Returns:
            ``chunk.n_tokens`` vectors of ``n_embd`` floats each. The native
            buffer is owned by the batch and invalidated by the next
            :meth:`encode` or by :meth:`close`, so values are copied out.
        """
        if self._batch is NULL:
            raise RuntimeError("Batch is closed")
        if chunk._chunk is NULL:
            raise RuntimeError("Chunk not initialized")
        if n_embd <= 0:
            raise ValueError(f"n_embd must be positive, got {n_embd}")

        cdef float * embd = mtmd_batch_get_output_embd(self._batch, chunk._chunk)
        if embd is NULL:
            raise RuntimeError(
                "No output embeddings for this chunk; was it added to the "
                "batch and was encode() called?"
            )

        cdef size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk._chunk)
        cdef size_t i, j
        cdef size_t c_n_embd = <size_t>n_embd
        embeddings = []
        for i in range(n_tokens):
            embeddings.append([embd[i * c_n_embd + j] for j in range(c_n_embd)])
        return embeddings

    def close(self) -> None:
        """Release the native batch. Idempotent."""
        if self._batch is not NULL:
            mtmd_batch_free(self._batch)
            self._batch = NULL
        self._ctx_ref = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


cdef class MtmdVideo:
    """Frame-by-frame reader over a video file, decoded via ffmpeg.

    Video exists only at the mtmd helper level: this yields ordinary image
    bitmaps that the vision projector handles like any other image, plus
    optional timestamp text chunks so the model can reason about when a
    frame occurred.

    Create via :meth:`MtmdContext.open_video`. Iterating yields
    ``(kind, value)`` pairs where ``kind`` is ``"image"`` with an
    :class:`MtmdBitmap`, or ``"text"`` with a ``str``.

    Example:
        >>> with ctx.open_video("clip.mp4", fps_target=1.0) as video:
        ...     print(video.info)
        ...     for kind, value in video:
        ...         ...
    """

    cdef mtmd_helper_video * _video
    cdef object _ctx_ref  # keeps the owning MtmdContext alive

    def __cinit__(self):
        self._video = NULL
        self._ctx_ref = None

    def __dealloc__(self):
        if self._video is not NULL:
            mtmd_helper_video_free(self._video)
            self._video = NULL

    @property
    def info(self) -> dict:
        """Video geometry: ``width``, ``height``, ``fps``, ``n_frames``.

        ``fps`` is the effective sampling rate, and ``n_frames`` is an
        estimate at that rate (-1 when ffprobe could not determine it).
        """
        if self._video is NULL:
            raise RuntimeError("Video is closed")
        cdef mtmd_helper_video_info inf = mtmd_helper_video_get_info(self._video)
        return {
            "width": inf.width,
            "height": inf.height,
            "fps": inf.fps,
            "n_frames": inf.n_frames,
        }

    def read_next(self):
        """Read the next item from the stream.

        Returns:
            ``("image", MtmdBitmap)`` or ``("text", str)`` for each item,
            or ``None`` at end of stream.

        Raises:
            RuntimeError: The decoder reported an error.
        """
        if self._video is NULL:
            raise RuntimeError("Video is closed")

        cdef mtmd_bitmap * out_bitmap = NULL
        cdef char * out_text = NULL
        cdef mtmd_helper_video * vid = self._video
        cdef int32_t result
        with nogil:
            result = mtmd_helper_video_read_next(vid, &out_bitmap, &out_text)

        if result == -1:
            return None
        if result != 0:
            raise RuntimeError(f"Video read failed with error code: {result}")

        if out_bitmap is not NULL:
            bitmap = MtmdBitmap()
            bitmap._bitmap = out_bitmap
            bitmap._owner = True
            return ("image", bitmap)

        if out_text is not NULL:
            try:
                text = out_text.decode('utf-8')
            finally:
                # Documented as heap-allocated via strdup/malloc.
                free(out_text)
            return ("text", text)

        # Neither output set: treat as end of stream rather than looping.
        return None

    def __iter__(self):
        while True:
            item = self.read_next()
            if item is None:
                return
            yield item

    def close(self) -> None:
        """Release the native video reader. Idempotent."""
        if self._video is not NULL:
            mtmd_helper_video_free(self._video)
            self._video = NULL
        self._ctx_ref = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def get_default_media_marker() -> str:
    """Get the default media marker string."""
    cdef const char* marker = mtmd_default_marker()
    return marker.decode('utf-8')


def get_mmproj_caps(mmproj_path: str) -> dict:
    """Report an mmproj file's input modalities without loading a full context.

    Cheap way to decide whether a projector is usable for a given input
    before paying for :class:`MtmdContext` initialization.

    Returns:
        A dict with ``vision`` and ``audio`` booleans.
    """
    if not os.path.exists(mmproj_path):
        raise FileNotFoundError(f"Multimodal projector file not found: {mmproj_path}")

    cdef bytes path_bytes = mmproj_path.encode('utf-8')
    cdef mtmd_caps caps = mtmd_get_cap_from_file(<const char*>path_bytes)
    return {"vision": bool(caps.inp_vision), "audio": bool(caps.inp_audio)}