# mtmd.pxd - Cython header declarations for libmtmd multimodal support
#
# This file provides Cython declarations for the mtmd C API from llama.cpp
# Based on mtmd.h and mtmd-helper.h headers

from libc.stdint cimport uint32_t, int32_t
from libc.stddef cimport size_t

from .ggml cimport ggml_log_level, ggml_log_callback
from .llama cimport llama_model, llama_context, llama_token, llama_pos, llama_seq_id, llama_flash_attn_type

# Forward declarations for Cython classes from llama_cpp
# We'll import these at runtime to avoid circular imports

cdef extern from "mtmd.h":
    # Enums
    ctypedef enum mtmd_input_chunk_type:
        MTMD_INPUT_CHUNK_TYPE_TEXT
        MTMD_INPUT_CHUNK_TYPE_IMAGE
        MTMD_INPUT_CHUNK_TYPE_AUDIO

    # Opaque types
    ctypedef struct mtmd_context:
        pass

    ctypedef struct mtmd_bitmap:
        pass

    ctypedef struct mtmd_image_tokens:
        pass

    ctypedef struct mtmd_input_chunk:
        pass

    ctypedef struct mtmd_input_chunks:
        pass

    # Structs
    ctypedef struct mtmd_input_text:
        const char * text
        bint add_special
        bint parse_special

    ctypedef struct mtmd_context_params:
        bint use_gpu
        bint print_timings
        int n_threads
        const char * image_marker  # deprecated
        const char * media_marker
        llama_flash_attn_type flash_attn_type
        int image_min_tokens  # minimum number of tokens for image input (default: read from metadata)
        int image_max_tokens  # maximum number of tokens for image input (default: read from metadata)
        bint warmup  # whether to run a warmup encode pass after initialization

    # Constants and defaults
    cdef const char * mtmd_default_marker()
    cdef mtmd_context_params mtmd_context_params_default()

    # Context management
    cdef mtmd_context * mtmd_init_from_file(const char * mmproj_fname,
                                       const llama_model * text_model,
                                       const mtmd_context_params ctx_params)
    cdef void mtmd_free(mtmd_context * ctx)

    # Context queries
    cdef bint mtmd_decode_use_non_causal(mtmd_context * ctx)
    cdef bint mtmd_decode_use_mrope(mtmd_context * ctx)
    cdef bint mtmd_support_vision(mtmd_context * ctx)
    cdef bint mtmd_support_audio(mtmd_context * ctx)
    cdef int mtmd_get_audio_bitrate(mtmd_context * ctx)

    # Bitmap management
    cdef mtmd_bitmap * mtmd_bitmap_init(uint32_t nx, uint32_t ny, const unsigned char * data)
    cdef mtmd_bitmap * mtmd_bitmap_init_from_audio(size_t n_samples, const float * data)
    cdef uint32_t mtmd_bitmap_get_nx(const mtmd_bitmap * bitmap)
    cdef uint32_t mtmd_bitmap_get_ny(const mtmd_bitmap * bitmap)
    cdef const unsigned char * mtmd_bitmap_get_data(const mtmd_bitmap * bitmap)
    cdef size_t mtmd_bitmap_get_n_bytes(const mtmd_bitmap * bitmap)
    cdef bint mtmd_bitmap_is_audio(const mtmd_bitmap * bitmap)
    cdef void mtmd_bitmap_free(mtmd_bitmap * bitmap)
    cdef const char * mtmd_bitmap_get_id(const mtmd_bitmap * bitmap)
    cdef void mtmd_bitmap_set_id(mtmd_bitmap * bitmap, const char * id)

    # Input chunks management
    cdef mtmd_input_chunks * mtmd_input_chunks_init()
    cdef size_t mtmd_input_chunks_size(const mtmd_input_chunks * chunks)
    cdef const mtmd_input_chunk * mtmd_input_chunks_get(const mtmd_input_chunks * chunks, size_t idx)
    cdef void mtmd_input_chunks_free(mtmd_input_chunks * chunks)

    # Input chunk queries
    cdef mtmd_input_chunk_type mtmd_input_chunk_get_type(const mtmd_input_chunk * chunk)
    cdef const llama_token * mtmd_input_chunk_get_tokens_text(const mtmd_input_chunk * chunk, size_t * n_tokens_output)
    cdef const mtmd_image_tokens * mtmd_input_chunk_get_tokens_image(const mtmd_input_chunk * chunk)
    cdef size_t mtmd_input_chunk_get_n_tokens(const mtmd_input_chunk * chunk)
    cdef const char * mtmd_input_chunk_get_id(const mtmd_input_chunk * chunk)
    cdef llama_pos mtmd_input_chunk_get_n_pos(const mtmd_input_chunk * chunk)

    # Input chunk management
    cdef mtmd_input_chunk * mtmd_input_chunk_copy(const mtmd_input_chunk * chunk)
    cdef void mtmd_input_chunk_free(mtmd_input_chunk * chunk)

    # Image tokens queries
    cdef size_t mtmd_image_tokens_get_n_tokens(const mtmd_image_tokens * image_tokens)
    cdef size_t mtmd_image_tokens_get_nx(const mtmd_image_tokens * image_tokens)
    cdef size_t mtmd_image_tokens_get_ny(const mtmd_image_tokens * image_tokens)
    cdef const char * mtmd_image_tokens_get_id(const mtmd_image_tokens * image_tokens)
    cdef llama_pos mtmd_image_tokens_get_n_pos(const mtmd_image_tokens * image_tokens)

    # Core processing
    cdef int32_t mtmd_tokenize(mtmd_context * ctx,
                          mtmd_input_chunks * output,
                          const mtmd_input_text * text,
                          const mtmd_bitmap ** bitmaps,
                          size_t n_bitmaps)

    cdef int32_t mtmd_encode(mtmd_context * ctx,
                        const mtmd_image_tokens * image_tokens)  # deprecated

    cdef int32_t mtmd_encode_chunk(mtmd_context * ctx,
                              const mtmd_input_chunk * chunk)

    cdef float * mtmd_get_output_embd(mtmd_context * ctx)

    # Logging
    cdef void mtmd_log_set(ggml_log_callback log_callback, void * user_data)

    # Test function
    cdef mtmd_input_chunks * mtmd_test_create_input_chunks()


cdef extern from "mtmd-helper.h":
    # Logging
    cdef void mtmd_helper_log_set(ggml_log_callback log_callback, void * user_data)

    # Helper functions for file/buffer loading
    cdef mtmd_bitmap * mtmd_helper_bitmap_init_from_file(mtmd_context * ctx, const char * fname)
    cdef mtmd_bitmap * mtmd_helper_bitmap_init_from_buf(mtmd_context * ctx,
                                                   const unsigned char * buf,
                                                   size_t len)

    # Helper functions for chunk processing
    cdef size_t mtmd_helper_get_n_tokens(const mtmd_input_chunks * chunks)
    cdef llama_pos mtmd_helper_get_n_pos(const mtmd_input_chunks * chunks)

    # Helper functions for evaluation
    cdef int32_t mtmd_helper_eval_chunks(mtmd_context * ctx,
                                    llama_context * lctx,
                                    const mtmd_input_chunks * chunks,
                                    llama_pos n_past,
                                    llama_seq_id seq_id,
                                    int32_t n_batch,
                                    bint logits_last,
                                    llama_pos * new_n_past)

    cdef int32_t mtmd_helper_eval_chunk_single(mtmd_context * ctx,
                                          llama_context * lctx,
                                          const mtmd_input_chunk * chunk,
                                          llama_pos n_past,
                                          llama_seq_id seq_id,
                                          int32_t n_batch,
                                          bint logits_last,
                                          llama_pos * new_n_past)

    cdef int32_t mtmd_helper_decode_image_chunk(mtmd_context * ctx,
                                           llama_context * lctx,
                                           const mtmd_input_chunk * chunk,
                                           float * encoded_embd,
                                           llama_pos n_past,
                                           llama_seq_id seq_id,
                                           int32_t n_batch,
                                           llama_pos * new_n_past)
