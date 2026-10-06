# mtmd.pxd - Cython header declarations for libmtmd multimodal support
#
# This file provides Cython declarations for the mtmd C API from llama.cpp
# Based on mtmd.h and mtmd-helper.h headers

from libc.stdint cimport uint32_t, int32_t, int64_t
from libc.stddef cimport size_t
from libcpp cimport bool as cppbool

from .ggml cimport ggml_log_level, ggml_log_callback, ggml_backend_dev_t, ggml_backend_sched_eval_callback
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
        size_t text_len
        bint add_special
        bint parse_special

    ctypedef struct mtmd_input_part:
        # only text or bitmap can be set, not both
        const mtmd_input_text * text
        const mtmd_bitmap * bitmap

    ctypedef bint (*mtmd_progress_callback)(float progress, void * user_data)

    ctypedef struct mtmd_context_params:
        bint use_gpu
        ggml_backend_dev_t device
        bint print_timings
        int n_threads
        const char * image_marker  # deprecated
        const char * media_marker
        llama_flash_attn_type flash_attn_type
        int image_min_tokens  # minimum number of tokens for image input (default: read from metadata)
        int image_max_tokens  # maximum number of tokens for image input (default: read from metadata)
        bint warmup  # whether to run a warmup encode pass after initialization
        int32_t batch_max_tokens  # maximum number of output tokens in a batch (default: 1024)
        ggml_backend_sched_eval_callback cb_eval
        void * cb_eval_user_data
        # Called with a progress value between 0.0 and 1.0; returning false aborts loading
        mtmd_progress_callback progress_callback
        void * progress_callback_user_data

    # Constants and defaults
    cdef const char * mtmd_default_marker()
    cdef const char * mtmd_get_marker(const mtmd_context * ctx)
    cdef mtmd_context_params mtmd_context_params_default()

    # Context management
    cdef mtmd_context * mtmd_init_from_file(const char * mmproj_fname,
                                       const llama_model * text_model,
                                       const mtmd_context_params ctx_params)
    cdef void mtmd_free(mtmd_context * ctx)

    # Context queries
    cdef bint mtmd_decode_use_non_causal(const mtmd_context * ctx, const mtmd_input_chunk * chunk)
    cdef bint mtmd_decode_use_mrope(const mtmd_context * ctx)
    cdef bint mtmd_support_vision(const mtmd_context * ctx)
    cdef bint mtmd_support_audio(const mtmd_context * ctx)
    cdef int mtmd_get_audio_sample_rate(const mtmd_context * ctx)

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
    # if true, video models may merge this bitmap with an adjacent mergeable one (temporal merge)
    cdef void mtmd_bitmap_set_mergeable(mtmd_bitmap * bitmap, bint mergeable)

    # Lazy bitmap: holds no data, expanded by mtmd_tokenize through the callback.
    # The callback returns 0 on success, -1 on EOF, -2 on error.
    ctypedef int (*mtmd_bitmap_lazy_callback)(size_t chunk_idx, void * user_data,
                                              mtmd_bitmap ** out_bitmap, char ** out_text)
    cdef mtmd_bitmap * mtmd_bitmap_init_lazy(const mtmd_context * ctx, const char * id,
                                             void * user_data, mtmd_bitmap_lazy_callback callback)

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
    cdef mtmd_input_chunk * mtmd_input_chunk_get_placeholder(const mtmd_input_chunk * chunk)
    # metadata only; a loaded chunk is a placeholder. out_buf may be NULL to query the size.
    cdef int32_t mtmd_input_chunk_save(const mtmd_input_chunk * chunk, char * out_buf,
                                       size_t out_len, size_t * expected_out_len)
    cdef mtmd_input_chunk * mtmd_input_chunk_load(const char * buf, size_t len)

    # Image tokens queries
    cdef size_t mtmd_image_tokens_get_n_tokens(const mtmd_image_tokens * image_tokens)
    cdef const char * mtmd_image_tokens_get_id(const mtmd_image_tokens * image_tokens)
    cdef llama_pos mtmd_image_tokens_get_n_pos(const mtmd_image_tokens * image_tokens)

    # Decoder position for M-RoPE models (replaces deprecated get_nx/get_ny)
    cdef struct mtmd_decoder_pos:
        uint32_t t
        uint32_t x
        uint32_t y
        uint32_t z  # unused for now, reserved for future use

    # i is the index of the embedding token, ranging from 0 to mtmd_image_tokens_get_n_tokens() - 1
    # pos_0 is the absolute position of the first token
    cdef mtmd_decoder_pos mtmd_image_tokens_get_decoder_pos(const mtmd_image_tokens * image_tokens, llama_pos pos_0, size_t i)

    # Core processing
    cdef int32_t mtmd_tokenize(mtmd_context * ctx,
                          mtmd_input_chunks * output,
                          const mtmd_input_text * text,
                          const mtmd_bitmap ** bitmaps,
                          size_t n_bitmaps)

    # as mtmd_tokenize, from text and bitmap parts instead of media markers; per-part add_special is ignored
    cdef int32_t mtmd_tokenize_from_parts(const mtmd_context * ctx,
                                          mtmd_input_chunks * output,
                                          const mtmd_input_part ** parts,
                                          size_t n_parts,
                                          bint add_special)

    cdef int32_t mtmd_encode(mtmd_context * ctx,
                        const mtmd_image_tokens * image_tokens)  # deprecated

    cdef int32_t mtmd_encode_chunk(mtmd_context * ctx,
                              const mtmd_input_chunk * chunk)

    cdef float * mtmd_get_output_embd(mtmd_context * ctx)

    # Logging
    cdef void mtmd_log_set(ggml_log_callback log_callback, void * user_data)

    # Batch encoding API
    # chunks are not owned by the batch, they will not be freed by mtmd_batch_free()
    # batch is valid for a given context, cannot be shared across contexts
    ctypedef struct mtmd_batch:
        pass

    cdef mtmd_batch * mtmd_batch_init(mtmd_context * ctx)
    cdef void mtmd_batch_free(mtmd_batch * batch)

    # only media chunks are allowed, text chunks will be rejected
    # 0 = success, 1 = generic error, 2 = batch too large, 3 = cannot batch with existing chunks
    cdef int32_t mtmd_batch_add_chunk(mtmd_batch * batch, const mtmd_input_chunk * chunk)
    cdef int32_t mtmd_batch_encode(mtmd_batch * batch) nogil
    cdef float * mtmd_batch_get_output_embd(mtmd_batch * batch, const mtmd_input_chunk * chunk)

    # EXPERIMENTAL: mmproj capabilities without initializing the full context
    ctypedef struct mtmd_caps:
        bint inp_vision
        bint inp_audio

    cdef mtmd_caps mtmd_get_cap_from_file(const char * mmproj_fname)

    # EXPERIMENTAL: audio generation (TTS) pipeline info
    ctypedef enum mtmd_gen_audio_type:
        MTMD_GEN_AUDIO_TYPE_NONE
        MTMD_GEN_AUDIO_TYPE_QWEN3TTS
        MTMD_GEN_AUDIO_TYPE_POCKETTTS

    ctypedef struct mtmd_gen_audio_info:
        mtmd_gen_audio_type type
        int32_t sample_rate
        const char * model_variant  # may be NULL

    cdef mtmd_gen_audio_info mtmd_gen_audio_get_info(const mtmd_context * ctx)

    ctypedef enum mtmd_gen_process_type:
        MTMD_GEN_PROCESS_TYPE_GEN_CODE  # h_state to semantic (codes, mel-spectrogram, etc.)
        MTMD_GEN_PROCESS_TYPE_GEN_WAV   # semantic to PCM audio

    ctypedef struct mtmd_gen_inp:
        mtmd_gen_process_type type
        # GEN_CODE
        int32_t code0
        float * embd
        int32_t top_k
        float top_p
        uint32_t seed
        float temp
        # GEN_WAV: codes (discrete) or feats (continuous), depending on the pipeline
        int32_t * codes
        size_t n_codes
        const float * feats
        size_t n_feats
        const char * state_data
        size_t state_size

    ctypedef struct mtmd_gen_out:
        # owned by the context, valid until the next process() call
        const int32_t * codes
        size_t n_codes
        const float * feats
        size_t n_feats
        const float * embd
        bint is_eos
        const float * audio
        size_t n_samples
        const char * state_data
        size_t state_size

    cdef mtmd_gen_inp mtmd_gen_inp_default(const mtmd_context * ctx)
    # stateless: the caller manages state and accumulates audio frames
    cdef int32_t mtmd_gen_audio_process(mtmd_context * ctx, const mtmd_gen_inp * inp, mtmd_gen_out * out)

    # Test function
    cdef mtmd_input_chunks * mtmd_test_create_input_chunks()


cdef extern from "mtmd-helper.h":
    # Video input helpers (require ffmpeg/ffprobe on the system PATH).
    # Video only exists at the helper level; the core mtmd library sees
    # the decoded frames as ordinary image bitmaps.
    ctypedef struct mtmd_helper_video:
        pass

    ctypedef struct mtmd_helper_video_info:
        uint32_t width
        uint32_t height
        float fps        # effective fps (fps_target if set, else original)
        int32_t n_frames # estimated total frames at effective fps (-1 if unknown)

    ctypedef struct mtmd_helper_video_init_params:
        float fps_target
        const char * ffmpeg_bin_dir
        int64_t timestamp_interval_ms

    cdef mtmd_helper_video_init_params mtmd_helper_video_init_params_default()

    # opt for mtmd_helper_bitmap_init_from_*()
    ctypedef struct mtmd_helper_init_opt:
        mtmd_helper_video_init_params video_params

    cdef mtmd_helper_init_opt mtmd_helper_init_opt_default()

    cdef mtmd_helper_video * mtmd_helper_video_init(mtmd_context * mctx,
                                                    const char * path,
                                                    mtmd_helper_video_init_params params) nogil

    cdef mtmd_helper_video * mtmd_helper_video_init_from_buf(mtmd_context * mctx,
                                                             const unsigned char * buf,
                                                             size_t length,
                                                             mtmd_helper_video_init_params params) nogil

    cdef void mtmd_helper_video_free(mtmd_helper_video * ctx)
    cdef mtmd_helper_video_info mtmd_helper_video_get_info(const mtmd_helper_video * ctx)

    # Exactly one of out_bitmap / out_text is set per call.
    # returns 0 on success, -1 on EOF, -2 on error
    cdef int32_t mtmd_helper_video_read_next(mtmd_helper_video * ctx,
                                             mtmd_bitmap ** out_bitmap,
                                             char ** out_text) nogil

    # EXPERIMENTAL: audio generation (TTS) helper
    ctypedef struct mtmd_helper_gen_audio:
        pass

    ctypedef enum mtmd_helper_gen_audio_outtype:
        MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM  # raw float32 PCM
        MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV  # WAV PCM 16-bit LE, mono

    ctypedef struct mtmd_helper_gen_audio_inp:
        llama_seq_id seq_id
        const char * prompt
        size_t prompt_len
        mtmd_bitmap * speaker_ref  # optional, can be NULL
        const char * lang          # optional, can be NULL
        int32_t top_k
        float top_p
        uint32_t seed              # UINT32_MAX for random
        mtmd_helper_gen_audio_outtype out_type

    cdef mtmd_helper_gen_audio * mtmd_helper_gen_audio_init(llama_context * lctx, mtmd_context * mctx)
    cdef void mtmd_helper_gen_audio_free(mtmd_helper_gen_audio * ctx)
    cdef void mtmd_helper_gen_audio_reset(mtmd_helper_gen_audio * ctx)
    cdef int32_t mtmd_helper_gen_audio_set_input(mtmd_helper_gen_audio * ctx,
                                                 const mtmd_helper_gen_audio_inp * inp) nogil

    # returns: >0 = prompt tokens remaining, 0 = done, <0 = error
    cdef int32_t mtmd_helper_gen_audio_step_prompt(mtmd_helper_gen_audio * ctx, int32_t n_batch) nogil

    # h_state_out is NULL if no frame was generated
    cdef int32_t mtmd_helper_gen_audio_step_gen(mtmd_helper_gen_audio * ctx,
                                                llama_token sampled,
                                                const float * h_state_in,
                                                const float ** h_state_out,
                                                cppbool * out_stop) nogil

    # out_data is valid until the next get_output() or reset() call
    cdef int32_t mtmd_helper_gen_audio_get_output(mtmd_helper_gen_audio * ctx,
                                                  int32_t * out_sample_rate,
                                                  const char ** out_data,
                                                  size_t * out_data_len,
                                                  int64_t * out_n_samples) nogil

    # return true if model can be used for chat
    cdef bint mtmd_helper_model_can_chat(llama_context * lctx, mtmd_context * mctx)

    # Logging
    cdef void mtmd_helper_log_set(ggml_log_callback log_callback, void * user_data)

    # true if this build includes video support (MTMD_VIDEO at compile time)
    cdef bint mtmd_helper_support_video(const mtmd_context * ctx)

    # Bitmap loading returns a wrapper struct holding the bitmap (and an
    # optional video context, unused here).
    cdef struct mtmd_helper_bitmap_wrapper:
        mtmd_bitmap * bitmap
        void * video_ctx

    # Helper functions for file/buffer loading
    cdef mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_file(const mtmd_context * ctx,
                                                   const char * fname,
                                                   bint placeholder,
                                                   mtmd_helper_init_opt opt)
    cdef mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_buf(const mtmd_context * ctx,
                                                   const unsigned char * buf,
                                                   size_t len,
                                                   bint placeholder,
                                                   mtmd_helper_init_opt opt)

    # Helper functions for chunk processing
    cdef size_t mtmd_helper_get_n_tokens(const mtmd_input_chunks * chunks)
    cdef llama_pos mtmd_helper_get_n_pos(const mtmd_input_chunks * chunks)

    # Helper to get list of relative decoder positions for image embedding tokens (M-RoPE)
    # out_pos must have length == mtmd_image_tokens_get_n_tokens(image)
    cdef void mtmd_helper_image_get_decoder_pos(const mtmd_image_tokens * image, mtmd_decoder_pos * out_pos)

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

    # one decoded sub-batch of embeddings, passed to mtmd_helper_post_decode_callback
    ctypedef struct mtmd_helper_embd_batch:
        int32_t n_tokens
        const float * embd   # [n_tokens, n_embd]
        int32_t n_embd
        const llama_pos * pos  # [n_pos, n_tokens], section-major
        int32_t n_pos          # 4 for M-RoPE models, 1 otherwise
        llama_seq_id seq_id

    ctypedef int32_t (*mtmd_helper_post_decode_callback)(const mtmd_helper_embd_batch * batch, void * user_data)

    cdef int32_t mtmd_helper_decode_image_chunk(mtmd_context * ctx,
                                           llama_context * lctx,
                                           const mtmd_input_chunk * chunk,
                                           float * encoded_embd,
                                           llama_pos n_past,
                                           llama_seq_id seq_id,
                                           int32_t n_batch,
                                           llama_pos * new_n_past,
                                           mtmd_helper_post_decode_callback callback,
                                           void * user_data)
