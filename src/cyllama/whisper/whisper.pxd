# distutils: language = c++

# cython: language_level=3

from libc.stdint cimport int32_t, int64_t, uint32_t
from libc.stddef cimport size_t
from libc.stdio cimport FILE

cdef extern from "whisper.h":
    # Constants
    cdef int WHISPER_SAMPLE_RATE
    cdef int WHISPER_N_FFT
    cdef int WHISPER_HOP_LENGTH
    cdef int WHISPER_CHUNK_SIZE

    # Type definitions
    ctypedef int32_t whisper_pos
    ctypedef int32_t whisper_token
    ctypedef int32_t whisper_seq_id

    # Enums
    ctypedef enum whisper_alignment_heads_preset:
        WHISPER_AHEADS_NONE
        WHISPER_AHEADS_N_TOP_MOST
        WHISPER_AHEADS_CUSTOM
        WHISPER_AHEADS_TINY_EN
        WHISPER_AHEADS_TINY
        WHISPER_AHEADS_BASE_EN
        WHISPER_AHEADS_BASE
        WHISPER_AHEADS_SMALL_EN
        WHISPER_AHEADS_SMALL
        WHISPER_AHEADS_MEDIUM_EN
        WHISPER_AHEADS_MEDIUM
        WHISPER_AHEADS_LARGE_V1
        WHISPER_AHEADS_LARGE_V2
        WHISPER_AHEADS_LARGE_V3
        WHISPER_AHEADS_LARGE_V3_TURBO

    ctypedef enum whisper_sampling_strategy:
        WHISPER_SAMPLING_GREEDY
        WHISPER_SAMPLING_BEAM_SEARCH

    ctypedef enum whisper_gretype:
        WHISPER_GRETYPE_END
        WHISPER_GRETYPE_ALT
        WHISPER_GRETYPE_RULE_REF
        WHISPER_GRETYPE_CHAR
        WHISPER_GRETYPE_CHAR_NOT
        WHISPER_GRETYPE_CHAR_RNG_UPPER
        WHISPER_GRETYPE_CHAR_ALT

    # Structures
    ctypedef struct whisper_ahead:
        int n_text_layer
        int n_head

    ctypedef struct whisper_aheads:
        size_t n_heads
        const whisper_ahead * heads

    ctypedef struct whisper_context_params:
        bint use_gpu
        bint flash_attn
        int gpu_device
        bint dtw_token_timestamps
        whisper_alignment_heads_preset dtw_aheads_preset
        int dtw_n_top
        whisper_aheads dtw_aheads
        size_t dtw_mem_size

    ctypedef struct whisper_token_data:
        whisper_token id
        whisper_token tid
        float p
        float plog
        float pt
        float ptsum
        int64_t t0
        int64_t t1
        int64_t t_dtw
        float vlen

    ctypedef struct whisper_model_loader:
        void * context
        size_t (*read)(void * ctx, void * output, size_t read_size)
        bint (*eof)(void * ctx)
        void (*close)(void * ctx)

    ctypedef struct whisper_grammar_element:
        whisper_gretype type
        uint32_t value

    ctypedef struct whisper_vad_params:
        float threshold
        int min_speech_duration_ms
        int min_silence_duration_ms
        float max_speech_duration_s
        int speech_pad_ms
        float samples_overlap

    ctypedef struct whisper_timings:
        float sample_ms
        float encode_ms
        float decode_ms
        float batchd_ms
        float prompt_ms

    ctypedef struct whisper_vad_context_params:
        int n_threads
        bint use_gpu
        int gpu_device

    # Forward declarations
    cdef struct whisper_context
    cdef struct whisper_state
    # cdef struct whisper_full_params
    cdef struct whisper_vad_context
    cdef struct whisper_vad_segments

    # Callback types
    ctypedef void (*whisper_new_segment_callback)(whisper_context * ctx, whisper_state * state, int n_new, void * user_data)
    ctypedef void (*whisper_progress_callback)(whisper_context * ctx, whisper_state * state, int progress, void * user_data)
    ctypedef bint (*whisper_encoder_begin_callback)(whisper_context * ctx, whisper_state * state, void * user_data)
    ctypedef void (*whisper_logits_filter_callback)(whisper_context * ctx, whisper_state * state, const whisper_token_data * tokens, int n_tokens, float * logits, void * user_data)

    # Full params structure
    ctypedef struct whisper_full_params:
        whisper_sampling_strategy strategy
        int n_threads
        int n_max_text_ctx
        int offset_ms
        int duration_ms
        bint translate
        bint no_context
        bint no_timestamps
        bint single_segment
        bint print_special
        bint print_progress
        bint print_realtime
        bint print_timestamps
        bint token_timestamps
        float thold_pt
        float thold_ptsum
        int max_len
        bint split_on_word
        int max_tokens
        bint debug_mode
        int audio_ctx
        bint tdrz_enable
        const char * suppress_regex
        const char * initial_prompt
        const whisper_token * prompt_tokens
        int prompt_n_tokens
        const char * language
        bint detect_language
        bint suppress_blank
        bint suppress_nst
        float temperature
        float max_initial_ts
        float length_penalty
        float temperature_inc
        float entropy_thold
        float logprob_thold
        float no_speech_thold
        # struct:
        #     int best_of
        # greedy
        # struct:
        #     int beam_size
        #     float patience
        # beam_search
        whisper_new_segment_callback new_segment_callback
        void * new_segment_callback_user_data
        whisper_progress_callback progress_callback
        void * progress_callback_user_data
        whisper_encoder_begin_callback encoder_begin_callback
        void * encoder_begin_callback_user_data
        void * abort_callback  # ggml_abort_callback
        void * abort_callback_user_data
        whisper_logits_filter_callback logits_filter_callback
        void * logits_filter_callback_user_data
        const whisper_grammar_element ** grammar_rules
        size_t n_grammar_rules
        size_t i_start_rule
        float grammar_penalty
        bint vad
        const char * vad_model_path
        whisper_vad_params vad_params

    # Version and info functions
    cdef const char * whisper_version()
    cdef const char * whisper_print_system_info()

    # Context initialization functions
    cdef whisper_context * whisper_init_from_file_with_params(const char * path_model, whisper_context_params params)
    cdef whisper_context * whisper_init_from_buffer_with_params(void * buffer, size_t buffer_size, whisper_context_params params)
    cdef whisper_context * whisper_init_with_params(whisper_model_loader * loader, whisper_context_params params)
    cdef whisper_context * whisper_init_from_file_with_params_no_state(const char * path_model, whisper_context_params params)
    cdef whisper_context * whisper_init_from_buffer_with_params_no_state(void * buffer, size_t buffer_size, whisper_context_params params)
    cdef whisper_context * whisper_init_with_params_no_state(whisper_model_loader * loader, whisper_context_params params)

    # State management
    cdef whisper_state * whisper_init_state(whisper_context * ctx)

    # OpenVINO support
    cdef int whisper_ctx_init_openvino_encoder_with_state(whisper_context * ctx, whisper_state * state, const char * model_path, const char * device, const char * cache_dir)
    cdef int whisper_ctx_init_openvino_encoder(whisper_context * ctx, const char * model_path, const char * device, const char * cache_dir)

    # Memory management
    cdef void whisper_free(whisper_context * ctx)
    cdef void whisper_free_state(whisper_state * state)
    # void whisper_free_params(whisper_full_params * params)
    cdef void whisper_free_context_params(whisper_context_params * params)

    # Audio processing
    cdef int whisper_pcm_to_mel(whisper_context * ctx, const float * samples, int n_samples, int n_threads)
    cdef int whisper_pcm_to_mel_with_state(whisper_context * ctx, whisper_state * state, const float * samples, int n_samples, int n_threads)
    cdef int whisper_set_mel(whisper_context * ctx, const float * data, int n_len, int n_mel)
    cdef int whisper_set_mel_with_state(whisper_context * ctx, whisper_state * state, const float * data, int n_len, int n_mel)

    # Encoder/decoder functions
    cdef int whisper_encode(whisper_context * ctx, int offset, int n_threads)
    cdef int whisper_encode_with_state(whisper_context * ctx, whisper_state * state, int offset, int n_threads)
    cdef int whisper_decode(whisper_context * ctx, const whisper_token * tokens, int n_tokens, int n_past, int n_threads)
    cdef int whisper_decode_with_state(whisper_context * ctx, whisper_state * state, const whisper_token * tokens, int n_tokens, int n_past, int n_threads)

    # Tokenization
    cdef int whisper_tokenize(whisper_context * ctx, const char * text, whisper_token * tokens, int n_max_tokens)
    cdef int whisper_token_count(whisper_context * ctx, const char * text)

    # Language functions
    cdef int whisper_lang_max_id()
    cdef int whisper_lang_id(const char * lang)
    cdef const char * whisper_lang_str(int id)
    cdef const char * whisper_lang_str_full(int id)
    cdef int whisper_lang_auto_detect(whisper_context * ctx, int offset_ms, int n_threads, float * lang_probs)
    cdef int whisper_lang_auto_detect_with_state(whisper_context * ctx, whisper_state * state, int offset_ms, int n_threads, float * lang_probs)

    # Model info functions
    cdef int whisper_n_len(whisper_context * ctx)
    cdef int whisper_n_len_from_state(whisper_state * state)
    cdef int whisper_n_vocab(whisper_context * ctx)
    cdef int whisper_n_text_ctx(whisper_context * ctx)
    cdef int whisper_n_audio_ctx(whisper_context * ctx)
    cdef int whisper_is_multilingual(whisper_context * ctx)
    cdef int whisper_model_n_vocab(whisper_context * ctx)
    cdef int whisper_model_n_audio_ctx(whisper_context * ctx)
    cdef int whisper_model_n_audio_state(whisper_context * ctx)
    cdef int whisper_model_n_audio_head(whisper_context * ctx)
    cdef int whisper_model_n_audio_layer(whisper_context * ctx)
    cdef int whisper_model_n_text_ctx(whisper_context * ctx)
    cdef int whisper_model_n_text_state(whisper_context * ctx)
    cdef int whisper_model_n_text_head(whisper_context * ctx)
    cdef int whisper_model_n_text_layer(whisper_context * ctx)
    cdef int whisper_model_n_mels(whisper_context * ctx)
    cdef int whisper_model_ftype(whisper_context * ctx)
    cdef int whisper_model_type(whisper_context * ctx)

    # Logits and tokens
    cdef float * whisper_get_logits(whisper_context * ctx)
    cdef float * whisper_get_logits_from_state(whisper_state * state)
    cdef const char * whisper_token_to_str(whisper_context * ctx, whisper_token token)
    cdef const char * whisper_model_type_readable(whisper_context * ctx)

    # Special tokens
    cdef whisper_token whisper_token_eot(whisper_context * ctx)
    cdef whisper_token whisper_token_sot(whisper_context * ctx)
    cdef whisper_token whisper_token_solm(whisper_context * ctx)
    cdef whisper_token whisper_token_prev(whisper_context * ctx)
    cdef whisper_token whisper_token_nosp(whisper_context * ctx)
    cdef whisper_token whisper_token_not(whisper_context * ctx)
    cdef whisper_token whisper_token_beg(whisper_context * ctx)
    cdef whisper_token whisper_token_lang(whisper_context * ctx, int lang_id)
    cdef whisper_token whisper_token_translate(whisper_context * ctx)
    cdef whisper_token whisper_token_transcribe(whisper_context * ctx)

    # Performance and timing
    cdef whisper_timings * whisper_get_timings(whisper_context * ctx)
    cdef void whisper_print_timings(whisper_context * ctx)
    cdef void whisper_reset_timings(whisper_context * ctx)

    # Parameter functions
    cdef whisper_context_params * whisper_context_default_params_by_ref()
    cdef whisper_context_params whisper_context_default_params()
    cdef whisper_full_params * whisper_full_default_params_by_ref(whisper_sampling_strategy strategy)
    cdef whisper_full_params whisper_full_default_params(whisper_sampling_strategy strategy)

    # Main processing functions
    cdef int whisper_full(whisper_context * ctx, whisper_full_params params, const float * samples, int n_samples)
    cdef int whisper_full_with_state(whisper_context * ctx, whisper_state * state, whisper_full_params params, const float * samples, int n_samples)
    cdef int whisper_full_parallel(whisper_context * ctx, whisper_full_params params, const float * samples, int n_samples, int n_processors)

    # Result extraction
    cdef int whisper_full_n_segments(whisper_context * ctx)
    cdef int whisper_full_n_segments_from_state(whisper_state * state)
    cdef int whisper_full_lang_id(whisper_context * ctx)
    cdef int whisper_full_lang_id_from_state(whisper_state * state)
    cdef int64_t whisper_full_get_segment_t0(whisper_context * ctx, int i_segment)
    cdef int64_t whisper_full_get_segment_t0_from_state(whisper_state * state, int i_segment)
    cdef int64_t whisper_full_get_segment_t1(whisper_context * ctx, int i_segment)
    cdef int64_t whisper_full_get_segment_t1_from_state(whisper_state * state, int i_segment)
    cdef bint whisper_full_get_segment_speaker_turn_next(whisper_context * ctx, int i_segment)
    cdef bint whisper_full_get_segment_speaker_turn_next_from_state(whisper_state * state, int i_segment)
    cdef const char * whisper_full_get_segment_text(whisper_context * ctx, int i_segment)
    cdef const char * whisper_full_get_segment_text_from_state(whisper_state * state, int i_segment)
    cdef int whisper_full_n_tokens(whisper_context * ctx, int i_segment)
    cdef int whisper_full_n_tokens_from_state(whisper_state * state, int i_segment)
    cdef const char * whisper_full_get_token_text(whisper_context * ctx, int i_segment, int i_token)
    cdef const char * whisper_full_get_token_text_from_state(whisper_context * ctx, whisper_state * state, int i_segment, int i_token)
    cdef whisper_token whisper_full_get_token_id(whisper_context * ctx, int i_segment, int i_token)
    cdef whisper_token whisper_full_get_token_id_from_state(whisper_state * state, int i_segment, int i_token)
    cdef whisper_token_data whisper_full_get_token_data(whisper_context * ctx, int i_segment, int i_token)
    cdef whisper_token_data whisper_full_get_token_data_from_state(whisper_state * state, int i_segment, int i_token)
    cdef float whisper_full_get_token_p(whisper_context * ctx, int i_segment, int i_token)
    cdef float whisper_full_get_token_p_from_state(whisper_state * state, int i_segment, int i_token)
    cdef float whisper_full_get_segment_no_speech_prob(whisper_context * ctx, int i_segment)
    cdef float whisper_full_get_segment_no_speech_prob_from_state(whisper_state * state, int i_segment)

    # VAD functions
    cdef whisper_vad_params whisper_vad_default_params()
    cdef whisper_vad_context_params whisper_vad_default_context_params()
    cdef whisper_vad_context * whisper_vad_init_from_file_with_params(const char * path_model, whisper_vad_context_params params)
    cdef whisper_vad_context * whisper_vad_init_with_params(whisper_model_loader * loader, whisper_vad_context_params params)
    cdef bint whisper_vad_detect_speech(whisper_vad_context * vctx, const float * samples, int n_samples)
    cdef int whisper_vad_n_probs(whisper_vad_context * vctx)
    cdef float * whisper_vad_probs(whisper_vad_context * vctx)
    cdef whisper_vad_segments * whisper_vad_segments_from_probs(whisper_vad_context * vctx, whisper_vad_params params)
    cdef whisper_vad_segments * whisper_vad_segments_from_samples(whisper_vad_context * vctx, whisper_vad_params params, const float * samples, int n_samples)
    cdef int whisper_vad_segments_n_segments(whisper_vad_segments * segments)
    cdef float whisper_vad_segments_get_segment_t0(whisper_vad_segments * segments, int i_segment)
    cdef float whisper_vad_segments_get_segment_t1(whisper_vad_segments * segments, int i_segment)
    cdef void whisper_vad_free_segments(whisper_vad_segments * segments)
    cdef void whisper_vad_free(whisper_vad_context * ctx)

    # Benchmark functions
    cdef int whisper_bench_memcpy(int n_threads)
    cdef const char * whisper_bench_memcpy_str(int n_threads)
    cdef int whisper_bench_ggml_mul_mat(int n_threads)
    cdef const char * whisper_bench_ggml_mul_mat_str(int n_threads)

    # Logging
    cdef void whisper_log_set(void * log_callback, void * user_data)  # ggml_log_callback

