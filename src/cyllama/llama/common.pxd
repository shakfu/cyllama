# distutils: language=c++
#
# Declarations for structs and enums from common.h used by the Python wrappers.
# Only type declarations (compile-time); no function declarations (no link dependency).

from libc.stdint cimport int32_t, int8_t, int64_t, uint16_t, uint32_t, uint64_t, uint8_t
from libcpp.string cimport string as std_string
from libcpp.vector cimport vector as std_vector
from libcpp.set cimport set as std_set
from libcpp.pair cimport pair as std_pair
from libcpp.map cimport map as std_map

cimport ggml
cimport llama

#------------------------------------------------------------------------------
# common.h - type declarations only

cdef extern from "common.h":

    ctypedef struct common_adapter_lora_info:
        std_string path
        float scale
        llama.llama_adapter_lora * ptr

    ctypedef std_vector[llama.llama_token] llama_tokens

    # -------------------------------------------------------------------------
    # CPU utils

    ctypedef struct cpu_params:
        int      n_threads
        bint     cpumask[ggml.GGML_MAX_N_THREADS]
        bint     mask_valid
        ggml.ggml_sched_priority  priority
        bint     strict_cpu
        uint32_t poll

    # -------------------------------------------------------------------------
    # Enums

    cdef enum common_sampler_type:
        COMMON_SAMPLER_TYPE_NONE        = 0
        COMMON_SAMPLER_TYPE_DRY         = 1
        COMMON_SAMPLER_TYPE_TOP_K       = 2
        COMMON_SAMPLER_TYPE_TOP_P       = 3
        COMMON_SAMPLER_TYPE_MIN_P       = 4
        COMMON_SAMPLER_TYPE_TYPICAL_P   = 6
        COMMON_SAMPLER_TYPE_TEMPERATURE = 7
        COMMON_SAMPLER_TYPE_XTC         = 8
        COMMON_SAMPLER_TYPE_INFILL      = 9
        COMMON_SAMPLER_TYPE_PENALTIES   = 10
        COMMON_SAMPLER_TYPE_TOP_N_SIGMA = 11
        COMMON_SAMPLER_TYPE_ADAPTIVE_P  = 12

    cdef enum dimre_method:
        DIMRE_METHOD_PCA
        DIMRE_METHOD_MEAN

    cdef enum common_conversation_mode:
        COMMON_CONVERSATION_MODE_DISABLED = 0
        COMMON_CONVERSATION_MODE_ENABLED  = 1
        COMMON_CONVERSATION_MODE_AUTO     = 2

    cdef enum common_grammar_trigger_type:
        COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN
        COMMON_GRAMMAR_TRIGGER_TYPE_WORD
        COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN
        COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL

    cdef enum common_speculative_type:
        COMMON_SPECULATIVE_TYPE_NONE
        COMMON_SPECULATIVE_TYPE_DRAFT
        COMMON_SPECULATIVE_TYPE_EAGLE3
        COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE
        COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K
        COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V
        COMMON_SPECULATIVE_TYPE_NGRAM_MOD
        COMMON_SPECULATIVE_TYPE_NGRAM_CACHE
        COMMON_SPECULATIVE_TYPE_COUNT

    cdef enum common_reasoning_format:
        COMMON_REASONING_FORMAT_NONE
        COMMON_REASONING_FORMAT_AUTO
        COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY
        COMMON_REASONING_FORMAT_DEEPSEEK
        COMMON_REASONING_FORMAT_GRANITE

    # -------------------------------------------------------------------------
    # Param structs

    ctypedef struct common_grammar_trigger:
        common_grammar_trigger_type type
        std_string value
        llama.llama_token token

    ctypedef struct common_params_sampling:
        uint32_t seed
        int32_t n_prev
        int32_t n_probs
        int32_t min_keep
        int32_t top_k
        float   top_p
        float   min_p
        float   xtc_probability
        float   xtc_threshold
        float   typ_p
        float   temp
        float   dynatemp_range
        float   dynatemp_exponent
        int32_t penalty_last_n
        float   penalty_repeat
        float   penalty_freq
        float   penalty_present
        float   dry_multiplier
        float   dry_base
        int32_t dry_allowed_length
        int32_t dry_penalty_last_n
        float   adaptive_target
        float   adaptive_decay
        int32_t mirostat
        float   top_n_sigma
        float   mirostat_tau
        float   mirostat_eta
        bint    ignore_eos
        bint    no_perf
        bint    timing_per_token
        uint64_t user_sampling_config
        std_vector[std_string] dry_sequence_breakers
        std_vector[common_sampler_type] samplers
        std_string grammar
        bint grammar_lazy
        std_vector[common_grammar_trigger] grammar_triggers
        std_set[llama.llama_token] preserved_tokens
        std_vector[llama.llama_logit_bias] logit_bias
        std_vector[llama.llama_logit_bias] logit_bias_eog
        bint backend_sampling
        std_string print()

    ctypedef struct common_params_model:
        std_string path
        std_string url
        std_string hf_repo
        std_string hf_file
        std_string docker_repo
        std_string name

    ctypedef struct common_params_speculative:
        common_speculative_type type
        int32_t n_max
        int32_t n_min
        float   p_split
        float   p_min
        uint16_t ngram_size_n
        uint16_t ngram_size_m
        uint16_t ngram_min_hits
        std_string lookup_cache_static
        std_string lookup_cache_dynamic
        common_params_model mparams_dft
        int32_t n_ctx
        int32_t n_gpu_layers
        ggml.ggml_type cache_type_k
        ggml.ggml_type cache_type_v
        cpu_params cpuparams
        cpu_params cpuparams_batch
        std_vector[ggml.ggml_backend_dev_t] devices
        std_vector[std_pair[std_string, std_string]] replacements

    ctypedef struct common_params_vocoder:
        common_params_model model
        std_string speaker_file
        bint use_guide_tokens

    ctypedef struct common_params_diffusion:
        int32_t steps
        bint visual_mode
        float eps
        int32_t block_length
        int32_t algorithm
        float alg_temp
        float cfg_scale
        bint add_gumbel_noise

    # -------------------------------------------------------------------------
    # Main params struct

    ctypedef struct common_params:
        int32_t n_predict
        int32_t n_ctx
        int32_t n_batch
        int32_t n_ubatch
        int32_t n_keep
        int32_t n_chunks
        int32_t n_parallel
        int32_t n_sequences
        int32_t grp_attn_n
        int32_t grp_attn_w
        int32_t n_print
        float   rope_freq_base
        float   rope_freq_scale
        float   yarn_ext_factor
        float   yarn_attn_factor
        float   yarn_beta_fast
        float   yarn_beta_slow
        int32_t yarn_orig_ctx

        std_vector[ggml.ggml_backend_dev_t] devices

        int32_t n_gpu_layers
        int32_t main_gpu
        float   tensor_split[128]
        bint    fit_params
        int32_t fit_params_min_ctx
        std_vector[size_t] fit_params_target

        llama.llama_split_mode split_mode

        cpu_params cpuparams
        cpu_params cpuparams_batch

        ggml.ggml_backend_sched_eval_callback cb_eval
        void * cb_eval_user_data

        ggml.ggml_numa_strategy numa

        llama.llama_rope_scaling_type rope_scaling_type
        llama.llama_pooling_type      pooling_type
        llama.llama_attention_type    attention_type

        common_params_sampling    sampling
        common_params_speculative speculative
        common_params_vocoder     vocoder
        common_params_diffusion   diffusion

        common_params_model model

        std_set[std_string] model_alias
        std_string hf_token
        std_string prompt
        std_string system_prompt
        std_string prompt_file
        std_string path_prompt_cache
        std_string input_prefix
        std_string input_suffix
        std_string logits_file

        std_vector[std_string] in_files
        std_vector[std_string] antiprompt
        std_vector[llama.llama_model_kv_override] kv_overrides

        bint lora_init_without_apply

        int32_t verbosity
        int32_t control_vector_layer_start
        int32_t control_vector_layer_end

        int32_t ppl_stride
        int32_t ppl_output_type

        bint   hellaswag
        size_t hellaswag_tasks

        bint   winogrande
        size_t winogrande_tasks

        bint   multiple_choice
        size_t multiple_choice_tasks

        bint   kl_divergence

        bint usage
        bint use_color
        bint special
        bint interactive
        bint interactive_first
        bint conversation
        bint prompt_cache_all
        bint prompt_cache_ro

        bint escape
        bint multiline_input
        bint simple_io
        bint cont_batching
        llama.llama_flash_attn_type flash_attn_type
        bint no_perf
        bint show_timings
        bint ctx_shift
        bint swa_full
        bint kv_unified

        bint input_prefix_bos
        bint use_mmap
        bint use_direct_io
        bint use_mlock
        bint verbose_prompt
        bint display_prompt
        bint infill
        bint dump_kv_cache
        bint no_kv_offload
        bint warmup
        bint check_tensors
        bint no_op_offload
        bint no_extra_bufts
        bint no_host
        bint single_turn

        ggml.ggml_type cache_type_k
        ggml.ggml_type cache_type_v

        common_conversation_mode conversation_mode

        common_params_model mmproj
        bint mmproj_use_gpu
        bint no_mmproj
        std_vector[std_string] image
        int image_min_tokens
        int image_max_tokens

        bint embedding
        int32_t embd_normalize
        std_string embd_out
        std_string embd_sep
        std_string cls_sep

        int32_t port
        int32_t timeout_read
        int32_t timeout_write
        int32_t n_threads_http
        int32_t n_cache_reuse
        bint    cache_prompt
        int32_t n_ctx_checkpoints
        int32_t cache_ram_mib

        std_string hostname
        std_string public_path
        std_string api_prefix
        std_string chat_template
        bint use_jinja
        bint enable_chat_template
        common_reasoning_format reasoning_format
        int reasoning_budget
        bint prefill_assistant
        int sleep_idle_seconds

        std_vector[std_string] api_keys

        std_string ssl_file_key
        std_string ssl_file_cert

        std_map[std_string, std_string] default_template_kwargs

        bint webui
        std_string webui_config_json

        bint endpoint_slots
        bint endpoint_props
        bint endpoint_metrics

        std_string models_dir
        std_string models_preset
        int models_max
        bint models_autoload

        bint log_json

        std_string slot_save_path
        std_string media_path

        float slot_prompt_similarity

        bint is_pp_shared
        bint is_tg_separate

        std_vector[int32_t] n_pp
        std_vector[int32_t] n_tg
        std_vector[int32_t] n_pl

        std_vector[std_string] context_files

        int32_t chunk_size
        std_string chunk_separator

        int32_t n_junk
        int32_t i_pos

        std_string out_file

        int32_t n_out_freq
        int32_t n_save_freq
        int32_t i_chunk
        int8_t  imat_dat

        bint process_output
        bint compute_ppl
        bint show_statistics
        bint parse_special

        int n_pca_batch
        int n_pca_iterations
        dimre_method cvector_dimre_method
        std_string cvector_positive_file
        std_string cvector_negative_file

        bint spm_infill

        bint batched_bench_output_jsonl

        llama.llama_progress_callback load_progress_callback
        void * load_progress_callback_user_data
