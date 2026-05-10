# distutils: language=c++

from libc.stdint cimport int32_t, uint32_t, int64_t, int8_t, uint64_t, uint16_t
from libcpp.string cimport string as std_string
from libcpp.vector cimport vector as std_vector
from libcpp.set cimport set as std_set
from libcpp.map cimport map as std_map
from libcpp.utility cimport pair as std_pair


#------------------------------------------------------------------------------
# ggml.h

cdef extern from "ggml.h":

    cpdef enum:
        GGML_DEFAULT_N_THREADS
        GGML_MAX_DIMS
        GGML_MAX_N_THREADS
        GGML_MAX_NAME
        GGML_MAX_OP_PARAMS
        GGML_MAX_SRC


    cpdef enum:
        GGML_ROPE_TYPE_NORMAL
        GGML_ROPE_TYPE_NEOX
        GGML_ROPE_TYPE_MROPE
        GGML_ROPE_TYPE_VISION
        GGML_ROPE_TYPE_IMROPE


    cpdef enum ggml_sched_priority:
        GGML_SCHED_PRIO_LOW
        GGML_SCHED_PRIO_NORMAL
        GGML_SCHED_PRIO_MEDIUM
        GGML_SCHED_PRIO_HIGH
        GGML_SCHED_PRIO_REALTIME


    cpdef enum ggml_numa_strategy:
        GGML_NUMA_STRATEGY_DISABLED
        GGML_NUMA_STRATEGY_DISTRIBUTE
        GGML_NUMA_STRATEGY_ISOLATE
        GGML_NUMA_STRATEGY_NUMACTL
        GGML_NUMA_STRATEGY_MIRROR
        GGML_NUMA_STRATEGY_COUNT


    cpdef enum ggml_type:
        GGML_TYPE_F32
        GGML_TYPE_F16
        GGML_TYPE_Q4_0
        GGML_TYPE_Q4_1
        # GGML_TYPE_Q4_2 = 4, support has been removed
        # GGML_TYPE_Q4_3 = 5, support has been removed
        GGML_TYPE_Q5_0
        GGML_TYPE_Q5_1
        GGML_TYPE_Q8_0
        GGML_TYPE_Q8_1
        GGML_TYPE_Q2_K
        GGML_TYPE_Q3_K
        GGML_TYPE_Q4_K
        GGML_TYPE_Q5_K
        GGML_TYPE_Q6_K
        GGML_TYPE_Q8_K
        GGML_TYPE_IQ2_XXS
        GGML_TYPE_IQ2_XS
        GGML_TYPE_IQ3_XXS
        GGML_TYPE_IQ1_S
        GGML_TYPE_IQ4_NL
        GGML_TYPE_IQ3_S
        GGML_TYPE_IQ2_S
        GGML_TYPE_IQ4_XS
        GGML_TYPE_I8
        GGML_TYPE_I16
        GGML_TYPE_I32
        GGML_TYPE_I64
        GGML_TYPE_F64
        GGML_TYPE_IQ1_M
        GGML_TYPE_BF16
        # GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
        # GGML_TYPE_Q4_0_4_8 = 32,
        # GGML_TYPE_Q4_0_8_8 = 33,
        GGML_TYPE_TQ1_0
        GGML_TYPE_TQ2_0
        # GGML_TYPE_IQ4_NL_4_4 = 36,
        # GGML_TYPE_IQ4_NL_4_8 = 37,
        # GGML_TYPE_IQ4_NL_8_8 = 38,
        GGML_TYPE_MXFP4 # MXFP4 (1 block)
        GGML_TYPE_NVFP4 # NVFP4 (4 blocks, E4M3 scale)
        GGML_TYPE_Q1_0
        GGML_TYPE_COUNT


    ctypedef struct ggml_backend_buffer_type:
        pass

    ctypedef struct ggml_backend_buffer:
        pass


    ctypedef ggml_backend_buffer_type * ggml_backend_buffer_type_t

    # -------------------------------------------------------------------------
    # n-dimensional tensor

    ctypedef struct ggml_tensor:
        pass


#------------------------------------------------------------------------------
# ggml-backend.h


cdef extern from "ggml-backend.h":
    cpdef enum ggml_backend_dev_type:
        # CPU device using system memory
        GGML_BACKEND_DEVICE_TYPE_CPU
        # GPU device using dedicated memory
        GGML_BACKEND_DEVICE_TYPE_GPU
        # integrated GPU device using host memory
        GGML_BACKEND_DEVICE_TYPE_IGPU
        # accelerator devices intended to be used together with the CPU backend (e.g. BLAS or AMX)
        GGML_BACKEND_DEVICE_TYPE_ACCEL
        GGML_BACKEND_DEVICE_TYPE_META

    # functionality supported by the device
    ctypedef struct ggml_backend_dev_caps:
        # asynchronous operations
        bint async
        # pinned host buffer
        bint host_buffer
        # creating buffers from host ptr
        bint buffer_from_host_ptr
        # event synchronization
        bint events

    # all the device properties
    ctypedef struct ggml_backend_dev_props:
        # device name
        const char * name
        # device description
        const char * description
        # device free memory in bytes
        size_t memory_free
        # device total memory in bytes
        size_t memory_total
        # device type
        ggml_backend_dev_type type
        # device id
        #   for PCI devices, this should be the PCI bus id formatted as "domain:bus:device.function" (e.g. "0000:01:00.0")
        #   if the id is unknown, this should be NULL
        const char * device_id
        # device capabilities
        ggml_backend_dev_caps caps

    ctypedef bint (*ggml_backend_sched_eval_callback)(ggml_tensor * t, bint ask, void * user_data)

    ctypedef bint (*ggml_abort_callback)(void *)

    ctypedef struct ggml_backend_device: pass

    ctypedef ggml_backend_device * ggml_backend_dev_t


#------------------------------------------------------------------------------
# llama.h

cdef extern from "llama.h":

    ctypedef int32_t llama_token

    cpdef enum llama_rope_scaling_type:
        LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED
        LLAMA_ROPE_SCALING_TYPE_NONE
        LLAMA_ROPE_SCALING_TYPE_LINEAR
        LLAMA_ROPE_SCALING_TYPE_YARN
        LLAMA_ROPE_SCALING_TYPE_LONGROPE
        LLAMA_ROPE_SCALING_TYPE_MAX_VALUE

    cpdef enum llama_pooling_type:
        LLAMA_POOLING_TYPE_UNSPECIFIED
        LLAMA_POOLING_TYPE_NONE
        LLAMA_POOLING_TYPE_MEAN
        LLAMA_POOLING_TYPE_CLS
        LLAMA_POOLING_TYPE_LAST
        LLAMA_POOLING_TYPE_RANK  # used by reranking models to attach the classification head to the graph
 
    cpdef enum llama_attention_type:
        LLAMA_ATTENTION_TYPE_UNSPECIFIED
        LLAMA_ATTENTION_TYPE_CAUSAL
        LLAMA_ATTENTION_TYPE_NON_CAUSAL

    cpdef enum llama_flash_attn_type:
        LLAMA_FLASH_ATTN_TYPE_AUTO
        LLAMA_FLASH_ATTN_TYPE_DISABLED
        LLAMA_FLASH_ATTN_TYPE_ENABLED

    cpdef enum llama_split_mode:
        LLAMA_SPLIT_MODE_NONE   # single GPU
        LLAMA_SPLIT_MODE_LAYER  # split layers and KV across GPUs
        LLAMA_SPLIT_MODE_ROW    # split layers and KV across GPUs, use tensor parallelism if supported
        LLAMA_SPLIT_MODE_TENSOR

    cpdef enum llama_model_kv_override_type:
        LLAMA_KV_OVERRIDE_TYPE_INT
        LLAMA_KV_OVERRIDE_TYPE_FLOAT
        LLAMA_KV_OVERRIDE_TYPE_BOOL
        LLAMA_KV_OVERRIDE_TYPE_STR

    ctypedef struct llama_model_kv_override: # FLATTENED nested union enum
        llama_model_kv_override_type tag
        char key[128]
        int64_t val_i64
        double  val_f64
        bint    val_bool
        char    val_str[128]

    ctypedef struct llama_model_tensor_buft_override:
        const char * pattern
        ggml_backend_buffer_type_t buft

    ctypedef struct llama_logit_bias:
        llama_token token
        float bias

    ctypedef bint (*llama_progress_callback)(float progress, void * user_data);

    ctypedef struct llama_sampler_seq_config: pass

    ctypedef struct llama_context_params: pass


#------------------------------------------------------------------------------
# build-info.h

cdef extern from "build-info.h":

    int llama_build_number()
    const char * llama_commit()
    const char * llama_compiler()
    const char * llama_build_target()


#------------------------------------------------------------------------------
# common.h

cdef extern from "common.h":

    ctypedef std_vector[llama_token] llama_tokens

    ctypedef struct llama_adapter_lora: pass

    cdef cppclass common_adapter_lora_info:
        std_string path
        float scale
        llama_adapter_lora *ptr

    ctypedef struct common_control_vector_load_info: pass

    # -------------------------------------------------------------------------
    # CPU utils

    ctypedef struct common_cpu_params:
        int      n_threads
        bint     cpumask[GGML_MAX_N_THREADS] # CPU affinity mask.
        bint     mask_valid             # Default: any CPU
        ggml_sched_priority  priority   # Scheduling prio : (0 - normal, 1 - medium, 2 - high, 3 - realtime)
        bint     strict_cpu             # Use strict CPU placement
        uint32_t poll                   # Polling (busywait) level (0 - no polling, 100 - mostly polling)

    # -------------------------------------------------------------------------
    # Common params

    cdef enum common_sampler_type:
        pass

    # dimensionality reduction methods, used by cvector-generator
    cpdef enum dimre_method:
        DIMRE_METHOD_PCA
        DIMRE_METHOD_MEAN

    cpdef enum common_conversation_mode:
        COMMON_CONVERSATION_MODE_DISABLED
        COMMON_CONVERSATION_MODE_ENABLED
        COMMON_CONVERSATION_MODE_AUTO

    cpdef enum common_grammar_trigger_type:
        COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN
        COMMON_GRAMMAR_TRIGGER_TYPE_WORD
        COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN
        COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL

    ctypedef struct common_grammar_trigger:
        common_grammar_trigger_type type
        std_string value
        bint at_start
        llama_token token

    cpdef enum common_grammar_type:
        COMMON_GRAMMAR_TYPE_NONE
        COMMON_GRAMMAR_TYPE_USER
        COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT
        COMMON_GRAMMAR_TYPE_TOOL_CALLS

    cdef cppclass common_grammar:
        common_grammar_type type
        std_string grammar
        common_grammar()
        common_grammar(common_grammar_type t, std_string g)
        bint empty()

    cpdef enum  common_params_sampling_config:
        COMMON_PARAMS_SAMPLING_CONFIG_SAMPLERS
        COMMON_PARAMS_SAMPLING_CONFIG_TOP_K
        COMMON_PARAMS_SAMPLING_CONFIG_TOP_P
        COMMON_PARAMS_SAMPLING_CONFIG_MIN_P
        COMMON_PARAMS_SAMPLING_CONFIG_XTC_PROBABILITY
        COMMON_PARAMS_SAMPLING_CONFIG_XTC_THRESHOLD
        COMMON_PARAMS_SAMPLING_CONFIG_TEMP
        COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_LAST_N
        COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT
        COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT
        COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_TAU
        COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_ETA

    # sampler parameters
    ctypedef struct common_params_sampling:
        uint32_t seed  # the seed used to initialize llama_sampler

        int32_t n_prev                     # number of previous tokens to remember
        int32_t n_probs                    # if greater than 0, output the probabilities of top n_probs tokens.
        int32_t min_keep                   # 0 = disabled, otherwise samplers should return at least min_keep tokens
        int32_t top_k                      # <= 0 to use vocab size
        float   top_p                      # 1.0 = disabled
        float   min_p                      # 0.0 = disabled
        float   xtc_probability            # 0.0 = disabled
        float   xtc_threshold              # > 0.5 disables XTC
        float   typ_p                      # typical_p, 1.0 = disabled
        float   temp                       # <= 0.0 to sample greedily, 0.0 to not output probabilities
        float   dynatemp_range             # 0.0 = disabled
        float   dynatemp_exponent          # controls how entropy maps to temperature in dynamic temperature sampler
        int32_t penalty_last_n             # last n tokens to penalize (0 = disable penalty, -1 = context size)
        float   penalty_repeat             # 1.0 = disabled
        float   penalty_freq               # 0.0 = disabled
        float   penalty_present            # 0.0 = disabled
        float   dry_multiplier             # 0.0 = disabled; DRY repetition penalty for tokens extending repetition:
        float   dry_base                   # 0.0 = disabled; multiplier * base ^ (length of sequence before token - allowed length)
        int32_t dry_allowed_length         # tokens extending repetitions beyond this receive penalty
        int32_t dry_penalty_last_n         # how many tokens to scan for repetitions (0 = disable penalty, -1 = context size)
        float   adaptive_target            # select tokens near this probability (valid range 0.0 to 1.0; negative = disabled)
        float   adaptive_decay             # EMA decay for adaptation; history ≈ 1/(1-decay) tokens (0.0 - 0.99)
        int32_t mirostat                   # 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
        float   top_n_sigma                # -1.0 = disabled
        float   mirostat_tau               # target entropy
        float   mirostat_eta               # learning rate
        bint    ignore_eos                 # ignore end-of-sentence
        bint    no_perf                    # disable performance metrics
        bint    timing_per_token
        uint64_t user_sampling_config      # bitfield to track user-specified samplers

        std_vector[std_string] dry_sequence_breakers

        std_vector[common_sampler_type] samplers

        common_grammar                      grammar           # optional grammar constraint (user / output-format / tool-calls)
        bint                                grammar_lazy
        std_vector[common_grammar_trigger]  grammar_triggers  # optional triggers (for lazy grammars)
        std_set[llama_token]                preserved_tokens

        std_vector[llama_logit_bias] logit_bias      # logit biases to apply
        std_vector[llama_logit_bias] logit_bias_eog  # pre-calculated logit biases for EOG tokens

        # The assistant generation prompt already prefilled into the prompt.
        # Fed to the grammar sampler (to advance past pre-existing tokens) and used
        # to determine the reasoning budget sampler's initial state.
        # Only applied when the grammar is of output-format or tool-calls type.
        std_string generation_prompt

        # reasoning budget sampler parameters
        # these are populated by the server/CLI based on chat template params
        int32_t  reasoning_budget_tokens   # -1 = disabled, >= 0 = token budget
        std_vector[llama_token] reasoning_budget_start  # start tag token sequence
        std_vector[llama_token] reasoning_budget_end    # end tag token sequence
        std_vector[llama_token] reasoning_budget_forced # forced sequence (message + end tag)
        std_string              reasoning_budget_message  # message injected before end tag when budget exhausted

        bint backend_sampling


    cpdef enum common_speculative_type:
        COMMON_SPECULATIVE_TYPE_NONE          # no speculative decoding
        COMMON_SPECULATIVE_TYPE_DRAFT         # draft model
        COMMON_SPECULATIVE_TYPE_EAGLE3        # eagle draft model
        COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE  # simple self-speculative decoding
        COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K   # self-speculative decoding with n-gram keys only
        COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V # self-speculative decoding with n-gram keys and 4 m-gram values
        COMMON_SPECULATIVE_TYPE_NGRAM_MOD
        COMMON_SPECULATIVE_TYPE_NGRAM_CACHE   # self-speculative decoding with 3-level n-gram cache
        COMMON_SPECULATIVE_TYPE_COUNT         # number of types, unknown type


    ctypedef struct common_params_model:
        std_string path          # model local path                                           // NOLINT
        std_string url           # model url to download                                      // NOLINT
        std_string hf_repo       # HF repo                                                    // NOLINT
        std_string hf_file       # HF file                                                    // NOLINT
        std_string docker_repo   # Docker repo                                                // NOLINT
        std_string name          # in format <user>/<model>[:<tag>] (tag is optional)         // NOLINT

    ctypedef struct common_ngram_mod
    ctypedef struct llama_model

    # draft-model-based speculative decoding parameters
    ctypedef struct common_params_speculative_draft:
        int32_t n_max   # maximum number of tokens to draft during speculative decoding
        int32_t n_min   # minimum number of draft tokens to use for speculative decoding

        float p_split   # speculative decoding split probability
        float p_min     # minimum speculative decoding probability (greedy)

        common_params_model mparams  # draft model parameters

        llama_model * model  # a llama_model that can be shared by multiple speculative contexts (runtime only)

        llama_context_params cparams  # parameters for the draft llama_context (runtime only)

        int32_t n_ctx         # draft context size
        int32_t n_gpu_layers  # number of layers to store in VRAM for the draft model (-1 - use default)

        ggml_type cache_type_k  # KV cache data type for the K
        ggml_type cache_type_v  # KV cache data type for the V

        common_cpu_params cpuparams
        common_cpu_params cpuparams_batch

        std_vector[ggml_backend_dev_t] devices  # devices to use for offloading

        std_vector[std_pair[std_string, std_string]] replacements  # main to speculative model replacements
        std_vector[llama_model_tensor_buft_override] tensor_buft_overrides

    ctypedef struct common_params_speculative_ngram_mod:
        int32_t n_match
        int32_t n_max
        int32_t n_min
        # std::shared_ptr<common_ngram_mod> obj  # runtime only, not exposed to Python

    ctypedef struct common_params_speculative_ngram_map:
        uint16_t size_n    # ngram size for lookup
        uint16_t size_m    # mgram size for speculative tokens
        uint16_t min_hits  # minimum hits at ngram/mgram lookup for mgram to be proposed

    ctypedef struct common_params_speculative_ngram_cache:
        std_string lookup_cache_static   # path of static ngram cache file for lookup decoding
        std_string lookup_cache_dynamic  # path of dynamic ngram cache file for lookup decoding

    ctypedef struct common_params_speculative:
        common_speculative_type type    # type of speculative decoding

        common_params_speculative_draft draft

        common_params_speculative_ngram_mod ngram_mod
        common_params_speculative_ngram_map ngram_simple
        common_params_speculative_ngram_map ngram_map_k
        common_params_speculative_ngram_map ngram_map_k4v

        common_params_speculative_ngram_cache ngram_cache


    ctypedef struct common_params_vocoder:
        common_params_model model
        std_string speaker_file # speaker file path                                      // NOLINT
        bint use_guide_tokens  # enable guide tokens to improve TTS accuracy            // NOLINT


    ctypedef struct common_params_diffusion:
        int32_t steps        # number of diffusion steps
        bint    visual_mode  # show progressive diffusion on screen

        float   eps          # epsilon for timesteps
        int32_t block_length # block length for generation

        int32_t algorithm    # default algorithm: low-confidence
        float   alg_temp     # algorithm temperature

        float   cfg_scale    # classifier-free guidance scale
        bint    add_gumbel_noise  # add gumbel noise to the logits if temp > 0.0


    # reasoning API response format (not to be confused as chat template's reasoning format)
    cpdef enum common_reasoning_format:
        COMMON_REASONING_FORMAT_NONE
        COMMON_REASONING_FORMAT_AUTO            # Same as deepseek, using `message.reasoning_content`
        COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY # Extract thinking tag contents and return as `message.reasoning_content`, or leave inline in <think> tags in stream mode
        COMMON_REASONING_FORMAT_DEEPSEEK        # Extract thinking tag contents and return as `message.reasoning_content`, including in streaming deltas.
        # do not extend this enum unless you absolutely have to
        # in most cases, use COMMON_REASONING_FORMAT_AUTO
        # see: https://github.com/ggml-org/llama.cpp/pull/15408

    ctypedef struct common_params:
        int32_t n_predict          # new tokens to predict
        int32_t n_ctx              # context size
        int32_t n_batch            # logical batch size for prompt processing (must be >=32 to use BLAS)
        int32_t n_ubatch           # physical batch size for prompt processing (must be >=32 to use BLAS)
        int32_t n_keep             # number of tokens to keep from initial prompt
        int32_t n_chunks           # max number of chunks to process (-1 = unlimited)
        int32_t n_parallel         # number of parallel sequences to decode
        int32_t n_sequences        # number of sequences to decode
        int32_t grp_attn_n         # group-attention factor
        int32_t grp_attn_w         # group-attention width
        int32_t n_print            # print token count every n tokens (-1 = disabled)
        float   rope_freq_base     # RoPE base frequency
        float   rope_freq_scale    # RoPE frequency scaling factor
        float   yarn_ext_factor    # YaRN extrapolation mix factor
        float   yarn_attn_factor   # YaRN magnitude scaling factor
        float   yarn_beta_fast     # YaRN low correction dim
        float   yarn_beta_slow     # YaRN high correction dim
        int32_t yarn_orig_ctx      # YaRN original context length

        std_vector[ggml_backend_dev_t] devices # devices to use for offloading
        int32_t n_gpu_layers       # number of layers to store in VRAM (-1 - use default)
        int32_t n_gpu_layers_draft # number of layers to store in VRAM for the draft model (-1 - use default)
        int32_t main_gpu           # the GPU that is used for scratch and small tensors
        float   tensor_split[128]  # how split tensors should be distributed across GPUs
        bint    fit_params         # whether to fit unset model/context parameters to free device memory
        bint    fit_params_print   # print the estimated required memory to run the model
        int32_t fit_params_min_ctx # minimum context size to set when trying to reduce memory use

        # margin per device in bytes for fitting parameters to free memory:
        std_vector[size_t] fit_params_target

        llama_split_mode        split_mode         # how to split the model across GPUs

        common_cpu_params cpuparams
        common_cpu_params cpuparams_batch

        ggml_backend_sched_eval_callback cb_eval
        void * cb_eval_user_data

        ggml_numa_strategy numa

        llama_rope_scaling_type rope_scaling_type
        llama_pooling_type      pooling_type       # pooling type for embeddings
        llama_attention_type    attention_type     # attention type for embeddings
        llama_flash_attn_type   flash_attn_type    # whether to use Flash Attention

        common_params_sampling sampling
        common_params_speculative speculative
        common_params_vocoder     vocoder
        common_params_diffusion   diffusion
        common_params_model model

        std_set[std_string] model_alias     # model aliases
        std_set[std_string] model_tags      # model tags (informational, not used for routing)
        std_string hf_token             # HF token
        std_string prompt               #
        std_string prompt_file          # store the external prompt file name
        std_string path_prompt_cache    # path to file for saving/loading prompt eval state
        std_string input_prefix         # string to prefix user inputs with
        std_string input_suffix         # string to suffix user inputs with
        std_string logits_file          # file for saving *all* logits

        # llama-debug specific options
        std_string logits_output_dir         # directory for saving logits output files
        bint        save_logits              # whether to save logits to files
        std_vector[std_string] tensor_filter # filter tensor names for debug output (regex)

        std_vector[std_string] in_files     # all input files
        std_vector[std_string] antiprompt   # strings upon which more user input is prompted (a.k.a. reverse prompts)
        std_vector[llama_model_kv_override] kv_overrides
        std_vector[llama_model_tensor_buft_override] tensor_buft_overrides


        bint lora_init_without_apply # only load lora to memory, but do not apply it to ctx (user can manually apply lora later using llama_lora_adapter_apply)
        std_vector[common_adapter_lora_info] lora_adapters # lora adapter path with user defined scale

        std_vector[common_control_vector_load_info] control_vectors # control vector with user defined scale

        int32_t verbosity
        int32_t control_vector_layer_start # layer range for control vector
        int32_t control_vector_layer_end   # layer range for control vector
        bint    offline

        int32_t ppl_stride          # stride for perplexity calculations. If left at 0, the pre-existing approach will be used.
        int32_t ppl_output_type     # = 0 -> ppl output is as usual, = 1 -> ppl output is num_tokens, ppl, one per line

        bint   hellaswag            # compute HellaSwag score over random tasks from datafile supplied in prompt
        size_t hellaswag_tasks      # number of tasks to use when computing the HellaSwag score

        bint   winogrande           # compute Winogrande score over random tasks from datafile supplied in prompt
        size_t winogrande_tasks     # number of tasks to use when computing the Winogrande score. If 0, all tasks will be computed

        bint   multiple_choice      # compute TruthfulQA score over random tasks from datafile supplied in prompt
        size_t multiple_choice_tasks # number of tasks to use when computing the TruthfulQA score. If 0, all tasks will be computed

        bint   kl_divergence        # compute KL divergence

        bint   check                # check rather than generate results for llama-results

        bint usage                  # print usage
        bint completion             # print source-able completion script
        bint use_color              # use color to distinguish generations and inputs
        bint special                # enable special token output
        bint interactive            # interactive mode
        bint interactive_first      # wait for user input immediately
        bint prompt_cache_all       # save user input and generations to prompt cache
        bint prompt_cache_ro        # open the prompt cache read-only and do not update it

        bint escape                 # escape "\n", "\r", "\t", "\'", "\"", and "\\"
        bint multiline_input        # reverse the usage of `\`
        bint simple_io              # improves compatibility with subprocesses and limited consoles
        bint cont_batching          # insert new sequences for decoding on-the-fly
        bint no_perf                # disable performance metric
        bint show_timings           # show timing information on CLI
        bint ctx_shift              # context shift on inifinite text generation
        bint swa_full               # use full-size SWA cache (https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
        bint kv_unified             # enable unified KV cache

        bint input_prefix_bos       # prefix BOS to user inputs, preceding input_prefix
        bint use_mmap               # use mmap for faster loads
        bint use_direct_io          # read from disk without buffering
        bint use_mlock              # use mlock to keep model in memory
        bint verbose_prompt         # print prompt tokens before generation
        bint display_prompt         # print prompt before generation
        bint no_kv_offload          # disable KV offloading
        bint warmup                 # warmup run
        bint check_tensors          # validate tensor data
        bint no_op_offload          # globally disable offload host tensor operations to device
        bint no_extra_bufts         # disable extra buffer types (used for weight repacking)
        bint no_host                # bypass host buffer allowing extra buffers to be used
        bint single_turn            # single turn chat conversation

        ggml_type cache_type_k      # KV cache data type for the K
        ggml_type cache_type_v      # KV cache data type for the V

        common_conversation_mode conversation_mode

        # multimodal models (see tools/mtmd)
        common_params_model mmproj
        bint mmproj_use_gpu         # use GPU for multimodal model
        bint no_mmproj              # explicitly disable multimodal model

        std_vector[std_string] image # path to image file(s)
        int32_t image_min_tokens
        int32_t image_max_tokens

        # finetune
        # We do not need to export finetune fields to Python

        # embedding
        bint embedding              # get only sentence embedding
        int32_t embd_normalize      # normalisation for embeddings (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)
        std_string embd_out         # empty = default, "array" = [[],[]...], "json" = openai style, "json+" = same "json" + cosine similarity matrix
        std_string embd_sep         # separator of embeddings
        std_string cls_sep          # separator of classification sequences

        # server params
        int32_t port                # server listens on this network port
        bint    reuse_port          # allow multiple sockets to bind to the same port
        int32_t timeout_read        # http read timeout in seconds
        int32_t timeout_write       # http write timeout in seconds
        int32_t n_threads_http      # number of threads to process HTTP requests (TODO: support threadpool)
        int32_t n_cache_reuse       # min chunk size to reuse from the cache via KV shifting
        bint    cache_prompt        # whether to enable prompt caching
        bint    cache_idle_slots    # save and clear idle slots upon starting a new task
        int32_t n_ctx_checkpoints   # max number of context checkpoints per slot
        int32_t checkpoint_every_nt   # make a checkpoint every n tokens during prefill
        int32_t cache_ram_mib       # -1 = no limit, 0 - disable, 1 = 1 MiB, etc.

        std_string hostname
        std_string public_path
        std_string api_prefix
        std_string chat_template
        bint use_jinja
        bint enable_chat_template
        bint force_pure_content_parser

        common_reasoning_format reasoning_format
        int32_t enable_reasoning    # -1 = auto, 0 = disable, 1 = enable
        bint prefill_assistant      # if true, any trailing assistant message will be prefilled into the response
        int32_t sleep_idle_seconds  # if >0, server will sleep after this many seconds of idle time

        std_vector[std_string] api_keys

        std_string ssl_file_key 
        std_string ssl_file_cert

        std_map[std_string, std_string] default_template_kwargs

        # webui configs
        bint webui
        bint webui_mcp_proxy
        std_string webui_config_json

        bint endpoint_slots
        bint endpoint_props
        bint endpoint_metrics

        # enable built-in tools
        std_vector[std_string] server_tools

        # router server configs
        std_string models_dir    # directory containing models for the router server
        std_string models_preset # directory containing model presets for the router server
        int32_t models_max       # maximum number of models to load simultaneously
        bint models_autoload     # automatically load models when requested via the router server

        bint log_json

        std_string slot_save_path
        std_string media_path    # path to directory for loading media files

        float slot_prompt_similarity

        # batched-bench params
        bint is_pp_shared
        bint is_tg_separate

        std_vector[int32_t] n_pp
        std_vector[int32_t] n_tg
        std_vector[int32_t] n_pl

        # retrieval params
        std_vector[std_string] context_files # context files to embed

        int32_t chunk_size      # chunk size for context embedding

        std_string chunk_separator # chunk separator for context embedding

        # passkey params
        int32_t n_junk      # number of times to repeat the junk text
        int32_t i_pos       # position of the passkey in the junk text

        # imatrix params
        int32_t n_out_freq       # output the imatrix every n_out_freq iterations
        int32_t n_save_freq      # save the imatrix every n_save_freq iterations
        int32_t i_chunk          # start processing from this chunk
        int8_t  imat_dat         # whether the legacy imatrix.dat format should be output (gguf <= 0 < dat)

        bint process_output      # collect data for the output tensor
        bint compute_ppl         # whether to compute perplexity
        bint show_statistics     # show imatrix statistics per tensor
        bint parse_special       # whether to parse special tokens during imatrix tokenization

        # cvector-generator params
        int n_pca_batch
        int n_pca_iterations
        dimre_method cvector_dimre_method
        std_string cvector_positive_file
        std_string cvector_negative_file

        bint spm_infill

        # batched-bench params
        bint batched_bench_output_jsonl
    
        # common params
        std_string out_file      # output filename for all example programs

        # optional callback for model loading progress and cancellation:
        # called with a progress value between 0.0 and 1.0.
        # return false from callback to abort model loading or true to continue
        llama_progress_callback load_progress_callback
        void *                  load_progress_callback_user_data
        bint no_alloc # Don't allocate model buffers


    
cdef extern from "sampling.h":

    std_vector[common_sampler_type] common_sampler_types_from_names(const std_vector[std_string] & names, bint allow_alt_names)
    
    std_string common_sampler_type_to_str(common_sampler_type cnstr)
