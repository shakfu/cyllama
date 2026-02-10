# distutils: language=c++

from libc.stdint cimport int32_t, int8_t, int64_t, uint32_t, uint64_t, uint8_t
from libc.stdio cimport FILE
from libcpp.string cimport string as std_string
from libcpp.string cimport string_view as std_string_view
from libcpp.vector cimport vector as std_vector
from libcpp.set cimport set as std_set
from libcpp.memory cimport unique_ptr
from libcpp.set cimport set as std_set
from libcpp.pair cimport pair as std_pair
from libcpp.map cimport map as std_map

cimport ggml
cimport llama

#------------------------------------------------------------------------------
# common.h

cdef extern from "common.h":

    ctypedef struct common_adapter_lora_info:
        std_string path
        float scale
        llama.llama_adapter_lora * ptr

    ctypedef std_vector[llama.llama_token] llama_tokens

    ctypedef struct common_control_vector_load_info: pass

    # -------------------------------------------------------------------------
    # Build info

    cdef int LLAMA_BUILD_NUMBER
    cdef const char * LLAMA_COMMIT
    cdef const char * LLAMA_COMPILER
    cdef const char * LLAMA_BUILD_TARGET

    # -------------------------------------------------------------------------
    # CPU utils

    ctypedef struct cpu_params:
        int      n_threads
        bint     cpumask[ggml.GGML_MAX_N_THREADS] # CPU affinity mask.
        bint     mask_valid             # Default: any CPU
        ggml.ggml_sched_priority  priority   # Scheduling prio : (0 - normal, 1 - medium, 2 - high, 3 - realtime)
        bint     strict_cpu             # Use strict CPU placement
        uint32_t poll                   # Polling (busywait) level (0 - no polling, 100 - mostly polling)


    cdef int32_t cpu_get_num_physical_cores()
    cdef int32_t cpu_get_num_math()

    # -------------------------------------------------------------------------
    # Common params

    cdef enum llama_example:
        LLAMA_EXAMPLE_BATCHED
        LLAMA_EXAMPLE_DEBUG
        LLAMA_EXAMPLE_COMMON
        LLAMA_EXAMPLE_SPECULATIVE
        LLAMA_EXAMPLE_COMPLETION
        LLAMA_EXAMPLE_CLI
        LLAMA_EXAMPLE_EMBEDDING
        LLAMA_EXAMPLE_PERPLEXITY
        LLAMA_EXAMPLE_RETRIEVAL
        LLAMA_EXAMPLE_PASSKEY
        LLAMA_EXAMPLE_IMATRIX
        LLAMA_EXAMPLE_BENCH
        LLAMA_EXAMPLE_SERVER
        LLAMA_EXAMPLE_CVECTOR_GENERATOR
        LLAMA_EXAMPLE_EXPORT_LORA
        LLAMA_EXAMPLE_MTMD
        LLAMA_EXAMPLE_LOOKUP
        LLAMA_EXAMPLE_PARALLEL
        LLAMA_EXAMPLE_TTS
        LLAMA_EXAMPLE_DIFFUSION
        LLAMA_EXAMPLE_FINETUNE
        LLAMA_EXAMPLE_FIT_PARAMS
        LLAMA_EXAMPLE_COUNT

    cdef enum common_sampler_type:
        COMMON_SAMPLER_TYPE_NONE        = 0
        COMMON_SAMPLER_TYPE_DRY         = 1
        COMMON_SAMPLER_TYPE_TOP_K       = 2
        COMMON_SAMPLER_TYPE_TOP_P       = 3
        COMMON_SAMPLER_TYPE_MIN_P       = 4
        #COMMON_SAMPLER_TYPE_TFS_Z       = 5
        COMMON_SAMPLER_TYPE_TYPICAL_P   = 6
        COMMON_SAMPLER_TYPE_TEMPERATURE = 7
        COMMON_SAMPLER_TYPE_XTC         = 8
        COMMON_SAMPLER_TYPE_INFILL      = 9
        COMMON_SAMPLER_TYPE_PENALTIES   = 10
        COMMON_SAMPLER_TYPE_TOP_N_SIGMA = 11
        COMMON_SAMPLER_TYPE_ADAPTIVE_P  = 12

    # dimensionality reduction methods, used by cvector-generator
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

    ctypedef struct common_grammar_trigger:
        common_grammar_trigger_type type
        std_string value
        llama.llama_token token


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
        float   adaptive_decay             # EMA decay for adaptation; history = 1/(1-decay) tokens (0.0 - 0.99)
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

        std_string grammar # optional BNF-like grammar to constrain sampling
        bint grammar_lazy
        std_vector[common_grammar_trigger] grammar_triggers
        std_set[llama.llama_token] preserved_tokens

        std_vector[llama.llama_logit_bias] logit_bias     # logit biases to apply
        std_vector[llama.llama_logit_bias] logit_bias_eog # pre-calculated logit biases for EOG tokens

        bint backend_sampling

        # print the parameters into a string
        std_string print()


    ctypedef struct common_params_model:
        std_string path         # model local path                                           // NOLINT
        std_string url          # model url to download                                      // NOLINT
        std_string hf_repo      # HF repo                                                    // NOLINT
        std_string hf_file      # HF file                                                    // NOLINT
        std_string docker_repo  # Docker repo                                                // NOLINT
        std_string name         # in format <user>/<model>[:<tag>] (tag is optional)         // NOLINT

    ctypedef struct common_params_speculative:
        std_vector[ggml.ggml_backend_dev_t] devices # devices to use for offloading
        int32_t n_ctx           # draft context size
        int32_t n_max           # maximum number of tokens to draft during speculative decoding
        int32_t n_min           # minimum number of draft tokens to use for speculative decoding
        int32_t n_gpu_layers    # number of layers to store in VRAM for the draft model (-1 - use default)
        float   p_split         # speculative decoding split probability
        float   p_min           # minimum speculative decoding probability (greedy)
        std_vector[std_pair[std_string, std_string]] replacements

        ggml.ggml_type cache_type_k # KV cache data type for the K
        ggml.ggml_type cache_type_v # KV cache data type for the V

        cpu_params cpuparams
        cpu_params cpuparams_batch

        std_string lookup_cache_static   # path of static ngram cache file for lookup decoding
        std_string lookup_cache_dynamic  # path of dynamic ngram cache file for lookup decoding

        common_params_model mparams_dft  # draft model for speculative decoding

    ctypedef struct common_params_vocoder:
        common_params_model model

        std_string speaker_file     # speaker file path
        bint use_guide_tokens       # enable guide tokens to improve TTS accuracy

    ctypedef struct common_params_diffusion:
        int32_t steps
        bint visual_mode
        float eps               # epsilon for timesteps
        int32_t block_length    # block length for generation
        int32_t algorithm       # default algorithm: low-confidence
        float alg_temp          # algorithm temperature
        float cfg_scale         # classifier-free guidance scale
        bint add_gumbel_noise   # add gumbel noise to the logits if temp > 0.0

    cdef enum common_reasoning_format:
        COMMON_REASONING_FORMAT_NONE
        COMMON_REASONING_FORMAT_AUTO
        COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY # Extract thinking tag contents and return as `message.reasoning_content`, or leave inline in <think> tags in stream mode
        COMMON_REASONING_FORMAT_DEEPSEEK        # Extract thinking tag contents and return as `message.reasoning_content`
        COMMON_REASONING_FORMAT_GRANITE         # Extract thinking tag contents and return as `message.reasoning_content`, including in streaming deltas.

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
        # float   defrag_thold       # KV cache defragmentation threshold

        std_vector[ggml.ggml_backend_dev_t] devices # devices to use for offloading
        
        int32_t n_gpu_layers       # number of layers to store in VRAM (-1 - use default)
        int32_t main_gpu           # the GPU that is used for scratch and small tensors
        float   tensor_split[128]  # how split tensors should be distributed across GPUs
        bint    fit_params         # whether to fit unset model/context parameters to free device memory
        int32_t fit_params_min_ctx # minimum context size to set when trying to reduce memory use

        # margin per device in bytes for fitting parameters to free memory:
        std_vector[size_t] fit_params_target

        llama.llama_split_mode split_mode # how to split the model across GPUs

        cpu_params cpuparams
        cpu_params cpuparams_batch

        ggml.ggml_backend_sched_eval_callback cb_eval
        void * cb_eval_user_data

        ggml.ggml_numa_strategy numa

        llama.llama_rope_scaling_type rope_scaling_type
        llama.llama_pooling_type      pooling_type       # pooling type for embeddings
        llama.llama_attention_type    attention_type     # attention type for embeddings

        common_params_sampling    sampling
        common_params_speculative speculative
        common_params_vocoder     vocoder
        common_params_diffusion   diffusion

        common_params_model model 

        std_string model_alias          # model alias
        std_string hf_token             # HF token
        std_string prompt               #
        std_string system_prompt
        std_string prompt_file          # store the external prompt file name
        std_string path_prompt_cache    # path to file for saving/loading prompt eval state
        std_string input_prefix         # string to prefix user inputs with
        std_string input_suffix         # string to suffix user inputs with
        std_string logits_file          # file for saving *all* logits

        std_vector[std_string] in_files     # all input files
        std_vector[std_string] antiprompt   # strings upon which more user input is prompted (a.k.a. reverse prompts)
        std_vector[llama.llama_model_kv_override] kv_overrides
        # std_vector[llama.llama_model_tensor_buft_override] tensor_buft_overrides

        bint lora_init_without_apply # only load lora to memory, but do not apply it to ctx (user can manually apply lora later using llama_adapter_lora_apply)
        # vector[common_adapter_lora_info] lora_adapters # lora adapter path with user defined scale

        # vector[common_control_vector_load_info] control_vectors # control vector with user defined scale

        int32_t verbosity
        int32_t control_vector_layer_start # layer range for control vector
        int32_t control_vector_layer_end   # layer range for control vector

        int32_t ppl_stride          # stride for perplexity calculations. If left at 0, the pre-existing approach will be used.
        int32_t ppl_output_type     # = 0 -> ppl output is as usual, = 1 -> ppl output is num_tokens, ppl, one per line

        bint   hellaswag            # compute HellaSwag score over random tasks from datafile supplied in prompt
        size_t hellaswag_tasks      # number of tasks to use when computing the HellaSwag score

        bint   winogrande           # compute Winogrande score over random tasks from datafile supplied in prompt
        size_t winogrande_tasks     # number of tasks to use when computing the Winogrande score. If 0, all tasks will be computed

        bint   multiple_choice      # compute TruthfulQA score over random tasks from datafile supplied in prompt
        size_t multiple_choice_tasks # number of tasks to use when computing the TruthfulQA score. If 0, all tasks will be computed

        bint   kl_divergence        # compute KL divergence

        # std::function<void(int, char **)> print_usage # print example-specific usage and example
        bint usage                  # print usage
        bint use_color              # use color to distinguish generations and inputs
        bint special                # enable special token output
        bint interactive            # interactive mode
        bint interactive_first      # wait for user input immediately
        bint conversation           # conversation mode (does not print special tokens and suffix/prefix)
        bint prompt_cache_all       # save user input and generations to prompt cache
        bint prompt_cache_ro        # open the prompt cache read-only and do not update it

        bint escape                 # escape "\n", "\r", "\t", "\'", "\"", and "\\"
        bint multiline_input        # reverse the usage of `\`
        bint simple_io              # improves compatibility with subprocesses and limited consoles
        bint cont_batching          # insert new sequences for decoding on-the-fly
        bint flash_attn             # flash attention
        bint no_perf                # disable performance metric
        bint show_timings           # show timing information on CLI
        bint ctx_shift              # context shift on inifinite text generation
        bint swa_full               # use full-size SWA cache (https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
        bint kv_unified             # enable unified KV cache

        bint input_prefix_bos       # prefix BOS to user inputs, preceding input_prefix
        bint use_mmap               # enable mmap to use filesystem cache
        bint use_direct_io          # read from disk without buffering for faster model loading
        bint use_mlock              # use mlock to keep model in memory
        bint verbose_prompt         # print prompt tokens before generation
        bint display_prompt         # print prompt before generation
        bint infill                 # use infill mode
        bint dump_kv_cache          # dump the KV cache contents for debugging purposes
        bint no_kv_offload          # disable KV offloading
        bint warmup                 # warmup run
        bint check_tensors          # validate tensor data
        bint no_op_offload          # globally disable offload host tensor operations to device
        bint no_extra_bufts         # disable extra buffer types (used for weight repacking)
        bint no_host                # bypass host buffer allowing extra buffers to be used
        bint single_turn            # single turn chat conversation

        ggml.ggml_type cache_type_k      # KV cache data type for the K
        ggml.ggml_type cache_type_v      # KV cache data type for the V

        common_conversation_mode conversation_mode

        # multimodal models (see tools/mtmd)
        common_params_model mmproj
        bint mmproj_use_gpu          # use GPU for multimodal model
        bint no_mmproj               # explicitly disable multimodal model
        std_vector[std_string] image # path to image file(s)
        int image_min_tokens         # minimum number of tokens for image input
        int image_max_tokens         # maximum number of tokens for image input

        # embedding
        bint embedding              # get only sentence embedding
        int32_t embd_normalize      # normalisation for embeddings (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)
        std_string embd_out         # empty = default, "array" = [[],[]...], "json" = openai style, "json+" = same "json" + cosine similarity matrix
        std_string embd_sep         # separator of embeddings
        std_string cls_sep          # separator of classification sequences

        # server params
        int32_t port                # server listens on this network port
        int32_t timeout_read        # http read timeout in seconds
        int32_t timeout_write       # http write timeout in seconds
        int32_t n_threads_http      # number of threads to process HTTP requests (TODO: support threadpool)
        int32_t n_cache_reuse       # min chunk size to reuse from the cache via KV shifting
        bint    cache_prompt        # whether to enable prompt caching
        int32_t n_ctx_checkpoints   # max number of context checkpoints per slot
        int32_t cache_ram_mib       # -1 = no limit, 0 = disable, 1 = 1 MiB, etc.

        std_string hostname
        std_string public_path
        std_string api_prefix
        std_string chat_template
        bint use_jinja
        bint enable_chat_template
        common_reasoning_format reasoning_format
        int reasoning_budget
        bint prefill_assistant
        int sleep_idle_seconds      # if >0, server will sleep after this many seconds of idle time

        std_vector[std_string] api_keys

        std_string ssl_file_key
        std_string ssl_file_cert

        std_map[std_string, std_string] default_template_kwargs

        # webui configs
        bint webui
        std_string webui_config_json

        # "advanced" endpoints are disabled by default for better security
        bint endpoint_slots
        bint endpoint_props
        bint endpoint_metrics

        # router server configs
        std_string models_dir     # directory containing models for the router server
        std_string models_preset  # directory containing model presets for the router server
        int models_max            # maximum number of models to load simultaneously
        bint models_autoload      # automatically load models when requested via the router server

        bint log_json

        std_string slot_save_path
        std_string media_path     # path to directory for loading media files

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
        std_string out_file # save the resulting imatrix to this file

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
        std_string out_file # output filename for all example programs
        # optional callback for model loading progress and cancellation:
        # called with a progress value between 0.0 and 1.0.
        # return false from callback to abort model loading or true to continue
        llama.llama_progress_callback load_progress_callback
        void * load_progress_callback_user_data


    # call once at the start of a program if it uses libcommon
    # initializes the logging system and prints info about the build
    cdef void common_init()

    cdef std_string common_params_get_system_info(const common_params & params)

    cdef bint parse_cpu_range(const std_string & range_, bint(&boolmask)[ggml.GGML_MAX_N_THREADS])
    cdef bint parse_cpu_mask(const std_string & mask, bint(&boolmask)[ggml.GGML_MAX_N_THREADS])
    cdef void postprocess_cpu_params(cpu_params & cpuparams, const cpu_params * role_model)
    cdef bint set_process_priority(ggml.ggml_sched_priority prio)

    # -------------------------------------------------------------------------
    # String utils

    cdef std_string string_format(const char * fmt, ...)

    cdef std_string string_strip(const std_string & str)
    cdef std_string string_get_sortable_timestamp()

    cdef std_string string_join(const std_vector[std_string] & values, const std_string & separator)
    cdef std_vector[std_string] string_split(const std_string & str, const std_string & delimiter)
    cdef std_string string_repeat(const std_string & str, size_t n)

    cdef void string_replace_all(std_string & s, const std_string & search, const std_string & replace)

    cdef std_string regex_escape(const std_string & s)

    cdef bint string_starts_with(const std_string & str, const std_string & prefix)
    cdef bint string_ends_with(const std_string_view & str, const std_string_view & suffix)
    cdef bint string_remove_suffix(std_string & str, const std_string_view & suffix)
    cdef size_t string_find_partial_stop(const std_string_view & str, const std_string_view & stop)

    cdef bint string_parse_kv_override(const char * data, std_vector[llama.llama_model_kv_override] & overrides)
    cdef void string_process_escapes(std_string & input)

    cdef std_string string_from(bint value)
    cdef std_string string_from(const std_vector[int] & values)
    cdef std_string string_from(const llama.llama_context * ctx, const std_vector[llama.llama_token] & tokens)
    cdef std_string string_from(const llama.llama_context * ctx, const llama.llama_batch & batch)

    cdef std_vector[std_string] string_split(const std_string & str, char delim)
    cdef std_vector[std_string] string_split[std_string](const std_string & input, char separator)

    # -------------------------------------------------------------------------
    # Filesystem utils

    cdef bint fs_validate_filename(const std_string & filename, bint allow_subdirs)
    cdef bint fs_create_directory_with_parents(const std_string & path)
    cdef bint fs_is_directory(const std_string & path)

    cdef std_string fs_get_cache_directory()
    cdef std_string fs_get_cache_file(const std_string & filename)

    ctypedef struct common_file_info:
        std_string path
        std_string name
        size_t size     # in bytes
        bint   is_dir

    cdef std_vector[common_file_info] fs_list(const std_string & path, bint include_directories)

    # -------------------------------------------------------------------------
    # Model utils

    # note: defines object's lifetime
    ctypedef struct common_init_result:
        llama.llama_model_ptr   model
        llama.llama_context_ptr context

        std_vector[llama.llama_adapter_lora_ptr] lora

    cdef common_init_result common_init_from_params(common_params & params)

    cdef llama.llama_model_params common_model_params_to_llama(common_params & params)
    cdef llama.llama_context_params common_context_params_to_llama(const common_params & params)
    cdef ggml.ggml_threadpool_params ggml_threadpool_params_from_cpu_params(const cpu_params & params)

    # clear LoRA adapters from context, then apply new list of adapters
    cdef void common_set_adapter_lora(llama.llama_context * ctx, std_vector[common_adapter_lora_info] & lora);

    cdef std_string get_model_endpoint()

    # -------------------------------------------------------------------------
    # Batch utils

    void common_batch_clear(llama.llama_batch & batch)

    void common_batch_add(
        llama.llama_batch & batch,
        llama.llama_token id,
        llama.llama_pos pos,
        const std_vector[llama.llama_seq_id] & seq_ids,
        bint logits)

    # -------------------------------------------------------------------------
    # Token utils

    # longest common prefix
    cdef size_t common_lcp(const llama_tokens & a, const llama_tokens & b)

    # longet common subsequence
    cdef size_t common_lcs(const llama_tokens & a, const llama_tokens & b)

    # -------------------------------------------------------------------------
    # Vocab utils

    # tokenizes a string into a vector of tokens
    # should work similar to Python's `tokenizer.encode`
    std_vector[llama.llama_token] common_tokenize(
            const llama.llama_context * ctx,
            const std_string & text,
            bint add_special,
            bint parse_special)

    std_vector[llama.llama_token] common_tokenize(
            const llama.llama_vocab * vocab,
            const std_string & text,
            bint add_special,
            bint parse_special)

    # tokenizes a token into a piece, optionally renders special/control tokens
    # should work similar to Python's `tokenizer.id_to_piece`
    std_string common_token_to_piece(
            const llama.llama_context * ctx,
            llama.llama_token token,
            bint special)

    std_string common_token_to_piece(
            const llama.llama_vocab * vocab,
            llama.llama_token   token,
            bint special)

    # detokenizes a vector of tokens into a string
    # should work similar to Python's `tokenizer.decode`
    # optionally renders special/control tokens
    std_string common_detokenize(
            const llama.llama_context * ctx,
            const std_vector[llama.llama_token] & tokens,
            bint special)

    std_string common_detokenize(
            const llama.llama_vocab * vocab,
            const std_vector[llama.llama_token] & tokens,
            bint special)

    # -------------------------------------------------------------------------
    # Embedding utils

    # TODO: repace embd_norm with an enum
    cdef void common_embd_normalize(const float * inp, float * out, int n, int embd_norm)

    cdef float common_embd_similarity_cos(const float * embd1, const float * embd2, int n)

    # -------------------------------------------------------------------------
    # Control vector utils


    ctypedef struct common_control_vector_data:
        int n_embd

        # stores data for layers [1, n_layer] where n_layer = data.size() / n_embd
        std_vector[float] data

    ctypedef struct common_control_vector_load_info:
        float strength

        std_string fname


    # Load control vectors, scale each by strength, and add them together.
    # On error, returns {-1, empty}
    cdef common_control_vector_data common_control_vector_load(const std_vector[common_control_vector_load_info] & load_infos);


    # -------------------------------------------------------------------------
    # Split utils

    const char * const LLM_KV_SPLIT_NO            = "split.no"
    const char * const LLM_KV_SPLIT_COUNT         = "split.count"
    const char * const LLM_KV_SPLIT_TENSORS_COUNT = "split.tensors.count"


    # -------------------------------------------------------------------------
    # MoE utils

    const char * const LLM_FFN_EXPS_REGEX = "\\.ffn_(up|down|gate)_exps"

    cdef std_string llm_ffn_exps_block_regex(int idx)
    cdef llama.llama_model_tensor_buft_override llm_ffn_exps_cpu_override()

    # -------------------------------------------------------------------------
    # Training utils

    cdef ggml.ggml_opt_dataset_t common_opt_dataset_init(llama.llama_context * ctx, const std_vector[llama.llama_token] & tokens, int64_t stride);

