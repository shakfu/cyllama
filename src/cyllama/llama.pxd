# distutils: language=c++

from libc.stdint cimport int32_t, int8_t, int64_t, uint32_t, uint64_t, uint8_t
from libc.stdio cimport FILE
from libcpp.string cimport string as std_string
from libcpp.vector cimport vector as std_vector
from libcpp.set cimport set as std_set
from libcpp.memory cimport unique_ptr as std_unique_ptr
from libcpp.set cimport set as std_set

cimport ggml

#------------------------------------------------------------------------------
# constants


#------------------------------------------------------------------------------
# llama.h

cdef extern from "llama.h":

    long LLAMA_DEFAULT_SEED
    int LLAMA_TOKEN_NULL

    ctypedef struct llama_vocab: pass
    ctypedef struct llama_model: pass
    ctypedef struct llama_context: pass
    ctypedef struct llama_sampler: pass
    ctypedef struct llama_memory_i: pass

    ctypedef llama_memory_i * llama_memory_t

    ctypedef int32_t llama_pos
    ctypedef int32_t llama_token
    ctypedef int32_t llama_seq_id

    cdef enum llama_vocab_type:
        LLAMA_VOCAB_TYPE_NONE = 0   # For models without vocab
        LLAMA_VOCAB_TYPE_SPM  = 1   # LLaMA tokenizer based on byte-level BPE with byte fallback
        LLAMA_VOCAB_TYPE_BPE  = 2   # GPT-2 tokenizer based on byte-level BPE
        LLAMA_VOCAB_TYPE_WPM  = 3   # BERT tokenizer based on WordPiece
        LLAMA_VOCAB_TYPE_UGM  = 4   # T5 tokenizer based on Unigram
        LLAMA_VOCAB_TYPE_RWKV = 5   # RWKV tokenizer based on greedy tokenization
        LLAMA_VOCAB_TYPE_PLAMO2 = 6 # PLaMo-2 tokenizer based on Aho-Corasick with dynamic programming

    cdef enum llama_rope_type:
        LLAMA_ROPE_TYPE_NONE   = -1
        LLAMA_ROPE_TYPE_NORM   = 0
        LLAMA_ROPE_TYPE_NEOX   = ggml.GGML_ROPE_TYPE_NEOX
        LLAMA_ROPE_TYPE_MROPE  = ggml.GGML_ROPE_TYPE_MROPE
        LLAMA_ROPE_TYPE_VISION = ggml.GGML_ROPE_TYPE_VISION

    cdef enum llama_token_type:
        LLAMA_TOKEN_TYPE_UNDEFINED    = 0
        LLAMA_TOKEN_TYPE_NORMAL       = 1
        LLAMA_TOKEN_TYPE_UNKNOWN      = 2
        LLAMA_TOKEN_TYPE_CONTROL      = 3
        LLAMA_TOKEN_TYPE_USER_DEFINED = 4
        LLAMA_TOKEN_TYPE_UNUSED       = 5
        LLAMA_TOKEN_TYPE_BYTE         = 6

    cdef enum llama_token_attr:
        LLAMA_TOKEN_ATTR_UNDEFINED    = 0
        LLAMA_TOKEN_ATTR_UNKNOWN      = 1 << 0
        LLAMA_TOKEN_ATTR_UNUSED       = 1 << 1
        LLAMA_TOKEN_ATTR_NORMAL       = 1 << 2
        LLAMA_TOKEN_ATTR_CONTROL      = 1 << 3
        LLAMA_TOKEN_ATTR_USER_DEFINED = 1 << 4
        LLAMA_TOKEN_ATTR_BYTE         = 1 << 5
        LLAMA_TOKEN_ATTR_NORMALIZED   = 1 << 6
        LLAMA_TOKEN_ATTR_LSTRIP       = 1 << 7
        LLAMA_TOKEN_ATTR_RSTRIP       = 1 << 8
        LLAMA_TOKEN_ATTR_SINGLE_WORD  = 1 << 9

    cdef enum llama_ftype:
        LLAMA_FTYPE_ALL_F32              = 0
        LLAMA_FTYPE_MOSTLY_F16           = 1
        LLAMA_FTYPE_MOSTLY_Q4_0          = 2
        LLAMA_FTYPE_MOSTLY_Q4_1          = 3
        # LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4  # tok_embeddings.weight and output.weight are F16
        # LLAMA_FTYPE_MOSTLY_Q4_2       = 5     # support has been removed
        # LLAMA_FTYPE_MOSTLY_Q4_3       = 6     # support has been removed
        LLAMA_FTYPE_MOSTLY_Q8_0          = 7
        LLAMA_FTYPE_MOSTLY_Q5_0          = 8
        LLAMA_FTYPE_MOSTLY_Q5_1          = 9
        LLAMA_FTYPE_MOSTLY_Q2_K          = 10
        LLAMA_FTYPE_MOSTLY_Q3_K_S        = 11
        LLAMA_FTYPE_MOSTLY_Q3_K_M        = 12
        LLAMA_FTYPE_MOSTLY_Q3_K_L        = 13
        LLAMA_FTYPE_MOSTLY_Q4_K_S        = 14
        LLAMA_FTYPE_MOSTLY_Q4_K_M        = 15
        LLAMA_FTYPE_MOSTLY_Q5_K_S        = 16
        LLAMA_FTYPE_MOSTLY_Q5_K_M        = 17
        LLAMA_FTYPE_MOSTLY_Q6_K          = 18
        LLAMA_FTYPE_MOSTLY_IQ2_XXS       = 19
        LLAMA_FTYPE_MOSTLY_IQ2_XS        = 20
        LLAMA_FTYPE_MOSTLY_Q2_K_S        = 21
        LLAMA_FTYPE_MOSTLY_IQ3_XS        = 22
        LLAMA_FTYPE_MOSTLY_IQ3_XXS       = 23
        LLAMA_FTYPE_MOSTLY_IQ1_S         = 24
        LLAMA_FTYPE_MOSTLY_IQ4_NL        = 25
        LLAMA_FTYPE_MOSTLY_IQ3_S         = 26
        LLAMA_FTYPE_MOSTLY_IQ3_M         = 27
        LLAMA_FTYPE_MOSTLY_IQ2_S         = 28
        LLAMA_FTYPE_MOSTLY_IQ2_M         = 29
        LLAMA_FTYPE_MOSTLY_IQ4_XS        = 30
        LLAMA_FTYPE_MOSTLY_IQ1_M         = 31
        LLAMA_FTYPE_MOSTLY_BF16          = 32
        # LLAMA_FTYPE_MOSTLY_Q4_0_4_4      = 33, # removed from gguf files, use Q4_0 and runtime repack
        # LLAMA_FTYPE_MOSTLY_Q4_0_4_8      = 34, # removed from gguf files, use Q4_0 and runtime repack
        # LLAMA_FTYPE_MOSTLY_Q4_0_8_8      = 35, # removed from gguf files, use Q4_0 and runtime repack
        LLAMA_FTYPE_MOSTLY_TQ1_0         = 36   # except 1d tensors
        LLAMA_FTYPE_MOSTLY_TQ2_0         = 37   # except 1d tensors
        LLAMA_FTYPE_MOSTLY_MXFP4_MOE     = 38   # except 1d tensors
        LLAMA_FTYPE_GUESSED              = 1024

    cdef enum llama_rope_scaling_type:
        LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1
        LLAMA_ROPE_SCALING_TYPE_NONE        = 0
        LLAMA_ROPE_SCALING_TYPE_LINEAR      = 1
        LLAMA_ROPE_SCALING_TYPE_YARN        = 2
        LLAMA_ROPE_SCALING_TYPE_LONGROPE    = 3
        LLAMA_ROPE_SCALING_TYPE_MAX_VALUE   = LLAMA_ROPE_SCALING_TYPE_LONGROPE

    cdef enum llama_pooling_type:
        LLAMA_POOLING_TYPE_UNSPECIFIED = -1
        LLAMA_POOLING_TYPE_NONE = 0
        LLAMA_POOLING_TYPE_MEAN = 1
        LLAMA_POOLING_TYPE_CLS  = 2
        LLAMA_POOLING_TYPE_LAST = 3
        LLAMA_POOLING_TYPE_RANK = 4 # used by reranking models to attach the classification head to the graph

    cdef enum llama_attention_type:
        LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1
        LLAMA_ATTENTION_TYPE_CAUSAL      = 0
        LLAMA_ATTENTION_TYPE_NON_CAUSAL  = 1

    cdef enum llama_flash_attn_type:
        LLAMA_FLASH_ATTN_TYPE_AUTO     = -1
        LLAMA_FLASH_ATTN_TYPE_DISABLED = 0
        LLAMA_FLASH_ATTN_TYPE_ENABLED  = 1

    cdef const char * llama_flash_attn_type_name(llama_flash_attn_type flash_attn_type)

    cdef enum llama_split_mode:
        LLAMA_SPLIT_MODE_NONE  = 0 # single GPU
        LLAMA_SPLIT_MODE_LAYER = 1 # split layers and KV across GPUs
        LLAMA_SPLIT_MODE_ROW   = 2 # split layers and KV across GPUs, use tensor parallelism if supported

    ctypedef struct llama_token_data:
        llama_token id  # token id
        float logit     # log-odds of the token
        float p         # probability of the token

    ctypedef struct llama_token_data_array:
        # NOTE: this pointer can be modified by the samplers
        llama_token_data * data
        size_t size
        int64_t selected  # this is the index in the data array (i.e. not the token id)
        bint sorted       # note: do not assume the data is sorted - always check this flag

    ctypedef bint (*llama_progress_callback)(float progress, void * user_data)

    # Input data for llama_encode/llama_decode
    # A llama_batch object can contain input about one or many sequences
    # The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
    #
    # - token  : the token ids of the input (used when embd is NULL)
    # - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
    # - pos    : the positions of the respective token in the sequence
    #            (if set to NULL, the token position will be tracked automatically by llama_encode/llama_decode)
    # - seq_id : the sequence to which the respective token belongs
    #            (if set to NULL, the sequence ID will be assumed to be 0)
    # - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
    #            (if set to NULL:
    #               - if embeddings: all tokens are output
    #               - if not:        only the last token is output
    #            )
    #
    ctypedef struct llama_batch:
        int32_t n_tokens

        llama_token  *  token
        float        *  embd
        llama_pos    *  pos
        int32_t      *  n_seq_id
        llama_seq_id ** seq_id
        int8_t       *  logits    # TODO: rename this to "output"

    cdef enum llama_model_kv_override_type:
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
       # ggml_backend_buffer_type_t buft # TODO

    ctypedef struct llama_model_params:
        ggml.ggml_backend_dev_t * devices   # NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
        # NULL-terminated list of buffer types to use for tensors that match a pattern
        const llama_model_tensor_buft_override * tensor_buft_overrides;
        int32_t n_gpu_layers           # number of layers to store in VRAM
        llama_split_mode split_mode    # how to split the model across multiple GPUs
        int32_t main_gpu               # the GPU that is used for the entire model when split_mode is LLAMA_SPLIT_MODE_NONE
        const float * tensor_split     # proportion of the model (layers or rows) to offload to each GPU, size: llama_max_devices()
        # Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
        # If the provided progress_callback returns true, model loading continues.
        # If it returns false, model loading is immediately aborted.
        llama_progress_callback progress_callback
        void * progress_callback_user_data
        const llama_model_kv_override * kv_overrides
        # Keep the booleans together to avoid misalignment during copy-by-value.
        bint vocab_only       # only load the vocabulary, no weights
        bint use_mmap         # use mmap if possible
        bint use_mlock        # force system to keep model in RAM
        bint check_tensors    # validate model tensor data
        bint use_extra_bufts  # use extra buffer types (used for weight repacking)

    ctypedef struct llama_context_params:
        uint32_t n_ctx             # text context, 0 = from model
        uint32_t n_batch           # logical maximum batch size that can be submitted to llama_decode
        uint32_t n_ubatch          # physical maximum batch size
        uint32_t n_seq_max         # max number of sequences (i.e. distinct states for recurrent models)
        uint32_t n_threads         # number of threads to use for generation
        uint32_t n_threads_batch   # number of threads to use for batch processing

        llama_rope_scaling_type rope_scaling_type # RoPE scaling type
        llama_pooling_type      pooling_type      # whether to pool (sum) embedding results by sequence id
        llama_attention_type    attention_type    # attention type to use for embeddings
        llama_flash_attn_type   flash_attn_type   # when to enable Flash Attention

        float    rope_freq_base   # RoPE base frequency, 0 = from model
        float    rope_freq_scale  # RoPE frequency scaling factor, 0 = from model
        float    yarn_ext_factor  # YaRN extrapolation mix factor, negative = from model
        float    yarn_attn_factor # YaRN magnitude scaling factor
        float    yarn_beta_fast   # YaRN low correction dim
        float    yarn_beta_slow   # YaRN high correction dim
        uint32_t yarn_orig_ctx    # YaRN original context size

        ggml.ggml_backend_sched_eval_callback cb_eval
        void * cb_eval_user_data

        ggml.ggml_type type_k # data type for K cache [EXPERIMENTAL]
        ggml.ggml_type type_v # data type for V cache [EXPERIMENTAL]

        # Abort callback
        # if it returns true, execution of llama_decode() will be aborted
        # currently works only with CPU execution
        ggml.ggml_abort_callback abort_callback
        void * abort_callback_data

        # Keep the booleans together and at the end of the struct to avoid misalignment during copy-by-value.
        bint offload_kqv  # offload the KQV ops (including the KV cache) to GPU
        bint no_perf      # measure performance timings
        bint op_offload   # offload host tensor operations to device
        bint swa_full     # use full-size SWA cache (https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
                          # NOTE: setting to false when n_seq_max > 1 can cause bad performance in some cases
                          #       ref: https://github.com/ggml-org/llama.cpp/pull/13845#issuecomment-2924800573
        bint kv_unified   # use a unified buffer across the input sequences when computing the attention
                          # try to disable when n_seq_max > 1 for improved performance when the sequences do not share a large prefix
                          # ref: https://github.com/ggml-org/llama.cpp/pull/14363

    ctypedef struct llama_model_quantize_params:
        int32_t nthread                     # number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
        llama_ftype ftype                   # quantize to this llama_ftype
        ggml.ggml_type output_tensor_type        # output tensor type
        ggml.ggml_type token_embedding_type      # itoken embeddings tensor type
        bint allow_requantize               # allow quantizing non-f32/f16 tensors
        bint quantize_output_tensor         # quantize output.weight
        bint only_copy                      # only copy tensors - ftype, allow_requantize and quantize_output_tensor are ignored
        bint pure                           # quantize all tensors to the default type
        bint keep_split                     # quantize to the same number of shards
        void * imatrix                      # pointer to importance matrix data
        void * kv_overrides                 # pointer to vector containing overrides
        void * tensor_types                 # pointer to vector containing tensor types
        void * prune_layers                 # pointer to vector containing layer indices to prune

    ctypedef struct llama_logit_bias:
        llama_token token
        float bias

    ctypedef struct llama_sampler_chain_params:
        bint no_perf # whether to measure performance timings

    ctypedef struct llama_chat_message:
        const char * role
        const char * content

    ctypedef struct llama_adapter_lora: pass


    # -------------------------------------------------------------------------
    # llama.h functions

    # TODO: update API to start accepting pointers to params structs (https://github.com/ggerganov/llama.cpp/discussions/9172)
    cdef llama_model_params llama_model_default_params()
    cdef llama_context_params llama_context_default_params()
    cdef llama_sampler_chain_params  llama_sampler_chain_default_params()
    cdef llama_model_quantize_params llama_model_quantize_default_params()


    # Initialize the llama + ggml backend
    # If numa is true, use NUMA optimizations
    # Call once at the start of the program
    cdef void llama_backend_init()

    # Call once at the end of the program - currently only used for MPI
    cdef void llama_backend_free()

    #optional:
    cdef void llama_numa_init(ggml.ggml_numa_strategy numa)

    # Optional: an auto threadpool gets created in ggml if not passed explicitly
    cdef void llama_attach_threadpool(llama_context * ctx, ggml.ggml_threadpool_t threadpool, ggml.ggml_threadpool_t threadpool_batch)
    cdef void llama_detach_threadpool(llama_context * ctx)

    # Load the model from a file
    # If the file is split into multiple parts, the file name must follow this pattern: <name>-%05d-of-%05d.gguf
    # If the split file name does not follow this pattern, use llama_model_load_from_splits
    cdef llama_model * llama_model_load_from_file(
            const char * path_model,
            llama_model_params params)

    # Load the model from multiple splits (support custom naming scheme)
    # The paths must be in the correct order
    cdef llama_model * llama_model_load_from_splits(
                             const char ** paths,
                                 size_t    n_paths,
                     llama_model_params    params)

    cdef void llama_model_save_to_file(
            const llama_model * model,
                const char * path_model);

    cdef void llama_model_free(llama_model * model)

    cdef llama_context * llama_init_from_model(
                     llama_model * model,
            llama_context_params   params)

    # Frees all allocated memory
    cdef void llama_free(llama_context * ctx)

    cdef int64_t llama_time_us()

    cdef size_t llama_max_devices()
    cdef size_t llama_max_parallel_sequences()

    cdef bint llama_supports_mmap       ()
    cdef bint llama_supports_mlock      ()
    cdef bint llama_supports_gpu_offload()
    cdef bint llama_supports_rpc()

    cdef uint32_t llama_n_ctx      (const llama_context * ctx)
    cdef uint32_t llama_n_batch    (const llama_context * ctx)
    cdef uint32_t llama_n_ubatch   (const llama_context * ctx)
    cdef uint32_t llama_n_seq_max  (const llama_context * ctx)

    cdef const llama_model * llama_get_model(const llama_context * ctx)
    cdef llama_memory_t   llama_get_memory  (const llama_context * ctx)
    # cdef llama_pooling_type llama_pooling_type(const llama_context * ctx) # TODO: rename to llama_get_pooling_type

    cdef llama_pooling_type  llama_get_pooling_type "llama_pooling_type" (const llama_context * ctx)
    cdef const llama_vocab * llama_model_get_vocab(const llama_model * model)
    cdef llama_rope_type     llama_get_model_rope_type "llama_model_rope_type" (const llama_model * model)

    cdef int32_t llama_model_n_ctx_train(const llama_model * model)
    cdef int32_t llama_model_n_embd     (const llama_model * model)
    cdef int32_t llama_model_n_layer    (const llama_model * model)
    cdef int32_t llama_model_n_head     (const llama_model * model)
    cdef int32_t llama_model_n_head_kv  (const llama_model * model)
    cdef int32_t llama_model_n_swa      (const llama_model * model)

    # Get the model's RoPE frequency scaling factor
    cdef float llama_model_rope_freq_scale_train(const llama_model * model)

    # Returns the number of classifier outputs (only valid for classifier models)
    # Undefined behavior for non-classifier models
    cdef uint32_t llama_model_n_cls_out(const llama_model * model)

    # Returns label of classifier output by index (<n_cls_out). Returns nullptr if no label provided
    cdef const char * llama_model_cls_label(const llama_model * model, uint32_t i)

    cdef llama_vocab_type llama_get_vocab_type "llama_vocab_type" (const llama_vocab * vocab)

    cdef int32_t llama_vocab_n_tokens(const llama_vocab * vocab)

    # Functions to access the model's GGUF metadata scalar values
    # - The functions return the length of the string on success, or -1 on failure
    # - The output string is always null-terminated and cleared on failure
    # - When retrieving a string, an extra byte must be allocated to account for the null terminator
    # - GGUF array values are not supported by these functions

    # Get metadata value as a string by key name
    cdef int32_t llama_model_meta_val_str(const llama_model * model, const char * key, char * buf, size_t buf_size)

    # Get the number of metadata key/value pairs
    cdef int32_t llama_model_meta_count(const llama_model * model)

    # Get metadata key name by index
    cdef int32_t llama_model_meta_key_by_index(const llama_model * model, int32_t i, char * buf, size_t buf_size)

    # Get metadata value as a string by index
    cdef int32_t llama_model_meta_val_str_by_index(const llama_model * model, int32_t i, char * buf, size_t buf_size)

    # Get a string describing the model type
    cdef int32_t llama_model_desc(const llama_model * model, char * buf, size_t buf_size)

    # Returns the total size of all the tensors in the model in bytes
    cdef uint64_t llama_model_size(const llama_model * model)

    # Get the default chat template. Returns nullptr if not available
    # If name is NULL, returns the default chat template
    cdef const char * llama_model_chat_template(const llama_model * model, const char * name)

    # Returns the total number of parameters in the model
    cdef uint64_t llama_model_n_params(const llama_model * model)

    # Returns true if the model contains an encoder that requires llama_encode() call
    cdef bint llama_model_has_encoder(const llama_model * model)

    # Returns true if the model contains a decoder that requires llama_decode() call
    cdef bint llama_model_has_decoder(const llama_model * model)

    # For encoder-decoder models, this function returns id of the token that must be provided
    # to the decoder to start generating output sequence. For other models, it returns -1.
    cdef llama_token llama_model_decoder_start_token(const llama_model * model)

    # Returns true if the model is recurrent (like Mamba, RWKV, etc.)
    cdef bint llama_model_is_recurrent(const llama_model * model)

    # Returns true if the model is diffusion-based (like LLaDA, Dream, etc.)
    cdef bint llama_model_is_diffusion(const llama_model * model)

    # Returns 0 on success
    cdef uint32_t llama_model_quantize(
            const char * fname_inp,
            const char * fname_out,
            const llama_model_quantize_params * params)

    #
    # Adapters
    #

    # Load a LoRA adapter from file
    cdef llama_adapter_lora * llama_adapter_lora_init(llama_model * model, const char * path_lora)

    # Functions to access the adapter's GGUF metadata scalar values
    # - The functions return the length of the string on success, or -1 on failure
    # - The output string is always null-terminated and cleared on failure
    # - When retrieving a string, an extra byte must be allocated to account for the null terminator
    # - GGUF array values are not supported by these functions

    # Get metadata value as a string by key name
    cdef int32_t llama_adapter_meta_val_str(const llama_adapter_lora * adapter, const char * key, char * buf, size_t buf_size)

    # Get the number of metadata key/value pairs
    cdef int32_t llama_adapter_meta_count(const llama_adapter_lora * adapter)

    # Get metadata key name by index
    cdef int32_t llama_adapter_meta_key_by_index(const llama_adapter_lora * adapter, int32_t i, char * buf, size_t buf_size)

    # Get metadata value as a string by index
    cdef int32_t llama_adapter_meta_val_str_by_index(const llama_adapter_lora * adapter, int32_t i, char * buf, size_t buf_size)

    # Manually free a LoRA adapter
    # Note: loaded adapters will be free when the associated model is deleted
    cdef void llama_adapter_lora_free(llama_adapter_lora * adapter)

    # Get the invocation tokens if the current lora is an alora
    cdef uint64_t            llama_adapter_get_alora_n_invocation_tokens(const llama_adapter_lora * adapter)
    cdef const llama_token * llama_adapter_get_alora_invocation_tokens  (const llama_adapter_lora * adapter)


    # The following functions operate on a llama_context, hence the naming: llama_verb_...

    # Add a loaded LoRA adapter to given context
    # This will not modify model's weight
    cdef int32_t llama_set_adapter_lora(
            llama_context * ctx,
            llama_adapter_lora * adapter,
            float scale)

    # Remove a specific LoRA adapter from given context
    # Return -1 if the adapter is not present in the context
    cdef int32_t llama_rm_adapter_lora(
            llama_context * ctx,
            llama_adapter_lora * adapter)

    # Remove all LoRA adapters from given context
    cdef void llama_clear_adapter_lora(llama_context * ctx)

    # Apply a loaded control vector to a llama_context, or if data is NULL, clear
    # the currently loaded vector.
    # n_embd should be the size of a single layer's control, and data should point
    # to an n_embd x n_layers buffer starting from layer 1.
    # il_start and il_end are the layer range the vector should apply to (both inclusive)
    # See llama_control_vector_load in common to load a control vector.
    cdef int32_t llama_apply_adapter_cvec(
            llama_context * ctx,
                     const float * data,
                          size_t   len,
                         int32_t   n_embd,
                         int32_t   il_start,
                         int32_t   il_end)


    # -------------------------------------------------------------------------
    # Memory

    #
    # Memory
    #

    # Clear the memory contents
    # If data == true, the data buffers will also be cleared together with the metadata
    cdef void llama_memory_clear(
            llama_memory_t mem,
                      bint data)

    # Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
    # Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
    # seq_id < 0 : match any sequence
    # p0 < 0     : [0,  p1]
    # p1 < 0     : [p0, inf)
    cdef bint llama_memory_seq_rm(
            llama_memory_t mem,
              llama_seq_id seq_id,
                 llama_pos p0,
                 llama_pos p1)

    # Copy all tokens that belong to the specified sequence to another sequence
    # p0 < 0 : [0,  p1]
    # p1 < 0 : [p0, inf)
    cdef void llama_memory_seq_cp(
            llama_memory_t mem,
              llama_seq_id seq_id_src,
              llama_seq_id seq_id_dst,
                 llama_pos p0,
                 llama_pos p1)

    # Removes all tokens that do not belong to the specified sequence
    cdef void llama_memory_seq_keep(
            llama_memory_t mem,
              llama_seq_id seq_id)

    # Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
    # p0 < 0 : [0,  p1]
    # p1 < 0 : [p0, inf)
    cdef void llama_memory_seq_add(
            llama_memory_t mem,
              llama_seq_id seq_id,
                 llama_pos p0,
                 llama_pos p1,
                 llama_pos delta)

    # Integer division of the positions by factor of `d > 1`
    # p0 < 0 : [0,  p1]
    # p1 < 0 : [p0, inf)
    cdef void llama_memory_seq_div(
            llama_memory_t mem,
              llama_seq_id seq_id,
                 llama_pos p0,
                 llama_pos p1,
                       int d)

    # Returns the smallest position present in the memory for the specified sequence
    # This is typically non-zero only for SWA caches
    # Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
    # Return -1 if the sequence is empty
    cdef llama_pos llama_memory_seq_pos_min(
            llama_memory_t mem,
              llama_seq_id seq_id)

    # Returns the largest position present in the memory for the specified sequence
    # Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
    # Return -1 if the sequence is empty
    cdef llama_pos llama_memory_seq_pos_max(
            llama_memory_t mem,
              llama_seq_id seq_id)

    # Check if the memory supports shifting
    cdef bint llama_memory_can_shift(llama_memory_t mem)

    # -------------------------------------------------------------------------
    # State / sessions

    # Returns the *actual* size in bytes of the state
    # (logits, embedding and memory)
    # Only use when saving the state, not when restoring it, otherwise the size may be too small.
    cdef size_t llama_state_get_size( llama_context * ctx)

    # Copies the state to the specified destination address.
    # Destination needs to have allocated enough memory.
    # Returns the number of bytes copied
    cdef size_t llama_state_get_data(
             llama_context * ctx,
                         uint8_t * dst,
                          size_t   size)

    # Set the state reading from the specified address
    # Returns the number of bytes read
    cdef size_t llama_state_set_data(
             llama_context * ctx,
                   const uint8_t * src,
                          size_t   size)

    # Save/load session file
    cdef bint llama_state_load_file(
             llama_context * ctx,
                      const char * path_session,
                     llama_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out)

    cdef bint llama_state_save_file(
             llama_context * ctx,
                      const char * path_session,
               const llama_token * tokens,
                          size_t   n_token_count)

    # Get the exact size needed to copy the state of a single sequence
    cdef size_t llama_state_seq_get_size(
             llama_context * ctx,
                    llama_seq_id   seq_id)

    # Copy the state of a single sequence into the specified buffer
    cdef size_t llama_state_seq_get_data(
             llama_context * ctx,
                         uint8_t * dst,
                          size_t   size,
                    llama_seq_id   seq_id)

    # Copy the sequence data (originally copied with `llama_state_seq_get_data`) into the specified sequence
    # Returns:
    #  - Positive: Ok
    #  - Zero: Failed to load
    cdef size_t llama_state_seq_set_data(
             llama_context * ctx,
                   const uint8_t * src,
                          size_t   size,
                    llama_seq_id   dest_seq_id)

    cdef size_t llama_state_seq_save_file(
             llama_context * ctx,
                      const char * filepath,
                    llama_seq_id   seq_id,
               const llama_token * tokens,
                          size_t   n_token_count)

    cdef size_t llama_state_seq_load_file(
             llama_context * ctx,
                      const char * filepath,
                    llama_seq_id   dest_seq_id,
                     llama_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out)

    #define LLAMA_STATE_SEQ_FLAGS_SWA_ONLY 1

    ctypedef uint32_t llama_state_seq_flags

    cdef size_t llama_state_seq_get_size_ext(
                   llama_context * ctx,
                    llama_seq_id   seq_id,
           llama_state_seq_flags   flags)

    cdef size_t llama_state_seq_get_data_ext(
                   llama_context * ctx,
                         uint8_t * dst,
                          size_t   size,
                    llama_seq_id   seq_id,
           llama_state_seq_flags   flags)

    cdef size_t llama_state_seq_set_data_ext(
                   llama_context * ctx,
                   const uint8_t * src,
                          size_t   size,
                    llama_seq_id   dest_seq_id,
           llama_state_seq_flags   flags)

    # -------------------------------------------------------------------------
    # Decoding

    # Return batch for single sequence of tokens
    # The sequence ID will be fixed to 0
    # The position of the tokens will be tracked automatically by llama_decode
    #
    # NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
    #
    cdef llama_batch llama_batch_get_one(llama_token * tokens, int32_t n_tokens)


    # Allocates a batch of tokens on the heap that can hold a maximum of n_tokens
    # Each token can be assigned up to n_seq_max sequence ids
    # The batch has to be freed with llama_batch_free()
    # If embd != 0, llama_batch.embd will be allocated with size of n_tokens * embd * sizeof(float)
    # Otherwise, llama_batch.token will be allocated to store n_tokens llama_token
    # The rest of the llama_batch members are allocated with size n_tokens
    # All members are left uninitialized
    cdef llama_batch llama_batch_init(
            int32_t n_tokens,
            int32_t embd,
            int32_t n_seq_max)

    # Frees a batch of tokens allocated with llama_batch_init()
    cdef void llama_batch_free( llama_batch batch)

    # Processes a batch of tokens with the ecoder part of the encoder-decoder model.
    # Stores the encoder output internally for later use by the decoder cross-attention layers.
    #   0 - success
    # < 0 - error. the memory state is restored to the state before this call
    cdef int32_t llama_encode(llama_context * ctx, llama_batch batch)

    # Process a batch of tokens.
    # Requires the context to have a memory.
    # For encode-decoder contexts, processes the batch using the decoder.
    # Positive return values does not mean a fatal error, but rather a warning.
    # Upon fatal-error or abort, the ubatches that managed to be been processed will remain in the memory state of the context
    #   To handle this correctly, query the memory state using llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
    # Upon other return values, the memory state is restored to the state before this call
    #    0 - success
    #    1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
    #    2 - aborted     (processed ubatches will remain in the context's memory)
    #   -1 - invalid input batch
    # < -1 - fatal error (processed ubatches will remain in the context's memory)
    cdef int32_t llama_decode(llama_context * ctx, llama_batch batch)

    # Set the number of threads used for decoding
    # n_threads is the number of threads used for generation (single token)
    # n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
    cdef void llama_set_n_threads( llama_context * ctx, uint32_t n_threads, uint32_t n_threads_batch)

    # Get the number of threads used for generation of a single token.
    cdef uint32_t llama_n_threads( llama_context * ctx)

    # Get the number of threads used for prompt and batch processing (multiple token).
    cdef uint32_t llama_n_threads_batch( llama_context * ctx)

    # Set whether the context outputs embeddings or not
    # TODO: rename to avoid confusion with llama_get_embeddings()
    cdef void llama_set_embeddings( llama_context * ctx, bint embeddings)

    # Set whether to use causal attention or not
    # If set to true, the model will only attend to the past tokens
    cdef void llama_set_causal_attn( llama_context * ctx, bint causal_attn)

    # Set abort callback
    cdef void llama_set_abort_callback( llama_context * ctx, ggml.ggml_abort_callback abort_callback, void * abort_callback_data)

    # Wait until all computations are finished
    # This is automatically done when using one of the functions below to obtain the computation results
    # and is not necessary to call it explicitly in most cases
    cdef void llama_synchronize( llama_context * ctx)

    # Token logits obtained from the last call to llama_decode()
    # The logits for which llama_batch.logits[i] != 0 are stored contiguously
    # in the order they have appeared in the batch.
    # Rows: number of tokens for which llama_batch.logits[i] != 0
    # Cols: n_vocab
    # TODO: deprecate in favor of llama_get_logits_ith() (ref: https://github.com/ggml-org/llama.cpp/pull/14853#issuecomment-3113143522)
    cdef float * llama_get_logits( llama_context * ctx)

    # FIXME: should this be added
    # cdef int32_t llama_n_outputs( llama_context * ctx)

    # Logits for the ith token. For positive indices, Equivalent to:
    # llama_get_logits(ctx) + ctx->output_ids[i]*n_vocab
    # Negative indicies can be used to access logits in reverse order, -1 is the last logit.
    # returns NULL for invalid ids.
    cdef float * llama_get_logits_ith( llama_context * ctx, int32_t i)

    # Get all output token embeddings.
    # when pooling_type == LLAMA_POOLING_TYPE_NONE or when using a generative model,
    # the embeddings for which llama_batch.logits[i] != 0 are stored contiguously
    # in the order they have appeared in the batch.
    # shape: [n_outputs*n_embd]
    # Otherwise, returns NULL.
    # TODO: deprecate in favor of llama_get_embeddings_ith() (ref: https://github.com/ggml-org/llama.cpp/pull/14853#issuecomment-3113143522)
    cdef float * llama_get_embeddings( llama_context * ctx)

    # Get the embeddings for the ith token. For positive indices, Equivalent to:
    # llama_get_embeddings(ctx) + ctx->output_ids[i]*n_embd
    # Negative indicies can be used to access embeddings in reverse order, -1 is the last embedding.
    # returns NULL for invalid ids.
    cdef float * llama_get_embeddings_ith( llama_context * ctx, int32_t i)

    # Get the embeddings for a sequence id
    # Returns NULL if pooling_type is LLAMA_POOLING_TYPE_NONE
    # when pooling_type == LLAMA_POOLING_TYPE_RANK, returns float[n_cls_out] with the rank(s) of the sequence
    # otherwise: float[n_embd] (1-dimensional)
    cdef float * llama_get_embeddings_seq( llama_context * ctx, llama_seq_id seq_id)


    # -------------------------------------------------------------------------
    # Vocab

    cdef const char * llama_vocab_get_text(const llama_vocab * vocab, llama_token token)

    cdef float llama_vocab_get_score(const llama_vocab * vocab, llama_token token)

    cdef llama_token_attr llama_vocab_get_attr(const llama_vocab * model, llama_token token)

    # Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)
    cdef bint llama_vocab_is_eog(const llama_vocab * vocab, llama_token token)

    # Identify if Token Id is a control token or a render-able token
    cdef bint llama_vocab_is_control(const llama_vocab * vocab, llama_token token)

    # Special tokens
    cdef llama_token llama_vocab_bos(const llama_vocab * vocab)  # beginning-of-sentence
    cdef llama_token llama_vocab_eos(const llama_vocab * vocab)  # end-of-sentence
    cdef llama_token llama_vocab_eot(const llama_vocab * vocab)  # end-of-turn
    cdef llama_token llama_vocab_cls(const llama_vocab * vocab)  # classification
    cdef llama_token llama_vocab_sep(const llama_vocab * vocab)  # sentence separator
    cdef llama_token llama_vocab_nl (const llama_vocab * vocab)  # next-line
    cdef llama_token llama_vocab_pad(const llama_vocab * vocab)  # padding
    cdef llama_token llama_vocab_mask(const llama_vocab * vocab) # mask

    cdef bint llama_vocab_get_add_bos(const llama_vocab * vocab)
    cdef bint llama_vocab_get_add_eos(const llama_vocab * vocab)
    cdef bint llama_vocab_get_add_sep(const llama_vocab * vocab)

    cdef llama_token llama_vocab_fim_pre(const llama_vocab * vocab)
    cdef llama_token llama_vocab_fim_suf(const llama_vocab * vocab)
    cdef llama_token llama_vocab_fim_mid(const llama_vocab * vocab)
    cdef llama_token llama_vocab_fim_pad(const llama_vocab * vocab)
    cdef llama_token llama_vocab_fim_rep(const llama_vocab * vocab)
    cdef llama_token llama_vocab_fim_sep(const llama_vocab * vocab)

    # -------------------------------------------------------------------------
    # Tokenization (The API is thread-safe)

    # @details Convert the provided text into tokens.
    # @param tokens The tokens pointer must be large enough to hold the resulting tokens.
    # @return Returns the number of tokens on success, no more than n_tokens_max
    # @return Returns a negative number on failure - the number of tokens that would have been returned
    # @return Returns INT32_MIN on overflow (e.g., tokenization result size exceeds int32_t limit)
    # @param add_special Allow to add BOS and EOS tokens if model is configured to do so.
    # @param parse_special Allow tokenizing special and/or control tokens which otherwise are not exposed and treated
    #                      as plaintext. Does not insert a leading space.
    cdef int32_t llama_tokenize(
        const  llama_vocab * vocab,
                      const char * text,
                         int32_t   text_len,
                     llama_token * tokens,
                         int32_t   n_tokens_max,
                            bint   add_special,
                            bint   parse_special)

    # Token Id -> Piece.
    # Uses the vocabulary in the provided context.
    # Does not write null terminator to the buffer.
    # User can skip up to 'lstrip' leading spaces before copying (useful when encoding/decoding multiple tokens with 'add_space_prefix')
    # @param special If true, special tokens are rendered in the output.
    cdef int32_t llama_token_to_piece(
              const  llama_vocab * vocab,
                           llama_token   token,
                                  char * buf,
                               int32_t   length,
                               int32_t   lstrip,
                                  bint   special)


    # @details Convert the provided tokens into text (inverse of llama_tokenize()).
    # @param text The char pointer must be large enough to hold the resulting text.
    # @return Returns the number of chars/bytes on success, no more than text_len_max.
    # @return Returns a negative number on failure - the number of chars/bytes that would have been returned.
    # @param remove_special Allow to remove BOS and EOS tokens if model is configured to do so.
    # @param unparse_special If true, special tokens are rendered in the output.
    cdef int32_t llama_detokenize(
        const  llama_vocab * vocab,
               const llama_token * tokens,
                         int32_t   n_tokens,
                            char * text,
                         int32_t   text_len_max,
                            bint   remove_special,
                            bint   unparse_special)


    # -------------------------------------------------------------------------
    # Chat templates
    #

    # Apply chat template. Inspired by hf apply_chat_template() on python.
    # Both "model" and "custom_template" are optional, but at least one is required. "custom_template" has higher precedence than "model"
    # NOTE: This function does not use a jinja parser. It only support a pre-defined list of template. See more: https:#github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
    # @param tmpl A Jinja template to use for this chat. If this is nullptr, the model’s default chat template will be used instead.
    # @param chat Pointer to a list of multiple llama_chat_message
    # @param n_msg Number of llama_chat_message in this chat
    # @param add_ass Whether to end the prompt with the token(s) that indicate the start of an assistant message.
    # @param buf A buffer to hold the output formatted prompt. The recommended alloc size is 2 * (total number of characters of all messages)
    # @param length The size of the allocated buffer
    # @return The total number of bytes of the formatted prompt. If is it larger than the size of buffer, you may need to re-alloc it and then re-apply the template.
    cdef int32_t llama_chat_apply_template(
                            const char * tmpl,
             const  llama_chat_message * chat,
                                size_t   n_msg,
                                  bint   add_ass,
                                  char * buf,
                               int32_t   length)

    # Get list of built-in chat templates
    cdef int32_t llama_chat_builtin_templates(const char ** output, size_t len)

    # -------------------------------------------------------------------------
    # Sampling API

    # Sample usage:
    #
    #    # prepare the sampling chain at the start
    #    auto sparams = llama_sampler_chain_default_params();
    #
    #    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    #
    #    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(50));
    #    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9, 1));
    #    llama_sampler_chain_add(smpl, llama_sampler_init_temp (0.8));
    #
    #    # typically, the chain should end with a sampler such as "greedy", "dist" or "mirostat"
    #    # this sampler will be responsible to select the actual token
    #    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    #
    #    ...
    #
    #    # decoding loop:
    #    while (...) {
    #        ...
    #
    #        llama_decode(ctx, batch);
    #
    #        # sample from the logits of the last token in the batch
    #        const llama_token id = llama_sampler_sample(smpl, ctx, -1);
    #
    #        # accepting the token updates the internal state of certain samplers (e.g. grammar, repetition, etc.)
    #        llama_sampler_accept(smpl, id);
    #        ...
    #    }
    #
    #    llama_sampler_free(smpl);
    #
    # TODO: In the future, llama_sampler will be utilized to offload the sampling to the backends (e.g. GPU).

    ctypedef void * llama_sampler_context_t

    # user code can implement the interface below in order to create custom llama_sampler
    ctypedef struct llama_sampler_i:
        const char *           (*name)  (const llama_sampler * smpl)                                 # can be NULL
        void                   (*accept)(      llama_sampler * smpl, llama_token token)              # can be NULL
        void                   (*apply) (      llama_sampler * smpl, llama_token_data_array * cur_p) # required
        void                   (*reset) (      llama_sampler * smpl)                                 # can be NULL
        llama_sampler *        (*clone) (const llama_sampler * smpl)                                 # can be NULL if ctx is NULL
        void                   (*free)  (      llama_sampler * smpl)

    ctypedef struct llama_sampler:
        const llama_sampler_i * iface
        llama_sampler_context_t ctx

    # mirror of llama_sampler_i:
    cdef llama_sampler *        llama_sampler_init  (const llama_sampler_i * iface, llama_sampler_context_t ctx)
    cdef const char *           llama_sampler_name  (const llama_sampler * smpl)
    cdef void                   llama_sampler_accept(      llama_sampler * smpl, llama_token token)
    cdef void                   llama_sampler_apply (      llama_sampler * smpl, llama_token_data_array * cur_p)
    cdef void                   llama_sampler_reset (      llama_sampler * smpl)
    cdef llama_sampler *        llama_sampler_clone (const llama_sampler * smpl)
    # important: do not free if the sampler has been added to a llama_sampler_chain (via llama_sampler_chain_add)
    cdef void                   llama_sampler_free  (      llama_sampler * smpl)

    # llama_sampler_chain
    # a type of llama_sampler that can chain multiple samplers one after another

    cdef llama_sampler * llama_sampler_chain_init(llama_sampler_chain_params params)

    # important: takes ownership of the sampler object and will free it when llama_sampler_free is called
    cdef void                   llama_sampler_chain_add(       llama_sampler * chain, llama_sampler * smpl)
    cdef llama_sampler *        llama_sampler_chain_get(const  llama_sampler * chain, int32_t i)
    cdef int                    llama_sampler_chain_n  (const  llama_sampler * chain)

    # after removing a sampler, the chain will no longer own it, and it will not be freed when the chain is freed
    cdef llama_sampler * llama_sampler_chain_remove( llama_sampler * chain, int32_t i)

    # available samplers:

    cdef llama_sampler * llama_sampler_init_greedy()
    cdef llama_sampler * llama_sampler_init_dist(uint32_t seed)

    # @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https:#arxiv.org/abs/1904.09751
    cdef llama_sampler * llama_sampler_init_top_k(int32_t k)

    # @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https:#arxiv.org/abs/1904.09751
    cdef llama_sampler * llama_sampler_init_top_p (float p, size_t min_keep)

    # @details Minimum P sampling as described in https:#github.com/ggerganov/llama.cpp/pull/3841
    cdef llama_sampler * llama_sampler_init_min_p (float p, size_t min_keep)

    # @details Locally Typical Sampling implementation described in the paper https:#arxiv.org/abs/2202.00666.
    cdef llama_sampler * llama_sampler_init_typical (float p, size_t min_keep)

    # @details Updates the logits l_i` = l_i/t. When t <= 0.0f, the maximum logit is kept at it's original value, the rest are set to -inf
    cdef llama_sampler * llama_sampler_init_temp (float t)

    # @details Dynamic temperature implementation described in the paper https:#arxiv.org/abs/2309.02772.
    cdef llama_sampler * llama_sampler_init_temp_ext (float t, float delta, float exponent)

    # @details XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
    cdef llama_sampler * llama_sampler_init_xtc (float p, float t, size_t min_keep, uint32_t seed)

    # @details Top n sigma sampling as described in academic paper "Top-nσ: Not All Logits Are You Need" https://arxiv.org/pdf/2411.07641
    cdef llama_sampler * llama_sampler_init_top_n_sigma(float n)

    # @details Mirostat 1.0 algorithm described in the paper https:#arxiv.org/abs/2007.14966. Uses tokens instead of words.
    # @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    # @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    # @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    # @pam m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
    # @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    cdef llama_sampler * llama_sampler_init_mirostat(
                 int32_t   n_vocab,
                uint32_t   seed,
                   float   tau,
                   float   eta,
                 int32_t   m)

    # @details Mirostat 2.0 algorithm described in the paper https:#arxiv.org/abs/2007.14966. Uses tokens instead of words.
    # @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    # @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    # @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    # @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    cdef llama_sampler * llama_sampler_init_mirostat_v2(
                                uint32_t   seed,
                                   float   tau,
                                   float   eta)

    # @details Intializes a GBNF grammar, see grammars/README.md for details.
    # @param vocab The vocabulary that this grammar will be used with.
    # @param grammar_str The production rules for the grammar, encoded as a string. Returns an empty grammar if empty. Returns NULL if parsing of grammar_str fails.
    # @param grammar_root The name of the start symbol for the grammar.
    cdef llama_sampler * llama_sampler_init_grammar(
                          const llama_vocab * vocab,
                          const char * grammar_str,
                          const char * grammar_root)

    # @details Lazy grammar sampler, introduced in https://github.com/ggml-org/llama.cpp/pull/9639
    # @param trigger_patterns A list of patterns that will trigger the grammar sampler. Pattern will be matched from the start of the generation output, and grammar sampler will be fed content starting from its first match group.
    # @param trigger_tokens A list of tokens that will trigger the grammar sampler. Grammar sampler will be fed content starting from the trigger token included.
    cdef llama_sampler * llama_sampler_init_grammar_lazy_patterns(
               const llama_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                     const char ** trigger_patterns,
                            size_t num_trigger_patterns,
               const llama_token * trigger_tokens,
                            size_t num_trigger_tokens)


    # NOTE: Avoid using on the full vocabulary as searching for repeated tokens can become slow. For example, apply top-k or top-p sampling first.
    cdef llama_sampler * llama_sampler_init_penalties(
                             int32_t   penalty_last_n,   # last n tokens to penalize (0 = disable penalty, -1 = context size)
                               float   penalty_repeat,   # 1.0 = disabled
                               float   penalty_freq,     # 0.0 = disabled
                               float   penalty_present)  # 0.0 = disabled

    # @details DRY sampler, designed by p-e-w, as described in: https://github.com/oobabooga/text-generation-webui/pull/5677, porting Koboldcpp implementation authored by pi6am: https://github.com/LostRuins/koboldcpp/pull/982
    cdef llama_sampler * llama_sampler_init_dry(
                   const llama_vocab *  vocab,
                             int32_t    n_ctx_train,
                               float    dry_multiplier,
                               float    dry_base,
                             int32_t    dry_allowed_length,
                             int32_t    dry_penalty_last_n,
                          const char ** seq_breakers,
                              size_t    num_breakers)

    cdef llama_sampler * llama_sampler_init_logit_bias(
                             int32_t   n_vocab,
                             int32_t   n_logit_bias,
              const llama_logit_bias * logit_bias)

    # this sampler is meant to be used for fill-in-the-middle infilling
    # it's supposed to be used after top_k + top_p sampling
    #
    # 1. if the sum of the EOG probs times the number of candidates is higher than the sum of the other probs -> pick EOG
    # 2. combine probs of tokens that have the same prefix
    #
    # example:
    #
    # - before:
    #   "hel":   0.5
    #   "hell":  0.2
    #   "hello": 0.1
    #   "dummy": 0.1
    #
    # - after:
    #   "hel":   0.8
    #   "dummy": 0.1
    #
    # 3. discard non-EOG tokens with low prob
    # 4. if no tokens are left -> pick EOT
    #
    cdef llama_sampler * llama_sampler_init_infill(const llama_vocab * vocab)


    # Returns the seed used by the sampler if applicable, LLAMA_DEFAULT_SEED otherwise
    cdef uint32_t llama_sampler_get_seed(const llama_sampler * smpl)

    # Sample and accept a token from the idx-th output of the last evaluation
    #
    # Shorthand for:
    #
    #    const auto * logits = llama_get_logits_ith(ctx, idx)
    #    llama_token_data_array cur_p = { ... init from logits ... }
    #    llama_sampler_apply(smpl, &cur_p)
    #    return cur_p.data[cur_p.selected].id
    #
    # At this point, this is mostly a convenience function.

    cdef llama_token llama_sampler_sample(llama_sampler * smpl, llama_context * ctx, int32_t idx)

    # TODO: extend in the future
    # void llama_decode_with_sampler(llama_context * ctx, llama_sampler * smpl, llama_batch batch, ...)

    # -------------------------------------------------------------------------
    # Model split

    # @details Build a split GGUF final path for this chunk.
    #          llama_split_path(split_path, sizeof(split_path), "/models/ggml-model-q4_0", 2, 4) => split_path = "/models/ggml-model-q4_0-00002-of-00004.gguf"
    #  Returns the split_path length.
    cdef int llama_split_path(char * split_path, size_t maxlen, const char * path_prefix, int split_no, int split_count)

    # @details Extract the path prefix from the split_path if and only if the split_no and split_count match.
    #          llama_split_prefix(split_prefix, 64, "/models/ggml-model-q4_0-00002-of-00004.gguf", 2, 4) => split_prefix = "/models/ggml-model-q4_0"
    #  Returns the split_prefix length.
    cdef int llama_split_prefix(char * split_prefix, size_t maxlen, const char * split_path, int split_no, int split_count)

    # Print system information
    cdef const char * llama_print_system_info()

    # Set callback for all future logging events.
    # If this is not called, or NULL is supplied, everything is output on stderr.
    cdef void llama_log_set(ggml.ggml_log_callback log_callback, void * user_data)

    # -------------------------------------------------------------------------
    # Performance utils

    # NOTE: Used by llama.cpp examples/tools, avoid using in third-party apps. Instead, do your own performance measurements.
    #

    cdef struct llama_perf_context_data:
        # ms == milliseconds
        double t_start_ms  # absolute start time
        double t_load_ms   # time needed for loading the model
        double t_p_eval_ms # time needed for processing the prompt
        double t_eval_ms   # time needed for generating tokens

        int32_t n_p_eval   # number of prompt tokens
        int32_t n_eval     # number of generated tokens
        int32_t n_reused   # number of times a ggml compute graph had been reused

    ctypedef struct llama_perf_sampler_data:
        double t_sample_ms # time needed for sampling in ms

        int32_t n_sample   # number of sampled tokens

    cdef llama_perf_context_data llama_perf_context(const llama_context * ctx)
    cdef void llama_perf_context_print(const llama_context * ctx)
    cdef void llama_perf_context_reset(      llama_context * ctx)

    # NOTE: the following work only with samplers constructed via llama_sampler_chain_init
    cdef llama_perf_sampler_data llama_perf_sampler(const llama_sampler * chain)
    cdef void llama_perf_sampler_print(const llama_sampler * chain)
    cdef void llama_perf_sampler_reset(      llama_sampler * chain)

    # print a breakdown of per-device memory use via LLAMA_LOG:
    cdef void llama_memory_breakdown_print(const llama_context * ctx)

    #
    # training
    #

    # function that returns whether or not a given tensor contains trainable parameters
    ctypedef bint (*llama_opt_param_filter)(const ggml.ggml_tensor * tensor, void * userdata)

    # always returns true
    cdef bint llama_opt_param_filter_all(const ggml.ggml_tensor * tensor, void * userdata)

    ctypedef struct llama_opt_params:
        uint32_t n_ctx_train # assumed context size post training, use context size specified in llama_context if 0

        llama_opt_param_filter param_filter # callback for determining which tensors contain trainable parameters
        void * param_filter_ud              # userdata for determining which tensors contain trainable parameters

        ggml.ggml_opt_get_optimizer_params get_opt_pars # callback for calculating optimizer parameters
        void * get_opt_pars_ud                     # userdata for calculating optimizer parameters

        ggml.ggml_opt_optimizer_type optimizer_type

    cdef void llama_opt_init(llama_context * lctx, llama_model * model, llama_opt_params lopt_params)

    cdef void llama_opt_epoch(
            llama_context             * lctx,
            ggml.ggml_opt_dataset_t        dataset,
            ggml.ggml_opt_result_t         result_train,
            ggml.ggml_opt_result_t         result_eval,
            int64_t                   idata_split,
            ggml.ggml_opt_epoch_callback   callback_train,
            ggml.ggml_opt_epoch_callback   callback_eval)



#------------------------------------------------------------------------------
# llama-cpu.h

cdef extern from "llama-cpp.h":

    cdef cppclass llama_model_deleter:
        void operator()(llama_model * model)

    cdef cppclass llama_context_deleter:
        void operator()(llama_context * context)

    cdef cppclass llama_sampler_deleter:
        void operator()(llama_sampler * sampler)

    cdef cppclass llama_adapter_lora_deleter:
        void operator()(llama_adapter_lora * adapter)

    ctypedef std_unique_ptr[llama_model, llama_model_deleter] llama_model_ptr
    ctypedef std_unique_ptr[llama_context, llama_context_deleter] llama_context_ptr
    ctypedef std_unique_ptr[llama_sampler, llama_sampler_deleter] llama_sampler_ptr
    ctypedef std_unique_ptr[llama_adapter_lora, llama_adapter_lora_deleter] llama_adapter_lora_ptr
