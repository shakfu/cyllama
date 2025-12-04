import pytest
# pytest.skip(allow_module_level=True)

import platform

from pytest import approx

PLATFORM = platform.system()
# ARCH = platform.machine()

import cyllama.llama.llama_cpp as cy


def test_default_model_params():
    params = cy.LlamaModelParams()
    # n_gpu_layers defaults to 999 when any GPU backend is enabled
    # (Metal on macOS, CUDA/HIP/Vulkan on Linux/Windows)
    # and 0 when CPU-only
    assert params.n_gpu_layers in (0, 999)
    assert params.split_mode == 1  # LLAMA_SPLIT_MODE_LAYER = 1
    assert params.main_gpu == 0
    assert params.vocab_only == False
    assert params.use_mmap == True
    assert params.use_mlock == False
    assert params.check_tensors == False
    assert params.progress_callback is None


def test_model_params_progress_callback():
    """Test that progress callback can be set and retrieved."""
    params = cy.LlamaModelParams()

    # Initially None
    assert params.progress_callback is None

    # Set a callback
    progress_values = []
    def on_progress(progress: float) -> bool:
        progress_values.append(progress)
        return True  # continue loading

    params.progress_callback = on_progress
    assert params.progress_callback is on_progress

    # Set to None to disable
    params.progress_callback = None
    assert params.progress_callback is None


def test_model_params_progress_callback_with_model(model_path):
    """Test that progress callback is actually called during model loading."""
    progress_values = []

    def on_progress(progress: float) -> bool:
        progress_values.append(progress)
        return True  # continue loading

    params = cy.LlamaModelParams()
    params.progress_callback = on_progress

    # Load model - this should trigger progress callbacks
    model = cy.LlamaModel(model_path, params)

    # Verify callback was called with progress values
    assert len(progress_values) > 0, "Progress callback should have been called"
    assert all(0.0 <= p <= 1.0 for p in progress_values), "Progress values should be between 0 and 1"
    # The final progress should be 1.0 (complete)
    assert progress_values[-1] == 1.0, "Final progress should be 1.0"


def test_model_params_progress_callback_abort(model_path):
    """Test that returning False from progress callback aborts loading."""
    progress_values = []

    def on_progress(progress: float) -> bool:
        progress_values.append(progress)
        # Abort after first callback
        return False

    params = cy.LlamaModelParams()
    params.progress_callback = on_progress

    # Loading should fail due to abort
    with pytest.raises(Exception):
        model = cy.LlamaModel(model_path, params)

    # Callback should have been called at least once
    assert len(progress_values) >= 1, "Progress callback should have been called before abort"

def test_default_context_params():
    params = cy.LlamaContextParams()
    assert params.n_ctx               == 512
    assert params.n_batch             == 2048
    assert params.n_ubatch            == 512
    assert params.n_seq_max           == 1
    assert params.n_threads           == cy.GGML_DEFAULT_N_THREADS
    assert params.n_threads_batch     == cy.GGML_DEFAULT_N_THREADS
    assert params.rope_scaling_type   == -1  # LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1
    assert params.pooling_type        == -1  # LLAMA_POOLING_TYPE_UNSPECIFIED = -1
    assert params.attention_type      == -1  # LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1
    assert params.rope_freq_base      == 0.0
    assert params.rope_freq_scale     == 0.0
    assert params.yarn_ext_factor     == -1.0
    assert params.yarn_attn_factor    == approx(-1.0)
    assert params.yarn_beta_fast      == -1.0
    assert params.yarn_beta_slow      == -1.0
    assert params.yarn_orig_ctx       == 0
    # defrag_thold is not available in LlamaContextParams
    # assert params.defrag_thold        == -1.0
    # assert params.cb_eval             == nullptr
    # assert params.cb_eval_user_data   == nullptr
    assert params.type_k              == 1  # GGML_TYPE_F16 = 1
    assert params.type_v              == 1  # GGML_TYPE_F16 = 1
    # logits_all and embeddings are not available in LlamaContextParams
    # assert params.logits_all          == False
    # assert params.embeddings          == False
    assert params.offload_kqv         == True
    # flash_attn is not available in LlamaContextParams
    # assert params.flash_attn          == False
    assert params.no_perf             == True
    # assert params.abort_callback      == nullptr
    # assert params.abort_callback_data == nullptr

def test_default_model_quantize_params():
    params = cy.LlamaModelQuantizeParams()
    assert params.nthread                     == 0
    assert params.ftype                       == 9  # LLAMA_FTYPE_MOSTLY_Q5_1 = 9
    assert params.output_tensor_type          == 40  # GGML_TYPE_COUNT = 40
    assert params.token_embedding_type        == 40  # GGML_TYPE_COUNT = 40
    assert params.allow_requantize            == False
    assert params.quantize_output_tensor      == True
    assert params.only_copy                   == False
    assert params.pure                        == False
    assert params.keep_split                  == False
    # assert params.imatrix                     == nullptr
    # assert params.kv_overrides                == nullptr

def test_default_ggml_threadpool_params():
    params = cy.GgmlThreadPoolParams(n_threads=10)
    assert params.n_threads == 10
    assert params.prio == 0
    assert params.poll == 50
    assert params.strict_cpu == False
    assert params.paused == False
    assert params.cpumask == [False] * cy.GGML_MAX_N_THREADS

def test_common_params_sampling():
    params = cy.CommonParamsSampling()
    assert params.seed == 4294967295  # LLAMA_DEFAULT_SEED = 0xFFFFFFFF
    assert params.n_prev             == 64
    assert params.n_probs            == 0
    assert params.min_keep           == 0
    assert params.top_k              == 40
    assert params.top_p              == approx(0.95)
    assert params.min_p              == approx(0.05)
    assert params.xtc_probability    == 0.00
    assert params.xtc_threshold      == approx(0.10)
    assert params.typ_p              == approx(1.00)
    assert params.temp               == approx(0.80)
    assert params.dynatemp_range     == 0.00
    assert params.dynatemp_exponent  == approx(1.00)
    assert params.penalty_last_n     == 64
    assert params.penalty_repeat     == approx(1.00)
    assert params.penalty_freq       == 0.00
    assert params.penalty_present    == 0.00
    assert params.dry_multiplier     == 0.0
    assert params.dry_base           == approx(1.75)
    assert params.dry_allowed_length == 2
    assert params.dry_penalty_last_n == -1
    assert params.mirostat           == 0
    assert params.mirostat_tau       == approx(5.00)
    assert params.mirostat_eta       == approx(0.10)
    assert params.ignore_eos         == False
    assert params.no_perf            == False

def test_common_params():
    params = cy.CommonParams()
    assert params.n_predict            ==    -1
    assert params.n_ctx                ==  4096
    assert params.n_batch              ==  2048
    assert params.n_ubatch             ==   512
    assert params.n_keep               ==     0
    assert params.n_chunks             ==    -1
    assert params.n_parallel           ==     1
    assert params.n_sequences          ==     1
    # assert params.p_split              ==   approx(0.1)
    assert params.n_gpu_layers         ==    -1
    # assert params.n_gpu_layers_draft   ==    -1
    assert params.main_gpu             ==     0
    assert params.tensor_split         ==   [0.0]*128  # First 128 elements are 0.0
    assert params.grp_attn_n           ==     1
    assert params.grp_attn_w           ==   512
    assert params.n_print              ==    -1
    assert params.rope_freq_base       ==   0.0
    assert params.rope_freq_scale      ==   0.0
    assert params.yarn_ext_factor      ==  approx(-1.0)
    assert params.yarn_attn_factor     ==  approx(-1.0)
    assert params.yarn_beta_fast       ==  approx(-1.0)
    assert params.yarn_beta_slow       ==  approx(-1.0)
    assert params.yarn_orig_ctx        ==  0
    # defrag_thold is not available in CommonParams
    # assert params.defrag_thold         ==  approx(0.1)

    assert params.cpuparams.n_threads  == -1
    assert params.cpuparams.cpumask    == [False] * cy.GGML_MAX_N_THREADS
    assert params.cpuparams.mask_valid == False
    assert params.cpuparams.priority   == 0  # GGML_SCHED_PRIO_NORMAL = 0
    assert params.cpuparams.strict_cpu == False
    assert params.cpuparams.poll       == 50

    # assert params.cpuparams_batch      ==
    # assert params.draft_cpuparams      ==
    # assert params.draft_cpuparams_batch ===

    # assert params.cb_eval             == nullptr;
    # assert params.cb_eval_user_data   == nullptr;

    assert params.numa                 ==  0  # GGML_NUMA_STRATEGY_DISABLED = 0
    assert params.split_mode           ==  1  # LLAMA_SPLIT_MODE_LAYER = 1
    assert params.rope_scaling_type    ==  -1  # LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1
    assert params.pooling_type         ==  -1  # LLAMA_POOLING_TYPE_UNSPECIFIED = -1
    assert params.attention_type       ==  -1  # LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1

    # common_sampler_params sparams

    assert params.model                == {'path': b'', 'url': b'', 'hf_repo': b'', 'hf_file': b''}
    # assert params.model_draft          == ""
    assert params.model_alias          == ""
    # model_url, hf_token, hf_repo, hf_file, prompt, prompt_file, path_prompt_cache, 
    # input_prefix, input_suffix, lookup_cache_static, lookup_cache_dynamic, 
    # logits_file, rpc_servers are not available in CommonParams
    # assert params.model_url            == ""
    # assert params.hf_token             == ""
    # assert params.hf_repo              == ""
    # assert params.hf_file              == ""
    # assert params.prompt               == ""
    # assert params.prompt_file          == ""
    # assert params.path_prompt_cache    == ""
    # assert params.input_prefix         == ""
    # assert params.input_suffix         == ""
    # assert params.lookup_cache_static  == ""
    # assert params.lookup_cache_dynamic == ""
    # assert params.logits_file          == ""
    # assert params.rpc_servers          == ""

    assert params.verbosity            >= 0  # Default varies by build config
    assert params.control_vector_layer_start == -1
    assert params.control_vector_layer_end   == -1
    assert params.ppl_stride           == 0
    assert params.ppl_output_type      == 0

    assert params.hellaswag            == False
    assert params.hellaswag_tasks      == 400
    assert params.winogrande           == False
    assert params.winogrande_tasks     == 0
    assert params.multiple_choice      == False
    assert params.multiple_choice_tasks == 0
    assert params.kl_divergence        == False
    assert params.usage                == False 
    assert params.use_color            == False 
    assert params.special              == False 
    assert params.interactive          == False 
    assert params.interactive_first    == False 
    # conversation, prompt_cache_all, prompt_cache_ro, flash_attn, ctx_shift, 
    # input_prefix_bos, logits_all, dump_kv_cache are not available in CommonParams
    # assert params.conversation         == False 
    # assert params.prompt_cache_all     == False 
    # assert params.prompt_cache_ro      == False 
    assert params.escape               == True  
    assert params.multiline_input      == False 
    assert params.simple_io            == False 
    assert params.cont_batching        == True  
    # assert params.flash_attn           == False 
    assert params.no_perf              == False 
    # assert params.ctx_shift            == True  
    # assert params.input_prefix_bos     == False 
    # assert params.logits_all           == False 
    assert params.use_mmap             == True  
    assert params.use_mlock            == False 
    assert params.verbose_prompt       == False 
    assert params.display_prompt       == True  
    # assert params.dump_kv_cache        == False 
    assert params.no_kv_offload        == False 
    assert params.warmup               == True  
    assert params.check_tensors        == False 

    assert params.cache_type_k         == 1  # GGML_TYPE_F16 = 1
    assert params.cache_type_v         == 1  # GGML_TYPE_F16 = 1

    assert params.mmproj               == {'path': b'', 'url': b'', 'hf_repo': b'', 'hf_file': b''}
    assert params.image                == []

    assert params.embedding            == False
    assert params.embd_normalize       == 2
    # embd_out, embd_sep, reranking are not available in CommonParams
    # assert params.embd_out             == ""
    # assert params.embd_sep             == "\n"
    # assert params.reranking            == False

    assert params.port                 == 8080
    assert params.timeout_read         == 600
    assert params.timeout_write        == 600
    assert params.n_threads_http       == -1
    assert params.n_cache_reuse        == 0

    assert params.hostname             == "127.0.0.1"
    # public_path is not available in CommonParams
    # assert params.public_path          == ""
    assert params.chat_template        == ""
    assert params.enable_chat_template == True

    assert params.api_keys             == []
    assert params.ssl_file_key         == ""
    assert params.ssl_file_cert        == ""

    assert params.webui                == True
    assert params.endpoint_slots       == True  # endpoint_slots is True, not False
    assert params.endpoint_props       == False
    assert params.endpoint_metrics     == False

    assert params.log_json             == False

    assert params.slot_save_path       == ""

    assert params.slot_prompt_similarity == approx(0.1)

    assert params.is_pp_shared         == False

    assert params.n_pp                 == []
    assert params.n_tg                 == []
    assert params.n_pl                 == []

    assert params.context_files        == []
    assert params.chunk_size           == 64
    assert params.chunk_separator      == "\n"

    assert params.n_junk               == 250
    assert params.i_pos                == -1
    assert params.out_file             == ""

    assert params.n_out_freq           == 10
    assert params.n_save_freq          == 0
    assert params.i_chunk              == 0

    assert params.process_output       == False
    assert params.compute_ppl          == True

    assert params.n_pca_batch           == 100
    assert params.n_pca_iterations      == 1000
    # assert params.cvector_dimre_method  == cy.DIMRE_METHOD_PCA
    # assert params.cvector_outfile       == "control_vector.gguf"
    # assert params.cvector_positive_file == "examples/cvector-generator/positive.txt"
    # assert params.cvector_negative_file == "examples/cvector-generator/negative.txt"

    # assert params.spm_infill            == False

    # assert params.lora_outfile          == "ggml-lora-merged-f16.gguf"

    # assert params.batched_bench_output_jsonl == False


    # ... rest not yet implemented
