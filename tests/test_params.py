import platform
import sys
from pathlib import Path

from pytest import approx

PLATFORM = platform.system()
ARCH = platform.machine()
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'src'))

import cyllama.cyllama as cy


def test_default_model_params():
    params = cy.ModelParams()
    if (PLATFORM, ARCH) == ("Darwin", "arm64"): # i.e. GGML_USE_METAL=ON
        assert params.n_gpu_layers == 999
    else:
        assert params.n_gpu_layers == 0
    assert params.split_mode == cy.LLAMA_SPLIT_MODE_LAYER
    assert params.main_gpu == 0
    assert params.vocab_only == False
    assert params.use_mmap == True
    assert params.use_mlock == False
    assert params.check_tensors == False

def test_default_context_params():
    params = cy.ContextParams()
    assert params.n_ctx               == 512
    assert params.n_batch             == 2048
    assert params.n_ubatch            == 512
    assert params.n_seq_max           == 1
    assert params.n_threads           == cy.GGML_DEFAULT_N_THREADS
    assert params.n_threads_batch     == cy.GGML_DEFAULT_N_THREADS
    assert params.rope_scaling_type   == cy.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED
    assert params.pooling_type        == cy.LLAMA_POOLING_TYPE_UNSPECIFIED
    assert params.attention_type      == cy.LLAMA_ATTENTION_TYPE_UNSPECIFIED
    assert params.rope_freq_base      == 0.0
    assert params.rope_freq_scale     == 0.0
    assert params.yarn_ext_factor     == -1.0
    assert params.yarn_attn_factor    == 1.0
    assert params.yarn_beta_fast      == 32.0
    assert params.yarn_beta_slow      == 1.0
    assert params.yarn_orig_ctx       == 0
    assert params.defrag_thold        == -1.0
    # assert params.cb_eval             == nullptr
    # assert params.cb_eval_user_data   == nullptr
    assert params.type_k              == cy.GGML_TYPE_F16
    assert params.type_v              == cy.GGML_TYPE_F16
    assert params.logits_all          == False
    assert params.embeddings          == False
    assert params.offload_kqv         == True
    assert params.flash_attn          == False
    assert params.no_perf             == True
    # assert params.abort_callback      == nullptr
    # assert params.abort_callback_data == nullptr

def test_default_model_quantize_params():
    params = cy.ModelQuantizeParams()
    assert params.nthread                     == 0
    assert params.ftype                       == cy.LLAMA_FTYPE_MOSTLY_Q5_1
    assert params.output_tensor_type          == cy.GGML_TYPE_COUNT
    assert params.token_embedding_type        == cy.GGML_TYPE_COUNT
    assert params.allow_requantize            == False
    assert params.quantize_output_tensor      == True
    assert params.only_copy                   == False
    assert params.pure                        == False
    assert params.keep_split                  == False
    # assert params.imatrix                     == nullptr
    # assert params.kv_overrides                == nullptr

def test_default_ggml_threadpool_params():
    params = cy.GGMLThreadPoolParams(n_threads=10)
    assert params.n_threads == 10
    assert params.prio == 0
    assert params.poll == 50
    assert params.strict_cpu == False
    assert params.paused == False
    assert params.cpumask == [False] * cy.GGML_MAX_N_THREADS

def test_common_sampler_params():
    params = cy.CommonSamplerParams()
    assert params.seed == cy.LLAMA_DEFAULT_SEED
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
    assert params.penalize_nl        == False
    assert params.ignore_eos         == False
    assert params.no_perf            == False

def test_common_params():
    params = cy.CommonParams()
    assert params.n_predict            ==    -1
    assert params.n_ctx                ==  4096
    assert params.n_batch              ==  2048
    assert params.n_ubatch             ==   512
    assert params.n_keep               ==     0
    assert params.n_draft              ==     5
    assert params.n_chunks             ==    -1
    assert params.n_parallel           ==     1
    assert params.n_sequences          ==     1
    assert params.p_split              ==   approx(0.1)
    assert params.n_gpu_layers         ==    -1
    assert params.n_gpu_layers_draft   ==    -1
    assert params.main_gpu             ==     0
    assert params.tensor_split         ==   [0]*128
    assert params.grp_attn_n           ==     1
    assert params.grp_attn_w           ==   512
    assert params.n_print              ==    -1
    assert params.rope_freq_base       ==   0.0
    assert params.rope_freq_scale      ==   0.0
    assert params.yarn_ext_factor      ==  approx(-1.0)
    assert params.yarn_attn_factor     ==  approx(1.0)
    assert params.yarn_beta_fast       ==  approx(32.0)
    assert params.yarn_beta_slow       ==  approx(1.0)
    assert params.yarn_orig_ctx        ==  0
    assert params.defrag_thold         ==  approx(-1.0)
    assert params.numa                 ==  cy.GGML_NUMA_STRATEGY_DISABLED
    assert params.split_mode           ==  cy.LLAMA_SPLIT_MODE_LAYER
    assert params.rope_scaling_type    ==  cy.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED
    assert params.pooling_type         ==  cy.LLAMA_POOLING_TYPE_UNSPECIFIED
    assert params.attention_type       ==  cy.LLAMA_ATTENTION_TYPE_UNSPECIFIED

    assert params.model                == ""
    assert params.model_draft          == ""
    assert params.model_alias          == "unknown"
    assert params.model_url            == ""
    assert params.hf_token             == ""
    assert params.hf_repo              == ""
    assert params.hf_file              == ""
    assert params.prompt               == ""
    assert params.prompt_file          == ""
    assert params.path_prompt_cache    == ""
    assert params.input_prefix         == ""
    assert params.input_suffix         == ""
    assert params.logdir               == ""
    assert params.lookup_cache_static  == ""
    assert params.lookup_cache_dynamic == ""
    assert params.logits_file          == ""
    assert params.rpc_servers          == ""

    assert params.verbosity            == 0
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
    assert params.conversation         == False 
    assert params.prompt_cache_all     == False 
    assert params.prompt_cache_ro      == False 
    assert params.escape               == True  
    assert params.multiline_input      == False 
    assert params.simple_io            == False 
    assert params.cont_batching        == True  
    assert params.flash_attn           == False 
    assert params.no_perf              == False 
    assert params.ctx_shift            == True  
    assert params.input_prefix_bos     == False 
    assert params.logits_all           == False 
    assert params.use_mmap             == True  
    assert params.use_mlock            == False 
    assert params.verbose_prompt       == False 
    assert params.display_prompt       == True  
    assert params.dump_kv_cache        == False 
    assert params.no_kv_offload        == False 
    assert params.warmup               == True  
    assert params.check_tensors        == False 
