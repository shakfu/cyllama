# xllamacpp Parameters Reference

This document is auto-generated from the Cython wrapper (`.pyx`) and the C++ `common.h` header.
It lists **all Python-accessible properties** of each configuration class, along with their types, default values, and descriptions.

> **Regenerate**: `python scripts/generate_params_doc.py`

## Table of Contents

- [CommonAdapterLoraInfo](#commonadapterlorainfo)
- [CommonParams](#commonparams)
- [CommonParamsDiffusion](#commonparamsdiffusion)
- [CommonParamsModel](#commonparamsmodel)
- [CommonParamsSampling](#commonparamssampling)
- [CommonParamsSpeculative](#commonparamsspeculative)
- [CommonParamsVocoder](#commonparamsvocoder)
- [CpuParams](#cpuparams)

---

## CommonAdapterLoraInfo

LoRA adapter info. Create via `xlc.CommonAdapterLoraInfo(path, scale)`.

| Property | Type | Default | R/W | Description |
|:---------|:-----|:--------|:---:|:------------|
| `path` | str | `` | R/W | LoRA adapter file path. |
| `scale` | float | `` | R/W | LoRA adapter scale factor. |

---

## CommonParams

The central configuration object. Controls model loading, inference, sampling, server behavior, and more. Create via `xlc.CommonParams()`.

### Attention / KV Cache

| Property | Type | Default | R/W | Description |
|:---------|:-----|:--------|:---:|:------------|
| `attention_type` | llama_attention_type | `LLAMA_ATTENTION_TYPE_UNSPECIFIED` | R/W | attention type for embeddings. |
| `cache_type_k` | ggml_type | `GGML_TYPE_F16` | R/W | KV cache data type for the K |
| `cache_type_v` | ggml_type | `GGML_TYPE_F16` | R/W | KV cache data type for the V |
| `ctx_shift` | bool | `false` | R/W | context shift on inifinite text generation |
| `flash_attn_type` | llama_flash_attn_type | `LLAMA_FLASH_ATTN_TYPE_AUTO` | R/W | whether to use Flash Attention. |
| `kv_unified` | bool | `false` | R/W | enable unified KV cache |
| `pooling_type` | llama_pooling_type | `LLAMA_POOLING_TYPE_UNSPECIFIED` | R/W | pooling type for embeddings. |
| `swa_full` | bool | `false` | R/W | use full-size SWA cache (https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055) |

### Batched Bench

| Property | Type | Default | R/W | Description |
|:---------|:-----|:--------|:---:|:------------|
| `is_pp_shared` | bool | `false` | R/W | batched-bench params |
| `is_tg_separate` | bool | `false` | R/W | batched-bench params |
| `n_pl` | list[int] | `` | R/W |  |
| `n_pp` | list[int] | `` | R/W |  |
| `n_tg` | list[int] | `` | R/W |  |

### Behavior Flags

| Property | Type | Default | R/W | Description |
|:---------|:-----|:--------|:---:|:------------|
| `check_tensors` | bool | `false` | R/W | validate tensor data |
| `cont_batching` | bool | `true` | R/W | insert new sequences for decoding on-the-fly |
| `display_prompt` | bool | `true` | R/W | print prompt before generation |
| `escape` | bool | `true` | R/W | escape "\n", "\r", "\t", "\'", "\"", and "\\" |
| `interactive` | bool | `false` | R/W | interactive mode |
| `multiline_input` | bool | `false` | R/W | reverse the usage of "\" |
| `no_perf` | bool | `false` | R/W | disable performance metrics |
| `offline` | bool | `false` | R/W |  |
| `prompt_cache_all` | bool | `false` | R/W | save user input and generations to prompt cache |
| `prompt_cache_ro` | bool | `false` | R/W | open the prompt cache read-only and do not update it |
| `show_timings` | bool | `true` | R/W | show timing information on CLI |
| `simple_io` | bool | `false` | R/W | improves compatibility with subprocesses and limited consoles |
| `single_turn` | bool | `false` | R/W | single turn chat conversation |
| `special` | bool | `false` | R/W | enable special token output |
| `usage` | bool | `false` | R/W | print usage |
| `use_color` | bool | `false` | R/W | use color to distinguish generations and inputs |
| `use_direct_io` | bool | `false` | R/W | read from disk without buffering |
| `use_mlock` | bool | `false` | R/W | use mlock to keep model in memory |
| `use_mmap` | bool | `true` | R/W | enable mmap to use filesystem cache |
| `verbose_prompt` | bool | `false` | R/W | print prompt tokens before generation |
| `warmup` | bool | `true` | R/W | warmup run |

### CPU

| Property | Type | Default | R/W | Description |
|:---------|:-----|:--------|:---:|:------------|
| `cpuparams` | [CpuParams](#cpuparams) | `` | R/W |  |
| `cpuparams_batch` | [CpuParams](#cpuparams) | `` | R/W |  |
| `numa` | ggml_numa_strategy | `GGML_NUMA_STRATEGY_DISABLED` | R/W | KV cache defragmentation threshold. |

### CVector Generator

| Property | Type | Default | R/W | Description |
|:---------|:-----|:--------|:---:|:------------|
| `n_pca_batch` | int | `100` | R/W | start processing from this chunk |
| `n_pca_iterations` | int | `1000` | R/W | start processing from this chunk |

### Control Vector

| Property | Type | Default | R/W | Description |
|:---------|:-----|:--------|:---:|:------------|
| `control_vector_layer_end` | int | `-1` | R/W | layer range for control vector |
| `control_vector_layer_start` | int | `-1` | R/W | layer range for control vector |

### Embedding

| Property | Type | Default | R/W | Description |
|:---------|:-----|:--------|:---:|:------------|
| `cls_sep` | str | `"\t"` | R/W | separator of classification sequences |
| `embd_normalize` | int | `2` | R/W | normalisation for embendings (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm) |
| `embd_out` | str | `""` | R/W | empty = default, "array" = [[],[]...], "json" = openai style, "json+" = same "json" + cosine similarity matrix |
| `embd_sep` | str | `"\n"` | R/W | separator of embendings |
| `embedding` | bool | `false` | R/W | get only sentence embedding |

### GPU / Offloading

| Property | Type | Default | R/W | Description |
|:---------|:-----|:--------|:---:|:------------|
| `fit_params` | bool | `true` | R/W | whether to fit unset model/context parameters to free device memory |
| `fit_params_min_ctx` | int | `4096` | R/W | minimum context size to set when trying to reduce memory use |
| `fit_params_print` | bool | `false` | R/W | print the estimated required memory to run the model |
| `fit_params_target` | list[int] | `std::vector<size_t>(llama_max_devices(), 1024 * 1024*1024)` | R/W | margin per device in bytes for fitting parameters to free memory |
| `main_gpu` | int | `0` | R/W | the GPU that is used for scratch and small tensors |
| `n_gpu_layers` | int | `-1` | R/W | number of layers to store in VRAM, -1 is auto, <= -2 is all |
| `no_extra_bufts` | bool | `false` | R/W | disable extra buffer types (used for weight repacking) |
| `no_host` | bool | `false` | R/W | bypass host buffer allowing extra buffers to be used |
| `no_kv_offload` | bool | `false` | R/W | disable KV offloading |
| `no_op_offload` | bool | `false` | R/W | globally disable offload host tensor operations to device |
| `split_mode` | llama_split_mode | `LLAMA_SPLIT_MODE_LAYER` | R/W | how to split the model across GPUs. |
| `tensor_split` | list[float] | `{0}` | R/W | how split tensors should be distributed across GPUs. |

### IMatrix

| Property | Type | Default | R/W | Description |
|:---------|:-----|:--------|:---:|:------------|
| `compute_ppl` | bool | `true` | R/W | whether to compute perplexity |
| `i_chunk` | int | `0` | R/W | start processing from this chunk |
| `imat_dat` | int | `0` | R/W | whether the legacy imatrix.dat format should be output (gguf <= 0 < dat) |
| `n_out_freq` | int | `10` | R/W | output the imatrix every n_out_freq iterations |
| `n_save_freq` | int | `0` | R/W | save the imatrix every n_save_freq iterations |
| `parse_special` | bool | `false` | R/W | whether to parse special tokens during imatrix tokenization |
| `process_output` | bool | `false` | R/W | collect data for the output tensor |
| `show_statistics` | bool | `false` | R/W | show imatrix statistics per tensor |

### Inference

| Property | Type | Default | R/W | Description |
|:---------|:-----|:--------|:---:|:------------|
| `grp_attn_n` | int | `1` | R/W | group-attention factor. |
| `grp_attn_w` | int | `512` | R/W | group-attention width. |
| `n_batch` | int | `2048` | R/W | logical batch size for prompt processing (must be >=32 to use BLAS) |
| `n_chunks` | int | `-1` | R/W | max number of chunks to process (-1 = unlimited). |
| `n_ctx` | int | `0` | R/W | context size, 0 == context the model was trained with |
| `n_keep` | int | `0` | R/W | number of tokens to keep from initial prompt. |
| `n_parallel` | int | `1` | R/W | number of parallel sequences to decode. |
| `n_predict` | int | `-1` | R/W | max. number of new tokens to predict, -1 == no limit |
| `n_print` | int | `-1` | R/W | print token count every n tokens (-1 = disabled). |
| `n_sequences` | int | `1` | R/W | number of sequences to decode. |
| `n_ubatch` | int | `512` | R/W | physical batch size for prompt processing (must be >=32 to use BLAS) |

### LoRA

| Property | Type | Default | R/W | Description |
|:---------|:-----|:--------|:---:|:------------|
| `lora_adapters` | list | `` | R/W | Get the list of LoRA adapters as a list of [CommonAdapterLoraInfo](#commonadapterlorainfo) objects. |
| `lora_init_without_apply` | bool | `false` | R/W | only load lora to memory, but do not apply it to ctx (user can manually apply lora later using llama_lora_adapter_apply). |

### Model

| Property | Type | Default | R/W | Description |
|:---------|:-----|:--------|:---:|:------------|
| `hf_token` | str | `""` | R/W | HF token (aka bearer token) |
| `model` | [CommonParamsModel](#commonparamsmodel) | `` | R/W |  |
| `model_alias` | set[str] | `` | R/W | model aliases |
| `model_tags` | set[str] | `` | R/W | model tags (informational, not used for routing) |

### Multimodal

| Property | Type | Default | R/W | Description |
|:---------|:-----|:--------|:---:|:------------|
| `image` | list[str] | `` | R/W | path to image file(s) ; TODO: change the name to "media" |
| `image_max_tokens` | int | `-1` | R/W |  |
| `image_min_tokens` | int | `-1` | R/W |  |
| `mmproj` | [CommonParamsModel](#commonparamsmodel) | `` | R/W |  |
| `mmproj_use_gpu` | bool | `true` | R/W | use GPU for multimodal model |
| `no_mmproj` | bool | `false` | R/W | explicitly disable multimodal model |

### Output / Logging

| Property | Type | Default | R/W | Description |
|:---------|:-----|:--------|:---:|:------------|
| `logits_file` | str | `""` | R/W | file for saving *all* logits |
| `logits_output_dir` | str | `"data"` | R/W | directory for saving logits output files |
| `out_file` | str | `` | R/W | output filename for all example programs |
| `save_logits` | bool | `false` | R/W | whether to save logits to files |
| `tensor_filter` | list[str] | `` | R/W | filter tensor names for debug output (regex) |
| `verbosity` | int | `3` | R/W | LOG_LEVEL_INFO |

### Perplexity / Benchmarks

| Property | Type | Default | R/W | Description |
|:---------|:-----|:--------|:---:|:------------|
| `check` | bool | `false` | R/W | check rather than generate results for llama-results |
| `hellaswag` | bool | `false` | R/W | compute HellaSwag score over random tasks from datafile supplied in prompt |
| `hellaswag_tasks` | int | `400` | R/W | number of tasks to use when computing the HellaSwag score |
| `kl_divergence` | bool | `false` | R/W | compute KL divergence |
| `multiple_choice` | bool | `false` | R/W | compute TruthfulQA score over random tasks from datafile supplied in prompt |
| `multiple_choice_tasks` | int | `0` | R/W | number of tasks to use when computing the TruthfulQA score. If 0, all tasks will be computed |
| `ppl_output_type` | int | `0` | R/W | = 0 -> ppl output is as usual, = 1 -> ppl output is num_tokens, ppl, one per line |
| `ppl_stride` | int | `0` | R/W | stride for perplexity calculations. If left at 0, the pre-existing approach will be used. |
| `winogrande` | bool | `false` | R/W | compute Winogrande score over random tasks from datafile supplied in prompt |
| `winogrande_tasks` | int | `0` | R/W | number of tasks to use when computing the Winogrande score. If 0, all tasks will be computed |

### Prompt / Input

| Property | Type | Default | R/W | Description |
|:---------|:-----|:--------|:---:|:------------|
| `antiprompt` | list[str] | `` | R/W | strings upon which more user input is prompted (a.k.a. reverse prompts). |
| `in_files` | list[str] | `` | R/W | all input files. |
| `input_prefix` | str | `""` | R/W | string to prefix user inputs with |
| `input_prefix_bos` | bool | `false` | R/W | prefix BOS to user inputs, preceding input_prefix |
| `input_suffix` | str | `""` | R/W | string to suffix user inputs with |
| `path_prompt_cache` | str | `""` | R/W | path to file for saving/loading prompt eval state |
| `prompt` | str | `""` | R/W | the prompt text |
| `prompt_file` | str | `""` | R/W | store the external prompt file name |

### Retrieval / Passkey

| Property | Type | Default | R/W | Description |
|:---------|:-----|:--------|:---:|:------------|
| `chunk_separator` | str | `"\n"` | R/W | chunk separator for context embedding |
| `chunk_size` | int | `64` | R/W | chunk size for context embedding |
| `context_files` | list[str] | `` | R/W | context files to embed |
| `i_pos` | int | `-1` | R/W | position of the passkey in the junk text |
| `n_junk` | int | `250` | R/W | number of times to repeat the junk text |

### RoPE / YaRN

| Property | Type | Default | R/W | Description |
|:---------|:-----|:--------|:---:|:------------|
| `rope_freq_base` | float | `0.0` | R/W | RoPE base frequency. |
| `rope_freq_scale` | float | `0.0` | R/W | RoPE frequency scaling factor. |
| `rope_scaling_type` | llama_rope_scaling_type | `LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED` | R/W | rope scaling type. |
| `yarn_attn_factor` | float | `-1.0` | R/W | YaRN magnitude scaling factor. |
| `yarn_beta_fast` | float | `-1.0` | R/W | YaRN low correction dim. |
| `yarn_beta_slow` | float | `-1.0` | R/W | YaRN high correction dim. |
| `yarn_ext_factor` | float | `-1.0` | R/W | YaRN extrapolation mix factor. |
| `yarn_orig_ctx` | int | `0` | R/W | YaRN original context length. |

### Server

| Property | Type | Default | R/W | Description |
|:---------|:-----|:--------|:---:|:------------|
| `api_keys` | list[str] | `` | R/W | list of api keys |
| `api_prefix` | str | `""` | R/W |  |
| `cache_idle_slots` | bool | `true` | R/W | save and clear idle slots upon starting a new task |
| `cache_prompt` | bool | `true` | R/W | whether to enable prompt caching |
| `cache_ram_mib` | int | `8192` | R/W | -1 = no limit, 0 - disable, 1 = 1 MiB, etc. |
| `chat_template` | str | `""` | R/W | chat template |
| `default_template_kwargs` | dict | `` | R/W |  |
| `enable_chat_template` | bool | `true` | R/W | enable chat template |
| `enable_reasoning` | int | `-1` | R/W | -1 = auto, 0 = disable, 1 = enable |
| `endpoint_metrics` | bool | `false` | R/W | endpoint metrics |
| `endpoint_props` | bool | `false` | R/W | only control POST requests, not GET |
| `endpoint_slots` | bool | `true` | R/W | endpoint slots |
| `hostname` | str | `"127.0.0.1"` | R/W | server hostname |
| `log_json` | bool | `false` | R/W | log json |
| `media_path` | str | `` | R/W | path to directory for loading media files |
| `models_autoload` | bool | `true` | R/W | automatically load models when requested via the router server |
| `models_dir` | str | `""` | R/W | directory containing models for the router server |
| `models_max` | int | `4` | R/W | maximum number of models to load simultaneously |
| `models_preset` | str | `""` | R/W | directory containing model presets for the router server |
| `n_cache_reuse` | int | `0` | R/W | min chunk size to reuse from the cache via KV shifting |
| `n_ctx_checkpoints` | int | `32` | R/W | max number of context checkpoints per slot |
| `n_threads_http` | int | `-1` | R/W | number of threads to process HTTP requests (TODO: support threadpool) |
| `port` | int | `8080` | R/W | server listens on this network port |
| `prefill_assistant` | bool | `true` | R/W | if true, any trailing assistant message will be prefilled into the response |
| `public_path` | str | `""` | R/W | server public_path |
| `reasoning_format` | common_reasoning_format | `COMMON_REASONING_FORMAT_DEEPSEEK` | R/W |  |
| `sleep_idle_seconds` | int | `-1` | R/W | if >0, server will sleep after this many seconds of idle time |
| `slot_prompt_similarity` | float | `0.1` | R/W | slot prompt similarity. |
| `slot_save_path` | str | `` | R/W | slot save path |
| `ssl_file_cert` | str | `""` | R/W | ssl file cert |
| `ssl_file_key` | str | `""` | R/W | ssl file key |
| `timeout_read` | int | `3600` | R/W | http read timeout in seconds |
| `timeout_write` | int | `timeout_read` | R/W | http write timeout in seconds |
| `use_jinja` | bool | `true` | R/W |  |
| `webui` | bool | `ui` | R/W | deprecated alias for ui |
| `webui_config_json` | str | `` | R/W | deprecated alias for ui_config_json |
| `webui_mcp_proxy` | bool | `false` | R/W | deprecated alias for ui_mcp_proxy |

### Sub-params

| Property | Type | Default | R/W | Description |
|:---------|:-----|:--------|:---:|:------------|
| `diffusion` | [CommonParamsDiffusion](#commonparamsdiffusion) | `` | R/W | common params diffusion. |
| `sampling` | [CommonParamsSampling](#commonparamssampling) | `` | R/W | common params sampling. |
| `speculative` | [CommonParamsSpeculative](#commonparamsspeculative) | `` | R/W | common params speculative. |
| `vocoder` | [CommonParamsVocoder](#commonparamsvocoder) | `` | R/W | common params vocoder. |

### Tensor Buffer Overrides

| Property | Type | Default | R/W | Description |
|:---------|:-----|:--------|:---:|:------------|
| `tensor_buft_overrides` | str | `` | R/W |  |

### Other

| Property | Type | Default | R/W | Description |
|:---------|:-----|:--------|:---:|:------------|
| `checkpoint_min_step` | int | `256` | R/W | minimum spacing between context checkpoints |
| `force_pure_content_parser` | bool | `false` | R/W | force pure content parser |
| `models_preset_hf` | str | `""` | R/W | show a warning about remote presets on router loaded (if not empty) |
| `mtmd_batch_max_tokens` | int | `1024` | R/W |  |
| `n_outputs_max` | int | `0` | R/W | max outputs in a batch (0 = n_batch). |
| `no_alloc` | bool | `false` | R/W | Don't allocate model buffers |
| `path_prompts_log_dir` | str | `""` | R/W | directory with logged prompts |
| `reuse_port` | bool | `false` | R/W | allow multiple sockets to bind to the same port |
| `server_tools` | list[str] | `` | R/W | enable built-in tools |
| `skip_download` | bool | `false` | R/W | skip model file downloading |
| `sse_ping_interval` | int | `30` | R/W | SSE ping interval in seconds |
| `system_prompt` | str | `""` | R/W | the system prompt text |
| `ui` | bool | `true` | R/W | enable UI |
| `ui_config_json` | str | `` | R/W | UI config json |
| `ui_mcp_proxy` | bool | `false` | R/W | UI MCP proxy |

---

## CommonParamsDiffusion

Diffusion model parameters. Access via `params.diffusion`.

| Property | Type | Default | R/W | Description |
|:---------|:-----|:--------|:---:|:------------|
| `add_gumbel_noise` | bool | `false` | R/W | add gumbel noise to the logits if temp > 0.0 |
| `alg_temp` | float | `0.0` | R/W | algorithm temperature |
| `algorithm` | int | `4` | R/W | diffusion algorithm (0=ORIGIN, 1=MASKGIT_PLUS, 2=TOPK_MARGIN, 3=ENTROPY) |
| `block_length` | int | `0` | R/W | block length for generation |
| `cfg_scale` | float | `0` | R/W | classifier-free guidance scale |
| `eps` | float | `0` | R/W | epsilon for timesteps |
| `steps` | int | `128` | R/W | number of diffusion steps |
| `visual_mode` | bool | `false` | R/W | show progressive diffusion on screen |

---

## CommonParamsModel

Model path and source parameters. Access via `params.model`.

| Property | Type | Default | R/W | Description |
|:---------|:-----|:--------|:---:|:------------|
| `docker_repo` | str | `""` | R/W | Docker repo |
| `hf_file` | str | `""` | R/W | HF file |
| `hf_repo` | str | `""` | R/W | HF repo |
| `name` | str | `""` | R/W | in format <user>/<model>[:<tag>] (tag is optional) |
| `path` | str | `""` | R/W | model local path |
| `url` | str | `""` | R/W | model url to download |

---

## CommonParamsSampling

Sampling parameters that control token generation strategy. Access via `params.sampling`.

| Property | Type | Default | R/W | Description |
|:---------|:-----|:--------|:---:|:------------|
| `adaptive_decay` | float | `0.90` | R/W | EMA decay for adaptation; history ≈ 1/(1-decay) tokens (0.0 - 0.99) |
| `adaptive_target` | float | `-1.0` | R/W | select tokens near this probability (valid range 0.0 to 1.0; negative = disabled) |
| `backend_sampling` | bool | `` | R/W | enable backend sampling |
| `dry_allowed_length` | int | `2` | R/W | tokens extending repetitions beyond this receive penalty |
| `dry_base` | float | `1.75` | R/W | 0.0 = disabled;      multiplier * base ^ (length of sequence before token - allowed length) |
| `dry_multiplier` | float | `0.0` | R/W | 0.0 = disabled;      DRY repetition penalty for tokens extending repetition: |
| `dry_penalty_last_n` | int | `-1` | R/W | how many tokens to scan for repetitions (0 = disable penalty, -1 = context size) |
| `dynatemp_exponent` | float | `1.00` | R/W | controls how entropy maps to temperature in dynamic temperature sampler |
| `dynatemp_range` | float | `0.00` | R/W | 0.0 = disabled |
| `generation_prompt` | str | `` | R/W | The assistant generation prompt already prefilled into the prompt. |
| `grammar` | CommonGrammar | `` | R/W | optional grammar constraint (user / output-format / tool-calls) |
| `ignore_eos` | bool | `false` | R/W | ignore end-of-sentence |
| `logit_bias` | list[LlamaLogitBias] | `` | R/W | logit biases to apply |
| `logit_bias_eog` | list[LlamaLogitBias] | `` | R/W | pre-calculated logit biases for EOG tokens |
| `min_keep` | int | `0` | R/W | 0 = disabled, otherwise samplers should return at least min_keep tokens |
| `min_p` | float | `0.05` | R/W | 0.0 = disabled |
| `mirostat` | int | `0` | R/W | 0 = disabled, 1 = mirostat, 2 = mirostat 2.0 |
| `mirostat_eta` | float | `0.10` | R/W | learning rate |
| `mirostat_tau` | float | `5.00` | R/W | target entropy |
| `n_prev` | int | `64` | R/W | number of previous tokens to remember |
| `n_probs` | int | `0` | R/W | if greater than 0, output the probabilities of top n_probs tokens. |
| `no_perf` | bool | `false` | R/W | disable performance metrics |
| `penalty_freq` | float | `0.00` | R/W | 0.0 = disabled |
| `penalty_last_n` | int | `64` | R/W | last n tokens to penalize (0 = disable penalty, -1 = context size) |
| `penalty_present` | float | `0.00` | R/W | 0.0 = disabled |
| `penalty_repeat` | float | `1.00` | R/W | 1.0 = disabled |
| `reasoning_budget_end` | list[int] | `` | R/W | end tag token sequence |
| `reasoning_budget_forced` | list[int] | `` | R/W | forced sequence (message + end tag) |
| `reasoning_budget_message` | str | `` | R/W | message injected before end tag when budget exhausted |
| `reasoning_budget_start` | list[int] | `` | R/W | start tag token sequence |
| `reasoning_budget_tokens` | int | `` | R/W | -1 = disabled, >= 0 = token budget |
| `reasoning_control` | bool | `` | R/W | create the budget sampler on demand so reasoning can be ended at runtime |
| `samplers` | str | `` | R/W | get/set sampler types |
| `seed` | int | `LLAMA_DEFAULT_SEED` | R/W | the seed used to initialize llama_sampler. |
| `temp` | float | `0.80` | R/W | <= 0.0 to sample greedily, 0.0 to not output probabilities |
| `timing_per_token` | bool | `false` | R/W |  |
| `top_k` | int | `40` | R/W | <= 0 to use vocab size. |
| `top_p` | float | `0.95` | R/W | 1.0 = disabled |
| `typ_p` | float | `1.00` | R/W | typical_p, 1.0 = disabled |
| `user_sampling_config` | int | `0` | R/W | bitfield to track user-specified samplers |
| `xtc_probability` | float | `0.00` | R/W | 0.0 = disabled |
| `xtc_threshold` | float | `0.10` | R/W | > 0.5 disables XTC |

---

## CommonParamsSpeculative

Speculative decoding parameters. Access via `params.speculative`.

| Property | Type | Default | R/W | Description |
|:---------|:-----|:--------|:---:|:------------|
| `draft` | [CommonParamsSpeculative](#commonparamsspeculative)Draft | `` | R | draft-model-based speculative decoding parameters. |
| `ngram_cache` | [CommonParamsSpeculative](#commonparamsspeculative)NgramCache | `` | R | ngram cache parameters. |
| `ngram_map_k` | [CommonParamsSpeculative](#commonparamsspeculative)NgramMap | `` | R | ngram map k parameters. |
| `ngram_map_k4v` | [CommonParamsSpeculative](#commonparamsspeculative)NgramMap | `` | R | ngram map k4v parameters. |
| `ngram_mod` | [CommonParamsSpeculative](#commonparamsspeculative)NgramMod | `` | R | ngram mod parameters. |
| `ngram_simple` | [CommonParamsSpeculative](#commonparamsspeculative)NgramMap | `` | R | ngram simple map parameters. |
| `types` | list[common_speculative_type] | `{ COMMON_SPECULATIVE_TYPE_NONE }` | R/W | types of speculative decoding. |

---

## CommonParamsVocoder

Text-to-speech (vocoder) parameters. Access via `params.vocoder`.

| Property | Type | Default | R/W | Description |
|:---------|:-----|:--------|:---:|:------------|
| `model` | [CommonParamsModel](#commonparamsmodel) | `` | R/W |  |
| `speaker_file` | str | `` | R/W | speaker file path |

---

## CpuParams

CPU threading and scheduling parameters. Access via `params.cpuparams` or `params.cpuparams_batch`.

| Property | Type | Default | R/W | Description |
|:---------|:-----|:--------|:---:|:------------|
| `cpumask` | list[bool] | `` | R/W | CPU affinity mask: mask of cpu cores (all-zeros means use default affinity settings) |
| `mask_valid` | bool | `` | R/W | Default: any CPU. |
| `n_threads` | int | `` | R/W | number of threads. |
| `poll` | uint32_t | `` | R/W | Polling (busywait) level (0 - no polling, 100 - mostly polling) |
| `priority` | ggml_sched_priority | `` | R/W | Scheduling prio : (0 - normal, 1 - medium, 2 - high, 3 - realtime). |
| `strict_cpu` | bool | `` | R/W | Use strict CPU placement. |
