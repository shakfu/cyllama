# TODO

## Bugs

- [ ] **Ctrl-C does not interrupt `cyllama.sd` generation** ([#8](https://github.com/shakfu/cyllama/issues/8)) -- `generate_image()` is a single blocking C call with no abort path; the LLM cancellation work in `0.2.14` does not transfer (separate compute graph). Fix is gated on upstream PR [leejet/stable-diffusion.cpp#1124](https://github.com/leejet/stable-diffusion.cpp/pull/1124) which adds `sd_cancel_generation(sd_ctx, sd_cancel_mode_t)`. Once that merges and `--sd-version` bumps to a release containing it: extend `stable_diffusion.pxd` with the enum + extern, add `SDContext.cancel(mode)` and `SDContext.install_sigint_handler()` mirroring the LLM helpers (`src/cyllama/api.py` `_SigintHandle`), wire into the cyllama-desktop sidecar's `asyncio.CancelledError` path. Tests in `tests/test_sd_cancel.py` modeled on `tests/test_cancel.py`.

## Medium Priority

- [ ] **Expose video-with-audio output from `SDContext.generate_video`** -- stable-diffusion.cpp `master-652-92dc726` changed `generate_video` to emit an optional audio track via a new `sd_audio_t** audio_out` out-param (some video models, e.g. WAN-family, produce audio alongside frames). The binding currently allocates and immediately frees it: `src/cyllama/sd/stable_diffusion.pyx:generate_video` passes `&audio_out` then calls `free_sd_audio(audio_out)` without exposing the data. The `sd_audio_t` struct (`sample_rate`, `channels`, `sample_count`, `float* data`) and `free_sd_audio()` are already declared in `src/cyllama/sd/stable_diffusion.pxd`. To expose: add a Python-side `SDAudio` wrapper (mirroring `SDImage._from_c_image`, copying the float PCM buffer before the `free_sd_audio` call), and change `generate_video`'s return from `List[SDImage]` to a `(frames, audio)` pair or a small result object (keep `audio=None` when the model produces none, to avoid breaking existing callers). Trigger: a user runs an audio-producing video model, or we add a video-gen test that exercises a model known to emit audio. No runtime test covers `generate_video` today (needs a video model not in the suite), so land a smoke test alongside.

- [ ] Performance regression detection -- CI-integrated baseline capture/comparison to catch speed or memory regressions across commits

- [ ] Structured logging system (JSON output option, agent decision flow logging)

## Wheel / Packaging

- [ ] **Centralize ggml backend registration in `ensure_backends_loaded()`** -- the 0.2.17 hotfix calls `ggml_backend_load_all()` from `src/cyllama/sd/__init__.py` to fix the post-`master-592` "No devices found!" regression. The loader logic itself lives inside the sd Cython module (`src/cyllama/sd/stable_diffusion.pyx:170-191`), duplicating path resolution that also exists in `src/cyllama/_internal/backend_dl.py` (`libs_to_load`). The proper fix is to extract a single `ensure_backends_loaded()` helper into `_internal/backend_dl.py` (idempotent, env-var opt-out e.g. `CYLLAMA_DISABLE_GPU=1`) and call it once from `cyllama.utils.platform.ensure_native_deps()` so llama / whisper / sd all share one registration path. Then drop or thin-wrap the sd-local `ggml_backend_load_all` Python shim. Trigger: next time llama.cpp or whisper.cpp upstreams switch to runtime backend discovery (sd.cpp already did in [#1448](https://github.com/leejet/stable-diffusion.cpp/pull/1448)) -- doing this proactively avoids a repeat of the 0.2.16 SD regression.

- [ ] stable-diffusion.cpp uses compile-time `#ifdef SD_USE_CUDA` for backend selection instead of dynamic `ggml_backend_load_all()` like llama.cpp and whisper.cpp -- propose dynamic backend discovery upstream or patch locally for consistency *(NOTE: superseded by sd.cpp `master-592` [#1448](https://github.com/leejet/stable-diffusion.cpp/pull/1448) which made the switch -- verify this item can be closed)*

## Explore

- [ ] MCP server (`cyllama/mcp/`): expose local inference (`complete`, `chat`, `embed`, `transcribe`, `generate_image`) as MCP tools and model listing as resources. Two transports: stdio entrypoint for subprocess clients (Claude Desktop), and Streamable-HTTP routes mounted on `EmbeddedServer` (`src/cyllama/llama/server/embedded.pyx`) for Claude Code / remote clients. Reuse `agents/jsonrpc.py` framing and the high-level API in `src/cyllama/api.py` -- no new heavy deps. (Client side already shipped: `LLM.add_mcp_server()` in `src/cyllama/api.py:1378` wraps `agents/mcp.py` for non-agent callers.)

## Agent framework

These three items are the residue of `AGENT_TOOL_REVIEW.md` after every concrete proposal landed or was explicitly dropped. Each has a clear trigger; none is urgent.

- [ ] **Stop-pattern migration in `_extract_answer`** -- `src/cyllama/agents/react.py` keeps a hand-maintained list of ~24 hallucination stop-patterns (code blocks, `Note:`, `Let's`, `def `, `class `, etc.) and post-processes generated text against them. The principled fix is to extend `GenerationConfig.stop_sequences` for the answer-extraction generation step instead; the default config already wires `stop_sequences` for `Observation:` patterns (`react.py:200-206`) and `LLM.__call__` honors them (`api.py:1276` runs `_find_stop_sequence`). Sketch: `cfg = replace(self.generation_config, stop_sequences=self.generation_config.stop_sequences + [...])` for the answer turn only; the regex strip becomes a fallback for models that ignore stop sequences. Trigger: refactor the next time the stop-pattern list grows from a new model-specific failure mode -- accreting another regex is the wrong response.

- [ ] **MCP SSE transport** -- `src/cyllama/agents/mcp.py` implements `McpStdioConnection` (line 148) and `McpHttpConnection` (line 271) but `McpTransportType.SSE` (line 38) is reserved-but-unwired. A symmetric `McpSseConnection` would slot in next to the HTTP one, dispatched from `McpClient._connect_server` (line 400). Bulk of the cost is an integration test harness with a real SSE-speaking MCP server; the protocol class itself is small. Trigger: an MCP server you want to use exposes SSE-only. The ecosystem is mostly stdio/HTTP today, so this is unlikely soon.

- [ ] **ACP protocol-version negotiation** -- `src/cyllama/agents/acp.py` hardcodes `ACP_PROTOCOL_VERSION = "2025-01-01"` (line 53) and embeds it directly in initialize responses (line 480). The module is marked experimental for this and other reasons, so the warning currently buys time -- but if ACP graduates from POC to a genuinely-used integration point, parameterize on the client's announced version (negotiate during initialize). Trigger: an ACP client surfaces with a different version. (A reference-client conformance test was also flagged but doesn't belong in a TODO until a harness target exists.)

### Pattern-coverage refinements (future, no urgency)

These are residual refinements documented under each pattern's "Gap" line in `docs/agents/patterns.md`. None block the pattern; each is a possible extension when a use case appears.

**Note:** every entry in this section should land in `inferna` too. The two projects share the agent layer byte-identical modulo namespace; refinements ported one-way only would drift the surfaces.

- [ ] **Streaming sub-agent events across MCP** -- `mcp_agent_tool` returns a single value per call; streaming would require the MCP server-streaming RFC to stabilize.
- [ ] **Streaming RAG results to the agent** -- `rag_as_tool` returns a single concatenated observation today.
- [ ] **Filtered deletion in `SemanticMemory`** -- `forget()` raises `NotImplementedError` pending a RAG-side metadata-filtered delete API.
- [ ] **Parallel critic ensembles in `ReflectionLoop`** -- multiple critics voting, reward-model-based acceptance.
- [ ] **Unified streaming for `plan_and_execute` steps** -- one iterator surfacing events from all steps in sequence (e.g. an `aplan_and_execute` async-generator variant, or a `stream=True` flag on the existing helper). `cyllama-desktop`'s sidecar bypasses the wrapper today and reimplements the loop with `planner.stream()` / `executor.stream()` precisely to get incremental events flowing into its SSE channel; a streaming variant in cyllama would let that ~100 LoC of reimplementation go away. Trigger: the next consumer (after cyllama-desktop) that needs per-step events.

- [ ] **`ReflectionLoop.stream()` per-attempt `source` labels.** Today `composition.py` sets `event.source = "worker"` / `"critic"` (role only -- see `_reflect_loop` in `composition.py`). Downstream consumers (cyllama-desktop today) want `worker-1` / `critic-1` / `worker-2` / etc. so the trace renderer can distinguish attempts. ~5 LoC change: `f"worker-{attempt + 1}"` in the source-tagging block. Same surface for ReflectionLoop's async variant if/when it lands. Trigger: a consumer wants per-attempt distinction (cyllama-desktop already does -- it currently reimplements the loop in the sidecar for this reason).

- [ ] **Document `SemanticMemory`'s actual RAG-shaped protocol.** The class docstring says it wraps a `cyllama.rag.RAG` instance; in practice the implementation only calls `.add_texts(texts, metadata, split)` and `.search(query, k, threshold)` on the wrapped object. Any duck-typed shape works, but the docstring buries this. `cyllama-desktop`'s sidecar built a ~20 LoC `_MemoryRagShim` around its `Embedder` + `SqliteVectorStore` precisely because a real `RAG` instance requires a `generation_model` it doesn't have. Two changes: (a) docstring rewrite stating the protocol; (b) optional `MemoryRagProtocol` (or similar) type alias under `agents.memory` that consumers can use for type checking. Trigger: a follow-up doc pass.

- [ ] **`ContractPolicy.from_name(s)` classmethod.** Today consumers wanting to map a UI string (`"OBSERVE"`, etc.) to a `ContractPolicy` enum member call `getattr(ContractPolicy, name)` and handle the `AttributeError` themselves. A `from_name` factory + a `Literal["IGNORE", "OBSERVE", "ENFORCE", "QUICK_ENFORCE"]` type alias would tighten the boundary and remove the boilerplate from every consumer. ~10 LoC. Trigger: the next consumer that has to do this dance.

- [ ] **Expose `Workflow.inputs_schema` (typed inputs, not just names).** `compiled.dry_run().inputs_required` is `Tuple[str, ...]` -- names only. Layer-C nodes have parameter annotations the framework already reads (`_extract_param_names`, `typing.get_type_hints`); surfacing those as `{key: type}` would let consumers render typed input forms instead of always-text fields. `cyllama-desktop`'s Workflows pane uses text inputs for everything and the user has to know that `count` is an `int` etc. ~20 LoC to populate `inputs_schema` on the `DryRunPlan` dataclass. Trigger: a consumer with a workflow whose inputs are numeric / boolean / enum.

### Pattern gaps -- explicitly **not on the roadmap**

These appear in `docs/agents/patterns.md` but won't be addressed without a forcing use case. Listed here to make the position explicit rather than implicit.

- **Tree of Thoughts (ToT)** -- requires public `LlamaContext.snapshot()` / `restore()` (currently absent at the API surface) plus a branching agent loop that maintains a frontier of candidate states with scoring. Significant new machinery; non-trivial value for cyllama's typical user. Skip unless a user with a concrete ToT use case shows up.

- **Autonomous / AutoGPT-style** -- structurally opposed to cyllama's design stance (bounded loops, loop detection, max_iterations, contracts for budget invariants). Unbounded goal-decomposition is what the framework actively prevents. Document the stance, don't accommodate it.

## CI / Workflows

### Medium Priority

- [ ] **Path-filtered `push` / `pull_request` triggers** -- llama.cpp's `build-vulkan.yml:5-25`, `release.yml:10-28`, `editorconfig.yml:9-14` all auto-trigger on narrow `paths:` filters. cyllama is entirely `workflow_dispatch` today, so real regressions can ship to users. Auto-trigger CPU cibw on PRs touching `src/cyllama/**`, `scripts/manage.py`, `pyproject.toml`; keep GPU wheels on `workflow_dispatch`

- [ ] **Lightweight Python lint / type-check workflows** -- llama.cpp has `python-lint.yml`, `python-type-check.yml`, `python-check-requirements.yml`, `editorconfig.yml` using `runs-on: ubuntu-slim`, triggered only on `**/*.py` / config path changes, running in <1 min. Cheap pre-filter before the 40-minute wheel matrix. Add `.github/workflows/python-lint.yml` with `ruff check` and optionally `mypy` / `ty`

- [ ] **Composite actions for repeated toolchain setup** -- llama.cpp factors into `.github/actions/{windows-setup-cuda,linux-setup-vulkan,windows-setup-rocm,unarchive-tar,get-tag-name}/action.yml` and reuses them across `build-vulkan.yml`, `release.yml`, `build-cache.yml`. cyllama duplicates the Vulkan-SDK pwsh install (~15 lines) and a version-reading Python snippet (`build-cibw.yml:234-239`, `build-gpu-wheels.yml:664-670`). Extract `.github/actions/setup-vulkan-windows` and `.github/actions/get-version`

- [ ] **Reusable `workflow_call` smoke-test** -- cyllama's wheel-find + venv + import + inference block is duplicated across all three workflow files with minor variations. Wrap `scripts/run_wheel_test.py` in `.github/workflows/_smoke-test.yml` with `on: workflow_call` (inputs: `artifact-name`, `runs-on`, `run-inference`) and delete ~200 lines of duplication

### Wheel Coverage (additional backend variants)

Gap analysis vs. llama.cpp b8893 release assets. Ordered by effort/payoff.

- [ ] **Windows SYCL (Intel Arc + Xe)** -- Linux SYCL is shipped (`build_sycl` in `build-gpu-wheels-abi3.yml:319`, wheel name `cyllama-sycl`). Windows SYCL still pending: follows the same pattern as windows-cuda/vulkan -- download prebuilt, synthesize `.lib` via existing `_generate_import_libs()`, `delvewheel --include ggml-sycl.dll` with `--no-dll` for `sycl[78].dll`, `pi_level_zero.dll`, `pi_opencl.dll`, `svml_dispmd.dll`, `libmmd.dll`, `libiomp5md.dll` (user-installed Intel oneAPI runtime). Build-time dep: Intel oneAPI DPC++ on the Windows runner -- use `oneapi-src/setup-oneapi` or similar. Needs `_release_asset_name()` + `_dylib_names` extended to recognize Windows SYCL assets

- [ ] **Windows HIP Radeon (AMD GPUs)** -- upstream ships `llama-b8893-bin-win-hip-radeon-x64.zip`. Same download+synthesize+delvewheel pattern; `--include ggml-hip.dll`, `--no-dll` for `amdhip64_6.dll`, `hipblas.dll`, `rocblas.dll`, `amd_comgr_*.dll` (user-installed AMD HIP SDK / Adrenalin runtime). Main obstacle: AMD HIP SDK Windows install on CI has no compact GitHub Action -- needs manual `Invoke-WebRequest` of AMD's installer (~2-3 GB) plus silent-install args, or a `choco install` package if one exists. Highest effort of the three Windows GPU gaps

- [ ] **Linux ROCm 7.2 prebuilt** -- upstream now ships `llama-b8893-bin-ubuntu-rocm-7.2-x64.tar.gz`. Current `build_rocm` job compiles ROCm 6.3 from source (20-40 min); switching to the prebuilt would cut CI time dramatically. Tradeoff: constrained to upstream's arch list (we currently target `gfx90a;gfx942;gfx1100` explicitly). Evaluate whether upstream's default architectures are acceptable before committing

- [ ] **ARM64 variants** -- growing relevance (Copilot+ PCs, Ampere/Graviton clouds, Apple Silicon KleidiAI). Currently commented out in `build-cibw.yml` for `ubuntu-24.04-arm` and `windows-11-arm`; upstream ships `ubuntu-arm64`, `ubuntu-vulkan-arm64`, `win-cpu-arm64`, `macos-arm64-kleidiai`. Needs its own wheel variant names and separate investigation of build-time toolchain availability on ARM runners

- [ ] **Linux OpenVINO (Intel CPU accelerator)** -- upstream ships `llama-b8893-bin-ubuntu-openvino-2026.0-x64.tar.gz` as a new backend. Would require cyllama-side integration work (build flags, runtime loader, backend detection in `build_config.json`) on top of the wheel packaging. Lower priority until there's user demand

### Lower Priority

- [ ] **Separate SDK caches from `thirdparty/` build-artifact caches** -- llama.cpp's `build-cache.yml` caches SDK install dirs (`C:\Program Files\AMD\ROCm`, `./vulkan_sdk`, `./openvino_toolkit`) under keys scoped to `HIPSDK_INSTALLER_VERSION`, separate from compile caches. cyllama's single `deps-<backend>-<linkmode>` key mixes SDK install with built artifacts, so a `manage.py` edit invalidates cached SDK binaries too. For `build-gpu-wheels.yml`: consider `--manylinux-image` with pre-built CUDA/ROCm image, or a container-volume trick to cache `/usr/local/cuda`

- [ ] **Monotonic `b<commit-count>` build-tag scheme** -- llama.cpp's `get-tag-name/action.yml` uses `fetch-depth: 0` + `git rev-list --count HEAD` to produce `b${BUILD_NUMBER}` on `master`, `${branch}-b${count}-${sha7}` on branches. Lets every CI-built wheel be distinguishable without bumping `pyproject.toml` version on every pre-release. Apply in `build-cibw.yml` / `build-gpu-wheels.yml` upload steps

## RAG Scaling (see docs/dev/scaling_rag.md)

- [ ] Persistent quantization state in database metadata (quantize() exists but state is in-memory only)

- [ ] Metadata pre-filtering in vector search (filter by source, date, etc.)

- [ ] Async embedding generation (`embed_batch_async()`)

- [ ] Parallel document loading in DirectoryLoader

- [ ] Batch query processing in RAG pipeline

- [ ] Sharding for 1M+ vector workloads