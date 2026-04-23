# TODO

## Bugs

## Medium Priority

- [ ] Performance regression detection -- CI-integrated baseline capture/comparison to catch speed or memory regressions across commits

- [ ] Structured logging system (JSON output option, agent decision flow logging)

## Wheel / Packaging

- [ ] stable-diffusion.cpp uses compile-time `#ifdef SD_USE_CUDA` for backend selection instead of dynamic `ggml_backend_load_all()` like llama.cpp and whisper.cpp -- propose dynamic backend discovery upstream or patch locally for consistency

## Explore

- [ ] MCP integration (two directions):

  - **Server** (`cyllama/mcp/`): expose local inference (`complete`, `chat`, `embed`, `transcribe`, `generate_image`) as MCP tools and model listing as resources. Two transports: stdio entrypoint for subprocess clients (Claude Desktop), and Streamable-HTTP routes mounted on `EmbeddedServer` (`src/cyllama/llama/server/embedded.pyx`) for Claude Code / remote clients. Reuse `agents/jsonrpc.py` framing and the high-level API in `src/cyllama/api.py` -- no new heavy deps.

  - **Client (high-level surface)**: stdio + HTTP client transports already exist in `src/cyllama/agents/mcp.py` and are wired into the agent `Tool` abstraction. Lift that onto the top-level `LLM`/`chat()` API (`LLM.add_mcp_server(...)`) so non-agent callers can attach MCP servers and have tools dispatched inside tool-calling loops. Llama.cpp upstream has no server-side MCP -- only the Svelte webui ships a TS client (`build/llama.cpp/tools/server/webui/src/lib/utils/mcp.ts`) -- so this stays a cyllama-layer concern.

## CI / Workflows

### High Priority

- [ ] **Add `concurrency:` groups to all workflows** -- every llama.cpp workflow has `concurrency: { group: ${{ github.workflow }}-${{ github.head_ref && github.ref || github.run_id }}, cancel-in-progress: true }` (e.g. `release.yml:30-32`, `build-vulkan.yml:28-30`). Currently a second push while a cyllama GPU matrix is running wastes ~2h of runtime. 2-line change per file, immediate savings. Apply to `build-cibw.yml`, `build-gpu-wheels.yml`, `build-new-wheels.yml`

- [x] **Vendor-drift guard workflow (`check-vendor.yml` pattern)** -- llama.cpp's `check-vendor.yml` re-runs `scripts/sync_vendor.py` in CI and fails if the tree differs. cyllama's `git status` currently shows modified vendored headers (`mtmd.h`, `log.h`, `ggml-backend.h`, `common.h`) -- exactly the class of silent drift that caused the `GGML_MAX_NAME=128` ABI-match incident. Add `.github/workflows/check-thirdparty.yml` that runs the vendor sync step and diffs. Implemented as `manage.py check_vendor` subcommand + `.github/workflows/check-vendor.yml`

- [ ] **`ggml-org/ccache-action` for C/C++ compile caching** -- used in every native build job in `release.yml` (e.g. lines 67-71, 122-127, 189-193, 252-256, 403-408), keyed per `os-arch-backend`, with `evict-old-files: 1d` and `save: ${{ github.event_name == 'push' && github.ref == 'refs/heads/master' }}`. Biggest win on CUDA/ROCm where compile dominates and the `thirdparty/` cache key busts often. Wire into `CIBW_ENVIRONMENT` via `CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache` and `CIBW_ENVIRONMENT_PASS_LINUX: CCACHE_DIR`. Note: Windows (MSVC) ccache skipped -- debug-format / cmake-integration caveats need a separate investigation

### Medium Priority

- [ ] **Path-filtered `push` / `pull_request` triggers** -- llama.cpp's `build-vulkan.yml:5-25`, `release.yml:10-28`, `editorconfig.yml:9-14` all auto-trigger on narrow `paths:` filters. cyllama is entirely `workflow_dispatch` today, so real regressions can ship to users. Auto-trigger CPU cibw on PRs touching `src/cyllama/**`, `scripts/manage.py`, `pyproject.toml`; keep GPU wheels on `workflow_dispatch`

- [ ] **Lightweight Python lint / type-check workflows** -- llama.cpp has `python-lint.yml`, `python-type-check.yml`, `python-check-requirements.yml`, `editorconfig.yml` using `runs-on: ubuntu-slim`, triggered only on `**/*.py` / config path changes, running in <1 min. Cheap pre-filter before the 40-minute wheel matrix. Add `.github/workflows/python-lint.yml` with `ruff check` and optionally `mypy` / `ty`

- [ ] **Composite actions for repeated toolchain setup** -- llama.cpp factors into `.github/actions/{windows-setup-cuda,linux-setup-vulkan,windows-setup-rocm,unarchive-tar,get-tag-name}/action.yml` and reuses them across `build-vulkan.yml`, `release.yml`, `build-cache.yml`. cyllama duplicates the Vulkan-SDK pwsh install (~15 lines) and a version-reading Python snippet (`build-cibw.yml:234-239`, `build-gpu-wheels.yml:664-670`). Extract `.github/actions/setup-vulkan-windows` and `.github/actions/get-version`

- [ ] **Reusable `workflow_call` smoke-test** -- cyllama's wheel-find + venv + import + inference block is duplicated across all three workflow files with minor variations. Wrap `scripts/run_wheel_test.py` in `.github/workflows/_smoke-test.yml` with `on: workflow_call` (inputs: `artifact-name`, `runs-on`, `run-inference`) and delete ~200 lines of duplication

### Wheel Coverage (additional backend variants)

Gap analysis vs. llama.cpp b8893 release assets. Ordered by effort/payoff.

- [ ] **Windows CUDA 13.1** -- upstream ships `llama-*-bin-win-cuda-13.1-x64.zip` alongside the 12.4 variant. Trivial addition: second job in `build-gpu-wheels2.yml` mirroring `build_cuda_windows`, Jimver/cuda-toolkit supports 13.1, wheel name `cyllama-cuda13`. Pays off for Blackwell/Ada users. `manage.py download_release()` already fetches the companion `cudart-llama-bin-win-cuda-{cuda_ver}-x64.zip` asset, so only `cuda_ver` changes

- [ ] **Windows SYCL (Intel Arc + Xe)** -- upstream ships `llama-b8893-bin-win-sycl-x64.zip`. Follows the same pattern as windows-cuda/vulkan: download prebuilt, synthesize `.lib` via existing `_generate_import_libs()`, `delvewheel --include ggml-sycl.dll` with `--no-dll` for `sycl[78].dll`, `pi_level_zero.dll`, `pi_opencl.dll`, `svml_dispmd.dll`, `libmmd.dll`, `libiomp5md.dll` (user-installed Intel oneAPI runtime). Build-time dep: Intel oneAPI DPC++ on the Windows runner for SD's own SYCL kernels -- use `oneapi-src/setup-oneapi` or similar. Needs `_release_asset_name()` + `_dylib_names` extended to recognize SYCL

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

## RAG Pipeline Integration

- [ ] **Pipeline-integrated reranking + `RerankerProtocol`** -- Plumb `Reranker` (currently user-facing only in `src/cyllama/rag/advanced.py`) into `RAGPipeline` so callers can turn on reranking via config instead of stitching retrieval + rerank together by hand. Proposed surface: `RAGConfig(rerank=True, rerank_top_k=20, reranker=<instance>)` where `top_k` is the pre-rerank retrieval depth and the final result count falls back to `cfg.top_k`. `RAGPipeline._retrieve` (currently `self.store.search(query_embedding, k=cfg.top_k, ...)`) becomes: retrieve `rerank_top_k` candidates, call `reranker.rerank(query, candidates, top_k=cfg.top_k)`, return the reranked head. Introduce `RerankerProtocol` (`score`, `rerank`, `close`) in `src/cyllama/rag/types.py` in the same PR so the pipeline's call site has a real consumer to contract against -- shipping the protocol without the pipeline hook is speculative abstraction (see the reasoning in CHANGELOG's "Embedder/Store/Agent protocols" entry). Existing `Reranker` class inherits from the protocol explicitly, same pattern as `Embedder` → `EmbedderProtocol`. Keep the concrete reranker optional -- `RAGConfig.rerank=False` stays the default so unchanged code paths pay nothing.

## Alternative vector-store backends

- [ ] **Qdrant adapter (`cyllama/rag/stores/qdrant.py`)** -- The cross-backend seam landed (`VectorStoreProtocol` in `src/cyllama/rag/store.py`, `RAG(store=...)` injection point in `src/cyllama/rag/rag.py`). Ship a first reference adapter so users have a worked example: thin wrapper around `qdrant_client.QdrantClient` implementing the seven protocol methods (`search`, `add`, `is_source_indexed`, `get_source_by_label`, `clear`, `close`, `__len__`). Source dedup maps to a Qdrant payload field (e.g. `content_hash`); `is_source_indexed` becomes a count query, `get_source_by_label` a payload filter. Behind an optional dep group (`pip install cyllama[qdrant]`); skip-if-not-installed test pattern from `tests/test_pdf.py`. Once Qdrant works, Chroma / LanceDB / pgvector adapters follow the same template -- consider whether to ship them in-tree or document the protocol and let the community own them.