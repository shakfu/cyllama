# TODO

## Bugs

## Medium Priority

- [ ] Performance regression detection -- CI-integrated baseline capture/comparison to catch speed or memory regressions across commits

- [ ] Structured logging system (JSON output option, agent decision flow logging)

## Wheel / Packaging

- [ ] stable-diffusion.cpp uses compile-time `#ifdef SD_USE_CUDA` for backend selection instead of dynamic `ggml_backend_load_all()` like llama.cpp and whisper.cpp -- propose dynamic backend discovery upstream or patch locally for consistency

## Explore

- [ ] MCP server (`cyllama/mcp/`): expose local inference (`complete`, `chat`, `embed`, `transcribe`, `generate_image`) as MCP tools and model listing as resources. Two transports: stdio entrypoint for subprocess clients (Claude Desktop), and Streamable-HTTP routes mounted on `EmbeddedServer` (`src/cyllama/llama/server/embedded.pyx`) for Claude Code / remote clients. Reuse `agents/jsonrpc.py` framing and the high-level API in `src/cyllama/api.py` -- no new heavy deps. (Client side already shipped: `LLM.add_mcp_server()` in `src/cyllama/api.py:1378` wraps `agents/mcp.py` for non-agent callers.)

## CI / Workflows

### High Priority

- [x] **Add `concurrency:` groups to all workflows** -- applied across `build-gpu-wheels.yml`, `build-gpu-wheels-abi3.yml`, `build-new-wheels.yml`, `check-vendor.yml`, `build-cibw.yml`, `build-cibw-abi3.yml`

- [x] **Vendor-drift guard workflow (`check-vendor.yml` pattern)** -- llama.cpp's `check-vendor.yml` re-runs `scripts/sync_vendor.py` in CI and fails if the tree differs. cyllama's `git status` currently shows modified vendored headers (`mtmd.h`, `log.h`, `ggml-backend.h`, `common.h`) -- exactly the class of silent drift that caused the `GGML_MAX_NAME=128` ABI-match incident. Add `.github/workflows/check-thirdparty.yml` that runs the vendor sync step and diffs. Implemented as `manage.py check_vendor` subcommand + `.github/workflows/check-vendor.yml`

- [x] **Extend ccache to CPU cibw workflows** -- `hendrikmuhs/ccache-action@v1.2` now wired into `build-cibw.yml` + `build-cibw-abi3.yml` for Linux and macOS, keyed per-matrix-OS, with `CMAKE_{C,CXX}_COMPILER_LAUNCHER=ccache` passed through `CIBW_ENVIRONMENT_{LINUX,MACOS}`. Linux manylinux container installs ccache via `pyproject.toml`'s `before-all`. Windows (MSVC) ccache remains skipped pending debug-format / cmake-integration investigation

### Medium Priority

- [ ] **Path-filtered `push` / `pull_request` triggers** -- llama.cpp's `build-vulkan.yml:5-25`, `release.yml:10-28`, `editorconfig.yml:9-14` all auto-trigger on narrow `paths:` filters. cyllama is entirely `workflow_dispatch` today, so real regressions can ship to users. Auto-trigger CPU cibw on PRs touching `src/cyllama/**`, `scripts/manage.py`, `pyproject.toml`; keep GPU wheels on `workflow_dispatch`

- [ ] **Lightweight Python lint / type-check workflows** -- llama.cpp has `python-lint.yml`, `python-type-check.yml`, `python-check-requirements.yml`, `editorconfig.yml` using `runs-on: ubuntu-slim`, triggered only on `**/*.py` / config path changes, running in <1 min. Cheap pre-filter before the 40-minute wheel matrix. Add `.github/workflows/python-lint.yml` with `ruff check` and optionally `mypy` / `ty`

- [ ] **Composite actions for repeated toolchain setup** -- llama.cpp factors into `.github/actions/{windows-setup-cuda,linux-setup-vulkan,windows-setup-rocm,unarchive-tar,get-tag-name}/action.yml` and reuses them across `build-vulkan.yml`, `release.yml`, `build-cache.yml`. cyllama duplicates the Vulkan-SDK pwsh install (~15 lines) and a version-reading Python snippet (`build-cibw.yml:234-239`, `build-gpu-wheels.yml:664-670`). Extract `.github/actions/setup-vulkan-windows` and `.github/actions/get-version`

- [ ] **Reusable `workflow_call` smoke-test** -- cyllama's wheel-find + venv + import + inference block is duplicated across all three workflow files with minor variations. Wrap `scripts/run_wheel_test.py` in `.github/workflows/_smoke-test.yml` with `on: workflow_call` (inputs: `artifact-name`, `runs-on`, `run-inference`) and delete ~200 lines of duplication

### Wheel Coverage (additional backend variants)

Gap analysis vs. llama.cpp b8893 release assets. Ordered by effort/payoff.

- [x] **Windows CUDA 13.1** -- shipped as `build_cuda13_windows` job in `build-new-wheels.yml:44` (wheel name `cyllama-cuda13`, Jimver/cuda-toolkit@13.1.0)

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

## RAG Pipeline Integration

- [x] **Pipeline-integrated reranking + `RerankerProtocol`** -- `RerankerProtocol` shipped in `src/cyllama/rag/types.py`; `Reranker` inherits from it. `RAGConfig(rerank=True, rerank_top_k=20, reranker=<instance>)` wired into `RAGPipeline._retrieve` (used by `query`, `stream`, `retrieve`). Default `rerank=False` preserves the legacy path.

## Alternative vector-store backends

- [x] **Qdrant adapter (`cyllama/rag/stores/qdrant.py`)** -- Reference adapter shipped. `QdrantVectorStore` implements the seven `VectorStoreProtocol` methods against `qdrant_client.QdrantClient`; source dedup lives in per-point payload fields (`content_hash`, `source_label`, `indexed_at`), `is_source_indexed` is a filtered count, `get_source_by_label` is a scroll+count. Lazy-imported from `cyllama.rag.QdrantVectorStore` and `cyllama.rag.stores`. Optional `qdrant` dep group in `pyproject.toml`. Tests in `tests/test_rag_qdrant.py` (21 cases) skip when `qdrant-client` isn't installed. Chroma / LanceDB / pgvector can follow the same template.