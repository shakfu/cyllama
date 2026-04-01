# Build Options Analysis: Local vs CI vs Upstream llama.cpp

This document compares the build options used across three contexts:

1. **Local builds** -- developer machine via `make` / `manage.py`
2. **CI wheel builds** -- GitHub Actions via `build-gpu-wheels.yml` (the active workflow)
3. **Upstream llama.cpp** -- options documented in [llama.cpp/docs/build.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)

> **Note:** `build-gpu-wheels-cached.yml` is an experimental workflow not currently used in production. It has been synced to mirror `build-gpu-wheels.yml` (settings, flags, and cache keys), but `build-gpu-wheels.yml` remains the authoritative CI workflow.

## Build Flow Overview

Cyllama's build is a two-phase process:

1. **Phase 1 -- Build dependencies** (`manage.py build --deps-only`): Clones llama.cpp (and whisper.cpp, stable-diffusion.cpp), runs CMake to build them, and copies headers + libraries to `thirdparty/`.
2. **Phase 2 -- Build cyllama** (`uv pip install .` or `uv sync`): scikit-build-core invokes the top-level `CMakeLists.txt`, which finds the pre-built libraries in `thirdparty/` and links them into the Cython extension modules.

Phase 1 is where upstream llama.cpp CMake options matter. Phase 2 uses cyllama's own `CMakeLists.txt`, which reads the same `GGML_*` environment variables to determine which backend libraries to link and which system dependencies to find.

## Backend Support Matrix

### Currently Supported

| Backend | Env Var | Local | CI | Upstream |
|---------|---------|-------|----|----------|
| Metal (macOS) | `GGML_METAL` | Default ON on macOS | OFF (Linux CI) | Default ON on macOS |
| CUDA (NVIDIA) | `GGML_CUDA` | OFF by default | CUDA 12.4 | ON with `-DGGML_CUDA=ON` |
| Vulkan | `GGML_VULKAN` | OFF by default | Supported | ON with `-DGGML_VULKAN=ON` |
| HIP/ROCm (AMD) | `GGML_HIP` | OFF by default | ROCm 6.3 | ON with `-DGGML_HIP=ON` |
| SYCL (Intel) | `GGML_SYCL` | OFF by default | Supported | ON with `-DGGML_SYCL=ON` |
| OpenCL | `GGML_OPENCL` | OFF by default | Not built | ON with `-DGGML_OPENCL=ON` |
| BLAS (CPU) | `GGML_BLAS` | `--blas` / `GGML_BLAS_VENDOR` | Not built | ON with vendor selection |

### Upstream-Only (Not Supported in Cyllama)

| Backend | Upstream Flag | Notes |
|---------|--------------|-------|
| MUSA (Moore Threads) | `GGML_MUSA` | Chinese GPU vendor, niche |
| CANN (Ascend NPU) | `GGML_CANN` | Huawei NPU |
| ZenDNN (AMD EPYC) | `GGML_ZENDNN` | CPU-only optimization for EPYC |
| WebGPU | `GGML_WEBGPU` | In-progress upstream, requires Dawn |
| KleidiAI (Arm) | `GGML_CPU_KLEIDIAI` | Arm-specific CPU optimization |
| OpenVINO | -- | Intel inference runtime |

These are deliberately excluded as they serve niche hardware. If demand arises, they can be added following the same pattern as existing backends in `manage.py:get_backend_cmake_options()`.

## Detailed Option Comparison

### Core Build Options

| Option | Local | CI Workflows | Upstream Default | Notes |
|--------|-------|-------------|------------------|-------|
| `GGML_NATIVE` | Not set (ON) | `OFF` | ON | **Deliberate.** Local builds optimize for the developer's CPU. CI builds set OFF for portable wheels. |
| `BUILD_SHARED_LIBS` | `False` (static) | Depends on `link_mode` input | ON | Local default is static linking. CI supports both via workflow `link_mode` parameter. |
| `CMAKE_POSITION_INDEPENDENT_CODE` | `True` | `True` (inherited) | Not set | Required for static libs that get linked into shared Python extensions. |
| `LLAMA_CURL` | `False` | `False` (inherited) | Not documented | Disabled; cyllama doesn't need libcurl for model downloads. |
| `LLAMA_OPENSSL` | `True` | `True` (inherited) | Not documented | Enabled for HTTPS support in cpp-httplib (embedded server). |
| `LLAMA_BUILD_SERVER` | `False` | `False` (inherited) | ON | cyllama has its own Python server; upstream C++ server not needed. |
| `LLAMA_BUILD_TESTS` | `False` | `False` (inherited) | ON | No need to build llama.cpp's test suite. |
| `LLAMA_BUILD_EXAMPLES` | `False` | `False` (inherited) | ON | No need for example binaries. |
| `GGML_OPENMP` | ON (disable with `--no-openmp` or `GGML_OPENMP=0`) | Not set (ON) | ON | Forwarded to all three builders. `CMakeLists.txt` does `find_package(OpenMP)` on Linux. |
| `GGML_BACKEND_DL` | Not set (OFF) | Not set (OFF) | OFF | Dynamic backend loading at runtime. Not used. |

### CUDA-Specific Options

| Option | Local | CI (`build-gpu-wheels.yml`) | Upstream |
|--------|-------|---------------------------|----------|
| `CMAKE_CUDA_ARCHITECTURES` | User-provided or llama.cpp default | `"75"` | User-provided or auto |
| `CMAKE_CUDA_COMPILER` | Forwarded from env | Not set | Custom nvcc path |
| `GGML_CUDA_FORCE_MMQ` | Forwarded from env | Not set | Force quantized matrix kernels |
| `GGML_CUDA_FORCE_CUBLAS` | Forwarded from env | Not set | Force FP16 cuBLAS |
| `GGML_CUDA_PEER_MAX_BATCH_SIZE` | Forwarded from env | Not set | Multi-GPU peer access batch size (default: 128) |
| `GGML_CUDA_FA_ALL_QUANTS` | Forwarded from env | Not set | Compile all KV cache quant types |
| `GGML_NATIVE` | Not set (ON) | `OFF` | ON |
| `CUDA_PATH` | User's env | `/usr/local/cuda` | User's env |

The CUDA performance tuning flags (`GGML_CUDA_FORCE_MMQ`, `GGML_CUDA_FORCE_CUBLAS`, `GGML_CUDA_PEER_MAX_BATCH_SIZE`, `GGML_CUDA_FA_ALL_QUANTS`) are compile-time options that affect kernel selection in the built libraries. They are forwarded from environment variables to CMake by `get_backend_cmake_options()` in all three builders (llama.cpp, whisper.cpp, stable-diffusion.cpp). `CMAKE_CUDA_COMPILER` is similarly forwarded, allowing users with multiple CUDA toolkit installations to select a specific `nvcc`.

### HIP/ROCm-Specific Options

| Option | Local | CI | Upstream |
|--------|-------|----|----------|
| `CMAKE_HIP_ARCHITECTURES` | User-provided | `"gfx90a;gfx942;gfx1100"` | User-provided via `GPU_TARGETS` |
| `HIP_PATH` | User's env | `/opt/rocm` | From `hipconfig -R` |
| `GGML_HIP_ROCWMMA_FATTN` | Forwarded from env (`GGML_HIP_ROCWMMA_FATTN=1`) | Not set | rocWMMA flash attention |

### Strip/Debug Options

| Option | Local | CI | Notes |
|--------|-------|----|-------|
| `CFLAGS=-s` | Not set | Set | CI strips debug symbols for smaller wheels |
| `CXXFLAGS=-s` | Not set | Set | Same |
| `CMAKE_BUILD_TYPE` | Not set (Release via scikit-build-core) | Release | `pyproject.toml` sets `cmake.build-type = "Release"` |

### Visibility Options (set in `manage.py` for Phase 1)

| Option | Value | Purpose |
|--------|-------|---------|
| `CMAKE_CXX_VISIBILITY_PRESET` | `hidden` | Hide C++ symbols from exported API |
| `CMAKE_C_VISIBILITY_PRESET` | `hidden` | Hide C symbols from exported API |
| `CMAKE_VISIBILITY_INLINES_HIDDEN` | `True` | Hide inline function symbols |

These are set in `manage.py:LlamaCppBuilder.build()` and do not appear in upstream llama.cpp defaults. They reduce symbol pollution and wheel size.

## Dynamic Linking (`--dynamic` mode)

The `--dynamic` flag changes Phase 1 behavior:

1. **If a pre-built release asset exists** for the platform/backend combo, it downloads that archive (DLLs/`.so`/`.dylib` files) from llama.cpp GitHub releases.
2. **If no asset exists**, it falls back to building from source with `BUILD_SHARED_LIBS=ON`.

### Pre-built Release Asset Names

Defined in `manage.py:LlamaCppBuilder._release_asset_name()`:

| Platform | Backend | Asset Name |
|----------|---------|------------|
| macOS arm64 | Metal | `llama-{ver}-bin-macos-arm64.tar.gz` |
| macOS x86_64 | Metal | `llama-{ver}-bin-macos-x64.tar.gz` |
| Linux x86_64 | CPU | `llama-{ver}-bin-ubuntu-x64.tar.gz` |
| Linux x86_64 | Vulkan | `llama-{ver}-bin-ubuntu-vulkan-x64.tar.gz` |
| Linux x86_64 | CUDA | `None` (must build from source) |
| Linux x86_64 | HIP | `None` (must build from source) |
| Linux x86_64 | SYCL | `None` (must build from source) |
| Windows x64 | CPU | `llama-{ver}-bin-win-cpu-x64.zip` |
| Windows x64 | CUDA | `llama-{ver}-bin-win-cuda-{cuda_ver}-x64.zip` |

The Windows CUDA asset version defaults to `12.4` but can be overridden via the `LLAMACPP_CUDA_RELEASE` environment variable (e.g., `LLAMACPP_CUDA_RELEASE=13.1`). This only works if llama.cpp publishes a matching release asset. Users whose CUDA version has no pre-built asset should build from source (omit `--dynamic`).

## Design Decisions

### CUDA Architecture Coverage in CI -- Deliberate PTX-Only Strategy

The active CI workflow (`build-gpu-wheels.yml`) sets `CMAKE_CUDA_ARCHITECTURES="75"`. This is **deliberate**, not an oversight. The history:

1. **v0.2.1** (`GGML_NATIVE=ON`, no explicit architectures): llama.cpp auto-detected the CI runner's GPU (sm_52), producing a 99 MB `libggml-cuda.so` with SASS for sm_52 only + PTX fallback.

2. **v0.2.2 initial** (`GGML_NATIVE=OFF`, no explicit architectures): Without a GPU to detect, llama.cpp fell back to its full default architecture list, producing a 449 MB `libggml-cuda.so` (see `docs/dev/0.2.1-release-issue.md` for the full investigation).

3. **v0.2.2 fix attempt** (`GGML_NATIVE=OFF`, `CMAKE_CUDA_ARCHITECTURES="70-real;75-real;80-real;86-real;89-real;90"`): The `-real` suffix compiles SASS (native code) for each architecture. This produced a 762 MB `libggml-cuda.so` -- even larger, because SASS for 5 architectures is bigger than PTX for 5.

4. **v0.2.2 final** (`GGML_NATIVE=OFF`, `CMAKE_CUDA_ARCHITECTURES="75"`): A single architecture without `-real` produces **PTX only** for sm_75. The CUDA driver JIT-compiles the PTX to the user's actual GPU architecture at runtime (one-time startup cost). This keeps the wheel small while supporting all Turing-and-newer GPUs (RTX 20xx+, T4+). This is also forward-proof: CUDA 13.x drops pre-Turing (sm < 75) support, so sm_75 PTX is the correct minimum.

The tradeoff is clear: PTX-only means a JIT compilation penalty on first load (~seconds), but the wheel is ~45 MB compressed instead of ~300+ MB. This matches v0.2.1's behavior (which also relied on PTX JIT, just from sm_52 instead of sm_75) and was verified working on RTX 4060 (sm_89).

**This is not an issue to fix** -- it is a deliberate size/compatibility tradeoff. If future users report unacceptable JIT latency, the alternative is shipping multiple wheels per architecture or using `-real` for popular architectures (80, 86, 89) and accepting the size increase.

## Resolved Issues

The following gaps were identified during the initial analysis and have since been addressed:

| Issue | Resolution |
|-------|-----------|
| Cached workflow stale (`GGML_NATIVE`, architectures) | Synced `build-gpu-wheels-cached.yml` to match active workflow. Cache keys now hash the workflow file. |
| CUDA tuning flags not forwarded | `GGML_CUDA_FORCE_MMQ`, `GGML_CUDA_FORCE_CUBLAS`, `GGML_CUDA_PEER_MAX_BATCH_SIZE`, `GGML_CUDA_FA_ALL_QUANTS` forwarded in all three builders. |
| `CMAKE_CUDA_COMPILER` not exposed | Forwarded from env var in all three builders. |
| Windows CUDA dynamic asset hardcoded to 12.4 | Parameterized via `LLAMACPP_CUDA_RELEASE` env var (default `"12.4"`). |
| `GGML_BLAS` / `GGML_BLAS_VENDOR` not exposed | Added `--blas` CLI flag; vendor set via `GGML_BLAS_VENDOR` env var. Forwarded in llama.cpp and whisper.cpp builders. |
| `GGML_OPENMP` not exposed | Added `--no-openmp` CLI flag and `GGML_OPENMP` env var pass-through in all three builders. |
| `GGML_HIP_ROCWMMA_FATTN` not forwarded | Forwarded from env var in llama.cpp and whisper.cpp builders. |

## Remaining Recommendations

1. **Consider `GGML_BACKEND_DL`** -- dynamic backend loading could allow shipping a single wheel that loads GPU backends at runtime, though this is a significant architectural change.

## Environment Variable Flow

```
User sets GGML_CUDA=1
    |
    v
Makefile exports GGML_CUDA
    |
    v
manage.py reads via getenv("GGML_CUDA")
    |
    v
get_backend_cmake_options() returns {"GGML_CUDA": "ON"}
    |
    v
cmake_config() passes -DGGML_CUDA=ON to llama.cpp's CMake (Phase 1)
    |
    v
llama.cpp builds with CUDA backend, produces libggml-cuda.a
    |
    v
copy_backend_libs() copies libggml-cuda.a to thirdparty/llama.cpp/lib/
    |
    v
Phase 2: cyllama's CMakeLists.txt reads $ENV{GGML_CUDA}
    |
    v
find_package(CUDAToolkit REQUIRED) + link CUDA::cudart, CUDA::cublas, CUDA::cuda_driver
    |
    v
Static link libggml-cuda.a into Cython extension
```

## File Reference

| File | Role |
|------|------|
| `Makefile` | Top-level build entry points, backend flag defaults and exports |
| `scripts/manage.py` | Phase 1 builder: clones, configures, builds, copies llama.cpp/whisper.cpp/sd.cpp |
| `CMakeLists.txt` | Phase 2: links pre-built libs into Cython extensions |
| `pyproject.toml` | scikit-build-core config, cibuildwheel settings for CPU wheels |
| `.github/workflows/build-gpu-wheels.yml` | CI: GPU wheel builds (active workflow) |
| `.github/workflows/build-gpu-wheels-cached.yml` | CI: GPU wheel builds with caching (experimental, synced but not currently used) |
| `docs/build_backends.md` | User-facing backend build documentation |
