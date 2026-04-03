# CHANGELOG

All notable project-wide changes will be documented in this file. Note that each subproject has its own CHANGELOG.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Commons Changelog](https://common-changelog.org). This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Types of Changes

- Added: for new features.
- Changed: for changes in existing functionality.
- Deprecated: for soon-to-be removed features.
- Removed: for now removed features.
- Fixed: for any bug fixes.
- Security: in case of vulnerabilities.

---

## [0.2.3]

### Added

- **Wheel smoke tests in CI** - New `build-cibw2.yaml` workflow adds a `smoke_test` job that runs after wheel builds on Linux, macOS ARM, and Windows. Installs each platform's wheel in a clean venv and validates: all core imports (API, Cython extensions, agents, integrations, optional whisper/sd), plus a minimal inference call with the 1B test model. Test model is cached across runs via `actions/cache`. Release upload is now gated on smoke test success
- **Build-time backend config (`_backend.py`)** - `manage.py` now generates `src/cyllama/_backend.py` alongside `_build_info.py`, recording which GPU backends (CUDA, Metal, Vulkan, HIP, SYCL, OpenCL, BLAS) were enabled and their configuration options (CUDA architectures, compiler, tuning flags, HIP architectures, BLAS vendor, OpenMP). Queryable at runtime via `from cyllama import _backend`
- **Windows CUDA DLL discovery** - New `cyllama.utils.platform` module with `ensure_native_deps()`, called automatically before native extension loads. On Windows with a CUDA build, registers CUDA toolkit DLL directories via `os.add_dll_directory()` (env vars, PATH, and standard install locations). No-op on other platforms or non-CUDA builds. Guards placed in `llama/__init__.py`, `whisper/__init__.py`, and `sd/__init__.py` so all three backends are covered
- **CUDA performance tuning flags** - `GGML_CUDA_FORCE_MMQ`, `GGML_CUDA_FORCE_CUBLAS`, `GGML_CUDA_PEER_MAX_BATCH_SIZE`, and `GGML_CUDA_FA_ALL_QUANTS` are now forwarded from environment variables to CMake in all three builders (llama.cpp, whisper.cpp, stable-diffusion.cpp)
- **`CMAKE_CUDA_COMPILER` passthrough** - Users with multiple CUDA toolkit installations can set `CMAKE_CUDA_COMPILER=/path/to/nvcc` to select a specific compiler
- **Parameterized CUDA version for dynamic builds** - `_release_asset_name()` reads `LLAMACPP_CUDA_RELEASE` env var (default `"12.4"`) instead of hardcoding the CUDA version in the Windows dynamic download asset name
- **`--blas` CLI flag and `GGML_BLAS`/`GGML_BLAS_VENDOR` passthrough** - Enables explicit BLAS backend selection (OpenBLAS, Intel MKL, etc.) for llama.cpp and whisper.cpp builds. Vendor is set via `GGML_BLAS_VENDOR` env var
- **`--no-openmp` CLI flag and `GGML_OPENMP` passthrough** - Allows disabling OpenMP for Arm and embedded builds. Forwarded to all three builders
- **`GGML_HIP_ROCWMMA_FATTN` passthrough** - Enables rocWMMA-accelerated flash attention for AMD GPUs with supported hardware, forwarded to llama.cpp and whisper.cpp builders
- **Build options analysis** - Added `docs/dev/build-options.md` documenting the comparison of local, CI, and upstream llama.cpp build configurations
- **Advanced Build Options guide** - Added `docs/build_options.md` covering CUDA tuning, architecture targeting, BLAS/OpenMP configuration, dynamic linking, Windows builds, and complete environment variable reference
- **`ggml_backend_unload()` exposed in llama_cpp** - New Python-callable function to unload a dynamically-loaded backend by name (e.g. `"Vulkan"`, `"CUDA"`) and unregister it from the ggml backend registry. Only works with backends loaded via `ggml_backend_load_all()` (i.e. `GGML_BACKEND_DL` builds)
- **`GGML_CPU_ALL_VARIANTS` build option** - New `--cpu-all-variants` CLI flag and `GGML_CPU_ALL_VARIANTS=1` env var for `manage.py build`. Builds the ggml-cpu backend for multiple x86 ISAs (AVX, AVX2, AVX512, etc.) as separate shared libraries; the optimal one is selected at runtime. Automatically disables `GGML_NATIVE` (they are incompatible). Requires `GGML_BACKEND_DL` (set automatically by `build_shared`)
- **`GGML_BACKEND_DL` enabled for dynamic builds** - `LlamaCppBuilder.build_shared()` now passes `GGML_BACKEND_DL=ON` to CMake so that GPU backend shared libraries (Vulkan, CUDA, etc.) are built as loadable modules with the `ggml_backend_score` entry point. Without this, `auditwheel`-repaired wheels contained the backend `.so` files but ggml's runtime loader silently skipped them

### Fixed

- **Crash on failed image generation** - `save_outputs` now skips images with no valid data instead of raising `ValueError: Image has no valid data`. All image generation commands (`txt2img`, `img2img`, `inpaint`, `controlnet`) report a clean error and exit with code 1 when no images are generated successfully (e.g. due to CUDA OOM)
- **`--offload-to-cpu` now enables `free_params_immediately`** - When CPU offloading is requested, model parameters are freed from VRAM immediately after each component finishes, maximizing available VRAM for the next component
- **GPU backend cross-contamination in wheel builds** - All three dep builders (llama.cpp, whisper.cpp, stable-diffusion.cpp) now explicitly set unused GPU backends to `OFF` in CMake, preventing stale cache entries from enabling the wrong backend. This fixes an issue where the Vulkan wheel's `libstable-diffusion.a` could be compiled with `SD_USE_CUDA` instead of `SD_USE_VULKAN`
- **GPU backend discovery in dynamic wheels** - `ggml_backend_load_all()` in all three modules (llama, whisper, sd) now also scans the `cyllama_<variant>.libs/` directory where `auditwheel` places backend shared libraries. Previously only `cyllama/llama/` was searched, so GPU backends like `libggml-vulkan` were linked but never registered with ggml's backend registry
- **Incorrect ggml version reported for sd.cpp in dynamic builds** - `_write_build_info` reported sd.cpp's vendored ggml version (0.9.5) instead of the shared llama.cpp ggml version (0.9.8) when llama.cpp sources were unavailable (e.g. `--dynamic` builds that download pre-built binaries). Now falls back to whisper.cpp's ggml version (which matches llama.cpp's) and warns if neither is available
- **Cached GPU workflow synced with active workflow** - `build-gpu-wheels-cached.yml` was stale: CUDA job was missing `GGML_NATIVE=OFF` (risking SIGILL on user CPUs) and used the broad architecture list instead of PTX-only `"75"`. All four backends (CUDA, ROCm, SYCL, Vulkan) now match `build-gpu-wheels.yml` settings. Cache keys now hash both `scripts/manage.py` and the workflow file itself, preventing stale binaries after workflow-only changes
- **GPU offload reported as False in dynamic wheels** - `cmd_info()` in `__main__.py` called `llama_backend_init()` but never called `ggml_backend_load_all()` for the llama.cpp section, so the backend registry was empty and `llama_supports_gpu_offload()` returned `False`. The whisper and sd sections already had this call. Now all three sections load backends before querying them

## [0.2.2]

### Fixed

- **CUDA wheel size stability** - The `CMAKE_CUDA_ARCHITECTURES` passthrough added in the previous fix was compiling SASS for 5 architectures plus PTX for 1, producing a 762 MB `libggml-cuda.so`. Changed to `CMAKE_CUDA_ARCHITECTURES="75"` (PTX-only for sm_75/Turing), which lets the CUDA driver JIT-compile for the user's actual GPU at runtime. Supports Turing and newer GPUs (RTX 20xx+, T4+), forward-proof for CUDA 13.x which drops pre-Turing support
- **`GGML_NATIVE=OFF` moved from `manage.py` to CI workflows** - The portability flags were previously hardcoded in `manage.py`'s `get_backend_cmake_options()`, affecting local development builds. Now set exclusively in `CIBW_BEFORE_ALL_LINUX` and `CIBW_ENVIRONMENT_LINUX` for all four GPU backends (CUDA, ROCm, SYCL, Vulkan), keeping local builds native while ensuring CI wheels are portable

### Added

- **Automatic GitHub pre-release uploads** - Both `build-cibw` and `build-gpu-wheels` workflows now upload wheels to a GitHub pre-release tagged with the `pyproject.toml` version

### Changed

- **GitHub Release upload is now opt-in** - The `upload_release` job in all four wheel workflows (`build-cibw`, `build-cibw-cached`, `build-gpu-wheels`, `build-gpu-wheels-cached`) is gated behind a new "Upload wheels to GitHub Release" checkbox (default off), preventing accidental releases during test builds
- **Experimental cached CI workflows** - `build-cibw-cached` and `build-gpu-wheels-cached` variants cache `thirdparty/` build artifacts between runs using `actions/cache`, keyed on `scripts/manage.py` hash. On cache hit, `CIBW_BEFORE_ALL` skips the deps build (including shaderc for Vulkan), reducing CI time on unchanged dependencies

## [0.2.1]

### Added

- **Per-request sampler parameters in PythonServer** - `temperature`, `min_p`, and `seed` are now configurable per request via the OpenAI-compatible `/v1/chat/completions` endpoint instead of being hardcoded (seed 1337, temp 0.8, min_p 0.05)
- **`GenerationConfig.to_dict()`** - New method that converts a config to a dictionary with mutable values copied, replacing duplicated dict-building logic in `LLM.__init__` and `AsyncLLM._build_config()` (also fixes missing `main_gpu`, `split_mode`, `tensor_split` fields in the async variant)
- **Configurable TTS token IDs** - `TTSGenerator` now accepts `guide_token_id` and `audio_code_range` parameters instead of hardcoding token ID 198 and range 151672-155772 (OuteTTS defaults preserved)
- **Timeout support for `AsyncLLM.stream()`** - New `timeout` parameter (seconds) limits how long the consumer waits for each chunk, raising `asyncio.TimeoutError` if the model stalls
- **Memory-aware LRU cache for embeddings** - `Embedder` now accepts `cache_max_memory_bytes` to cap cache memory usage in addition to count-based eviction. `CacheInfo` includes a `memory_bytes` field

### Tests

- **Concurrent VectorStore access** - Tests verify cross-thread safety: shared instances correctly reject cross-thread use, and separate instances on the same file support concurrent reads and writes
- **DirectoryLoader symlink handling** - Tests cover symlinks to files, directories, broken symlinks, and symlinks pointing outside the base directory
- **Server request validation** - Tests for PythonServer with malformed inputs: missing messages, missing role/content keys, stop word truncation, unknown roles
- **Memory-aware LRU cache** - Unit tests for count-based eviction, memory-based eviction, memory tracking accuracy, and duplicate key handling
- **TextLoader errors parameter** - Tests for invalid and valid codec error handler names
- **`make leaks` RSS-growth leak detector** - New Makefile target runs `scripts/leak_check.py`, which exercises model load/unload and inference in a loop, measuring RSS per cycle and failing if growth exceeds a threshold (default 20%). Detects native memory leaks in Cython wrappers without the false-positive noise of `leaks --atExit` on CPython

### Changed

- **CI Python path auto-discovered in manylinux** - Replaced hardcoded `/opt/python/cp310-cp310/bin/python` in `build-gpu-wheels.yml` with runtime discovery (`ls -d /opt/python/cp3*/bin | head -1`), removing the need for a workflow input
- **Windows `get_lib_path()` searches more build configurations** - Now tries `RelWithDebInfo/`, `MinSizeRel/`, and `Debug/` before falling back, with a warning listing all searched paths
- **CI artifact naming deterministic** - Replaced `strategy.job-index` with `matrix.os` in `build-cibw.yaml` artifact names, preventing name changes when the matrix definition is modified

### Fixed

- **`TextLoader` validates `errors` parameter** - Invalid codec error handler names (e.g. `"invalid"`) now raise `ValueError` at construction time instead of producing a cryptic `LookupError` deep in file I/O
- **CUDA dynamic wheel build failed on missing `libggml-rpc.so`** - `ggml-rpc` was incorrectly listed as a required core library in `CMakeLists.txt`, but it's an optional RPC backend not always present in self-builds or release tarballs. Moved to optional; build warns instead of failing when absent
- **`LLAMACPP_DYLIB_DIR` validated when `WITH_DYLIB=ON`** - CMake now emits `FATAL_ERROR` if the directory does not exist or is empty, instead of silently defaulting to an absent `thirdparty/llama.cpp/dynamic`
- **Broken symlinks in `build_shared()` now logged** - Previously, symlinks with missing targets were silently skipped during shared library collection, potentially producing incomplete wheels. A warning now identifies the broken symlink and its target
- **`sed` package-rename validated in GPU wheel CI** - Each `sed` rename in `build-gpu-wheels.yml` is now followed by a `grep -q` check that fails the build if the pattern did not match, preventing wheels from shipping with the wrong package name
- **`LD_LIBRARY_PATH` directory validated in wheel repair** - `CIBW_REPAIR_WHEEL_COMMAND_LINUX` now checks that the dynamic lib directory exists before running `auditwheel`, instead of silently falling back to system paths
- **Chat message validation in `apply_chat_template()`** - `msg.get("role", "user")` replaced with explicit validation: messages must be dicts with a non-empty string `role` and a present `content` key. Raises `TypeError`/`ValueError` with the offending index
- **`AsyncLLM.stream()` producer task cleanup** - If the consumer raises before the producer starts or finishes, the producer task is now cancelled and awaited cleanly instead of being left dangling
- **GIL released during `whisper_full()` and `llama_encode()`** - Long-running C inference calls now release the GIL, allowing other Python threads to run concurrently during whisper transcription and encoder-decoder encoding
- **Stable diffusion callback exceptions now logged** - `_c_log_callback` and `_c_progress_callback` now log exceptions via `logging.warning` instead of silently swallowing them with bare `except: pass`
- **Overflow-safe image size calculations in stable_diffusion.pyx** - `SDImage.from_numpy()` and `SDImage.load()` now check for integer overflow before `malloc`, raising `OverflowError` instead of silently truncating or allocating undersized buffers
- **`*.a` files excluded from sdist** - `sdist.exclude` in `pyproject.toml` now filters out `thirdparty/*/lib/*.a` and `*.lib` files that were previously included via `sdist.include` of `thirdparty/*/lib` directories
- **`_release_url()` guarded against `None` asset name** - `_release_asset_name()` can return `None` for unsupported platform/backend combinations; `_release_url()` now returns `None` instead of interpolating `"None"` into the URL, and `download_release()` raises a clear `RuntimeError`
- **Invalid chat template names now warn** - When a template name that looks like an identifier (e.g. `"chatml"`) is not found in the model, a `UserWarning` is emitted before falling back to treating it as a raw Jinja string
- **`LLM.__del__` wrapped in try-except** - `close()` called from the destructor can raise during interpreter shutdown or partial initialization; exceptions are now suppressed
- **Shared metadata dicts in `VectorStore.add()`** - `metadata = [{}] * len(embeddings)` replaced with list comprehension to prevent all entries sharing the same dict reference
- **Metadata JSON-serialization validated upfront** - `VectorStore.add()` now validates all metadata dicts are JSON-serializable before starting the insert loop, giving a clear `ValueError` with the offending index
- **HybridStore FTS triggers documented for exclusive access** - `_create_fts_table()` docstring now documents that the INSERT/DELETE/UPDATE triggers require exclusive write access to the SQLite database
- **whisper.cpp CUDA backend not loading** - whisper.cpp was running inference on CPU only because `ggml_backend_load_all()` was never called before creating a `WhisperContext`. Added `ggml_backend_load_all()` to the whisper module (matching llama.cpp's existing pattern) and updated `whisper/cli.py` to call it before context creation
- **whisper.cpp backend detection in `cyllama info`** - The info command used outdated `KEY = 1` parsing for whisper backends, but modern whisper.cpp reports dynamically-loaded backends in `BACKEND : feature = val |` format. Fixed to parse the new format and call `ggml_backend_load_all()` before querying
- **stable-diffusion.cpp backend reporting in `cyllama info`** - Added `ggml_backend_load_all()` to the sd module so `cyllama info` correctly reports GPU backends without relying on llama.cpp having been initialized first
- **Duplicate ggml header conflicts** - `COMMON_INCLUDE_DIRS` included all thirdparty include paths (llama.cpp, whisper.cpp, stable-diffusion.cpp), causing `#pragma once` path-based include guard failures when identical `ggml.h`/`ggml-backend.h` headers existed under multiple directories. Cleaned up so `COMMON_INCLUDE_DIRS` only contains llama.cpp + base dirs; whisper and sd targets now get only their own + llama.cpp's include dirs via per-target configuration
- **Missing malloc NULL checks** - Added NULL checks after `malloc()` in whisper `tokenize()`, llama `tokenize()`, and llama `detokenize()` to raise `MemoryError` instead of crashing on allocation failure
- **Resource leaks in convenience functions** - `complete()`, `chat()`, and `batch_generate()` now use context managers to ensure GPU memory and model resources are freed after use
- **SQL injection risk in VectorStore** - Table names are now validated against `^[a-zA-Z_][a-zA-Z0-9_]*$` in `VectorStore.__init__()`, `VectorStore.open()`, and `HybridStore.__init__()` to prevent SQL injection via f-string interpolated identifiers
- **Silent build failures with missing libraries** - Dynamic library discovery in CMakeLists.txt now uses `FATAL_ERROR` instead of `WARNING` when required shared libraries are missing, preventing broken wheels. `copy_lib()` in manage.py now raises `FileNotFoundError` for required libraries instead of silently returning `False`
- **Missing cmake source directory validation** - `cmake_config()` now validates that `src_dir` exists before invoking cmake, raising `FileNotFoundError` instead of producing cryptic cmake errors
- **Unvalidated CI before-build scripts** - `before-build` and `before-all` scripts in pyproject.toml now verify that `manage.py` actually produced libraries before proceeding with wheel builds, preventing broken wheels from silent dependency build failures
- **Illegal instruction crash with pre-built dynamic wheels** - Dynamic builds linked against an arbitrary CPU variant (e.g. `libggml-cpu-alderlake.so`) chosen by non-deterministic GLOB order, which could contain instructions unsupported by the target CPU. Now links against the most portable variant (`x64` then `sse42`) for symbol resolution; ggml's runtime dispatcher still selects the optimal variant for the actual CPU
- **Callback use-after-free in `set_log_callback()`** - The Python callback passed to `llama_log_set()` was not stored, allowing it to be garbage collected while C code still held a pointer. Now kept alive via a module-level reference
- **Missing bounds check in `LlamaBatch.add()`** - Writing past batch capacity caused undefined behavior. Now raises `IndexError` when the batch is full
- **Async event loop race in `AsyncLLM.stream()`** - `asyncio.get_event_loop()` replaced with `asyncio.get_running_loop()` to avoid acquiring the wrong loop after `asyncio.to_thread()`
- **Missing model path validation** - `LLM.__init__()` now raises `FileNotFoundError` early if the model file doesn't exist, instead of failing deep in GPU initialization
- **Division by zero in `Embedder._mean_pool_manual()`** - Returns a zero vector when `n_tokens` is 0 instead of raising `ZeroDivisionError`
- **`VectorStore.open()` lost metric and vector_type** - Store configuration (metric, vector_type, dimension) is now persisted in a `{table}_meta` table and restored on reopen, instead of silently defaulting to cosine/float32. Backwards compatible with databases created before this change

## [0.2.0]

### Added

- **GPU variant packages on PyPI** - Dynamically linked GPU wheels published to PyPI for the first time:
  - `pip install cyllama-cuda12` -- NVIDIA GPU (CUDA 12.4, architectures: Volta through Hopper + PTX)
  - `pip install cyllama-rocm` -- AMD GPU (ROCm 6.3, requires glibc >= 2.35)
  - `pip install cyllama-sycl` -- Intel GPU (oneAPI SYCL 2025.3, requires glibc >= 2.35)
  - `pip install cyllama-vulkan` -- Cross-platform GPU (Vulkan, requires glibc >= 2.35)
  - All variants install the same `cyllama` Python package (same import, different backends)
  - Dynamic linking reduces wheel sizes to PyPI-publishable levels: CUDA 45MB, ROCm 51MB, Vulkan 28MB, SYCL 84MB

- **Dynamic Linking Support** - New `WITH_DYLIB=1` build mode links against pre-built llama.cpp shared libraries instead of building from source
  - Set `LLAMACPP_DYLIB_DIR=/path/to/release` to point at a pre-built release tarball
  - Shared libraries (`libllama.dylib`, `libggml*.dylib`, `libmtmd.dylib`) are copied alongside the extension for runtime resolution
  - Extension size drops from ~15 MB (static) to ~1.6 MB (dynamic) -- inference engine code is external
  - Static linking (`WITH_DYLIB=OFF`) remains the default and is unchanged

- **Dynamic whisper.cpp linking** - In `WITH_DYLIB` mode, `whisper_cpp` now reuses llama.cpp's ggml shared libraries instead of statically linking its own copy
  - `libwhisper.a` and `libcommon.a` (whisper-specific code) are still linked statically
  - ggml symbols resolve at runtime from the already-bundled `libggml*.dylib` in `cyllama/llama/` via `@loader_path/../llama`
  - Eliminates duplicate ggml code in the wheel

- **GPU wheel CI workflow** (`build-gpu-wheels.yml`) - Builds CUDA, ROCm, SYCL, and Vulkan wheels via `cibuildwheel` with both static and dynamic linking modes, with checkbox-based backend selection for parallel builds

- **LlamaContext memory management methods** - Added `memory_seq_rm`, `memory_seq_cp`, `memory_seq_keep`, `memory_seq_add`, `memory_seq_pos_min`, `memory_seq_pos_max` for direct KV cache sequence manipulation

### Changed

- **Eliminated all `common.h`/`libcommon` dependencies** - The Cython extension now uses only public C APIs (`llama.h`, `ggml.h`, `gguf.h`, `mtmd.h`). No internal llama.cpp C++ APIs are linked.
  - **Sampling** -- `CommonSampler` rewritten to build sampler chains via public `llama_sampler_chain_init()` + `llama_sampler_init_*()` with grammar rejection sampling
  - **Speculative decoding** -- `Speculative` class rewritten using `LlamaContext`, `LlamaSampler`, and memory management public APIs
  - **N-gram cache** -- `NgramCache` rewritten as pure Python (`dict`-based, binary-compatible save/load format)
  - **Download functions** -- `get_hf_file`, `download_model`, `list_cached_models`, `resolve_docker_model` rewritten using `urllib.request` (stdlib, no external deps)
  - **Batch helpers** -- `common_batch_add`/`common_batch_clear` inlined as direct array assignment
  - **Parameter conversion** -- `common_context_params_to_llama` inlined as field-by-field assignment

- **Pure Python JSON schema-to-grammar** - Replaced the C++ `json-schema-to-grammar.cpp`/`json-partial.cpp` compilation with a vendored pure Python implementation (`src/cyllama/utils/json_schema_to_grammar.py`). This eliminates the need for a llama.cpp source checkout during `build-dynamic`, removes `build_info_stub.cpp` and `json_schema.cpp` C++ helpers, and simplifies the CMake build for both static and dynamic linking modes. Agents import directly from `cyllama.utils` instead of going through the llama layer

- **Vendored sqlite-vector source** - sqlite-vector is now vendored in `thirdparty/sqlite-vector/` and built from source via CMake as part of the normal build, replacing the separate `manage.py` build step and pre-built binary copy. This ensures `vector.so` is always included in wheels (including `uv build` sdist->wheel pipeline) without needing `.gitignore` hacks or binary files in the source tree

- **Unified ggml 0.9.8 across all extensions** - Both static and dynamic builds now use llama.cpp's ggml 0.9.8 consistently for llama.cpp, whisper.cpp, and stable-diffusion.cpp (previously SD vendored ggml 0.9.5). In dynamic builds, all extensions share a single set of `libggml*.so`. In static builds, each extension links llama.cpp's ggml static libs (symbols are hidden, so no runtime conflicts). Set `SD_USE_VENDORED_GGML=1` (env var or CMake option) to link stable-diffusion against its own vendored ggml instead; available via `manage.py build --sd-vendored-ggml`

- **Unified GPU backend flags** - All components (llama.cpp, whisper.cpp, stable-diffusion.cpp) now use the same `GGML_*` environment variables for backend selection. Removed the separate `--sd-metal` / `SD_METAL` flag; `GGML_METAL` (and `--metal`) now applies to all components consistently

- **Documentation** - Moved Installation section above Quick Start in README; added feature summary list; updated docs to remove "book" language; fixed Python version requirements (3.10+); updated ROCm/Vulkan backend docs

### Removed

- **OpenSSL linking** -- Removed `find_package(OpenSSL)` from CMake. The embedded server (`cpp-httplib`) does not expose SSL configuration, and the absolute homebrew paths (`/opt/homebrew/opt/openssl@3/`) broke wheel portability on other machines
- **`common.h` wrapper layer** -- Deleted `common.pxd`, `common.pxi`, `sampling.pxi` (~2600 lines). These wrapped the internal `common_params` mega-struct which duplicated the public API configuration path (`LlamaContextParams`, `LlamaSampler`, etc.) and was unused by the Python API or agents
- **Internal C++ declaration files** -- Deleted `chat.pxd`, `log.pxd`, `sampling.pxd`, `download.pxd`, `ngram_cache.pxd`, `speculative.pxd`
- **`CommonParams`, `CommonParamsSampling`** and related wrapper classes -- Use `LlamaContextParams`, `LlamaSampler` instead
- **`CommonSampler`** -- Use `LlamaSampler` with `add_*` methods for the public API path

### Fixed

- **GPU dynamic wheel builds** - Fixed `--dynamic` mode silently downloading CPU-only pre-built binaries for CUDA, ROCm, and SYCL on Linux, where no matching pre-built GPU release assets exist. The build now falls back to compiling llama.cpp from source with `BUILD_SHARED_LIBS=ON` and collects the resulting shared libraries (including backend-specific libs like `libggml-cuda.so`) into the dynamic lib directory

- **GPU wheel duplication** - Shared libraries are no longer installed twice in Linux GPU wheels. CMake `install()` of dynamic libs is now macOS-only (`@loader_path`); on Linux, `auditwheel` handles vendoring into `<pkg>.libs/` via `LD_LIBRARY_PATH`. This roughly halves CUDA wheel size (81MB -> 45MB compressed)

- **GPU Wheel Import Error** - Fixed `.so` files installing to wrong directory (e.g. `cyllama_cuda12/llama/` instead of `cyllama/llama/`) by hardcoding install destination to `cyllama` instead of using `SKBUILD_PROJECT_NAME` which changed with the package rename

- **ROCm 6.3 Wheel Build** - Fixed multiple issues preventing ROCm wheels from building
  - Patched sqlite-vector to define `_GNU_SOURCE` for `strcasestr` on manylinux/AlmaLinux 8
  - Fixed auditwheel repair by targeting `manylinux_2_35` (matching ROCm 6.3 system requirements) and excluding transitive ROCm runtime libraries

- **Vulkan Wheel Build** - Restructured Vulkan job to build deps inside manylinux_2_28 container
  - Built `glslc` shader compiler from source (Google shaderc) to avoid glibc 2.29 dependency
  - Added `--plat manylinux_2_35_x86_64` for auditwheel repair (gcc-toolset-14 symbol compatibility)

- **SYCL Wheel Build** - Fixed Intel oneAPI installation, compiler selection, and SYCL kernel linking
  - Switched from direct RPM download (403 errors) to Intel YUM repository
  - Pinned to `intel-oneapi-dpcpp-cpp-2025.3` to resolve dependency conflicts
  - Set `CC=icx CXX=icpx` for SYCL compiler support
  - Added `-fsycl` as link-only flag (not compile flag) to preserve SYCL device code from `libggml-sycl.a` without forcing C files to compile as C++

- **CUDA Wheel Size** - Restored CUDA wheels to ~148 MB static (was ~682 MB); dynamic wheels are 45MB
  - Removed `CMAKE_CUDA_ARCHITECTURES` passthrough from deps build; llama.cpp defaults use virtual/PTX for older architectures, producing much smaller static libraries
  - Added free disk space step to prevent `strip` failing with "No space left on device"
  - Added `install.strip = true` in pyproject.toml (scikit-build-core default changed in recent versions)
  - Added `--plat manylinux_2_35_x86_64` to auditwheel repair for glibc compatibility

- **sqlite-vector Missing from Wheels** - Added CMake `install(FILES ...)` directive to explicitly include `vector.so`/`vector.dylib` (was excluded by `.gitignore` filtering in scikit-build-core)

- **argparse crash with `--sd-vendored-ggml`** - Fixed `TypeError: 'NoneType' object is not subscriptable` when `opt()` was called with `None` as the short option name, which caused `manage.py` to crash on Python 3.10+ in CI

- **`_build_info.py` reported wrong ggml version for stable-diffusion** -- `stable_diffusion_cpp_ggml_version` was read from SD's vendored source (0.9.5) instead of the actually-linked llama.cpp ggml (0.9.8). `_write_build_info` now uses llama.cpp's ggml version as canonical for all backends, unless `SD_USE_VENDORED_GGML=1` is set
- **`common.pxd` struct field mismatch** -- `flash_attn` (bool) corrected to `flash_attn_type` (enum) matching upstream `common_params`
- **`llama.pxd` missing field** -- Added `embeddings` bool to `llama_context_params` (was causing struct layout mismatch)

## [0.1.21]

### Added

- **CLI Info Command** - Added `python -m cyllama info` to show build and backend information
  - Displays cyllama version, Python version, and platform
  - llama.cpp: ggml version/commit, registered backends (MTL, CUDA, CPU, etc.), device enumeration with type and description, GPU offload/MMAP/MLOCK/RPC support
  - whisper.cpp: version, detected backends, CPU features
  - stable-diffusion.cpp: inferred backends, CPU features
  - Added `ggml_backend_reg_count()`, `ggml_backend_reg_names()`, `ggml_backend_dev_count()`, `ggml_backend_dev_info()` Python wrappers in `llama_cpp.pyx`
  - Added `ggml_backend_reg_name()` declaration to `ggml.pxd`
  - Also supports `python -m cyllama version`

- **Code Quality Tooling** - Added Makefile targets and configuration for linting, formatting, and type checking
  - `make lint`: ruff check with auto-fix
  - `make format`: ruff format
  - `make typecheck`: mypy on pure-Python modules (rag, utils)
  - `make qa`: runs lint, typecheck, format in sequence
  - Added `[tool.ruff]` config in `pyproject.toml`: line-length 120, per-file ignores for `__init__.py` re-exports, test relaxations, agents/api conditional imports
  - Added `[tool.mypy]` config: Python 3.10 target, ignore missing imports, `--follow-imports=skip` for Cython-dependent modules
  - Fixed all lint errors and type check errors across the codebase

### Fixed

- **Cross-Platform Wheel Build (cibuildwheel)** - Fixed numerous issues for CI wheel builds across all platforms
  - Fixed circular import in `cyllama/llama/__init__.py` by making all imports lazy via `__getattr__` (installed wheels have different import semantics than editable installs)
  - Fixed cibuildwheel test pythonpath: cleared `pythonpath=` override so installed wheel is used instead of source tree
  - Added `pytest-mock` and `pytest-asyncio` to cibuildwheel `test-groups` (uses `dev` dependency group via cibuildwheel v3.4.0 `test-groups` feature)
  - Fixed Linux manylinux2014 build: installed cmake/ninja via pip (yum cmake 2.8 too old), used `/opt/python/cp310-cp310/bin/python` (system Python 3.6 too old for `list[str]` syntax in manage.py), added `git` to container
  - Fixed Linux `lib64` vs `lib` issue: added `CMAKE_INSTALL_LIBDIR="lib"` to whisper.cpp and stable-diffusion.cpp cmake configs
  - Fixed Linux httplib undefined symbol: added `libcpp-httplib.a` to `--whole-archive` link group (GNU linker requires this for static lib symbol resolution)
  - Fixed macOS delocate OpenSSL conflict: excluded `libssl`/`libcrypto` from delocate bundling (Homebrew OpenSSL targets macOS 15.0, incompatible with wheel target 11.0)
  - Enabled `LLAMA_OPENSSL=ON` in manage.py for HTTPS support in cpp-httplib
  - Fixed Windows 32-bit: skipped `*-win32` and `*-manylinux_i686` builds (C++ backends are 64-bit only)
  - Fixed Windows `delvewheel` not found: added `pip install delvewheel` to Windows `before-build`
  - Fixed Windows ngram cache crash: skip `test_ngram_cache.py` on Windows (C++ divide-by-zero bug)
  - Fixed Windows temp file locking: moved `os.unlink` outside `NamedTemporaryFile` context in `test_tts_logic.py`
  - Fixed SQLite extension loading: added `hasattr` check for `enable_load_extension` in RAG store tests (CI Python may lack extension support)
  - Added `collect_wheels` job to `build-cibw.yaml` to combine all platform wheels into single downloadable artifact

## [0.1.20]

### Added

- **MkDocs Documentation** - Migrated documentation from Quarto to MkDocs with Material theme
  - Converted 24 `.qmd` files to standard Markdown in `docs/`
  - Created `mkdocs.yml` with full navigation structure
  - Added Makefile targets: `docs-serve`, `docs-build`, `docs-deploy`, `docs-clean`
  - Removed `docs/book/` Quarto directory
  - Updated all doc links in `README.md` to new paths
  - Updated outdated content: speculative decoding API, `flow_shift` move to SDSampleParams, version/date references

- **GPU Backend Wheel CI** - Added GitHub Actions workflow for building GPU-accelerated wheels
  - `build-gpu-wheels.yml`: Manual dispatch with backend selector (cuda-12.4, cuda-12.8, vulkan, all)
  - CUDA wheels built on Linux x86_64 with CUDA toolkit
  - Vulkan wheels built on Linux x86_64 and Windows x86_64
  - Wheels organized into PEP 503-compliant index, deployable to GitHub Pages
  - Users install GPU variants via `pip install cyllama --extra-index-url https://shakfu.github.io/cyllama/<backend>/`

- **PyPI Publishing** - Added Makefile targets and verified build pipeline for PyPI release
  - `make check`: Validate wheels with `twine check`
  - `make publish`: Upload to PyPI (runs check first)
  - `make publish-test`: Upload to TestPyPI (runs check first)
  - Added `license` field and updated author email in `pyproject.toml`
  - Added `pip install cyllama` instructions to `README.md`

- **LLM Response Cache** - Added optional response caching with TTL support for the `LLM` class
  - `LLM`: New `cache_size` parameter (default 0 = disabled) and `cache_ttl` parameter (seconds, None = no expiration)
  - `cache_info()`: Returns `ResponseCacheInfo` namedtuple with hits, misses, maxsize, currsize, ttl
  - `cache_clear()`: Clears cache and resets statistics
  - `cache_enabled`: Property to check if caching is active
  - LRU eviction when cache reaches capacity
  - TTL expiration support for time-based cache invalidation
  - Automatic cache bypass for random seed (`seed=-1`) since output is non-deterministic
  - Streaming responses are not cached (defeats streaming purpose)
  - Cache key includes only output-affecting parameters (temperature, top_k, top_p, etc.)
  - Exported `ResponseCacheInfo` from `cyllama`

- **Embedder LRU Cache** - Added optional embedding cache for repeated queries
  - `Embedder`: New `cache_size` parameter (default 0 = disabled)
  - `cache_info()`: Returns `CacheInfo` namedtuple with hits, misses, maxsize, currsize
  - `cache_clear()`: Clears cache and resets statistics
  - `cache_enabled`: Property to check if caching is active
  - LRU eviction when cache reaches capacity
  - Exported `CacheInfo` from `cyllama.rag`

- **Multi-GPU Support** - Added comprehensive multi-GPU configuration to high-level API
  - `GenerationConfig`: Added `main_gpu`, `split_mode`, and `tensor_split` parameters
    - `main_gpu`: Select primary GPU device index (default: 0)
    - `split_mode`: Control model splitting (0=NONE, 1=LAYER, 2=ROW with tensor parallelism)
    - `tensor_split`: Custom work distribution across GPUs (e.g., `[0.3, 0.7]` for 30%/70% split)
  - `LlamaModelParams.tensor_split`: Now writable (was read-only), with proper memory management
  - `LLM` class: Accepts all GPU parameters via kwargs or `GenerationConfig`
  - Verbose mode now prints GPU configuration details

### Fixed

- **llama.cpp Build Compatibility** - Fixed build failures after llama.cpp upstream sync
  - Added jinja header copy step in `manage.py` (`chat.h` now depends on `jinja/parser.h`, `jinja/runtime.h`, `jinja/caps.h`)
  - Added `libcpp-httplib.a` to build artifacts and CMake link list (`libcommon.a` now depends on cpp-httplib)
  - Added OpenSSL and macOS Security/CoreFoundation framework linking (required by cpp-httplib with SSL support)
  - Renamed `mtmd_get_audio_bitrate` to `mtmd_get_audio_sample_rate` in `mtmd.pxd`, `mtmd.pxi`, `multimodal.py`, tests, and examples (upstream rename)
  - Updated `model_alias` from `std::string` to `std::set<std::string>` in `common.pxd` and `common.pxi` (upstream type change)

- **stable-diffusion.cpp Build Compatibility** - Fixed build failure after stable-diffusion.cpp upstream sync
  - Moved `flow_shift` from `sd_ctx_params_t` to `sd_sample_params_t` in `stable_diffusion.pxd` (upstream struct change)
  - Moved `flow_shift` property from `SDParams` to `SDSampleParams` in `stable_diffusion.pyx`
  - Added `flow_shift` parameter to `SDContext.generate()` method
  - Updated CLI (`__main__.py`) to pass `flow_shift` via `generate()` instead of context params
  - Added missing `sd_ctx_params_t` fields: `enable_mmap`, `circular_x`, `circular_y`, `qwen_image_zero_cond_t`
  - Added corresponding properties to `SDParams`: `enable_mmap`, `circular_x`, `circular_y`, `qwen_image_zero_cond_t`

- **HIP/ROCm Backend Build** - Fixed multiple issues with HIP backend configuration (Issue #9)
  - Added missing environment variable handling for `GGML_HIP`, `GGML_SYCL`, `GGML_OPENCL`
  - Added HIP system library linking (`hip::host`, `roc::rocblas`, `roc::hipblas`)
  - Fixed HIP compilation flags: removed `hip::device` target which incorrectly added HIP compiler flags (`-x hip --offload-arch`) to Cython extensions
  - Added SYCL and OpenCL system library linking support
  - Added backend library checks for stable-diffusion extension (HIP, SYCL, OpenCL)

### Changed

- **Speculative Decoding API Sync** - Updated speculative decoding bindings for latest llama.cpp API
  - `speculative.pxd`: Rewrote declarations to match upstream: `common_speculative_init()` now takes `(params, ctx_tgt)` instead of two contexts, renamed `common_speculative_are_compatible()` to `common_speculative_is_compat()` (single context), renamed `common_speculative_gen_draft()` to `common_speculative_draft()`, removed `common_speculative_add_replacement_tgt_dft()` and `common_speculative_params` struct (uses `common_params_speculative` from `common.pxd`), added `common_speculative_begin()`, `common_speculative_accept()`, `common_speculative_print_stats()`
  - `speculative.pxi`: Updated `SpeculativeParams` to wrap `common_params_speculative` with `n_max`, `n_min`, `p_split`, `p_min` properties; updated `Speculative` class: `__init__()` takes `(params, ctx_target)`, renamed `are_compatible()` to `is_compat()`, renamed `gen_draft()` to `draft()`, removed `add_replacement()`, added `begin()`, `accept()`, `print_stats()` methods
  - `common.pxd`: Renamed `common_params_speculative.model` to `mparams_dft`, moved `lookup_cache_static`/`lookup_cache_dynamic` from `common_params` into `common_params_speculative`
  - `common.pxi`: Updated `lookup_cache_static`/`lookup_cache_dynamic` properties to access via `speculative` sub-struct, fixed `CommonParamsSpeculative.model` property to access `mparams_dft.path`

- **llama.cpp API Sync** - Updated wrappers for latest llama.cpp header changes
  - `llama.pxd`: Added `use_direct_io` field to `llama_model_params`, added `llama_params_fit_status` enum, updated `llama_params_fit()` return type and `margins` parameter, added `llama_model_n_embd_out()` function, added `llama_sampler_init_adaptive_p()` sampler, updated `llama_sampler_chain_get()` to non-const, added `llama_set_sampler()` function, updated `llama_split_path()` and `llama_split_prefix()` parameter and return types from `int` to `int32_t`, updated `use_direct_io` comment
  - `llama_cpp.pyx`: Added `use_direct_io` property to `LlamaModelParams` class, added `std_set` cimport for `model_alias` type change
  - `common.pxd`: Added `LLAMA_EXAMPLE_BATCHED`/`LLAMA_EXAMPLE_DEBUG` enum values, added `COMMON_SAMPLER_TYPE_ADAPTIVE_P` sampler type, added `adaptive_target`/`adaptive_decay`/`backend_sampling` to `common_params_sampling`, changed `fit_params_target` from `size_t` to `std_vector[size_t]`, added `use_direct_io`/`cache_prompt`/`sleep_idle_seconds`/`webui_config_json` to `common_params`, added `common_speculative_type` enum, added `type` field and reordered `common_params_speculative` fields to match upstream with new `ngram_size_n`/`ngram_size_m`/`ngram_min_hits` fields, updated `common_init_result` from struct to cppclass with methods and `common_init_result_ptr` typedef, updated `common_init_from_params()` return type to `common_init_result_ptr`, changed `model_alias` from `std_string` to `std_set[std_string]`
  - `common.pxi`: Updated `model_alias` property to return `set` of strings, setter accepts `str` or iterable
  - `mtmd.pxd`: Renamed `mtmd_get_audio_bitrate()` to `mtmd_get_audio_sample_rate()`
  - `mtmd.pxi`: Renamed `audio_bitrate` property to `audio_sample_rate`
  - `multimodal.py`: Renamed `audio_bitrate` to `audio_sample_rate` throughout
  - `ggml.pxd`: Added `use_ref` field to `ggml_cplan` struct, added `ggml_backend_cpu_set_use_ref()` function, `GGML_TYPE_COUNT` updated from 40 to 41
  - `ngram_cache.pxd`: Updated `common_ngram_cache_save()` and `common_ngram_cache_load()` filename parameters from `string &` to `const string &`
  - `test_chat.py`: Updated builtin templates list (added `exaone-moe`, `solar-open`)
  - `test_context.py`: Updated `get_state_size()` expected value for empty context (37 -> 17)
  - `test_params.py`: Updated `n_gpu_layers` default to `-1` (auto-detect), `model_alias` default to `set()`, `GGML_TYPE_COUNT` to 41
  - `test_mtmd.py`: Renamed `audio_bitrate` references to `audio_sample_rate`

- **stable-diffusion.cpp API Sync** - Updated wrappers for latest stable-diffusion.cpp header changes
  - `stable_diffusion.pxd`: Replaced `sd_easycache_params_t` with new `sd_cache_params_t` struct and `sd_cache_mode_t` enum (supports EASYCACHE, UCACHE, DBCACHE, TAYLORSEER, CACHE_DIT modes), updated `sd_img_gen_params_t` and `sd_vid_gen_params_t` to use new cache system, added `vae_tiling_params` to `sd_vid_gen_params_t`, updated `sd_get_default_scheduler()` signature (added `sample_method` parameter), updated `convert()` signature (added `convert_name` parameter), added `RES_MULTISTEP_SAMPLE_METHOD`/`RES_2S_SAMPLE_METHOD` to `sample_method_t`, added `KL_OPTIMAL_SCHEDULER`/`BONG_TANGENT_SCHEDULER` to `scheduler_t`, added `flash_attn` field to `sd_ctx_params_t`
  - `stable_diffusion.pyx`: Added `cache_mode`, `cache_threshold`, `cache_range` properties, kept backward-compatible `easycache_*` properties (now map to new cache system), updated `get_default_scheduler()` to accept optional `sample_method`, updated `convert_model()` to accept optional `convert_name`, added `RES_MULTISTEP`/`RES_2S` to `SampleMethod` enum, added `KL_OPTIMAL`/`BONG_TANGENT` to `Scheduler` enum, added `flash_attn` property to `SDParams`

### Deprecated

- **SDImgGenParams**: `easycache_enabled`, `easycache_threshold`, `easycache_range` properties deprecated in favor of new `cache_mode`, `cache_threshold`, `cache_range` properties

## [0.1.19]

### Changed

- **llama.cpp API Sync** - Updated wrappers for latest llama.cpp header changes
- LLAMACPP_VERSION to `b7442`, SDCPP_VERSION = `master-423-c3ad6a1`
  - `sampling.pxd/pxi`: Removed `grammar_first` parameter from `common_sampler_sample()` and `common_sampler_sample_and_accept_n()` functions
  - `llama.pxd`: Added `no_alloc` field to `llama_model_params`, added `llama_params_fit()`, `llama_max_tensor_buft_overrides()`, `llama_log_get()` functions
  - `llama_cpp.pyx`: Added `use_extra_bufts`, `no_host`, `no_alloc` properties to `LlamaModelParams` class
  - `common.pxd`: Updated `llama_example` enum (`LLAMA_EXAMPLE_MAIN` -> `LLAMA_EXAMPLE_COMPLETION`, added `LLAMA_EXAMPLE_CLI`, `LLAMA_EXAMPLE_FINETUNE`, `LLAMA_EXAMPLE_FIT_PARAMS`), added `user_sampling_config` to `common_params_sampling`, added `docker_repo`/`name` to `common_params_model`, added multiple new fields to `common_params` (`fit_params`, `fit_params_target`, `fit_params_min_ctx`, `show_timings`, `models_dir`, `models_preset`, `models_max`, `models_autoload`, `media_path`), updated filesystem utils (`fs_validate_filename` signature, added `fs_is_directory()`, renamed `fs_list_files()` to `fs_list()`)
  - `ggml.pxd`: Added `GGML_OP_TOP_K` enum value
  - `chat.pxd`: Added new chat format enum values (`COMMON_CHAT_FORMAT_GLM_4_5`, `COMMON_CHAT_FORMAT_MINIMAX_M2`, `COMMON_CHAT_FORMAT_KIMI_K2`, `COMMON_CHAT_FORMAT_QWEN3_CODER_XML`, `COMMON_CHAT_FORMAT_APRIEL_1_5`, `COMMON_CHAT_FORMAT_XIAOMI_MIMO`, `COMMON_CHAT_FORMAT_PEG_SIMPLE`, `COMMON_CHAT_FORMAT_PEG_NATIVE`, `COMMON_CHAT_FORMAT_PEG_CONSTRUCTED`)
  - `mtmd.pxd`: Added `warmup` field to `mtmd_context_params`
  - `log.pxd`: Added `LOG_LEVEL_DEBUG`, `LOG_LEVEL_INFO`, `LOG_LEVEL_WARN`, `LOG_LEVEL_ERROR`, `LOG_LEVEL_OUTPUT` constants, added `common_log_flush()` function
  - `test_params.py`: Updated tests for new default values (`n_ctx` 4096->0, new `common_params_model` fields)

- **stable-diffusion.cpp API Sync** - Updated wrappers for latest stable-diffusion.cpp header changes
  - `stable_diffusion.pxd`: Updated `prediction_t` enum (removed `DEFAULT_PRED`, renamed `SD3_FLOW_PRED` to `FLOW_PRED`), added `sd_embedding_t` and `sd_lora_t` structs, updated `sd_ctx_params_t` (removed `lora_model_dir`/`embedding_dir`, added `embeddings`/`embedding_count`), added `custom_sigmas`/`custom_sigmas_count` to `sd_sample_params_t`, added `loras`/`lora_count` to `sd_img_gen_params_t` and `sd_vid_gen_params_t`, added `tile_size` parameter to `new_upscaler_ctx()`, added `sd_commit()` and `sd_version()` functions
  - `stable_diffusion.pyx`: Updated `Prediction` enum (removed `DEFAULT`, renamed `SD3_FLOW` to `FLOW`), removed `lora_model_dir`/`embedding_dir` from `SDContextParams`, added `tile_size` parameter to `SDUpscaler`, removed `lora_model_dir` from `text_to_image()` function

### Removed

- **SDContextParams**: Removed `lora_model_dir` and `embedding_dir` properties (upstream API change - LoRAs and embeddings now specified per-generation via new struct fields)
- **Prediction enum**: Removed `DEFAULT` member, renamed `SD3_FLOW` to `FLOW` (upstream enum change)
- **CommonSampler**: Removed `grammar_first` parameter from `sample()` and `sample_and_accept_n()` methods (upstream API change)

### Changed (Build System)

- **Build System Consolidation** - Unified build management in `scripts/manage.py`
  - Added `info` subcommand - Shows version info (tag/commit) for llama.cpp, whisper.cpp, stable-diffusion.cpp, sqlite-vector
    - `--snapshot` / `-s` option: Commits and pushes with dependency version info in commit message
  - Added `download` subcommand - Downloads models from HuggingFace (llama, whisper)
  - Added `bins` subcommand - Builds llama.cpp CLI binaries (llama-cli, llama-server, etc.)
  - Added `bench` subcommand - Runs performance benchmarks (prefill/decode speed)
  - Added `profile` subcommand - Profiles cyllama operations using cProfile with selectable targets:
    - `--tokenization`, `--inference`, `--logits`, `--batch`, `--properties`, `--all`
    - Saves `.prof` files for analysis with snakeviz or pstats
  - Added `bump` subcommand - Semantic version bumping with git tag creation
    - Default: patch increment (`0.1.18` -> `0.1.19`)
    - `--minor` / `-m`: minor increment (`0.1.18` -> `0.2.0`)
    - `--major` / `-M`: major increment (`0.1.18` -> `1.0.0`)
    - `--dry-run` / `-n`: preview changes without modifying files
    - Updates `pyproject.toml` and `src/cyllama/__init__.py`, commits, tags, and pushes
  - Added `--sd-metal` / `-M` build option for experimental stable-diffusion.cpp Metal support
  - Added backend configuration for whisper.cpp (`GGML_*` env vars) and stable-diffusion.cpp (`SD_*` env vars)
  - Added mtmd (multimodal) header copying for llama.cpp builds
  - Converted `scripts/setup2.sh` to thin wrapper delegating to `manage.py`

- **Stable Diffusion Metal Backend** - Fixed Metal support configuration
  - SD Metal disabled by default due to missing `GGML_OP_DIAG_MASK_INF` in ggml-metal
  - Use `SD_METAL=1` environment variable to opt-in (experimental)
  - SD extension now links against its own ggml libraries instead of llama.cpp's

### Deprecated

The following scripts are now superseded by `manage.py` subcommands:

- `scripts/info.sh` -> `manage.py info`
- `scripts/snap.sh` -> `manage.py info --snapshot`
- `scripts/bump.sh` -> `manage.py bump`
- `scripts/download-ggml-model.sh` -> `manage.py download --whisper`
- `scripts/build-llama-bins.sh` -> `manage.py bins`
- `scripts/benchmark.py` -> `manage.py bench`
- `scripts/*_profile.py`, `scripts/*_benchmark.py` -> `manage.py profile`

## [0.1.18]

### Changed

- **Build System Improvements** ([@xxnuo](https://github.com/xxnuo))
  - Added parallel build support (`--parallel` flag) for faster compilation
  - Fixed llama.cpp build targets to include backend-specific libraries (ggml-metal, ggml-cuda, etc.)
  - Backend libraries are now properly built before being copied

- **Stable Diffusion Module Renamed** - `cyllama.stablediffusion` renamed to `cyllama.sd`
  - All imports should now use `from cyllama.sd import ...`
  - Old module name deprecated

- **CLI Restructured with Subcommands** - Complete CLI overhaul for `python -m cyllama.sd`
  - `txt2img` (alias: `generate`) - Text to image generation
  - `img2img` - Image to image transformation with `--init-img` and `--strength`
  - `inpaint` - Inpainting with `--mask` (white areas = inpaint region)
  - `controlnet` - ControlNet guided generation with `--control-net`, `--control-image`, `--canny`
  - `video` - Video generation (text-to-video, image-to-video, frame interpolation)
  - `upscale` - ESRGAN upscaling with `--repeats` for multiple passes
  - `convert` - Model format conversion with quantization
  - `info` - System information and available options

- **Model Loading Flexibility** - `--model` is now optional when `--diffusion-model` is provided
  - Supports split model architectures (FLUX, SD3, etc.)
  - Either `--model` or `--diffusion-model` required, both accepted

- **Cleaner Public API** - Slimmed down `cyllama` namespace
  - Low-level bindings no longer exported at top level (use `from cyllama.llama.llama_cpp import ...`)
  - `apply_chat_template`, `get_chat_template` moved to `cyllama.api`
  - `agents`, `utils` modules now require explicit import
  - Reduces namespace pollution and clarifies API boundaries

### Removed

- **Top-level Low-Level Exports** - The following are no longer exported from `cyllama`:
  - All `Llama*` classes (LlamaModel, LlamaContext, etc.) - use `from cyllama.llama.llama_cpp import ...`
  - `ggml_*` functions - use `from cyllama.llama.llama_cpp import ...`
  - `json_schema_to_grammar` - use `from cyllama.llama.llama_cpp import ...`
  - `GGUFContext`, `NgramCache`, `Speculative*` - use `from cyllama.llama.llama_cpp import ...`
  - `download_model`, `list_cached_models` - use `from cyllama.llama.llama_cpp import ...`
  - `apply_chat_template`, `get_chat_template` - use `from cyllama.api import ...`
  - `stream_complete_async` - use `complete_async` with streaming
  - `MemoryEstimate` - returned by functions, not constructed directly
  - `agents`, `utils`, `mtmd` modules - import explicitly when needed

### Added

- **New SDContextParams Properties**
  - `clip_vision_path` - CLIP vision model path
  - `llm_path` - LLM text encoder path (FLUX2/Qwen)
  - `llm_vision_path` - LLM vision encoder path
  - `taesd_path` - TAESD model for fast preview
  - `control_net_path` - ControlNet model path
  - `photo_maker_path` - PhotoMaker model path
  - `high_noise_diffusion_model_path` - High-noise model (Wan2.2 MoE)
  - `tensor_type_rules` - Mixed precision rules (e.g., `"^vae\\.=f16"`)
  - `sampler_rng_type` - Separate RNG type for sampler
  - `lora_apply_mode` - LoRA application mode (auto, immediately, at_runtime)
  - `keep_clip_on_cpu`, `keep_vae_on_cpu`, `keep_control_net_on_cpu` - Memory optimization flags
  - `diffusion_conv_direct`, `vae_conv_direct` - Direct convolution options
  - `tae_preview_only` - Use TAESD only for preview
  - `flow_shift` - Flow shift parameter (SD3.x/Wan)
  - `chroma_use_dit_mask`, `chroma_use_t5_mask`, `chroma_t5_mask_pad` - Chroma model options

- **New SDSampleParams Properties**
  - `slg_scale`, `slg_layer_start`, `slg_layer_end` - Skip Layer Guidance (SLG) parameters
  - `img_cfg_scale` - Image CFG scale for inpainting
  - `distilled_guidance` - Distilled guidance for FLUX models
  - `shifted_timestep` - Shifted timestep for NitroFusion models

- **New SDImageGenParams Properties**
  - `vae_tiling_enabled`, `vae_tile_size`, `vae_tile_overlap` - VAE tiling for large images
  - `easycache_enabled`, `easycache_threshold`, `easycache_range` - EasyCache acceleration
  - `control_strength` - ControlNet strength
  - `auto_resize_ref_image` - Auto-resize reference images
  - `set_mask_image()` - Method to set inpainting mask

- **New Enums**
  - `Prediction.FLUX2_FLOW` - FLUX2 flow matching prediction type
  - `LoraApplyMode` - LoRA application modes (AUTO, IMMEDIATELY, AT_RUNTIME)
  - `PreviewMode` - Preview modes (NONE, PROJ, TAE, VAE)

- **Enhanced text_to_image() Function** - New parameters:
  - `taesd_path`, `control_net_path` - Additional model paths
  - `eta`, `slg_scale` - Sampler parameters
  - `vae_tiling` - Enable VAE tiling
  - `offload_to_cpu`, `keep_clip_on_cpu`, `keep_vae_on_cpu` - Memory optimization
  - `diffusion_flash_attn` - Flash attention flag

- **Enhanced SDContext.generate() Method** - New parameters:
  - `mask_image` - Mask for inpainting
  - `control_image`, `control_strength` - ControlNet parameters
  - `eta`, `slg_scale` - Sampler parameters
  - `vae_tiling` - Enable VAE tiling

- **CLI Options** - Comprehensive CLI with 50+ options:
  - Memory: `--offload-to-cpu`, `--clip-on-cpu`, `--vae-on-cpu`, `--control-net-cpu`
  - Performance: `--diffusion-fa`, `--diffusion-conv-direct`, `--vae-conv-direct`
  - Guidance: `--slg-scale`, `--skip-layer-start`, `--skip-layer-end`, `--guidance`, `--img-cfg-scale`
  - VAE tiling: `--vae-tiling`, `--vae-tile-size`, `--vae-tile-overlap`
  - Preview: `--preview`, `--preview-path`, `--preview-interval`, `--preview-noisy`
  - Chroma: `--chroma-disable-dit-mask`, `--chroma-enable-t5-mask`, `--chroma-t5-mask-pad`
  - Video: `--video-frames`, `--fps`, `--init-img`, `--end-img`

### Fixed

- **stable-diffusion.cpp API Compatibility** - Updated bindings for latest upstream changes:
  - `get_num_physical_cores()` renamed to `sd_get_num_physical_cores()`
  - `sd_preview_cb_t` callback signature updated with `void* data` parameter
  - `sd_set_preview_callback()` updated to 6 parameters
  - `qwen2vl_path` renamed to `llm_path`
  - `qwen2vl_vision_path` renamed to `llm_vision_path`

## [0.1.17]

### Added

- **RAG Support Phase 1: Core Embedding API** - Text embedding generation using llama.cpp
  - `Embedder` class - Generate vector embeddings from text using GGUF models
  - `embed()` - Embed a single text string
  - `embed_batch()` - Embed multiple texts efficiently
  - `embed_documents()` - Embed documents with optional progress tracking
  - `embed_with_info()` - Get embedding with token count metadata
  - `embed_iter()` - Generator for memory-efficient batch embedding
  - Pooling strategies: `mean`, `cls`, `last`, `none`
  - L2 normalization (optional, enabled by default)
  - Context manager support for proper resource cleanup
  - Data classes: `EmbeddingResult`, `SearchResult`, `Document`, `Chunk`
  - 22 unit tests in `tests/test_rag_embedder.py`

- **sqlite-vector Build Support** - Build system integration for sqlite-vector extension
  - `scripts/setup.sh` - Added `get_sqlitevector()` function to build sqlite-vector
  - `scripts/manage.py` - Added `SqliteVectorBuilder` class with `--sqlite-vector` flag
  - Extension installed to `src/cyllama/rag/` for runtime inclusion in wheel

- **RAG Support Phase 2: VectorStore** - SQLite-based vector storage with sqlite-vector
  - `VectorStore` class - High-performance vector similarity search
  - `add()`, `add_one()` - Add embeddings with text and optional metadata
  - `search()` - Find k most similar embeddings with threshold filtering
  - `get()`, `get_vector()` - Retrieve stored embeddings by ID
  - `delete()`, `clear()` - Remove embeddings from store
  - `quantize()` - Quantize vectors for 4-5x faster search on large datasets
  - `preload_quantization()` - Preload quantized data into memory
  - `VectorStore.open()` - Open existing database from disk
  - Distance metrics: `cosine`, `l2`, `dot`, `l1`, `squared_l2`
  - Vector types: `float32`, `float16`, `int8`, `uint8`
  - Context manager support for automatic cleanup
  - 49 unit tests in `tests/test_rag_store.py`

- **RAG Support Phase 3: Text Processing** - Document splitting and loading utilities
  - `TextSplitter` class - Recursive character splitting with configurable chunk size/overlap
  - `TokenTextSplitter` - Token-based splitting using custom tokenizer functions
  - `MarkdownSplitter` - Markdown-aware splitting respecting headers, code blocks, lists
  - `TextLoader` - Load plain text files
  - `MarkdownLoader` - Load Markdown with optional YAML frontmatter parsing
  - `JSONLoader` - Load JSON with configurable text key and jq-like filtering
  - `JSONLLoader` - Load JSON Lines with lazy loading support
  - `DirectoryLoader` - Batch load files from directories with glob patterns
  - `PDFLoader` - Load PDF files using docling (optional `pdf` dependency group)
  - `load_document()` - Convenience function for loading single files
  - `load_directory()` - Convenience function for loading directories
  - 72 unit tests in `tests/test_rag_splitter.py` and `tests/test_rag_loaders.py`

- **RAG Support Phase 4: RAG Pipeline** - Complete retrieval-augmented generation
  - `RAGConfig` dataclass - Configuration for retrieval and generation settings
    - `top_k`, `similarity_threshold` - Retrieval parameters
    - `max_tokens`, `temperature` - Generation parameters
    - `prompt_template`, `context_separator`, `include_metadata` - Prompt formatting
    - Validation for all configuration values
  - `RAGResponse` dataclass - Response wrapper with sources and statistics
    - `text`, `sources`, `stats`, `query` attributes
    - `to_dict()` method for JSON serialization
  - `RAGPipeline` class - Orchestrates retrieval and generation
    - `query(question, config=None)` - Full RAG query with response
    - `stream(question, config=None)` - Stream response tokens
    - `retrieve(question, config=None)` - Retrieve documents without generation
    - Customizable prompt templates with `{context}` and `{question}` placeholders
  - `RAG` class - High-level interface with sensible defaults
    - `add_texts(texts, metadata=None, split=True)` - Add text to knowledge base
    - `add_documents(paths, split=True)` - Load and add files
    - `add_document(document, split=True)` - Add single Document object
    - `query(question, config=None)` - Query knowledge base
    - `stream(question, config=None)` - Stream response tokens
    - `retrieve(question, config=None)` - Retrieve without generation
    - `search(query, k=5, threshold=None)` - Direct vector search
    - Context manager support for proper resource cleanup
  - 25 unit tests in `tests/test_rag_pipeline.py`

- **RAG Support Phase 5: Advanced Features** - Async, agent integration, hybrid search
  - `AsyncRAG` class - Async wrapper for non-blocking RAG operations
    - `add_texts()`, `add_documents()` - Async document ingestion
    - `query()`, `stream()`, `retrieve()` - Async query methods
    - `search()`, `clear()` - Async utility methods
    - Async context manager support
  - `create_rag_tool(rag)` - Create agent tools from RAG instances
    - Compatible with ReActAgent, ConstrainedAgent, ContractAgent
    - Configurable name, description, top_k, and score inclusion
    - Auto-generates proper JSON schema for tool parameters
  - `Reranker` class - Cross-encoder reranking for improved quality
    - `score(query, document)` - Score query-document pairs
    - `rerank(query, results, top_k)` - Rerank search results
    - Lazy model loading for efficiency
  - `HybridStore` class - Combined FTS5 + vector search
    - SQLite FTS5 integration with automatic triggers
    - Reciprocal Rank Fusion (RRF) for combining results
    - Configurable alpha for vector vs FTS weighting
    - `search(query_embedding, query_text, k, alpha)` - Hybrid search
  - `async_search_knowledge()` - Async helper function
  - 37 unit tests in `tests/test_rag_advanced.py`

## [0.1.16]

### Added

- **Chat Template Support** - Integrated llama.cpp's built-in chat template system
  - `apply_chat_template(messages, model_path, template=None)` - Format chat messages using model's template
  - `get_chat_template(model_path, template_name=None)` - Retrieve template string from model metadata
  - `LLM.chat(messages, config=None, stream=False, template=None)` - Chat with template formatting
  - `LLM.get_chat_template(template_name=None)` - Get template from loaded model
  - `AsyncLLM.chat(messages, config=None, template=None)` - Async chat with templates
  - `AsyncLLM.get_chat_template(template_name=None)` - Get template from async LLM
  - `chat()` function now supports `template` parameter for explicit template selection
  - Supports all llama.cpp built-in templates: llama2, llama3, llama4, chatml, mistral, phi3, phi4, deepseek, gemma, falcon3, command-r, vicuna, zephyr, and many more
  - Automatic fallback to simple `User:/Assistant:` format when no template available
  - 8 new tests in `tests/test_chat.py`

- **Async API Support** - Full async/await support for text generation
  - `AsyncLLM` class - Async wrapper around `LLM` with `async with` context manager support
  - `complete_async()` - Async convenience function for one-off completions
  - `chat_async()` - Async chat-style generation
  - `stream_complete_async()` - Async streaming generator
  - Uses `asyncio.to_thread()` to avoid blocking the event loop during inference
  - Lock-based serialization prevents concurrent access issues
  - All async functions support same kwargs as sync versions

- **Async Agent Support** - Async wrappers for agent execution
  - `AsyncReActAgent` - Async wrapper for ReActAgent
  - `AsyncConstrainedAgent` - Async wrapper for ConstrainedAgent
  - `run_agent_async()` - Helper function to run any agent asynchronously
  - Async streaming via `async for event in agent.stream(task)`
  - Suitable for use in FastAPI, aiohttp, and other async frameworks

- **Response Class** - Structured response object for all generation functions
  - `Response` dataclass with `text`, `stats`, `finish_reason`, `model` attributes
  - Backward compatible via `__str__` - existing code using string operations continues to work
  - Full string protocol support: `__eq__`, `__len__`, `__iter__`, `__contains__`, `__add__`, `__radd__`
  - `to_dict()` method for dictionary serialization
  - `to_json(indent=None)` method for JSON serialization
  - `stats` contains `GenerationStats` with timing and token metrics when available
  - Returned by: `complete()`, `chat()`, `LLM()`, `LLM.chat()`, `batch_generate()`
  - Async support: `complete_async()`, `chat_async()`, `AsyncLLM()`, `AsyncLLM.chat()`
  - 19 new tests in `tests/test_response.py`

### Changed

- **Framework Integrations Updated for Response** - OpenAI and LangChain integrations now properly use Response objects
  - `OpenAICompatibleClient` uses `response.stats` for accurate token counts instead of re-tokenizing
  - `OpenAICompatibleClient` uses `response.finish_reason` for completion finish reason
  - `CyllamaLLM` (LangChain) now includes generation stats in `generation_info`:
    - `prompt_tokens`, `completion_tokens`, `total_time_seconds`, `tokens_per_second`, `finish_reason`
  - Internal `_call_internal()` method added to LangChain integration returning `Response` objects
  - Both integrations maintain backward compatibility with their respective framework APIs

- **LLM Class Direct Parameters** - `LLM` class now accepts generation parameters directly
  - Can now use `LLM("model.gguf", temperature=0.9, max_tokens=100)` instead of requiring `GenerationConfig`
  - Supports three patterns: direct kwargs, explicit `config=`, or config with kwargs overrides
  - Maintains full backward compatibility with existing `config=GenerationConfig(...)` usage
  - Validation still runs through `GenerationConfig.__post_init__`

- **GitHub Actions Workflow Fixes** - Fixed wheel collection in CI workflows
  - Fixed `build-matrix.yml` wheel Python version tagging using `--python` and `--python-preference only-system`
  - Root cause: `.python-version` file caused `uv build` to ignore `setup-python` configured interpreter
  - Updated `build-wheels.yml` and `publish-wheels.yml` to use consistent patterns with `uv` and `make sync`
  - Updated `download-artifact` to v5 and removed problematic `merge-multiple: true`
  - Added proper wheel collection using `find` command to flatten artifact directories
  - Simplified `ci.yml` with consistent runner versions and build patterns

- **CI/CD Automation Enabled** - Full CI pipeline now runs on push/PR
  - Enabled push triggers for `main` and `dev` branches
  - Enabled pull request triggers for `main` and `dev` branches
  - Added code coverage reporting with `pytest-cov` (XML and terminal output)
  - Coverage report uploaded as artifact for ubuntu-22.04/py3.12 job
  - Added mypy type checking job (runs separately, continues on error initially)
  - Added `mypy>=1.13.0` to dev dependencies

- **Test Suite Cleanup** - Removed redundant and obsolete test files
  - Deleted `scratch.py` (not a test file), `test_api.py` (redundant with `test_simple.py`)
  - Deleted `test_highlevel.py` and `test_common.py` (entirely skipped, tested deprecated/non-existent APIs)
  - Consolidated 5 small mserver test files into `test_mserver_embedded.py`
  - Improved `conftest.py` with LLM fixtures that ensure proper resource cleanup
  - Added `llm`, `llm_deterministic`, `llm_shared` fixtures for model instance management
  - Added `fast_config`, `deterministic_config` fixtures for common generation configs
  - Added custom pytest markers: `@pytest.mark.slow`, `@pytest.mark.requires_model`, `@pytest.mark.gpu`
  - Test count: 862 passed, 29 skipped (reduced from 863/34 by removing redundant tests)

## [0.1.15]

### Changed

- **Build System Migration to scikit-build-core** - Replaced setuptools with modern CMake-based build
  - Migrated from `setup.py` + `setuptools` to `scikit-build-core` + `CMakeLists.txt`
  - Added `CMakeLists.txt` for building Cython extensions via CMake
  - Added `scripts/cmake/` directory with vendored `cython-cmake` modules (`FindCython.cmake`, `UseCython.cmake`)
  - Updated `pyproject.toml` with `[tool.scikit-build]` configuration
  - Updated `Makefile` targets to use `uv pip install -e .` for editable installs and `uv build --wheel` for wheels
  - Updated `scripts/manage.py` to use scikit-build-core commands with `--deps-only` flag for CI builds
  - Build now uses CMake for cross-platform compatibility and better IDE integration

- **Cross-Platform Build Support** - cyllama now builds on macOS, Linux, and Windows
  - Full support for macOS (arm64/x86_64) with Metal GPU acceleration
  - Full support for Linux (x86_64) with CPU and optional GPU backends
  - Full support for Windows (x86_64) with MSVC compiler
  - Platform-specific wheel repair tools: `delocate` (macOS), `auditwheel` (Linux), `delvewheel` (Windows)
  - Thirdparty libraries (llama.cpp, whisper.cpp, stable-diffusion.cpp) included in sdist for isolated builds

- **GitHub Actions Workflows** - Automated wheel building for all platforms
  - `build-simple.yml` - Single Python version builds for macOS, Linux, and Windows
  - `build-matrix.yml` - Matrix builds across Python 3.9-3.13 on all three platforms
  - `build-manage.yml` - Builds using `scripts/manage.py` for dependency management
  - All workflows produce distributable wheels with proper platform tags
  - Wheels are uploaded as artifacts for easy distribution

- **Version Management in manage.py** - Added command-line options to specify dependency versions
  - New `--llama-version` option (default: `b7126`)
  - New `--whisper-version` option (default: `v1.8.2`)
  - New `--sd-version` option (default: `master-377-2034588`)
  - Changed `STABLE_BUILD` default to `True` for reproducible builds with pinned versions

### Removed

- **setup.py** - Replaced by `CMakeLists.txt` and `pyproject.toml` configuration
- **MANIFEST.in** - Replaced by `[tool.scikit-build]` wheel configuration in `pyproject.toml`

### Security

- **Cython Input Validation** - Added critical input validation to prevent crashes and security issues
  - Fixed buffer overflow in `get_state_seq_data()` and `get_state_seq_data_with_flags()` - now dynamically allocates buffer based on actual required size instead of fixed 512-byte stack buffer
  - Added file path validation to `lora_adapter_init()` - raises `FileNotFoundError` if LoRA file doesn't exist
  - Added file path validation to `load_state_file()` and `load_state_seq_file()` - raises `FileNotFoundError` if state file doesn't exist
  - Added parent directory validation to `save_state_file()` and `save_state_seq_file()` - raises `FileNotFoundError` if parent directory doesn't exist
  - Added NULL pointer check to `LlamaContext.__init__` - raises `ValueError` if model is None or has been freed, preventing segfaults

## [0.1.14]

### Fixed

- **Python 3.8-3.9 Compatibility** - Fixed type hint syntax incompatibility
  - Changed `str | Iterator[str]` to `Union[str, Iterator[str]]` in `api.py`
  - Now compatible with declared `requires-python = ">=3.8"` in pyproject.toml

- **Bare Except Clauses** - Replaced unsafe bare `except:` with specific exceptions
  - `memory.py:47` - Changed to `except (OSError, IOError):` for file operations
  - `memory.py:80` - Changed to `except (AttributeError, TypeError):` for vocab access
  - `tts.py:419, 430` - Changed to `except (UnicodeDecodeError, ValueError, AttributeError):` for debug output

- **Silent Unicode Errors** - Added warning logs for UnicodeDecodeError in token decoding
  - `api.py` - Now logs warning with token ID when decoding fails
  - `batching.py` - Now logs warning with token ID and sequence ID when decoding fails
  - `constrained.py` - Now logs warning with token ID when decoding fails
  - Errors are logged via Python's `logging` module at WARNING level

- **Progress Callback Crash** - Fixed crash when using `progress_callback` on `LlamaModelParams`
  - The setter now correctly sets both the C wrapper function and stores Python callback reference
  - Added `_progress_callback` attribute to prevent garbage collection of Python callback
  - Progress callback now works correctly to monitor model loading progress
  - Returning `False` from callback properly aborts model loading
  - Added 4 new tests for progress callback functionality

- **GenerationConfig Validation** - Added parameter validation to `GenerationConfig`
  - `max_tokens` must be >= 0 (0 means "generate nothing")
  - `temperature` must be >= 0.0
  - `top_k` must be >= 0
  - `top_p` must be between 0.0 and 1.0
  - `min_p` must be between 0.0 and 1.0
  - `repeat_penalty` must be >= 0.0
  - `n_gpu_layers` must be >= 0
  - `n_ctx` must be >= 1 or None
  - `n_batch` must be >= 1
  - `seed` must be >= -1
  - Multiple validation errors are reported together in a single exception
  - Added 11 new validation tests

- **Sampler Docstrings and Implementation** - Fixed XXX/FIXME markers in `LlamaSampler`
  - Fixed incorrect docstrings for `add_mirostat()` and `add_mirostat_v2()` methods
    - Removed references to non-existent parameters (`candidates`, `mu`)
    - Added proper Args documentation matching actual function signatures
    - Fixed URL format (https:# -> https://)
  - Implemented `add_logit_bias()` method (was commented out)
    - Allows biasing specific token probabilities during sampling
    - Takes list of (token_id, bias) tuples

- **MCP Race Condition** - Fixed thread safety issue in `McpStdioConnection.send_notification()`
  - Now acquires `_read_lock` before writing to stdin, matching `send_request()` behavior
  - Prevents message interleaving when notifications and requests are sent concurrently

- **Additional Python 3.9 Compatibility** - Fixed `tuple[...]` syntax in more files
  - `api.py:384` - Changed `tuple[str, GenerationStats]` to `Tuple[str, GenerationStats]`
  - `agents/react.py:557` - Changed `tuple[str, Dict[str, Any]]` to `Tuple[str, Dict[str, Any]]`
  - `whisper/cli.py:107` - Changed `tuple[np.ndarray, int]` to `Tuple[np.ndarray, int]`

- **EnhancedConstrainedAgent Stub** - Made non-functional class explicit
  - Now raises `NotImplementedError` on instantiation with helpful message
  - Directs users to use `ConstrainedAgent` instead

- **MCP Error Handling** - Improved robustness of MCP connections
  - Added stdin/stdout null checks before I/O operations
  - Added `BrokenPipeError` and `OSError` handling for connection failures
  - Errors now raise `RuntimeError` with descriptive messages

- **MCP Configurable Timeouts** - Added timeout configuration to `McpServerConfig`
  - New `request_timeout` field (default: 30.0 seconds)
  - New `shutdown_timeout` field (default: 5.0 seconds)
  - Module constants `DEFAULT_REQUEST_TIMEOUT` and `DEFAULT_SHUTDOWN_TIMEOUT`

- **Thread Safety in color.py** - Added lock protection for global color settings
  - `use_color_no_tty()` and `use_color()` now use threading lock
  - Prevents race conditions when color settings are modified concurrently

- **Session Storage Error Handling** - Improved `FileSessionStore.list_sessions()`
  - Added OSError handling for directory listing failures
  - Added logging for parse errors and file read errors
  - Continues processing remaining files if one fails

- **LLM Resource Management** - Improved context lifecycle and memory management
  - Added context reuse: contexts are cached and reused when size permits
  - Added `kv_cache_clear()` method to `LlamaContext` for clearing KV cache
  - Added `close()` method to `LLM` for explicit resource cleanup
  - Added `reset_context()` method to force context recreation
  - Added context manager support (`with LLM(...) as llm:`)
  - Added `__del__` destructor for automatic cleanup
  - Performance improvement: reduces context allocation overhead for repeated generations
  - 7 new tests for resource management

- **BatchGenerator Resource Management** - Added proper cleanup and validation to batch processing
  - Added `close()` method for explicit resource cleanup
  - Added `__del__` destructor for automatic cleanup
  - Added context manager support (`with BatchGenerator(...) as gen:`)
  - Added `is_closed` property to check generator state
  - Added `_check_closed()` internal method to prevent use after close
  - Improved input validation with detailed error messages:
    - `TypeError` for None or wrong type prompts/requests
    - `TypeError` with index and value info for invalid items in lists
    - Enhanced `ValueError` message for too many prompts (includes batch suggestion)
  - 22 new tests for cleanup, validation, and edge cases

- **ReActAgent Robust Parsing** - Improved tool call parsing and error handling
  - Added `ActionParseError` exception class with structured error information:
    - `message`: Human-readable error description
    - `action_str`: Original action that failed
    - `suggestion`: Helpful hint for fixing the format
    - `details`: List of parsing attempts made
  - Multi-strategy argument parsing:
    - Strategy 1: JSON object format with trailing comma fix
    - Strategy 2: Key=value pairs with proper quote handling
    - Strategy 3: Single positional argument
    - Strategy 4: Extract multiple quoted values for tool parameters
  - Handles common LLM output variations:
    - Trailing commas in JSON (`{"key": "value",}`)
    - Single-quoted JSON strings (`{'key': 'value'}`)
    - Escaped quotes within values
    - Multi-line argument values
  - Improved exception handling in tool execution:
    - `ActionParseError`: Parse failures with suggestions
    - `ValueError`: Unknown tools show available tools list
    - `TypeError`: Invalid arguments with tool info
    - Generic `Exception`: Unexpected errors with stack trace logging
  - Comprehensive loop detection documentation in `__init__` docstring:
    - Exact action matching mechanism
    - Same tool matching mechanism
    - Parse failure tracking
    - Recovery behavior and summary generation
  - 28 new tests for parsing, error handling, loop detection, argument types, and metrics

- **Tool Type System** - Enhanced type hint handling and schema generation
  - Added `_safe_get_type_hints()` for graceful error handling:
    - Catches `NameError` for unresolved forward references
    - Catches `TypeError` for invalid annotations
    - Falls back to raw `__annotations__` on failure
    - Logs warnings for debugging
  - Added `_python_type_to_json_schema()` with full generic type support:
    - `List[T]` -> `{"type": "array", "items": {...}}`
    - `Dict[K, V]` -> `{"type": "object", "additionalProperties": {...}}`
    - `Optional[T]` -> `{"type": "...", "nullable": true}`
    - `Union[A, B]` -> `{"anyOf": [...]}`
    - `Tuple[A, B]` -> `{"type": "array", "prefixItems": [...]}`
    - `Set[T]` -> `{"type": "array", "uniqueItems": true}`
    - `Literal["a", "b"]` -> `{"type": "string", "enum": [...]}`
    - `bytes` -> `{"type": "string", "contentEncoding": "base64"}`
    - Nested generics like `List[Dict[str, int]]` fully supported
  - Improved docstring parsing for parameter descriptions:
    - Google-style: `Args: param: description`
    - NumPy-style: `Parameters\n----------\nparam : type\n    description`
    - Sphinx/reST-style: `:param name: description`
    - Epytext-style: `@param name: description`
    - Multi-line description support for all formats
  - 23 new tests for type handling, generics, and docstring parsing

- **ContractAgent Documentation** - Enhanced documentation and test coverage
  - Comprehensive module docstring explaining Python vs C++26 differences:
    - Runtime-only checking (vs C++26 compile-time)
    - Dynamic predicate evaluation via callables
    - No undefined behavior (always well-defined policy handling)
    - Agent-specific extensions (task preconditions, answer postconditions)
  - ContractPolicy enum documentation with policy resolution hierarchy:
    - Individual contract policy (highest priority)
    - ContractAgent default policy
    - ENFORCE as fallback when no context
  - PreCondition/PostCondition class documentation with examples
  - contract_assert() documentation comparing to Python's assert statement
  - ContractAgent class documentation with usage examples
  - 28 new tests covering:
    - Policy resolution between contract and agent levels
    - ContractSpec dataclass
    - Default handler logging and verbose output
    - IterationState with all event types
    - Predicate string extraction
    - ContractViolation extended fields
    - ContractContext without handler
    - Agent with empty/None tools
    - Postcondition args edge cases
    - contract_assert outside agent context
    - Multiple contracts execution order
    - ContractTermination exception

- **Comprehensive Test Suite** - Added `test_comprehensive.py` with 53 new tests covering gaps identified in code review
  - Error condition tests (13 tests):
    - Invalid model path (nonexistent, directory, empty, invalid GGUF, truncated)
    - Context errors (size zero, batch zero, negative max_tokens)
    - BatchGenerator errors (invalid path, zero n_seq_max)
    - Memory estimation errors (invalid paths, GPU memory strings, negative sizes)
  - Unicode handling tests (11 tests):
    - Basic Unicode, CJK characters, emoji in prompts
    - Mixed scripts, special Unicode characters
    - Unicode in batch generation and streaming
    - Null bytes and surrogate pairs handling
  - Concurrent execution tests (6 tests):
    - Multiple LLM instances in parallel threads
    - Shared LLM sequential access
    - BatchGenerator separate instances in threads
    - GenerationConfig thread safety
    - Context manager cleanup in multithreaded environment
    - Batch pool concurrent access
  - Boundary condition tests (23 tests):
    - max_tokens (1, very large)
    - Context size (minimum, near limit)
    - Batch size (1, small)
    - Temperature (0, very high)
    - top_k (1), top_p (0, 1)
    - n_seq_max (minimum, exact match)
    - Stop sequences (empty, many, long)
    - Repeat penalty (0, high)
    - Special prompts (whitespace, newlines, tokens, repeated)
    - Resource limits (memory stability, generator reuse)

- **LLM Destructor Safety** - Fixed `__del__` and `close()` to handle partial initialization
  - Use `getattr()` for safe attribute access when instance may be partially initialized
  - Prevents AttributeError when constructor fails before all attributes are set

### Changed

- **Centralized Model Path Configuration** - Consolidated hardcoded model paths across the codebase
  - Added `DEFAULT_MODEL` constant in `tests/conftest.py` as single source of truth
  - Test files now use `model_path` pytest fixture from `conftest.py`
  - Subprocess tests import `DEFAULT_MODEL` from `conftest.py` where fixtures aren't available
  - Example files (`tests/examples/`) now use argparse with `-m/--model` argument
  - Script files (`scripts/`) now use argparse with `-m/--model` argument
  - Eliminates scattered hardcoded paths, simplifying model path changes

- **Type Hints** - Added missing type hints to remaining functions
  - `api.py:simple()` - Added `Optional[int]`, `bool`, and `-> bool` return type hints
  - `memory.py:parse_gpu_memory()` - Added `-> Union[int, List[int]]` return type
  - `memory.py:format_bytes()` - Added `Union[int, float]` parameter and `-> str` return type
  - `memory.py:main()` - Added `-> int` return type

- **Memory Module Improvements** - Enhanced `memory.py` with logging, validation, and documentation
  - Added module-level logger for error and diagnostic reporting
  - Added comprehensive docstrings explaining memory estimation formulas
  - Documented all magic numbers with named constants and references:
    - `FLASH_ATTN_FACTOR = 0.8` - Flash attention memory reduction from Dao et al., 2022
    - `NO_KQV_OFFLOAD_FACTOR = 1.2` - Memory increase without KQV offload
    - `SAFETY_MARGIN = 1.1` - Buffer for fragmentation and alignment
    - `PROJECTOR_SIZE_BYTES = 100MB` - LLaVA projector size estimate
    - `QUANTIZATION_FACTORS` dict with GGML type comments and bit calculations
  - Added input validation to main functions:
    - `graph_size()` - Validates n_layers, n_embd, n_ctx
    - `estimate_gpu_layers()` - Validates gpu_memory, ctx_size, batch_size
    - `estimate_memory_usage()` - Validates ctx_size, batch_size
    - `parse_gpu_memory()` - Validates string format and raises ValueError on invalid input
  - Added logging calls for error conditions:
    - File I/O errors in `get_file_host_endian()`
    - Metadata loading failures in `dump_metadata_json()`
    - Context size clamping warnings in `estimate_gpu_layers()`
    - Invalid parameter warnings throughout

- **Stop Sequence Logic Simplified** - Refactored stop sequence handling in `api.py`
  - Extracted `_find_stop_sequence()` helper method for cleaner code
  - Fixed buffer flush bug that was including stop sequences in output
  - Improved buffer management: only keeps `max_stop_len - 1` chars for sequence detection
  - Added 6 new tests for stop sequence handling (basic, multiple, streaming, edge cases)

### Added

- **Benchmark Script** - New `scripts/benchmark.py` for comprehensive performance measurement
  - Separates prefill (prompt processing) and decode (token generation) metrics
  - Reports time-to-first-token (TTFT)
  - Includes warmup run to exclude cold-start effects
  - Shows avg, median, min, max statistics
  - Configurable via `-m` (model), `-n` (runs), `-p` (prompt), `-t` (max tokens), `-c` (context size)

- **Batch Memory Pooling Integration** - Added optional memory pooling to `BatchGenerator`
  - Added `use_pooling` parameter to `BatchGenerator` (default: `False`)
  - When `use_pooling=True`, batches are reused instead of allocated/deallocated each generation
  - Reduces memory allocation overhead in high-throughput scenarios
  - 6 new tests for batch pooling functionality

## [0.1.13]

### Added

- **Zero-Dependency Image I/O** - Native PNG/JPEG/BMP support via bundled stb library
  - `SDImage.save_png(path)` - Save images as PNG format without PIL
  - `SDImage.save_jpg(path, quality=90)` - Save images as JPEG format without PIL
  - `SDImage.save_bmp(path)` - Save images as BMP format (pure Python)
  - `SDImage.save_ppm(path)` - Save images as PPM format (pure Python)
  - `SDImage.load(path, channels=0)` - Load PNG, JPEG, BMP, TGA, GIF, PSD, HDR, PIC formats via stb
  - `SDImage.load_ppm(path)` - Load PPM format (pure Python)
  - `SDImage.load_bmp(path)` - Load BMP format (pure Python)
  - Updated `save()` method to auto-detect format and use stb for PNG/JPEG/BMP
  - Channel conversion support on load (auto/grayscale/RGB/RGBA)
  - 5 new tests for PNG/JPEG roundtrip and channel conversion

- **stb Library Integration** - Bundled stb_image for image I/O
  - Added `stb_impl.cpp` to compile stb_image and stb_image_write implementations
  - Updated `setup.py` to include stb_impl.cpp in stable_diffusion extension
  - Updated `scripts/setup.sh` to copy stb headers from stable-diffusion.cpp
  - Updated `scripts/manage.py` StableDiffusionCppBuilder to copy stb headers

### Changed

- **Build Scripts** - Consistent stb header handling across build systems
  - `scripts/setup.sh` now copies `stb_image.h`, `stb_image_write.h`, `stb_image_resize.h` to thirdparty includes
  - `scripts/manage.py` StableDiffusionCppBuilder now copies stb headers during build
  - Both build methods (shell-based and Python-based) produce identical results

- **manage.py Version Constants** - Added version tracking for all dependencies
  - Added `WHISPERCPP_VERSION` constant for whisper.cpp version tracking
  - Added `SDCPP_VERSION` constant for stable-diffusion.cpp version tracking
  - `STABLE_BUILD` environment variable controls whether to use pinned versions

## [0.1.12]

### Fixed

- stable diffusion wasn't building by default via the `setup.sh` script. This is fixed now.
- cython generated .cpp files should not be included in the repository. This was inconsistently applied, and is now fixed.

### Added

- **Stable Diffusion Support** - Full integration of stable-diffusion.cpp for image generation
  - New `cyllama.stablediffusion` module with Cython bindings for stable-diffusion.cpp
  - `SDContext` class for model loading and image generation
  - `SDContextParams` class for context configuration (model paths, threads, backends)
  - `SDImage` class with numpy/PIL conversion (`to_numpy()`, `to_pil()`, `from_numpy()`, `save()`)
  - `SDImageGenParams` class for generation parameters (prompt, dimensions, seed, steps, CFG)
  - `SDSampleParams` class for sampling configuration (method, scheduler, steps, eta)
  - Convenience functions: `text_to_image()`, `image_to_image()` for simple usage
  - Utility functions: `get_num_cores()`, `get_system_info()`, `type_name()`, `sample_method_name()`, `scheduler_name()`
  - Callback support: `set_log_callback()`, `set_progress_callback()`, `set_preview_callback()` for monitoring
  - Full enum support: `RngType`, `SampleMethod`, `Scheduler`, `Prediction`, `SDType`, `LogLevel`, `PreviewMode`, `LoraApplyMode`
  - Support for GGUF, safetensors, and ckpt model formats
  - SDXL, SD 1.x/2.x, SD3, FLUX model architecture support
  - 29 comprehensive tests in `tests/test_stablediffusion.py`
  - Example: `tests/examples/stablediffusion_example.py` with CLI interface
  - Build configuration via `WITH_STABLEDIFFUSION` environment variable (default: enabled)

- **Video Generation** - Support for video generation models (Wan, CogVideoX)
  - `SDContext.generate_video()` method for video frame generation
  - Support for init/end image for video interpolation
  - Configurable frame count, dimensions, and sampling parameters

- **Upscaler Class** - ESRGAN image upscaling support
  - `Upscaler` class for loading ESRGAN models
  - `upscale()` method for image super-resolution
  - Automatic upscale factor detection from model
  - Context manager support for resource management

- **Model Conversion** - Convert models between formats and quantizations
  - `convert_model()` function for model format conversion
  - Support for all quantization types (F16, Q4_0, Q8_0, etc.)
  - Optional VAE path and tensor type rules

- **ControlNet Preprocessing** - Canny edge detection for ControlNet
  - `canny_preprocess()` function for in-place image preprocessing
  - Configurable high/low thresholds, weak/strong values, inverse option

- **Preview Callbacks** - Real-time generation preview support
  - `set_preview_callback()` for monitoring generation progress with preview images
  - Configurable preview modes (PROJ, TAE, VAE)
  - Interval and denoised/noisy preview options

- **CLI Tool** - Command-line interface for stable diffusion
  - `python -m cyllama.stablediffusion generate` - Generate images from text
  - `python -m cyllama.stablediffusion upscale` - Upscale images with ESRGAN
  - `python -m cyllama.stablediffusion convert` - Convert model formats
  - `python -m cyllama.stablediffusion info` - Show system info and available options

- **Testing & Documentation** - Comprehensive test suite and API documentation
  - 77 unit tests covering all wrapper classes (SDContext, SDContextParams, SDImage, SDSampleParams, SDImageGenParams, Upscaler)
  - Integration tests with real models (text_to_image, image_to_image generation)
  - Tests for Canny preprocessing, callbacks, enums, and utility functions
  - Advanced example script: `tests/examples/stablediffusion_advanced_example.py`
  - Complete API documentation in `docs/api_reference.md` (Stable Diffusion Integration section)
  - CLI usage documentation with command examples

### Changed

- **setup.py** - Added stable-diffusion.cpp extension build support
  - Added `SDCPP_INCLUDE` and `SDCPP_LIBS_DIR` paths for stable-diffusion headers and libraries
  - Added `libstable-diffusion.a` to static library linking
  - Added rpath configuration for stable-diffusion shared libraries
  - Cythonize support for `stable_diffusion.pyx`

- **MANIFEST.in** - Added stable-diffusion source files for distribution
  - Added `src/cyllama/stablediffusion/*.pxd`, `*.pyx`, `*.pxi`, `*.cpp`, `*.h`
  - Added `thirdparty/stable-diffusion.cpp/include` and `lib` directories

## [0.1.11]

### Added

- **manage.py Build Improvements** - Fixed compatibility with latest llama.cpp (main branch)
  - Added `cmake_build_targets()` method to build specific CMake targets without building all tools
  - Added nlohmann JSON header copying from `vendor/nlohmann/` for `json-partial.h` compatibility
  - Added `cmake_value()` helper to convert Python booleans to CMake ON/OFF values
  - Manual library copying to avoid cmake install failures when tools are disabled

- **Agent Client Protocol (ACP) Support** - Full ACP implementation for editor/IDE integration
  - `ACPAgent` class providing ACP-compliant agent that can be spawned by editors (Zed, Neovim, etc.)
  - JSON-RPC 2.0 transport layer over stdio for bidirectional communication
  - Session management with `session/new`, `session/load`, `session/prompt`, `session/cancel` methods
  - Tool permission flow with `session/request_permission` for user approval
  - File operations delegated to editor via `fs/read_text_file`, `fs/write_text_file`
  - Terminal operations via `terminal/create`, `terminal/output`, `terminal/kill`, `terminal/release`
  - Async bridge for sending notifications from synchronous agent execution
  - 30 comprehensive tests in `tests/test_acp.py`
  - Documentation: `docs/protocol_support.md`

- **Model Context Protocol (MCP) Client** - Connect to external MCP servers for tool/resource access
  - `McpClient` class for managing connections to multiple MCP servers
  - Stdio transport (subprocess) and HTTP transport support
  - Automatic tool and resource discovery from connected servers
  - MCP tools exposed as cyllama `Tool` instances for seamless agent integration
  - `McpServerConfig` for server connection configuration
  - 23 comprehensive tests in `tests/test_mcp.py`

- **Session Storage Backends** - Persistent session storage for ACP agents
  - `MemorySessionStore` - In-memory storage (default, non-persistent)
  - `FileSessionStore` - JSON file-based storage in a directory
  - `SqliteSessionStore` - SQLite database storage (Python built-in)
  - `Session` dataclass with messages, tool calls, and permission caching
  - Permission caching for "allow always" / "reject always" decisions
  - `create_session_store()` factory function
  - 27 comprehensive tests in `tests/test_session.py`

- **JSON-RPC 2.0 Transport Layer** - Foundation for ACP and MCP protocols
  - `JsonRpcRequest`, `JsonRpcResponse`, `JsonRpcError` message classes
  - `StdioTransport` for newline-delimited JSON over stdin/stdout
  - `JsonRpcServer` for request dispatching and handler registration
  - `AsyncBridge` for queue-based notification sending from sync code
  - Standard error codes (Parse Error, Method Not Found, Internal Error, etc.)
  - 23 comprehensive tests in `tests/test_jsonrpc.py`

- **Agents CLI** - Command-line interface for ACP server and agent operations
  - `python -m cyllama.agents.cli acp` - Start ACP server for editor integration
  - `python -m cyllama.agents.cli run` - Run single agent query
  - `python -m cyllama.agents.cli mcp-test` - Test MCP server connections
  - Support for MCP server configuration via command-line flags
  - Session storage configuration (memory, file, sqlite)

### Changed

- **Module Exports**: Enhanced `cyllama.agents` module with protocol support
  - Added `ACPAgent`, `serve_acp` for ACP functionality
  - Added `McpClient`, `McpServerConfig`, `McpTransportType`, `McpTool` for MCP
  - Added `Session`, `SessionStore`, `MemorySessionStore`, `FileSessionStore`, `SqliteSessionStore`
  - Added `JsonRpcServer`, `JsonRpcRequest`, `JsonRpcResponse`, `JsonRpcError`, `StdioTransport`

### Fixed

- **manage.py Build Script** - Fixed build errors when using latest llama.cpp
  - Disabled `LLAMA_HTTPLIB` to prevent linker errors with httplib symbols
  - Disabled `LLAMA_BUILD_SERVER`, `LLAMA_BUILD_TESTS`, `LLAMA_BUILD_EXAMPLES` (require httplib)
  - Build only required targets (`llama`, `common`, `mtmd`) to avoid httplib-dependent tools like `llama-run`
  - Fixed library paths for manual copying (`libggml-cpu.a` path correction)

- **Source Distribution (MANIFEST.in)** - Fixed missing files in sdist/wheel builds
  - Added `*.hpp` files for `nlohmann/json.hpp` header
  - Added `*.a` static libraries from `thirdparty/llama.cpp/lib` and `thirdparty/whisper.cpp/lib`
  - Ensures `uv build` and `pip install` from source work correctly

- **GitHub Workflow (build-wheels.yml)** - Fixed CI build failures
  - Added Cython installation step before `manage.py build` runs
  - Ensures Cython is available for thirdparty library compilation

- **Cython Build** - Fixed `setup.py` to generate C++ files instead of C files
  - Added `--cplus` flag to `run_cythonize()` function
  - Ensures `.cpp` files are generated for C++ language extensions

## [0.1.10]

### Added

- **ContractAgent** - C++26-inspired contract-based agent with preconditions, postconditions, and runtime assertions
  - `ContractAgent` class wrapping inner agents (ReActAgent or ConstrainedAgent) with contract verification
  - `@pre` decorator for tool preconditions (validate inputs before execution)
  - `@post` decorator for tool postconditions (validate outputs after execution)
  - `contract_assert()` function for runtime invariants within tool implementations
  - `ContractPolicy` enum with four evaluation modes: `IGNORE`, `OBSERVE`, `ENFORCE`, `QUICK_ENFORCE`
  - `ContractViolation` dataclass for detailed violation reporting
  - Agent-level contracts: `task_precondition`, `answer_postcondition`, `iteration_invariant`
  - Custom violation handlers for logging, alerting, or custom error handling
  - New event types: `CONTRACT_CHECK`, `CONTRACT_VIOLATION`
  - Thread-safe `ContractContext` for `contract_assert` integration
  - Postconditions receive actual typed return values (`raw_result`) for accurate type checking
  - 53 comprehensive tests in `tests/test_agents_contract.py`
  - Example: `tests/examples/agent_contract_example.py`
  - Design documentation: `CONTRACT_AGENT.md`, `CONTRACT_AGENT_IMPL.md`

- **ReActAgent Event Metadata Enhancement** - Added tool execution details to event metadata
  - ACTION events now include `tool_name` and `tool_args` in metadata
  - OBSERVATION events now include `tool_name`, `tool_args`, and `raw_result` in metadata
  - `raw_result` preserves actual typed return value (not stringified) for programmatic use
  - Enables ContractAgent and other wrappers to intercept and validate tool calls

- **HuggingFace Model Downloads** - Improved `download_model()` function with HuggingFace support
  - Auto-detection of HuggingFace repo format (e.g., `"user/repo:tag"`) as first positional argument
  - Resolves HuggingFace repos to download URLs via `get_hf_file()` API
  - Default download location: `~/.cache/llama.cpp/`
  - Supports `:latest` tag which auto-selects best quantization (Q4_K_M)
  - Custom download paths via `model_path` parameter
  - Specific file selection via `hf_file` parameter

### Changed

- **API Refactoring** - Renamed core API for clarity and consistency
  - `generate.py` → `api.py` - New unified API module
  - `Generator` class → `LLM` class - Better semantic naming
  - `generate()` function → `complete()` function - More precise terminology
  - `SimpleChat` class → `Chat` class - Clearer naming for chat interface
  - `EmbeddedLlamaServer` class → `PythonServer` class - Simpler naming
  - Merged `api.simple()` into new `api.py` module
  - Updated all integrations and tests to use new naming
  - All exports remain available from `cyllama` package root

- **Server Module Reorganization** - Renamed server implementations for clarity
  - `embedded.py` → `python.py` - Pure Python HTTP server implementation
  - `mongoose_server.pyx` → `embedded.pyx` - Embedded C server using Mongoose library
  - `MongooseServer` class → `EmbeddedServer` class - Better reflects embedded C implementation
  - Updated `__main__.py` server type choices: `["embedded", "mongoose"]` → `["embedded", "python"]`
  - Default server type is now `"embedded"` (high-performance C implementation)
  - All imports updated: `from cyllama.llama.server import PythonServer, EmbeddedServer`
  - Convenience functions renamed: `start_mongoose_server()` → `start_embedded_server()`
  - Updated documentation, tests, and examples to reflect new naming
  - Maintains backward compatibility through proper module exports

### Added

- **Multi-Backend GPU Support** - Environment variable configuration for all GPU acceleration backends
  - Added support for CUDA, Vulkan, SYCL, HIP/ROCm, and OpenCL backends (in addition to existing Metal support)
  - Environment variables: `GGML_METAL`, `GGML_CUDA`, `GGML_VULKAN`, `GGML_SYCL`, `GGML_HIP`, `GGML_OPENCL`
  - New Makefile targets: `build-cpu`, `build-cuda`, `build-vulkan`, `build-sycl`, `build-hip`, `build-all`
  - New `make show-backends` command to display current backend configuration
  - Backend detection in `setup.py` - automatically detects available GPU backends (CUDA, Vulkan, SYCL, ROCm, Metal)
  - Enhanced `scripts/setup.sh` to pass CMake backend flags based on environment variables
  - Enhanced `scripts/manage.py` with backend command-line flags (`--cuda`, `--vulkan`, `--metal`, `--sycl`, `--hip`, `--opencl`, `--cpu-only`)
  - Dynamic library linking in `setup.py` based on enabled backends
  - Comprehensive user documentation in `docs/BUILD_BACKENDS.md`
  - Updated README.md with GPU acceleration build instructions
  - Multi-backend builds supported (e.g., CUDA + Vulkan simultaneously)
  - Two build methods: Makefile (shell-based) or manage.py (Python-based)

- **Integration Improvements** - Cleaner import paths for framework integrations
  - Added `OpenAIClient` alias for `OpenAICompatibleClient` in `cyllama.integrations`
  - Now supports `from cyllama.integrations import OpenAIClient` (shorter import path)
  - Maintains backward compatibility with full path import

### Fixed

- **Batch Processing** - Implemented working batch processing functionality
  - Fixed `BatchGenerator` and `batch_generate()` which were using incorrect API and never worked
  - Implemented `LlamaBatch.add()` and `LlamaBatch.clear()` methods in Cython bindings
  - Added `n_seq_max` parameter to control maximum parallel sequences (default: 8)
  - Fixed batch index tracking for proper logit sampling in parallel sequences
  - Added comprehensive test suite with 13 tests covering all batch processing scenarios
  - Updated documentation with correct API usage examples
- **Logging** - Disabled verbose llama.cpp logging by default in `LLM`, `complete()`, `chat()`, and `BatchGenerator`
  - Added `verbose` parameter to control logging output
  - Calls `disable_logging()` when `verbose=False` (the default)
  - Significantly reduces debug output for cleaner user experience

### Security

- **scripts/manage.py Hardening** - Comprehensive security improvements to build manager
  - Enhanced `getenv()` function with robust error handling and warning logs for invalid values
  - Hardened `cmd()` method with path resolution to prevent path traversal exploits
  - Secured `download()` method with URL scheme validation (http/https only), path traversal prevention, and file size limits (100MB default)
  - Hardened `extract()` method with pre-extraction path validation to prevent zip slip attacks
  - Added warning logs for missing backend libraries during build
  - All security improvements are type-safe with full mypy compliance (0 errors)
  - All 260 tests passing with no regressions
  - Production readiness score upgraded from 8.5/10 to 9.5/10
  - See `docs/MANAGE_REVIEW.md` and `docs/MANAGE_SECURITY_IMPROVEMENTS.md` for details

## [0.1.9] - 2025-11-21

### Added

- **High-Level Generation API** (`src/cyllama/generate.py`)
  - Added `generate()` convenience function for one-line text generation
  - Added `chat()` function for multi-turn conversation interface
  - Added `Generator` class for efficient model reuse and caching
  - Added `GenerationConfig` dataclass for comprehensive generation parameters
  - Added `GenerationStats` dataclass for detailed performance metrics
  - Automatic context and sampler management with optimal sizing
  - Full streaming support with token-by-token callbacks
  - Support for temperature, top-k, top-p, min-p, repeat penalty, and seed parameters
  - Stop sequences and custom tokenization options
  - 60+ comprehensive tests in `tests/test_generate.py`

- **Batch Processing Utilities** (`src/cyllama/batching.py`)
  - Added `batch_generate()` convenience function for efficient batch processing
  - Added `BatchGenerator` class for parallel sequence processing
  - Added `BatchRequest` and `BatchResponse` dataclasses for structured batch operations
  - Utilizes llama.cpp's native batching for 3-10x throughput improvement
  - Detailed performance statistics per request
  - Automatic batch size optimization
  - Examples in documentation and tests

- **OpenAI-Compatible API** (`src/cyllama/integrations/openai_compat.py`)
  - Added `OpenAICompatibleClient` class providing drop-in replacement for OpenAI client
  - Full chat completions API compatibility
  - Streaming support with proper chunking
  - Compatible message format (system, user, assistant roles)
  - Usage statistics (prompt tokens, completion tokens)
  - Response objects matching OpenAI's format (ChatCompletion, ChatCompletionChunk)
  - 10+ comprehensive tests in `tests/test_integrations.py`

- **LangChain Integration** (`src/cyllama/integrations/langchain.py`)
  - Added `CyllamaLLM` class implementing LangChain's LLM interface
  - Works seamlessly with LangChain chains, agents, and tools
  - Streaming support with LangChain callback managers
  - Proper error handling when LangChain is not installed
  - Full parameter compatibility (temperature, max_tokens, top_k, top_p)
  - Example usage in documentation

- **Comprehensive Documentation**
  - Added `docs/USER_GUIDE.md` - Complete 450+ line user guide covering all APIs
  - Added `docs/COOKBOOK.md` - 350+ line cookbook with practical patterns and recipes
  - Added `docs/IMPROVEMENTS_SUMMARY.md` - Detailed summary of all improvements
  - Sections on text generation, chat apps, structured output, performance, integrations
  - Working examples for FastAPI, Flask, Gradio integrations
  - Error handling patterns, best practices, troubleshooting guides

### Changed

- **Module Exports**: Enhanced `src/cyllama/__init__.py` with convenient top-level imports
  - Exported high-level generation functions: `generate`, `chat`, `Generator`, `GenerationConfig`
  - Exported batching utilities: `batch_generate`, `BatchGenerator`, `BatchRequest`, `BatchResponse`
  - Exported memory utilities: `estimate_gpu_layers`, `estimate_memory_usage`, `MemoryEstimate`
  - All new APIs available directly from `import cyllama`

- **Documentation**: Updated `RECOMMENDED_TO_WRAP.md` to reflect completion status
  - All five high-priority APIs now marked as completed
  - Updated priorities for remaining optional features
  - Comprehensive status tracking and implementation notes

### Technical Implementation

- **High-Level API Architecture**: Designed for simplicity with power when needed
  - Automatic model and context lifecycle management
  - Lazy initialization with smart caching
  - Proper cleanup with Python context managers
  - Type hints throughout for IDE support

- **Streaming Implementation**: Efficient token-by-token generation
  - Generator-based streaming for memory efficiency
  - Optional token callbacks for real-time processing
  - Compatible with both sync and async patterns

- **Batch Processing**: Leverages llama.cpp's native batching
  - Parallel sequence processing with shared KV cache
  - Automatic batch size optimization based on context
  - Per-sequence logit computation
  - Efficient memory management

- **Integration Layer**: Minimal overhead adapters
  - OpenAI compatibility through adapter pattern
  - LangChain integration via interface implementation
  - Graceful degradation when optional dependencies missing
  - Zero-copy data passing where possible

- **Testing Strategy**: Comprehensive test coverage
  - Unit tests for all new APIs and configurations
  - Integration tests with real models
  - Edge case testing (empty prompts, zero tokens, etc.)
  - Performance validation tests
  - All 276 tests passing

### Performance Improvements

- **Model Reuse**: Generator class caches model between generations
  - Eliminates repeated model loading (5-10s saved per generation)
  - Smart context recreation only when necessary
  - Sampler recreation for each generation to respect config changes

- **Batch Processing**: Up to 10x throughput improvement
  - Parallel processing of multiple prompts
  - Shared model and context overhead
  - Efficient GPU utilization

- **Memory Management**: Automatic context sizing
  - Dynamic sizing based on prompt + max_tokens
  - Prevents over-allocation
  - Optimal batch sizes for available memory

## [0.1.8] - 2025-11-21

### Added

- **Speculative Decoding API** (`speculative.h` wrapper)
  - Added `SpeculativeParams` class for configuring speculative decoding parameters
  - Added `Speculative` class for managing speculative decoding with target and draft models
  - Methods: `are_compatible()`, `add_replacement()`, `gen_draft()`
  - Parameters: `n_draft` (max drafted tokens), `n_reuse` (token reuse), `p_min` (acceptance probability)
  - 17 comprehensive tests in `tests/test_speculative.py`
  - Example: `tests/examples/speculative_example.py` with parameter tuning demonstrations
  - Enables 2-3x inference speedup when using compatible draft/target model pairs
  - Supports token replacement mappings for models with different tokenizers

### Changed

- **Documentation**: Updated `RECOMMENDED_TO_WRAP.md` to mark speculative decoding as completed
  - All five high-priority APIs now fully implemented
  - Updated implementation status and remaining priorities

### Technical Implementation

- **Speculative API**: Created `speculative.pxd` with C API declarations, wrapper implementation in `speculative.pxi`
- **Context Management**: Proper handling of LlamaContext pointer access via `.ptr` attribute
- **Memory Safety**: Automatic resource cleanup with `__dealloc__` method
- **Exception Handling**: All C++ API bindings use `except +` for automatic exception translation
- **Integration**: Seamlessly integrated into main module via `llama_cpp.pyx` includes

## [0.1.7] - 2025-11-17

### Added

- **GGUF File Format API** (`gguf.h` wrapper)
  - Added `GGUFContext` class for reading and writing GGUF model files
  - Methods: `from_file()`, `write_to_file()`, `get_value()`, `get_all_metadata()`, `set_val_*()`, `get_all_tensor_info()`, `find_tensor()`, `remove_key()`
  - 6 comprehensive tests in `tests/test_gguf.py`
  - Example: `tests/examples/gguf_example.py`
  - Enables model inspection, metadata manipulation, and custom GGUF creation

- **JSON Schema to Grammar API** (`json-schema-to-grammar.h` wrapper)
  - Added `json_schema_to_grammar()` function to convert JSON schemas to GBNF grammars
  - Supports nested objects, arrays, enums, and complex schemas
  - Force GBNF mode with `force_gbnf` parameter
  - C++ wrapper layer to bridge nlohmann::json library
  - 11 comprehensive tests in `tests/test_json_schema.py`
  - Example: `tests/examples/json_schema_example.py`
  - Essential for structured JSON output from language models

- **Download Helper API** (`download.h` wrapper)
  - Added `download_model()` function for downloading from HuggingFace, URLs, and Docker registries
  - Added `get_hf_file()` function with Ollama-style quantization tags (`:q4`, `:q8`, etc.)
  - Added `list_cached_models()` function to enumerate cached models
  - Added `resolve_docker_model()` function for Docker registry integration
  - Support for bearer token authentication
  - 11 comprehensive tests in `tests/test_download.py`
  - Example: `tests/examples/download_example.py`
  - Models cached in `~/.cache/llama.cpp/`

- **N-gram Cache API** (`ngram-cache.h` wrapper)
  - Added `NgramCache` class for accelerating generation with repeated patterns
  - Methods: `update()`, `draft()`, `save()`, `load()`, `merge()`
  - Support for context/dynamic/static cache types
  - Configurable ngram_min and ngram_max parameters (2-4)
  - 14 comprehensive tests in `tests/test_ngram_cache.py`
  - Example: `tests/examples/ngram_cache_example.py`
  - Provides 2-10x speedup for repetitive text (code, templates, structured data)

### Changed

- **Exception Handling**: All new C++ API bindings use `except +` for automatic exception translation
- **Documentation**: Updated `RECOMMENDED_TO_WRAP.md` to reflect completion of 4 new high-priority APIs

### Technical Implementation

- **GGUF API**: Created `gguf.pxd` with complete C API declarations, wrapper methods in `llama_cpp.pyx`
- **JSON Schema**: C++ bridge (`json_schema.cpp/h`) for nlohmann::json, installed v3.12.0 headers
- **Download API**: Created `download.pxd`, Cython wrappers with memory-safe string handling
- **N-gram Cache**: Created `ngram_cache.pxd`, draft vector seed token initialization, proper memory management

## [0.1.6]

### Fixed

- **Multimodal (MTMD) Test Infrastructure**: Resolved critical test import and type issues for multimodal functionality
  - **Import Structure**: Fixed circular import issue in `mtmd` submodule by correcting import paths from `..mtmd` to `..llama_cpp`
  - **Data Type Compatibility**: Updated `MtmdBitmap.create_image()` parameter annotation from `str` to `bytes` to match actual Cython implementation
  - **Error Handling**: Added file existence check to `MultimodalProcessor` constructor for better error reporting before type validation
  - **Test Expectations**: Updated test assertions to match actual behavior (empty string vs None for bitmap IDs, OverflowError for invalid parameters)
  - **Mock Object Integration**: Properly configured Mock objects in tests to avoid Cython type checking conflicts
  - **Test Results**: All 27 multimodal tests now pass with 3 appropriately skipped integration tests

- **Circular Import Resolution**: Eliminated circular dependency issues in multimodal module structure
  - Fixed `src/cyllama/llama/mtmd/multimodal.py` import from `..mtmd` to `..llama_cpp`
  - Fixed `src/cyllama/llama/mtmd/__init__.py` import from `..mtmd` to `..llama_cpp`
  - Ensured proper import hierarchy where Cython classes are imported from the compiled extension module
  - Maintained backward compatibility for all existing multimodal API usage

### Changed

- **Multimodal Error Handling**: Enhanced robustness of multimodal processor initialization
  - Added early file existence validation in `MultimodalProcessor` constructor
  - Improved error messages with clearer context for file not found scenarios
  - Better separation of concerns between file validation and object initialization

### Technical Implementation

- **Import Architecture**: Corrected module import hierarchy for proper Cython class access
  - The `mtmd.pxi` include file defines Cython classes that are compiled into `llama_cpp.pyx`
  - High-level Python wrappers in `multimodal.py` now correctly import from the compiled extension
  - Eliminated self-referential imports that were causing circular dependency issues

- **Type System Compatibility**: Improved compatibility between Python test framework and Cython type checking
  - Fixed parameter type annotations to match actual implementation behavior
  - Ensured Mock objects are properly isolated from Cython type validation where appropriate
  - Maintained strict type checking for production code while enabling flexible testing

## [0.1.5]

### Added

- **High-Performance Embedded HTTP Server**: Production-ready C-based server alternative
  - New `src/cyllama/llama/server/embedded.pyx` (formerly `mongoose_server.pyx`) - Cython bindings for Mongoose web server
  - Complete integration of Mongoose v7.19 (single-file embedded web server)
  - `EmbeddedServer` class (formerly MongooseServer) providing high-performance C-based alternative to Python HTTP server
  - Zero external dependencies beyond existing cyllama requirements
  - Direct C networking with concurrent connection handling (vs. Python GIL limitations)
  - Uses same `ServerSlot` logic and OpenAI-compatible API as Python server
  - Production-ready performance for high-throughput LLM inference scenarios

- **Mongoose Server nogil Optimizations**: Advanced GIL-free operations for maximum performance
  - **Event Loop Optimization**: Core `_wait_for_shutdown_nogil()` method runs `mg_mgr_poll()` without GIL blocking
  - **Connection Management**: `_close_connections_nogil()` method for GIL-free connection cleanup operations
  - **HTTP Response Optimization**: `_send_reply_nogil()` method for non-blocking HTTP response transmission
  - **Core API Enhancement**: All Mongoose C API functions marked with `nogil` decorators for maximum efficiency
  - **Concurrent Thread Support**: Python threads can run concurrently during network I/O operations
  - **Performance Results**: 15.9μs average server lifecycle, excellent concurrent thread performance
  - **Zero API Changes**: All optimizations are transparent with full backward compatibility

- **REST API Server Infrastructure**: Complete Python wrapper for llama.cpp server functionality
  - New `src/cyllama/llama/server.py` module with comprehensive server management capabilities
  - `ServerConfig` class for complete configuration management of all llama-server parameters
  - `LlamaServer` class with full subprocess lifecycle management (start, stop, restart, status)
  - `LlamaServerClient` class providing OpenAI-compatible API client functionality
  - Automatic binary detection with fallback paths for llama-server executable
  - Context manager support for automatic server cleanup and resource management

- **OpenAI-Compatible API Support**: Full compatibility with OpenAI API standards
  - Chat completions endpoint (`/v1/chat/completions`) with streaming support
  - Embeddings endpoint (`/v1/embeddings`) for vector generation
  - Models endpoint (`/v1/models`) for available model listing
  - Health check endpoint (`/health`) for server monitoring
  - Complete request/response handling with proper error management
  - Authentication support with API keys and SSL certificates

- **Server Management Features**: Production-ready server control and monitoring
  - Graceful shutdown with configurable timeouts and fallback force-kill
  - Health checking and readiness detection with automatic retry logic
  - Server status monitoring with API readiness detection
  - Comprehensive logging and error reporting
  - Support for all llama-server configuration options and parameters
  - Web UI integration and metrics endpoint support

- **Developer Tools and Examples**: Complete development and integration support
  - `examples/server_example.py` - Full-featured server demonstration script
  - `examples/server_simple.py` - Minimal server setup example
  - Convenience `start_server()` function for quick server initialization
  - Comprehensive documentation and usage examples
  - Integration with existing cyllama module structure

- **Comprehensive Testing**: Extensive test coverage for reliability
  - `tests/test_server.py` with 28 comprehensive test cases covering all functionality
  - Unit tests for configuration, server lifecycle, and client operations
  - Integration tests with real model files and llama-server binary
  - Mock-based testing for edge cases and error conditions
  - Graceful handling of optional dependencies (requests library)
  - All tests passing with proper skip behavior for missing dependencies

### Changed

- **Module Structure**: Enhanced cyllama.llama module with server functionality
  - Added server classes to `src/cyllama/llama/__init__.py` exports
  - Updated module imports for easy access to server components
  - Maintained backward compatibility with existing API structure

- **Dependency Management**: Optional dependency handling for enhanced functionality
  - Graceful degradation when `requests` library is not available
  - Clear error messages and installation guidance for missing dependencies
  - Server functionality works without requests (health checking disabled)
  - Client functionality requires requests with helpful error messages

### Technical Implementation

- **Mongoose nogil Implementation**: Low-level GIL optimization techniques
  - **Cython nogil Decorators**: Applied to all core Mongoose C API functions including `cyllama_mg_mgr_init`, `cyllama_mg_mgr_free`, `cyllama_mg_mgr_poll`, `cyllama_mg_http_listen`, and `cyllama_mg_http_reply`
  - **C Pointer Extraction**: Safe conversion of Python bytes objects to C char pointers before entering nogil sections
  - **GIL Management**: Strategic use of `with gil:` blocks for minimal Python object access during long-running operations
  - **Thread Safety**: Preserved thread safety while enabling concurrent Python thread execution during network operations
  - **Memory Safety**: Maintained proper memory management and cleanup without introducing race conditions

- **Subprocess Management**: Robust process control and monitoring
  - Automatic binary discovery across multiple installation paths
  - Comprehensive parameter translation from Python config to command-line arguments
  - Process health monitoring with PID tracking and status detection
  - Proper signal handling for graceful shutdown sequences

- **Error Handling and Reliability**: Production-ready error management
  - Comprehensive exception handling with descriptive error messages
  - Timeout handling for server startup and shutdown operations
  - Resource cleanup and memory management for long-running servers
  - Proper handling of network connectivity issues and API failures

- **Performance and Scalability**: Optimized for production use cases
  - Minimal overhead Python wrapper around native llama-server binary
  - Efficient configuration management with parameter validation
  - Support for high-performance server configurations and GPU utilization
  - Integration with existing cyllama performance optimizations

- **Embedded Server Infrastructure**: Native Python server using existing cyllama bindings
  - New `src/cyllama/llama/server/embedded.py` module with direct llama.cpp integration
  - `EmbeddedLlamaServer` class providing OpenAI-compatible API without external binaries
  - `ServerSlot` class for concurrent request processing using native cyllama objects
  - Direct memory sharing with `LlamaModel`, `LlamaContext`, and `LlamaSampler` instances
  - Built-in HTTP server using Python's standard library for zero external dependencies
  - CLI interface via `python -m cyllama.llama.server` for easy deployment

- **Zero-Binary Deployment**: Complete server functionality without subprocess management
  - No requirement for llama-server executable or external process spawning
  - Direct integration with existing libllama.a linkage through cyllama bindings
  - Better error handling with Python-level exception management
  - Simplified deployment as single Python process with embedded functionality
  - Resource cleanup through context manager support and automatic slot management
  - Fixed critical issues with context creation, token processing, and state management

- **Native API Endpoints**: Full OpenAI-compatible server implementation
  - `/health` endpoint for server monitoring and readiness checks
  - `/v1/models` endpoint for available model listing and metadata
  - `/v1/chat/completions` endpoint with complete chat completion functionality
  - Proper JSON request/response handling with error management
  - Support for streaming responses and standard OpenAI parameters
  - Successfully generating responses like "2 + 2 = 4" with proper token handling

- **Server Implementation Fixes**: Critical bug fixes for production stability
  - Fixed `vocab.is_eog_token()` method name error to correct `vocab.is_eog()`
  - Corrected token conversion from `token_to_piece(token_id)` to `token_to_piece(token_id, 0, True)`
  - Resolved LlamaContext constructor parameter handling with proper `LlamaContextParams` objects
  - Refactored from creating new contexts per request to slot-based persistent contexts
  - Added proper context state reset between requests to prevent response contamination
  - Eliminated segmentation faults and server crashes during chat completion processing

- **Comprehensive Testing and Examples**: Production-ready development support
  - `tests/test_embedded_server.py` with 26 comprehensive test cases
  - `examples/embedded_server_example.py` - Full demonstration with API testing
  - Unit tests covering configuration, server lifecycle, and HTTP endpoints
  - Integration tests with real model files and complete request/response cycles
  - Mock-based testing for edge cases and error conditions with proper isolation
  - Verified working implementation with successful chat completion generation

## [0.1.4]

### Added

- **GPU Memory Estimation Module**: Advanced memory management and GPU allocation optimization
  - New `src/cyllama/memory.py` module with sophisticated memory estimation capabilities
  - `estimate_gpu_layers()` function for intelligent GPU layer allocation across single or multiple GPUs
  - `estimate_memory_usage()` function for comprehensive memory analysis without GPU constraints
  - `MemoryEstimate` dataclass for structured memory allocation results
  - Support for multi-GPU tensor splitting with optimal layer distribution

- **Memory CLI Tool**: Complete command-line interface for memory analysis
  - `src/cyllama/memory_cli.py` - Interactive memory estimation tool
  - Memory overview with model parameter analysis and architecture details
  - GPU allocation estimation with hardware-specific recommendations
  - Multi-GPU configuration support with tensor split visualization
  - Human-readable output formatting with size conversions (B/KB/MB/GB)
  - Performance guidance for optimal hardware utilization

- **Multi-Architecture Support**: Comprehensive model architecture compatibility
  - LLaMA, Gemma, Qwen2, StableLM, DeepSeek architecture-specific calculations
  - Automatic fallback handling for unknown architectures
  - Architecture-aware graph memory computation with optimization factors

- **Advanced Memory Features**: Professional-grade memory management capabilities
  - Multiple quantization level support (F32, F16, Q4_0, Q8_0, etc.)
  - KV cache precision options (F16/F32) with memory impact analysis
  - Context size and batch size memory scaling
  - Memory safety margins and optimization hints
  - Projector memory requirements for multimodal models

- **Integration and Testing**: Seamless codebase integration
  - Added memory estimation functions to main `__init__.py` exports
  - Comprehensive test suite with unit tests for all core functionality
  - Mock-based testing for model loading scenarios
  - Integration tests with real model files

### Changed

- **Module Exports**: Enhanced main module interface
  - Added `estimate_gpu_layers`, `estimate_memory_usage`, and `MemoryEstimate` to public API
  - Updated import structure for easy access to memory estimation features

- **Performance Optimizations**: Major performance improvements across core operations

  **Tokenization Optimizations** (Priority 2 - Medium Risk, High Benefit):
  - **Tokenization Speed**: Achieved 2.5x performance improvement (up to 4.6M tokens/s from 1.8M tokens/s)
  - **Smart Memory Allocation**: Replaced fixed vocab-size allocation with conservative text-length estimation
  - **Pre-allocated Lists**: Optimized token copying with direct assignment instead of append operations
  - **Reduced Python Overhead**: Eliminated list extension operations and optimized Cython variable declarations
  - **Memory Efficiency**: Reduced allocation overhead by ~90% for typical text lengths
  - Performance scaling across text sizes: 1.6M-4.6M tokens/s with 17K-537K calls/s

  **Property Caching Optimizations** (Priority 1 - Low Risk, Immediate Benefit):
  - **Property Access Speed**: Achieved exceptional performance with 18-21 million property accesses/second
  - **Microsecond-Level Access**: Average 0.05μs per property access (virtually instantaneous)
  - **Cached Model Properties**: Optimized n_embd, n_layer, n_head, n_head_kv, n_ctx_train, n_params, size
  - **Automatic Cache Management**: Transparent caching with zero API changes or user intervention required
  - **Property-Heavy Workload Optimization**: Perfect for memory estimation and analysis operations (3.2M workloads/s)
  - **Zero API Disruption**: Fully backward compatible with existing code and interfaces

  **Batch Operations Optimizations** (Priority 3 - Medium Risk, High Performance Benefit):
  - **Batch Processing Speed**: Achieved exceptional batch operation performance with nogil optimizations
  - **GIL-Free Operations**: Core batch setup loops run without Python GIL overhead using Cython nogil decorators
  - **Optimized Functions**: Enhanced `set_batch()`, `add_sequence()`, `set_last_logits_to_true()`, and `llama_batch_get_one()`
  - **Memory Access Patterns**: Separated Python object access from C array operations for maximum efficiency
  - **Performance Scaling**: 2.1M batch creations/s (small), 813K/s (medium), 469K/s (large), 113K/s (very large batches)
  - **Batch Workload Optimization**: 985K workloads/s for typical 32-token batch processing workflows
  - **Zero API Changes**: Fully backward compatible with existing batch processing code

  **Context Operations Optimizations** (Priority 5 - Medium Risk, High Performance Benefit):
  - **Inference Performance**: Optimized critical inference path operations with reduced Python overhead
  - **Decode Optimization**: Enhanced `LlamaContext.decode()` with streamlined error handling and optimized parameter access
  - **Sampling Optimization**: Improved `LlamaSampler.sample()` with explicit Cython variable usage and reduced overhead
  - **Conservative Approach**: Focused on Python/Cython overhead reduction while maintaining full API compatibility
  - **Inference Speed**: 22 inference cycles/s with 45.6ms average time per decode+sample cycle
  - **Error Handling**: Optimized branching with `elif` patterns for faster conditional execution
  - **Zero API Disruption**: Fully backward compatible with existing context and sampling code

  **Memory Management Optimizations** (Priority 4 - Higher Complexity, High Performance Benefit):
  - **Memory Pool Systems**: Implemented sophisticated token and batch memory pooling for efficient object reuse
  - **Token List Pooling**: `TokenMemoryPool` class provides reusable token lists for common sizes (8-512 tokens)
  - **Batch Object Pooling**: `BatchMemoryPool` class enables LlamaBatch object reuse across inference operations
  - **Tokenization Performance**: 8.6-10.6% improvement in tokenization speed through memory pool integration
  - **Batch Creation Performance**: 6.1-7.7% improvement for medium-to-large batches (32-128 tokens)
  - **High-Pressure Performance**: 22.1% improvement under intensive allocation patterns (1.08M → 1.39M allocs/s)
  - **Smart Allocation Strategy**: Automatic pool bypass for very large objects, optimal reuse for common sizes
  - **Comprehensive API**: Public functions for pool management, statistics, and explicit pooled object creation
  - **Overall Performance Gain**: 8.8% faster performance across combined memory-intensive operations

### Technical Implementation

- **xllamacpp Integration**: Adapted best practices from xllamacpp fork analysis
  - Implemented memory estimation algorithms based on xllamacpp's sophisticated approach
  - Maintained compatibility with existing cyllama architecture and design principles
  - Selective integration focusing on memory management without breaking existing functionality

- **Performance Optimization**: Efficient memory calculation algorithms
  - Architecture-specific memory computation with minimal overhead
  - Intelligent layer size estimation based on quantization schemes
  - Optimized graph memory calculations with attention mechanism considerations

## [0.1.3]

### Added

- **Whisper Support**: Added Whisper.cpp integration for speech-to-text functionality
  - New `src/cyllama/whisper/` module with Cython bindings for whisper.cpp
  - `whisper_cpp.pyx` - Primary Whisper Cython extension module
  - `tests/test_whisper.py` - Comprehensive Whisper test suite
  - `samples/jfk.wav` - Sample audio file for testing
  - `scripts/download-ggml-model.sh` - Script to download Whisper models

- **Whisper CLI**: Complete Python CLI wrapper equivalent to whisper.cpp CLI
  - `src/cyllama/whisper/cli.py` - Full command-line interface for speech-to-text
  - Support for all major whisper.cpp CLI parameters and options
  - Multiple output formats: TXT, SRT, VTT, CSV, JSON (basic and full), LRC
  - Audio file loading with automatic resampling to 16kHz
  - WAV format support for 8, 16, 24, and 32-bit audio files
  - GPU acceleration support with Metal backend on macOS
  - Language detection and translation capabilities
  - Comprehensive argument parsing with help documentation

### Changed

- **Major Code Restructuring**: Reorganized codebase to support multiple AI modalities
  - Moved LLaMA-specific code to `src/cyllama/llama/` subdirectory
  - Separated Whisper functionality into `src/cyllama/whisper/` subdirectory
  - Updated module imports and package structure
  - Added `src/cyllama/__main__.py` for CLI entry point

- **Text-to-Speech Improvements**: Enhanced TTS functionality with better C++ compatibility
  - Improved TTS generation to match llama.cpp reference implementation
  - Fixed audio quality issues and generation completeness
  - Better speaker template management and prompt construction

- **Build System Updates**: Enhanced build configuration for multi-modal support
  - Updated `Makefile` with Whisper-specific build targets
  - Enhanced `setup.py` for multi-extension compilation
  - Updated `MANIFEST.in` and `pyproject.toml` for new package structure

### Fixed

- **Token Decoding**: Fixed `token_to_piece` method corruption issues
  - Resolved text output with replacement characters
  - Proper buffer length handling for token decoding
  - Added error handling for negative return values

- **Whisper Transcription**: Enabled and fixed the `full()` method in Whisper wrapper
  - Uncommented and activated the main transcription functionality
  - Fixed Cython compilation issues with proper memory view handling
  - Corrected import paths for whisper.pxd module
  - Proper error handling for transcription failures

## [0.1.2]

- Updated to latest release of `llama.cpp`: `b6374`

- Added unit tests

- Changed `cyllama.pyx` and tests to apply more consistent naming of Llama-type classes.

## [0.1.0]

- Moved cyllama code from [llamalib](https://github.com/shakfu/llamalib) to this repo
- Added low-level simple wrapper using cyllama
- Added high-level simple wrapper using cyllama
