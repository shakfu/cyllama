# Static Linking vs Dynamic Linking Analysis

## Current Strategy: Static Linking + Cython Wrappers

llama.cpp is built from source via `manage.py`, producing `.a` archives that are linked into `.so` Python extension modules at build time. Cython `.pxd` files declare the C/C++ API, and everything gets baked into `llama_cpp.cpython-*.so`.

### Build Flow

1. `manage.py` clones llama.cpp at a pinned commit (currently `b8522`)
2. CMake builds static libraries (`libllama.a`, `libggml*.a`, `libcommon.a`, etc.)
3. Libraries and ~56 headers are copied to `thirdparty/llama.cpp/{lib,include}/`
4. scikit-build-core runs the root `CMakeLists.txt`, which:
   - Transpiles `.pyx` to `.cxx` via Cython

   - Compiles `.cxx` to `.o`

   - Links `.o` + all `.a` archives into final `.so` extension modules

## Alternative Strategy: Dynamic Linking Against Pre-built Releases

Link Cython extensions against pre-built `.dylib`/`.so` files from llama.cpp GitHub releases (e.g., `llama-b8522` tarball from <https://github.com/ggml-org/llama.cpp/releases>).

### What the Pre-built Releases Contain

The release tarball (e.g., `llama-b8522-bin-macos-arm64.tar.gz`) ships:

- **Dynamic libraries**: `libllama.dylib`, `libggml.dylib`, `libggml-base.dylib`, `libggml-cpu.dylib`, `libggml-metal.dylib`, `libggml-blas.dylib`, `libggml-rpc.dylib`, `libmtmd.dylib`

- **CLI tools**: `llama-cli`, `llama-server`, `llama-quantize`, etc.

- **No headers** (`include/` directory is absent)

- **No `libcommon`** or `libcpp-httplib`

### Exported Symbols

- **233 stable C API symbols** (`_llama_*`) -- these are the public API from `llama.h`

- **C++ mangled symbols** (`__Z*`) -- internal, compiler-specific, not ABI-stable

---

## Pros of Dynamic Linking

### 1. Dramatically Faster Builds

The llama.cpp compile is the build bottleneck (minutes). Skipping it reduces the build to just Cython transpile + link (seconds).

### 2. Decoupled Upgrade Cycle

Users can drop in a new llama.cpp release without rebuilding cyllama. Version bumps become a file swap rather than a full `make remake`.

### 3. Smaller Wheel Sizes

The `.so` extensions shrink significantly since llama.cpp code is external. You ship a thin binding layer, not the full inference engine.

### 4. Shared Memory Footprint

If other processes load the same `libllama.dylib`, the OS shares pages. With static linking each Python extension embeds its own copy.

### 5. Simpler CI Matrix

Test against pre-built release artifacts rather than building llama.cpp per-platform in CI. The release maintainers (ggml-org) already handle the platform matrix.

### 6. Eliminates Builder Complexity

~1000 lines of `manage.py` builder code for downloading, patching, building, and copying llama.cpp artifacts becomes unnecessary.

---

## Cons of Dynamic Linking

### 1. No Headers Shipped in Releases

The pre-built tarballs have no `include/` directory. The current build depends on ~56 headers. Options:

- Still clone the repo to get headers (partially defeats the purpose)

- Vendor the headers separately and pin them to the release version

- Use only the public C API headers fetched from the GitHub tag

### 2. `libcommon` Not in Release

The pre-built release exports `libllama`, `libggml*`, and `libmtmd`, but **not** `libcommon` or `libcpp-httplib`. The Cython wrappers (`common.pxd`) bind directly to `common.h` symbols (arg parsing, chat templates, sampling params). These are C++ internal symbols, not part of the stable C API. Options:

- Rewrite bindings to only use the 233 public `_llama_*` C symbols

- Still build `libcommon` from source (partially defeats the purpose)

> **Resolved**: All `libcommon` dependencies have been eliminated. Sampling, download, n-gram cache, speculative decoding, and batch helpers were rewritten to use public C APIs or pure Python. JSON schema-to-grammar conversion was replaced with a vendored pure Python implementation. `libcommon.a` is no longer linked in either static or dynamic builds.

### 3. C++ Name Mangling Fragility

Pre-built dylibs export C++ mangled symbols (`__Z*` names) that are compiler-specific and break across compiler versions, standard library versions, or optimization levels. The 233 `_llama_*` C symbols are stable; the C++ symbols are not. The `common.pxd` and `sampling.pxd` bindings depend on C++ APIs.

### 4. Runtime Dependency Management

Dylibs are needed at runtime, not just build time:

- `@rpath` resolution requires correct `install_name_tool` fixup or environment variables

- `pip install cyllama` would need to either bundle the dylibs (back to large wheels) or require the user to install llama.cpp separately

- Platform-specific dylib discovery (`DYLD_LIBRARY_PATH` on macOS, `LD_LIBRARY_PATH` on Linux, `PATH` on Windows)

### 5. ABI Compatibility Risk

llama.cpp has no ABI stability guarantee. Even the C API can change between releases (functions added/removed/signatures changed). With static linking you pin to an exact commit. With dynamic linking, a user swapping in a newer dylib could silently break things or segfault.

### 6. Backend Coverage Gaps

GitHub releases build one variant per platform (Metal for macOS-arm64, CUDA for specific Linux builds, etc.). The current system builds with exactly the backends the user wants. A pre-built release might not match -- e.g., no Vulkan macOS build, no SYCL build, no specific CUDA arch.

### 7. Whisper and Stable Diffusion Don't Apply

Neither whisper.cpp nor stable-diffusion.cpp ship pre-built releases in the same way. The from-source build pipeline (`manage.py`) is still required for those, so it cannot be fully eliminated.

### 8. `--whole-archive` Replaced by `dlopen` Complexity

The Linux `--whole-archive` workaround disappears, but you inherit the GGML backend plugin loading model, which uses `dlopen` internally -- a different kind of complexity.

---

## The Core Tension

The Cython wrappers bind **both** the public C API (`llama.h` -- 233 stable symbols) **and** internal C++ APIs (`common.h`, `sampling.h`, `chat.h` -- mangled, unstable). The pre-built releases only export the former reliably.

---

## Possible Hybrid Approach

- **Dynamic link against `libllama.dylib` + `libggml*.dylib`** for the core inference C API

- **Still build `libcommon.a` from source** (small, fast to compile) for C++ utility bindings

- **Vendor just the public headers** from the release tag

This buys less than it initially appears, because the common/sampling layer is where most of the Cython binding complexity lives.

> **Resolved**: The hybrid approach is no longer needed. All `libcommon` dependencies have been eliminated -- the extension uses only public C APIs and pure Python. Dynamic linking (`make build-dynamic`) now works without any source compilation of llama.cpp. Only `libcpp-httplib.a` is still built from source for the embedded server.

---

## C-API-Only Refactor: Detailed Scoping

> **Status: Complete.** All internal C++ API dependencies have been eliminated. The extension now uses only public C APIs (`llama.h`, `ggml.h`, `gguf.h`, `mtmd.h`) and pure Python. Dynamic linking is fully operational via `make build-dynamic`. The analysis below is retained for historical reference.

A refactor to eliminate internal C++ API dependencies would make dynamic linking viable. This section maps exactly which internal symbols are used, what public API replacements exist, and what must be reimplemented in Python.

### Current Internal C++ API Usage Heat Map

```text
HEADER          | SYMBOLS | ACTIVE | USAGE PATTERN
----------------+---------+--------+---------------------------
sampling.h      |   18    |   14   | 100% - core sampling chain
speculative.h   |    7    |    7   | 100% - draft model verify
ngram_cache.h   |    5    |    5   | 100% - self-speculative
download.h      |    4    |    4   | 100% - model downloading
mtmd.h          |   42    |   25   | 60%  - multimodal support
common.h        |  50+    |    3   | 6%   - batch utils + param conv
chat.h          |   25    |    0   | 0%   - declared but unused
log.h           |   10    |    0   | 0%   - declared but unused
gguf.h          |   40    |    0   | 0%   - declared but unused
```

### Dead Code: Immediate Cleanup

Three `.pxd` files declare symbols that are **never called** in any `.pyx`/`.pxi`:

- **chat.pxd** (202 lines): 12 structs, 13 functions -- zero usage

- **log.pxd** (60 lines): 10+ functions -- zero usage

**Note**: `gguf.pxd` was initially thought to be dead code but is actively used by the `GgufContext` class. It wraps the public `gguf.h` API (part of libggml), not an internal C++ API, so it does not block dynamic linking.

These can be removed immediately with no behavioral change, reducing the declared surface by ~260 lines.

### Module-by-Module Replacement Analysis

#### 1. Sampling (sampling.h) -- REPLACEABLE via public API

**Currently used** (14 functions via `sampling.pxi`, wrapping `CommonSampler` class):

| Internal C++ Function | Where Called | Public C API Replacement |
|---|---|---|
| `common_sampler_init()` | sampling.pxi:13 | Build chain with `llama_sampler_chain_init()` + individual `llama_sampler_init_*()` |
| `common_sampler_free()` | sampling.pxi:20 | `llama_sampler_free()` |
| `common_sampler_accept()` | sampling.pxi:25 | `llama_sampler_accept()` |
| `common_sampler_reset()` | sampling.pxi:29 | `llama_sampler_reset()` |
| `common_sampler_clone()` | sampling.pxi:33 | `llama_sampler_clone()` |
| `common_sampler_sample()` | sampling.pxi:40 | `llama_sampler_sample()` |
| `common_sampler_sample_and_accept_n()` | sampling.pxi:63 | Loop: `llama_sampler_sample()` + `llama_sampler_accept()` |
| `common_sampler_get_seed()` | sampling.pxi:68 | Track seed in Python when constructing chain |
| `common_sampler_last()` | sampling.pxi:80 | Track last token in Python |
| `common_sampler_print()` | sampling.pxi:84 | Reimplement in Python (debug utility) |
| `common_sampler_prev_str()` | sampling.pxi:88 | Reimplement in Python (debug utility) |
| `common_sampler_type_to_chr()` | sampling.pxi:94 | Reimplement in Python (string mapping) |
| `common_sampler_type_to_str()` | sampling.pxi:98 | Reimplement in Python (string mapping) |
| `common_sampler_types_from_names()` | sampling.pxi:100+ | Reimplement in Python (name lookup) |

**What `common_sampler_init()` actually does** (the key function): It reads `common_params_sampling` and constructs a `llama_sampler` chain by calling the public sampler init functions in order. This is the main value-add -- it's a ~100-line convenience wrapper.

**Replacement strategy**: Rewrite `CommonSampler.__init__()` in Python/Cython to directly call:

```text
llama_sampler_chain_init()
llama_sampler_chain_add(chain, llama_sampler_init_top_k(k))
llama_sampler_chain_add(chain, llama_sampler_init_top_p(p, min_keep))
llama_sampler_chain_add(chain, llama_sampler_init_min_p(p, min_keep))
llama_sampler_chain_add(chain, llama_sampler_init_temp(temp))
llama_sampler_chain_add(chain, llama_sampler_init_penalties(...))
llama_sampler_chain_add(chain, llama_sampler_init_grammar(vocab, grammar_str, root))
# ... etc for each enabled sampler
```

The public API now has **15+ sampler init functions** covering all sampler types:

- `llama_sampler_init_top_k`, `_top_p`, `_min_p`, `_temp`, `_temp_ext`

- `llama_sampler_init_xtc`, `_typical`, `_top_n_sigma`

- `llama_sampler_init_mirostat`, `_mirostat_v2`

- `llama_sampler_init_grammar`, `_grammar_lazy_patterns`

- `llama_sampler_init_penalties` (repetition/frequency/presence)

- `llama_sampler_init_dry` (DRY sampler -- actually NEW, not in common_sampler)

- `llama_sampler_init_adaptive_p`, `_logit_bias`, `_infill`

**Effort**: Medium. The chain construction logic is ~100 lines of C++ that becomes ~100 lines of Python. The debug/print utilities are trivial. `common_params_sampling` struct (30+ fields) needs a Python dataclass replacement.

**Risk**: Low. This is straightforward mapping.

#### 2. Batch Management (common.h) -- TRIVIALLY REPLACEABLE

**Currently used** (2 functions):

| Internal C++ Function | Where Called | Replacement |
|---|---|---|
| `common_batch_clear()` | llama_cpp.pyx:668 | Zero out `llama_batch.n_tokens` (1 line) |
| `common_batch_add()` | llama_cpp.pyx:664 | Set batch fields at index, increment n_tokens (~5 lines) |

**Replacement strategy**: Inline into `LlamaBatch` class methods. These are trivial array manipulation wrappers.

**Effort**: Trivial. ~10 lines of Cython.

#### 3. Parameter Conversion (common.h) -- ELIMINABLE

**Currently used** (1 function):

| Internal C++ Function | Where Called | Replacement |
|---|---|---|
| `common_context_params_to_llama()` | llama_cpp.pyx:1032 | Set `llama_context_params` fields directly in Python |

**Replacement strategy**: The Python code already sets most params individually. The converter just maps `common_params` fields to `llama_context_params` fields. Do this directly in `LlamaContextParams.__init__()`.

**Effort**: Trivial. Field-by-field assignment already partially exists.

#### 4. Speculative Decoding (speculative.h) -- REQUIRES REIMPLEMENTATION

**Currently used** (7 functions, 100% active):

| Internal C++ Function | Where Called | Public API Equivalent |
|---|---|---|
| `common_speculative_init()` | speculative.pxi:111 | None -- manages draft model state |
| `common_speculative_free()` | speculative.pxi:118 | None |
| `common_speculative_is_compat()` | speculative.pxi:131 | None -- checks vocab compatibility |
| `common_speculative_begin()` | speculative.pxi:143 | None -- prepares KV cache state |
| `common_speculative_draft()` | speculative.pxi:168 | None -- runs draft model inference |
| `common_speculative_accept()` | speculative.pxi:188 | None -- verify/accept tokens |
| `common_speculative_print_stats()` | speculative.pxi:192 | None |

**No public API equivalent exists.** Speculative decoding orchestrates two models (draft + target) with coordinated KV cache management. This is ~500 lines of C++ with nontrivial algorithmic content (tree-based verification, KV cache rollback).

**Replacement strategy options**:

1. **Reimplement in Python**: Use public `llama_decode()`, `llama_memory_seq_rm/cp()`, and sampler APIs to orchestrate the draft-verify loop. Feasible but significant effort.
2. **Keep as optional C++ dependency**: Build only `libcommon-speculative.a` from source for users who need this feature.
3. **Drop feature**: If speculative decoding is not a core use case.

**Effort**: High. ~500 lines of algorithmic C++ to reimplement, with subtle correctness requirements around KV cache state management.

**Risk**: Medium-high. KV cache coordination bugs would cause silent correctness issues.

#### 5. N-gram Cache (ngram_cache.h) -- REIMPLEMENTABLE IN PYTHON

**Currently used** (5 functions):

| Internal C++ Function | Where Called | Replacement |
|---|---|---|
| `common_ngram_cache_update()` | llama_cpp.pyx:3734 | Python dict-based n-gram tracking |
| `common_ngram_cache_draft()` | llama_cpp.pyx:3790 | Python lookup + draft generation |
| `common_ngram_cache_save()` | llama_cpp.pyx:3822 | Python pickle/json serialization |
| `common_ngram_cache_load()` | llama_cpp.pyx:3843 | Python pickle/json deserialization |
| `common_ngram_cache_merge()` | llama_cpp.pyx:3869 | Python dict merge |

**Replacement strategy**: The C++ implementation uses `std::unordered_map` with custom hash. A Python `dict` with tuple keys achieves the same thing. The cache is a mapping from n-gram (token tuple) to candidate next tokens.

**Effort**: Medium-low. ~200 lines of Python. Performance may be slightly worse for very large caches but acceptable for typical use.

#### 6. Model Downloading (download.h) -- REIMPLEMENTABLE IN PYTHON

**Currently used** (4 functions):

| Internal C++ Function | Where Called | Replacement |
|---|---|---|
| `common_get_hf_file()` | llama_cpp.pyx:3503 | Python `requests`/`httpx` + HuggingFace Hub API |
| `common_download_model()` | llama_cpp.pyx:3604 | Python HTTP download with progress |
| `common_list_cached_models()` | llama_cpp.pyx:3621 | Python filesystem scan of cache dir |
| `common_docker_resolve_model()` | llama_cpp.pyx:3655 | Python Docker registry API calls |

**Replacement strategy**: These are HTTP/filesystem operations. Python is arguably *better* for this than C++ (easier error handling, richer HTTP libraries, async support). The `huggingface_hub` Python package already provides most of this functionality.

**Effort**: Medium. ~300 lines of Python, but well-trodden ground with existing libraries.

**Risk**: Low. Network I/O is not performance-sensitive.

#### 7. Multimodal / MTMD (mtmd.h) -- NOT REPLACEABLE

**Currently used** (25 functions): Image/audio tokenization, bitmap handling, chunk processing, encoding.

**No public llama.h equivalent.** `mtmd.h` is itself a public C API (not C++ mangled), and `libmtmd.dylib` IS included in pre-built releases. The functions use `extern "C"` linkage.

**This module does NOT block dynamic linking.** It can link against `libmtmd.dylib` directly. Headers would still need to be vendored.

**Effort**: None for dynamic linking. Just need the header file.

#### 8. TTS Helpers (tts_helpers.pxi) -- CUSTOM C++ CODE

The TTS helper functions (`save_wav16`, `fill_hann_window`, `irfft`, `fold`, `process_text`, etc.) are **not from llama.cpp** -- they're custom C++ in `src/cyllama/llama/helpers/tts.cpp`.

**These do not block dynamic linking.** They're compiled directly into the extension module.

### Summary: What the Refactor Looks Like

#### Phase 1: Dead Code Removal (Immediate, ~0.5 day)

- Delete `chat.pxd` (or keep only struct declarations if needed for future)

- Delete `log.pxd`

- Delete `gguf.pxd`

- Remove corresponding unused declarations from `common.pxd`

- **Result**: ~400 fewer lines of C++ declarations to maintain

#### Phase 2: Trivial Replacements (Easy, ~1 day)

- Inline `common_batch_clear()` / `common_batch_add()` into `LlamaBatch` (~10 lines of Cython)

- Replace `common_context_params_to_llama()` with direct field assignment

- **Result**: `common.h` dependency drops to zero active functions

#### Phase 3: Sampling Refactor (Medium, ~3 days)

- Create Python `SamplingParams` dataclass replacing `common_params_sampling` struct

- Rewrite `CommonSampler.__init__()` to build `llama_sampler` chains via public API

- Replace `common_sampler_sample/accept/reset/clone` with direct `llama_sampler_*` calls

- Reimplement debug utilities (`print`, `prev_str`, `type_to_str`) in Python

- **Result**: `sampling.h` dependency eliminated entirely

#### Phase 4: Download/Cache in Python (Medium, ~2 days)

- Reimplement HuggingFace file resolution using `huggingface_hub` or raw HTTP

- Reimplement model download with progress reporting

- Reimplement cache listing as directory scan

- Reimplement Docker registry resolution

- **Result**: `download.h` dependency eliminated

#### Phase 5: N-gram Cache in Python (Medium-low, ~1 day)

- Reimplement n-gram cache as Python `dict[tuple[int, ...], list[int]]`

- Reimplement save/load as JSON or pickle

- **Result**: `ngram_cache.h` dependency eliminated

#### Phase 6: Speculative Decoding (Hard, ~5 days)

- Reimplement draft-verify loop using public `llama_decode()` + `llama_memory_seq_*()` APIs

- Careful testing required for KV cache state management correctness

- **Result**: `speculative.h` dependency eliminated

**OR**: Keep speculative decoding as an optional feature requiring from-source build.

### Post-Refactor State

After Phases 1-5 (skipping Phase 6):

```text
REMAINING INTERNAL C++ DEPENDENCIES:
  speculative.h  -- 7 functions (optional feature, can keep static build)

REMAINING PUBLIC C API DEPENDENCIES (all dynamically linkable):
  llama.h        -- 233 symbols via libllama.dylib
  ggml.h         -- via libggml*.dylib
  mtmd.h         -- 25 functions via libmtmd.dylib (already in releases)

CUSTOM C++ (compiled into extension, no external dependency):
  tts.cpp        -- TTS helpers
  mongoose.c     -- HTTP server

PURE PYTHON (no C++ compilation needed):
  json_schema_to_grammar.py -- JSON schema to GBNF grammar conversion
```

This state enables dynamic linking for the core use case (inference, sampling, tokenization, multimodal) with speculative decoding as an opt-in feature requiring from-source build.

### Estimated Total Effort

| Phase | Effort | Risk | Dependencies Eliminated |
|---|---|---|---|
| 1. Dead code removal | 0.5 day | None | chat.h, log.h, gguf.h |
| 2. Trivial replacements | 1 day | None | common.h (batch, params) |
| 3. Sampling refactor | 3 days | Low | sampling.h |
| 4. Download in Python | 2 days | Low | download.h |
| 5. N-gram cache | 1 day | Low | ngram_cache.h |
| 6. Speculative decoding | 5 days | Medium-high | speculative.h |
| **Total (Phases 1-5)** | **~7.5 days** | **Low** | **All except speculative** |
| **Total (all phases)** | **~12.5 days** | **Medium** | **All internal C++ APIs** |

### Key Insight: The 80/20 Split

Phases 1-3 (~4.5 days) eliminate the three highest-churn dependencies (common.h, sampling.h, dead declarations) and cover the core inference path. This alone makes dynamic linking viable for the primary use case. Phases 4-6 are incremental wins with diminishing returns.

---

## Conclusion

Dynamic linking is attractive for build speed and decoupling, but the current codebase depends too heavily on internal C++ symbols (`common`, `sampling`, `chat`) that are not exported stably in pre-built releases. The practical path to dynamic linking requires first narrowing the Cython binding surface to the public C API, which is a significant but tractable refactor (~7.5 days for the critical path, excluding speculative decoding). For a project that also wraps whisper.cpp and stable-diffusion.cpp (which lack pre-built releases), the from-source pipeline cannot be fully eliminated regardless.

The most impactful finding is that **llama.cpp's public sampler API has expanded dramatically** and now covers all sampler types. The `common_sampler_*` wrapper -- previously the primary blocker -- is now a thin convenience layer over public API functions. This makes the refactor significantly more feasible than it would have been even a few releases ago.

### Decision Matrix

| Factor | Static (Current) | Dynamic (Alternative) |
|---|---|---|
| Build speed | Slow (minutes) | Fast (seconds) |
| Upgrade friction | Full rebuild | File swap (if ABI-compatible) |
| Wheel size | Large | Small (if dylibs external) |
| ABI safety | Pinned at build | Risk of mismatch |
| Backend flexibility | Full control | Limited to release variants |
| C++ internal access | Full | Fragile / unavailable |
| Whisper/SD support | Same pipeline | Still needs from-source |
| Distribution simplicity | Self-contained | External dependency |

---

## Completed Refactoring (Phases 1-5 + field fix)

All phases implemented and verified (full test suite passing).

### Phase 1: Dead Code Removal

- Deleted `chat.pxd` (202 lines) and `log.pxd` (60 lines) -- cimported but never referenced

- Removed `cimport chat` and `cimport log` from `llama_cpp.pyx`

- **Kept** `gguf.pxd` (actively used by `GgufContext` class; wraps public `gguf.h` API)

### Phase 2: Inlined Batch Helpers

- Replaced `common.common_batch_add()` with direct array assignment in `LlamaBatch.add()`

- Replaced `common.common_batch_clear()` with `self.p.n_tokens = 0` in `LlamaBatch.clear()`

### Phase 3: Sampling Refactor

- Deleted `sampling.pxd` (117 lines) and removed `cimport sampling` from `llama_cpp.pyx`

- Rewrote `CommonSampler` in `sampling.pxi` to use only public `llama_sampler_*` API:

  - Chain construction via `llama_sampler_chain_init()` + `llama_sampler_init_*()` for each sampler type

  - Grammar as separate sampler with rejection sampling (fast path: sample then check; slow path: constrain then resample)

  - Token history tracked in Python `collections.deque` (replacing C++ `ring_buffer`)

- Reimplemented all sampler helper functions in Python:

  - `type_to_chr()`, `type_to_str()` via Python dicts

  - `types_from_names()`, `types_from_chars()` via Python lookups

  - `CommonSampler.print()` via `llama_sampler_chain_n/get/name`

  - `CommonSampler.prev_str()` via `llama_token_to_piece` public API

### Phase 4: Download Functions in Python

- Deleted `download.pxd` (55 lines) and removed `cimport download` from `llama_cpp.pyx`

- Rewrote all 4 download functions using Python `urllib.request` (stdlib, no external deps):

  - `get_hf_file()`: HF manifest API + JSON caching

  - `download_model()`: HTTP download with ETag caching, resume, retry

  - `list_cached_models()`: Filesystem scan of `manifest=*.json` files

  - `resolve_docker_model()`: Docker Hub auth + manifest + blob download

### Phase 5: N-gram Cache in Python

- Deleted `ngram_cache.pxd` (74 lines) and removed `cimport ngram_cache` from `llama_cpp.pyx`

- Rewrote `NgramCache` as a pure Python class:

  - Cache stored as `dict[tuple, dict[int, int]]` (ngram -> {next_token: count})

  - Three-tier draft algorithm (context/dynamic/static) with confidence thresholds

  - Binary save/load format compatible with C++ implementation via `struct` module

### Field Mismatch Fix + Param Conversion Inline

- Fixed `common.pxd`: `flash_attn` (bint) -> `flash_attn_type` (llama_flash_attn_type enum)

- Fixed `llama.pxd`: Added missing `embeddings` bool field to `llama_context_params`

- Inlined `common_context_params_to_llama()` as direct field assignment in `LlamaContextParams.from_common_params()`

- **Zero `common.h` function calls remain** -- only struct/enum type declarations

### Remaining Internal C++ Dependencies

| Module | Status | Reason |
|---|---|---|
| `common.h` (structs/enums) | Kept | Type declarations only (compile-time, no link dependency) |
| `speculative.h` | Kept | No public API equivalent, required for speculative decoding |

### Dynamic Linking Prototype (Validated)

Dynamic linking against pre-built llama.cpp releases has been implemented and validated:

```bash
# Build with dynamic linking against a pre-built release
WITH_DYLIB=1 LLAMACPP_DYLIB_DIR=/path/to/llama-b8522 make build
```

**CMake options**:

- `WITH_DYLIB=ON` -- link against shared `.dylib`/`.so` files instead of static `.a` archives

- `LLAMACPP_DYLIB_DIR=/path/to/release` -- directory containing pre-built shared libraries

- `SD_USE_VENDORED_GGML=ON` (default) -- link stable-diffusion against its own vendored ggml; set to `OFF` to share llama.cpp's ggml (not recommended for GPU backends)

**How it works**:

- Core llama.cpp libraries (`libllama`, `libggml*`, `libmtmd`) linked as shared libraries

- `libcpp-httplib.a` still built from source (not in releases, needed for embedded server)

- `libcommon.a` is no longer linked -- JSON schema-to-grammar conversion is now pure Python, and all other `common.h` dependencies have been eliminated

- Shared libraries copied alongside extension modules (`cyllama/llama/`) so `@loader_path`/`$ORIGIN` RPATH resolves correctly

- Whisper and Stable Diffusion remain statically linked (no pre-built releases available). Stable Diffusion uses its own vendored ggml by default; set `SD_USE_VENDORED_GGML=0` to share llama.cpp's ggml (not recommended for GPU backends due to ggml version incompatibilities)

**Validated results** (macOS arm64, b8522 release):

- Extension size: 1.6 MB (vs ~15 MB with static linking)

- All tests passing (120+ tests verified including generation, batching, chat, context)

- Pre-built dylibs installed alongside extension: `libllama.dylib` (2.3MB), `libggml*.dylib`, `libmtmd.dylib`

---

## Dev Branch Merge Analysis (2026-03-28)

Analysis of the `dev` branch changes since v0.1.21 (`main`), covering 11 commits, 63 files, +27,633 / -4,246 lines.

### Pros of merging

**1. Eliminates internal C++ API fragility (the biggest win)**
The branch removes all `common.h`/`libcommon` dependencies (~4200 lines deleted across `.pxd`/`.pxi` files). The extension now uses only public C APIs (`llama.h`, `ggml.h`, `gguf.h`, `mtmd.h`). This decouples cyllama from llama.cpp internals that break between releases, making future upstream version bumps significantly safer.

**2. Enables dynamic linking**
`WITH_DYLIB=1` is a new build mode that links against pre-built llama.cpp releases instead of compiling from source. Extension size drops from ~15 MB to ~1.6 MB, build time goes from minutes to seconds. This unblocks faster iteration and simpler CI.

**3. Build system consistency**

- Unified ggml across all three backends (was 0.9.5 for SD, 0.9.8 for the rest)

- Unified GPU backend flags (`GGML_*` applies to all components, no more separate `SD_METAL`)

- sqlite-vector vendored and built from CMake (no more `.gitignore` hacks for pre-built binaries)

**4. Correctness fixes**

- `flash_attn` / `flash_attn_type` struct mismatch (potential silent memory corruption)

- Missing `embeddings` field in `llama_context_params` (struct layout mismatch)

- `_build_info.py` reporting wrong ggml version for SD

**5. Cleaner test suite**
Removed ~220 lines of tests for `CommonParams`/`CommonParamsSampling` that tested deleted wrapper classes. The remaining tests cover the actual public API.

### Cons of merging

**1. Large, multi-concern changeset**
11 commits touching 63 files with +27,633 / -4,246 lines. Hard to review atomically or bisect if something regresses. The changes span: API refactor, build system overhaul, new build mode, vendored dependency, bug fixes. Ideally these would be separate PRs, but they're intertwined (e.g., public API refactor enables dynamic linking).

**2. Removed public Python API surface**
`CommonParams`, `CommonParamsSampling`, `CommonSampler` are gone. Any downstream code using these classes will break. The replacements (`LlamaContextParams`, `LlamaSampler`) exist but the migration isn't documented beyond the changelog. This is a breaking change that warrants a minor version bump.

**3. Vendored sqlite-vector adds ~24K lines of C to the repo**
The `thirdparty/sqlite-vector/` directory includes `sqlite3.h` (13,773 lines) and several SIMD distance implementations. This bloats the repo and creates a maintenance burden for keeping it in sync with upstream. The alternative (the pre-built binary approach) had its own problems, but the tradeoff is worth being explicit about.

**4. Dynamic linking is new and lightly validated**
The changelog says "120+ tests verified" for dynamic mode, but the test matrix for dynamic linking across platforms (Linux, macOS) and backends (CUDA, Vulkan) is likely thin. A regression in dynamic mode could be hard to catch without CI coverage.

**5. `SD_USE_VENDORED_GGML` adds configuration complexity**
A build option with interactions across static/dynamic modes, multiple backends, and two different ggml versions. Now defaults to ON (vendored) after CUDA image generation crashes were traced to ggml version incompatibilities between llama.cpp and stable-diffusion.cpp. The shared-ggml path (`SD_USE_VENDORED_GGML=0`) remains available but is not recommended for GPU backends.

### Recommendation

The public API refactor and correctness fixes alone justify merging. The core risk is the breaking API change (`CommonParams` removal) -- if there are downstream consumers, they need a migration path. If this is primarily internal/personal use, the tradeoff is clearly positive.

One option to reduce risk: merge now, tag as `0.2.0` (semver minor bump for breaking changes), and add CI for the dynamic linking path before advertising it as stable.
