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

Link Cython extensions against pre-built `.dylib`/`.so` files from llama.cpp GitHub releases (e.g., `llama-b8522` tarball from https://github.com/ggml-org/llama.cpp/releases).

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

---

## C-API-Only Refactor: Detailed Scoping

A refactor to eliminate internal C++ API dependencies would make dynamic linking viable. This section maps exactly which internal symbols are used, what public API replacements exist, and what must be reimplemented in Python.

### Current Internal C++ API Usage Heat Map

```
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
- **gguf.pxd** (138 lines): 40+ functions -- zero usage

These can be removed immediately with no behavioral change, reducing the declared surface by ~400 lines.

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
```
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

```
REMAINING INTERNAL C++ DEPENDENCIES:
  speculative.h  -- 7 functions (optional feature, can keep static build)

REMAINING PUBLIC C API DEPENDENCIES (all dynamically linkable):
  llama.h        -- 233 symbols via libllama.dylib
  ggml.h         -- via libggml*.dylib
  mtmd.h         -- 25 functions via libmtmd.dylib (already in releases)

CUSTOM C++ (compiled into extension, no external dependency):
  tts.cpp        -- TTS helpers
  json_schema.cpp -- JSON schema helpers
  mongoose.c     -- HTTP server
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
