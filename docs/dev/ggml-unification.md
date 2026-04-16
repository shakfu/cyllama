# ggml unification: eliminating the 210 MB SD bloat in CUDA wheels

## Summary

The `cyllama_cuda12` wheel shipped at **157 MB compressed / ~427 MB
uncompressed** in 0.2.9, exceeding PyPI's 100 MB default limit. Two files
dominated: `cyllama/sd/stable_diffusion.cpython-*.so` (~210 MB) and
`cyllama/llama/libggml-cuda.so` (~202 MB). Both carried the same set of ggml
CUDA kernels -- the SD extension had `libggml-cuda.a` whole-archive linked
into it, while llama.cpp shipped the same backend as a separate shared library.

The fix required three things:

1. **Source sync** (`_sync_ggml_abi()`): replace SD's vendored ggml with
   llama.cpp's copy so enum values and API signatures match.
2. **Struct layout sync** (`GGML_MAX_NAME=128`): propagate SD's larger
   `GGML_MAX_NAME` to the llama.cpp shared lib build so `ggml_tensor` struct
   layout is identical on both sides.
3. **Dynamic link path** (`WITH_DYLIB=1`, `SD_USE_VENDORED_GGML=0`): link SD
   against llama.cpp's shared ggml dylibs instead of whole-archiving the
   static `.a` files.

All three are now implemented. The SD `.so` dropped from 210 MB to 23 MB.

---

## Problem statement

### Wheel size

From `dist/cyllama_cuda12-0.2.9-cp313-cp313-manylinux2014_x86_64.*.whl`:

| File | Compressed (approx) | Uncompressed | Notes |
|---|---:|---:|---|
| `cyllama/sd/stable_diffusion.cpython-*.so` | ~75 MB | 210 MB | Contains whole-archive-linked `libggml-cuda.a` |
| `cyllama/llama/libggml-cuda.so` | ~75 MB | 202 MB | Standalone ggml CUDA backend for llama bindings |
| `cyllama/llama/libllama.so.0` | ~1 MB | 3.2 MB | |
| `cyllama/llama/llama_cpp.cpython-*.so` | ~0.6 MB | 1.4 MB | Cython bindings |
| everything else | ~5 MB | ~10 MB | Python sources, small libs, libs duplicated by auditwheel |
| **total** | **~157 MB** | **~427 MB** | |

The two CUDA-carrying `.so` files held effectively the same compiled kernel
set. Confirmed by `nm -D` showing identical ggml_cuda kernel symbols in both.

### PyPI constraint

PyPI's default per-file limit is 100 MB. At 157 MB the wheel required a
size-increase grant.

### Non-goal: splitting the distribution

SD is a core feature of cyllama, not an optional add-on. This document
pursues the single-wheel path.

---

## Why the current shape exists: the 0.2.9 workaround

Release 0.2.9 made `SD_USE_VENDORED_GGML=ON` the default. The rationale:

> Fixes a `ggml_backend_tensor_copy` assertion crash ("cannot copy tensors
> with different layouts") during CUDA image generation caused by subtle ggml
> version incompatibilities between llama.cpp and stable-diffusion.cpp.

### Historical root cause (pre-0.2.9)

SD's vendored ggml was at an older release. Upstream inserted new operations
into the `ggml_op` enum, shifting every subsequent op's integer ordinal. SD
was compiled against older headers but called into llama.cpp's newer
`libggml-cuda.so` at runtime. The mismatched op ids produced compute graphs
with wrong output shapes, tripping `GGML_ASSERT(ggml_are_same_layout(src,
dst))` during the first cross-backend tensor copy.

0.2.9 short-circuited this by statically linking SD against its own vendored
ggml with `-Wl,--whole-archive`. Correct, but every symbol in
`libggml-cuda.a` ended up inside the SD extension.

---

## The two ABI problems

Unifying the ggml library required solving two distinct ABI mismatches.

### Problem 1: enum ordinal drift (solved by `_sync_ggml_abi`)

SD's vendored ggml (0.9.8) and llama.cpp's ggml (0.9.11) define different
integer values for `ggml_op` enum members. The fix replaces SD's vendored
`ggml/` directory with llama.cpp's copy at build time:

```python
# scripts/manage.py -- StableDiffusionCppBuilder._sync_ggml_abi()
shutil.rmtree(sd_ggml)
shutil.copytree(llama_ggml, sd_ggml)
```

Triggered when `SD_USE_VENDORED_GGML=0`. After the sync, SD compiles against
the same ggml headers that the shared dylibs were built from. The enum
values match; the original 0.2.9 crash mode is eliminated.

This was sufficient for simple models (SDXL Turbo, 512x512). More complex
models revealed a second, subtler problem.

### Problem 2: GGML_MAX_NAME struct layout mismatch

#### Discovery

With the source sync in place, SDXL Turbo image generation worked
perfectly. But FLUX-like models (`z_image_turbo`) with `--offload-to-cpu
--vae-on-cpu` crashed with the same `ggml_are_same_layout` assertion:

```
ggml-backend.cpp:478: GGML_ASSERT(ggml_are_same_layout(src, dst)
    && "cannot copy tensors with different layouts") failed
```

The native `sd-cli` binary (statically linked, same ggml source after sync)
worked fine for the same model and parameters.

#### Diagnosis

Instrumenting `ggml_backend_tensor_copy` to print tensor details before the
assertion revealed:

```
src name=leaf_382 type=0 ne=[0,1,1,1]    nb=[4,10240,10240,10240]
dst name=          type=0 ne=[2560,1,1,1] nb=[4,10240,10240,10240]
```

The source tensor had `ne[0]=0` but strides consistent with `ne[0]=2560`
(`nb[1] = 2560 * 4 = 10240`). This meant the tensor was originally created
with `ne[0]=2560` but the field was subsequently overwritten with zeros --
classic struct layout corruption.

#### Root cause

`ggml.h` defines `GGML_MAX_NAME` with a `#ifndef` guard:

```c
// build/llama.cpp/ggml/include/ggml.h
#ifndef GGML_MAX_NAME
#   define GGML_MAX_NAME  64
#endif
```

stable-diffusion.cpp's `CMakeLists.txt:233` overrides this for its own build:

```cmake
if (NOT SD_USE_SYSTEM_GGML)
    add_definitions(-DGGML_MAX_NAME=128)
endif()
```

And `ggml_extend.hpp:94` enforces it:

```cpp
static_assert(GGML_MAX_NAME >= 128, "GGML_MAX_NAME must be at least 128");
```

The `ggml_tensor` struct contains `char name[GGML_MAX_NAME]`. With the
mismatch:

- **SD's compiled code** sees `sizeof(ggml_tensor)` with a 128-byte `name`
  field.
- **llama.cpp's shared libs** see `sizeof(ggml_tensor)` with a 64-byte
  `name` field.

Every field after `name` in the struct (`extra`, `padding`) is at a
different byte offset. When SD writes to a tensor's `ne[0]` through what it
thinks is the correct offset, the shared lib reads from a different location.

The `_sync_ggml_abi()` function synced the ggml *source* but not the
*compile-time defines* -- SD's cmake still injected `-DGGML_MAX_NAME=128`
before including the synced headers, preempting the `#ifndef` guard.

#### Why simple models passed

SDXL Turbo's tensor names are short and its inference path doesn't require
cross-backend tensor copies with `--offload-to-cpu`. The 64-byte padding
difference happened not to corrupt the specific `ne[]` values accessed
during that model's execution.

FLUX models use `--offload-to-cpu --vae-on-cpu` (necessary because the ~10
GB model exceeds 8 GB VRAM), which forces
`offload_params_to_runtime_backend()` to copy hundreds of tensors between
CPU and CUDA backends via `ggml_backend_tensor_copy`. This exercises the
struct layout at every copy, exposing the corruption.

#### Why the native binary worked

`sd-cli` is statically linked. SD's cmake builds its own ggml libraries
with `-DGGML_MAX_NAME=128`, and that same value is used throughout the
binary. No ABI boundary, no mismatch.

---

## The fix

Three coordinated changes:

### 1. Source sync (existing)

`_sync_ggml_abi()` replaces SD's vendored `ggml/` tree with llama.cpp's
copy. This eliminates enum ordinal drift.

### 2. GGML_MAX_NAME propagation (new)

When `SD_USE_VENDORED_GGML=0`, both `LlamaCppBuilder.build()` and
`LlamaCppBuilder.build_shared()` now pass `-DGGML_MAX_NAME=128` via
`CMAKE_C_FLAGS` and `CMAKE_CXX_FLAGS`:

```python
# scripts/manage.py
_SD_GGML_MAX_NAME = 128

def _sd_uses_shared_ggml() -> bool:
    return os.environ.get("SD_USE_VENDORED_GGML") == "0"

# In LlamaCppBuilder.build() and build_shared():
if _sd_uses_shared_ggml():
    _def = f"-DGGML_MAX_NAME={_SD_GGML_MAX_NAME}"
    extra["CMAKE_C_FLAGS"] = _def
    extra["CMAKE_CXX_FLAGS"] = _def
```

The main `CMakeLists.txt` also propagates it to the scikit-build-core wheel
build (the Cython extensions):

```cmake
if(NOT SD_USE_VENDORED_GGML)
    add_definitions(-DGGML_MAX_NAME=128)
endif()
```

This ensures every component -- llama.cpp shared libs, SD extension, Cython
bindings -- agrees on `sizeof(ggml_tensor)`.

### 3. Dynamic link path (existing)

`CMakeLists.txt:843` links SD against llama.cpp's shared ggml dylibs when
`WITH_DYLIB=ON` and `SD_USE_VENDORED_GGML=OFF`. The Makefile dynamic
targets now set `SD_USE_VENDORED_GGML=0` by default.

---

## Validation results

Validated on RTX 4060 (sm_89), CUDA 12.0, `CMAKE_CUDA_ARCHITECTURES=native`:

| Check | Result |
|---|---|
| SD `.so` size | 23 MB (down from 210 MB) |
| `libggml-cuda.so` size | 137 MB (native sm_89 only) |
| ggml_cuda symbols in SD `.so` | 0 |
| `import cyllama.sd` | OK |
| SDXL Turbo image generation | 512x512 at 4.16 it/s, no assertion |
| FLUX z_image_turbo with `--offload-to-cpu --vae-on-cpu` | 512x1024, no assertion |
| Full test suite | 1432 passed, 35 skipped |
| Wheel size (compressed, native arch) | 44 MB |
| auditwheel repair | No duplicated libs |
| Repaired wheel in clean venv | Loads + generates images OK |

---

## Changes made

### Makefile

All GPU dynamic build and wheel targets now set `SD_USE_VENDORED_GGML=0`.
CUDA targets default `CMAKE_CUDA_ARCHITECTURES` to `native` (overridable
via env var).

### scripts/manage.py

- Module-level `_SD_GGML_MAX_NAME = 128` and `_sd_uses_shared_ggml()` helper.
- `LlamaCppBuilder.build()`: passes `-DGGML_MAX_NAME=128` via CMAKE_C/CXX_FLAGS
  when SD shares ggml.
- `LlamaCppBuilder.build_shared()`: same.
- `StableDiffusionCppBuilder._sync_ggml_abi()`: unchanged (source sync).

### CMakeLists.txt

- `add_definitions(-DGGML_MAX_NAME=128)` when `NOT SD_USE_VENDORED_GGML`,
  so Cython extensions also see the correct struct layout.

### .github/workflows/build-gpu-wheels.yml

All four auditwheel repair blocks (CUDA, ROCm, SYCL, Vulkan) now exclude
bundled project libs (`libllama`, `libggml-*`, `libmtmd`) and `libgomp` in
addition to GPU runtime system libs. This prevents auditwheel from
SONAME-rewriting the bundled libs, which was the root cause of the
double-free crash documented in `docs/dev/cuda-double-free.md`.

---

## Follow-up work (out of scope for the first landing)

- **Flip the CMake default** (Option 4b). After a release cycle of
  real-world coverage, change `SD_USE_VENDORED_GGML` default to `OFF`.
- **Extend unification to Metal / Vulkan / HIP / SYCL / OpenCL** wheels.
  Same mechanism applies; each backend needs its own validation on matching
  hardware.
- **Default `GGML_NATIVE=ON` for local static builds** and
  `CMAKE_CUDA_ARCHITECTURES=native` for local dynamic builds. See
  `docs/dev/ggml-config.md` for the analysis and recommendation.
- **Audit whisper** for the same pattern. Whisper also wraps ggml; if its
  static path whole-archives ggml backends, the same unification can apply.
- **Propose the sync mechanism upstream**. `_sync_ggml_abi()` is a
  cyllama-specific workaround. An upstream fix -- either
  stable-diffusion.cpp tracking ggml's HEAD more closely, or ggml itself
  committing to stable ABI guarantees -- would remove the need for the sync
  entirely.

---

## References

- `CHANGELOG.md` -- 0.2.9 entry documenting the workaround this plan
  reverses.
- `CMakeLists.txt:11` -- `SD_USE_VENDORED_GGML` option default.
- `CMakeLists.txt:15-19` -- `GGML_MAX_NAME=128` propagation.
- `CMakeLists.txt:849-869` -- dynamic-link branch.
- `CMakeLists.txt:891-894`, `CMakeLists.txt:913-917` -- the whole-archive
  link whose bloat we eliminated.
- `scripts/manage.py` -- `_SD_GGML_MAX_NAME`, `_sd_uses_shared_ggml()`,
  `_sync_ggml_abi()`, and the CMAKE_C/CXX_FLAGS injection in both
  `LlamaCppBuilder.build()` and `LlamaCppBuilder.build_shared()`.
- `build/stable-diffusion.cpp/CMakeLists.txt:233` --
  `-DGGML_MAX_NAME=128` that SD applies.
- `build/stable-diffusion.cpp/src/ggml_extend.hpp:94` --
  `static_assert(GGML_MAX_NAME >= 128)`.
- `build/llama.cpp/ggml/include/ggml.h` -- `#ifndef GGML_MAX_NAME` /
  `#define GGML_MAX_NAME 64` guard.
- `Makefile` -- `build-*-dynamic` and `wheel-*-dynamic` targets.
- `.github/workflows/build-gpu-wheels.yml` -- auditwheel exclude sets.
- `docs/dev/cuda-double-free.md` -- auditwheel `--exclude` rationale.
- `docs/dev/ggml-config.md` -- `GGML_NATIVE` and `CMAKE_CUDA_ARCHITECTURES`
  analysis.
