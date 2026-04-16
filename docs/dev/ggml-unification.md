# ggml unification: eliminating the 210 MB SD bloat in CUDA wheels

## Summary

The `cyllama_cuda12` wheel currently ships at **157 MB compressed / ~427 MB uncompressed**, exceeding PyPI's 100 MB default limit and the 150 MB cap typically granted on request. Two files dominate: `cyllama/sd/stable_diffusion.cpython-*.so` (~210 MB) and `cyllama/llama/libggml-cuda.so` (~202 MB). Both carry the same set of ggml CUDA kernels -- the SD extension has `libggml-cuda.a` whole-archive linked into it, while llama.cpp ships the same backend as a separate shared library.

The intuitive fix -- "patch stable-diffusion.cpp to work with llama.cpp's ggml so they can share one `libggml-cuda.so`" -- is **not needed**. The required infrastructure already exists in this repo:

- `scripts/manage.py:1623` defines `_sync_ggml_abi()`, which replaces stable-diffusion.cpp's vendored `ggml/` tree with llama.cpp's copy at build time. This makes SD compile against the exact same headers the shared dylibs use, eliminating the enum/ABI drift that originally motivated the 0.2.9 workaround.
- `CMakeLists.txt:843` defines a dynamic-link branch (`WITH_DYLIB AND NOT SD_USE_VENDORED_GGML`) that links `libstable-diffusion.a` statically but references llama.cpp's shared ggml dylibs at runtime. No whole-archive of `libggml-cuda.a`.
- `Makefile:284` defines `wheel-cuda-dynamic`, which sets `GGML_CUDA=1 WITH_DYLIB=1`.

The missing piece is a single environment toggle: `SD_USE_VENDORED_GGML=0`. With it set, the sync runs, the dynamic CMake branch is taken, and the SD `.so` drops to roughly the size of the SD logic alone (estimated 15-25 MB). The resulting CUDA wheel should land near ~80 MB compressed, well under any PyPI limit.

This document lays out:
1. Why 0.2.9 made `SD_USE_VENDORED_GGML=ON` the default.
2. Why the underlying correctness problem is already solved by `_sync_ggml_abi()`.
3. A concrete plan to switch CUDA release wheels to the shared-ggml path.
4. A validation protocol that exercises the historical crash scenario.
5. A fallback procedure if validation reveals a residual incompatibility that does require a source patch to stable-diffusion.cpp.

---

## Problem statement

### Wheel size

From `dist/cyllama_cuda12-0.2.9-cp313-cp313-manylinux2014_x86_64.*.whl`:

| File | Compressed (approx) | Uncompressed | Notes |
|---|---:|---:|---|
| `cyllama/sd/stable_diffusion.cpython-313-x86_64-linux-gnu.so` | ~75 MB | 210 MB | Contains whole-archive-linked `libggml-cuda.a` |
| `cyllama/llama/libggml-cuda.so` | ~75 MB | 202 MB | Standalone ggml CUDA backend for llama bindings |
| `cyllama/llama/libllama.so.0` | ~1 MB | 3.2 MB | |
| `cyllama/llama/llama_cpp.cpython-*.so` | ~0.6 MB | 1.4 MB | Cython bindings |
| everything else | ~5 MB | ~10 MB | Python sources, small libs, libs duplicated by auditwheel |
| **total** | **~157 MB** | **~427 MB** | |

The two CUDA-carrying `.so` files hold effectively the same compiled kernel set -- the ggml CUDA backend, compiled for whichever SM architectures the build targeted, for all dtype/template permutations. Confirmed by inspecting the SD `.so` with `nm -D`:

```
_Z25launch_mul_mat_vec_f_cudaIffLi3ELb0EEvPKT_...
_Z21ggml_cuda_op_topk_moeR25ggml_backend_cuda_context...
_Z19ggml_cuda_mul_mat_fR25ggml_backend_cuda_context...
```

and strings `cublas`, `cudart`, `compute_16f`, `sm_75`. The SD extension contains the entire ggml CUDA backend as static code, in addition to its own model/sampler/VAE implementation.

### PyPI constraint

PyPI's default per-file limit is 100 MB. Projects can request a per-project increase; 150 MB is commonly granted, higher values are increasingly hard to justify. At 157 MB the wheel cannot be uploaded without a size-increase grant and is at the edge of even the 150 MB threshold.

### Non-goal: splitting the distribution

A natural workaround is to publish `cyllama-cuda12` and `cyllama-sd-cuda12` as separate distributions. This is explicitly **not** the approach we want: SD is a core feature of cyllama, not an optional add-on, and splitting complicates install and version skew for users. This document pursues the single-wheel path.

---

## Why the current shape exists: the 0.2.9 workaround

Release 0.2.9 (see `CHANGELOG.md`, top-of-file entry) made `SD_USE_VENDORED_GGML=ON` the default build-time behavior for the SD extension. The changelog rationale:

> **stable-diffusion.cpp now uses its own vendored ggml by default** -- The SD extension statically links stable-diffusion.cpp's own vendored ggml instead of sharing llama.cpp's ggml dylibs. Fixes a `ggml_backend_tensor_copy` assertion crash ("cannot copy tensors with different layouts") during CUDA image generation caused by subtle ggml version incompatibilities between llama.cpp and stable-diffusion.cpp.

### Historical root cause (pre-0.2.9)

At the time of the 0.2.9 fix, SD's vendored ggml was at an older release while llama.cpp's had moved forward. The upstream project inserted new operations into the `ggml_op` enum between those versions. Because the enum is defined in a public header and the C/C++ compilers encode enum values as integer ordinals at compile time:

1. SD was compiled against its older ggml headers, assigning low integer values to ops like `GGML_OP_MUL_MAT`, `GGML_OP_ADD`, etc.
2. At runtime, SD's compiled code called into llama.cpp's newer `libggml-cuda.so`, which had the new ops inserted earlier in the enum, shifting every subsequent op's integer value by N.
3. SD's compute graphs were constructed with integer op ids that the newer runtime now interpreted as *different* operations. The resulting tensor graphs had mismatched output shapes, and the first `ggml_backend_tensor_copy` between a CPU-allocated source and a CUDA-allocated destination tripped `GGML_ASSERT(ggml_are_same_layout(src, dst))` in `ggml-backend.cpp`.

0.2.9 short-circuited the problem by ensuring SD compiled and linked against the same ggml it was designed for -- its own vendored copy, statically, with `-Wl,--whole-archive` on the ggml `.a` files so no symbol resolution ever crosses into llama.cpp's ggml dylibs.

### Correct but expensive

The workaround is correct. It also bundles the entire ggml CUDA backend inside the SD `.so` (`CMakeLists.txt:885-888` and the `-Wl,--whole-archive` block at `CMakeLists.txt:907-911`), which is where the 210 MB comes from.

Quoting the CMakeLists directly:

```cmake
# CMakeLists.txt:885-888
if(GGML_CUDA)
    static_lib(SD_LIB_GGML_CUDA "${SD_GGML_LIB_DIR}" "ggml-cuda")
    list(APPEND SD_GGML_LIBS ${SD_LIB_GGML_CUDA})
endif()
```

```cmake
# CMakeLists.txt:907-911  (UNIX AND NOT APPLE branch)
target_link_libraries(stable_diffusion PRIVATE
    -Wl,--whole-archive ${SD_GGML_LIBS} -Wl,--no-whole-archive
    "${LIB_SD}"
    ${SYSTEM_LIBS}
)
```

`-Wl,--whole-archive` is used because ggml-cuda exposes kernel launchers through function-pointer tables that the linker would otherwise dead-strip. The consequence is that *every* symbol in `libggml-cuda.a` -- including the tens of megabytes of CUDA kernel object code for each SM architecture and each template instantiation -- ends up inside the SD extension.

---

## Why "patch stable-diffusion.cpp" is not the right framing

The intuition behind the patch framing is sound: if SD's source is compatible with llama.cpp's newer ggml, SD can build against llama's headers and link against llama's shared `libggml-cuda.so` at runtime. One copy of the CUDA backend, ~200 MB saved.

However, two observations change what the "fix" actually looks like:

### Observation 1: the ABI sync is already implemented

`scripts/manage.py:1623-1650`:

```python
def _sync_ggml_abi(self) -> None:
    """Sync ggml ABI between stable-diffusion.cpp and llama.cpp.

    stable-diffusion.cpp vendors its own ggml (potentially older), but the
    final extension links against llama.cpp's ggml dylibs.  If enum values
    (ggml_op, ggml_type) diverge between versions, the SD code will build
    compute graphs with wrong op ids, causing assertion failures at runtime.

    We replace SD's vendored ggml directory with llama.cpp's ggml so that
    headers, source, and the runtime dylibs all use the same version.
    """
    import shutil
    llama_ggml = self.project.src / "llama.cpp" / "ggml"
    sd_ggml = self.src_dir / "ggml"
    ...
    shutil.rmtree(sd_ggml)
    shutil.copytree(llama_ggml, sd_ggml)
```

And the trigger, at `scripts/manage.py:1663`:

```python
if os.environ.get("SD_USE_VENDORED_GGML") == "0":
    self._sync_ggml_abi()
```

This is the compile-time half of the fix. When invoked, SD is rebuilt against llama.cpp's current ggml source tree, so the `ggml_op` enum values its object files contain match what `libggml-cuda.so` will serve at runtime. The runtime divergence that caused the 0.2.9 crash cannot occur.

### Observation 2: the dynamic CMake branch is already present

`CMakeLists.txt:843-863`:

```cmake
if(WITH_DYLIB AND NOT SD_USE_VENDORED_GGML)
    # Dynamic: link libstable-diffusion.a statically, reuse llama.cpp's ggml dylibs
    target_link_directories(stable_diffusion PRIVATE "${SDCPP_LIB}" ${COMMON_LINK_DIRS})
    target_link_libraries(stable_diffusion PRIVATE
        "${LIB_SD}"
        ${DYLIBS}
        ${SYSTEM_LIBS}
    )
    ...
    set_target_properties(stable_diffusion PROPERTIES
        BUILD_RPATH "${LLAMACPP_DYLIB_DIR};${SDCPP_LIB}"
        INSTALL_RPATH "$ORIGIN/../llama"
    )
```

`${DYLIBS}` resolves to the llama.cpp shared libraries already bundled under `cyllama/llama/`, which on a CUDA build include `libggml-cuda.so`. The `$ORIGIN/../llama` RPATH ensures the SD extension finds those libraries at import time.

### Observation 3: current ggml divergence is additive

Inspection of the two trees now (April 2026) shows:

- `build/stable-diffusion.cpp/ggml/include/ggml.h` -- ggml 0.9.8
- `build/llama.cpp/ggml/include/ggml.h` -- ggml 0.9.11

The public `ggml_tensor` struct is byte-identical between these versions. The `ggml_are_same_layout` helper (`ggml-impl.h`) is likewise identical (checks `type`, `ne[]`, `nb[]`). The difference is purely additive: llama's 0.9.11 has more ops in `ggml_op` and additional helper functions. SD's compiled source doesn't invoke any of those new ops, so the only failure mode -- the enum-ordinal shift -- is *exactly* what `_sync_ggml_abi()` neutralizes by rebuilding SD against the newer enum.

There is therefore no SD source-level change needed for the unification to work. Compile SD against llama's ggml headers, and its emitted enum values match the runtime. The required "patch" is a build-system configuration flip, not an edit to `stable_diffusion.cpp` files.

---

## The plan

### Step 1: produce a dynamic CUDA wheel and validate it

The existing `wheel-cuda-dynamic` Makefile target builds a CUDA wheel with `WITH_DYLIB=1`, but does not currently set `SD_USE_VENDORED_GGML=0`. That combination takes the `else()` branch at `CMakeLists.txt:864` (still static-link SD, just with ggml picked from llama.cpp's build tree), which does *not* avoid the whole-archive of `libggml-cuda.a`. To hit the dynamic branch at `CMakeLists.txt:843` we need both flags.

On a Linux + CUDA machine:

```bash
make reset                 # ensure clean tree
make download              # test model
GGML_CUDA=1 WITH_DYLIB=1 SD_USE_VENDORED_GGML=0 python scripts/manage.py build --all --dynamic
GGML_CUDA=1 WITH_DYLIB=1 SD_USE_VENDORED_GGML=0 uv build --wheel
```

Expected wheel layout:

- `cyllama/sd/stable_diffusion.cpython-*.so` -- roughly 15-25 MB (SD model/sampler/VAE code, no ggml CUDA kernels).
- `cyllama/llama/libggml-cuda.so` -- still ~200 MB (unchanged; this is the one authoritative copy).
- Total wheel: ~80 MB compressed.

The two preceding commands are indicative -- the actual invocation should match however `wheel-cuda-dynamic` is wired today, extended with the `SD_USE_VENDORED_GGML=0` variable. The simplest concrete form:

```bash
make reset
GGML_CUDA=1 WITH_DYLIB=1 SD_USE_VENDORED_GGML=0 make wheel-cuda-dynamic
```

If that target does not already export `SD_USE_VENDORED_GGML`, either export it in the shell or amend the Makefile recipe (see Step 4 below).

### Step 2: run the regression-critical CUDA test

The failure that motivated 0.2.9 was a runtime assertion inside `ggml_backend_tensor_copy` during `generate_image()`. That exact code path needs to pass before this wheel ships. A minimal reproducer:

```bash
# Install the dynamic wheel into a clean venv
uv venv .venv-dyn && source .venv-dyn/bin/activate
uv pip install dist/cyllama_cuda12-*.whl

# Run SD CUDA image generation
python -c "
from cyllama.sd import SDContext, SDContextParams, SDImageGenParams
params = SDContextParams(model_path='models/<some-sd-gguf>')
ctx = SDContext(params)
gen = SDImageGenParams(prompt='a cat on a rocket')
img = ctx.generate_image(gen)
img.save('out.png')
del ctx
import gc; gc.collect()
"
```

If this prints no assertion and produces `out.png`, the ABI-sync fix is confirmed working against the real workload. If it crashes with `ggml_are_same_layout` or any other `ggml_backend_tensor_copy` assertion, we fall back to Step 5.

Run the full `make test` suite in addition:

```bash
make test
```

All 1150+ tests must pass. Pay particular attention to `tests/test_sd.py` (which already applies the `del ctx; gc.collect()` discipline required for Metal-era SD tests -- see `docs/dev/test-cleanup.md`).

### Step 3: verify the wheel is installable via auditwheel

After the raw `uv build --wheel` step, run the manylinux repair pass used by CI. Keep the `--exclude` set from the double-free fix documented in `docs/dev/cuda-double-free.md`, and extend it to also exclude the already-bundled llama.cpp libraries (the auditwheel dedup we identified separately):

```bash
auditwheel repair dist/cyllama_cuda12-*.whl \
    --exclude libcuda.so.1 \
    --exclude libcudart.so.12 \
    --exclude libcublas.so.12 \
    --exclude libcublasLt.so.12 \
    --exclude libgomp.so.1 \
    --exclude 'libllama*' \
    --exclude 'libggml*' \
    --exclude 'libmtmd*'
```

The last three excludes cover the previously-duplicated `cyllama_cuda12.libs/libllama-<hash>.so.*`, `libggml-<hash>.so.*`, and `libggml-base-<hash>.so.*`, saving another ~5 MB and eliminating the double-bundling noise. This is independent of the SD unification work and can ship on its own.

Re-confirm the repaired wheel still loads and passes Step 2.

### Step 4: flip the default for CUDA release wheels

Once Step 2 is confirmed on real CUDA hardware, change the Makefile release target so `SD_USE_VENDORED_GGML=0` is the default for CUDA wheels. Two reasonable options:

**Option 4a -- flip only the dynamic targets** (lower-blast-radius):

`Makefile:284-285`:

```make
wheel-cuda-dynamic:
    @GGML_CUDA=1 WITH_DYLIB=1 SD_USE_VENDORED_GGML=0 uv build --wheel
```

Do the same for `wheel-metal-dynamic`, `wheel-vulkan-dynamic`, etc. once those are separately validated. This leaves the static `wheel-cuda` target untouched, preserving the 0.2.9 behavior for anyone who specifically wants a statically-linked build.

**Option 4b -- flip the CMake default** (simpler user model):

`CMakeLists.txt:11`:

```cmake
option(SD_USE_VENDORED_GGML "Link stable-diffusion against its own vendored ggml" OFF)
```

And ensure the `wheel-cuda` target runs with `WITH_DYLIB=1` as well, so the SD extension always takes the dynamic path. This is the cleaner end state but is a larger policy change -- any downstream consumer of `SD_USE_VENDORED_GGML=ON` (e.g. for isolation reasons) would have to opt in explicitly.

Recommendation: start with 4a. Take 4b in a later release once a few weeks of GPU-machine coverage have validated the dynamic default.

### Step 5: CHANGELOG and release

Draft the 0.2.10 (or next-version) changelog entry under `## [Unreleased]`:

```markdown
### Changed

- **CUDA wheel shrunk from 157 MB to ~80 MB compressed** -- The stable-diffusion
  extension no longer statically embeds `libggml-cuda.a`. It now links against
  llama.cpp's shared `libggml-cuda.so`, the same copy already bundled for the
  llama/whisper bindings. The ggml ABI is synchronized at build time via
  `scripts/manage.py:_sync_ggml_abi()`, which overlays llama.cpp's ggml source
  onto stable-diffusion.cpp before compilation, eliminating the enum-ordinal
  drift that caused the 0.2.9 `ggml_are_same_layout` assertion under CUDA. The
  0.2.9 workaround (`SD_USE_VENDORED_GGML=ON` default) is reversed for the
  dynamic CUDA target. The old static-link behavior remains available via
  `SD_USE_VENDORED_GGML=1`.
```

---

## Validation protocol

Before promoting any wheel produced by this plan to PyPI:

1. **Build host**: Linux x86_64 with CUDA 12.x, a non-trivial NVIDIA GPU (minimum Turing / sm_75), glibc matching the manylinux tag we target.
2. **Clean build**: `make reset && GGML_CUDA=1 WITH_DYLIB=1 SD_USE_VENDORED_GGML=0 make wheel-cuda-dynamic`.
3. **Size check**:
    - Compressed wheel size: must be under 100 MB to allow default PyPI upload; target is ~80 MB.
    - `cyllama/sd/stable_diffusion.cpython-*.so`: must be under 30 MB uncompressed. If it is still >100 MB, the whole-archive CUDA link was not avoided -- confirm `_sync_ggml_abi` ran and the `WITH_DYLIB AND NOT SD_USE_VENDORED_GGML` branch was taken (inspect `CMakeCache.txt`).
4. **Symbol check**: `nm -D cyllama/sd/stable_diffusion.cpython-*.so | grep -c ggml_cuda` should be 0 (or, at most, reference undefined `U` entries that will resolve at load time). Presence of *defined* `T`/`W` ggml_cuda symbols in the SD `.so` means the static embedding is still occurring.
5. **Load check**: `python -c "import cyllama.sd"` with the dynamic wheel installed. Must not raise `ImportError` or `OSError` about missing `libggml-cuda.so`.
6. **Regression check**: the CUDA `generate_image()` smoke test from Step 2. Must produce a valid image with no assertion.
7. **Full suite**: `make test`. Must pass.
8. **auditwheel repair**: with the exclude set in Step 3. Repaired wheel must re-pass checks 4-7.

Only when every step is green should the wheel be uploaded.

---

## Fallback: what if validation fails

If Step 2 crashes -- either with the original `ggml_are_same_layout` assertion or with a different failure -- the hypothesis that the sync alone is sufficient is wrong, and we need to investigate what in SD's code is incompatible with llama's newer ggml.

### Diagnostic order

1. **Confirm the sync ran**. `build/stable-diffusion.cpp/ggml/include/ggml.h` should have the same `ggml_op` enum as `build/llama.cpp/ggml/include/ggml.h`. If not, `SD_USE_VENDORED_GGML=0` was not picked up by the build driver -- fix that first and re-try.
2. **Check the symbol resolution**. `ldd cyllama/sd/stable_diffusion.cpython-*.so` must show `libggml-cuda.so => .../cyllama/llama/libggml-cuda.so`. If it resolves to a different path (e.g. a system copy, or an SD-local copy), the RPATH is wrong and the wheel is linking against the wrong runtime.
3. **Run under `GGML_LOG_LEVEL=DEBUG`** (or whatever the current ggml env-var is) to capture which op the crash occurs on.
4. **Compare the failing op's struct usage**. If the op exists in both 0.9.8 and 0.9.11 but its `src[]` / `op_params[]` packing changed, SD's code path for that op needs updating to match the 0.9.11 convention.

### Where an SD source patch would live

If a genuine source patch to stable-diffusion.cpp turns out to be needed, it would go in `build/stable-diffusion.cpp/` (the upstream clone) and be carried as a patch file in `patches/` (convention: `patches/sd-ggml-0.9.11-compat.patch`), applied by `scripts/manage.py` at the same stage as `_sync_ggml_abi`. Candidate files most likely to need edits:

- `build/stable-diffusion.cpp/model.cpp`
- `build/stable-diffusion.cpp/ggml_extend.hpp`
- `build/stable-diffusion.cpp/stable-diffusion.cpp`

These are the files that construct compute graphs and invoke ggml op builders directly. A patch would most likely replace a deprecated builder call or initialize a newly-added field on an op's parameter struct.

### Scope estimate

Based on the Explore-agent review of the two trees (SD 0.9.8 vs llama 0.9.11), the public tensor struct is unchanged and no APIs SD uses have been removed or renamed between those versions. A residual incompatibility, if any, is likely a handful of lines. The patch is not expected to be invasive.

If upstream stable-diffusion.cpp has moved to newer ggml in their own master by the time we investigate, the cleanest form of the "patch" is a version bump of the pinned SD commit in `scripts/manage.py`, not a carried diff.

---

## Follow-up work (out of scope for the first landing)

Items that are adjacent but not strictly required for the size win:

- **Flip the CMake default** (Option 4b). After a release cycle of real-world coverage.
- **Extend unification to Metal / Vulkan / HIP / SYCL / OpenCL** wheels. Same mechanism applies; each backend needs its own validation on matching hardware.
- **Narrow `CMAKE_CUDA_ARCHITECTURES`**. Currently unset; the build inherits ggml-cuda's default arch list. Setting e.g. `75;80;86;89;90` for release wheels can cut `libggml-cuda.so` by another 30-50% at the cost of narrower GPU coverage. This is independent of SD unification and can land separately.
- **Audit whisper** for the same pattern. Whisper also wraps ggml; if its static path whole-archives ggml backends, the same unification can apply.
- **Propose the sync mechanism upstream**. `_sync_ggml_abi()` is a cyllama-specific workaround for a problem every aggregator of llama.cpp + stable-diffusion.cpp faces. An upstream fix -- either stable-diffusion.cpp tracking ggml's HEAD more closely, or ggml itself committing to stable op-id numbering -- would remove the need for the sync entirely.

---

## Expected outcome

| Metric | Before (0.2.9) | After (this plan) |
|---|---:|---:|
| CUDA wheel compressed | 157 MB | ~80 MB |
| CUDA wheel uncompressed | ~427 MB | ~250 MB |
| SD `.so` size | 210 MB | 15-25 MB |
| `libggml-cuda.so` size | 202 MB | 202 MB (unchanged, now the single copy) |
| Ships on PyPI under default 100 MB limit | no | yes |
| Changes to stable-diffusion.cpp source | none | none |
| Risk of re-introducing 0.2.9 crash | n/a | mitigated by `_sync_ggml_abi` + validation protocol |

The win is large, the code changes are small, and the original crash mode is addressed at its root rather than worked around.

---

## References

- `CHANGELOG.md` -- 0.2.9 entry documenting the workaround this plan reverses.
- `CMakeLists.txt:11` -- `SD_USE_VENDORED_GGML` option default.
- `CMakeLists.txt:843-863` -- dynamic-link branch that this plan activates.
- `CMakeLists.txt:885-888`, `CMakeLists.txt:907-911` -- the whole-archive link whose bloat we are eliminating.
- `scripts/manage.py:1623-1650` -- `_sync_ggml_abi` implementation.
- `scripts/manage.py:1663-1664` -- sync trigger condition.
- `Makefile:284-285` -- `wheel-cuda-dynamic` target.
- `docs/dev/cuda-double-free.md` -- auditwheel `--exclude` set we must continue to use.
- `docs/dev/static-vs-dynamic.md` -- prior analysis of static vs dynamic linking tradeoffs.
- `docs/dev/packaging-options.md` -- broader survey of packaging strategies, including the shared-library question this plan answers concretely.
- `docs/dev/test-cleanup.md` -- `SDContext` lifecycle requirements that the validation protocol must respect.
