# CUDA double free or corruption (!prev)

## Summary

Dynamic-linked CUDA wheels (`WITH_DYLIB=1`) crash with `double free or corruption (!prev)` during Python interpreter shutdown. The crash is non-deterministic and only affects wheels processed by `auditwheel repair`.

First observed against commit `36db368`. The initial fix attempted Python-level cleanup before shutdown, but this caused instability on the Metal backend as well. Reverted in `fb522c8`.

## Root cause

The issue is caused by `auditwheel repair`, specifically its use of `patchelf` to rewrite ELF SONAME headers on bundled shared libraries.

During repair, auditwheel:

1. Copies bundled `.so` files (`libllama.so`, `libggml-cuda.so`, etc.) into `cyllama.libs/`
2. Renames them with hash suffixes (e.g., `libllama-91896a1c.so.0.0.1`)
3. Rewrites their ELF SONAME headers via `patchelf` so the dynamic linker resolves the renamed copies

The SONAME rewrite changes the dependency graph that glibc's dynamic linker uses to determine `dlclose` unload ordering. CUDA's runtime (`libcudart`) registers internal `atexit` handlers during initialization. When Python shuts down, the altered unload order can cause CUDA's handlers to fire after the memory they reference (in `libggml-cuda.so`) has already been unmapped.

### Why it only affects CUDA

Vulkan wheels go through the same `auditwheel repair` process on their bundled llama/ggml libs and do not crash. The difference is that CUDA's runtime manages teardown via `atexit` handlers registered internally by `libcudart`. Vulkan cleanup is explicit (the application calls `vkDestroy*` functions), so unload ordering is irrelevant. ROCm and SYCL have not been tested but may exhibit similar issues if their runtimes use atexit-based teardown.

### Why it is non-deterministic

Several factors vary between runs:

- **ASLR** randomizes library mapping addresses, which affects whether a double-free hits unmapped memory (segfault) vs. still-valid memory (silent or no error)
- **`dlclose` ordering** depends on the full set of loaded shared objects, which varies with installed packages (numpy, etc.) and load timing
- **CUDA runtime state** depends on what GPU operations actually ran. Import-only tests may exit cleanly because CUDA never fully initialized its teardown hooks
- **Python GC timing** determines whether Cython destructors run before or during interpreter shutdown. If contexts are freed before shutdown starts, the `dlclose` race is avoided

### Reproduction matrix

| Build method | Clean exit? |
|---|---|
| `uv build --wheel` static (no `WITH_DYLIB`) | Yes |
| `uv build --wheel` `WITH_DYLIB=1`, no auditwheel | Yes |
| `uv build --wheel` `WITH_DYLIB=1` + `auditwheel repair` (current CI) | No, double free |
| `uv build --wheel` `WITH_DYLIB=1` + `auditwheel repair --exclude` (bundled libs) | Yes |

## Solutions

Seven approaches are described below, ordered from most recommended to least.

### 1. `auditwheel addtag` instead of `auditwheel repair`

Replace `auditwheel repair` entirely with `auditwheel addtag`, which stamps the `manylinux` platform tag on the wheel without copying, renaming, or patching any libraries.

```yaml
CIBW_REPAIR_WHEEL_COMMAND_LINUX: >
  auditwheel addtag -w {dest_dir} {wheel}
```

The build already places libraries under `cyllama/llama/` with correct RPATHs, so auditwheel's relocation is unnecessary.

**Pros:**
- Eliminates the entire class of SONAME-rewriting bugs
- No exclude list to maintain
- One-line change

**Cons:**
- `addtag` refuses if the wheel doesn't already meet the target manylinux policy; verify with `auditwheel show` first
- If any non-bundled system libs genuinely need vendoring (unlikely for this project), they won't be

**Status:** Requires auditwheel >= 6.0. Stock manylinux2014 and manylinux_2_28 images ship older versions that do not include `addtag` (available commands: `show`, `repair`, `lddtree`). To use this approach, upgrade auditwheel in the container first (e.g. `pip install 'auditwheel>=6.0'` in `CIBW_BEFORE_BUILD`). Alternatively, skip repair entirely with `CIBW_REPAIR_WHEEL_COMMAND_LINUX: ""` -- the wheel keeps its `linux_x86_64` tag but avoids all SONAME rewriting. The test workflow at `.github/workflows/test-cuda-wheel.yml` validates both the skip-repair and `--exclude` strategies.

### 2. `--exclude` bundled libraries from auditwheel repair

Add `--exclude` flags for every bundled `.so` so auditwheel leaves them untouched (no relocation, no SONAME rewrite).

```yaml
CIBW_REPAIR_WHEEL_COMMAND_LINUX: >
  auditwheel repair -w {dest_dir} {wheel}
  --plat manylinux_2_35_x86_64
  --exclude libcuda.so.1
  --exclude libcudart.so.12
  --exclude libcublas.so.12
  --exclude libcublasLt.so.12
  --exclude libllama.so.0
  --exclude libggml.so.0
  --exclude libggml-base.so.0
  --exclude libggml-cuda.so
  --exclude libggml-cpu.so
  --exclude libmtmd.so.0
  --exclude libgomp.so.1
```

**Pros:**
- Directly addresses the root cause
- Validated in the reproduction matrix
- Zero runtime cost

**Cons:**
- Maintenance burden: every new upstream `.so` (new ggml backend, new library) requires a new `--exclude` line. Missing one reintroduces the crash
- Weaker manylinux compliance: trusts the build's RPATHs rather than auditwheel's relocation guarantees
- Theoretical risk of SONAME collisions if another installed package bundles a different version of the same library (unlikely in practice)

**Status:** Validated by colleague's report. Test workflow at `.github/workflows/test-cuda-wheel.yml`.

### 3. Static linking

Build with `WITH_DYLIB=0` so llama.cpp/ggml are statically linked into the Cython extension `.so`. No shared libraries to relocate, no `dlclose` ordering, no SONAME rewriting.

```yaml
CIBW_ENVIRONMENT_LINUX: >
  WITH_DYLIB=0
```

**Pros:**
- Simplest possible fix -- no auditwheel complexity at all
- Single `.so` extension file with no dependency graph
- Already validated: static builds exit cleanly in the reproduction matrix

**Cons:**
- Larger wheel size: each Cython extension (llama, whisper, sd) embeds its own copy of ggml
- Cannot share loaded libraries across extensions at runtime
- May break if CUDA expects to `dlopen` ggml backends dynamically at runtime
- The project already offers both link modes; this would mean abandoning dynamic linking for CUDA

**Status:** Known to work. Already the default for non-GPU wheels.

### 4. Explicit Python-level cleanup before shutdown

Register a Python `atexit` handler that tears down native state before the interpreter starts unloading modules.

```python
# In cyllama/__init__.py or the CUDA backend init path
import atexit
import gc

def _cuda_cleanup():
    gc.collect()
    # If exposed: llama_backend_free()

atexit.register(_cuda_cleanup)
```

**Pros:**
- Doesn't touch the build system at all
- Works with unmodified auditwheel

**Cons:**
- Fragile. Python's `atexit` execution order is not guaranteed relative to extension module `__del__` methods or module `__del__` cleanup
- Races against the same non-determinism that causes the bug -- may reduce crash frequency without eliminating it
- Requires that all native contexts are freed during the `gc.collect()` call, which depends on no circular references holding them alive
- Does not fix the underlying `dlclose` ordering problem

**Status:** Untested. Not recommended as a primary fix due to inherent fragility.

### 5. `RTLD_NODELETE` on the CUDA-linked library

Mark `libggml-cuda.so` so the dynamic linker never unloads it, ensuring CUDA's atexit handlers always find valid memory.

From C (in a Cython init path):
```c
dlopen("libggml-cuda.so", RTLD_NOW | RTLD_NODELETE);
```

Or from Python:
```python
import ctypes
ctypes.CDLL("libggml-cuda.so", mode=ctypes.RTLD_GLOBAL | 0x1000)  # RTLD_NODELETE
```

**Pros:**
- Surgically prevents the specific unload-ordering bug
- The library stays mapped until process exit, so CUDA's atexit handlers always find valid memory
- No build system changes required

**Cons:**
- Library memory is never reclaimed (minor, since the process is exiting anyway)
- Requires knowing the exact `.so` path, which differs after auditwheel renames it
- Platform-specific: `RTLD_NODELETE` is a Linux/glibc feature
- Obscure: future maintainers won't immediately understand why a manual `dlopen` with unusual flags exists

**Status:** Untested. Viable as a targeted fix if build-system changes are undesirable.

### 6. Patch auditwheel to skip SONAME rewriting

Fork or configure auditwheel to perform file relocation without rewriting SONAME headers. The trigger is specifically the SONAME change, not the file copy.

**Pros:**
- Preserves auditwheel's portability benefits (library vendoring, path normalization) while avoiding the specific trigger

**Cons:**
- auditwheel does not currently expose this granularity
- Requires maintaining a fork or contributing upstream
- High effort relative to other solutions
- The SONAME rewrite and the relocation are tightly coupled in auditwheel's internals

**Status:** Not investigated. Unlikely to be worth the effort given simpler alternatives.

### 7. Do nothing, document the issue

Ship the wheel as-is and document the crash as a known issue.

**Pros:**
- Zero effort

**Cons:**
- The crash manifests as a segfault or glibc abort on interpreter exit with no actionable error message
- Users cannot reasonably diagnose or work around it
- Undermines confidence in the CUDA wheel

**Status:** Not recommended.

## Recommendation

Try **solution 1** (`auditwheel addtag`) first. If the wheel already meets the target manylinux policy (verify with `auditwheel show`), it is a one-line fix with no maintenance burden. If `addtag` doesn't work (policy violation), fall back to **solution 2** (`--exclude`), which is validated and whose maintenance cost is low.

If neither build-system approach is viable, **solution 5** (`RTLD_NODELETE`) is the most targeted runtime fix.

## Test workflow

`.github/workflows/test-cuda-wheel.yml` is a standalone CI workflow for validating the fix. It builds a single CUDA wheel (cp312 only) with the `--exclude` fix applied and runs three shutdown-specific tests:

1. Basic import and exit
2. Import all extensions and exit
3. Repeated import/gc cycles

Trigger it manually from the Actions tab. Note that the smoke tests run on a CPU-only runner (no GPU available), so they validate clean `dlclose` ordering but cannot exercise the full CUDA atexit path. A passing test gives moderate confidence; a failing test is definitive.

## References

- Reproduction report: colleague's analysis of commit `2755d96`
- Revert: `fb522c8` (reverted `36db368`)
- auditwheel SONAME rewriting: [pypa/auditwheel#289](https://github.com/pypa/auditwheel/issues/289)
- patchelf SONAME behavior: [NixOS/patchelf#275](https://github.com/NixOS/patchelf/issues/275)
