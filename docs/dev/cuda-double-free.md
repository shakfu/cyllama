# CUDA double free or corruption (!prev)

## Summary

Dynamic-linked CUDA wheels (`WITH_DYLIB=1`) crash with `double free or corruption (!prev)` during Python interpreter shutdown. The crash is non-deterministic and only affects wheels processed by `auditwheel repair`.

This issue is **Linux-specific**. The entire chain — `auditwheel repair`, `patchelf` SONAME rewriting, glibc `dlclose` unload ordering — only exists on Linux. macOS and Windows are not affected:

- **macOS** uses `delocate` for wheel repair, which rewrites Mach-O load commands via `install_name_tool` rather than ELF SONAME headers. `delocate` does not alter the dyld unload order in the way that triggers this crash. macOS wheels have their own issues (e.g. duplicate OpenMP runtimes causing segfaults when co-installed with PyTorch — see [LightGBM#6595](https://github.com/lightgbm-org/LightGBM/issues/6595)), but those are a different problem with different root causes.
- **Windows** uses no wheel repair tool. DLLs are bundled as-is with no SONAME equivalent to rewrite, and the Windows loader does not have the same unload-ordering sensitivity as glibc's `dlclose`.

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

### Multiple CUDA versions installed simultaneously

When a system has multiple CUDA toolkit versions installed (e.g. CUDA 12 and CUDA 13), the dynamic linker may resolve `libcudart.so` to a different major version than the one the wheel was built against. This causes the same double-free symptom through a different mechanism:

1. The wheel's bundled `libggml-cuda.so` is linked against CUDA 12 (built with `cuda-12.4`)
2. At runtime, the dynamic linker resolves `libcudart.so` to the CUDA 13 version from the system, depending on `LD_LIBRARY_PATH` ordering and ldconfig priority
3. The mismatched runtime initializes its own internal atexit handlers with different memory layout assumptions
4. On shutdown, the CUDA 13 teardown attempts to free structures allocated by CUDA 12 conventions (or vice versa), triggering the double-free

This is a well-documented pattern in the CUDA ecosystem. PyTorch, CuPy, and other projects that ship CUDA wheels use `--exclude` to avoid bundling CUDA runtime libraries for exactly this reason — letting the system provide a single consistent CUDA installation. However, `--exclude` alone does not prevent the issue when multiple system CUDA versions coexist and the linker picks the wrong one.

#### Diagnosing version mismatch

```bash
# Check which libcudart is actually loaded at runtime
LD_DEBUG=libs python -c "from cyllama.llama import llama_cpp" 2>&1 | grep cudart

# List all CUDA runtime versions available on the system
ldconfig -p | grep cudart

# Check the version the extension was linked against
ldd $(python -c "import cyllama.llama.llama_cpp as m; print(m.__file__)") | grep cudart
```

If the loaded version differs from the linked version, the mismatch is confirmed. The user-side fix is to ensure `LD_LIBRARY_PATH` or the linker configuration (`/etc/ld.so.conf.d/`) prioritizes the CUDA version matching the wheel.

### Why it is non-deterministic

Several factors vary between runs:

- **ASLR** randomizes library mapping addresses, which affects whether a double-free hits unmapped memory (segfault) vs. still-valid memory (silent or no error)
- **`dlclose` ordering** depends on the full set of loaded shared objects, which varies with installed packages (numpy, etc.) and load timing
- **CUDA runtime state** depends on what GPU operations actually ran. Import-only tests may exit cleanly because CUDA never fully initialized its teardown hooks
- **Python GC timing** determines whether Cython destructors run before or during interpreter shutdown. If contexts are freed before shutdown starts, the `dlclose` race is avoided
- **Multiple CUDA versions** on the same system change which `libcudart.so` the linker resolves, varying the atexit handler behavior depending on environment variable ordering and ldconfig state

### Reproduction matrix

| Build method | Runtime environment | Clean exit? |
|---|---|---|
| `uv build --wheel` static (no `WITH_DYLIB`) | Any | Yes |
| `uv build --wheel` `WITH_DYLIB=1`, no auditwheel | Matching CUDA | Yes |
| `uv build --wheel` `WITH_DYLIB=1` + `auditwheel repair` | Single CUDA version | No, double free |
| `uv build --wheel` `WITH_DYLIB=1` + `auditwheel repair` | Multiple CUDA versions | No, double free |
| `uv build --wheel` `WITH_DYLIB=1` + `auditwheel repair --exclude` | Matching CUDA | Yes |
| `uv build --wheel` `WITH_DYLIB=1` + `auditwheel repair --exclude` | Mismatched CUDA version | No, double free |

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

**Status:** Untested. Should be investigated first -- if it works, it is strictly better than the `--exclude` approach.

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

#### Installed wheel structure (0.2.7, cp312, `--exclude` build)

A wheel built with the `--exclude` list above was installed and inspected. The `--exclude` strategy works as intended — the six bundled libraries in `cyllama/llama/` retain their original names with no hash suffixes or SONAME rewriting:

```
cyllama/llama/libggml-base.so.0
cyllama/llama/libggml-cpu.so
cyllama/llama/libggml-cuda.so
cyllama/llama/libggml.so.0
cyllama/llama/libllama.so.0
cyllama/llama/libmtmd.so.0
```

However, `libgomp.so.1` (GNU OpenMP runtime) was **not** in the `--exclude` list, so auditwheel relocated and SONAME-renamed it:

```
cyllama_cuda12.libs/libgomp-a34b3233.so.1.0.0
```

This is a potential problem. Bundling a private copy of libgomp can conflict with other packages (PyTorch, NumPy with OpenMP, etc.) that load the system's `libgomp.so.1` in the same process. While this is unlikely to cause the same `dlclose`-ordering crash as CUDA (OpenMP teardown is less dependent on unload order), it is in the same family of issues — see [LightGBM#6595](https://github.com/lightgbm-org/LightGBM/issues/6595) for a documented case of duplicate OpenMP runtimes causing segfaults on macOS when LightGBM and PyTorch are co-installed.

Adding `--exclude libgomp.so.1` to the exclude list would force the system-provided OpenMP to be used consistently and avoid this class of conflict.

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

## RPATH hardening

The project's shared libraries use `$ORIGIN`-relative RPATHs to resolve bundled dependencies:

| Extension location | INSTALL_RPATH | Resolves to |
|---|---|---|
| `cyllama/llama/llama_cpp.so` | `$ORIGIN` | `cyllama/llama/` |
| `cyllama/whisper/whisper_cpp.so` | `$ORIGIN/../llama` | `cyllama/llama/` |
| `cyllama/sd/stable_diffusion.so` | `$ORIGIN/../llama` | `cyllama/llama/` |

This ensures the bundled project libraries (`libllama.so.0`, `libggml-base.so.0`, `libggml-cpu.so`, etc.) in `cyllama/llama/` are always found before any system copies.

However, CUDA system libraries (`libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12`) are **not bundled** in the wheel (excluded via `--exclude`). They are resolved entirely by the system dynamic linker's default search order: `LD_LIBRARY_PATH`, then `RUNPATH`/`RPATH`, then `/etc/ld.so.conf`, then `/lib` and `/usr/lib`. There is no way to harden RPATH for libraries the wheel does not ship.

This is the same model used by PyTorch (`torch`), CuPy, and other CUDA wheels — CUDA runtime libraries are a system dependency because:

- The user must have a compatible GPU driver installed regardless (a system-level dependency)
- CUDA's forward-compatibility model ties the runtime to the driver version
- Bundling CUDA libs causes version conflicts when other packages (PyTorch, etc.) bundle different versions

### User-side mitigation for multi-version CUDA systems

When multiple CUDA toolkit versions are installed, users must ensure the correct version is found first:

```bash
# Option 1: Set LD_LIBRARY_PATH to prefer the matching CUDA version
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

# Option 2: Use update-alternatives to set the default CUDA version
sudo update-alternatives --set cuda /usr/local/cuda-12.4

# Option 3: Configure ldconfig to prioritize the correct version
echo "/usr/local/cuda-12.4/lib64" | sudo tee /etc/ld.so.conf.d/cuda-12.conf
sudo ldconfig
```

## Recommendation

Try **solution 1** (`auditwheel addtag`) first. If the wheel already meets the target manylinux policy (verify with `auditwheel show`), it is a one-line fix with no maintenance burden. If `addtag` doesn't work (policy violation), fall back to **solution 2** (`--exclude`), which is validated and whose maintenance cost is low.

If neither build-system approach is viable, **solution 5** (`RTLD_NODELETE`) is the most targeted runtime fix.

### Applies to all GPU backends, not just CUDA

Although the double-free crash was first observed with CUDA, the recommendation to use `auditwheel addtag` (or `--exclude`) applies equally to Vulkan, ROCm, SYCL, and any future GPU backend. The reasoning is backend-agnostic:

1. **The build already handles library placement.** All bundled `.so` files are placed in `cyllama/llama/` with correct `$ORIGIN` RPATHs. auditwheel's relocation machinery adds no value.
2. **SONAME rewriting is actively harmful.** It alters the `dlclose` unload ordering that glibc uses, which is the root cause of the crash. Any backend whose runtime registers `atexit` handlers (CUDA does; ROCm and SYCL may) is vulnerable.
3. **Vulkan wheels are affected too, just silently.** Vulkan uses explicit cleanup (`vkDestroy*`) rather than `atexit`, so the altered unload order doesn't cause a crash. But the unnecessary relocation still occurs, and the SONAME-renamed libraries serve no purpose.
4. **libgomp bundling is cross-backend.** Any GPU variant that links against OpenMP (which ggml uses for CPU threading) will have auditwheel bundle a private, SONAME-renamed copy of `libgomp.so.1`. This can conflict with other packages (PyTorch, NumPy) that load the system's libgomp in the same process — see [LightGBM#6595](https://github.com/lightgbm-org/LightGBM/issues/6595).

Inspection of an installed `--exclude` build (v0.2.7, cp312) confirmed that the only library auditwheel actually relocated was `libgomp.so.1` — everything else was excluded. The sole useful work `auditwheel repair` performed was stamping the manylinux platform tag, which is exactly what `auditwheel addtag` does without any relocation.

All Linux GPU wheel variants (CUDA, Vulkan, ROCm, SYCL) should use `auditwheel addtag` if the wheel meets the manylinux policy, or `auditwheel repair` with a comprehensive `--exclude` list (including `--exclude libgomp.so.1`) as a fallback.

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
