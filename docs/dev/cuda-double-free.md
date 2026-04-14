# CUDA double free or corruption (!prev)

## Summary

Dynamic-linked CUDA wheels (`WITH_DYLIB=1`) crash with `double free or corruption (!prev)` during Python interpreter shutdown. The crash is non-deterministic and only affects wheels processed by `auditwheel repair`.

This issue is **Linux-specific**. The entire chain -- `auditwheel repair`, `patchelf` SONAME rewriting, glibc `dlclose` unload ordering -- only exists on Linux. macOS and Windows are not affected:

- **macOS** uses `delocate` for wheel repair, which rewrites Mach-O load commands via `install_name_tool` rather than ELF SONAME headers. `delocate` does not alter the dyld unload order in the way that triggers this crash. macOS wheels have their own issues (e.g. duplicate OpenMP runtimes causing segfaults when co-installed with PyTorch -- see [LightGBM#6595](https://github.com/lightgbm-org/LightGBM/issues/6595)), but those are a different problem with different root causes.
- **Windows** uses no wheel repair tool. DLLs are bundled as-is with no SONAME equivalent to rewrite, and the Windows loader does not have the same unload-ordering sensitivity as glibc's `dlclose`.

First observed against commit `36db368`. The initial fix attempted Python-level cleanup before shutdown, but this caused instability on the Metal backend as well. Reverted in `fb522c8`.

## Root cause

The issue is caused by `auditwheel repair`, specifically its use of `patchelf` to rewrite ELF SONAME headers on bundled shared libraries.

During repair, auditwheel:

1. Copies bundled `.so` files (`libllama.so`, `libggml-cuda.so`, etc.) into `cyllama.libs/`
2. Renames them with hash suffixes (e.g., `libllama-91896a1c.so.0.0.1`)
3. Rewrites their ELF SONAME headers via `patchelf` so the dynamic linker resolves the renamed copies

The SONAME rewrite changes the dependency graph that glibc's dynamic linker uses to determine `dlclose` unload ordering. CUDA's runtime (`libcudart`) registers internal `atexit` handlers during initialization. When Python shuts down, the altered unload order can cause CUDA's handlers to fire after the memory they reference (in `libggml-cuda.so`) has already been unmapped.

The primary suspect is the SONAME rewrite on `libggml-cuda.so` specifically, since it is the library that links directly against `libcudart` and whose unload ordering relative to CUDA's atexit handlers matters. However, this has not been isolated -- the validated fix excludes all bundled project libs at once. See [Open question: minimal exclude set](#open-question-minimal-exclude-set) below.

### Why excluding only CUDA runtime libs is insufficient

The original CI workflow already excluded the CUDA runtime system libraries (`libcuda.so.1`, `libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12`) from `auditwheel repair`. **The double-free still occurred.** This is because those excludes only prevent auditwheel from bundling the CUDA SDK -- they do not prevent it from SONAME-rewriting the project's own bundled libraries.

With only runtime excludes, auditwheel still:

1. Copies `libggml-cuda.so`, `libllama.so.0`, etc. from `cyllama/llama/` into `cyllama_cuda12.libs/`
2. Renames them with hash suffixes (e.g., `libggml-cuda-3e3d7523.so`)
3. Rewrites their SONAME headers

This SONAME rewrite on `libggml-cuda.so` (and possibly the other bundled libs -- see open question below) alters the `dlclose` unload graph, triggering the crash. The fix requires excluding the bundled project libs from auditwheel's relocation, in addition to the runtime system libs.

### What each category of excludes does

The `--exclude` flags serve two distinct purposes:

**GPU runtime system libs** (e.g., `libcuda.so.1`, `libcudart.so.12`, `libcublas.so.12`): These are system-installed libraries that the wheel should never bundle. Excluding them ensures the user's own CUDA installation is used at runtime, avoiding version conflicts. This is the same approach used by PyTorch and CuPy. Note: excluding these alone does **not** fix the double-free.

**Bundled project libs** (e.g., `libllama.so.0`, `libggml-cuda.so`, `libggml.so.0`, `libmtmd.so.0`): These are already placed in `cyllama/llama/` by the build system with correct `$ORIGIN` RPATHs. Excluding them prevents auditwheel from relocating them to `cyllama_cuda12.libs/` and SONAME-rewriting them. This is where the double-free fix actually lies -- though it is not yet known whether all of these need to be excluded or only `libggml-cuda.so`.

**`libgomp.so.1`** (GCC OpenMP runtime): A system library linked by `libggml-cpu.so`. Without excluding it, auditwheel bundles a private, SONAME-renamed copy. This can conflict with other packages (PyTorch, NumPy) that load the system's libgomp in the same process -- see [LightGBM#6595](https://github.com/lightgbm-org/LightGBM/issues/6595).

### Why all bundled project libs must be excluded

It might seem that only `libggml-cuda.so` needs to be excluded (since it's the only bundled lib that links directly against `libcudart`). However, a minimal exclude test (runtime libs + `libggml-cuda.so` only) revealed that **auditwheel refuses to produce a wheel** unless all bundled project libs are excluded:

```
auditwheel: error: cannot repair "cyllama_cuda12-0.2.7-cp312-cp312-linux_x86_64.whl"
to "manylinux_2_35_x86_64" ABI because of the presence of too-recent versioned symbols.
You'll need to compile the wheel on an older toolchain.
```

When `libggml-cuda.so` alone is excluded, auditwheel attempts to relocate the remaining bundled libs (`libllama.so.0`, `libggml.so.0`, `libggml-base.so.0`, `libggml-cpu.so`, `libmtmd.so.0`, `libgomp.so.1`). These libs contain glibc symbols newer than what the `manylinux_2_35` policy allows, so auditwheel aborts. When **all** bundled libs are excluded, auditwheel has nothing to relocate and simply stamps the manylinux tag -- it never inspects the symbol versions.

This means the full exclude list is not just precautionary -- it is **required** for the repair step to succeed. It is also not possible to isolate whether `libggml-cuda.so` is the sole double-free trigger via `--exclude`, since auditwheel won't produce a partially-excluded wheel. (Isolating the trigger would require manually running `patchelf` on individual libs in an otherwise-unrepaired wheel, which is not worth the effort given that the full exclude approach works and has no downsides.)

### Why it only affects CUDA

Vulkan wheels go through the same `auditwheel repair` process on their bundled llama/ggml libs and do not crash. The difference is that CUDA's runtime manages teardown via `atexit` handlers registered internally by `libcudart`. Vulkan cleanup is explicit (the application calls `vkDestroy*` functions), so unload ordering is irrelevant. ROCm and SYCL have not been tested but may exhibit similar issues if their runtimes use atexit-based teardown.

### Multiple CUDA versions installed simultaneously

When a system has multiple CUDA toolkit versions installed (e.g. CUDA 12 and CUDA 13), the dynamic linker may resolve `libcudart.so` to a different major version than the one the wheel was built against. This causes the same double-free symptom through a different mechanism:

1. The wheel's bundled `libggml-cuda.so` is linked against CUDA 12 (built with `cuda-12.4`)
2. At runtime, the dynamic linker resolves `libcudart.so` to the CUDA 13 version from the system, depending on `LD_LIBRARY_PATH` ordering and ldconfig priority
3. The mismatched runtime initializes its own internal atexit handlers with different memory layout assumptions
4. On shutdown, the CUDA 13 teardown attempts to free structures allocated by CUDA 12 conventions (or vice versa), triggering the double-free

This is a well-documented pattern in the CUDA ecosystem. PyTorch, CuPy, and other projects that ship CUDA wheels use `--exclude` to avoid bundling CUDA runtime libraries for exactly this reason -- letting the system provide a single consistent CUDA installation. However, `--exclude` alone does not prevent the issue when multiple system CUDA versions coexist and the linker picks the wrong one.

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

| Build method | Runtime environment | Clean exit? | Notes |
|---|---|---|---|
| `uv build --wheel` static (no `WITH_DYLIB`) | Any | Yes | No shared libs to relocate |
| `uv build --wheel` `WITH_DYLIB=1`, no auditwheel | Matching CUDA | Yes | No SONAME rewriting |
| `uv build --wheel` `WITH_DYLIB=1` + `auditwheel repair` (no excludes) | Single CUDA version | No | auditwheel rewrites all bundled libs |
| `uv build --wheel` `WITH_DYLIB=1` + `auditwheel repair` (runtime excludes only) | Single CUDA version | No | auditwheel still rewrites bundled project libs |
| `uv build --wheel` `WITH_DYLIB=1` + `auditwheel repair` (runtime + `libggml-cuda.so` only) | N/A | N/A | auditwheel refuses: remaining libs have too-recent glibc symbols for manylinux_2_35 |
| `uv build --wheel` `WITH_DYLIB=1` + `auditwheel repair` (runtime + all bundled excludes) | Matching CUDA | Yes | Validated fix |
| `uv build --wheel` `WITH_DYLIB=1` + `auditwheel repair` (runtime + all bundled excludes) | Mismatched CUDA version | No | CUDA version mismatch, different root cause |

## Solutions

Seven approaches are described below, ordered from most recommended to least.

### 1. Skip `auditwheel repair` entirely

Set `CIBW_REPAIR_WHEEL_COMMAND_LINUX: ""` so cibuildwheel skips the repair step. The wheel keeps its `linux_x86_64` platform tag. No SONAME rewriting, no library relocation, no `dlclose` ordering changes.

**Pros:**
- Eliminates the entire class of SONAME-rewriting bugs
- No exclude list to maintain
- One-line change

**Cons:**
- The wheel gets a `linux_x86_64` tag instead of `manylinux`, which pip may refuse to install on some systems and PyPI will reject for upload
- Only viable for self-hosted wheel indexes or direct installation

**Status:** Validated. The test workflow at `.github/workflows/test-cuda-wheel.yml` includes a no-repair strategy that produces an identical wheel (in content) to the `--exclude` strategy, differing only in the platform tag.

### 2. `--exclude` bundled libraries from auditwheel repair

Add `--exclude` flags for **both** GPU runtime system libs and bundled project libs so auditwheel leaves them all untouched (no relocation, no SONAME rewrite). The only work auditwheel performs is stamping the manylinux platform tag.

```yaml
# CUDA example -- other backends substitute their own runtime libs
# and backend-specific ggml lib (libggml-hip.so, libggml-vulkan.so, etc.)
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

The exclude list has two categories:

- **GPU runtime system libs** (`libcuda*`, `libcublas*`): Prevents auditwheel from bundling system-installed CUDA libraries. The user's own CUDA installation is used at runtime. Excluding these alone does not fix the double-free.
- **Bundled project libs** (`libllama*`, `libggml*`, `libmtmd*`) and **`libgomp.so.1`**: Prevents auditwheel from relocating and SONAME-rewriting libraries that the build system already placed in `cyllama/llama/` with correct `$ORIGIN` RPATHs. This is where the double-free fix lies. All bundled project libs must be excluded -- auditwheel refuses to produce a wheel if any non-excluded libs contain glibc symbols too recent for the target manylinux policy (see [why all bundled project libs must be excluded](#why-all-bundled-project-libs-must-be-excluded)).

**Pros:**
- Directly addresses the root cause (SONAME rewriting of bundled project libs)
- Produces a manylinux-tagged wheel that pip and PyPI accept
- Validated in the reproduction matrix and on hardware (RTX 4060)
- Zero runtime cost

**Cons:**
- Maintenance burden: every new upstream `.so` (new ggml backend, new library) requires a new `--exclude` line. Missing one will cause auditwheel to fail (too-recent glibc symbols) or, if the symbols happen to pass the policy check, reintroduce the SONAME rewriting
- Weaker manylinux compliance: trusts the build's RPATHs rather than auditwheel's relocation guarantees

**Status:** Validated. Applied to all GPU variants in `.github/workflows/build-gpu-wheels2.yml`. Test workflow at `.github/workflows/test-cuda-wheel.yml`.

### 3. `auditwheel addtag` instead of `auditwheel repair`

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
- Requires auditwheel >= 6.0. Stock manylinux2014 and manylinux_2_28 images ship older versions that do not include `addtag` (available commands: `show`, `repair`, `lddtree`). Must upgrade auditwheel in the container first (e.g. `pip install 'auditwheel>=6.0'` in `CIBW_BEFORE_BUILD`)

**Status:** Not viable with current CI images. The manylinux containers used in CI do not ship auditwheel >= 6.0, and the `addtag` subcommand is not available. Upgrading auditwheel in-container is possible but adds complexity for no benefit over `--exclude`.

### 4. Static linking

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

### 5. Explicit Python-level cleanup before shutdown

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

### 6. `RTLD_NODELETE` on the CUDA-linked library

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

This is the same model used by PyTorch (`torch`), CuPy, and other CUDA wheels -- CUDA runtime libraries are a system dependency because:

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

Use **solution 2** (`--exclude` with both runtime and bundled project libs). It is validated, produces manylinux-tagged wheels, and works with the auditwheel version shipped in stock manylinux containers.

The `--exclude` list must include **both categories**:
1. GPU runtime system libs (backend-specific: `libcuda*`, `libamdhip*`, `libsycl*`, `libvulkan*`, etc.)
2. Bundled project libs (`libllama.so.0`, `libggml*.so`, `libmtmd.so.0`) and `libgomp.so.1`

Excluding only the runtime libs (category 1) is insufficient -- the double-free persisted with the original CI workflow that excluded only `libcuda.so.1`, `libcudart.so.12`, `libcublas.so.12`, and `libcublasLt.so.12`. The SONAME rewrite on the bundled project libs (particularly `libggml-cuda.so`) is the actual trigger.

All bundled project libs must be excluded -- not just `libggml-cuda.so`. A minimal exclude test (runtime libs + `libggml-cuda.so` only) showed that auditwheel refuses to process the remaining bundled libs because they contain glibc symbols too recent for the `manylinux_2_35` policy. The full exclude list is the only viable configuration.

### Applies to all GPU backends, not just CUDA

Although the double-free crash was first observed with CUDA, the `--exclude` approach applies equally to Vulkan, ROCm, SYCL, and any future GPU backend. The reasoning is backend-agnostic:

1. **The build already handles library placement.** All bundled `.so` files are placed in `cyllama/llama/` with correct `$ORIGIN` RPATHs. auditwheel's relocation machinery adds no value.
2. **SONAME rewriting is actively harmful for CUDA, and unnecessary for others.** It alters the `dlclose` unload ordering that glibc uses, which is the root cause of the CUDA crash. Any backend whose runtime registers `atexit` handlers (CUDA does; ROCm and SYCL may) is vulnerable.
3. **Vulkan wheels are affected too, just silently.** Vulkan uses explicit cleanup (`vkDestroy*`) rather than `atexit`, so the altered unload order doesn't cause a crash. But the unnecessary relocation still occurs, and the SONAME-renamed libraries serve no purpose.
4. **libgomp bundling is cross-backend.** Any GPU variant that links against OpenMP (which ggml uses for CPU threading) will have auditwheel bundle a private, SONAME-renamed copy of `libgomp.so.1`. This can conflict with other packages (PyTorch, NumPy) that load the system's libgomp in the same process -- see [LightGBM#6595](https://github.com/lightgbm-org/LightGBM/issues/6595).

The full `--exclude` list per backend (runtime libs vary, bundled project libs are the same):

| Backend | Runtime system lib excludes | Bundled project lib excludes |
|---|---|---|
| CUDA | `libcuda.so.1`, `libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12` | `libllama.so.0`, `libggml.so.0`, `libggml-base.so.0`, `libggml-cuda.so`, `libggml-cpu.so`, `libmtmd.so.0`, `libgomp.so.1` |
| ROCm | `libamdhip64.so.6`, `libhipblas.so.2`, `librocblas.so.4`, `libhsa-runtime64.so.1`, `librocsolver.so.0`, `libhipblaslt.so.0`, `libamd_comgr.so.2`, `librocprofiler-register.so.0` | `libllama.so.0`, `libggml.so.0`, `libggml-base.so.0`, `libggml-hip.so`, `libggml-cpu.so`, `libmtmd.so.0`, `libgomp.so.1` |
| SYCL | `libsycl.so.8`, `libOpenCL.so.1`, `libsvml.so`, `libimf.so`, `libintlc.so.5`, `libtbb.so.12` | `libllama.so.0`, `libggml.so.0`, `libggml-base.so.0`, `libggml-sycl.so`, `libggml-cpu.so`, `libmtmd.so.0`, `libgomp.so.1` |
| Vulkan | `libvulkan.so.1` | `libllama.so.0`, `libggml.so.0`, `libggml-base.so.0`, `libggml-vulkan.so`, `libggml-cpu.so`, `libmtmd.so.0`, `libgomp.so.1` |

### Installed wheel structure (0.2.7, cp312, `--exclude` build)

A wheel built with the full `--exclude` list was installed and inspected. The bundled libraries in `cyllama/llama/` retain their original names with no hash suffixes or SONAME rewriting:

```
cyllama/llama/libggml-base.so.0
cyllama/llama/libggml-cpu.so
cyllama/llama/libggml-cuda.so
cyllama/llama/libggml.so.0
cyllama/llama/libllama.so.0
cyllama/llama/libmtmd.so.0
```

No `cyllama_cuda12.libs/` directory exists -- auditwheel did not relocate any libraries. The only work auditwheel performed was stamping the manylinux platform tag.

### Runtime validation

The installed wheel was tested on a system with an NVIDIA GeForce RTX 4060 (compute capability 8.9). All three extensions load the CUDA and CPU backends from the correct unbundled paths in `cyllama/llama/`, confirming that the `$ORIGIN` RPATHs work without auditwheel relocation:

```
$ uv run cyllama info

llama.cpp:
  load_backend: loaded CUDA backend from .../cyllama/llama/libggml-cuda.so
  load_backend: loaded CPU backend from .../cyllama/llama/libggml-cpu.so
  built: CUDA | registries: CUDA, CPU | GPU offload: True

whisper.cpp:
  load_backend: loaded CUDA backend from .../cyllama/llama/libggml-cuda.so
  load_backend: loaded CPU backend from .../cyllama/llama/libggml-cpu.so
  built: CUDA | backends: CUDA, CUDA

stable-diffusion.cpp:
  load_backend: loaded CUDA backend from .../cyllama/llama/libggml-cuda.so
  load_backend: loaded CPU backend from .../cyllama/llama/libggml-cpu.so
  built: CUDA | backends: CUDA, CUDA, CUDA
```

All three extensions share the same `libggml-cuda.so` and `libggml-cpu.so` from `cyllama/llama/` -- no duplicated or SONAME-renamed copies.

The bundled `libggml-cuda.so` correctly resolves CUDA system libraries via the dynamic linker rather than bundling them:

```
$ ldd .../cyllama/llama/libggml-cuda.so | grep -E 'cuda|cublas'
  libcudart.so.12   => /lib/x86_64-linux-gnu/libcudart.so.12
  libcublas.so.12   => /lib/x86_64-linux-gnu/libcublas.so.12
  libcuda.so.1      => /lib/x86_64-linux-gnu/libcuda.so.1
  libcublasLt.so.12 => /lib/x86_64-linux-gnu/libcublasLt.so.12
```

This confirms the `--exclude` strategy works end-to-end: the wheel's bundled libraries use `$ORIGIN` RPATHs to find each other, while CUDA runtime libraries are resolved from the system installation. This is the same model used by PyTorch and CuPy -- CUDA libs are a system dependency, not bundled in the wheel.

## Test workflow

`.github/workflows/test-cuda-wheel.yml` is a standalone CI workflow for validating the fix. It builds a single CUDA wheel (cp312 only) once and then tests two repair strategies as lightweight downstream jobs on the same artifact:

1. **No-repair** (Strategy A): `CIBW_REPAIR_WHEEL_COMMAND_LINUX: ""` -- the unrepaired wheel with a `linux_x86_64` tag. Serves as a baseline to confirm the build itself is correct.
2. **Full exclude** (Strategy B): `auditwheel repair --exclude ...` with the complete exclude list (runtime libs + all bundled project libs + libgomp) -- the production-equivalent wheel with a manylinux tag.

A third strategy (minimal exclude: runtime libs + `libggml-cuda.so` only) was attempted but auditwheel refuses to produce the wheel because the remaining non-excluded bundled libs contain glibc symbols too recent for `manylinux_2_35`. This confirms the full exclude list is required.

The build happens once; the repair strategies run as downstream jobs, avoiding duplicate CUDA builds.

Trigger it manually from the Actions tab. Note that tests run on CPU-only runners (no GPU available), so they validate clean `dlclose` ordering but cannot exercise the full CUDA atexit path. A passing test gives moderate confidence; a failing test is definitive.

## References

- Reproduction report: colleague's analysis of commit `2755d96`
- Revert: `fb522c8` (reverted `36db368`)
- auditwheel SONAME rewriting: [pypa/auditwheel#289](https://github.com/pypa/auditwheel/issues/289)
- patchelf SONAME behavior: [NixOS/patchelf#275](https://github.com/NixOS/patchelf/issues/275)
