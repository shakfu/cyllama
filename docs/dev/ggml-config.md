# ggml build configuration: GGML_NATIVE and CMAKE_CUDA_ARCHITECTURES

## Summary

Local CUDA builds are unnecessarily slow to compile and produce bloated
binaries because neither `GGML_NATIVE` nor `CMAKE_CUDA_ARCHITECTURES` is
defaulted for local development.  This document describes the current
behavior, the upstream constraints, and a concrete recommendation for
sensible defaults.

---

## Current behavior

### GGML_NATIVE

`GGML_NATIVE` tells ggml's cmake to optimize for the build machine's CPU
(native ISA detection: AVX2, AVX-512, etc.) and, when CUDA is enabled, to
set `CMAKE_CUDA_ARCHITECTURES=native` so only the installed GPU's SM
architecture is targeted.

| Build mode | What happens today | Location |
|---|---|---|
| Static (`build()`) | Only set if explicitly provided via `GGML_NATIVE=1` in env. Otherwise not passed to cmake, so cmake's own default applies (typically OFF). | `scripts/manage.py:1036-1041` |
| Dynamic (`build_shared()`) | Forcibly set to OFF because it is incompatible with `GGML_BACKEND_DL` (the dynamic backend plugin system). | `scripts/manage.py:1187-1190` |

The incompatibility between `GGML_NATIVE` and `GGML_BACKEND_DL` is enforced
by ggml-cpu's cmake at `ggml/src/ggml-cpu/CMakeLists.txt:382`:

```
GGML_NATIVE is not compatible with GGML_BACKEND_DL, consider using
GGML_CPU_ALL_VARIANTS
```

This is a CPU-side constraint -- `GGML_NATIVE` enables compile-time ISA
selection (`-march=native`), which conflicts with the runtime plugin loading
model where each CPU variant is a separate `.so`.  The CUDA architecture
selection (`CMAKE_CUDA_ARCHITECTURES=native`) has no such conflict.

### CMAKE_CUDA_ARCHITECTURES

Pure passthrough from the environment.  Only set if the user explicitly
provides it (`scripts/manage.py:964-966`).

If not set, ggml-cuda's own default logic applies
(`build/llama.cpp/ggml/src/ggml-cuda/CMakeLists.txt:8-55`).  With CUDA 12.0
and `GGML_NATIVE` unset, this produces:

```
50-virtual  61-virtual  70-virtual  75-virtual  80-virtual  86-real  89-real
```

Seven architectures.  Each one roughly multiplies the compiled kernel code.

### Net effect on local builds

A local `make build-cuda-dynamic` with no env vars:

- Does not optimize for the local CPU (no `GGML_NATIVE`)

- Builds CUDA kernels for 7+ SM architectures instead of the one installed GPU

- `libggml-cuda.so` is ~509 MB (vs ~137 MB with `native`)

- Compilation is slow due to redundant architecture codegen

### CI builds

CI workflows already set both variables explicitly:

```yaml
# .github/workflows/build-gpu-wheels.yml:103,116
# .github/workflows/build-new-wheels.yml:280
CMAKE_CUDA_ARCHITECTURES="75"
GGML_NATIVE=OFF
```

This has been `"75"` (sm_75 / Turing) since the first CUDA CI workflow.
Any new defaults must not override these explicit settings.

---

## Observed sizes

Measured on an RTX 4060 (sm_89) with CUDA 12.0, dynamic build,
`SD_USE_VENDORED_GGML=0`:

| Configuration | `libggml-cuda.so` | SD `.so` | Notes |
|---|---:|---:|---|
| No arch set (ggml defaults) | 509 MB | 23 MB | 7 architectures |
| `CMAKE_CUDA_ARCHITECTURES=native` | 137 MB | 23 MB | sm_89 only |
| CI (`CMAKE_CUDA_ARCHITECTURES=75`) | ~202 MB | ~210 MB* | sm_75 only, *vendored ggml |

---

## Recommendation

### Local builds: default to native

**Static builds**: Default `GGML_NATIVE=ON` when the environment variable is
not set.  This gives native CPU ISA optimization and
`CMAKE_CUDA_ARCHITECTURES=native` in one flag.

**Dynamic builds**: Cannot use `GGML_NATIVE` (BACKEND_DL conflict).  Instead,
when CUDA is enabled and `CMAKE_CUDA_ARCHITECTURES` is not explicitly set,
default it to `native`.  This targets only the installed GPU without
triggering the CPU-side incompatibility.

### CI builds: no change needed

CI already sets `GGML_NATIVE=OFF` and `CMAKE_CUDA_ARCHITECTURES="75"`
explicitly.  These override any defaults.

### Implementation

Two changes in `scripts/manage.py`:

**1. Default `GGML_NATIVE=ON` for static builds (`get_backend_cmake_options`,
~line 1036):**

```python
# Before:
ggml_native = os.environ.get("GGML_NATIVE")
if ggml_native is not None:
    options["GGML_NATIVE"] = "ON" if ggml_native == "1" else "OFF"
    self.log.info(f"  GGML_NATIVE={options['GGML_NATIVE']}")

# After:
ggml_native = os.environ.get("GGML_NATIVE")
if ggml_native is not None:
    options["GGML_NATIVE"] = "ON" if ggml_native == "1" else "OFF"
else:
    options["GGML_NATIVE"] = "ON"
self.log.info(f"  GGML_NATIVE={options['GGML_NATIVE']}")
```

`build_shared()` already forces `GGML_NATIVE=OFF` after calling
`get_backend_cmake_options()`, so the new default is harmless for dynamic
builds.

**2. Default `CMAKE_CUDA_ARCHITECTURES=native` for dynamic builds
(`build_shared`, ~line 1190):**

```python
# After forcing GGML_NATIVE=OFF:
if "GGML_NATIVE" not in backend_options:
    backend_options["GGML_NATIVE"] = "OFF"

# Add: default CUDA to native arch when not explicitly set
if (backend_options.get("GGML_CUDA") == "ON"
        and "CMAKE_CUDA_ARCHITECTURES" not in backend_options):
    backend_options["CMAKE_CUDA_ARCHITECTURES"] = "native"
    self.log.info("  CMAKE_CUDA_ARCHITECTURES=native (default for local build)")
```

### Behavior matrix after changes

| | `GGML_NATIVE` | `CMAKE_CUDA_ARCHITECTURES` | Source |
|---|---|---|---|
| Local static | ON | native (via GGML_NATIVE) | New default |
| Local dynamic | OFF (forced) | native | New default |
| CI static | OFF | 75 | Explicit env |
| CI dynamic | OFF (forced) | 75 | Explicit env |
| User override | Respected | Respected | Env var takes precedence |

---

## References

- `scripts/manage.py:964-966` -- `CMAKE_CUDA_ARCHITECTURES` passthrough

- `scripts/manage.py:1036-1041` -- `GGML_NATIVE` handling

- `scripts/manage.py:1187-1190` -- `GGML_NATIVE` forced OFF in `build_shared()`

- `build/llama.cpp/ggml/src/ggml-cuda/CMakeLists.txt:8-55` -- ggml-cuda default arch logic

- `build/llama.cpp/ggml/src/ggml-cpu/CMakeLists.txt:382` -- NATIVE vs BACKEND_DL incompatibility

- `.github/workflows/build-gpu-wheels.yml:103,116` -- CI arch settings

- `.github/workflows/build-new-wheels.yml:280` -- CI arch settings
