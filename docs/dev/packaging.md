# Wheel packaging: dylibs and rpaths

## Current approach (canonical delocate / auditwheel pattern)

cyllama does not install dynamic libs into `cyllama/llama/` inside the wheel.
The repair tool (`delocate` on macOS, `auditwheel` on Linux) is the single
authority for bundling and rewiring runtime libs.

Flow:

1. `manage.py build --all --dynamic` builds llama.cpp as shared libs into
   `thirdparty/llama.cpp/dynamic/`. Upstream install names are preserved
   (`LC_ID_DYLIB = @rpath/libllama.0.dylib`, etc.).
2. CMake builds the Cython extensions (`llama_cpp.*.so`, `whisper_cpp.*.so`,
   `embedded.*.so`, `stable_diffusion.*.so`) linking against those shared
   libs. Each extension gets:
   - `BUILD_RPATH = <build-tree lib dirs>` so editable / in-tree use works.
   - `INSTALL_RPATH = <same build-tree lib dirs>` so the repair tool can
     resolve `@rpath/libX.0.dylib` during wheel repair.
3. scikit-build-core produces the wheel. Extensions are inside, dylibs are
   not.
4. Under cibuildwheel, `delocate-wheel` walks the extension rpaths, resolves
   each `@rpath/libX.0.dylib` to the build-tree real file, copies it into
   `cyllama/.dylibs/`, rewrites every `LC_LOAD_DYLIB` to
   `@loader_path/../.dylibs/libX.0.dylib`, and sanitises the absolute
   build-tree paths out of the final wheel. Linux `auditwheel` does the
   equivalent with `cyllama.libs/` and `$ORIGIN`.
5. Final wheel contains one copy per lib in `cyllama/.dylibs/`, and
   extensions with relative load commands that don't depend on any
   external paths.

## Why this over shipping libs in `cyllama/llama/`

An earlier scheme installed dylibs into `cyllama/llama/` *and* let
`delocate` process them, producing duplicated copies (one in
`cyllama/llama/`, one in `cyllama/.dylibs/`, plus Python's zipfile
flattened each SONAME symlink chain into three identical real files
per lib — ~4x bloat). The canonical pattern removes the package-dir
install, so there's a single source of truth.

## macOS-specific: `libggml-blas` on non-Metal backends

GGML auto-links the Accelerate framework on macOS regardless of the
primary backend, so `libllama`/`libggml`/`libmtmd` carry a runtime
dep on `@rpath/libggml-blas.0.dylib` even on Vulkan builds. The dylib
must exist on the filesystem so `delocate` can resolve the chain.
`CMakeLists.txt` appends `ggml-blas` to `_OPTIONAL_DYLIB_NAMES` on
APPLE when `GGML_METAL=OFF` (Metal already requires it).

Accelerate itself is system-provided and not bundled.

## `libllama-common`

Upstream's `libllama-common` links against Homebrew's OpenSSL. On
`macos-15-intel` runners the Homebrew OpenSSL is built for min macOS
15.0, which clashes with our `MACOSX_DEPLOYMENT_TARGET=11.0`. No
cyllama extension references `libllama-common`, so it's not a concern
unless it ends up on the filesystem in a location `delocate` walks
into. Under the current canonical pattern nothing pulls it in, so no
exclusion is needed — the extension's rpath drives what gets bundled.

## Wheel repair command

```
delocate-wheel --require-archs <arch> -w <out-dir> -v <wheel.whl>
               --exclude libvulkan
               --exclude libMoltenVK
```

`libvulkan` and `libMoltenVK` are excluded because we expect users to
have the Vulkan loader / MoltenVK installed system-wide (typically via
Homebrew or the LunarG SDK). Bundling them would add ~100 MB and
constrain the MoltenVK version. Revisit if this becomes a user-facing
friction point.

## Windows: `delvewheel` and `GGML_BACKEND_DL` plugins

Builds use `GGML_BACKEND_DL=ON`, so GPU backends (`ggml-cuda`, `ggml-vulkan`)
are compiled as **plugin DLLs loaded at runtime via `LoadLibrary`**, not
linked into `llama.dll`. `scripts/manage.py` builds them explicitly as
separate targets and copies them into `thirdparty/llama.cpp/dynamic/`:

```python
targets = ["llama", "llama-common", "mtmd", "ggml-cpu"]
if backend_options.get("GGML_VULKAN") == "ON":
    targets.append("ggml-vulkan")
if backend_options.get("GGML_CUDA") == "ON":
    targets.append("ggml-cuda")
```

`delvewheel` only vendors DLLs that are **transitively referenced by a PE
import table** in the wheel's `.pyd` / `.dll` files. Because the backend
plugins are loaded via `LoadLibrary` and nothing in the wheel imports
them by name, delvewheel skips them unless told otherwise. The result:
`build_config.json` reports `cuda.enabled: true` / `vulkan.enabled: true`,
but at runtime `ggml_backend_load_all_from_path()` finds no GPU backend
and llama/whisper fall back to CPU silently.

(`auditwheel` on Linux and `delocate` on macOS pick up the plugin libs
via build-tree layout / rpath resolution, so the same source setup
produces correct wheels there without extra flags.)

Fix: extend `CIBW_REPAIR_WHEEL_COMMAND_WINDOWS` with `--include` for each
plugin:

```yaml
CIBW_REPAIR_WHEEL_COMMAND_WINDOWS: >
  delvewheel repair -w {dest_dir} {wheel}
  --add-path {project}\thirdparty\llama.cpp\dynamic
  --add-path {project}\thirdparty\whisper.cpp\dynamic
  --add-path {project}\thirdparty\stable-diffusion.cpp\dynamic
  --include ggml-vulkan.dll        # vulkan job
  # --include ggml-cuda.dll        # cuda job
```

For the CUDA job, `ggml-cuda.dll`'s transitive CUDA runtime deps
(`cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll`) must remain
`--no-dll`-excluded — users install the CUDA runtime themselves.
delvewheel will see those deps once `ggml-cuda.dll` is included, so the
existing exclusions stay necessary.

Verification after repair:

```bash
unzip -l cyllama_cuda12-*-win_*.whl  | grep -i ggml-cuda     # must show ggml-cuda-*.dll
unzip -l cyllama_vulkan-*-win_*.whl  | grep -i ggml-vulkan   # must show ggml-vulkan-*.dll
```

Note: smoke tests that only check `import cyllama` pass even when the GPU
backend plugin is missing. To catch this class of bug, assert the
expected backend is actually registered (e.g. via `ggml_backend_reg_names`
or `llama_supports_gpu_offload`).

## `SD_USE_VENDORED_GGML` — sharing ggml with stable-diffusion.cpp

Under `link_mode=dynamic`, stable-diffusion.cpp should link against
llama.cpp's shared ggml instead of vendoring its own. `manage.py` exposes
this via the `SD_USE_VENDORED_GGML` env var (or `--sd-shared-ggml` flag).
It must be set in `CIBW_ENVIRONMENT_*` for every job:

```yaml
SD_USE_VENDORED_GGML=${{ inputs.link_mode == 'dynamic' && '0' || '1' }}
```

Without it, `manage.py` defaults to vendored ggml and SD statically
embeds its own ggml with GPU kernels baked in. Observed impact on a
Windows CUDA wheel with the flag missing: `stable_diffusion.pyd` was
216 MB (vendored ggml + CUDA kernels) vs. ~23 MB on Linux where SD
shares the single `libggml-cuda-*.so` with llama/whisper.

This also causes `build_config.json` to omit `llama_cpp_ggml_version`:
with vendored ggml, `write_build_config` records only per-project
vendored versions, not a single shared llama.cpp ggml version — a useful
signal for auditing whether ggml is actually being shared.

Reference workflows with the pattern already applied:
`build-gpu-wheels.yml`, `build-gpu-wheels2.yml`.

## Installing from sdist (non-cibuildwheel) on macOS / Linux

Not directly supported under the current pattern: CMake does not
install any dylibs into the package, and without `delocate` /
`auditwheel` the extensions' `INSTALL_RPATH` points at a build-tree
path that won't survive `pip install .`. Use the published wheels or
an editable install.

## References

- [delocate docs](https://github.com/matthew-brett/delocate)
- [auditwheel docs](https://github.com/pypa/auditwheel)
- `CMakeLists.txt` — extension rpath setup (`add_cython_extension`
  function and per-extension overrides for `embedded`, `whisper_cpp`,
  `stable_diffusion`).
- `scripts/manage.py` — `LlamaCppBuilder.install_shared_libs` copies
  upstream dylibs into `thirdparty/llama.cpp/dynamic/` with symlinks
  preserved.
- `.github/workflows/build-new-wheels.yml` — `build_vulkan_macos_intel`
  job.
