# Windows CUDA wheel: prebuilt DLL path

## Summary

llama.cpp ships prebuilt CUDA Windows DLLs with every release (e.g.
[b8828](https://github.com/ggml-org/llama.cpp/releases/tag/b8828) includes
`llama-b8828-bin-win-cuda-12.4-x64.zip`). `manage.py` already knows how to
consume them — but the current `build-new-wheels.yml` CUDA-Windows job does
not use them. This document explains why, what changes to make, and the
tradeoffs.

## Current state

The CUDA-Windows job in `.github/workflows/build-new-wheels.yml` pins
`WITH_DYLIB=0` and does not pass `--dynamic` to `manage.py build`. That
forces a full static compile of llama.cpp under cibuildwheel, which invokes
`nvcc` against the ggml-cuda source tree. This is by far the slowest part
of the CI job and requires the full CUDA toolkit to be installed on the
runner.

Meanwhile, `manage.py:1329-1331` already constructs the right asset
filename for a Windows CUDA release:

```python
if getenv("GGML_CUDA", default=False):
    cuda_ver = os.environ.get("LLAMACPP_CUDA_RELEASE", "12.4")
    return f"llama-{version}-bin-win-cuda-{cuda_ver}-{arch_tag}.zip"
```

And the build dispatcher at `manage.py:2259` selects between
`download_release()` (fast) and `build_shared()` (slow) based on whether
an asset exists and whether SD is sharing llama.cpp's ggml.

## Recommended config

Switch the CUDA-Windows job to the dynamic + vendored-SD config. Three
env changes in `CIBW_ENVIRONMENT_WINDOWS`:

```yaml
CIBW_ENVIRONMENT_WINDOWS: >
  GGML_CUDA=1
  GGML_METAL=0
  GGML_NATIVE=OFF
  CMAKE_CUDA_ARCHITECTURES="75"
  WITH_DYLIB=1
  SD_USE_VENDORED_GGML=1
```

And pass `--dynamic` in the build step:

```yaml
python scripts/manage.py build --all --deps-only --no-sd-examples --dynamic &&
```

`SD_USE_VENDORED_GGML=1` is the critical pairing: the upstream prebuilt
Windows DLLs are compiled with stock `GGML_MAX_NAME=64`. If SD tried to
share those DLLs' ggml it would hit the same `ggml_tensor` layout
mismatch documented in `ggml_max_name.md`. By having SD statically link
its own vendored ggml (self-consistent at whatever `GGML_MAX_NAME` SD
defaults to), the layout hazard disappears.

With that pairing in place, the gate at `manage.py:2259`
(`if asset is None or _sd_uses_shared_ggml()`) evaluates false and takes
the `download_release()` fast path.

## Tradeoffs

**Wins:**

- Much faster CI: no `nvcc` compile of ggml-cuda from source — the
  longest single step in today's Linux CUDA job is eliminated on Windows

- Smaller runner footprint: arguably no need for the full CUDA toolkit
  install, just the runtime DLLs that the prebuilt zip already contains
  (verify before relying on this — nvcc may still be needed for Cython
  bindings that touch CUDA headers)

- Matches the upstream llama.cpp binary exactly, which simplifies bug
  reports

**Costs:**

- SD statically links its own ggml, so the wheel carries two ggml copies
  (llama.cpp's in the DLLs + SD's embedded in `libstable-diffusion.a`).
  Slightly larger than the shared-ggml Linux CUDA wheel

- Locked to the CUDA version upstream ships (currently 12.4). Users on
  older drivers may have compatibility issues. `LLAMACPP_CUDA_RELEASE`
  lets you pick a different release variant if upstream publishes more
  than one

## delvewheel repair

When `WITH_DYLIB=1`, the following DLLs will need to land in the wheel
from `thirdparty/llama.cpp/dynamic/`:

- `llama.dll`

- `ggml.dll`

- `ggml-base.dll`

- `ggml-cpu.dll`

- `ggml-cuda.dll`

Verify that `delvewheel repair` picks them up via the build artifact path
after scikit-build-core installs them into the wheel tree. The current
`--no-dll` exclusions for `nvcuda.dll`, `cudart64_12.dll`,
`cublas64_12.dll`, `cublasLt64_12.dll` are still correct (those are
system libs the user must provide via the NVIDIA driver / CUDA runtime).

If delvewheel starts mangling the bundled DLL names (its default for
non-system libs), add `--no-mangle` for the project libs above — this is
the Windows analogue of the Linux auditwheel SONAME-rewrite issue
documented in `cuda-double-free.md`.

## Alternative: shared-ggml + source build

To match the Linux-CUDA dynamic wheel exactly (shared ggml across
llama/whisper/SD, `GGML_MAX_NAME=128`, single ggml copy), the config
would be:

```
WITH_DYLIB=1
SD_USE_VENDORED_GGML=0
```

But this forces `build_shared()` from source (the gate at
`manage.py:2259` takes the slow path), which eliminates the speed win
that motivated using the prebuilt DLLs in the first place. Not
recommended for CI unless wheel size becomes a hard constraint.

## Rollout

For the new-wheels testbed, start with dynamic + vendored. Once CI is
green and a test install works on a Windows box with an NVIDIA GPU,
decide whether the few MB of duplicated ggml matters enough to switch to
the source-build shared-ggml config.
