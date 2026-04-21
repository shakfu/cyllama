# GGML_MAX_NAME ABI mismatch in dynamic Vulkan wheels

## Summary

Vulkan wheels built from the `build-gpu-wheels.yml` workflow with
`link_mode=dynamic` crash during stable-diffusion image generation with:

```
ggml/src/ggml-backend.cpp:478: GGML_ASSERT(ggml_are_same_layout(src, dst) &&
  "cannot copy tensors with different layouts") failed
```

The crash fires on the first `ggml_backend_tensor_copy()` that transfers a
CPU-offloaded tensor to the Vulkan buffer. CUDA dynamic wheels built from
the same workflow do **not** crash, which is the clue that isolates the
bug.

The symptom looks identical to the `GGML_MAX_NAME` ABI issue that the
0.2.9 → Unreleased work is supposed to fix (via `_sync_ggml_abi()` and
`CMAKE_C_FLAGS=-DGGML_MAX_NAME=128` propagation in
`LlamaCppBuilder.build_shared()`). The fix is in place; Vulkan just
doesn't go through that code path.

## What has to match, and why it's not a Vulkan-specific rule

Vulkan does **not** have a special `GGML_MAX_NAME` requirement that CUDA
lacks. The rule is the same for every backend: when
`SD_USE_VENDORED_GGML=0`, **every** ggml-related `.so` that SD links
against must be compiled with `GGML_MAX_NAME=128`. That set is
`libggml-base.so`, `libggml.so`, `libggml-cpu.so`, and the backend
plugin (`libggml-cuda.so`, `libggml-vulkan.so`, `libggml-hip.so`, …).
They all read and write `ggml_tensor` structs across the boundary with
SD; any one of them seeing a different layout corrupts field offsets.

What's different between backends is *how the `.so` files end up on
disk*:

| Backend | Delivery mechanism in dynamic mode | Compiled with `GGML_MAX_NAME=128`? |
|---------|------------------------------------|-----------------------------------|
| CUDA    | built from source (`build_shared`)   | yes — `CMAKE_C_FLAGS` injection at `manage.py:1215` |
| SYCL    | built from source (`build_shared`)   | yes — same path |
| HIP     | built from source (`build_shared`)   | yes — same path |
| Vulkan  | downloaded upstream release (`download_release`) | **no** — upstream ships `=64` |

So the Vulkan crash isn't about Vulkan having a different requirement —
it's that the requirement was silently not being met because the libs
came pre-compiled from a third party. The fix just routes Vulkan
through the same `build_shared()` path the other GPU backends already
use, so the existing `GGML_MAX_NAME=128` injection actually reaches
`libggml-vulkan.so`.

## Root cause

`scripts/manage.py:2252` chooses between two strategies when building
llama.cpp in dynamic mode:

```python
if args.dynamic and BuilderClass == LlamaCppBuilder:
    asset = builder._release_asset_name()
    if asset is None:
        builder.build_shared()     # compile from source
    else:
        builder.download_release() # grab upstream pre-built tarball
```

`_release_asset_name()` (manage.py:1300) returns:

| Backend | Linux return value |
|---------|--------------------|
| CUDA    | `None` (no upstream release) |
| SYCL    | `None` |
| HIP     | `None` |
| Vulkan  | `llama-{version}-bin-ubuntu-vulkan-x64.tar.gz` |
| CPU     | `llama-{version}-bin-ubuntu-x64.tar.gz` |

So on CUDA the code falls into `build_shared()`, which (since b378cba)
injects `-DGGML_MAX_NAME=128` into `CMAKE_C_FLAGS` / `CMAKE_CXX_FLAGS`
whenever `SD_USE_VENDORED_GGML=0`. The resulting `libggml-base.so`,
`libggml.so`, `libggml-cuda.so` all have `ggml_tensor` structs with
`char name[128]`. stable-diffusion.cpp's `libstable-diffusion.a` uses
`GGML_MAX_NAME=128` (its own default, see
`build/stable-diffusion.cpp/CMakeLists.txt:233`). Both sides agree → no
mismatch.

On Vulkan the code goes to `download_release()`, which pulls the
upstream pre-built `llama-bXXXX-bin-ubuntu-vulkan-x64.tar.gz`. Upstream
binaries are compiled with the default `GGML_MAX_NAME=64`. So:

- `libstable-diffusion.a` (SD): `char name[128]`

- `libggml-base.so` (downloaded): `char name[64]`

- `libggml-vulkan.so` (downloaded): `char name[64]`

`sizeof(ggml_tensor)` diverges by 64 bytes between the two sides, and
every field after `name` (`extra`, `padding`) sits at a different offset
depending on which side allocated the struct.

### How this manifests as an `ggml_are_same_layout` failure

`ggml_are_same_layout()` (ggml/src/ggml-impl.h:73) itself only reads
`type`, `ne[4]`, and `nb[4]` — all fields that are **before** `name`,
so their offsets don't depend on `GGML_MAX_NAME`. The ABI mismatch
doesn't corrupt those fields directly.

The mechanism is indirect: SD code assumes `name[128]` and writes up to
128 bytes via `ggml_set_name()` / `strncpy(tensor->name, ..., 128)`.
When the tensor was allocated by the downloaded `libggml-base.so`, the
struct only has 64 bytes reserved for `name`, so the trailing 64 bytes
spill into `extra` (pointer) and `padding[8]`. Later calls inside
`libggml-base.so` or `libggml-vulkan.so` read `tensor->extra`
expecting an `ggml-backend.cpp` buffer handle or similar — and get
garbage. Downstream code dereferences that garbage, tensors look
malformed, and `ggml_are_same_layout` fires on what looks like a
size/stride mismatch.

The CPU-offload path is the trigger because that's where SD constructs a
host tensor, calls `ggml_backend_tensor_copy(host, device)`, and crosses
the ABI boundary.

## Why the existing fixes don't cover this path

Two defenses landed in `b378cba`:

1. **`_sync_ggml_abi()`** (`scripts/manage.py:1654`) — overlays
   llama.cpp's ggml source tree onto stable-diffusion.cpp so enum
   ordinals (`ggml_op`, `ggml_type`) match. This runs during SD's
   `build()` regardless of which llama.cpp strategy was used, so Vulkan
   benefits from it.

2. **`GGML_MAX_NAME=128` injection into `CMAKE_C_FLAGS`** — only runs
   inside `LlamaCppBuilder.build()` (static path, manage.py:1140) and
   `LlamaCppBuilder.build_shared()` (dynamic-from-source path,
   manage.py:1215).

`download_release()` doesn't compile anything, it just extracts a
pre-built tarball. There's nowhere for the define to apply. So Vulkan
dynamic wheels escape the fix entirely.

Root-level `CMakeLists.txt:16-18`:

```cmake
if(NOT SD_USE_VENDORED_GGML)
    add_definitions(-DGGML_MAX_NAME=128)
endif()
```

is ordered before `CMakeLists.txt:105` which reads the
`SD_USE_VENDORED_GGML` env var, so this `add_definitions` never fires
when the env var is the only source of `SD_USE_VENDORED_GGML=OFF`.
This is a separate-but-related ordering bug — not the proximal cause of
the Vulkan crash (the project-level Cython modules don't dereference
`ggml_tensor` fields), but worth fixing while touching this area.

## Proposed fix

Route Vulkan (and any future backend that gains an upstream release)
through `build_shared()` whenever SD is configured to share ggml:

```python
# scripts/manage.py, in do_build()
if args.dynamic and BuilderClass == LlamaCppBuilder:
    asset = builder._release_asset_name()
    # When SD shares llama.cpp's ggml, the shared libs must be built
    # with GGML_MAX_NAME=128 so ggml_tensor's layout matches what SD
    # was compiled with. Upstream pre-built releases use the default
    # GGML_MAX_NAME=64, so skip them and build from source to
    # propagate the define.
    if asset is None or _sd_uses_shared_ggml():
        if asset is not None:
            self.log.info(
                "SD_USE_VENDORED_GGML=0: building llama.cpp from "
                "source to propagate GGML_MAX_NAME=128 (skipping "
                "upstream pre-built release)"
            )
        builder.build_shared()
    else:
        builder.download_release()
```

### Tradeoff

Building llama.cpp from source adds CI time (Vulkan CI now has to build
llama.cpp in addition to shaderc). For a local Vulkan dynamic build on
this machine the llama.cpp build is ~3-4 minutes. On CI this cost is
paid once and cached via the `actions/cache@v5` step that already keys
on `scripts/manage.py` + `build-gpu-wheels.yml`.

### Alternative considered

Keeping `download_release()` and forcing `SD_USE_VENDORED_GGML=1` on
Vulkan reverts to the 0.2.9 behavior: SD statically embeds its own ggml
backend, giving a ~2x wheel size. Rejected because it contradicts the
entire Unreleased "GPU wheel size reduced ~50%" goal and reintroduces
the duplicate-ggml-backend runtime risk that motivated unification.

## Backend coverage after the fix

| Backend | `_release_asset_name()` | Dynamic path (SD_USE_VENDORED_GGML=0) |
|---------|-------------------------|---------------------------------------|
| CUDA    | `None`                  | `build_shared()` (unchanged)          |
| SYCL    | `None`                  | `build_shared()` (unchanged)          |
| HIP     | `None`                  | `build_shared()` (unchanged)          |
| Vulkan  | vulkan tarball URL      | `build_shared()` (was `download_release`) |
| Metal   | n/a (macOS, no Vulkan)  | `build_shared()` (unchanged)          |

The CPU dynamic path (no SD) is unaffected: `_sd_uses_shared_ggml()`
returns `False` when `SD_USE_VENDORED_GGML` is unset, so
`download_release()` is still used for plain CPU dynamic wheels.

## References

- `scripts/manage.py:905-914` — `_SD_GGML_MAX_NAME`, `_sd_uses_shared_ggml()`

- `scripts/manage.py:1140-1145`, `1215-1219` — `CMAKE_C_FLAGS` injection

- `scripts/manage.py:1300` — `_release_asset_name()`

- `scripts/manage.py:1346` — `download_release()`

- `scripts/manage.py:2252` — the fix site

- `CMakeLists.txt:16-18`, `:105-111` — root-level ordering bug (separate)

- `build/stable-diffusion.cpp/CMakeLists.txt:233` — SD's own `-DGGML_MAX_NAME=128`

- `build/llama.cpp/ggml/include/ggml.h:228` — llama.cpp's `#define GGML_MAX_NAME 64` default

- `CHANGELOG.md` — Unreleased "GPU wheel size reduced ~50%"

- `docs/dev/ggml-unification.md` — broader unification plan
