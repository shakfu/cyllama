# Patches

Fixes for vendored C++ dependencies, with their upstream rationale. Where the
issue can be handled from the Cython wrapper layer, cyllama does that and the
patch here is only a proposed upstream change. Where it cannot — e.g. a hard
`GGML_ABORT`/`abort()` that no Python-level code can intercept — the fix is
applied to the cloned source at build time by `GgmlBuilder._apply_source_patches()`
in `scripts/manage.py` (idempotent, guarded to become a no-op once upstream merges).

Naming decides where a patch lands. `ggml-*.patch` files are applied to every
ggml-backed tree (llama.cpp, whisper.cpp, stable-diffusion.cpp), since each
vendors its own ggml copy and shares its bugs. `<project>-*.patch` files
(e.g. `llama.cpp-*.patch`) are applied only to that project's tree.

`proposed/` holds patches that are **not** applied — the glob is non-recursive,
so anything in that subdirectory is inert. Use it for changes that are correct
in isolation but not safe to ship yet; keep the analysis alongside them here.

Note that stable-diffusion.cpp and whisper.cpp pull ggml in as a git
*submodule*, while llama.cpp vendors it in-tree. `git apply` writes through
the submodule boundary either way, but `git checkout <path>` from the
superproject does not — to undo a patch by hand there, run git from inside
`build/<project>/ggml`, not from `build/<project>`.

## Applied

### `ggml-metal-pin-msl-version-set-lang.patch`

**Target:** `ggml/src/ggml-metal/ggml-metal-device.m` -- all three ggml trees:
llama.cpp `v0.4.0`+, whisper.cpp `v1.9.4`+, and stable-diffusion.cpp's
vendored copy. By default SD compiles llama.cpp's ggml via
`SD_GGML_SOURCE_DIR`, so its own copy is unused.

**Problem:** ggml compiles its embedded Metal shader library without setting
`languageVersion` except on the tensor branch of
`ggml_metal_compile_options_set_lang()`. Metal then derives the MSL version
from the SDK **the host process was linked against** -- not from the running
OS. ggml is a library loaded into whichever binary the user runs, so shader
compilation depended on the interpreter rather than the machine.

python.org's CPython 3.12 links the macOS **12.1** SDK. Under it the library
fails to compile outright:

```
ggml_metal_library_init: error: MTLLibraryErrorDomain Code=3
  program_source: error: no matching constructor for initialization of
  'threadgroup metal::half4x4[512]'
ggml_metal_device_init: error: failed to create library
```

The Metal backend never initializes, and the first `llama_context` creation
fails. Model loading succeeds first, so the traceback points at the context, not
at Metal. Below MSL 3.1 there is a quieter variant: `ggml-metal.metal:36`
`#undef`s the bf16 kernels while `props.has_bfloat` stays true.

This is not CI-only. `cp312-abi3` wheels are what 3.12 users install, and the
python.org installer is a mainstream way to get Python on macOS.

**Fix:** Pin `languageVersion` in `ggml_metal_compile_options_set_lang()` via
an `@available` ladder -- 3.2 on macOS 15+, 3.1 on macOS 14+, 3.0 on macOS 13+.
Versions are spelled numerically so the file still builds against SDKs
predating the enum constants, matching the existing `MTLGPUFamilyMetal4_GGML`
precedent.

**Why the ladder stops below MSL 4.0:** 4.0 makes the Metal 4 tensor headers
available, which enables the tensor matmul kernels that blank stable-diffusion
output -- the reason `proposed/llama.cpp-metal-tensor-msl4.patch` is not
applied. Upstream enables the tensor API on M5/M6/A19/A20 and asks for MSL 4.0
to get its headers, so the cap overrides an upstream default. Capping at 3.2
makes the tensor probe's dummy-kernel compile fail, so `ggml_metal_device_init`
clears `props.has_tensor` and the backend keeps the simdgroup kernels.
`GGML_METAL_TENSOR_ENABLE=1` cannot override it. Untested on Metal 4 hardware.

**Verified** against the exact failing interpreter (the `actions/python-versions`
3.12.10 darwin-arm64 build, expanded with `pkgutil --expand-full`): before the
fix it reproduces the CI failure; after it, `ggml_metal_library_init` succeeds
with `has bfloat = true` and `has tensor = false`. This was measured with the
predecessor patch, which set the same ladder at the two bare
`[MTLCompileOptions new]` sites that pre-`v0.4.0` ggml had.

**History:** that predecessor, `ggml-metal-pin-msl-version.patch`, stopped
applying when llama.cpp `v0.4.0` introduced `ggml_metal_compile_options_set_lang()`.
A skipped patch is silent, so this surfaced as nothing at all. It was removed
once whisper.cpp `v1.9.4`, the last tree on the old shape, adopted the new one.

### `stable-diffusion.cpp-msvc-bigobj.patch`

**Target:** `CMakeLists.txt` (stable-diffusion.cpp `master-812-ea7f0c8`)

**Problem:** `src/stable-diffusion.cpp` has grown past the 65,279-section
limit of the COFF object format, so the Windows wheel build dies with
`error C1128: number of sections exceeded object file format limit: compile
with /bigobj`. llama.cpp hit the same wall long ago and carries `/bigobj` in
its own top-level `CMakeLists.txt`; stable-diffusion.cpp's `if (MSVC)` block
sets `/MP` and `/utf-8` but not `/bigobj`.

**Fix:** Add `/bigobj` alongside `/utf-8`, using the same
`COMPILE_LANGUAGE` generator expressions so the scoping matches its
neighbours. Upstreamable as-is.

**Why a patch and not a CMake cache variable:** passing
`-DCMAKE_CXX_FLAGS=/bigobj` from `manage.py` would *replace* the value CMake
initialises for MSVC (`/DWIN32 /D_WINDOWS /W3 /GR /EHsc`) rather than append
to it, silently dropping `/EHsc` and changing unwind semantics for the whole
tree. `add_compile_options()` is purely additive.

Two stable-diffusion.cpp patches were dropped at `master-898-2bb7294`, both
fixed upstream:

- `graph-cut-budget-clamp`: the budgeted plan it clamped is gone. Segmentation
  is now decided per run against live free VRAM
  ([#1905](https://github.com/leejet/stable-diffusion.cpp/pull/1905),
  [#1940](https://github.com/leejet/stable-diffusion.cpp/pull/1940)).
- `conditioner-compute-failure`: the LLM conditioner logs and returns an empty
  condition instead of asserting, and the pipeline turns it into a failed
  generation ([#1958](https://github.com/leejet/stable-diffusion.cpp/pull/1958),
  [#1973](https://github.com/leejet/stable-diffusion.cpp/pull/1973),
  [#2020](https://github.com/leejet/stable-diffusion.cpp/pull/2020)).

An earlier llama.cpp patch is no longer carried: the gemma4a
`clip_n_mmproj_embd()` abort fix was merged upstream in
[ggml-org/llama.cpp#24091](https://github.com/ggml-org/llama.cpp/pull/24091)
(released in `b9503`).

## Not applied

### `proposed/llama.cpp-metal-tensor-msl4.patch`

**Target:** `ggml/src/ggml-metal/ggml-metal-device.m` (llama.cpp `c0bc859`, ggml 0.17.0)

**Status:** correct in isolation, **not applied** — it enables ggml's Metal
tensor kernels, which produce blank images in stable-diffusion.cpp. See
"Why this is not applied" below.

**Problem:** On M5/A19-class hardware ggml probes for Metal 4 tensor API support
by compiling a small kernel at runtime, then compiles its main shader library
from embedded source (`GGML_METAL_EMBED_LIBRARY=ON`, the cyllama default).
Neither call sets `MTLCompileOptions.languageVersion`, so Metal picks a default
derived from the SDK that the *host process* was linked against — not from the
running OS. Loaded into a Python interpreter built against an older SDK (uv's
CPython 3.13 links the macOS 15.5 SDK), the default falls below MSL 4.0 and the
tensor headers are unavailable, so the probe fails with `use of undeclared
identifier 'mpp'`, ggml logs `the tensor API is not supported in this
environment - disabling`, and falls back to simdgroup kernels. Measured on an
M5 Pro / macOS 26.5 with Z-Image Turbo at 512x1024: 13.99 s/it disabled vs
7.78 s/it enabled, a ~1.8x loss.

**Fix:** Request MSL 4.0 explicitly on both compile paths when the device
reports the Metal 4 GPU family. `MTLLanguageVersion4_0` is spelled as a local
constant, matching ggml's existing `MTLGPUFamilyMetal4_GGML` workaround for
building against SDKs that predate the symbol.

**Not fixable from the wrapper layer:** the language version is chosen inside
ggml before any cyllama code runs.

**Why this is not applied.** The language version bump is numerically inert on
its own: with the tensor path off, forcing MSL 4.0 on every compile produces
bit-identical output (verified by SHA over generated PNGs). What it does is
*enable* ggml's tensor matmul kernels — and those break stable-diffusion.cpp.
Measured on M5 Pro / macOS 26.5:

- llama.cpp, ggml 0.17.0: greedy output bit-identical with the tensor path on
  and off (Llama-3.2-1B-Q8_0, Qwen3-4B-Q8_0). Correct, and faster.
- stable-diffusion.cpp: blank white image whenever the tensor path is on.
  Reproduced on **both** SD's old vendored ggml 0.15.3 and the shared ggml
  0.17.0, at 512x512 and 512x1024, with and without `--diffusion-fa` /
  `--vae-on-cpu`, cfg 1.0 and 7.0. Tensor off renders correctly every time.
  The speedup is real (13.99 -> 7.78 s/it) and the output is worthless.

So this is not a stale-ggml problem, as first suspected — ggml's tensor
kernels are simply wrong for something in SD's graph, in a way llama.cpp's
text path never hits. Worth reporting upstream with the repro above.

There is no per-extension escape hatch: since SD shares llama.cpp's ggml, both
extensions link the same `libggml-metal.a`, and ggml's Metal device (and its
`has_tensor` decision) is initialised once per process. Tensor kernels are
therefore all-or-nothing for a process that loads both. Correct-and-slower
wins, so the patch stays in `proposed/`.

**Failure mode to watch for:** nothing is logged. The image is simply blank —
an ~8K PNG where a real 512x512 render is ~300K. A build can look completely
healthy and still be producing white squares.

To re-evaluate after a ggml bump: apply the patch, rebuild, and render with a
fixed seed. `GGML_METAL_TENSOR_DISABLE=1` toggles the tensor path at runtime
without rebuilding, which makes the A/B cheap.

## stable-diffusion.cpp

**Target:** commit `545fac4` (tag `master-537-545fac4`)

**Upstream issue:** https://github.com/leejet/stable-diffusion.cpp/issues/1367

**Problem:** `alloc_params_buffer()` in `GGMLRunner` (ggml_extend.hpp) returns `bool`, but all wrapper classes in `DiffusionModel`, `Conditioner`, `T5Embedder`, and `LLM` declare their overrides as `void`, discarding the return value. The call sites in `stable-diffusion.cpp` also never check the result. When allocation fails (e.g. CUDA out of memory), execution silently continues with unallocated tensors, producing garbage output.

**Current cyllama workaround:** The Cython wrapper (`stable_diffusion.pyx`) validates each generated `SDImage.is_valid` and raises `RuntimeError` when all images have invalid data.
