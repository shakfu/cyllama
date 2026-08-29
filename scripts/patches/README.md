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

### `llama.cpp-subprocess-glibc217.patch`

**Target:** `vendor/sheredom/subprocess.h` (llama.cpp `b10271`)

**Problem:** `subprocess_create_ex()` honours a working-directory argument via
`posix_spawn_file_actions_addchdir_np()`, which glibc only gained in 2.29.
cyllama builds its manylinux wheels on the manylinux2014 image (glibc 2.17,
see the `manylinux-*-image` rationale in `pyproject.toml`), where that call is
undeclared and the compile fails. It takes down two targets cyllama needs:
`llama-common` (via `common/subproc.cpp`) and `mtmd` when `MTMD_VIDEO` is on
(via `tools/mtmd/mtmd-helper.cpp`, which includes the vendored header
directly). This first appeared with the `b10271` bump, which added
`common/subproc.cpp`.

**Fix:** Report `ENOSYS` through the function's existing error path when the
libc is glibc older than 2.29, instead of emitting the call. There is no
posix_spawn equivalent on that glibc, so the cwd feature genuinely cannot be
supported there; nothing else in `subprocess_create_ex()` is affected. No
caller in llama.cpp passes a cwd today — `mtmd-helper.cpp` spawns ffmpeg
through plain `subprocess_create()`, so video decoding keeps working on
manylinux wheels.

**Alternative rejected:** configuring with `LLAMA_SUBPROCESS=OFF` compiles
`subproc.cpp` down to a stub and avoids the header entirely, but it also
forces `MTMD_VIDEO` off (llama.cpp's top-level `CMakeLists.txt` ties the two
together). cyllama binds video through `MtmdContext.open_video()`, so that
would have silently dropped a working API on every platform to fix one image.

**Note on the version test:** it is written as nested `#if` rather than
`defined(__GLIBC_PREREQ) && !__GLIBC_PREREQ(2, 29)`. The preprocessor
macro-expands the whole controlling expression before evaluating it, so where
`__GLIBC_PREREQ` is undefined the second operand degrades to `1 (2, 29)` — a
syntax error, not a false branch.

### `ggml-metal-pin-msl-version.patch`

**Target:** `ggml/src/ggml-metal/ggml-metal-device.m` — all three ggml trees
(llama.cpp, whisper.cpp, stable-diffusion.cpp each carry the same two compile
sites).

**Problem:** ggml compiles its embedded Metal shader library with a bare
`[MTLCompileOptions new]` and never sets `languageVersion`. Metal then derives
the MSL version from the SDK **the host process was linked against** — not from
the running OS. ggml is a library loaded into whichever binary the user runs, so
shader compilation depended on the interpreter rather than the machine.

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

**Fix:** Pin `languageVersion` on both compile paths via an `@available`
ladder — 3.2 on macOS 15+, 3.1 on macOS 14+, 3.0 on macOS 13+. Versions are
spelled numerically so the file still builds against SDKs predating the enum
constants, matching the existing `MTLGPUFamilyMetal4_GGML` precedent.

**Why the ladder stops below MSL 4.0:** 4.0 makes the Metal 4 tensor headers
available, which enables the tensor matmul kernels that blank stable-diffusion
output — the reason `proposed/llama.cpp-metal-tensor-msl4.patch` is not applied.
Capping at 3.2 keeps the tensor probe on its existing unsupported path, so this
patch makes today's behaviour deterministic rather than changing it.

**Verified** against the exact failing interpreter (the `actions/python-versions`
3.12.10 darwin-arm64 build, expanded with `pkgutil --expand-full`): before the
patch it reproduces the CI failure; after it, `ggml_metal_library_init` succeeds
with `has bfloat = true` and `has tensor = false`.

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

### `stable-diffusion.cpp-graph-cut-budget-clamp.patch`

**Target:** `src/core/ggml_extend.hpp` (stable-diffusion.cpp `master-816-487de75`)

**Problem:** `--max-vram` is a per-graph VRAM budget, but it is resolved once
per module at context-init time (`SDContext::max_graph_vram_bytes_for_module()`
-> `MaxVramAssignment::bytes_for_backend()`, which also memoises the result per
backend). With the auto budget (`"-1"`) that resolves to *free VRAM minus 1
GiB* on a machine where nothing else is loaded yet -- so on a single-GPU box
every module is told it may use the whole card.

Nothing reclaims that. `runner_done()` asks the model manager to release the
module's params, but `ModelManager::release_params_storage_blocks()` only frees
a block whose tensors are `ResidencyMode::Disk`; ordinary resident weights stay
in VRAM for the lifetime of the context. So the text encoder's weights are
still on the card when the diffusion model plans its graph against an init-time
budget that assumed an empty GPU. The planner duly merges the segments back
into one blob, and the allocation fails:

```
[ERROR] ggml_backend_cuda_buffer_type_alloc_buffer: allocating 5638.16 MiB on device 0: cudaMalloc failed: out of memory
[ERROR] model_manager.cpp:737  - model manager alloc params backend buffer failed, size = 5638.16MB
[ERROR] ggml_extend.hpp:1948 - z_image prepare graph weights failed
```

Reproduced with Z-Image Turbo Q6_K (5.5 GiB) + Qwen3-4B-Q8_0 (3.9 GiB) at
512x1024 on an 8 GiB RTX 4060: the text encoder loads 3974 MB, and the
diffusion model then asks for a single 5638 MB buffer out of the ~3.6 GiB that
is left. No `--max-vram` value avoids it, because the two modules need 8.5 GiB
between them and neither is ever evicted.

**Fix:** `GGMLRunner::resolve_graph_cut_plan()` already clamps the budget to
the VRAM that is actually free at plan time -- it just does it only when
`--stream-layers` is on. Drop that gate so the clamp applies whenever a budget
is set. The planner then cuts the diffusion graph into segments that fit what
is genuinely free, instead of one that fitted at init. A clamp can only lower
the budget, so an explicit `--max-vram N` still means "at most N GiB".

The budgeted plan cache is already keyed on the budget
(`PlanCache::budgeted_graph_cut_plan_max_vram_bytes`), so a budget that moves
between steps re-plans rather than reusing a stale plan; that path is what
`--stream-layers` has always exercised.

**Not fixable from the wrapper layer:** the budget is consumed inside the
runner, and the wrapper cannot observe -- let alone correct -- a plan that has
already been merged. What the wrapper *can* do, and now does, is expose
`params_backend` (see the CHANGELOG entry for `SDContextParams.backend` /
`.params_backend`), which sidesteps the budget entirely by keeping the text
encoder's weights in RAM.

### `stable-diffusion.cpp-conditioner-compute-failure.patch`

**Target:** `src/core/ggml_extend.hpp`, `src/conditioning/conditioner.hpp`,
`src/stable-diffusion.cpp` (stable-diffusion.cpp `master-816-487de75`)

**Problem:** `GGMLRunner::compute()` reports failure as an empty
`std::optional`, but `take_or_empty()` flattens that to an empty tensor and the
conditioners then `GGML_ASSERT(!hidden_states.empty())` on the result. A failed
text-encoder graph therefore kills the process:

```
conditioner.hpp:2058: GGML_ASSERT(!hidden_states.empty()) failed
```

Reproduced with `--max-vram 3.5` (any budget that forces the *text encoder*
itself through the graph-cut segmented path) on the Z-Image Turbo + Qwen3-4B
setup above. Worse, several of `compute()`'s failure paths log nothing at all,
so the abort arrives with no indication of what actually went wrong.

That silence is not hypothetical. The empty result in that repro turned out to
be cyllama's own `GGML_MAX_NAME` pin having gone stale (see the CHANGELOG entry
-- SD was compiled against a `struct ggml_tensor` 32 bytes longer than the one
ggml allocates, so the segmented path scribbled over the compute context's
object list). Finding it meant instrumenting eight separate `return
std::nullopt` sites by hand, precisely because the abort carried no cause. That
build-side bug is fixed and the segmented text encoder now runs, but the two
defects this patch addresses -- a failure that logs nothing, and an abort where
an error channel already exists -- are independent of it and remain.

An abort is the one failure mode a Python wrapper cannot contain: it takes the
interpreter with it, so no `RuntimeError`, no traceback, nothing an application
can retry or report.

**Fix:** two parts.

1. `take_or_empty()` logs `runner compute failed` before returning the empty
   tensor, so the cause is at least named at the point the error is dropped.
2. The LLM conditioner's `encode_prompt()` returns an empty tensor instead of
   asserting, `get_learned_condition()` propagates that as an empty
   `SDCondition` (which already has an `empty()` predicate), and
   `prepare_image_generation_embeds()` turns it into `std::nullopt` -- the
   error channel it already returns. Generation then fails cleanly and cyllama
   raises `RuntimeError` as it does for any other failed generation.

**Scope:** deliberately limited to the LLM/Qwen conditioner path (Z-Image,
Qwen-Image, MiniMax-H3) that reproduces. The same
`GGML_ASSERT(!hidden_states.empty())` pattern appears at 14 sites across the
older CLIP/T5 conditioners; converting all of them means threading a failure
result through every `get_learned_condition()` override, which is an upstream
refactor rather than a vendored patch. Part 1 covers those sites diagnostically.

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
