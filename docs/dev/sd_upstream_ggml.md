# Building stable-diffusion.cpp against upstream ggml

Status: deferred. Not for 0.4.10. Revisit in the next release, once
upstream's upstream-ggml mode has seen some real-world use.

## Problem

`SDCPP_VERSION` is pinned at `master-816-487de75` (`scripts/manage.py`).
From `master-817-bcc7e29` on, sd.cpp calls ggml APIs that only exist in
leejet's ggml fork. cyllama compiles SD against llama.cpp's ggml
(`_sync_ggml_abi`), so every later sd.cpp fails to compile.

## What changed upstream

Two PRs, both merged 2026-09-19:

| PR | First tag | Effect |
|-|-|-|
| [#1999](https://github.com/leejet/stable-diffusion.cpp/pull/1999) | `master-883-137f740` | Adds `SD_USE_UPSTREAM_GGML` and `SD_GGML_SOURCE_DIR` |
| [#2001](https://github.com/leejet/stable-diffusion.cpp/pull/2001) | `master-884-008ca5b` | Restores FP8 safetensors -> F16 conversion at load time in upstream mode |

Details of #1999:

- `SD_USE_UPSTREAM_GGML` (default `OFF`) only adds a `PUBLIC` compile
  definition. It does not fetch or replace any ggml source.
- `SD_GGML_SOURCE_DIR` (default `${CMAKE_CURRENT_SOURCE_DIR}/ggml`) selects the
  ggml tree that `add_subdirectory` builds.
- ggml setup moves to `cmake/ggml.cmake`. It fails the configure step unless
  the chosen tree contains `ggml-impl.h`.
- `#include "ggml/src/ggml-impl.h"` becomes `#include "ggml-impl.h"`. The
  private include dir now comes from the chosen tree.
- With `SD_USE_SYSTEM_GGML=ON`, `SD_GGML_SOURCE_DIR` must still point at the
  matching source tree, for private headers. Upstream's `docs/build.md` says
  the installed library must use the same `GGML_MAX_NAME`.

Features disabled in upstream mode:

- INT8 tensorwise/convrot. Those model files are rejected at load.
- Native FP8 storage. FP8 safetensors load as F16, which doubles RAM and VRAM
  for those tensors. FP8 GGUF files and FP8 `--type` requests are rejected.
- SageAttention (`ggml_sage_attn`, added in #2005).

cyllama at 816 has none of these, so nothing is lost relative to today. This
is inferred from upstream calling FP8 -> F16 the "legacy" behaviour. It is
not tested against 816.

## Verification done (2026-09-21, against `master-889-c678dfe`)

1. Fork-only symbols. Every `ggml_*` identifier called in `src/` was compared
   against `build/llama.cpp/ggml/include/*.h` and `src/ggml-impl.h`. Excluding
   SD's own `ggml_ext_*` helpers, 3 functions are missing:
   `ggml_mul_mat_i8_tensorwise`, `ggml_quantize_i8_convrot`,
   `ggml_sage_attn`. `GGML_TYPE_F8_*` is also missing. All 5 call sites sit
   inside `#ifndef SD_USE_UPSTREAM_GGML`:
   - `src/core/ggml_extend.cpp` (2 sites)
   - `src/pipeline/diffusion_engine.cpp`
   - `src/model/common/ggml_block.hpp` (2 sites)
   - `src/core/util.cpp`
2. `GGML_MAX_NAME` is still 160 (`CMakeLists.txt:326`). This matches
   `StableDiffusionCppBuilder.GGML_MAX_NAME`.
3. cyllama patches, checked with `git apply --check`:
   - `stable-diffusion.cpp-msvc-bigobj.patch`: applies.
   - `stable-diffusion.cpp-conditioner-compute-failure.patch`: fails.
   - `stable-diffusion.cpp-graph-cut-budget-clamp.patch`: fails.

   Both failures come from #1945 (`31ab2b2`), which split
   `src/core/ggml_extend.hpp` into `.h`/`.cpp`. Neither bug is fixed
   upstream. `GGML_ASSERT(!hidden_states.empty())` is still in
   `src/conditioning/conditioner.hpp`.

Nothing was built or run.

## Implementation plan

1. **Pin.** Set `SDCPP_VERSION` to `master-884-008ca5b` or later in both
   branches of `STABLE_BUILD`. Replace the "Ceiling" comment.
2. **Configure.** In `StableDiffusionCppBuilder.build()`, when
   `uses_shared_ggml()`:
   - pass `SD_USE_UPSTREAM_GGML=ON`;
   - pass `SD_GGML_SOURCE_DIR=<project.src>/llama.cpp/ggml`;
   - replace the `rmtree` + `copytree` in `_sync_ggml_abi()` with that
     option. Keep the step that drops SD's cmake build dir, because stale
     objects from the old ggml remain a hazard;
   - check that `_apply_source_patches()` still patches the ggml tree that
     actually gets compiled. After this change that is llama.cpp's tree, not
     `build/stable-diffusion.cpp/ggml`. `ggml-*.patch` must not be applied
     to it twice (the patcher is meant to be idempotent; confirm it).
   - update `do_check_vendor()`'s SD special case to match.
3. **Patches.** Rebase `conditioner-compute-failure` and
   `graph-cut-budget-clamp` onto the split `ggml_extend.h`/`.cpp`. Re-check
   the `LLMEmbedder` hunks against the current line numbers.
4. **Bindings.** `include/stable-diffusion.h` changed between 816 and 889:
   - `sd_type_t`: adds `SD_TYPE_Q2_0 = 42`, `SD_TYPE_F8_E4M3 = 43`,
     `SD_TYPE_F8_E5M2 = 44`. `SD_TYPE_COUNT` goes from 42 to 45.
   - New enum values: `SD_LOG_VERBOSE`, `LLADA_IMAGE_SCHEDULER`,
     `SENSENOVA_U1_FLOW_PRED`.
   - New exports: `sample_method_to_str[]`, `scheduler_to_str[]`,
     `sd_get_model_version_name()`.
   - Context params: `stream_layers` removed. Added `disable_prefetch`,
     `disable_segmented_compute`, `linear_scale`, `attn_scale`, `tokenizer`,
     `sage_attn`, `audio_encoder_path`.
   - `max_vram` semantics changed. It was a graph-cut budget (0 = off,
     -1 = auto). It is now a per-device budget, where 0 means use live free
     VRAM. Review the Python defaults and docs that pass it.
   - The video generation entry point gains `int* fps_out`.

   `tests/test_enum_drift.py` should catch the enum changes. Struct field
   changes need a manual diff against the `.pxd`.
5. **Guard.** Consider having the builder check that the cloned
   `cmake/ggml.cmake` exists. A pin moved back below 883 would then fail
   loudly, instead of silently building the fork ggml path.
6. **Test.** `make test` on CPU, then the Metal, CUDA and Vulkan wheels. The
   GPU wheels matter most: upstream warns that some operators may be missing
   or slower on upstream ggml.

## Open risks

- **Operator coverage.** Models added after 816 may use ops that llama.cpp's
  ggml implements only partly, or runs slowly on some backends. Upstream
  emits a warning but does not test this combination.
- **No pinned ggml.** Upstream mode means "compiles against upstream ggml",
  not "tested against ggml X". A llama.cpp bump can break SD without any
  change on the SD side. Keep one SD test in the suite that exercises the
  shared-ggml path on each llama.cpp bump.
- **Maintenance.** Upstream keeps the fork ggml as the default, so new
  features will keep landing behind the fork. Each SD bump needs the
  symbol check from "Verification done" repeated.

## Relation to `one-ggml-refactor.md`

`one-ggml-refactor.md` proposes building ggml once and consuming it through
each project's `*_USE_SYSTEM_GGML`. Before #1999 that could not work for SD
past 816, because of the fork-only calls. #1999 removes that blocker.
Under `SD_USE_SYSTEM_GGML=ON`, SD also needs
`SD_GGML_SOURCE_DIR` for `ggml-impl.h`, which the llama.cpp ggml tree
provides.

Recommended order: do this bump first, using `SD_GGML_SOURCE_DIR`. Do the
one-ggml refactor as a separate change. Combining them mixes a large API
update with a build-system change, which makes failures harder to bisect.

## Re-running the checks

```sh
git clone --filter=blob:none https://github.com/leejet/stable-diffusion.cpp.git sd
cd sd && git checkout <tag>
L=<cyllama>/build/llama.cpp/ggml
grep -rhoE '\bggml_[a-z0-9_]+\s*\(' src | sed 's/\s*($//' | sort -u > used.txt
cat $L/include/*.h $L/src/ggml-impl.h | grep -oE '\bggml_[a-z0-9_]+\b' | sort -u > have.txt
comm -23 used.txt have.txt | grep -v '^ggml_ext_'   # then confirm each is #ifndef-guarded
grep -n GGML_MAX_NAME CMakeLists.txt
for p in <cyllama>/scripts/patches/stable-diffusion.cpp-*.patch; do git apply --check "$p"; done
git diff master-816-487de75 <tag> -- include/stable-diffusion.h
```
