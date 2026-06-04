# Patches

Fixes for vendored C++ dependencies, with their upstream rationale. Where the
issue can be handled from the Cython wrapper layer, cyllama does that and the
patch here is only a proposed upstream change. Where it cannot — e.g. a hard
`GGML_ABORT`/`abort()` that no Python-level code can intercept — the fix is
applied to the cloned source at build time by `LlamaCppBuilder._apply_source_patches()`
in `scripts/manage.py` (idempotent, guarded to become a no-op once upstream merges).

## llama.cpp

**Target:** `b9493`-`b9498` (`tools/mtmd/clip.cpp`); still required as of b9498.

**Status:** applied at build time via `manage.py` (`_apply_source_patches`).

**Problem:** `clip_n_mmproj_embd()` lost its `PROJECTOR_TYPE_GEMMA4A` case when
the gemma4 "ultra" audio variant (`GEMMA4UA`) was added. `GEMMA4A` is still
handled in every other function in the file, so this is an unintentional
omission. Loading a gemma4a mmproj (e.g. `mmproj-gemma-4-E4B-it-BF16.gguf`) and
warming up the audio encoder falls through to `default: GGML_ABORT("Unknown
projector type")`, which calls `abort()` and takes down the whole process
(SIGABRT). It cannot be caught from the Cython layer, hence the source patch.

**Fix:** group `PROJECTOR_TYPE_GEMMA4A` with `PROJECTOR_TYPE_GEMMA4UA` in
`clip_n_mmproj_embd()` (both return `ctx->model.hparams.projection_dim`).

**Upstream status:** unfixed on `master` as of 2026-06-04; not merged in any
tag. The original symptom was filed as ggml-org/llama.cpp#21325, which was
closed without addressing this `clip_n_mmproj_embd` omission. A ready-to-post
issue draft (with this fix) lives at `docs/dev/patch-mmproj-gemma4a.md`.

## stable-diffusion.cpp

**Target:** commit `545fac4` (tag `master-537-545fac4`)

**Upstream issue:** https://github.com/leejet/stable-diffusion.cpp/issues/1367

**Problem:** `alloc_params_buffer()` in `GGMLRunner` (ggml_extend.hpp) returns `bool`, but all wrapper classes in `DiffusionModel`, `Conditioner`, `T5Embedder`, and `LLM` declare their overrides as `void`, discarding the return value. The call sites in `stable-diffusion.cpp` also never check the result. When allocation fails (e.g. CUDA out of memory), execution silently continues with unallocated tensors, producing garbage output.

**Current cyllama workaround:** The Cython wrapper (`stable_diffusion.pyx`) validates each generated `SDImage.is_valid` and raises `RuntimeError` when all images have invalid data.
