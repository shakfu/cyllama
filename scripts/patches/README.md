# Patches

Fixes for vendored C++ dependencies, with their upstream rationale. Where the
issue can be handled from the Cython wrapper layer, cyllama does that and the
patch here is only a proposed upstream change. Where it cannot — e.g. a hard
`GGML_ABORT`/`abort()` that no Python-level code can intercept — the fix is
applied to the cloned source at build time by `LlamaCppBuilder._apply_source_patches()`
in `scripts/manage.py` (idempotent, guarded to become a no-op once upstream merges).

There are currently no build-time llama.cpp patches. The gemma4a
`clip_n_mmproj_embd()` abort fix was merged upstream in
[ggml-org/llama.cpp#24091](https://github.com/ggml-org/llama.cpp/pull/24091)
(released in `b9503`), so cyllama no longer carries it. `_apply_source_patches()`
remains in place for any future patch.

## stable-diffusion.cpp

**Target:** commit `545fac4` (tag `master-537-545fac4`)

**Upstream issue:** https://github.com/leejet/stable-diffusion.cpp/issues/1367

**Problem:** `alloc_params_buffer()` in `GGMLRunner` (ggml_extend.hpp) returns `bool`, but all wrapper classes in `DiffusionModel`, `Conditioner`, `T5Embedder`, and `LLM` declare their overrides as `void`, discarding the return value. The call sites in `stable-diffusion.cpp` also never check the result. When allocation fails (e.g. CUDA out of memory), execution silently continues with unallocated tensors, producing garbage output.

**Current cyllama workaround:** The Cython wrapper (`stable_diffusion.pyx`) validates each generated `SDImage.is_valid` and raises `RuntimeError` when all images have invalid data.
