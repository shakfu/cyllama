# Upstream issue draft: gemma4a audio mmproj aborts in `clip_n_mmproj_embd`

This is a ready-to-post issue for the llama.cpp tracker
(https://github.com/ggml-org/llama.cpp/issues). It documents the
`PROJECTOR_TYPE_GEMMA4A` omission that cyllama currently works around with a
build-time source patch (`scripts/patches/llama.cpp-gemma4a-clip_n_mmproj_embd.patch`).

Related: the original symptom was filed as
[#21325](https://github.com/ggml-org/llama.cpp/issues/21325) ("Eval bug: Gemma
4 audio support is missing"), which was closed on 2026-04-12 without a linked
PR. The specific `clip_n_mmproj_embd` omission below is still present on
`master` as of 2026-06-04, so a fresh issue (referencing #21325) is warranted.

Everything below the line is the issue body. Trim the "Verified against" line to
whatever build you reproduce on before posting.

---

## Eval bug: `clip_n_mmproj_embd()` aborts for `gemma4a` audio projector (missing switch case)

### Summary

Loading a Gemma-4 multimodal projector that uses the audio projector type
`gemma4a` (`clip.audio.projector_type = gemma4a`) aborts the process with
`GGML_ABORT("Unknown projector type")`. `clip_n_mmproj_embd()` in
`tools/mtmd/clip.cpp` has cases for every other current projector type --
including the sibling audio variant `GEMMA4UA` -- but is missing the
`PROJECTOR_TYPE_GEMMA4A` case, so a `gemma4a` mmproj falls through to the
`default:` arm and calls `abort()`. The abort is uncatchable from any
higher-level binding (it is a hard `abort()`, not a return code), so it takes
down the whole host process.

### Name and Version

Reproduced on `master` (commit checked 2026-06-04) and on release `b9498`.
The omission is also present in `b9493`.

```
$ ./llama-cli --version
version: b9498 (or later master)
built with Apple clang for arm64-apple-darwin
```

### Operating systems

Platform-independent (it is a logic bug in a `switch`). Observed on macOS
(Metal) but the affected code path is not backend-specific.

### Which llama.cpp modules do you know to be affected?

libmtmd / multimodal (`tools/mtmd/clip.cpp`).

### Steps to Reproduce

1. Obtain a Gemma-4 model and its audio-capable mmproj, e.g.
   `mmproj-gemma-4-E4B-it-BF16.gguf` (the mmproj reports
   `clip.audio.projector_type = gemma4a`).
2. Initialize an mtmd context against that mmproj, e.g. via `llama-mtmd-cli`
   with the audio mmproj, or programmatically through libmtmd. The abort fires
   during clip-context construction (the `clip_ctx` constructor evaluates
   `n_mmproj_embd(clip_n_mmproj_embd(ctx))`), before any inference runs.

### Expected behavior

`clip_n_mmproj_embd()` returns the projector's output embedding dimension for
`gemma4a`, exactly as it already does for the sibling audio projector
`gemma4ua`, and the context loads successfully.

### Actual behavior

```
GGML_ABORT("Unknown projector type")  -> abort() / SIGABRT
```

### Root cause

`PROJECTOR_TYPE_GEMMA4A` is a fully supported projector type and is handled in
every other projector `switch` in `clip.cpp` on `master`:

- `clip_image_build_graph` dispatch -- builds `clip_graph_gemma4a` (~line 972)
- graph input setup (~line 1590)
- `load_tensors` (~line 2475, loads the SSCP conv stack)
- `n_patches` (~line 3380)
- the projector-output builder (~line 4128)

It is missing from exactly one function: `clip_n_mmproj_embd()` (`master`
~line 4312). The relevant arm currently reads:

```cpp
        case PROJECTOR_TYPE_LFM2A:
            return ctx->model.position_embeddings->ne[0];
        case PROJECTOR_TYPE_GEMMA4UA:                 // <- GEMMA4A is not grouped here
            return ctx->model.hparams.projection_dim;
        case PROJECTOR_TYPE_GRANITE_SPEECH:
            return ctx->model.qf_proj_linear_w->ne[1];
        ...
        default:
            GGML_ABORT("Unknown projector type");
```

Because `GEMMA4A` is absent, it hits `default`. Given it is handled everywhere
else, this looks like an unintentional omission introduced when the `GEMMA4UA`
("ultra" audio) variant was added.

### Proposed fix

Group `PROJECTOR_TYPE_GEMMA4A` with `PROJECTOR_TYPE_GEMMA4UA`, returning
`hparams.projection_dim`:

```diff
@@ int clip_n_mmproj_embd(const struct clip_ctx * ctx) {
         case PROJECTOR_TYPE_LFM2A:
             return ctx->model.position_embeddings->ne[0];
+        case PROJECTOR_TYPE_GEMMA4A:
         case PROJECTOR_TYPE_GEMMA4UA:
             return ctx->model.hparams.projection_dim;
         case PROJECTOR_TYPE_GRANITE_SPEECH:
             return ctx->model.qf_proj_linear_w->ne[1];
```

This makes the audio embeddings inject into the text model. For the Gemma-4 E4B
mmproj, `clip.audio.projection_dim` equals the text model's embedding length
(2560), which is the value `clip_n_mmproj_embd` must return.

### Note for maintainers (one thing worth confirming)

`gemma4a` and `gemma4ua` are distinct projectors with different graph builders
(`clip_graph_gemma4a` vs `clip_graph_gemma4ua`) and different tensors
(`gemma4a` loads the SSCP conv stack; `gemma4ua` loads `mm_input_proj`), so
grouping them is an assumption that both project to `hparams.projection_dim`. In
testing against the E4B mmproj this is correct, but please confirm the intended
output dimension for `gemma4a` rather than assuming parity with `gemma4ua`. If
they differ, the case should return the `gemma4a`-specific dimension instead of
being grouped.
