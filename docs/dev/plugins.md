# cyllama Plugins Architecture

Status: proposal / design recommendation Context: GitHub issue [#16](https://github.com/shakfu/cyllama/issues/16) — request to integrate [acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp) (music generation) and [omnivoice.cpp](https://github.com/ServeurpersoCom/omnivoice.cpp) (multilingual TTS + voice cloning), both by @ServeurpersoCom.

This document records the recommendation for how to integrate self-contained `.cpp` model libraries into the cyllama family as **plugins** rather than baking them into the core wheel.

## TL;DR

- A plugin model is **feasible and recommended** for self-contained `.cpp` libs that vendor their own ggml (omnivoice, acestep). It is *not* suitable for libs that extend llama.cpp internals and share core's ggml ABI.

- Ship **one repo (`cyllama-plugins`), one wheel per plugin** (`cyllama-omnivoice`, `cyllama-acestep`, …) — not a single fat bundle.

- Surface plugins **both** standalone (`import cyllama_omnivoice`) **and** under the `cyllama.` namespace via entry-point discovery.

- Plugins stay **independent** of core; core integration (e.g. whisper → TTS) lives in examples, not as a hard dependency.

- **CPU-first**; add GPU wheels per plugin on demand.

- **Prototype in-tree first** (behind `WITH_OMNIVOICE=OFF`), prove real synthesis + live ggml coexistence, *then* extract into the separate plugin repo.

## Why this works: the Phase 0 isolation result

The enabling fact is that these libraries are fully self-contained at the ABI level. Building omnivoice.cpp with the recipe below produces a `libomnivoice.so` that:

- has **no external `libggml` dependency** (ggml is statically baked in; only `libgomp`/`libstdc++`/`libm`/`libgcc_s`/`libc` remain as `NEEDED`),

- exports **0 `ggml_*` and 0 `gguf_*` symbols**, and exactly the **13 `ov_*`** public ABI functions,

- in omnivoice's case, has **no dependency on llama.cpp at all**.

Because a plugin shares nothing with core at the ABI level, it can be built, versioned, and shipped as a completely independent wheel. That is the entire precondition for a plugin system, and it was verified empirically (symbol + `ldd`/`readelf` inspection, and an ABI consumer smoke test).

### The proven build recipe

```
-DBUILD_SHARED_LIBS=OFF                       # ggml -> static, baked into the .so
-DOMNIVOICE_SHARED=ON                         # omnivoice itself -> shared (only ov_* visible)
-DGGML_BACKEND_DL=OFF                          # backends linked in, not dlopen'd
-DCMAKE_C_VISIBILITY_PRESET=hidden
-DCMAKE_CXX_VISIBILITY_PRESET=hidden
-DCMAKE_VISIBILITY_INLINES_HIDDEN=ON
-DCMAKE_POSITION_INDEPENDENT_CODE=ON
-DCMAKE_SHARED_LINKER_FLAGS=-Wl,--exclude-libs,ALL
```

> **Critical correction discovered during Phase 0:** `OMNIVOICE_SHARED=ON` > *without* `BUILD_SHARED_LIBS=OFF` looks clean on a `nm` symbol check but > secretly carries `NEEDED libggml.so.0` — which would collide with llama.cpp's > identically-named `libggml.so.0` at load time (the dynamic linker resolves the > SONAME to a single file → crash). Forcing ggml **static** is mandatory, not > optional. Always verify with `ldd`/`readelf -d`, not `nm` alone.

## Scope: where the plugin model applies

| Library shape | Plugin-friendly? | Why |
|---|---|---|
| Self-contained, vendors own ggml (omnivoice, acestep) | yes | Sealed `.so`, no ABI lockstep with core |
| Extends llama.cpp internals, shares core's ggml ABI | no | Needs version lockstep with core; defeats independent packaging |

"Plugins" here means *a family of self-contained native model wrappers*, not an arbitrary extension point into core internals.

## Recommended architecture

### Repo layout — one repo, one wheel per plugin

`cyllama-plugins` is the **repo/umbrella**, shipping **a wheel per plugin**, not one fat wheel. A single bundled wheel would force every user to download every model's native lib + its own ggml copy + every GPU variant just to get one modality — throwing away the main benefit (small, à la carte installs).

```
cyllama-plugins/                 # the repo
├─ shared/                       # the Phase-0 Builder recipe + CMake helpers, factored once
├─ plugins/
│  ├─ omnivoice/                 # -> wheel: cyllama-omnivoice   (multilingual TTS + voice clone)
│  │   ├─ pyproject.toml         #    scikit-build-core, name="cyllama-omnivoice"
│  │   ├─ CMakeLists.txt         #    builds self-contained libomnivoice.so + Cython ext
│  │   └─ src/cyllama_omnivoice/
│  └─ acestep/                   # -> wheel: cyllama-acestep     (music gen; later)
└─ (optional) cyllama-plugins meta-wheel -> just deps on the individual plugin wheels
```

Each plugin is its own scikit-build-core project reusing the proven recipe above.

### Import / CLI surface — both, via entry-point discovery

Each plugin is importable standalone **and** lazily surfaced under `cyllama.`:

```python
# 1. Standalone — omnivoice needs nothing from core
from cyllama_omnivoice import OmniVoice
tts = OmniVoice(model="omnivoice-base-Q8_0.gguf", codec="omnivoice-tokenizer-F32.gguf")
tts.save("hello.wav", "Hello world", lang="en")

# 2. Unified namespace — core lazily exposes installed plugins
import cyllama
cyllama.omnivoice.OmniVoice(...)        # works if the plugin wheel is present,
                                        # else: "pip install cyllama-omnivoice"
```

Discovery is via entry points, **not** namespace packages (which are fragile when mixing a regular core package with separately-built contributions). Each plugin declares:

```toml
[project.entry-points."cyllama.plugins"]
omnivoice = "cyllama_omnivoice:register"
```

Core grows a tiny loader (~30 lines, zero per-plugin coupling): `cyllama/__init__.py` gets a `__getattr__` that resolves `cyllama.<name>` against the `cyllama.plugins` entry-point group, and the `cyllama` CLI auto-registers a subcommand per discovered plugin (`cyllama omnivoice -m … --codec … -p "Hi"`).

### Relationship to core — independent, optional one-way integration

Plugins depend on **nothing** in core (that is what keeps the isolation clean — don't give it up). Valuable glue such as whisper transcript → omnivoice TTS lives in **examples/recipes**, not as a hard dependency. A plugin may optionally `import cyllama` if present, but must work without it.

## The one real cost: GPU wheel matrix

| | Plugin repo (per-plugin wheels) | In-core `WITH_*` flag |
|---|---|---|
| Core wheel size | unchanged | grows per model |
| Release cadence | **independent** (upstreams move fast: 528 / 73 commits) | locked to core releases |
| Per-plugin backend gaps (e.g. no SYCL) | isolated — don't block core | complicate the `cyllama-sycl` wheel |
| GPU packaging | **each plugin needs its own cpu/cuda/rocm/vulkan wheels** → matrix multiplies | one shared matrix |
| Per-process cost | each plugin carries its own ggml copy (~MBs RAM/disk) | one shared ggml |

The cadence + backend-gap decoupling is why the plugin repo wins for these fast-moving, partial-backend upstreams. The cost is the GPU matrix — mitigated by **starting CPU-only** and adding GPU wheels per plugin on demand.

## Recommended first step

**Prototype omnivoice in-tree behind `WITH_OMNIVOICE=OFF`, drive it to a real 24 kHz WAV, and confirm `import cyllama` + omnivoice coexist live.** That single milestone validates everything the plugin architecture rests on. (Phase 0 only proved static linkage; it never loaded both ggml copies into one process at once — the in-tree synthesis test does.) If it holds, extracting to `cyllama-plugins` is mechanical.

In this repo:

1. `OmniVoiceCppBuilder` in `scripts/manage.py` encoding the proven recipe; copy `src/omnivoice.h` + `libomnivoice.so` into `thirdparty/omnivoice.cpp/{include,lib}`.

2. `src/cyllama/omnivoice/omnivoice.{pxd,pyx}` wrapping the 13 `ov_*` functions, plus a small `OmniVoice` class returning NumPy samples / writing a WAV.

3. `WITH_OMNIVOICE` option in `CMakeLists.txt` (default OFF), dynamic extension linking only `libomnivoice.so` (mirrors the dynamic `whisper_cpp` target).

4. Smoke test: download a Q8_0 base + F32 tokenizer from [`Serveurperso/OmniVoice-GGUF`](https://huggingface.co/Serveurperso/OmniVoice-GGUF), synthesize "Hello world", assert a non-empty 24 kHz mono WAV — in a process that has also imported `cyllama`.

Once the pilot generates a WAV and coexists with core in one process, acestep and future libs are repetition. acestep is a bigger lift and should follow omnivoice: it currently exposes **no public library API** (only executables + an internal `acestep-core` static lib with no install rules), so wrapping it needs either an upstream `libacestep` + public header, or a subprocess/HTTP-server integration style cyllama does not currently use. Its SYCL support is also still experimental (acestep.cpp#54).

## Public ABI reference (omnivoice.h, OV_ABI_VERSION = 3)

The 13 exported functions wrapped by the plugin:

- `ov_version`, `ov_last_error`

- `ov_init_default_params`, `ov_init`, `ov_free`

- `ov_tts_default_params`, `ov_synthesize`

- `ov_audio_free`

- `ov_duration_sec_to_tokens`, `ov_num_codebooks`

- `ov_extract_voice_ref`, `ov_voice_ref_free`

- `ov_log_set`

Output is mono float PCM at 24 kHz (`struct ov_audio`). Voice cloning is supported via either pre-encoded RVQ tokens or raw 24 kHz reference audio (`ov_tts_params.ref_audio_tokens` / `ref_audio_24k`).
