# Cython → nanobind: analysis and advice

This document analyses whether to migrate cyllama's bindings from Cython to
[nanobind](https://github.com/wjakko/nanobind), and the related question of
whether to consolidate the three extension modules into a single statically
linked binary.

## Current state

cyllama wraps three upstreams via Cython:

| Module                     | `.pyx` LOC | `.pxd` LOC | `cdef class`es |
| -------------------------- | ---------- | ---------- | -------------- |
| `cyllama.llama` (+ pxi)    | ~5,600     | ~2,500     | 25             |
| `cyllama.sd`               | 3,168      | 433        | 14             |
| `cyllama.whisper`          | 990        | 385        | 10             |
| `cyllama.llama.server`     | 594        | 107        | (mongoose)     |
| **Total**                  | **~12.4K** | **~3.4K**  | **~50**        |

All three upstreams (llama.cpp, whisper.cpp, stable-diffusion.cpp) are pinned
to the **same ggml commit** and currently link dynamically against a single
`libggml.so`. ggml's global state (backend registry, log callback, Vulkan/Metal
device tables) therefore lives exactly once in the process today.

## Two independent questions

The two ideas often discussed together are actually orthogonal:

1. **Binding-tool choice** — Cython vs nanobind.
2. **Binary layout** — three extensions + shared `libggml.so` vs one
   extension with statically linked ggml.

Each can be done without the other.

---

## Question 1: Cython → nanobind

### Pros

- **Delete the `.pxd` files (~3.4K LOC).** nanobind reads upstream C/C++
  headers directly. No more re-declaring llama.cpp/whisper.cpp/sd.cpp APIs in
  parallel — no more silent drift when upstream changes a struct.
- **Real C++ ergonomics.** `std::vector`, `std::function`, smart pointers,
  templates, overloads work natively. Today's Cython needs `libcpp.*` shims
  and awkward casts for function pointers.
- **Smaller binaries, faster compiles.** nanobind is the leanest of the C++
  binding libraries; often 2–3× faster compile and smaller `.so` than
  Cython-generated C++.
- **Better callback story.** The
  `(<object>py_callback)(...)` pattern (see `log_callback`, `abort_callback`)
  becomes idiomatic `std::function<...>` with `nb::cpp_function`.
- **Modern toolchain.** CMake-native (cyllama already uses CMake), good
  stub generation (`nanobind-stubgen`), better IDE/typing support.
- **One language: C++.** No more pyx/pxi/pxd file zoo, no `# distutils:`
  pragmas.

### Cons / risks

- **Pure-Python-flavoured glue gets harder.** Much of the `.pyx` is
  Python-style code (default args, optional kwargs, string decoding,
  list↔vector conversions). In Cython that's free; in nanobind it's C++.
  Some of it migrates better to a thin Python layer on top of a minimal C++
  binding.
- **`cpdef enum` blocks** become `nb::enum_<>` — straightforward but tedious
  given how many there are.
- **`include "*.pxi"`** (textual inclusion sharing C-level state across
  translation units — `mtmd.pxi`, `speculative.pxi`, `tts_helpers.pxi`) has
  no clean nanobind analogue. Refactor into normal C++ headers + separate
  binding modules with cross-module type sharing (nanobind supports it, but
  it's a design step).
- **Embedded mongoose server (`embedded.pyx`, 594 LOC)** — anything relying
  on Cython `nogil` semantics needs re-thinking with `nb::gil_scoped_release`.
- **Cython memoryviews → `nb::ndarray<>`.** Different API, mostly an
  improvement, but every call site changes.
- **Stricter compiler requirement.** C++17 minimum, C++20 preferred. Cython
  is more forgiving.
- **Loss of `.pyx`-level debugger / line profiling** (minor).
- **Stub regeneration** — typed Python API surface needs new stubs.

### Effort estimate

Treating each `cdef class` as ~0.5–1 day once a binding template is
established, plus upfront design work (module layout, ndarray strategy,
callback strategy, GIL strategy):

| Bucket                                                    | Estimate     |
| --------------------------------------------------------- | ------------ |
| Setup + first module skeleton + CMake/nanobind wiring     | 2–4 days     |
| llama bindings (~5.6K LOC, 25 classes, mtmd, speculative) | 3–5 weeks    |
| stable-diffusion (3.2K LOC, 14 classes)                   | 1.5–2 weeks  |
| whisper (~1K LOC, 10 classes)                             | 3–5 days     |
| embedded mongoose server (threading/IO)                   | 3–7 days     |
| Test parity, stubs, packaging (incl. Vulkan wheels), docs | 1–2 weeks    |

**Total: ~6–10 weeks of full-time work**, single developer, assuming the
public Python API stays roughly compatible. Add 2–3 weeks if redesigning
the Python surface as part of the port.

---

## Question 2: One statically linked binary

Since cyllama already pins one ggml across all three upstreams, the hardest
part of consolidation is already done. Remaining work is mostly mechanical:

1. **Flip ggml from SHARED to STATIC** in CMake (`BUILD_SHARED_LIBS=OFF` for
   the ggml subtree, or a dedicated `ggml_static` target). Ensure
   llama/whisper/sd all link the same static target.
2. **Merge the three Python extensions into one `.so`** with three
   submodules. In Cython: one top-level package init, three `.pyx` compiled
   and linked together against the single static ggml + static
   llama/whisper/sd. In nanobind: natural `NB_MODULE` + `def_submodule`.
3. **Drop `libggml.so` from the wheel** and remove the RPATH / loader
   plumbing.

**Estimated effort: 3–7 days** of CMake + packaging work. No binding rewrite
required.

### What this gains

- **No dynamic loader fragility.** No RPATH, no `DT_NEEDED`, no auditwheel /
  delocate / delvewheel having to repair `libggml.so` paths. Static linking
  removes a category of bugs.
- **Smaller wheel.** Modest — saves per-`.so` ELF/Mach-O overhead and some
  PLT/GOT churn, not ggml duplication. Probably single-digit MB.
- **Flat dependency graph.** One `.so`, no transitive shared-lib loading.
- **Cross-module C++ type sharing becomes trivial.** Relevant if a
  `ggml_backend_t` ever needs to flow between `cyllama.llama` and
  `cyllama.sd` without round-tripping through opaque pointers.
- **LTO across the boundary.** With LTO on, the compiler can inline ggml
  calls into llama/whisper/sd call sites. Small perf win, free once static.

### What this does *not* gain

- **Backend-registry correctness** — already correct via shared
  `libggml.so`.
- **ggml version coherence** — already pinned.
- **"No ggml duplication"** — there is no duplication today.

### Costs

- **Bigger single `.so`** — no lazy loading; importing `cyllama.whisper`
  pulls in llama+sd code too. Probably not material.
- **Chunkier CUDA/Vulkan build matrix.** One failing backend fails the whole
  wheel instead of one submodule.
- **Loses the option to ship `cyllama-llama` / `cyllama-whisper` /
  `cyllama-sd` as separable PyPI packages.**

---

## Recommendation

Treat these as two independent decisions and sequence them.

### Step 1 — Consolidate to one binary + static ggml in Cython (1 week)

Captures the loader/wheel wins immediately. Doesn't require touching binding
code. De-risks any future nanobind port by establishing the single-`.so`
structure first.

### Step 2 — Live with it for a release cycle

The strongest argument for nanobind is **eliminating `.pxd` drift against
upstream headers**. Whether that's actually painful is something you'll feel
in practice over a release or two. If `.pxd` maintenance is rarely a problem,
nanobind is mostly lateral motion.

### Step 3 — Port to nanobind only if Step 2 hurts

And even then, do it one submodule at a time. A nanobind submodule and a
Cython submodule can coexist in the same package during migration — there's
no need for a flag-day rewrite. Whisper is the cheapest first target
(~1 week) and a good spike to validate the approach before committing to
llama (3–5 weeks).

## Summary table

| Action                            | Effort     | Primary win                                  |
| --------------------------------- | ---------- | -------------------------------------------- |
| Static ggml + single `.so` (Cython) | 3–7 days   | Loader robustness, simpler wheels            |
| Whisper module → nanobind (spike) | ~1 week    | Validate ergonomics on a small surface       |
| Full Cython → nanobind port       | 6–10 weeks | Delete `.pxd`s, modern C++ ergonomics        |
