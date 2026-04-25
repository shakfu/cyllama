# Consolidating to a single Cython extension binary

This document analyses whether to collapse cyllama's three Cython extensions
(`llama_cpp`, `whisper_cpp`, `stable_diffusion`) — plus the `embedded`
server extension — into a single shared object, and sketches how to do it
in Cython if the answer is yes.

The binding-tool question (Cython vs nanobind) is treated separately in
[`nanobind.md`](nanobind.md). This document is about binary layout only.

## TL;DR

cyllama ships with **dynamic linking** (`WITH_DYLIB=ON`). Under that mode,
single-binary consolidation is mostly a structural change with small
payoffs — **not recommended unless there is a specific reason**. The big
wins (eliminating ggml duplication, unifying the backend registry) only
exist in the static-linking mode, which is not the shipping path.

Skip to [When this matters](#when-this-matters) for the short answer.

## Current state: two modes

cyllama supports two link modes via the `WITH_DYLIB` CMake option.

### `WITH_DYLIB=ON` — the shipping path

ggml is built once as a shared library; all three extensions link against
it at runtime. One ggml registry per process. The wheel ships
`libggml.so` plus backend dylibs (`libggml-cuda.so`,
`libggml-cpu-haswell.so`, etc.) plus the three extension `.so`s, with
RPATH plumbing handled by auditwheel / delocate / delvewheel.

Backend dispatch uses `GGML_BACKEND_DL=ON`, so ggml `dlopen`s the
appropriate `libggml-cpu-*.so` variant at runtime. Those variant dylibs
must remain as separate files regardless of any consolidation.

### `WITH_DYLIB=OFF` — static linking, fallback only

Each extension statically links its own copy of `libggml.a`,
`libggml-base.a`, `libggml-cpu.a` (and the active backend). This means
three copies of ggml in the wheel and three independent backend
registries, isolated by Python's `RTLD_LOCAL` extension loading plus
`-fvisibility=hidden`. Works in practice because each extension sets up
the backends it needs itself, but the registries diverge.

This mode exists as a fallback for environments where shipping dylibs is
awkward; it is **not** the production path.

## What consolidation buys you under dynamic linking

Under `WITH_DYLIB=ON`, the big-ticket wins evaporate:

| Claimed win                              | Under dynamic linking                       |
| ---------------------------------------- | ------------------------------------------- |
| One ggml backend registry                | **Already have it** via shared `libggml.so` |
| No 3× ggml duplication on disk           | **Already none** — one `libggml.so`         |
| Cross-module backend visibility          | **Already correct**                         |

What actually remains:

- **Three extension `.so`s become one.** Saves the per-`.so` ELF/Mach-O
  overhead and a bit of PLT/GOT churn. Single-digit MB at most.
- **Slightly simpler RPATH.** One consumer of `libggml.so` instead of
  three. auditwheel/delocate already handle the three-consumer case
  cleanly, so this is not a real pain point today.
- **Cross-submodule C++ type sharing.** Only matters if you later want to
  pass a `ggml_backend_t` Python wrapper or similar directly between
  `cyllama.llama` and `cyllama.sd`. No concrete need today.
- **Preparation for a future nanobind port.** Single-`.so` is the natural
  shape there.

Critically, **the dylib zoo cannot be eliminated even with one extension**.
With `GGML_BACKEND_DL=ON`, ggml `dlopen`s backend variants at runtime, so
the wheel still ships:

- `libggml-cpu-*.so` variants — **must remain separate** for runtime
  dispatch.
- `libggml-cuda.so` / `libggml-vulkan.so` / etc. — **must remain separate**
  for runtime dispatch.
- `libggml.so`, `libggml-base.so` could in principle be linked into the
  extension if `GGML_BACKEND_DL` were turned off, but that would lose
  runtime CPU-variant dispatch.
- `libllama.so`, `libmtmd.so`, `libwhisper.so` could be linked into the
  extension, but doing so loses the option of using upstream's pre-built
  release tarballs via `LLAMACPP_DYLIB_DIR`.

Net effect: you'd save maybe 2–4 dylibs out of a wheel that already ships
6–10+, and the mental model gets *more* complicated (some things linked
in, some still `dlopen`'d).

## When this matters

Single-binary consolidation is worth doing only if one of these applies:

- **You want to ship a statically linked wheel variant** — e.g. for
  distros that dislike vendored dylibs, or for Windows where DLL search
  rules are painful. Consolidation is a prerequisite there, because
  static linking without consolidation gives you 3× ggml duplication and
  registry divergence.
- **You hit a concrete cross-module sharing need.** Passing C++ objects
  between `cyllama.llama` and `cyllama.sd` directly, sharing a
  `ggml_backend_t` wrapper, etc.
- **You are preparing to migrate to nanobind.** Single-`.so` is the
  natural target shape; doing this first establishes the packaging
  layout before any binding code changes.

Otherwise the current dynamic-linking story is the right answer for the
production workload. `libggml.so` is exactly the sharing mechanism shared
libraries exist for.

## How to do it in Cython (if you decide to)

The rest of this document sketches the consolidation. The sketch targets
the **static** case, because that is where consolidation has real value;
under dynamic linking the same mechanical structure applies but the
payoff is smaller.

### Target shape

```
src/cyllama/
├── __init__.py
├── _core.cpython-312-x86_64-linux-gnu.so   ← single binary
├── llama/
│   └── __init__.py    # `from cyllama._core.llama import *`
├── whisper/
│   └── __init__.py    # `from cyllama._core.whisper import *`
└── sd/
    └── __init__.py    # `from cyllama._core.sd import *`
```

One `.so`, exporting `PyInit__core`, which internally creates three
Python submodules. The embedded server lives in the same `.so` — no
separate `embedded` extension.

### Cython side

Cython compiles one `.pyx` per Python module. To get three submodules
into one extension, compile each `.pyx` to `.cpp` independently, then
link all generated `.cpp` files into a single `python_add_library`. Each
generated `.cpp` defines a `PyInit_<modname>`. Suppress all but one and
call the others manually from a small `_core.pyx`:

```cython
# src/cyllama/_core.pyx
# distutils: language = c++

cdef extern from "Python.h":
    object PyImport_AddModule(const char*)
    int PyModule_AddObject(object, const char*, object) except -1

# Forward-declare the per-submodule init functions Cython generates.
cdef extern from *:
    """
    extern "C" PyObject* PyInit_llama_cpp(void);
    extern "C" PyObject* PyInit_whisper_cpp(void);
    extern "C" PyObject* PyInit_stable_diffusion(void);
    """
    object PyInit_llama_cpp()
    object PyInit_whisper_cpp()
    object PyInit_stable_diffusion()

def _bootstrap():
    """Attach submodules under cyllama._core."""
    pkg = PyImport_AddModule("cyllama._core")
    PyModule_AddObject(pkg, "llama",   PyInit_llama_cpp())
    PyModule_AddObject(pkg, "whisper", PyInit_whisper_cpp())
    PyModule_AddObject(pkg, "sd",      PyInit_stable_diffusion())

_bootstrap()
```

Then in `src/cyllama/llama/__init__.py`:

```python
from cyllama._core import llama as _ext
from cyllama._core.llama import *   # noqa
```

Tidier variants exist using multi-phase init / `PyModuleDef_Init`, but
the manual `PyInit_*` call from `_core.pyx` is the most pragmatic for an
existing Cython codebase.

### CMake side

Replace the three `add_cython_extension(...)` calls with one. Sketch (for
the static case — which is the case where consolidation pays off):

```cmake
# Transpile each .pyx independently.
cython_transpile(src/cyllama/_core.pyx              LANGUAGE CXX OUTPUT_VARIABLE _core_cpp
    CYTHON_ARGS ${CYTHON_INCLUDE_ARGS})
cython_transpile(src/cyllama/llama/llama_cpp.pyx    LANGUAGE CXX OUTPUT_VARIABLE llama_cpp
    CYTHON_ARGS ${CYTHON_INCLUDE_ARGS})
cython_transpile(src/cyllama/whisper/whisper_cpp.pyx LANGUAGE CXX OUTPUT_VARIABLE whisper_cpp
    CYTHON_ARGS ${CYTHON_INCLUDE_ARGS})
cython_transpile(src/cyllama/sd/stable_diffusion.pyx LANGUAGE CXX OUTPUT_VARIABLE sd_cpp
    CYTHON_ARGS ${CYTHON_INCLUDE_ARGS})

# One Python extension, all translation units linked together.
python_add_library(_core MODULE WITH_SOABI ${_SABI_ARGS}
    ${_core_cpp}
    ${llama_cpp}
    ${whisper_cpp}
    ${sd_cpp}
    ${EMBEDDED_SOURCES}      # mongoose etc. — no longer a separate .so
)

target_include_directories(_core PRIVATE ${COMMON_INCLUDE_DIRS} ...)

# Whole-archive ggml ONCE, not per-extension. This is the duplication fix
# for the static path.
if(UNIX AND NOT APPLE)
    target_link_libraries(_core PRIVATE
        -Wl,--whole-archive
            ${LIB_GGML} ${LIB_GGML_BASE} ${LIB_GGML_CPU}
            ${_BACKEND_GGML_LIBS}     # ggml-vulkan, ggml-cuda, etc.
        -Wl,--no-whole-archive
        ${LIB_LLAMA} ${LIB_MTMD}
        ${LIB_WHISPER_COMMON} ${LIB_WHISPER}
        ${LIB_SD}
        ${SYSTEM_LIBS})
elseif(APPLE)
    target_link_libraries(_core PRIVATE
        -Wl,-force_load,${LIB_GGML}
        -Wl,-force_load,${LIB_GGML_BASE}
        -Wl,-force_load,${LIB_GGML_CPU}
        ${_BACKEND_FORCE_LOAD_FLAGS}
        ${LIB_LLAMA} ${LIB_MTMD}
        ${LIB_WHISPER_COMMON} ${LIB_WHISPER}
        ${LIB_SD}
        ${SYSTEM_LIBS})
else()  # Windows / MSVC
    target_link_libraries(_core PRIVATE
        ${STATIC_LIBS} ${LIB_SD}      # MSVC keeps unreferenced symbols by default
        ${SYSTEM_LIBS})
    target_link_options(_core PRIVATE
        /WHOLEARCHIVE:ggml
        /WHOLEARCHIVE:ggml-base
        /WHOLEARCHIVE:ggml-cpu)
endif()

install(TARGETS _core LIBRARY DESTINATION cyllama)
```

The whole-archive (or `-force_load` / `/WHOLEARCHIVE`) is what
guarantees backend registration constructors actually run — the same
reason the current Linux build uses `--whole-archive` on ggml libs per
extension.

For the **dynamic** case, the same single `python_add_library(_core ...)`
works; just link against `libggml.so` etc. as today and skip the
whole-archive flags. The loader handles backend registration via the
shared library's own constructors.

### Things to watch out for

1. **Multi-init safety.** Calling `PyInit_*` manually means each Cython
   submodule must be safe to initialize once. They are by default —
   Cython's generated init does its own once-guard. But if any of them
   register C-level globals that could conflict (log callbacks, etc.),
   audit carefully.
2. **Symbol visibility.** With `-fvisibility=hidden`, only `PyInit__core`
   should be exported. The forward-declared `PyInit_llama_cpp` etc. need
   to be visible *to the linker within the same .so* — they are, since
   they are in the same translation-unit set.
3. **`SD_USE_VENDORED_GGML=OFF` becomes mandatory.** With one binary you
   cannot have sd.cpp's own ggml linked alongside llama.cpp's ggml — ODR
   violation. The existing `GGML_MAX_NAME=128` define for this case
   already handles the layout-divergence concern.
4. **macOS `-force_load` is per-archive.** Each ggml backend needs its
   own `-force_load`; there is no batched form.
5. **Windows `/WHOLEARCHIVE`** uses bare library names, not paths; pass
   them as `target_link_options`.
6. **Test that `RTLD_GLOBAL` is not relied upon anywhere.** This used to
   be three `.so`s and is now one — anything that depended on a symbol
   being *not* visible across modules (unlikely) would break.
   Conversely, anything that depended on cross-extension symbol
   visibility would now Just Work.

### Effort breakdown

| Step                                                              | Effort      |
| ----------------------------------------------------------------- | ----------- |
| `_core.pyx` bootstrap + Python package shims                      | ~1 day      |
| CMake refactor (one extension, drop duplicated link paths)        | 1–2 days    |
| Fix any hidden cross-module assumptions revealed by tests         | 1–2 days    |
| Wheel matrix re-verification (CPU/Metal/CUDA/Vulkan/HIP/SYCL)     | 2–3 days    |
| **Total**                                                         | **~1 week** |

## Recommendation

**Do not pursue single-binary consolidation as a standalone project.**

Under the shipping `WITH_DYLIB=ON` mode, the structural wins are real but
small (one `.so` instead of three, slightly simpler RPATH) and the dylib
zoo cannot be eliminated anyway because of `GGML_BACKEND_DL` runtime
dispatch. The current setup is a good fit for the workload.

Do pursue it if and when one of the triggers in
[When this matters](#when-this-matters) appears — most likely either a
decision to ship a static wheel variant, or as the first step of a
nanobind migration. In those scenarios this document's sketch is the
starting point.

## Relationship to the nanobind question

This consolidation is independent of any future nanobind migration. If
nanobind is on the roadmap, doing the consolidation first establishes
the single-`.so` packaging layout before any binding code changes, and a
nanobind port can then proceed one submodule at a time within that
layout. See [`nanobind.md`](nanobind.md) for the broader analysis.
