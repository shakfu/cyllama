# Consolidating to a single Cython extension binary

This document describes how to collapse cyllama's three Cython extensions
(`llama_cpp`, `whisper_cpp`, `stable_diffusion`) — plus the `embedded` server
extension — into a single shared object, and what is gained by doing so.

The binding-tool question (Cython vs nanobind) is treated separately in
[`nanobind.md`](nanobind.md). This document is about binary layout only.

## Current state: two modes

cyllama already supports two link modes via the `WITH_DYLIB` CMake option.

### `WITH_DYLIB=OFF` (default) — static linking, three extensions

Each of `llama_cpp.so`, `whisper_cpp.so`, `stable_diffusion.so` is a separate
Python extension that statically links its own copy of `libggml.a`,
`libggml-base.a`, `libggml-cpu.a` (and the active backend, e.g.
`libggml-vulkan.a`).

This means:

- **Three copies of ggml** on disk in the wheel.
- **Three independent ggml backend registries** at runtime in the same Python
  process, each isolated by Python's default `RTLD_LOCAL` extension loading
  plus `-fvisibility=hidden`. `ggml_backend_register` calls in
  `cyllama.llama` are *not* visible to `cyllama.sd`.
- The `--whole-archive` linker pass that ensures backend self-registration
  constructors run is applied **per extension**, multiplying the static-code
  cost.

In practice this works because each extension sets up the backends it needs
itself, but the registries diverge — a real correctness gap, not just a size
issue.

### `WITH_DYLIB=ON` — dynamic linking via `libggml.so`

ggml is built once as a shared library; all three extensions link against it
at runtime. One ggml registry per process. Wheel ships `libggml.so` plus the
backend dylibs plus the three extension `.so`s plus RPATH plumbing handled
by auditwheel/delocate/delvewheel.

## Which is the consolidation target — static or dynamic?

**Static.** The single-binary goal is most valuable when ggml is statically
linked, because that is where the duplication and registry-divergence costs
exist today. With dynamic ggml, runtime already sees one ggml; the only
remaining win is dropping `libggml.so` from the wheel.

The rest of this document targets **one statically linked extension**.

## Target shape

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

One `.so`, exporting `PyInit__core`, which internally creates three Python
submodules. The embedded server lives in the same `.so` as well — no
separate `embedded` extension.

## Cython side

Cython compiles one `.pyx` per Python module. To get three submodules into
one extension, compile each `.pyx` to `.cpp` independently, then link all
generated `.cpp` files into a single `python_add_library`. Each generated
`.cpp` defines a `PyInit_<modname>`. Suppress all but one and call the
others manually from a small `_core.pyx`:

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

Tidier variants exist using multi-phase init / `PyModuleDef_Init`, but the
manual `PyInit_*` call from `_core.pyx` is the most pragmatic for an
existing Cython codebase.

## CMake side

Replace the three `add_cython_extension(...)` calls with one. Sketch:

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

# Whole-archive ggml ONCE, not per-extension. This is the duplication fix.
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

The whole-archive (or `-force_load` / `/WHOLEARCHIVE`) is what guarantees
backend registration constructors actually run — the same reason the
current Linux build uses `--whole-archive` on ggml libs per extension.

## What this achieves

- **One ggml backend registry** in the process. `ggml_backend_register`
  calls are visible across llama/whisper/sd code paths. Currently they are
  not (each extension has its own private `RTLD_LOCAL` copy).
- **One copy of ggml on disk.** Wheel size drops by
  `2 × sizeof(ggml + active backends)`. With Vulkan/CUDA backends statically
  linked, this is non-trivial — easily 30–80 MB for CUDA builds.
- **One copy of llama/whisper/sd code on disk** (each was duplicated less,
  since they are only linked into their own `.so`, but still).
- **No `--whole-archive` triplication.** The biggest disk-size win is here.
- **`embedded` server collapses into the same `.so`.** The `cpp-httplib`
  static lib stops being linked into both `llama_cpp` and `embedded`.

## Things to watch out for

1. **Multi-init safety.** Calling `PyInit_*` manually means each Cython
   submodule must be safe to initialize once. They are by default —
   Cython's generated init does its own once-guard. But if any of them
   register C-level globals that could conflict (log callbacks, etc.),
   audit carefully.
2. **Symbol visibility.** With `-fvisibility=hidden`, only `PyInit__core`
   should be exported. The forward-declared `PyInit_llama_cpp` etc. need to
   be visible *to the linker within the same .so* — they are, since they
   are in the same translation-unit set. No special action needed.
3. **`SD_USE_VENDORED_GGML=OFF` becomes mandatory.** With one binary you
   cannot have sd.cpp's own ggml linked alongside llama.cpp's ggml — ODR
   violation. The existing `GGML_MAX_NAME=128` define for this case
   already handles the layout-divergence concern.
4. **macOS `-force_load` is per-archive.** Each ggml backend needs its own
   `-force_load`; there is no batched form.
5. **Windows `/WHOLEARCHIVE`** uses bare library names, not paths; pass
   them as `target_link_options`.
6. **Test `RTLD_GLOBAL` is not relied upon anywhere.** Since this used to
   be three `.so`s and is now one, anything that depended on a symbol
   being *not* visible across modules (unlikely but possible) would break.
   Conversely, anything that depended on a symbol from one module being
   accessible to another via `RTLD_GLOBAL` would now Just Work.

## Effort breakdown

| Step                                                              | Effort      |
| ----------------------------------------------------------------- | ----------- |
| `_core.pyx` bootstrap + Python package shims                      | ~1 day      |
| CMake refactor (one extension, whole-archive once, drop dupes)    | 1–2 days    |
| Fix any hidden cross-module assumptions revealed by tests         | 1–2 days    |
| Wheel matrix re-verification (CPU/Metal/CUDA/Vulkan/HIP/SYCL)     | 2–3 days    |
| **Total**                                                         | **~1 week** |

## Relationship to the nanobind question

This consolidation is independent of any future nanobind migration. Doing
it first:

- Captures the registry-correctness and wheel-size wins immediately.
- Establishes the single-`.so` structure, so a future nanobind port can
  proceed one submodule at a time within the same packaging layout.
- Does not require touching any binding code.

See [`nanobind.md`](nanobind.md) for the broader analysis.
