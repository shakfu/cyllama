# abi3 Conversion Plan

## Motivation

cyllama currently builds a separate wheel for each supported Python version
(3.10, 3.11, 3.12, 3.13, 3.14) x each platform x each GPU backend.
Converting the extensions to use the CPython stable ABI (abi3 / limited API)
collapses the Python-version axis: one wheel tagged `cp310-abi3-<plat>`
works on CPython 3.10 and every newer version.

Effective reduction is ~5x (only the Python-version axis collapses; the
platform and GPU-backend axes are unaffected).

Cython 3.1+ supports `limited_api=True`, which generates code that avoids
non-stable CPython internals (no `PyTypeObject` field access; types via
`PyType_FromSpec`; etc.). The existing `.pyx` files do not use any features
that block abi3 (no direct `PyObject_HEAD` manipulation, no buffer-protocol
internals, exceptions handled implicitly by Cython).

## Design: abi3 as an opt-in build mode

abi3 is wired as a CMake option (`CYLLAMA_ABI3`, default `OFF`) so the
existing per-version build continues to work unchanged. The Cython
`limited_api` directive is applied via `CYTHON_ARGS` rather than a
pragma at the top of each `.pyx` file, and `USE_SABI` is added to each
`python_add_library` call only when the option is on.

Two ways to enable:

1. Local build: `cmake -DCYLLAMA_ABI3=ON ...` (or via scikit-build-core
   config settings: `pip install . --config-settings=cmake.define.CYLLAMA_ABI3=ON`).
2. CI abi3 job: environment variable `SKBUILD_CMAKE_DEFINE=CYLLAMA_ABI3=ON`
   plus `SKBUILD_WHEEL_PY_API=cp310` to tag the wheel correctly.

## 1. CMakeLists.txt changes

**Line 1** - bump minimum:

```cmake
cmake_minimum_required(VERSION 3.26...3.30)
```

`USE_SABI` requires CMake 3.26.

**After the existing options block** (around line 30) - add the option:

```cmake
option(CYLLAMA_ABI3 "Build extensions against the CPython stable ABI (abi3)" OFF)
```

**Line 134** - extend Python components conditionally:

```cmake
if(CYLLAMA_ABI3)
    find_package(Python COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)
else()
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
endif()
```

**Around the `CYTHON_INCLUDE_ARGS` block (line 574)** - append the
limited-api directive when opted in:

```cmake
if(CYLLAMA_ABI3)
    list(APPEND CYTHON_INCLUDE_ARGS "-X" "limited_api=True")
endif()
```

`cython_transpile` forwards `CYTHON_ARGS` to the Cython compiler, so `-X
limited_api=True` has the same effect as a `# cython: limited_api=True`
pragma without touching the source files.

**In `add_cython_extension` (line 592) and the three inline
`python_add_library` calls** (embedded ~725, whisper ~798,
stable_diffusion ~848) - gate `USE_SABI` on the option. Simplest form
using a helper variable near the top of the extensions section:

```cmake
if(CYLLAMA_ABI3)
    set(_SABI_ARGS USE_SABI 3.10)
else()
    set(_SABI_ARGS "")
endif()
```

Then at each `python_add_library` call:

```cmake
python_add_library(${target_name} MODULE WITH_SOABI ${_SABI_ARGS}
    ${cython_cpp}
    ${ARG_SOURCES}
)
```

`USE_SABI 3.10` (CMake 3.26+) links `Python::SABIModule`, defines
`Py_LIMITED_API=0x030a0000`, and uses the `.abi3.so` / `.pyd` suffix.

## 2. pyproject.toml changes

Do **not** hard-code `wheel.py-api` in `[tool.scikit-build]`, since that
would force every wheel to be tagged abi3 regardless of how the code was
compiled. Instead, the abi3 CI job sets it via environment variable:

```
SKBUILD_WHEEL_PY_API=cp310
SKBUILD_CMAKE_DEFINE=CYLLAMA_ABI3=ON
```

The default build reads neither and produces per-version wheels as today.

In `[tool.cibuildwheel]` the default `build = "cp310-* cp311-* cp312-*
cp313-* cp314-*"` remains untouched. The abi3 CI workflow overrides with
`CIBW_BUILD=cp310-*` and the two `SKBUILD_*` variables above.

## 3. Cython source changes

None. Not touching the `.pyx` files is the whole point of this design:
the limited-API directive is passed at build time.

## 4. Not affected

- `vector` (C shared lib, line 961) - does not use CPython API.
- `mongoose.c` / `mongoose_wrapper.c` (line 727) - pure C, no CPython.
- `scripts/manage.py` thirdparty builds - unchanged.
- `before-all` and `before-build` hooks - unchanged (thirdparty deps are
  Python-version-agnostic and already built once per platform).

## 5. Rollout

1. Branch, apply the CMake changes, verify the default `make && make
   test` on 3.12 still passes unchanged (CYLLAMA_ABI3 defaults to OFF).
2. Verify opt-in path: `pip install . --config-settings=cmake.define.CYLLAMA_ABI3=ON`
   on 3.12, confirm `.abi3.so` suffix on the installed modules, run
   `make test`.
3. Add a parallel CI job building `cp310-abi3` wheels alongside the
   existing matrix for one release.
4. Add a post-build verification job: download the abi3 wheel,
   `pip install` + `pytest` on 3.10, 3.13, and 3.14.
5. After one clean release with no regressions, consider whether to
   switch the default (or keep both paths indefinitely).

## 6. Known risks

- Cython `limited_api=True` occasionally generates code that fails to
  compile when a macro it expects is absent on a given Python version;
  the 3.14 build is the most likely to surface this (newest ABI). Verify
  explicitly.
- Any `.pxd` in dependencies that `cimport`s from `cpython.*` submodules
  may reach non-limited APIs. If the abi3 build fails, grep for `from
  cpython` and inspect.
- Wheel tag `cp310-abi3-<plat>.whl` is handled correctly by pip >= 20.3;
  very old pip versions will not resolve it.
- Free-threaded builds (`cp313t`, `cp314t`) are not covered by abi3 and
  still need their own wheels if supported.
- The minimum-Python floor (3.10) is locked in for the lifetime of the
  abi3 wheel: C API additions from 3.11+ cannot be used without either
  raising the floor (e.g. `cp311`) or dropping abi3 on that code path.

## 7. Benefit estimate

Current wheel matrix per release (approximate):

- Linux x86_64: 5 wheels/backend
- macOS Intel + ARM: 10 wheels/backend
- Windows x86_64: 5 wheels/backend

After abi3 (when the abi3 path is used): 1/2/1 wheels per backend
respectively - roughly a 5x reduction in total wheel count and a
proportional reduction in CI build time and artifact storage. Because
the non-abi3 path remains available, the two matrices can coexist during
transition.
