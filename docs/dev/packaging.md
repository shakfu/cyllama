# macOS x86_64 Vulkan Wheel Packaging

## Status

Proposal. Not yet applied. Supersedes the `_rewrite_dynamic_install_names`
approach introduced in commit `6d1511a`.

## Problem

`cibuildwheel` -> `delocate-wheel` fails to repair the `cyllama-vulkan`
macOS x86_64 wheel with two distinct error classes:

1. `@loader_path/libggml-blas.0.dylib not found` — needed by
   `cyllama/llama/libllama.0.dylib`, `libggml.0.dylib`, `libmtmd.0.dylib`.
2. `@rpath/libllama.dylib not found` (plus `libggml`, `libggml-base`,
   `libggml-cpu`, `libggml-vulkan`, `libmtmd`) — needed by
   `cyllama/llama/llama_cpp.cpython-312-darwin.so` and
   `cyllama/whisper/whisper_cpp.cpython-312-darwin.so`. Note the refs are
   the **unversioned** dylib names.

Both classes surface only after `_rewrite_dynamic_install_names` rewrote
`LC_ID_DYLIB` entries in `thirdparty/llama.cpp/dynamic/`.

## Root cause analysis

The current pipeline has four stages that interact subtly:

### Stage 1: upstream llama.cpp build

Upstream CMake produces, for each shared lib:

- `libllama.0.dylib` — real file, `LC_ID_DYLIB = @rpath/libllama.0.dylib`

- `libllama.dylib`   — symlink to `libllama.0.dylib`

Sibling load commands use `@rpath/libX.0.dylib` (versioned).

On macOS x86_64, GGML auto-detects the Accelerate framework and builds
`libggml-blas.dylib` regardless of whether `GGML_VULKAN=1` is set, so
`libggml`, `libllama`, and `libmtmd` carry `@rpath/libggml-blas.0.dylib`
load commands.

### Stage 2: `manage_macos_intel.py` copy into `thirdparty/.../dynamic/`

The copy loop (`LlamaCppBuilder`, around lines 1248–1257) dereferences
symlinks. After this step, `dynamic/` contains **two independent real
files** per library:

- `dynamic/libllama.dylib`   — real file, `LC_ID = @rpath/libllama.0.dylib`
  (inherited from the symlink target, not the filename)

- `dynamic/libllama.0.dylib` — real file, `LC_ID = @rpath/libllama.0.dylib`

Byte-identical, same install name. Harmless on its own.

### Stage 3: `_rewrite_dynamic_install_names` (6d1511a)

This function iterates every `*.dylib` in `dynamic/` and runs:

```
install_name_tool -id @rpath/<basename> <file>
```

For `dynamic/libllama.dylib` the basename is `libllama.dylib`, so its
`LC_ID` is rewritten to `@rpath/libllama.dylib` — **unversioned**. The
two copies now have *different* install names despite identical bytes.

It also rewrites sibling `LC_LOAD_DYLIB` entries from
`@rpath/libX.0.dylib` to `@loader_path/libX.0.dylib`.

### Stage 4: cyllama extension build + wheel install

`CMakeLists.txt` links extensions via `find_library(llama …)`, which the
linker resolves against `dynamic/libllama.dylib` (the unversioned name is
searched first). The linker records the file's `LC_ID_DYLIB` into the
extension's `LC_LOAD_DYLIB`, which post-rewrite is
`@rpath/libllama.dylib`.

Meanwhile `CMakeLists.txt`'s `_find_dylib` macro (lines 335–344) installs
**only the versioned soname** (`libllama.0.dylib`) into
`cyllama/llama/`. The unversioned file is dropped.

Result after staging the wheel:

| Consumer                                     | Reference                         |
|----------------------------------------------|-----------------------------------|
| `cyllama/llama/llama_cpp.…so`                | `@rpath/libllama.dylib`           |
| `cyllama/whisper/whisper_cpp.…so`            | `@rpath/libllama.dylib`           |
| `cyllama/llama/libllama.0.dylib` (sibling)   | `@loader_path/libggml-blas.0.dylib` |
| `cyllama/llama/` contents                    | only versioned `lib*.0.dylib`     |

`delocate-wheel` finds **no** `libllama.dylib` (unversioned) and **no**
`libggml-blas.0.dylib` anywhere in the wheel — hence both error classes.

### Why the rewrite was introduced

Commit `6d1511a`'s justification: prevent `delocate-wheel` from copying
duplicate sibling libs into `cyllama/.dylibs/` with distinct install
names, which would cause dyld to load two copies of `libggml` at
runtime.

The rewrite's *sibling-load* half (`@rpath/` -> `@loader_path/`) does
achieve that. The *LC_ID* half is the part that causes the current
failure — fabricating install names that don't match what gets shipped.

## Proposed fix

Remove the fabricated-install-name hack. Address each real problem at
its real source.

### Change 1: do not rewrite `LC_ID_DYLIB`

Upstream's versioned `LC_ID = @rpath/libllama.0.dylib` is correct and
relocatable. Keep it. Drop the
`install_name_tool -id @rpath/<basename>` call from
`_rewrite_dynamic_install_names` in
`scripts/manage_macos_intel.py`.

### Change 2: replace sibling-load rewrite with rpath injection

Instead of rewriting every sibling `LC_LOAD_DYLIB` from `@rpath/` to
`@loader_path/`, add `@loader_path` to each dylib's `LC_RPATH` once:

```
install_name_tool -add_rpath @loader_path <file>
```

Effect: upstream's `@rpath/libggml.0.dylib` references resolve to the
sibling at runtime with no install-name mutation. `delocate-wheel` walks
rpaths and matches `@rpath/libggml.0.dylib` against
`cyllama/llama/libggml.0.dylib` directly; no duplicate is copied into
`cyllama/.dylibs/`.

Wart: `install_name_tool -add_rpath` fails if the rpath is already
present. Either parse `otool -l` to check first, or tolerate the failure
(`... || true`). Prefer the check — silent `|| true` hides real errors.

### Change 3: ship `libggml-blas` on macOS

`libggml-blas.dylib` is a genuine runtime dependency of `libllama` /
`libggml` / `libmtmd` on macOS because GGML auto-links Accelerate. It
must be shipped in the wheel. Accelerate itself is present on every
macOS system, so no further bundling is required.

In `CMakeLists.txt` near line 305 (`_BACKEND_DYLIB_NAMES` block), add:

```cmake
if(APPLE)
    list(APPEND _OPTIONAL_DYLIB_NAMES ggml-blas)
endif()
```

Optional, not required — so non-Apple builds (Linux, Windows) don't
FATAL_ERROR if it isn't produced. On Apple builds where Accelerate is
found (the default), it will be picked up and installed.

### Change 4 (optional): preserve symlinks in the copy loop

In `scripts/manage_macos_intel.py`, stop dereferencing symlinks. Keep
`dynamic/libllama.dylib` as a symlink to `libllama.0.dylib`.

This is optional if Changes 1–3 are applied — no fabricated install name
is created, so the duplicate-with-different-id problem disappears. But
preserving symlinks matches upstream llama.cpp's layout and how other
macOS wheels ship dylibs (e.g. numpy, scipy).

Trade-off: Python's `wheel` tooling and `scikit-build-core` preserve
symlinks inside the tree at install time, but symlinks inside a `.whl`
zip archive depend on the zip entry's external attributes being read
correctly. Some older pip/wheel versions flatten them. Verify
end-to-end before committing to this change.

## Why this is less hackish than 6d1511a

| Dimension                  | 6d1511a (current)                    | Proposed                          |
|----------------------------|--------------------------------------|-----------------------------------|
| Mutates upstream `LC_ID`   | Yes — fabricates unversioned IDs    | No — upstream IDs preserved       |
| Creates phantom files      | Yes — unversioned id w/o unversioned install | No                          |
| Matches upstream conventions | No — diverges                      | Yes — `@rpath` + `@loader_path`   |
| Sibling dedup in wheel     | Via load-command rewrite             | Via rpath injection (dyld-native) |
| `libggml-blas` handling    | Silently dropped on Vulkan builds    | Shipped as optional dep on APPLE  |
| Special cases in CMake     | None (hidden in script)              | One `if(APPLE)` in CMake          |
| Lines of non-stdlib glue   | ~45                                  | ~10                               |

## Implementation steps

1. Edit `scripts/manage_macos_intel.py`:
   - Replace `_rewrite_dynamic_install_names` body with an `otool -l`
     check + `install_name_tool -add_rpath @loader_path` per dylib.

   - Drop the `install_name_tool -id ...` call entirely.
2. Edit `CMakeLists.txt` line ~305:
   - Add `if(APPLE) list(APPEND _OPTIONAL_DYLIB_NAMES ggml-blas) endif()`
     outside the backend-specific blocks.
3. (Optional, gated on symlink-in-wheel verification) Edit
   `scripts/manage_macos_intel.py` copy loop to preserve symlinks via
   `os.symlink(item.readlink(), dest)` when `item.is_symlink()`.
4. Test locally: `make build-vulkan` on a macOS x86_64 host, inspect
   `thirdparty/llama.cpp/dynamic/` install names with `otool -D` and
   `otool -l`, then run `delocate-wheel` on the built wheel manually.
5. CI: re-run `build-new-wheels.yml` with `vulkan_macos_intel=true`.

## Verification checklist

After applying:

- [ ] `otool -D dynamic/libllama.0.dylib` prints
      `@rpath/libllama.0.dylib`.

- [ ] `otool -l dynamic/libllama.0.dylib | grep -A2 LC_RPATH` includes
      `@loader_path`.

- [ ] `otool -L dynamic/libllama.0.dylib` still shows upstream
      `@rpath/libggml.0.dylib` (unmodified).

- [ ] Built extensions reference `@rpath/libllama.0.dylib` (versioned).

- [ ] `cyllama/llama/` contains `libggml-blas.0.dylib`.

- [ ] `delocate-wheel` completes without errors and **does not** copy
      any `libggml*` or `libllama*` into `cyllama/.dylibs/`.

- [ ] `unzip -l <repaired wheel>` shows exactly one copy of each
      versioned dylib.

- [ ] Installed wheel imports and runs a basic `complete()` call.

## References

- Commit `6d1511a` — introduced `_rewrite_dynamic_install_names`.

- `scripts/manage_macos_intel.py` — build / dylib-staging logic.

- `CMakeLists.txt` lines 215–380, 994–1005 — `_find_dylib` macro and
  dylib install rules.

- `.github/workflows/build-new-wheels.yml` lines 48–142 — Vulkan macOS
  Intel job.

- `delocate` docs on rpath resolution:
  https://github.com/matthew-brett/delocate

## Addendum: comparison with xorbitsai/xllamacpp

`xorbitsai/xllamacpp` is a fork with a substantially different packaging
pipeline. Its Vulkan macOS Intel build succeeds where ours fails, so the
divergences are instructive.

### Workflow inventory

Four workflows in `.github/workflows/`:

- `build-wheel.yaml` — CPU wheels for Linux (x86_64 + aarch64), macOS
  (x86_64 + arm64), Windows (AMD64). Publishes to PyPI on `v*` tags.

- `build-wheel-cuda-hip.yaml` — GPU wheels: HIP Linux (two ROCm
  versions), Vulkan (Linux/macOS-Intel/Windows), CUDA Linux
  (x86_64+aarch64, two CUDA versions), CUDA Windows. Publishes to
  GitHub Releases with backend-tagged tags.

- `ci.yaml` — lint (black + clang-format) + pytest matrix
  (Linux/macOS/Windows, Python 3.10 + 3.13).

- `release-github-pypi.yaml` — release plumbing.

### Architectural differences vs. cyllama

| Dimension | xllamacpp | cyllama |
|---|---|---|
| Wheel ABI | Single **abi3** wheel per platform (`cp310-*`, covers Py 3.10+) | Per-Python-version wheels |
| Build driver | `python -m build --wheel` in hand-managed envs; `cibuildwheel` only for non-Linux CPU | `cibuildwheel` everywhere |
| Linux CPU | `docker run quay.io/pypa/manylinux2014_{x86_64,aarch64}` hand-driven | `cibuildwheel` |
| Linux HIP | `docker run rocm/dev-ubuntu-22.04:<ver>` with host Python bind-mounted | N/A |
| Linux CUDA | `conda-incubator/setup-miniconda` + `mamba install cuda==X.Y.Z` on bare runner | `Jimver/cuda-toolkit` + Docker |
| Windows CUDA | `Jimver/cuda-toolkit@v0.2.30` + `w64devkit` MinGW | Similar |
| Vulkan macOS Intel | `brew install molten-vk shaderc vulkan-loader ...` on `macos-15-intel`, **no cibuildwheel**, `delocate-wheel -v dist/*.whl` with **no flags** | `cibuildwheel` + `delocate-wheel --exclude libvulkan --exclude libMoltenVK` |
| Wheel repair flags | Linux: `auditwheel repair --exclude libstdc++.so.6 --exclude libgomp.so.1` or `--plat manylinux_2_35_x86_64`; macOS: plain `delocate-wheel -v`; Windows: `delvewheel repair --exclude nvcuda.dll` | Extensive platform-specific `--exclude` rules |
| GPU wheel distribution | **GitHub Releases** via `softprops/action-gh-release`, tags like `v1.2.3-cu128`, `v1.2.3-rocm-6.4.1`, `v1.2.3-vulkan-macos-15-intel` | Different |
| `dynamic/` staging dir | None — dylibs are emitted with correct install names by CMake directly | Separate `thirdparty/.../dynamic/` with post-copy rewrites |
| `install_name_tool` usage | None in workflow | `_rewrite_dynamic_install_names` (6d1511a) |

### Why xllamacpp's Vulkan macOS Intel build succeeds

Three compounding reasons:

1. **No sibling staging directory, no install-name fabrication.** Their
   CMake builds llama.cpp inline via `make` and emits dylibs with
   upstream's own `@rpath/libX.0.dylib` install names. No
   dereferenced-symlink duplicate, no basename-derived `LC_ID` rewrite,
   so the failure mode diagnosed in this document does not exist.

2. **They do not exclude `libvulkan` / `libMoltenVK` from the wheel.**
   `delocate-wheel` is free to bundle them into `*.dylibs/` alongside
   the rest. The wheel is larger but self-contained — no runtime
   dependency on user-installed MoltenVK. Our `--exclude libMoltenVK`
   is what forced the install-name gymnastics that broke things.

3. **abi3 build with a single `.so` per module.** With
   `Py_LIMITED_API`, the Cython extension is one `.so` per Python
   target; combined with inline CMake, there is no manual dylib
   staging. Fewer moving parts to misalign.

### Worth borrowing

- **Ship MoltenVK in the wheel** (drop `--exclude libMoltenVK`) for
  macOS Vulkan. Larger wheel but removes a user-facing runtime
  prerequisite and sidesteps the current `delocate` pain. This is
  orthogonal to the main fix above — depending on how `delocate`
  handles the additional bundling, it may also obviate the
  `libggml-blas` leg by letting `delocate` resolve everything itself.
  **Decision: adopt.**

### Considered but deferred / rejected

- **abi3 wheels as default.** One wheel per platform covers Python
  3.10+ and cuts build time and storage roughly 5x. However, abi3
  extensions can be measurably slower than version-specific Cython
  extensions because the limited API forces indirection through
  accessor functions instead of inlined struct access. For a
  performance-sensitive binding like cyllama, the speed regression is
  not acceptable as the default.
  **Decision: keep per-Python-version wheels as default. Optionally
  offer an abi3 build variant as a separate wheel for users who prefer
  a single-wheel install and accept the perf cost.**

- **GitHub Releases with backend-tagged tag suffixes** (xllamacpp style:
  `v1.2.3-cu128`, `v1.2.3-rocm-6.4.1`).
  **Decision: not needed.** cyllama already publishes GitHub Releases
  of regular wheels, using package-name variants (`cyllama`,
  `cyllama-<gpu-variant>`) rather than tag suffixes. Same outcome,
  different naming convention.

- **llama.cpp as a git submodule.** Would simplify caching but
  sacrifices the flexibility of the current
  download-and-checkout-into-`thirdparty/build` system (swapping llama
  .cpp versions without touching the outer repo, independent fetch of
  whisper.cpp / stable-diffusion.cpp).
  **Decision: keep the current system.**

### Not worth borrowing

- Hand-rolled `docker run` for Linux CPU wheels is more fragile than
  `cibuildwheel`. Their HIP/CUDA jobs need `docker run` because the
  base images are ROCm/CUDA-specific, but the plain CPU Linux job has
  no such constraint.

- `mamba install cuda==X.Y.Z` on a bare runner is clever but adds
  ~10 minutes per build. `Jimver/cuda-toolkit` is faster.

- Multi-line `docker exec bash -c '...'` blocks with embedded
  heredocs are hard to debug in CI logs. Prefer short shell stages
  with explicit inputs.

### Actionable follow-ups

Independent of the main Vulkan macOS Intel fix:

1. **Drop `--exclude libMoltenVK` (and possibly `--exclude libvulkan`)
   from the `delocate-wheel` invocation** in
   `.github/workflows/build-new-wheels.yml`'s `build_vulkan_macos_intel`
   job. Verify the resulting wheel runs on a clean macOS system with no
   Vulkan SDK installed. If successful, this may be sufficient on its
   own — re-evaluate whether Changes 1–3 above are still needed.
2. **Add an optional abi3 build variant** as a separate wheel
   (`cyllama-abi3` or similar) for users who prefer a single wheel per
   platform. Keep per-Python-version wheels as the default so the
   perf-sensitive path is unchanged.
