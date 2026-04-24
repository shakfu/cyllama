"""Helpers for discovering ggml backend libraries in wheel repair directories.

Wheel repair tools rename and/or relocate shared libraries when bundling
them into wheels:

- **auditwheel** (Linux): ``cyllama*.libs/`` at site-packages level,
  filenames get a content hash (e.g. ``libggml-vulkan-3e3d7523.so``).
- **delvewheel** (Windows): ``cyllama*.libs/`` at site-packages level,
  filenames get a content hash (e.g. ``ggml-vulkan-3e3d7523.dll``).
- **delocate** (macOS): ``cyllama/.dylibs/`` inside the package,
  filenames are preserved.

ggml's built-in ``ggml_backend_load_all_from_path()`` cannot discover
non-variant backends (those without ``ggml_backend_score``) after hash
renaming because its fallback path expects the original filename.

The :func:`libs_to_load` function scans the platform-appropriate
directories for all ggml backend library candidates so the caller can
load them explicitly via ``ggml_backend_load()``, bypassing ggml's
filename-based discovery entirely.
"""

from __future__ import annotations

import glob
import os
import sys


# ggml's filename conventions  (ggml-backend-reg.cpp)
#   - Linux/macOS: libggml-<name>.so  (ggml builds modules as .so even on macOS)
#   - Windows:     ggml-<name>.dll
_PREFIX = "ggml-" if sys.platform == "win32" else "libggml-"
_EXT = ".dll" if sys.platform == "win32" else ".so"

# All three Cython modules (llama, whisper, sd) share the same ggml
# dynamic library and thus the same backend registry.  This flag ensures
# the scan runs only once to avoid duplicate registrations.
_libs_loaded = False


# Support libraries that are not loadable backends.  These lack the
# ggml_backend_init symbol and attempting to load them produces noisy
# log warnings.  The set contains the first segment of the stem after
# stripping the prefix (e.g. "base" from "libggml-base-<hash>.so").
_NON_BACKENDS = frozenset(("base",))


def _is_hex(s: str) -> bool:
    """Return True if *s* looks like a hex hash (8+ hex chars)."""
    return len(s) >= 8 and all(c in "0123456789abcdef" for c in s)


def _is_backend_candidate(fname: str, siblings: frozenset[str] = frozenset()) -> bool:
    """Return True if *fname* looks like a ggml backend library.

    Filters out support libraries that lack ``ggml_backend_init``::

        libggml-<hash>.so          → core ggml (skip)
        libggml-base-<hash>.so     → support lib (skip)
        libggml-cpu-<hash>.so      → CPU dispatch shim if variants exist, else the CPU backend itself
        libggml-cpu-x64-<hash>.so  → CPU variant (keep)
        libggml-vulkan-<hash>.so   → GPU backend (keep)
    """
    if not fname.startswith(_PREFIX) or not fname.endswith(_EXT):
        return False
    # Extract stem: "libggml-vulkan-3e3d7523.so" -> "vulkan-3e3d7523"
    stem = fname[len(_PREFIX) : -len(_EXT)]
    if not stem:
        return False
    parts = stem.split("-")
    # Skip the core library: libggml-<hash>.so — only a hex hash
    if len(parts) == 1 and _is_hex(parts[0]):
        return False
    # Skip known non-backend support libs
    if parts[0] in _NON_BACKENDS:
        return False
    # libggml-cpu-<hash>.so is a dispatch shim *only* when CPU variants
    # (libggml-cpu-<arch>-<hash>.so) are present as siblings. When no
    # variants exist — as in most GPU wheels — this file IS the CPU
    # backend and must be loaded, otherwise llama.cpp fails with
    # "make_cpu_buft_list: no CPU backend found".
    if parts[0] == "cpu" and len(parts) == 2 and _is_hex(parts[-1]):
        for sib in siblings:
            if sib == fname:
                continue
            if not sib.startswith(_PREFIX + "cpu-") or not sib.endswith(_EXT):
                continue
            sib_parts = sib[len(_PREFIX) : -len(_EXT)].split("-")
            if len(sib_parts) >= 3:
                return False
    return True


def _scan_dir(dirpath: str, results: list[bytes]) -> None:
    """Append encoded paths of ggml backend candidates in *dirpath*."""
    try:
        entries = os.listdir(dirpath)
    except OSError:
        return
    siblings = frozenset(entries)
    for fname in entries:
        if _is_backend_candidate(fname, siblings):
            results.append(os.path.join(dirpath, fname).encode())


def libs_to_load(site_dir: str) -> list[bytes]:
    """Return encoded paths of ggml backend libs to load, or ``[]`` if
    already called.

    Scans platform-appropriate wheel repair directories for files
    matching ``[lib]ggml-*.[so|dll]``.  Returns each path as ``bytes``
    ready for ``ggml_backend_load()``.  Subsequent calls return ``[]``
    so backends are only registered once across all modules sharing the
    same ggml registry.

    Directories searched:
    - ``cyllama*.libs/``  under *site_dir* (auditwheel / delvewheel)
    - ``cyllama/.dylibs/`` under *site_dir* (delocate on macOS)
    """
    global _libs_loaded
    if _libs_loaded:
        return []
    _libs_loaded = True

    results: list[bytes] = []
    # auditwheel (Linux) / delvewheel (Windows)
    for libs_dir in glob.glob(os.path.join(site_dir, "cyllama*.libs")):
        _scan_dir(libs_dir, results)
        # Windows: ggml's ggml_backend_load() calls LoadLibraryW(path, NULL, 0),
        # which uses the standard search order and does NOT include the
        # directory of the DLL being loaded. Bundled siblings (ggml-base-*,
        # vulkan-1-*, etc.) are only resolvable if the libs dir is on PATH.
        # os.add_dll_directory() (set by delvewheel's __init__ patch) does
        # not help here — that only affects LoadLibraryEx callers passing
        # LOAD_LIBRARY_SEARCH_USER_DIRS.
        if sys.platform == "win32":
            path = os.environ.get("PATH", "")
            if libs_dir not in path.split(os.pathsep):
                os.environ["PATH"] = libs_dir + os.pathsep + path
    # delocate (macOS) — libs inside the package with original names
    dylibs_dir = os.path.join(site_dir, "cyllama", ".dylibs")
    _scan_dir(dylibs_dir, results)
    return results
