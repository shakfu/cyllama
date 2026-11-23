#!/usr/bin/env python3
"""Audit a built wheel's external runtime dependencies against the project's
wheel-repair exclude lists.

Catches drift between what the GPU build actually links against and what
``_EXCLUDES_*`` in ``scripts/manage.py`` tells auditwheel /
delocate to leave external. If oneAPI / CUDA / ROCm bumps a soname and we
forget to update the exclude list, auditwheel either bloats the wheel by
bundling vendor runtimes or fails outright; running this auditor on the
built wheel in CI flags the mismatch before a release.

Linux: walks every ELF ``*.so*`` in the wheel, parses ``DT_NEEDED`` via
``llvm-readelf``/``readelf``. An entry is considered acceptable if (a)
it carries the auditwheel hash suffix ``-<8hex>.so`` (so the lib is
bundled into ``<pkg>.libs/`` with its NEEDED rewritten), (b) it's on
the manylinux baseline (libc/libm/libpthread/libdl/librt/libstdc++/
libgcc_s/ld-linux), or (c) it matches a pattern in
``_EXCLUDES_LINUX[<backend>]`` (fnmatch globs honored — the
SYCL list uses ``libmkl_*.so*``).

macOS: walks every Mach-O ``*.dylib``/``*.so`` via ``otool -L``. An
install_name is acceptable if it points into ``<pkg>/.dylibs/`` (where
delocate bundles), lives under ``/usr/lib/`` or ``/System/`` (OS-supplied),
or contains a substring from ``_EXCLUDES_DARWIN[<backend>]``
(delocate matches by substring, not glob).

Windows: walks every ``*.dll``/``*.pyd`` and reads its PE import and
delay-import tables directly (no pefile/dumpbin needed). An imported DLL
is acceptable if (a) a file of exactly that name is in the wheel, (b) it
is a Windows system DLL (``api-ms-win-*``, a core OS DLL, or -- when
auditing on Windows -- present in System32 and not a Visual C++
redistributable), (c) the interpreter supplies it (``python3*.dll``,
``vcruntime140*.dll``), or (d) it is a driver runtime the backend expects
(``WINDOWS_DRIVER_DLLS`` here, plus ``_WIN_EXCLUDES[<backend>]``).
Rule (a) deliberately ignores delvewheel's ``-<hash>.dll`` suffix: a
plugin delvewheel never traced keeps importing pre-rename names that no
longer exist, which is exactly the failure this catches.

Backend defaults to whatever appears between the first ``_`` and the
``-<version>`` in the wheel filename (``cyllama_sycl-...`` -> ``sycl``;
``cyllama-0.1.6-...`` -> no backend, treated as CPU/base build).

Also fails a wheel whose binaries carry a run path that is not relative
(``$ORIGIN`` / ``@loader_path``), or whose Linux extension modules lack
RELRO or BIND_NOW.

Exit codes:
    0 — no unexpected NEEDED entries / imports, run paths or hardening gaps
    1 — at least one unexpected NEEDED entry / import, run path or hardening gap
    2 — usage / tooling error (e.g. no readelf available)

Example::

    python scripts/audit_wheel.py dist/cyllama_sycl-0.3.0-*.whl
    python scripts/audit_wheel.py dist/cyllama_cuda-*.whl --backend cuda
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# Hoisted constants in manage.py; importing here is the single source of truth.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import manage  # noqa: E402

# manylinux_2_28+ allows these to remain external without an explicit
# `--exclude`; auditwheel's policy file does the same. Anything outside
# this set must be either bundled (hash-suffixed in NEEDED) or matched by
# the backend's exclude list — otherwise the wheel would bundle it.
MANYLINUX_BASELINE = frozenset(
    {
        "libc.so.6",
        "libm.so.6",
        "libpthread.so.0",
        "libdl.so.2",
        "librt.so.1",
        "libutil.so.1",
        "libresolv.so.2",
        "libstdc++.so.6",
        "libgcc_s.so.1",
        "ld-linux-x86-64.so.2",
        "ld-linux-aarch64.so.1",
    }
)

# auditwheel renames bundled libs to "<orig>-<8 hex chars>.so..." and
# rewrites the parent's DT_NEEDED to the new name. A NEEDED entry matching
# this pattern is therefore evidence of bundling, not an external dep.
BUNDLED_HASH_RE = re.compile(r"-[0-9a-f]{8}\.so")


# ---------------------------------------------------------------------------
# Tool discovery


def _find_readelf() -> str:
    """Return a path to a working readelf binary.

    Tries llvm-readelf first (ships with Homebrew/Xcode/Clang and is
    portable across hosts), then falls back to GNU readelf.
    """
    for cand in (
        "llvm-readelf",
        "readelf",
        "/opt/homebrew/opt/llvm/bin/llvm-readelf",
        "/usr/local/opt/llvm/bin/llvm-readelf",
    ):
        if cand.startswith("/"):
            if Path(cand).is_file():
                return cand
        else:
            found = shutil.which(cand)
            if found:
                return found
    print("ERROR: no readelf found (install LLVM or binutils).", file=sys.stderr)
    sys.exit(2)


def _find_otool() -> str:
    found = shutil.which("otool")
    if not found:
        print("ERROR: otool not available; macOS wheel audit requires Xcode CLI tools.", file=sys.stderr)
        sys.exit(2)
    return found


# ---------------------------------------------------------------------------
# Wheel walking


def _extract_wheel(wheel: Path, into: Path) -> None:
    with zipfile.ZipFile(wheel) as zf:
        zf.extractall(into)


# The wheel's distribution tag is not always the key used by the exclude
# lists in manage.py: the published variants are versioned (`cuda12`,
# `cuda13`) or named for the vendor SDK rather than the ggml backend
# (`rocm` builds the `hip` backend). Anything not listed here is assumed to
# already be an exclude-list key. Keep in sync with ALLOWED in
# scripts/ci_rename_package.py.
_WHEEL_TAG_TO_BACKEND: dict[str, str] = {
    "cuda12": "cuda",
    "cuda13": "cuda",
    "rocm": "hip",
}


class UnknownBackendError(ValueError):
    """The wheel carries a backend tag that maps to no exclude list."""


def _detect_backend(wheel_name: str) -> str:
    """Pull the backend tag out of the wheel filename.

    Convention (matches both inferna and cyllama): ``<pkg>[_<backend>]-<version>-...``.
    Returns ``""`` for plain CPU/base wheels (no backend suffix).

    Raises UnknownBackendError when a backend tag is present but maps to no
    exclude list. Falling back to ``""`` there would audit a GPU wheel against
    the CPU list, which allows nothing -- every legitimately runtime-supplied
    driver library then reports as an unexpected dependency, blaming the wheel
    for what is really a gap in this mapping.
    """
    # The tag may contain digits (`cuda12`), so it cannot be `[a-z]+`.
    m = re.match(r"[A-Za-z0-9]+(?:_([a-z][a-z0-9]*))?-\d", wheel_name)
    if not m:
        return ""
    tag = m.group(1) or ""
    if not tag:
        return ""
    backend = _WHEEL_TAG_TO_BACKEND.get(tag, tag)
    if backend not in manage._EXCLUDES_LINUX:
        raise UnknownBackendError(
            f"wheel {wheel_name!r} has backend tag {tag!r}, which maps to no entry in "
            f"manage.py:_EXCLUDES_LINUX (tried {backend!r}). Add it to "
            f"_WHEEL_TAG_TO_BACKEND in this script, or pass --backend explicitly."
        )
    return backend


def _wheel_platform(wheel_name: str) -> str:
    """Classify the wheel as linux / darwin / windows via its tag."""
    if "manylinux" in wheel_name or "linux_" in wheel_name:
        return "linux"
    if "macosx" in wheel_name:
        return "darwin"
    if "win_" in wheel_name or "win32" in wheel_name:
        return "windows"
    return ""


# ---------------------------------------------------------------------------
# Per-platform audit


_NEEDED_RE = re.compile(r"\(NEEDED\)\s+Shared library:\s+\[(.+?)\]")


def _so_needed(path: Path, readelf: str) -> set[str]:
    """Return the DT_NEEDED set for an ELF file, or empty if it's not ELF."""
    try:
        out = subprocess.run(
            [readelf, "-d", str(path)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    except OSError:
        return set()
    return set(_NEEDED_RE.findall(out))


def _audit_linux(wheel: Path, backend: str, root: Path) -> list[tuple[str, set[str]]]:
    """Return [(needed_lib, set_of_so_filenames)] for every unexpected entry."""
    readelf = _find_readelf()
    excludes = manage._EXCLUDES_LINUX.get(backend, [])

    # Collect NEEDED → which .so files referenced it. Multiple references
    # to the same lib are merged; the per-.so list is printed for context.
    refs: dict[str, set[str]] = {}
    for so in sorted(root.rglob("*.so*")):
        if not so.is_file() or so.is_symlink():
            continue
        for lib in _so_needed(so, readelf):
            refs.setdefault(lib, set()).add(str(so.relative_to(root)))

    unexpected: list[tuple[str, set[str]]] = []
    for lib in sorted(refs):
        if lib in MANYLINUX_BASELINE:
            continue
        if BUNDLED_HASH_RE.search(lib):
            # Bundled into <pkg>.libs/ — auditwheel rewrote the NEEDED.
            continue
        if any(fnmatch.fnmatchcase(lib, pat) for pat in excludes):
            continue
        unexpected.append((lib, refs[lib]))
    return unexpected


_DYLIB_LINE_RE = re.compile(r"^\s+(\S+)\s+\(compatibility version", re.M)


def _dylib_install_names(path: Path, otool: str) -> set[str]:
    try:
        out = subprocess.run(
            [otool, "-L", str(path)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    except OSError:
        return set()
    return set(_DYLIB_LINE_RE.findall(out))


_PE_IMPORT_DIR = 1
_PE_DELAY_IMPORT_DIR = 13


def _pe_imports(path: Path) -> set[str]:
    """Return the DLL names in a PE file's import and delay-import tables.

    Parsed directly from the headers so the audit needs neither pefile nor
    dumpbin. Returns an empty set for anything that is not a well-formed PE.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return set()
    try:
        if data[:2] != b"MZ":
            return set()
        (pe,) = struct.unpack_from("<I", data, 0x3C)
        if data[pe : pe + 4] != b"PE\0\0":
            return set()
        n_sections, _, _, _, opt_size = struct.unpack_from("<HIIIH", data, pe + 6)
        opt = pe + 24
        (magic,) = struct.unpack_from("<H", data, opt)
        # Data directories follow NumberOfRvaAndSizes, whose offset differs
        # between PE32 and PE32+.
        n_dirs_off = {0x10B: 92, 0x20B: 108}.get(magic)
        if n_dirs_off is None:
            return set()
        (n_dirs,) = struct.unpack_from("<I", data, opt + n_dirs_off)
        dirs = opt + n_dirs_off + 4
        sections: list[tuple[int, int, int]] = []
        for i in range(n_sections):
            off = opt + opt_size + 40 * i
            vsize, vaddr, rawsize, rawptr = struct.unpack_from("<IIII", data, off + 8)
            sections.append((vaddr, max(vsize, rawsize), rawptr))

        def file_offset(rva: int) -> int | None:
            for vaddr, size, rawptr in sections:
                if vaddr <= rva < vaddr + size:
                    return rawptr + rva - vaddr
            return None

        def c_string(rva: int) -> str | None:
            off = file_offset(rva)
            if off is None:
                return None
            end = data.find(b"\0", off)
            return data[off:end].decode("ascii", "replace") if end != -1 else None

        names: set[str] = set()
        # (directory index, descriptor size, offset of the DLL-name RVA)
        for index, desc_size, name_field in ((_PE_IMPORT_DIR, 20, 12), (_PE_DELAY_IMPORT_DIR, 32, 4)):
            if index >= n_dirs:
                continue
            rva, size = struct.unpack_from("<II", data, dirs + 8 * index)
            off = file_offset(rva) if rva and size else None
            if off is None:
                continue
            while off + desc_size <= len(data):
                desc = data[off : off + desc_size]
                if desc == b"\0" * desc_size:
                    break
                (name_rva,) = struct.unpack_from("<I", desc, name_field)
                name = c_string(name_rva)
                if name:
                    names.add(name)
                off += desc_size
        return names
    except struct.error:
        return set()


# Core OS DLLs present on every supported Windows install. Backstops the
# System32 lookup below, which is only possible when auditing on Windows.
WINDOWS_SYSTEM_DLLS = frozenset(
    {
        "advapi32.dll",
        "bcrypt.dll",
        "cfgmgr32.dll",
        "comctl32.dll",
        "crypt32.dll",
        "d3d12.dll",
        "dbghelp.dll",
        "dxgi.dll",
        "gdi32.dll",
        "iphlpapi.dll",
        "kernel32.dll",
        "ntdll.dll",
        "ole32.dll",
        "oleaut32.dll",
        "powrprof.dll",
        "psapi.dll",
        "rpcrt4.dll",
        "secur32.dll",
        "setupapi.dll",
        "shell32.dll",
        "shlwapi.dll",
        "ucrtbase.dll",
        "user32.dll",
        "userenv.dll",
        "version.dll",
        "winmm.dll",
        "ws2_32.dll",
    }
)

# Visual C++ redistributables. A dev box or CI runner usually has them in
# System32, but an end user's machine need not, so finding one there proves
# nothing: delvewheel must bundle them, and every import must name the
# bundled copy. vcruntime140*.dll is the exception -- the interpreter ships
# it and has already loaded it.
_MSVC_REDIST_RE = re.compile(r"^(msvcp|vcomp|concrt|vccorlib)\d+(_\w+)?\.dll$")
_PYTHON_SUPPLIED_RE = re.compile(r"^(python3\d*|vcruntime140(_\d+)?)\.dll$")

# Supplied by the GPU driver, never bundled: delvewheel finds these in
# System32 and leaves them external without being told to. Backend-specific
# runtimes that do need an explicit `--no-dll` live in
# manage._WIN_EXCLUDES and are allowed alongside these.
WINDOWS_DRIVER_DLLS: dict[str, list[str]] = {
    "vulkan": ["vulkan-1.dll"],
    "cuda": ["nvcuda.dll"],
}


def _is_windows_system_dll(name: str) -> bool:
    lower = name.lower()
    if lower.startswith(("api-ms-win-", "ext-ms-")) or lower in WINDOWS_SYSTEM_DLLS:
        return True
    if _MSVC_REDIST_RE.match(lower) or sys.platform != "win32":
        return False
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    return (system32 / name).is_file()


def _audit_windows(wheel: Path, backend: str, root: Path) -> list[tuple[str, set[str]]]:
    """Return [(imported_dll, set_of_importers)] for every unresolvable import.

    delvewheel renames each bundled DLL to `<name>-<hash>.dll` and rewrites the
    import tables of every binary it traced to match. A binary it did not
    trace -- a ggml backend plugin forced in with `--include`, which nothing in
    the wheel imports -- keeps the original names, and Windows refuses to load
    it once those names are gone. So rather than trust hash suffixes, check
    that each import names a file that is actually in the wheel.
    """
    allowed = {n.lower() for n in manage._WIN_EXCLUDES.get(backend, []) + WINDOWS_DRIVER_DLLS.get(backend, [])}
    binaries = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".dll", ".pyd"))
    bundled = {p.name.lower() for p in binaries}

    refs: dict[str, set[str]] = {}
    for path in binaries:
        for name in _pe_imports(path):
            refs.setdefault(name, set()).add(str(path.relative_to(root)))

    unexpected: list[tuple[str, set[str]]] = []
    for name in sorted(refs, key=str.lower):
        lower = name.lower()
        if lower in bundled or lower in allowed:
            continue
        if _PYTHON_SUPPLIED_RE.match(lower) or _is_windows_system_dll(name):
            continue
        unexpected.append((name, refs[name]))
    return unexpected


def _audit_darwin(wheel: Path, backend: str, root: Path) -> list[tuple[str, set[str]]]:
    otool = _find_otool()
    excludes = manage._EXCLUDES_DARWIN.get(backend, manage._DARWIN_BASE_EXCLUDES)

    refs: dict[str, set[str]] = {}
    for path in sorted(list(root.rglob("*.dylib")) + list(root.rglob("*.so"))):
        if not path.is_file() or path.is_symlink():
            continue
        for name in _dylib_install_names(path, otool):
            refs.setdefault(name, set()).add(str(path.relative_to(root)))

    unexpected: list[tuple[str, set[str]]] = []
    for name in sorted(refs):
        # delocate bundles into <pkg>/.dylibs/ and rewrites install_name to
        # @loader_path/../.dylibs/<name>. Anything routed through @loader_path,
        # @rpath, or @executable_path is bundled-or-relative — not external.
        if name.startswith(("@loader_path", "@rpath", "@executable_path")):
            continue
        # A bundled dylib's own install id, which otool -L lists first;
        # delocate sets it to /DLC/<pkg>/.dylibs/<name>. Not a dependency.
        if name.startswith("/DLC/"):
            continue
        # OS-supplied dylibs that delocate (and Apple) consider always-present.
        if name.startswith(("/usr/lib/", "/System/")):
            continue
        # delocate matches --exclude by substring against the install_name.
        if any(pat in name for pat in excludes):
            continue
        unexpected.append((name, refs[name]))
    return unexpected


# ---------------------------------------------------------------------------
# Run paths and hardening
#
# The 1.0.0a0 macOS CPU wheels shipped /Users/runner/work/... in LC_RPATH:
# repair tools leave a static wheel's binaries untouched. These checks keep
# that, and the Linux hardening flags, from regressing.

# Absolute run paths a backend may keep on purpose: the macOS Vulkan wheel
# resolves Homebrew's libvulkan (see manage.py _fix_macos_vulkan_wheel_impl).
ALLOWED_ABS_RPATHS = {"vulkan": ("/opt/homebrew/lib", "/usr/local/lib")}

_RUNPATH_RE = re.compile(r"\((?:RPATH|RUNPATH)\)\s+Library (?:rpath|runpath): \[([^\]]*)\]")


def _is_extension(path: Path, root: Path) -> bool:
    """True for cyllama's own extension modules, not libraries bundled from elsewhere."""
    rel = path.relative_to(root).as_posix()
    return rel.startswith("cyllama/") and (".cpython-" in path.name or ".abi3." in path.name)


def _audit_linux_paths(backend: str, root: Path) -> list[str]:
    readelf = _find_readelf()
    problems = []
    for so in sorted(root.rglob("*.so*")):
        if not so.is_file() or so.is_symlink():
            continue
        dyn = subprocess.run([readelf, "-d", str(so)], capture_output=True, text=True, check=False).stdout
        for entry in _RUNPATH_RE.findall(dyn):
            for d in entry.split(":"):
                if d and not d.startswith("$ORIGIN"):
                    problems.append(f"{so.relative_to(root)}: run path {d!r} is not $ORIGIN-relative")
        if _is_extension(so, root):
            seg = subprocess.run([readelf, "-lW", str(so)], capture_output=True, text=True, check=False).stdout
            if "GNU_RELRO" not in seg:
                problems.append(f"{so.relative_to(root)}: no RELRO (-Wl,-z,relro)")
            # (FLAGS) BIND_NOW, or (FLAGS_1) Flags: NOW ...
            if "BIND_NOW" not in dyn and not re.search(r"\(FLAGS_1\)\s+(?:Flags:)?[^\n]*\bNOW\b", dyn):
                problems.append(f"{so.relative_to(root)}: not BIND_NOW (-Wl,-z,now)")
    return problems


def _audit_darwin_paths(backend: str, root: Path) -> list[str]:
    otool = _find_otool()
    allowed = ALLOWED_ABS_RPATHS.get(backend, ())
    problems = []
    for lib in sorted([*root.rglob("*.dylib"), *root.rglob("*.so")]):
        if not lib.is_file() or lib.is_symlink():
            continue
        out = subprocess.run([otool, "-l", str(lib)], capture_output=True, text=True, check=False).stdout
        lines = out.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "cmd LC_RPATH" and i + 2 < len(lines):
                path = lines[i + 2].split()[1]
                if not path.startswith("@") and path not in allowed:
                    problems.append(f"{lib.relative_to(root)}: absolute LC_RPATH {path!r}")
    return problems


# ---------------------------------------------------------------------------
# Entry point


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("wheel", type=Path, help="Path to the .whl file to audit.")
    p.add_argument(
        "--backend",
        default=None,
        help="Backend name (cuda/hip/sycl/vulkan/opencl/cpu/metal). Inferred from the wheel filename if omitted.",
    )
    p.add_argument(
        "--platform",
        choices=["linux", "darwin", "windows"],
        default=None,
        help="Platform classification. Inferred from the wheel filename if omitted.",
    )
    args = p.parse_args(argv)

    wheel: Path = args.wheel
    if not wheel.is_file():
        print(f"ERROR: wheel not found: {wheel}", file=sys.stderr)
        return 2

    try:
        backend = args.backend if args.backend is not None else _detect_backend(wheel.name)
    except UnknownBackendError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    plat = args.platform or _wheel_platform(wheel.name)
    if not plat:
        print(f"ERROR: cannot classify platform from filename: {wheel.name}", file=sys.stderr)
        return 2

    print(f"wheel:    {wheel.name}")
    print(f"backend:  {backend or '(none)'}")
    print(f"platform: {plat}")
    print()

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _extract_wheel(wheel, root)

        if plat == "linux":
            unexpected = _audit_linux(wheel, backend, root)
            exclude_field = "_EXCLUDES_LINUX"
            path_problems = _audit_linux_paths(backend, root)
        elif plat == "darwin":
            unexpected = _audit_darwin(wheel, backend, root)
            exclude_field = "_EXCLUDES_DARWIN"
            path_problems = _audit_darwin_paths(backend, root)
        else:
            unexpected = _audit_windows(wheel, backend, root)
            exclude_field = "_WIN_EXCLUDES"
            path_problems = []

    if path_problems:
        print(f"FAIL: {len(path_problems)} run-path / hardening problem(s):")
        for problem in path_problems:
            print(f"  {problem}")
        print()
        if not unexpected:
            return 1

    if not unexpected:
        print(
            f"OK: every external dep is bundled, on the platform baseline, or matched by the {backend!r} exclude list."
        )
        return 0

    print(f"FAIL: {len(unexpected)} unexpected external dep(s):")
    for lib, sites in unexpected:
        print(f"  {lib}")
        for s in sorted(sites):
            print(f"      <- {s}")
    print()
    print(
        f"These are neither bundled nor matched by manage.py:{exclude_field}[{backend!r}]. "
        "Either add them to that list (if they should remain runtime-supplied) or "
        "investigate why the wheel was built against them."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
