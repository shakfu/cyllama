"""Tests for `scripts/audit_wheel.py`: backend detection and the Windows PE audit.

The audit decides which external dependencies a built wheel may leave
runtime-supplied by looking up `manage._EXCLUDES_LINUX[<backend>]` (or the
Darwin / Windows equivalent). The backend is inferred from the wheel filename,
and that filename does not always spell the backend the way the exclude lists
key it: published variants are versioned (`cyllama_cuda12`) or named for the
vendor SDK rather than the ggml backend (`cyllama_rocm` builds the `hip`
backend).

Getting this wrong is not a no-op. Falling back to `""`, whose exclude list is
empty, would report every driver library the wheel legitimately expects at
runtime as an unexpected dependency -- blaming the wheel for a gap in the
mapping. Ported from inferna, which hit exactly that.
"""

import fnmatch
import importlib.util
import struct
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(SCRIPTS))
audit_wheel = _load("audit_wheel")
manage = _load("manage")


WHEEL_SUFFIX = "-0.1.10-cp312-abi3-manylinux_2_35_x86_64.whl"


@pytest.mark.parametrize(
    "dist,expected",
    [
        ("cyllama", ""),
        ("cyllama_cuda12", "cuda"),
        ("cyllama_cuda13", "cuda"),
        ("cyllama_rocm", "hip"),
        ("cyllama_hip", "hip"),
        ("cyllama_vulkan", "vulkan"),
        ("cyllama_sycl", "sycl"),
        ("cyllama_opencl", "opencl"),
    ],
)
def test_detect_backend(dist, expected):
    assert audit_wheel._detect_backend(dist + WHEEL_SUFFIX) == expected


def test_every_published_variant_resolves():
    """Each name ci_rename_package.py allows must map to a real exclude list."""
    ci_rename = _load("ci_rename_package")

    for variant in sorted(ci_rename.ALLOWED):
        dist = variant.replace("-", "_")
        backend = audit_wheel._detect_backend(dist + WHEEL_SUFFIX)
        assert backend in manage._EXCLUDES_LINUX, f"{variant} detected as {backend!r}, which has no exclude list"


def test_unknown_backend_tag_raises():
    """An unmapped tag must fail loudly, not silently audit as a CPU wheel."""
    with pytest.raises(audit_wheel.UnknownBackendError):
        audit_wheel._detect_backend("cyllama_cuda99" + WHEEL_SUFFIX)


@pytest.mark.parametrize(
    "dist,needed",
    [
        # What inferna's CUDA and ROCm audits reported as "unexpected" while
        # the backend was misdetected as "".
        (
            "cyllama_cuda12",
            [
                "libcublas.so.12",
                "libcublasLt.so.12",
                "libcuda.so.1",
                "libcudart.so.12",
                "libgomp.so.1",
            ],
        ),
        (
            "cyllama_rocm",
            [
                "libamdhip64.so.6",
                "libgomp.so.1",
                "libhipblas.so.2",
                "librocblas.so.4",
            ],
        ),
    ],
)
def test_driver_libs_are_excluded_for_their_backend(dist, needed):
    """The libs those wheels link against must clear their own exclude list."""
    backend = audit_wheel._detect_backend(dist + WHEEL_SUFFIX)
    excludes = manage._EXCLUDES_LINUX[backend]

    unmatched = [lib for lib in needed if not any(fnmatch.fnmatchcase(lib, pat) for pat in excludes)]
    assert not unmatched, f"{dist} ({backend}) would still fail the audit on: {unmatched}"


def test_cpu_wheel_allows_nothing():
    """The empty-backend list stays empty -- the fallback that caused the bug."""
    assert manage._EXCLUDES_LINUX[""] == []


def test_exclude_list_key_sets_agree():
    """_detect_backend validates against the Linux dict but is used for both.

    The two dicts are keyed the same today; if they ever diverge, a darwin-only
    backend would be rejected as unknown.
    """
    assert set(manage._EXCLUDES_LINUX) == set(manage._EXCLUDES_DARWIN)


# ---------------------------------------------------------------------------
# Windows: PE import audit


def _make_pe(imports=(), delay_imports=()) -> bytes:
    """Build a minimal PE32+ image whose import tables name the given DLLs."""
    raw_ptr, section_rva = 0x200, 0x1000
    imp_size = 20 * (len(imports) + 1)
    delay_size = 32 * (len(delay_imports) + 1)
    names_off = imp_size + delay_size
    body = bytearray(names_off)
    for i, name in enumerate([*imports, *delay_imports]):
        rva = section_rva + len(body)
        body += name.encode() + b"\0"
        if i < len(imports):
            struct.pack_into("<I", body, 20 * i + 12, rva)
        else:
            struct.pack_into("<I", body, imp_size + 32 * (i - len(imports)) + 4, rva)

    head = bytearray(raw_ptr)
    head[0:2] = b"MZ"
    struct.pack_into("<I", head, 0x3C, 0x40)
    head[0x40:0x44] = b"PE\0\0"
    opt_size = 240
    struct.pack_into("<HHIIIHH", head, 0x44, 0x8664, 1, 0, 0, 0, opt_size, 0)
    opt = 0x58
    struct.pack_into("<H", head, opt, 0x20B)
    struct.pack_into("<I", head, opt + 108, 16)
    dirs = opt + 112
    if imports:
        struct.pack_into("<II", head, dirs + 8 * 1, section_rva, imp_size)
    if delay_imports:
        struct.pack_into("<II", head, dirs + 8 * 13, section_rva + imp_size, delay_size)
    section = opt + opt_size
    head[section : section + 8] = b".idata\0\0"
    struct.pack_into("<IIII", head, section + 8, len(body), section_rva, len(body), raw_ptr)
    return bytes(head + body)


def test_pe_imports_reads_import_and_delay_import_tables(tmp_path):
    dll = tmp_path / "x.dll"
    dll.write_bytes(_make_pe(["KERNEL32.dll", "ggml-base.dll"], ["nvcuda.dll"]))
    assert audit_wheel._pe_imports(dll) == {"KERNEL32.dll", "ggml-base.dll", "nvcuda.dll"}


def test_pe_imports_ignores_non_pe(tmp_path):
    junk = tmp_path / "junk.dll"
    junk.write_bytes(b"not a PE file at all")
    assert audit_wheel._pe_imports(junk) == set()


def _vulkan_wheel(root: Path, *, mangled: bool) -> None:
    """Lay out a repaired cyllama_vulkan wheel: a traced extension plus the
    --include'd ggml-vulkan.dll plugin, which delvewheel never rewrites."""
    libs = root / "cyllama_vulkan.libs"
    libs.mkdir(parents=True)
    base = "ggml-base-11e962b8f57276b0c0ecd825ee9db2e1.dll" if mangled else "ggml-base.dll"
    (libs / base).write_bytes(_make_pe(["KERNEL32.dll", "VCRUNTIME140.dll"]))
    (root / "cyllama").mkdir()
    (root / "cyllama" / "_llama_native.pyd").write_bytes(
        _make_pe([base, "KERNEL32.dll", "python3.dll", "api-ms-win-crt-heap-l1-1-0.dll"])
    )
    (libs / "ggml-vulkan.dll").write_bytes(_make_pe(["ggml-base.dll", "vulkan-1.dll", "KERNEL32.dll"]))


def test_windows_flags_plugin_importing_mangled_name(tmp_path):
    """The 0.4.2 vulkan wheel: ggml-base.dll was renamed, the plugin was not
    rewritten, and Windows refused to load ggml-vulkan.dll."""
    _vulkan_wheel(tmp_path, mangled=True)
    unexpected = audit_wheel._audit_windows(None, "vulkan", tmp_path)
    assert [lib for lib, _ in unexpected] == ["ggml-base.dll"]
    assert unexpected[0][1] == {str(Path("cyllama_vulkan.libs") / "ggml-vulkan.dll")}


def test_windows_passes_with_project_libs_unmangled(tmp_path):
    _vulkan_wheel(tmp_path, mangled=False)
    assert audit_wheel._audit_windows(None, "vulkan", tmp_path) == []


def test_windows_driver_dll_only_allowed_for_its_backend(tmp_path, monkeypatch):
    # A host with the Vulkan driver has vulkan-1.dll in System32, which would
    # pass the import as a system DLL; point the lookup at an empty dir.
    monkeypatch.setenv("SystemRoot", str(tmp_path / "Windows"))
    _vulkan_wheel(tmp_path, mangled=False)
    unexpected = audit_wheel._audit_windows(None, "cuda", tmp_path)
    assert [lib for lib, _ in unexpected] == ["vulkan-1.dll"]


def test_windows_flags_msvc_redist_even_if_installed(tmp_path):
    """A dev box has MSVCP140.dll in System32; a user's machine may not."""
    (tmp_path / "plugin.dll").write_bytes(_make_pe(["MSVCP140.dll", "VCRUNTIME140_1.dll"]))
    unexpected = audit_wheel._audit_windows(None, "", tmp_path)
    assert [lib for lib, _ in unexpected] == ["MSVCP140.dll"]


def test_no_mangle_covers_what_backend_plugins_import():
    """Every project lib a ggml backend plugin links against must keep its
    real name, or the --include'd plugin cannot resolve it."""
    no_mangle = {n.lower() for n in manage._WIN_NO_MANGLE}
    assert {"ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "msvcp140.dll"} <= no_mangle


def test_windows_driver_dll_keys_are_backends():
    assert set(audit_wheel.WINDOWS_DRIVER_DLLS) <= set(manage._EXCLUDES_LINUX)
