"""Platform-specific runtime setup: native extension loading and CPU topology."""

import functools
import os
import subprocess
import sys
from pathlib import Path

_initialized = False


@functools.cache
def physical_cores() -> int:
    """Physical cores available to this process.

    Decode is memory-bound, so SMT siblings slow it; llama.cpp's own tools default to this count.
    """
    if sys.platform.startswith("linux"):
        cores = set()
        for cpu in os.sched_getaffinity(0):
            topo = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            try:
                cores.add(((topo / "physical_package_id").read_text(), (topo / "core_id").read_text()))
            except OSError:
                pass
        if cores:
            return len(cores)
    elif sys.platform == "darwin":
        try:  # performance cores only, as llama.cpp's common library counts them
            out = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True)
            return max(1, int(out.stdout))
        except (OSError, ValueError):
            pass
    return max(1, (os.cpu_count() or 2) // 2)  # assume 2-way SMT


def resolve_n_threads(n_threads: int = -1, share: int = 1) -> int:
    """Return `n_threads` if positive, else physical cores divided among `share` concurrent contexts."""
    return n_threads if n_threads > 0 else max(1, physical_cores() // share)


def ensure_native_deps() -> None:
    """Ensure platform-specific shared libraries are discoverable.

    Idempotent. On Windows, registers DLL search paths for backends that
    require external toolkit DLLs (e.g. CUDA). No-op on other platforms
    or when the backend does not need runtime DLL discovery.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    if sys.platform != "win32":
        return

    from .._internal import build_config

    if build_config.backend_enabled("cuda"):
        _setup_cuda_dll_paths()


def _setup_cuda_dll_paths() -> None:
    """Register CUDA toolkit DLL directories on Windows."""
    import glob
    import os
    import re
    import shutil

    if not hasattr(os, "add_dll_directory"):
        return

    seen: set[str] = set()

    def add_bin(path: str) -> None:
        if path in seen or not os.path.isdir(path):
            return
        seen.add(path)
        try:
            os.add_dll_directory(path)  # type: ignore[attr-defined]
        except OSError:
            pass

    # 1. Explicit env vars (highest priority)
    for key in ("CUDA_PATH", "CUDA_HOME"):
        root = os.environ.get(key)
        if root:
            add_bin(os.path.join(root, "bin"))

    # 2. nvcc on PATH
    nvcc = shutil.which("nvcc")
    if nvcc:
        add_bin(os.path.dirname(os.path.abspath(nvcc)))

    # 3. Standard install location (newest version first)
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    cuda_root = os.path.join(pf, "NVIDIA GPU Computing Toolkit", "CUDA")
    if os.path.isdir(cuda_root):

        def ver_key(d: str) -> tuple[int, ...]:
            m = re.search(r"v(\d+)\.(\d+)", d)
            return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

        for vdir in sorted(
            glob.glob(os.path.join(cuda_root, "v*")),
            key=ver_key,
            reverse=True,
        ):
            add_bin(os.path.join(vdir, "bin"))
