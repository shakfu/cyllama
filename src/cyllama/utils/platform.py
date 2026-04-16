"""Platform-specific runtime setup for native extension loading."""

import sys

_initialized = False


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
