# Windows: CUDA-linked extensions depend on toolkit DLLs (e.g. cublas64_13.dll).
# Add CUDA bin dirs to the DLL search path before loading native modules.
# Many shells/IDEs do not set CUDA_PATH; discover standard install locations too.
import glob
import os
import re
import shutil
import sys

if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    _seen_bins: set[str] = set()

    def _add_dll_bin(path: str) -> None:
        if path in _seen_bins or not os.path.isdir(path):
            return
        _seen_bins.add(path)
        try:
            os.add_dll_directory(path)
        except OSError:
            pass

    def _cuda_ver_key(install_dir: str) -> tuple[int, ...]:
        m = re.search(r"v(\d+)\.(\d+)", install_dir)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        return (0, 0)

    _ordered_bins: list[str] = []
    for _key in ("CUDA_PATH", "CUDA_HOME"):
        _root = os.environ.get(_key)
        if _root:
            _ordered_bins.extend(
                [
                    os.path.join(_root, "bin"),
                    os.path.join(_root, "bin", "x64"),
                ]
            )
    _nvcc = shutil.which("nvcc")
    if _nvcc:
        _ordered_bins.append(os.path.dirname(os.path.abspath(_nvcc)))
    _pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    _cuda_root = os.path.join(_pf, "NVIDIA GPU Computing Toolkit", "CUDA")
    if os.path.isdir(_cuda_root):
        _vdirs = sorted(
            glob.glob(os.path.join(_cuda_root, "v*")),
            key=_cuda_ver_key,
            reverse=True,
        )
        for _vd in _vdirs:
            _ordered_bins.extend(
                [
                    os.path.join(_vd, "bin"),
                    os.path.join(_vd, "bin", "x64"),
                ]
            )
    for _b in _ordered_bins:
        _add_dll_bin(_b)

# High-level API
from .api import (
    LLM,
    complete,
    chat,
    simple,
    GenerationConfig,
    GenerationStats,
    Response,
    ResponseCacheInfo,
    # Async API
    AsyncLLM,
    complete_async,
    chat_async,
)

# Batching
from .batching import batch_generate, BatchGenerator, BatchRequest, BatchResponse

# Memory utilities
from .memory import estimate_gpu_layers, estimate_memory_usage

__version__ = "0.2.2"
