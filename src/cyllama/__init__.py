from .defaults import (  # noqa: F401
    LLAMA_DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_MIN_P,
    DEFAULT_REPEAT_PENALTY,
    DEFAULT_PENALTY_LAST_N,
    DEFAULT_PENALTY_FREQ,
    DEFAULT_PENALTY_PRESENT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_N_GPU_LAYERS,
    DEFAULT_N_BATCH,
    DEFAULT_MAIN_GPU,
    DEFAULT_SPLIT_MODE,
)

# High-level API.
#
# The first import below transitively loads the Cython extension, which has
# the linked llama.cpp shared libraries as DT_NEEDED entries. For SYCL
# wheels we deliberately do not vendor the Intel oneAPI runtime (libsycl,
# MKL, TBB, libiomp5, the Intel compiler runtimes) -- see
# docs/installation.md#cyllama-sycl-host-prerequisites for the rationale.
# Without the host runtime on the loader path, the bare dlopen error
# (`libsycl.so.8: cannot open shared object file`) is uninformative;
# rewrite it into something actionable.
try:
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
except (ImportError, OSError) as _exc:
    from ._internal import build_config as _bc

    # Sonames the SYCL wheel expects from the host oneAPI runtime
    # (kept in sync with the auditwheel --exclude list in
    # .github/workflows/build-gpu-wheels-abi3.yml).
    _SYCL_RUNTIME_SONAMES = (
        "libsycl.so",
        "libmkl_",
        "libtbb.so",
        "libiomp5.so",
        "libsvml.so",
        "libimf.so",
        "libintlc.so",
        "libirng.so",
        "libOpenCL.so",
    )
    _msg = str(_exc)
    if _bc.backend_enabled("sycl") and any(s in _msg for s in _SYCL_RUNTIME_SONAMES):
        raise ImportError(
            "cyllama-sycl could not load a required Intel oneAPI runtime "
            f"library:\n    {_msg}\n\n"
            "The SYCL wheel does not vendor the oneAPI userspace runtimes "
            "(libsycl, MKL, TBB, libiomp5, the Intel compiler runtimes). "
            "Install them and put them on the loader path before importing "
            "cyllama. On Debian/Ubuntu, after adding the Intel oneAPI APT "
            "repo:\n\n"
            "    sudo apt install \\\n"
            "        intel-oneapi-runtime-dpcpp-cpp \\\n"
            "        intel-oneapi-runtime-mkl \\\n"
            "        intel-oneapi-runtime-tbb \\\n"
            "        intel-oneapi-runtime-openmp\n"
            "    source /opt/intel/oneapi/setvars.sh\n\n"
            "See docs/installation.md#cyllama-sycl-host-prerequisites for "
            "RPM-based distros, the separate GPU/CPU device-driver "
            "requirement, and links to Intel's install guides."
        ) from _exc
    raise

# Batching
from .batching import batch_generate, BatchGenerator, BatchRequest, BatchResponse

# Memory utilities
from .memory import estimate_gpu_layers, estimate_memory_usage

__version__ = "0.3.0"
