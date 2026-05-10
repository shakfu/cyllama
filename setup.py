#!/usr/bin/python3

import os
import sys
import platform
import subprocess
from setuptools import Extension, setup

from Cython.Build import cythonize

# -----------------------------------------------------------------------------
# constants

BUILD_CUDA = os.getenv("XLLAMACPP_BUILD_CUDA")
BUILD_HIP = os.getenv("XLLAMACPP_BUILD_HIP")
BUILD_VULKAN = os.getenv("XLLAMACPP_BUILD_VULKAN")
NAME = "xllamacpp"
# NAME = "xllamacpp-cuda12x" if BUILD_CUDA else "xllamacpp"
CWD = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, CWD)
import versioneer

VERSION = versioneer.get_version()

PLATFORM = platform.system()

LLAMACPP_LIBS_DIR = os.path.join(CWD, "src/llama.cpp/lib")

# ABI3+ (Limited API) support for Python 3.10+
PY_LIMITED_API_VERSION = 0x030A0000  # Python 3.10

DEFINE_MACROS = [("Py_LIMITED_API", PY_LIMITED_API_VERSION)]
if PLATFORM == "Windows":
    EXTRA_COMPILE_ARGS = ["/std:c++17"]
else:
    EXTRA_COMPILE_ARGS = ["-std=c++17", "-fvisibility=hidden", "-fvisibility-inlines-hidden"]
    if PLATFORM == "Darwin":
        EXTRA_COMPILE_ARGS.append("-mmacosx-version-min=12.0")
EXTRA_LINK_ARGS = []
EXTRA_OBJECTS = []
INCLUDE_DIRS = [
    "src/xllamacpp",
    os.path.join(CWD, "thirdparty/llama.cpp/include"),
    os.path.join(CWD, "thirdparty/llama.cpp/common"),
    os.path.join(CWD, "thirdparty/llama.cpp/ggml/include"),
    os.path.join(
        CWD, "thirdparty/llama.cpp"
    ),  # For including 'common/base64.hpp' in server/utils.hpp
    os.path.join(
        CWD, "thirdparty/llama.cpp/build/tools/server"
    ),  # For including index.html.gz.hpp and loading.html.hpp
    os.path.join(CWD, "thirdparty/llama.cpp/tools/server"),
    os.path.join(CWD, "thirdparty/llama.cpp/tools/mtmd"),
    os.path.join(CWD, "thirdparty/llama.cpp/vendor"),
]
LIBRARY_DIRS = [
    LLAMACPP_LIBS_DIR,
]
LIBRARIES = []

if PLATFORM == "Windows":
    LIBRARIES.extend(
        [
            "llama-common-base",
            "llama-common",
            "llama",
            "ggml",
            "ggml-base",
            "ggml-cpu",
            "mtmd",
            "cpp-httplib",
            "server-context",
            "llguidance",
            "ssl",
            "crypto",
            "Advapi32",
            "userenv",
            "ntdll",
        ]
    )
    # Note: Windows builds use LLAMA_BUILD_BORINGSSL=ON which statically links OpenSSL
    # Add BoringSSL static libraries for proper symbol resolution
    if BUILD_CUDA:
        LIBRARY_DIRS.extend([os.getenv("CUDA_PATH", "") + "\\Lib\\x64"])
        LIBRARIES.extend(["ggml-cuda", "cudart", "cublas", "cublasLt", "cuda"])
    if BUILD_VULKAN:
        LIBRARY_DIRS.extend([os.getenv("VULKAN_SDK", "") + "\\Lib"])
        LIBRARIES.extend(["ggml-vulkan", "vulkan-1"])
else:
    LIBRARIES.extend(["pthread"])
    # Order matters for static linking: dependents before dependencies.
    # libssl.a/libcrypto.a must come AFTER libraries that reference OpenSSL symbols
    # (e.g., libcpp-httplib.a, libserver-context.a).
    EXTRA_OBJECTS.extend(
        [
            f"{LLAMACPP_LIBS_DIR}/libserver-context.a",
            f"{LLAMACPP_LIBS_DIR}/libcpp-httplib.a",
            f"{LLAMACPP_LIBS_DIR}/libmtmd.a",
            f"{LLAMACPP_LIBS_DIR}/libllama-common-base.a",
            f"{LLAMACPP_LIBS_DIR}/libllama-common.a",
            f"{LLAMACPP_LIBS_DIR}/libllguidance.a",
            f"{LLAMACPP_LIBS_DIR}/libllama.a",
            f"{LLAMACPP_LIBS_DIR}/libggml.a",
            f"{LLAMACPP_LIBS_DIR}/libggml-cpu.a",
            f"{LLAMACPP_LIBS_DIR}/libggml-base.a",
            # BoringSSL static libraries must be last (they are dependencies, not dependents)
            f"{LLAMACPP_LIBS_DIR}/libssl.a",
            f"{LLAMACPP_LIBS_DIR}/libcrypto.a",
        ]
    )
    if BUILD_CUDA:
        EXTRA_OBJECTS.extend(
            [
                f"{LLAMACPP_LIBS_DIR}/libggml-cuda.a",
            ]
        )
        LIBRARY_DIRS.extend(
            [
                os.getenv("CUDA_PATH", "") + "/lib/stubs",
                os.getenv("CUDA_PATH", "") + "/lib",
            ],
        )
        LIBRARIES.extend(["cudart", "cublas", "cublasLt", "cuda"])
    if BUILD_HIP:
        EXTRA_OBJECTS.extend(
            [
                f"{LLAMACPP_LIBS_DIR}/libggml-hip.a",
            ]
        )
        LIBRARY_DIRS.extend(["/opt/rocm/lib"])
        LIBRARIES.extend(["amdhip64", "hipblas", "rocblas"])
    if BUILD_VULKAN:
        EXTRA_OBJECTS.extend(
            [
                f"{LLAMACPP_LIBS_DIR}/libggml-vulkan.a",
            ]
        )
        LIBRARIES.extend(["vulkan"])

if PLATFORM == "Darwin":
    EXTRA_LINK_ARGS.append("-Wl,-rpath," + LLAMACPP_LIBS_DIR)
    os.environ["LDFLAGS"] = " ".join(
        [
            "-framework Accelerate",
            "-framework Foundation",
            "-framework Metal",
            "-framework MetalKit",
        ]
    )
    # Both the Intel and ARM platforms need to be linked with BLAS.
    EXTRA_OBJECTS.extend(
        [
            f"{LLAMACPP_LIBS_DIR}/libggml-blas.a",
        ]
    )
    if platform.processor() == "arm":
        EXTRA_OBJECTS.extend(
            [
                f"{LLAMACPP_LIBS_DIR}/libggml-metal.a",
            ]
        )
elif PLATFORM == "Linux":
    # Do not statically link to libstdc++; this will cause compatibility issues.
    # gcc-toolset-14's libgomp.a is not built with -fPIC on both x86_64 and aarch64,
    # so we dynamically link libgomp and exclude it via auditwheel instead.
    EXTRA_LINK_ARGS.extend(["-fopenmp", "-static-libgcc"])
    # Check if BLAS is enabled in environment
    if os.path.exists(f"{LLAMACPP_LIBS_DIR}/libggml-blas.a"):
        print("BLAS is enabled, adding ggml-blas to link targets")
        EXTRA_OBJECTS.extend([f"{LLAMACPP_LIBS_DIR}/libggml-blas.a"])
        EXTRA_LINK_ARGS.extend(["-lopenblas"])

INCLUDE_DIRS.append(os.path.join(CWD, "src/xllamacpp"))


def mk_extension(name, sources, define_macros=None):
    return Extension(
        name=name,
        sources=sources,
        define_macros=DEFINE_MACROS + (define_macros if define_macros else []),
        include_dirs=INCLUDE_DIRS,
        libraries=LIBRARIES,
        library_dirs=LIBRARY_DIRS,
        extra_objects=EXTRA_OBJECTS,
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
        language="c++",
        py_limited_api=True,
    )


# ----------------------------------------------------------------------------
# COMMON SETUP CONFIG

common = {
    "name": NAME,
    "version": VERSION,
    "description": "A cython wrapper of the llama.cpp inference engine.",
    "python_requires": ">=3.10",
    "cmdclass": versioneer.get_cmdclass(),
    "license": "MIT",
    # "include_package_data": True,
}


# forces cythonize in this case
subprocess.call("cythonize *.pyx", cwd="src/xllamacpp", shell=True)

if not os.path.exists("MANIFEST.in"):
    with open("MANIFEST.in", "w") as f:
        f.write("exclude src/xllamacpp/*.pxd\n")
        f.write("exclude src/xllamacpp/*.pyx\n")
        f.write("exclude src/xllamacpp/*.cpp\n")
        f.write("exclude src/xllamacpp/*.h\n")
        f.write("exclude src/xllamacpp/py.typed\n")

extensions = [
    mk_extension(
        "xllamacpp.xllamacpp",
        sources=[
            "src/xllamacpp/xllamacpp.pyx",
            "src/xllamacpp/server.cpp",
            "thirdparty/llama.cpp/tools/server/server-models.cpp",
            "thirdparty/llama.cpp/tools/server/server-http.cpp",
        ],
    ),
]

setup(
    **common,
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "embedsignature": False,  # default: False
            "emit_code_comments": False,  # default: True
            "warn.unused": True,  # default: False
            "binding": True,  # Required for ABI3+
        },
    ),
    package_dir={"": "src"},
    options={
        "bdist_wheel": {
            "py_limited_api": "cp310",
        }
    },
)
