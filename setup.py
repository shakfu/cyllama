#!/usr/bin/python3

import os
import platform
import subprocess
from setuptools import Extension, setup

from Cython.Build import cythonize

# -----------------------------------------------------------------------------
# constants

CWD = os.getcwd()
# VENDOR_DIR = os.path.join(CWD, "build/llama.cpp/vendor")
# SERVER_PUBLIC_DIR = os.path.join(CWD, "build/llama.cpp/build/tools/server")

VERSION = '0.1.10'

PLATFORM = platform.system()

WITH_WHISPER = os.getenv("WITH_WHISPER", True)

WITH_DYLIB = os.getenv("WITH_DYLIB", False)

# -----------------------------------------------------------------------------
# Backend detection helpers

def detect_cuda():
    """Check if CUDA toolkit is available."""
    try:
        result = subprocess.run(["nvcc", "--version"],
                               capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def detect_vulkan():
    """Check if Vulkan SDK is available."""
    vulkan_headers = [
        "/usr/include/vulkan/vulkan.h",
        "/usr/local/include/vulkan/vulkan.h",
        "/opt/vulkan/include/vulkan/vulkan.h",
    ]
    return any(os.path.exists(path) for path in vulkan_headers)

def detect_sycl():
    """Check if Intel oneAPI/SYCL is available."""
    return os.path.exists("/opt/intel/oneapi")

def detect_rocm():
    """Check if ROCm/HIP is available."""
    return os.path.exists("/opt/rocm")

def detect_metal():
    """Check if Metal is available (macOS only)."""
    if PLATFORM != 'Darwin':
        return False
    try:
        result = subprocess.run(["xcrun", "--sdk", "macosx", "--show-sdk-path"],
                               capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

# Print backend detection info
print("Backend detection:")
print(f"  CUDA available:    {detect_cuda()}")
print(f"  Vulkan available:  {detect_vulkan()}")
print(f"  SYCL available:    {detect_sycl()}")
print(f"  ROCm/HIP available: {detect_rocm()}")
print(f"  Metal available:   {detect_metal()}")

# Backend flags (read from environment)
GGML_METAL = os.getenv("GGML_METAL", "1") == "1"
GGML_CUDA = os.getenv("GGML_CUDA", "0") == "1"
GGML_VULKAN = os.getenv("GGML_VULKAN", "0") == "1"
GGML_SYCL = os.getenv("GGML_SYCL", "0") == "1"
GGML_HIP = os.getenv("GGML_HIP", "0") == "1"
GGML_OPENCL = os.getenv("GGML_OPENCL", "0") == "1"

print("\nEnabled backends:")
if GGML_METAL:
    print("  ✓ Metal")
if GGML_CUDA:
    print("  ✓ CUDA")
if GGML_VULKAN:
    print("  ✓ Vulkan")
if GGML_SYCL:
    print("  ✓ SYCL")
if GGML_HIP:
    print("  ✓ HIP/ROCm")
if GGML_OPENCL:
    print("  ✓ OpenCL")
if not any([GGML_METAL, GGML_CUDA, GGML_VULKAN, GGML_SYCL, GGML_HIP, GGML_OPENCL]):
    print("  CPU only (no GPU acceleration)")

LLAMACPP_INCLUDE = os.path.join(CWD, "thirdparty/llama.cpp/include")
LLAMACPP_LIBS_DIR = os.path.join(CWD, "thirdparty/llama.cpp/lib")

WHISPERCPP_INCLUDE = os.path.join(CWD, "thirdparty/whisper.cpp/include")
WHISPERCPP_LIBS_DIR = os.path.join(CWD, "thirdparty/whisper.cpp/lib")

DEFINE_MACROS = []
EXTRA_COMPILE_ARGS = ['-std=c++17']
EXTRA_LINK_ARGS = []
EXTRA_OBJECTS = []
INCLUDE_DIRS = [
    "src/cyllama",
    "src/cyllama/llama/helpers",
    "src/cyllama/llama/server",
    LLAMACPP_INCLUDE,
    # VENDOR_DIR,
    # SERVER_PUBLIC_DIR,
]
LIBRARY_DIRS = [
    LLAMACPP_LIBS_DIR,
]
LIBRARIES = ["pthread"]

if WITH_WHISPER:
    INCLUDE_DIRS.extend([
        WHISPERCPP_INCLUDE,
    ])
    LIBRARY_DIRS.extend([
        WHISPERCPP_LIBS_DIR,
    ])

if WITH_DYLIB:
    EXTRA_OBJECTS.append(f'{LLAMACPP_LIBS_DIR}/libcommon.a')
    LIBRARIES.extend([
        'common',
        'ggml',
        'llama',
    ])
    if WITH_WHISPER:
        LIBRARIES.extend([
            "whisper",
        ])

else:
    EXTRA_OBJECTS.extend([
        f'{LLAMACPP_LIBS_DIR}/libcommon.a',
        f'{LLAMACPP_LIBS_DIR}/libllama.a', 
        f'{LLAMACPP_LIBS_DIR}/libggml.a',
        f'{LLAMACPP_LIBS_DIR}/libggml-base.a',
        f'{LLAMACPP_LIBS_DIR}/libggml-cpu.a',
        f'{LLAMACPP_LIBS_DIR}/libmtmd.a',
    ])

    if WITH_WHISPER:
        EXTRA_OBJECTS.extend([
            f'{WHISPERCPP_LIBS_DIR}/libcommon.a',
            f'{WHISPERCPP_LIBS_DIR}/libwhisper.a',
        ])


INCLUDE_DIRS.append(os.path.join(CWD, 'include'))

# Platform-specific configurations
if PLATFORM == 'Darwin':
    # macOS-specific backends
    if GGML_METAL:
        EXTRA_OBJECTS.extend([
            f'{LLAMACPP_LIBS_DIR}/libggml-blas.a',
            f'{LLAMACPP_LIBS_DIR}/libggml-metal.a',
        ])
        os.environ['LDFLAGS'] = ' '.join([
            '-framework Accelerate',
            '-framework Foundation',
            '-framework Metal',
            '-framework MetalKit',
        ])

    EXTRA_LINK_ARGS.append('-mmacosx-version-min=14.7')
    # add local rpath
    EXTRA_LINK_ARGS.extend([
        '-Wl,-rpath,' + LLAMACPP_LIBS_DIR,
    ])

    if WITH_WHISPER:
        EXTRA_LINK_ARGS.extend([
            '-Wl,-rpath,' + WHISPERCPP_LIBS_DIR,
        ])

# Cross-platform backends
if GGML_CUDA:
    EXTRA_OBJECTS.append(f'{LLAMACPP_LIBS_DIR}/libggml-cuda.a')
    LIBRARIES.extend(['cuda', 'cudart', 'cublas'])
    # Try common CUDA paths
    cuda_paths = ['/usr/local/cuda/lib64', '/usr/local/cuda/lib', '/opt/cuda/lib64']
    for path in cuda_paths:
        if os.path.exists(path):
            LIBRARY_DIRS.append(path)
            break

if GGML_VULKAN:
    EXTRA_OBJECTS.append(f'{LLAMACPP_LIBS_DIR}/libggml-vulkan.a')
    LIBRARIES.append('vulkan')

if GGML_SYCL:
    EXTRA_OBJECTS.append(f'{LLAMACPP_LIBS_DIR}/libggml-sycl.a')
    # SYCL libraries will be added based on Intel oneAPI setup
    sycl_path = '/opt/intel/oneapi/compiler/latest/lib'
    if os.path.exists(sycl_path):
        LIBRARY_DIRS.append(sycl_path)

if GGML_HIP:
    EXTRA_OBJECTS.append(f'{LLAMACPP_LIBS_DIR}/libggml-hip.a')
    LIBRARIES.extend(['amdhip64', 'rocblas'])
    # Try common ROCm paths
    rocm_paths = ['/opt/rocm/lib', '/opt/rocm/hip/lib']
    for path in rocm_paths:
        if os.path.exists(path):
            LIBRARY_DIRS.append(path)
            break

if GGML_OPENCL:
    EXTRA_OBJECTS.append(f'{LLAMACPP_LIBS_DIR}/libggml-opencl.a')
    LIBRARIES.append('OpenCL')

if PLATFORM == 'Linux':
    EXTRA_LINK_ARGS.append('-fopenmp')


def mk_extension(name, sources, define_macros=None, extra_compile_args=None, language="c++"):
    return Extension(
        name=name,
        sources=sources,
        define_macros=define_macros if define_macros else [],
        include_dirs=INCLUDE_DIRS,
        libraries=LIBRARIES,
        library_dirs=LIBRARY_DIRS,
        extra_objects=EXTRA_OBJECTS,
        extra_compile_args=extra_compile_args if extra_compile_args else EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
        language=language,
    )


# ----------------------------------------------------------------------------
# COMMON SETUP CONFIG

common = {
    "name": "cyllama",
    "version": VERSION,
    "description": "A cython wrapper of the llama.cpp inference engine.",
    "python_requires": ">=3.8",
    # "include_package_data": True,
}


# forces cythonize in this case
# subprocess.call("cythonize *.pyx", cwd="src/cyllama", shell=True)
subprocess.call("cythonize llama_cpp.pyx", cwd="src/cyllama/llama", shell=True)
if WITH_WHISPER:
    subprocess.call("cythonize whisper_cpp.pyx", cwd="src/cyllama/whisper", shell=True)


if not os.path.exists('MANIFEST.in'):
    with open("MANIFEST.in", "w") as f:
        f.write("exclude src/cyllama/*.pxd\n")
        f.write("exclude src/cyllama/*.pyx\n")
        f.write("exclude src/cyllama/*.cpp\n")
        f.write("exclude src/cyllama/*.h\n")
        f.write("exclude src/cyllama/py.typed\n")

extensions = [
    mk_extension("cyllama.llama.llama_cpp", sources=[
        "src/cyllama/llama/llama_cpp.pyx",
        "src/cyllama/llama/helpers/tts.cpp",
        "src/cyllama/llama/helpers/json_schema.cpp",
        # "build/llama.cpp/tools/server/server.cpp",
    ]),
    Extension(
        name="cyllama.llama.server.embedded",
        sources=[
            "src/cyllama/llama/server/embedded.pyx",
            "src/cyllama/llama/server/mongoose.c",
            "src/cyllama/llama/server/mongoose_wrapper.c",
        ],
        include_dirs=INCLUDE_DIRS,
        libraries=LIBRARIES,
        library_dirs=LIBRARY_DIRS,
        extra_objects=EXTRA_OBJECTS,
        extra_compile_args=[],  # No C++ flags for the C code
        extra_link_args=EXTRA_LINK_ARGS,
        language="c++",  # Overall extension language
    ),
]

if WITH_WHISPER:
    extensions.append(
        mk_extension("cyllama.whisper.whisper_cpp", sources=[
            "src/cyllama/whisper/whisper_cpp.pyx",
        ])
    )


setup(
    **common,
    ext_modules=cythonize(
        extensions,
        compiler_directives = {
            'language_level' : '3',
            'embedsignature': False,     # default: False
            'emit_code_comments': False, # default: True
            'warn.unused': True,         # default: False
        },
    ),
    package_dir={"": "src"},
    # gdb_debug=True,
)

