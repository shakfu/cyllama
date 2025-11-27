#!/usr/bin/python3

import os
import platform
import subprocess
import sys
from setuptools import Extension, setup

from Cython.Build import cythonize

# -----------------------------------------------------------------------------
# constants

CWD = os.getcwd()

VERSION = '0.1.10'

PLATFORM = platform.system()
IS_WINDOWS = PLATFORM == 'Windows'
IS_MACOS = PLATFORM == 'Darwin'
IS_LINUX = PLATFORM == 'Linux'

WITH_WHISPER = os.getenv("WITH_WHISPER", "1") == "1"
WITH_STABLEDIFFUSION = os.getenv("WITH_STABLEDIFFUSION", "1") == "1"
WITH_DYLIB = os.getenv("WITH_DYLIB", "0") == "1"

# Library file extensions per platform
if IS_WINDOWS:
    STATIC_LIB_EXT = '.lib'
    SHARED_LIB_EXT = '.dll'
else:
    STATIC_LIB_EXT = '.a'
    SHARED_LIB_EXT = '.dylib' if IS_MACOS else '.so'

# -----------------------------------------------------------------------------
# Backend detection helpers

def detect_cuda():
    """Check if CUDA toolkit is available."""
    if IS_WINDOWS:
        cuda_path = os.environ.get('CUDA_PATH', r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA')
        return os.path.exists(cuda_path)
    try:
        result = subprocess.run(["nvcc", "--version"],
                               capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def detect_vulkan():
    """Check if Vulkan SDK is available."""
    if IS_WINDOWS:
        vulkan_sdk = os.environ.get('VULKAN_SDK', r'C:\VulkanSDK')
        return os.path.exists(vulkan_sdk)
    vulkan_headers = [
        "/usr/include/vulkan/vulkan.h",
        "/usr/local/include/vulkan/vulkan.h",
        "/opt/vulkan/include/vulkan/vulkan.h",
    ]
    return any(os.path.exists(path) for path in vulkan_headers)

def detect_sycl():
    """Check if Intel oneAPI/SYCL is available."""
    if IS_WINDOWS:
        return os.path.exists(r'C:\Program Files (x86)\Intel\oneAPI')
    return os.path.exists("/opt/intel/oneapi")

def detect_rocm():
    """Check if ROCm/HIP is available."""
    if IS_WINDOWS:
        return False  # ROCm not available on Windows
    return os.path.exists("/opt/rocm")

def detect_metal():
    """Check if Metal is available (macOS only)."""
    if not IS_MACOS:
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
# Metal is only available on macOS, default enabled there
GGML_METAL = os.getenv("GGML_METAL", "1" if IS_MACOS else "0") == "1" and IS_MACOS
GGML_CUDA = os.getenv("GGML_CUDA", "0") == "1"
GGML_VULKAN = os.getenv("GGML_VULKAN", "0") == "1"
GGML_SYCL = os.getenv("GGML_SYCL", "0") == "1"
GGML_HIP = os.getenv("GGML_HIP", "0") == "1" and not IS_WINDOWS  # ROCm not on Windows
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

SDCPP_INCLUDE = os.path.join(CWD, "thirdparty/stable-diffusion.cpp/include")
SDCPP_LIBS_DIR = os.path.join(CWD, "thirdparty/stable-diffusion.cpp/lib")

DEFINE_MACROS = []

# Platform-specific compiler flags
if IS_WINDOWS:
    EXTRA_COMPILE_ARGS = ['/std:c++17', '/EHsc', '/MD']
else:
    EXTRA_COMPILE_ARGS = ['-std=c++17']

EXTRA_LINK_ARGS = []
EXTRA_OBJECTS = []
INCLUDE_DIRS = [
    "src/cyllama",
    "src/cyllama/llama/helpers",
    "src/cyllama/llama/server",
    LLAMACPP_INCLUDE,
]
LIBRARY_DIRS = [
    LLAMACPP_LIBS_DIR,
]

# Platform-specific libraries
if IS_WINDOWS:
    LIBRARIES = []  # Windows doesn't need pthread
else:
    LIBRARIES = ["pthread"]

if WITH_WHISPER:
    INCLUDE_DIRS.extend([
        WHISPERCPP_INCLUDE,
    ])
    LIBRARY_DIRS.extend([
        WHISPERCPP_LIBS_DIR,
    ])

if WITH_STABLEDIFFUSION:
    INCLUDE_DIRS.extend([
        SDCPP_INCLUDE,
    ])
    LIBRARY_DIRS.extend([
        SDCPP_LIBS_DIR,
    ])

def static_lib(lib_dir, name):
    """Get platform-appropriate static library path."""
    if IS_WINDOWS:
        # Windows uses name.lib (no 'lib' prefix)
        return os.path.join(lib_dir, f'{name}.lib')
    else:
        # Unix uses libname.a
        return os.path.join(lib_dir, f'lib{name}.a')

if WITH_DYLIB:
    EXTRA_OBJECTS.append(static_lib(LLAMACPP_LIBS_DIR, 'common'))
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
        static_lib(LLAMACPP_LIBS_DIR, 'common'),
        static_lib(LLAMACPP_LIBS_DIR, 'llama'),
        static_lib(LLAMACPP_LIBS_DIR, 'ggml'),
        static_lib(LLAMACPP_LIBS_DIR, 'ggml-base'),
        static_lib(LLAMACPP_LIBS_DIR, 'ggml-cpu'),
        static_lib(LLAMACPP_LIBS_DIR, 'mtmd'),
    ])

    if WITH_WHISPER:
        EXTRA_OBJECTS.extend([
            static_lib(WHISPERCPP_LIBS_DIR, 'common'),
            static_lib(WHISPERCPP_LIBS_DIR, 'whisper'),
        ])

    if WITH_STABLEDIFFUSION:
        EXTRA_OBJECTS.extend([
            static_lib(SDCPP_LIBS_DIR, 'stable-diffusion'),
        ])


INCLUDE_DIRS.append(os.path.join(CWD, 'include'))

# Platform-specific configurations
if IS_MACOS:
    # macOS-specific backends
    if GGML_METAL:
        EXTRA_OBJECTS.extend([
            static_lib(LLAMACPP_LIBS_DIR, 'ggml-blas'),
            static_lib(LLAMACPP_LIBS_DIR, 'ggml-metal'),
        ])
        os.environ['LDFLAGS'] = ' '.join([
            '-framework Accelerate',
            '-framework Foundation',
            '-framework Metal',
            '-framework MetalKit',
        ])

    # macOS deployment target
    macos_target = os.getenv('MACOSX_DEPLOYMENT_TARGET', '11.0')
    EXTRA_LINK_ARGS.append(f'-mmacosx-version-min={macos_target}')
    # add local rpath
    EXTRA_LINK_ARGS.extend([
        '-Wl,-rpath,' + LLAMACPP_LIBS_DIR,
    ])

    if WITH_WHISPER:
        EXTRA_LINK_ARGS.extend([
            '-Wl,-rpath,' + WHISPERCPP_LIBS_DIR,
        ])

    if WITH_STABLEDIFFUSION:
        EXTRA_LINK_ARGS.extend([
            '-Wl,-rpath,' + SDCPP_LIBS_DIR,
        ])

elif IS_LINUX:
    # Linux-specific configuration
    EXTRA_LINK_ARGS.append('-fopenmp')
    # Add rpath for shared libraries
    EXTRA_LINK_ARGS.extend([
        '-Wl,-rpath,$ORIGIN',
        '-Wl,-rpath,' + LLAMACPP_LIBS_DIR,
    ])
    if WITH_WHISPER:
        EXTRA_LINK_ARGS.extend([
            '-Wl,-rpath,' + WHISPERCPP_LIBS_DIR,
        ])

    if WITH_STABLEDIFFUSION:
        EXTRA_LINK_ARGS.extend([
            '-Wl,-rpath,' + SDCPP_LIBS_DIR,
        ])

elif IS_WINDOWS:
    # Windows-specific configuration
    # Add Windows system libraries that may be needed
    LIBRARIES.extend(['kernel32', 'user32', 'advapi32'])

# Cross-platform backends
if GGML_CUDA:
    EXTRA_OBJECTS.append(static_lib(LLAMACPP_LIBS_DIR, 'ggml-cuda'))
    LIBRARIES.extend(['cuda', 'cudart', 'cublas'])
    # Try common CUDA paths
    if IS_WINDOWS:
        cuda_path = os.environ.get('CUDA_PATH', r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0')
        cuda_paths = [os.path.join(cuda_path, 'lib', 'x64')]
    else:
        cuda_paths = ['/usr/local/cuda/lib64', '/usr/local/cuda/lib', '/opt/cuda/lib64']
    for path in cuda_paths:
        if os.path.exists(path):
            LIBRARY_DIRS.append(path)
            break

if GGML_VULKAN:
    EXTRA_OBJECTS.append(static_lib(LLAMACPP_LIBS_DIR, 'ggml-vulkan'))
    LIBRARIES.append('vulkan')
    if IS_WINDOWS:
        vulkan_sdk = os.environ.get('VULKAN_SDK', '')
        if vulkan_sdk:
            LIBRARY_DIRS.append(os.path.join(vulkan_sdk, 'Lib'))

if GGML_SYCL:
    EXTRA_OBJECTS.append(static_lib(LLAMACPP_LIBS_DIR, 'ggml-sycl'))
    # SYCL libraries will be added based on Intel oneAPI setup
    if IS_WINDOWS:
        sycl_path = r'C:\Program Files (x86)\Intel\oneAPI\compiler\latest\lib'
    else:
        sycl_path = '/opt/intel/oneapi/compiler/latest/lib'
    if os.path.exists(sycl_path):
        LIBRARY_DIRS.append(sycl_path)

if GGML_HIP:
    EXTRA_OBJECTS.append(static_lib(LLAMACPP_LIBS_DIR, 'ggml-hip'))
    LIBRARIES.extend(['amdhip64', 'rocblas'])
    # Try common ROCm paths (Linux only)
    rocm_paths = ['/opt/rocm/lib', '/opt/rocm/hip/lib']
    for path in rocm_paths:
        if os.path.exists(path):
            LIBRARY_DIRS.append(path)
            break

if GGML_OPENCL:
    EXTRA_OBJECTS.append(static_lib(LLAMACPP_LIBS_DIR, 'ggml-opencl'))
    LIBRARIES.append('OpenCL')


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


# Force cythonize - cross-platform compatible
def run_cythonize(pyx_file, cwd):
    """Run cythonize in a cross-platform way."""
    cmd = [sys.executable, '-m', 'cython', '--cplus', pyx_file]
    subprocess.call(cmd, cwd=cwd)

run_cythonize("llama_cpp.pyx", cwd="src/cyllama/llama")
if WITH_WHISPER:
    run_cythonize("whisper_cpp.pyx", cwd="src/cyllama/whisper")
if WITH_STABLEDIFFUSION:
    run_cythonize("stable_diffusion.pyx", cwd="src/cyllama/stablediffusion")


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

if WITH_STABLEDIFFUSION:
    extensions.append(
        mk_extension("cyllama.stablediffusion.stable_diffusion", sources=[
            "src/cyllama/stablediffusion/stable_diffusion.pyx",
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

