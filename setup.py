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

VERSION = '0.1.5'

PLATFORM = platform.system()

WITH_WHISPER = os.getenv("WITH_WHISPER", True)

WITH_DYLIB = os.getenv("WITH_DYLIB", False)

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

if PLATFORM == 'Darwin':
    EXTRA_OBJECTS.extend([
        f'{LLAMACPP_LIBS_DIR}/libggml-blas.a',
        f'{LLAMACPP_LIBS_DIR}/libggml-metal.a',
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

    os.environ['LDFLAGS'] = ' '.join([
        '-framework Accelerate',
        '-framework Foundation',
        '-framework Metal',
        '-framework MetalKit',
    ])

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
        # "build/llama.cpp/tools/server/server.cpp",
    ]),
    Extension(
        name="cyllama.llama.server.mongoose_server",
        sources=[
            "src/cyllama/llama/server/mongoose_server.pyx",
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

