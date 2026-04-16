cmake_minimum_required(VERSION 3.21...3.30)
project(cyllama LANGUAGES C CXX)

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------

option(WITH_WHISPER "Build whisper.cpp wrapper" ON)
option(WITH_STABLEDIFFUSION "Build stable-diffusion.cpp wrapper" ON)
option(WITH_DYLIB "Use dynamic libraries instead of static" OFF)
option(SD_USE_VENDORED_GGML "Link stable-diffusion against its own vendored ggml" ON)

# stable-diffusion.cpp requires GGML_MAX_NAME=128 for long tensor names.
# When SD shares llama.cpp's ggml, both must use the same value or the
# ggml_tensor struct layout diverges.  Propagate to all targets.
if(NOT SD_USE_VENDORED_GGML)
    add_definitions(-DGGML_MAX_NAME=128)
endif()

# When WITH_DYLIB=ON, point this to a directory containing pre-built dylibs
# (e.g., from a llama.cpp release tarball). Defaults to thirdparty/llama.cpp/lib.
set(LLAMACPP_DYLIB_DIR "" CACHE PATH "Path to pre-built llama.cpp shared libraries")

# Backend options - set defaults based on platform
if(APPLE)
    set(_default_metal ON)
else()
    set(_default_metal OFF)
endif()

option(GGML_METAL "Enable Metal backend (macOS)" ${_default_metal})
option(GGML_CUDA "Enable CUDA backend" OFF)
option(GGML_VULKAN "Enable Vulkan backend" OFF)
option(GGML_SYCL "Enable SYCL backend" OFF)
option(GGML_HIP "Enable HIP/ROCm backend" OFF)
option(GGML_OPENCL "Enable OpenCL backend" OFF)

# Read from environment - override defaults
if(DEFINED ENV{WITH_WHISPER})
    if("$ENV{WITH_WHISPER}" STREQUAL "0")
        set(WITH_WHISPER OFF)
    endif()
endif()

if(DEFINED ENV{WITH_STABLEDIFFUSION})
    if("$ENV{WITH_STABLEDIFFUSION}" STREQUAL "0")
        set(WITH_STABLEDIFFUSION OFF)
    endif()
endif()

if(DEFINED ENV{GGML_METAL})
    if("$ENV{GGML_METAL}" STREQUAL "1")
        set(GGML_METAL ON CACHE BOOL "Enable Metal backend (macOS)" FORCE)
    elseif("$ENV{GGML_METAL}" STREQUAL "0")
        set(GGML_METAL OFF CACHE BOOL "Enable Metal backend (macOS)" FORCE)
    endif()
endif()

if(DEFINED ENV{GGML_CUDA})
    if("$ENV{GGML_CUDA}" STREQUAL "1")
        set(GGML_CUDA ON CACHE BOOL "Enable CUDA backend" FORCE)
    elseif("$ENV{GGML_CUDA}" STREQUAL "0")
        set(GGML_CUDA OFF CACHE BOOL "Enable CUDA backend" FORCE)
    endif()
endif()

if(DEFINED ENV{GGML_VULKAN})
    if("$ENV{GGML_VULKAN}" STREQUAL "1")
        set(GGML_VULKAN ON CACHE BOOL "Enable Vulkan backend" FORCE)
    elseif("$ENV{GGML_VULKAN}" STREQUAL "0")
        set(GGML_VULKAN OFF CACHE BOOL "Enable Vulkan backend" FORCE)
    endif()
endif()

if(DEFINED ENV{GGML_HIP})
    if("$ENV{GGML_HIP}" STREQUAL "1")
        set(GGML_HIP ON CACHE BOOL "Enable HIP/ROCm backend" FORCE)
    elseif("$ENV{GGML_HIP}" STREQUAL "0")
        set(GGML_HIP OFF CACHE BOOL "Enable HIP/ROCm backend" FORCE)
    endif()
endif()

if(DEFINED ENV{GGML_SYCL})
    if("$ENV{GGML_SYCL}" STREQUAL "1")
        set(GGML_SYCL ON CACHE BOOL "Enable SYCL backend" FORCE)
    elseif("$ENV{GGML_SYCL}" STREQUAL "0")
        set(GGML_SYCL OFF CACHE BOOL "Enable SYCL backend" FORCE)
    endif()
endif()

if(DEFINED ENV{GGML_OPENCL})
    if("$ENV{GGML_OPENCL}" STREQUAL "1")
        set(GGML_OPENCL ON CACHE BOOL "Enable OpenCL backend" FORCE)
    elseif("$ENV{GGML_OPENCL}" STREQUAL "0")
        set(GGML_OPENCL OFF CACHE BOOL "Enable OpenCL backend" FORCE)
    endif()
endif()

if(DEFINED ENV{WITH_DYLIB})
    if("$ENV{WITH_DYLIB}" STREQUAL "1")
        set(WITH_DYLIB ON CACHE BOOL "Use dynamic libraries instead of static" FORCE)
    endif()
endif()

if(DEFINED ENV{SD_USE_VENDORED_GGML})
    if("$ENV{SD_USE_VENDORED_GGML}" STREQUAL "0")
        set(SD_USE_VENDORED_GGML OFF CACHE BOOL "Link stable-diffusion against its own vendored ggml" FORCE)
    else()
        set(SD_USE_VENDORED_GGML ON CACHE BOOL "Link stable-diffusion against its own vendored ggml" FORCE)
    endif()
endif()

if(DEFINED ENV{LLAMACPP_DYLIB_DIR})
    set(LLAMACPP_DYLIB_DIR "$ENV{LLAMACPP_DYLIB_DIR}" CACHE PATH "Path to pre-built llama.cpp shared libraries" FORCE)
endif()

# -----------------------------------------------------------------------------
# Find dependencies
# -----------------------------------------------------------------------------

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Add cmake module path for cython-cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/scripts/cmake")

find_package(Cython MODULE REQUIRED VERSION 3.0)
include(UseCython)

# -----------------------------------------------------------------------------
# Platform detection
# -----------------------------------------------------------------------------

if(APPLE)
    set(IS_MACOS TRUE)
    set(STATIC_LIB_PREFIX "lib")
    set(STATIC_LIB_SUFFIX ".a")
elseif(WIN32)
    set(IS_WINDOWS TRUE)
    set(STATIC_LIB_PREFIX "")
    set(STATIC_LIB_SUFFIX ".lib")
else()
    set(IS_LINUX TRUE)
    set(STATIC_LIB_PREFIX "lib")
    set(STATIC_LIB_SUFFIX ".a")
endif()

# Print configuration
message(STATUS "Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "WITH_WHISPER: ${WITH_WHISPER}")
message(STATUS "WITH_STABLEDIFFUSION: ${WITH_STABLEDIFFUSION}")
message(STATUS "GGML_METAL: ${GGML_METAL}")
message(STATUS "GGML_CUDA: ${GGML_CUDA}")
message(STATUS "GGML_VULKAN: ${GGML_VULKAN}")
message(STATUS "GGML_HIP: ${GGML_HIP}")
message(STATUS "GGML_SYCL: ${GGML_SYCL}")
message(STATUS "GGML_OPENCL: ${GGML_OPENCL}")

# -----------------------------------------------------------------------------
# Library paths
# -----------------------------------------------------------------------------

set(LLAMACPP_DIR "${CMAKE_SOURCE_DIR}/thirdparty/llama.cpp")
set(LLAMACPP_INCLUDE "${LLAMACPP_DIR}/include")
set(LLAMACPP_LIB "${LLAMACPP_DIR}/lib")

set(WHISPERCPP_DIR "${CMAKE_SOURCE_DIR}/thirdparty/whisper.cpp")
set(WHISPERCPP_INCLUDE "${WHISPERCPP_DIR}/include")
set(WHISPERCPP_LIB "${WHISPERCPP_DIR}/lib")

set(SDCPP_DIR "${CMAKE_SOURCE_DIR}/thirdparty/stable-diffusion.cpp")
set(SDCPP_INCLUDE "${SDCPP_DIR}/include")
set(SDCPP_LIB "${SDCPP_DIR}/lib")

# Helper function to get static library path
function(static_lib result lib_dir name)
    set(${result} "${lib_dir}/${STATIC_LIB_PREFIX}${name}${STATIC_LIB_SUFFIX}" PARENT_SCOPE)
endfunction()

# -----------------------------------------------------------------------------
# Common settings
# -----------------------------------------------------------------------------

set(COMMON_INCLUDE_DIRS
    "${CMAKE_SOURCE_DIR}/src/cyllama"
    "${CMAKE_SOURCE_DIR}/src/cyllama/llama"
    "${CMAKE_SOURCE_DIR}/src/cyllama/llama/helpers"
    "${CMAKE_SOURCE_DIR}/src/cyllama/llama/server"
    "${LLAMACPP_INCLUDE}"
    "${CMAKE_SOURCE_DIR}/include"
)

set(COMMON_LINK_DIRS
    "${LLAMACPP_LIB}"
)

if(WITH_WHISPER)
    list(APPEND COMMON_LINK_DIRS "${WHISPERCPP_LIB}")
endif()

if(WITH_STABLEDIFFUSION)
    list(APPEND COMMON_LINK_DIRS "${SDCPP_LIB}")
endif()

# =============================================================================
# Library collection (static or dynamic)
# =============================================================================

set(STATIC_LIBS "")
set(DYLIBS "")       # shared libraries to link (WITH_DYLIB mode)
set(DYLIB_FILES "")  # shared library files to copy alongside extensions

if(WITH_DYLIB)
    # -------------------------------------------------------------------------
    # Dynamic linking: link against pre-built shared libraries
    # -------------------------------------------------------------------------
    if(NOT LLAMACPP_DYLIB_DIR)
        set(LLAMACPP_DYLIB_DIR "${LLAMACPP_DIR}/dynamic")
    endif()
    if(NOT EXISTS "${LLAMACPP_DYLIB_DIR}")
        message(FATAL_ERROR
            "WITH_DYLIB=ON but LLAMACPP_DYLIB_DIR does not exist: ${LLAMACPP_DYLIB_DIR}\n"
            "Either build shared libs first (make build-shared) or set LLAMACPP_DYLIB_DIR "
            "to a directory containing pre-built llama.cpp shared libraries.")
    endif()
    file(GLOB _dylib_contents "${LLAMACPP_DYLIB_DIR}/*")
    if(NOT _dylib_contents)
        message(FATAL_ERROR
            "WITH_DYLIB=ON but LLAMACPP_DYLIB_DIR is empty: ${LLAMACPP_DYLIB_DIR}\n"
            "The directory exists but contains no files. Build shared libs first or "
            "provide a directory with pre-built llama.cpp shared libraries.")
    endif()
    message(STATUS "WITH_DYLIB: ON (dylib dir: ${LLAMACPP_DYLIB_DIR})")

    # Find shared libraries in the dylib directory
    # On macOS: .dylib, on Linux: .so
    if(APPLE)
        set(_dylib_ext "dylib")
    else()
        set(_dylib_ext "so")
    endif()

    # Core llama.cpp shared libraries (available in release tarballs)
    set(_DYLIB_NAMES llama ggml ggml-base mtmd)
    # Optional core libraries (not always present in release tarballs or self-builds)
    set(_OPTIONAL_DYLIB_NAMES ggml-rpc)

    # ggml-cpu: with GGML_BACKEND_DL=ON and GGML_NATIVE=OFF, llama.cpp
    # may produce arch-specific variants (e.g. libggml-cpu-haswell.so,
    # libggml-cpu-x64.dylib) instead of a base libggml-cpu. This happens
    # on both Linux (release tarballs) and macOS (Vulkan/non-Metal builds).
    # All variants are installed for ggml's runtime dispatch, but we must
    # link against the most portable one for symbol resolution.
    file(GLOB _cpu_variants "${LLAMACPP_DYLIB_DIR}/libggml-cpu-*.${_dylib_ext}")
    if(_cpu_variants)
        # Install all CPU variants for runtime dispatch
        list(APPEND DYLIB_FILES ${_cpu_variants})
        # Link against the most portable variant for symbol resolution.
        # Preference order: base ggml-cpu > x64 > sse42 > first available.
        set(_cpu_link_lib "")
        find_library(_cpu_base ggml-cpu PATHS "${LLAMACPP_DYLIB_DIR}" NO_DEFAULT_PATH)
        if(_cpu_base)
            set(_cpu_link_lib "${_cpu_base}")
        else()
            foreach(_candidate x64 sse42)
                find_library(_cpu_portable ggml-cpu-${_candidate}
                    PATHS "${LLAMACPP_DYLIB_DIR}" NO_DEFAULT_PATH)
                if(_cpu_portable)
                    set(_cpu_link_lib "${_cpu_portable}")
                    break()
                endif()
                unset(_cpu_portable CACHE)
            endforeach()
        endif()
        if(NOT _cpu_link_lib)
            # Last resort: pick the first variant (may crash on incompatible CPUs)
            list(GET _cpu_variants 0 _cpu_link_lib)
            message(WARNING "No portable CPU variant (x64/sse42) found, linking against ${_cpu_link_lib}")
        endif()
        list(APPEND DYLIBS "${_cpu_link_lib}")
        list(APPEND DYLIB_FILES "${_cpu_link_lib}")
        get_filename_component(_link_name "${_cpu_link_lib}" NAME)
        message(STATUS "  CPU link variant: ${_link_name}")
        foreach(_v ${_cpu_variants})
            get_filename_component(_vn ${_v} NAME)
            message(STATUS "  Found CPU variant: ${_vn}")
        endforeach()
    else()
        # Fallback: try the base name (self-built with GGML_NATIVE=ON
        # or Metal-default builds typically produce libggml-cpu directly)
        list(APPEND _DYLIB_NAMES ggml-cpu)
    endif()

    # Backend-specific shared libraries (required when their backend is enabled)
    set(_BACKEND_DYLIB_NAMES "")
    if(GGML_METAL)
        list(APPEND _BACKEND_DYLIB_NAMES ggml-blas ggml-metal)
    endif()
    if(GGML_CUDA)
        list(APPEND _BACKEND_DYLIB_NAMES ggml-cuda)
    endif()
    if(GGML_VULKAN)
        list(APPEND _BACKEND_DYLIB_NAMES ggml-vulkan)
    endif()
    if(GGML_SYCL)
        list(APPEND _BACKEND_DYLIB_NAMES ggml-sycl)
    endif()
    if(GGML_HIP)
        list(APPEND _BACKEND_DYLIB_NAMES ggml-hip)
    endif()
    if(GGML_OPENCL)
        list(APPEND _BACKEND_DYLIB_NAMES ggml-opencl)
    endif()

    # Helper macro: find a shared lib, append to DYLIBS/DYLIB_FILES, or handle missing
    macro(_find_dylib _name _is_required)
        find_library(_found_${_name}
            NAMES ${_name}
            PATHS "${LLAMACPP_DYLIB_DIR}"
            NO_DEFAULT_PATH
        )
        if(_found_${_name})
            list(APPEND DYLIBS "${_found_${_name}}")
            # Install the soname file -- that's what @rpath / $ORIGIN references at runtime
            # macOS soname format: libllama.0.dylib
            # Linux soname format: libllama.so.0
            if(APPLE)
                file(GLOB _soname "${LLAMACPP_DYLIB_DIR}/lib${_name}.[0-9].${_dylib_ext}")
            else()
                file(GLOB _soname "${LLAMACPP_DYLIB_DIR}/lib${_name}.${_dylib_ext}.[0-9]")
            endif()
            if(_soname)
                list(APPEND DYLIB_FILES ${_soname})
            else()
                list(APPEND DYLIB_FILES "${_found_${_name}}")
            endif()
            message(STATUS "  Found shared lib: ${_found_${_name}}")
        else()
            if(${_is_required})
                message(FATAL_ERROR "  Required shared lib not found: lib${_name}.${_dylib_ext} in ${LLAMACPP_DYLIB_DIR}")
            else()
                message(WARNING "  Optional shared lib not found: lib${_name}.${_dylib_ext} in ${LLAMACPP_DYLIB_DIR}")
            endif()
        endif()
    endmacro()

    # Find core libraries (FATAL_ERROR if missing)
    foreach(_name ${_DYLIB_NAMES})
        _find_dylib(${_name} TRUE)
    endforeach()

    # Find optional core libraries (WARNING if missing)
    foreach(_name ${_OPTIONAL_DYLIB_NAMES})
        _find_dylib(${_name} FALSE)
    endforeach()

    # Find backend libraries (FATAL_ERROR if missing — they were explicitly enabled)
    foreach(_name ${_BACKEND_DYLIB_NAMES})
        _find_dylib(${_name} TRUE)
    endforeach()

    # cpp-httplib: not in release tarballs, needed for embedded server
    static_lib(LIB_CPP_HTTPLIB "${LLAMACPP_LIB}" "cpp-httplib")
    if(EXISTS "${LIB_CPP_HTTPLIB}")
        list(APPEND STATIC_LIBS "${LIB_CPP_HTTPLIB}")
        message(STATUS "  Static lib (embedded server): ${LIB_CPP_HTTPLIB}")
    endif()

    # Add dylib directory to link search path
    list(APPEND COMMON_LINK_DIRS "${LLAMACPP_DYLIB_DIR}")

else()
    # -------------------------------------------------------------------------
    # Static linking: original behavior
    # -------------------------------------------------------------------------
    message(STATUS "WITH_DYLIB: OFF (static linking)")

    static_lib(LIB_LLAMA "${LLAMACPP_LIB}" "llama")
    static_lib(LIB_GGML "${LLAMACPP_LIB}" "ggml")
    static_lib(LIB_GGML_BASE "${LLAMACPP_LIB}" "ggml-base")
    static_lib(LIB_GGML_CPU "${LLAMACPP_LIB}" "ggml-cpu")
    static_lib(LIB_MTMD "${LLAMACPP_LIB}" "mtmd")

    list(APPEND STATIC_LIBS
        "${LIB_LLAMA}"
        "${LIB_GGML}"
        "${LIB_GGML_BASE}"
        "${LIB_GGML_CPU}"
        "${LIB_MTMD}"
    )

    # cpp-httplib: optional, only needed for embedded server
    static_lib(LIB_CPP_HTTPLIB "${LLAMACPP_LIB}" "cpp-httplib")
    if(EXISTS "${LIB_CPP_HTTPLIB}")
        list(APPEND STATIC_LIBS "${LIB_CPP_HTTPLIB}")
        message(STATUS "  Static lib (embedded server): ${LIB_CPP_HTTPLIB}")
    endif()

    # Backend-specific static libraries
    if(GGML_METAL)
        static_lib(LIB_GGML_BLAS "${LLAMACPP_LIB}" "ggml-blas")
        static_lib(LIB_GGML_METAL "${LLAMACPP_LIB}" "ggml-metal")
        list(APPEND STATIC_LIBS "${LIB_GGML_BLAS}" "${LIB_GGML_METAL}")
    endif()

    if(GGML_CUDA)
        static_lib(LIB_GGML_CUDA "${LLAMACPP_LIB}" "ggml-cuda")
        list(APPEND STATIC_LIBS "${LIB_GGML_CUDA}")
    endif()

    if(GGML_VULKAN)
        static_lib(LIB_GGML_VULKAN "${LLAMACPP_LIB}" "ggml-vulkan")
        list(APPEND STATIC_LIBS "${LIB_GGML_VULKAN}")
    endif()

    if(GGML_SYCL)
        static_lib(LIB_GGML_SYCL "${LLAMACPP_LIB}" "ggml-sycl")
        list(APPEND STATIC_LIBS "${LIB_GGML_SYCL}")
    endif()

    if(GGML_HIP)
        static_lib(LIB_GGML_HIP "${LLAMACPP_LIB}" "ggml-hip")
        list(APPEND STATIC_LIBS "${LIB_GGML_HIP}")
    endif()

    if(GGML_OPENCL)
        static_lib(LIB_GGML_OPENCL "${LLAMACPP_LIB}" "ggml-opencl")
        list(APPEND STATIC_LIBS "${LIB_GGML_OPENCL}")
    endif()
endif()

# Whisper: static libs are always needed, but ggml comes from dylibs when available
if(WITH_WHISPER)
    static_lib(LIB_WHISPER_COMMON "${WHISPERCPP_LIB}" "common")
    static_lib(LIB_WHISPER "${WHISPERCPP_LIB}" "whisper")
    if(NOT WITH_DYLIB)
        # Static build: add whisper libs globally (linked into every extension)
        list(APPEND STATIC_LIBS "${LIB_WHISPER_COMMON}" "${LIB_WHISPER}")
    endif()
endif()

if(WITH_STABLEDIFFUSION)
    static_lib(LIB_SD "${SDCPP_LIB}" "stable-diffusion")
endif()

# System libraries
set(SYSTEM_LIBS "")

if(NOT WIN32)
    list(APPEND SYSTEM_LIBS pthread)
endif()

if(WIN32)
    list(APPEND SYSTEM_LIBS kernel32 user32 advapi32)
endif()

if(GGML_METAL)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)
    list(APPEND SYSTEM_LIBS
        ${ACCELERATE_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${METAL_FRAMEWORK}
        ${METALKIT_FRAMEWORK}
    )
endif()

# OpenSSL linking removed: EmbeddedServer doesn't expose SSL config,
# and absolute homebrew paths break wheel portability.

if(GGML_CUDA)
    find_package(CUDAToolkit REQUIRED)
    list(APPEND SYSTEM_LIBS CUDA::cudart CUDA::cublas CUDA::cuda_driver)
endif()

if(GGML_VULKAN)
    find_package(Vulkan REQUIRED)
    list(APPEND SYSTEM_LIBS Vulkan::Vulkan)
endif()

if(GGML_HIP)
    # HIP/ROCm backend requires hip, hipblas, and rocblas
    # Note: Only link hip::host (runtime), NOT hip::device (which adds HIP compiler flags)
    # The actual HIP kernels are in libggml-hip.a which was compiled with hipcc
    find_package(hip REQUIRED)
    find_package(hipblas REQUIRED)
    find_package(rocblas REQUIRED)
    # hip::host provides the HIP runtime without requiring HIP compilation
    # amdhip64 is the actual runtime library
    list(APPEND SYSTEM_LIBS hip::host roc::rocblas roc::hipblas)
endif()

if(GGML_SYCL)
    # SYCL backend - add -fsycl to link flags only (not compile flags).
    # Using IntelSYCL::SYCL_CXX adds -fsycl globally which forces C files
    # to be compiled as C++, causing duplicate symbol errors with mongoose.
    # Instead, we add -fsycl only at link time to preserve SYCL device code
    # from libggml-sycl.a, and link libsycl.so for the runtime.
    find_library(_sycl_lib NAMES sycl
        PATHS "$ENV{CMPLR_ROOT}/lib" "/opt/intel/oneapi/compiler/latest/lib")
    if(_sycl_lib)
        list(APPEND SYSTEM_LIBS "${_sycl_lib}")
    endif()
    add_link_options(-fsycl)
    message(STATUS "SYCL backend: linking with -fsycl and ${_sycl_lib}")
endif()

if(GGML_OPENCL)
    # OpenCL backend
    find_package(OpenCL REQUIRED)
    list(APPEND SYSTEM_LIBS ${OpenCL_LIBRARIES})
endif()

if(IS_LINUX)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        list(APPEND SYSTEM_LIBS OpenMP::OpenMP_CXX)
    endif()
endif()

# -----------------------------------------------------------------------------
# Compile options
# -----------------------------------------------------------------------------

if(WIN32)
    set(CXX_COMPILE_OPTIONS /std:c++17 /EHsc /MD)
else()
    set(CXX_COMPILE_OPTIONS -std=c++17 -fvisibility=hidden -fvisibility-inlines-hidden)
endif()

# -----------------------------------------------------------------------------
# Cython include paths (for .pxd files)
# -----------------------------------------------------------------------------

set(CYTHON_INCLUDE_DIRS
    "${CMAKE_SOURCE_DIR}/src/cyllama"
    "${CMAKE_SOURCE_DIR}/src/cyllama/llama"
    "${CMAKE_SOURCE_DIR}/src/cyllama/llama/server"
    "${CMAKE_SOURCE_DIR}/src/cyllama/whisper"
    "${CMAKE_SOURCE_DIR}/src/cyllama/sd"
)

# Build Cython include args
set(CYTHON_INCLUDE_ARGS "")
foreach(dir ${CYTHON_INCLUDE_DIRS})
    list(APPEND CYTHON_INCLUDE_ARGS "-I${dir}")
endforeach()

# -----------------------------------------------------------------------------
# Helper function to create Cython extension
# -----------------------------------------------------------------------------

function(add_cython_extension target_name pyx_file)
    cmake_parse_arguments(ARG "" "DESTINATION" "SOURCES;INCLUDE_DIRS;COMPILE_OPTIONS" ${ARGN})

    # Transpile Cython to C++
    cython_transpile(${pyx_file} LANGUAGE CXX OUTPUT_VARIABLE cython_cpp
        CYTHON_ARGS ${CYTHON_INCLUDE_ARGS}
    )

    # Create the module
    python_add_library(${target_name} MODULE WITH_SOABI
        ${cython_cpp}
        ${ARG_SOURCES}
    )

    # Set include directories
    target_include_directories(${target_name} PRIVATE
        ${COMMON_INCLUDE_DIRS}
        ${ARG_INCLUDE_DIRS}
    )

    # Link directories
    target_link_directories(${target_name} PRIVATE ${COMMON_LINK_DIRS})

    # Link libraries
    if(WITH_DYLIB)
        # Dynamic linking: shared libs + any remaining static libs
        target_link_libraries(${target_name} PRIVATE
            ${DYLIBS}
            ${STATIC_LIBS}
            ${SYSTEM_LIBS}
        )
    elseif(UNIX AND NOT APPLE)
        # Static linking on Linux: use whole-archive for ggml libs
        set(_GGML_LIBS ${LIB_GGML} ${LIB_GGML_BASE} ${LIB_GGML_CPU})
        if(GGML_METAL)
            list(APPEND _GGML_LIBS ${LIB_GGML_BLAS} ${LIB_GGML_METAL})
        endif()
        if(GGML_CUDA)
            list(APPEND _GGML_LIBS ${LIB_GGML_CUDA})
        endif()
        if(GGML_VULKAN)
            list(APPEND _GGML_LIBS ${LIB_GGML_VULKAN})
        endif()
        if(GGML_SYCL)
            list(APPEND _GGML_LIBS ${LIB_GGML_SYCL})
        endif()
        if(GGML_HIP)
            list(APPEND _GGML_LIBS ${LIB_GGML_HIP})
        endif()
        if(GGML_OPENCL)
            list(APPEND _GGML_LIBS ${LIB_GGML_OPENCL})
        endif()

        set(_OTHER_LIBS ${LIB_LLAMA} ${LIB_MTMD})
        if(EXISTS "${LIB_CPP_HTTPLIB}")
            list(APPEND _OTHER_LIBS ${LIB_CPP_HTTPLIB})
        endif()
        if(WITH_WHISPER)
            list(APPEND _OTHER_LIBS ${LIB_WHISPER_COMMON} ${LIB_WHISPER})
        endif()

        set(_WHOLE_ARCHIVE_LIBS ${_GGML_LIBS})
        if(EXISTS "${LIB_CPP_HTTPLIB}")
            list(APPEND _WHOLE_ARCHIVE_LIBS ${LIB_CPP_HTTPLIB})
        endif()

        target_link_libraries(${target_name} PRIVATE
            -Wl,--whole-archive ${_WHOLE_ARCHIVE_LIBS} -Wl,--no-whole-archive
            ${_OTHER_LIBS}
            ${SYSTEM_LIBS}
        )
    else()
        # Static linking on macOS/Windows
        target_link_libraries(${target_name} PRIVATE ${STATIC_LIBS} ${SYSTEM_LIBS})
    endif()

    # Compile options
    if(ARG_COMPILE_OPTIONS)
        target_compile_options(${target_name} PRIVATE ${ARG_COMPILE_OPTIONS})
    else()
        target_compile_options(${target_name} PRIVATE ${CXX_COMPILE_OPTIONS})
    endif()

    # Set C++ standard
    set_target_properties(${target_name} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    # RPATH settings
    set(_build_rpath "${LLAMACPP_LIB};${WHISPERCPP_LIB};${SDCPP_LIB}")
    if(WITH_DYLIB AND LLAMACPP_DYLIB_DIR)
        set(_build_rpath "${LLAMACPP_DYLIB_DIR};${_build_rpath}")
    endif()

    if(APPLE)
        set_target_properties(${target_name} PROPERTIES
            BUILD_RPATH "${_build_rpath}"
            INSTALL_RPATH "@loader_path"
        )
    elseif(UNIX)
        set_target_properties(${target_name} PROPERTIES
            BUILD_RPATH "${_build_rpath}"
            INSTALL_RPATH "$ORIGIN"
        )
    endif()

    # Install
    if(ARG_DESTINATION)
        install(TARGETS ${target_name} DESTINATION ${ARG_DESTINATION})
    else()
        install(TARGETS ${target_name} DESTINATION cyllama)
    endif()
endfunction()

# -----------------------------------------------------------------------------
# Build extensions
# -----------------------------------------------------------------------------

# llama_cpp extension
# JSON schema -> GBNF grammar conversion is now handled by pure Python
# (src/cyllama/json_schema_to_grammar.py), eliminating the need for
# json-schema-to-grammar.cpp, json-partial.cpp, and json_schema.cpp.
set(_LLAMA_CPP_SOURCES
    "${CMAKE_SOURCE_DIR}/src/cyllama/llama/helpers/tts.cpp"
)

add_cython_extension(llama_cpp
    "${CMAKE_SOURCE_DIR}/src/cyllama/llama/llama_cpp.pyx"
    SOURCES ${_LLAMA_CPP_SOURCES}
    DESTINATION "cyllama/llama"
)

# embedded server extension
cython_transpile("${CMAKE_SOURCE_DIR}/src/cyllama/llama/server/embedded.pyx"
    LANGUAGE CXX
    OUTPUT_VARIABLE embedded_cpp
    CYTHON_ARGS ${CYTHON_INCLUDE_ARGS}
)

python_add_library(embedded MODULE WITH_SOABI
    ${embedded_cpp}
    "${CMAKE_SOURCE_DIR}/src/cyllama/llama/server/mongoose.c"
    "${CMAKE_SOURCE_DIR}/src/cyllama/llama/server/mongoose_wrapper.c"
)

target_include_directories(embedded PRIVATE
    ${COMMON_INCLUDE_DIRS}
    "${CMAKE_SOURCE_DIR}/src/cyllama/llama/server"
)
target_link_directories(embedded PRIVATE ${COMMON_LINK_DIRS})
if(UNIX AND NOT APPLE)
    # Reuse the same selective whole-archive approach
    set(_GGML_LIBS ${LIB_GGML} ${LIB_GGML_BASE} ${LIB_GGML_CPU})
    if(GGML_METAL)
        list(APPEND _GGML_LIBS ${LIB_GGML_BLAS} ${LIB_GGML_METAL})
    endif()
    if(GGML_CUDA)
        list(APPEND _GGML_LIBS ${LIB_GGML_CUDA})
    endif()
    if(GGML_VULKAN)
        list(APPEND _GGML_LIBS ${LIB_GGML_VULKAN})
    endif()
    if(GGML_SYCL)
        list(APPEND _GGML_LIBS ${LIB_GGML_SYCL})
    endif()
    if(GGML_HIP)
        list(APPEND _GGML_LIBS ${LIB_GGML_HIP})
    endif()
    if(GGML_OPENCL)
        list(APPEND _GGML_LIBS ${LIB_GGML_OPENCL})
    endif()

    # Note: SD libs are NOT included here - SD extension links its own ggml
    set(_OTHER_LIBS ${LIB_LLAMA} ${LIB_MTMD})
    if(WITH_WHISPER)
        list(APPEND _OTHER_LIBS ${LIB_WHISPER_COMMON} ${LIB_WHISPER})
    endif()

    target_link_libraries(embedded PRIVATE
        -Wl,--whole-archive ${_GGML_LIBS} -Wl,--no-whole-archive
        ${_OTHER_LIBS}
        ${SYSTEM_LIBS}
    )
else()
    target_link_libraries(embedded PRIVATE ${STATIC_LIBS} ${SYSTEM_LIBS})
endif()
set_target_properties(embedded PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

if(APPLE)
    set_target_properties(embedded PROPERTIES
        BUILD_RPATH "${LLAMACPP_LIB};${WHISPERCPP_LIB};${SDCPP_LIB}"
        INSTALL_RPATH "@loader_path"
    )
elseif(UNIX)
    set_target_properties(embedded PROPERTIES
        BUILD_RPATH "${LLAMACPP_LIB};${WHISPERCPP_LIB};${SDCPP_LIB}"
        INSTALL_RPATH "$ORIGIN"
    )
endif()

install(TARGETS embedded DESTINATION "cyllama/llama/server")

# whisper_cpp extension
if(WITH_WHISPER)
    if(WITH_DYLIB)
        # Dynamic: link whisper-specific code statically, reuse llama.cpp's ggml dylibs
        cython_transpile("${CMAKE_SOURCE_DIR}/src/cyllama/whisper/whisper_cpp.pyx"
            LANGUAGE CXX
            OUTPUT_VARIABLE whisper_cpp_src
            CYTHON_ARGS ${CYTHON_INCLUDE_ARGS}
        )

        python_add_library(whisper_cpp MODULE WITH_SOABI ${whisper_cpp_src})

        # Whisper needs its own include dir + llama.cpp's (for shared ggml headers).
        # Do NOT add sd include dirs — their duplicate ggml headers conflict
        # via #pragma once (path-based guard).
        target_include_directories(whisper_cpp PRIVATE
            "${CMAKE_SOURCE_DIR}/src/cyllama"
            "${CMAKE_SOURCE_DIR}/src/cyllama/llama"
            "${LLAMACPP_INCLUDE}"
            "${WHISPERCPP_INCLUDE}"
            "${CMAKE_SOURCE_DIR}/include"
        )
        target_link_directories(whisper_cpp PRIVATE ${COMMON_LINK_DIRS})

        target_link_libraries(whisper_cpp PRIVATE
            ${LIB_WHISPER}
            ${LIB_WHISPER_COMMON}
            ${DYLIBS}
            ${SYSTEM_LIBS}
        )

        target_compile_options(whisper_cpp PRIVATE ${CXX_COMPILE_OPTIONS})
        set_target_properties(whisper_cpp PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
        )

        # Dylibs live in cyllama/llama/, whisper installs to cyllama/whisper/
        if(APPLE)
            set_target_properties(whisper_cpp PROPERTIES
                BUILD_RPATH "${LLAMACPP_DYLIB_DIR};${WHISPERCPP_LIB}"
                INSTALL_RPATH "@loader_path/../llama"
            )
        elseif(UNIX)
            set_target_properties(whisper_cpp PROPERTIES
                BUILD_RPATH "${LLAMACPP_DYLIB_DIR};${WHISPERCPP_LIB}"
                INSTALL_RPATH "$ORIGIN/../llama"
            )
        endif()

        install(TARGETS whisper_cpp DESTINATION "cyllama/whisper")
    else()
        add_cython_extension(whisper_cpp
            "${CMAKE_SOURCE_DIR}/src/cyllama/whisper/whisper_cpp.pyx"
            INCLUDE_DIRS "${WHISPERCPP_INCLUDE}"
            DESTINATION "cyllama/whisper"
        )
    endif()
endif()

# stable_diffusion extension
if(WITH_STABLEDIFFUSION)
    cython_transpile("${CMAKE_SOURCE_DIR}/src/cyllama/sd/stable_diffusion.pyx"
        LANGUAGE CXX
        OUTPUT_VARIABLE sd_cpp
        CYTHON_ARGS ${CYTHON_INCLUDE_ARGS}
    )

    python_add_library(stable_diffusion MODULE WITH_SOABI
        ${sd_cpp}
        "${CMAKE_SOURCE_DIR}/src/cyllama/sd/stb_impl.cpp"
    )

    # SD needs its own include dir + llama.cpp's (for shared ggml headers).
    # Do NOT add whisper include dirs — their duplicate ggml headers conflict
    # via #pragma once (path-based guard).
    target_include_directories(stable_diffusion PRIVATE
        "${CMAKE_SOURCE_DIR}/src/cyllama"
        "${CMAKE_SOURCE_DIR}/src/cyllama/llama"
        "${LLAMACPP_INCLUDE}"
        "${SDCPP_INCLUDE}"
        "${CMAKE_SOURCE_DIR}/include"
    )

    target_compile_options(stable_diffusion PRIVATE ${CXX_COMPILE_OPTIONS})
    set_target_properties(stable_diffusion PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    if(WITH_DYLIB AND NOT SD_USE_VENDORED_GGML)
        # Dynamic: link libstable-diffusion.a statically, reuse llama.cpp's ggml dylibs
        target_link_directories(stable_diffusion PRIVATE "${SDCPP_LIB}" ${COMMON_LINK_DIRS})
        target_link_libraries(stable_diffusion PRIVATE
            "${LIB_SD}"
            ${DYLIBS}
            ${SYSTEM_LIBS}
        )

        # Dylibs live in cyllama/llama/, SD installs to cyllama/sd/
        if(APPLE)
            set_target_properties(stable_diffusion PROPERTIES
                BUILD_RPATH "${LLAMACPP_DYLIB_DIR};${SDCPP_LIB}"
                INSTALL_RPATH "@loader_path/../llama"
            )
        elseif(UNIX)
            set_target_properties(stable_diffusion PROPERTIES
                BUILD_RPATH "${LLAMACPP_DYLIB_DIR};${SDCPP_LIB}"
                INSTALL_RPATH "$ORIGIN/../llama"
            )
        endif()
    else()
        # Static (or dynamic + SD_USE_VENDORED_GGML): choose ggml source
        target_link_directories(stable_diffusion PRIVATE "${SDCPP_LIB}" ${COMMON_LINK_DIRS})

        if(SD_USE_VENDORED_GGML)
            # Use stable-diffusion.cpp's own vendored ggml libs
            set(SD_GGML_LIB_DIR "${SDCPP_LIB}")
        else()
            # Default: use llama.cpp's ggml (unified across all backends)
            set(SD_GGML_LIB_DIR "${LLAMACPP_LIB}")
        endif()

        static_lib(SD_LIB_GGML "${SD_GGML_LIB_DIR}" "ggml")
        static_lib(SD_LIB_GGML_BASE "${SD_GGML_LIB_DIR}" "ggml-base")
        static_lib(SD_LIB_GGML_CPU "${SD_GGML_LIB_DIR}" "ggml-cpu")
        set(SD_GGML_LIBS ${SD_LIB_GGML} ${SD_LIB_GGML_BASE} ${SD_LIB_GGML_CPU})
        if(GGML_METAL)
            static_lib(SD_LIB_GGML_BLAS "${SD_GGML_LIB_DIR}" "ggml-blas")
            static_lib(SD_LIB_GGML_METAL "${SD_GGML_LIB_DIR}" "ggml-metal")
            list(APPEND SD_GGML_LIBS ${SD_LIB_GGML_BLAS} ${SD_LIB_GGML_METAL})
        endif()
        if(GGML_CUDA)
            static_lib(SD_LIB_GGML_CUDA "${SD_GGML_LIB_DIR}" "ggml-cuda")
            list(APPEND SD_GGML_LIBS ${SD_LIB_GGML_CUDA})
        endif()
        if(GGML_VULKAN)
            static_lib(SD_LIB_GGML_VULKAN "${SD_GGML_LIB_DIR}" "ggml-vulkan")
            list(APPEND SD_GGML_LIBS ${SD_LIB_GGML_VULKAN})
        endif()
        if(GGML_HIP)
            static_lib(SD_LIB_GGML_HIP "${SD_GGML_LIB_DIR}" "ggml-hip")
            list(APPEND SD_GGML_LIBS ${SD_LIB_GGML_HIP})
        endif()
        if(GGML_SYCL)
            static_lib(SD_LIB_GGML_SYCL "${SD_GGML_LIB_DIR}" "ggml-sycl")
            list(APPEND SD_GGML_LIBS ${SD_LIB_GGML_SYCL})
        endif()
        if(GGML_OPENCL)
            static_lib(SD_LIB_GGML_OPENCL "${SD_GGML_LIB_DIR}" "ggml-opencl")
            list(APPEND SD_GGML_LIBS ${SD_LIB_GGML_OPENCL})
        endif()

        if(UNIX AND NOT APPLE)
            target_link_libraries(stable_diffusion PRIVATE
                -Wl,--whole-archive ${SD_GGML_LIBS} -Wl,--no-whole-archive
                "${LIB_SD}"
                ${SYSTEM_LIBS}
            )
        else()
            target_link_libraries(stable_diffusion PRIVATE
                "${LIB_SD}" ${SD_GGML_LIBS} ${SYSTEM_LIBS}
            )
        endif()

        if(APPLE)
            set_target_properties(stable_diffusion PROPERTIES
                BUILD_RPATH "${SD_GGML_LIB_DIR};${SDCPP_LIB}"
                INSTALL_RPATH "@loader_path"
            )
        elseif(UNIX)
            set_target_properties(stable_diffusion PROPERTIES
                BUILD_RPATH "${SD_GGML_LIB_DIR};${SDCPP_LIB}"
                INSTALL_RPATH "$ORIGIN"
            )
        endif()
    endif()

    install(TARGETS stable_diffusion DESTINATION "cyllama/sd")
endif()

# -----------------------------------------------------------------------------
# sqlite-vector extension (vendored source, built as shared library)
# -----------------------------------------------------------------------------

set(_VECTOR_SRC_DIR "${CMAKE_SOURCE_DIR}/thirdparty/sqlite-vector")
set(_VECTOR_SOURCES
    "${_VECTOR_SRC_DIR}/sqlite-vector.c"
    "${_VECTOR_SRC_DIR}/distance-cpu.c"
    "${_VECTOR_SRC_DIR}/distance-avx2.c"
    "${_VECTOR_SRC_DIR}/distance-avx512.c"
    "${_VECTOR_SRC_DIR}/distance-sse2.c"
    "${_VECTOR_SRC_DIR}/distance-neon.c"
    "${_VECTOR_SRC_DIR}/distance-rvv.c"
)

add_library(vector SHARED ${_VECTOR_SOURCES})
target_include_directories(vector PRIVATE "${_VECTOR_SRC_DIR}")
if(MSVC)
    target_compile_options(vector PRIVATE /O2 /W3)
else()
    target_compile_options(vector PRIVATE -O3 -fPIC -Wall -Wextra -Wno-unused-parameter)
endif()

# No lib prefix — output is vector.so / vector.dylib / vector.dll
set_target_properties(vector PROPERTIES
    PREFIX ""
    C_STANDARD 11
)

if(APPLE)
    set_target_properties(vector PROPERTIES
        SUFFIX ".dylib"
        LINK_FLAGS "-dynamiclib -undefined dynamic_lookup"
    )
elseif(WIN32)
    set_target_properties(vector PROPERTIES SUFFIX ".dll")
else()
    set_target_properties(vector PROPERTIES SUFFIX ".so")
endif()

install(TARGETS vector DESTINATION "cyllama/rag")

# -----------------------------------------------------------------------------
# Install shared libraries (WITH_DYLIB mode)
# -----------------------------------------------------------------------------

if(WITH_DYLIB AND DYLIB_FILES)
    # Install dylibs alongside the extension so $ORIGIN / @loader_path resolves.
    # For wheel builds on Linux, auditwheel will later vendor and patch these
    # into <pkg>.libs/, but editable installs need them in place.
    install(FILES ${DYLIB_FILES} DESTINATION "cyllama/llama")
    foreach(_f ${DYLIB_FILES})
        get_filename_component(_n "${_f}" NAME)
        message(STATUS "  dylib: ${_n}")
    endforeach()
endif()

# -----------------------------------------------------------------------------
# Install Python files
# -----------------------------------------------------------------------------

# The wheel.packages setting in pyproject.toml handles the Python files
