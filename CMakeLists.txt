cmake_minimum_required(VERSION 3.21...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX)

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------

option(WITH_WHISPER "Build whisper.cpp wrapper" ON)
option(WITH_STABLEDIFFUSION "Build stable-diffusion.cpp wrapper" ON)
option(WITH_DYLIB "Use dynamic libraries instead of static" OFF)

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
    list(APPEND COMMON_INCLUDE_DIRS "${WHISPERCPP_INCLUDE}")
    list(APPEND COMMON_LINK_DIRS "${WHISPERCPP_LIB}")
endif()

if(WITH_STABLEDIFFUSION)
    list(APPEND COMMON_INCLUDE_DIRS "${SDCPP_INCLUDE}")
    list(APPEND COMMON_LINK_DIRS "${SDCPP_LIB}")
endif()

# Collect static libraries to link
set(STATIC_LIBS "")

static_lib(LIB_COMMON "${LLAMACPP_LIB}" "common")
static_lib(LIB_LLAMA "${LLAMACPP_LIB}" "llama")
static_lib(LIB_GGML "${LLAMACPP_LIB}" "ggml")
static_lib(LIB_GGML_BASE "${LLAMACPP_LIB}" "ggml-base")
static_lib(LIB_GGML_CPU "${LLAMACPP_LIB}" "ggml-cpu")
static_lib(LIB_MTMD "${LLAMACPP_LIB}" "mtmd")

list(APPEND STATIC_LIBS
    "${LIB_COMMON}"
    "${LIB_LLAMA}"
    "${LIB_GGML}"
    "${LIB_GGML_BASE}"
    "${LIB_GGML_CPU}"
    "${LIB_MTMD}"
)

if(WITH_WHISPER)
    static_lib(LIB_WHISPER_COMMON "${WHISPERCPP_LIB}" "common")
    static_lib(LIB_WHISPER "${WHISPERCPP_LIB}" "whisper")
    list(APPEND STATIC_LIBS "${LIB_WHISPER_COMMON}" "${LIB_WHISPER}")
endif()

if(WITH_STABLEDIFFUSION)
    # Define LIB_SD here but don't add to STATIC_LIBS - SD extension has its own linking
    static_lib(LIB_SD "${SDCPP_LIB}" "stable-diffusion")
endif()

# Backend-specific libraries
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
    # SYCL backend - try to find IntelSYCL (from Intel oneAPI)
    # Note: SYCL builds require a SYCL-capable compiler (icpx from oneAPI)
    find_package(IntelSYCL QUIET)
    if(IntelSYCL_FOUND)
        list(APPEND SYSTEM_LIBS IntelSYCL::SYCL_CXX)
        message(STATUS "Found IntelSYCL")
    else()
        message(STATUS "IntelSYCL not found - SYCL runtime linking may need manual configuration")
    endif()
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

    # Link static libraries
    # On Linux, use whole-archive selectively for ggml libs to ensure all symbols
    # are included (needed for symbols like ggml_repeat_4d called via function pointers)
    # Only ggml libs need this - applying to all libs causes duplicate symbol errors
    if(UNIX AND NOT APPLE)
        # Separate ggml libs (need whole-archive) from other libs
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

        # Other libs that don't need whole-archive
        # Note: SD libs are NOT included here - SD extension links its own ggml
        set(_OTHER_LIBS ${LIB_COMMON} ${LIB_LLAMA} ${LIB_MTMD})
        if(WITH_WHISPER)
            list(APPEND _OTHER_LIBS ${LIB_WHISPER_COMMON} ${LIB_WHISPER})
        endif()

        target_link_libraries(${target_name} PRIVATE
            -Wl,--whole-archive ${_GGML_LIBS} -Wl,--no-whole-archive
            ${_OTHER_LIBS}
            ${SYSTEM_LIBS}
        )
    else()
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

    # macOS-specific settings
    if(APPLE)
        set_target_properties(${target_name} PROPERTIES
            BUILD_RPATH "${LLAMACPP_LIB};${WHISPERCPP_LIB};${SDCPP_LIB}"
            INSTALL_RPATH "@loader_path"
        )
    elseif(UNIX)
        set_target_properties(${target_name} PROPERTIES
            BUILD_RPATH "${LLAMACPP_LIB};${WHISPERCPP_LIB};${SDCPP_LIB}"
            INSTALL_RPATH "$ORIGIN"
        )
    endif()

    # Install
    if(ARG_DESTINATION)
        install(TARGETS ${target_name} DESTINATION ${ARG_DESTINATION})
    else()
        install(TARGETS ${target_name} DESTINATION ${SKBUILD_PROJECT_NAME})
    endif()
endfunction()

# -----------------------------------------------------------------------------
# Build extensions
# -----------------------------------------------------------------------------

# llama_cpp extension
add_cython_extension(llama_cpp
    "${CMAKE_SOURCE_DIR}/src/cyllama/llama/llama_cpp.pyx"
    SOURCES
        "${CMAKE_SOURCE_DIR}/src/cyllama/llama/helpers/tts.cpp"
        "${CMAKE_SOURCE_DIR}/src/cyllama/llama/helpers/json_schema.cpp"
    DESTINATION "${SKBUILD_PROJECT_NAME}/llama"
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
    set(_OTHER_LIBS ${LIB_COMMON} ${LIB_LLAMA} ${LIB_MTMD})
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

install(TARGETS embedded DESTINATION "${SKBUILD_PROJECT_NAME}/llama/server")

# whisper_cpp extension
if(WITH_WHISPER)
    add_cython_extension(whisper_cpp
        "${CMAKE_SOURCE_DIR}/src/cyllama/whisper/whisper_cpp.pyx"
        DESTINATION "${SKBUILD_PROJECT_NAME}/whisper"
    )
endif()

# stable_diffusion extension - uses SD's own ggml libraries, not llama.cpp's
if(WITH_STABLEDIFFUSION)
    # Transpile Cython to C++
    cython_transpile("${CMAKE_SOURCE_DIR}/src/cyllama/sd/stable_diffusion.pyx"
        LANGUAGE CXX
        OUTPUT_VARIABLE sd_cpp
        CYTHON_ARGS ${CYTHON_INCLUDE_ARGS}
    )

    python_add_library(stable_diffusion MODULE WITH_SOABI
        ${sd_cpp}
        "${CMAKE_SOURCE_DIR}/src/cyllama/sd/stb_impl.cpp"
    )

    target_include_directories(stable_diffusion PRIVATE
        ${COMMON_INCLUDE_DIRS}
    )

    target_link_directories(stable_diffusion PRIVATE "${SDCPP_LIB}")

    # SD-specific ggml libraries (from stable-diffusion.cpp build)
    static_lib(LIB_SD_GGML "${SDCPP_LIB}" "ggml")
    static_lib(LIB_SD_GGML_BASE "${SDCPP_LIB}" "ggml-base")
    static_lib(LIB_SD_GGML_CPU "${SDCPP_LIB}" "ggml-cpu")

    set(SD_STATIC_LIBS
        "${LIB_SD}"
        "${LIB_SD_GGML}"
        "${LIB_SD_GGML_BASE}"
        "${LIB_SD_GGML_CPU}"
    )

    # Backend-specific ggml libraries for SD
    # Note: SD has its own Metal support that may differ from llama.cpp's GGML_METAL setting
    # We check if SD's ggml-metal library exists to determine if SD was built with Metal
    static_lib(_SD_METAL_LIB_PATH "${SDCPP_LIB}" "ggml-metal")
    if(EXISTS "${_SD_METAL_LIB_PATH}")
        message(STATUS "SD Metal library found: ${_SD_METAL_LIB_PATH}")
        static_lib(LIB_SD_GGML_BLAS "${SDCPP_LIB}" "ggml-blas")
        static_lib(LIB_SD_GGML_METAL "${SDCPP_LIB}" "ggml-metal")
        list(APPEND SD_STATIC_LIBS "${LIB_SD_GGML_BLAS}" "${LIB_SD_GGML_METAL}")
    else()
        message(STATUS "SD Metal library not found, SD will use CPU backend")
    endif()

    # Check for CUDA support in SD
    static_lib(_SD_CUDA_LIB_PATH "${SDCPP_LIB}" "ggml-cuda")
    if(EXISTS "${_SD_CUDA_LIB_PATH}")
        list(APPEND SD_STATIC_LIBS "${_SD_CUDA_LIB_PATH}")
    endif()

    # Check for Vulkan support in SD
    static_lib(_SD_VULKAN_LIB_PATH "${SDCPP_LIB}" "ggml-vulkan")
    if(EXISTS "${_SD_VULKAN_LIB_PATH}")
        list(APPEND SD_STATIC_LIBS "${_SD_VULKAN_LIB_PATH}")
    endif()

    # Check for HIP/ROCm support in SD
    static_lib(_SD_HIP_LIB_PATH "${SDCPP_LIB}" "ggml-hip")
    if(EXISTS "${_SD_HIP_LIB_PATH}")
        list(APPEND SD_STATIC_LIBS "${_SD_HIP_LIB_PATH}")
    endif()

    # Check for SYCL support in SD
    static_lib(_SD_SYCL_LIB_PATH "${SDCPP_LIB}" "ggml-sycl")
    if(EXISTS "${_SD_SYCL_LIB_PATH}")
        list(APPEND SD_STATIC_LIBS "${_SD_SYCL_LIB_PATH}")
    endif()

    # Check for OpenCL support in SD
    static_lib(_SD_OPENCL_LIB_PATH "${SDCPP_LIB}" "ggml-opencl")
    if(EXISTS "${_SD_OPENCL_LIB_PATH}")
        list(APPEND SD_STATIC_LIBS "${_SD_OPENCL_LIB_PATH}")
    endif()

    target_link_libraries(stable_diffusion PRIVATE ${SD_STATIC_LIBS} ${SYSTEM_LIBS})

    target_compile_options(stable_diffusion PRIVATE ${CXX_COMPILE_OPTIONS})

    set_target_properties(stable_diffusion PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    if(APPLE)
        set_target_properties(stable_diffusion PROPERTIES
            BUILD_RPATH "${SDCPP_LIB}"
            INSTALL_RPATH "@loader_path"
        )
    elseif(UNIX)
        set_target_properties(stable_diffusion PROPERTIES
            BUILD_RPATH "${SDCPP_LIB}"
            INSTALL_RPATH "$ORIGIN"
        )
    endif()

    install(TARGETS stable_diffusion DESTINATION "${SKBUILD_PROJECT_NAME}/sd")
endif()

# -----------------------------------------------------------------------------
# Install Python files
# -----------------------------------------------------------------------------

# The wheel.packages setting in pyproject.toml handles the Python files
