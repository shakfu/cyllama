#!/bin/sh

# scripts/build-llama-bins.sh
#
# Builds llama.cpp CLI binaries from existing source in build/llama.cpp
# Run this AFTER running setup.sh or make to build cyllama
#
# Usage:
#   ./scripts/build-llama-bins.sh          # Build all binaries
#   ./scripts/build-llama-bins.sh clean    # Clean and rebuild
#
# Binaries are installed to: thirdparty/llama.cpp/bin/

set -e

CWD=$(pwd)
BUILD_DIR="${CWD}/build/llama.cpp"
PREFIX="${CWD}/thirdparty/llama.cpp"
BIN_DIR="${PREFIX}/bin"

# Detect OS
OS_TYPE=$(uname -s)
IS_MACOS=0
CMAKE_BUILD_SUBDIR=""

case "$OS_TYPE" in
    Darwin)
        IS_MACOS=1
        METAL_DEFAULT=1
        ;;
    MINGW*|MSYS*|CYGWIN*)
        METAL_DEFAULT=0
        CMAKE_BUILD_SUBDIR="/Release"
        ;;
    *)
        METAL_DEFAULT=0
        ;;
esac

# Check if llama.cpp source exists
if [ ! -d "${BUILD_DIR}" ]; then
    echo "Error: llama.cpp source not found at ${BUILD_DIR}"
    echo "Run 'make' or 'scripts/setup.sh' first to download llama.cpp"
    exit 1
fi

# Handle clean argument
if [ "$1" = "clean" ]; then
    echo "Cleaning previous binary build..."
    rm -rf "${BUILD_DIR}/build-bins"
    rm -rf "${BIN_DIR}"
fi

# Create bin directory
mkdir -p "${BIN_DIR}"

# Build CMake args - enable examples/tools, disable tests
CMAKE_ARGS="-DBUILD_SHARED_LIBS=OFF"
CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_POSITION_INDEPENDENT_CODE=ON"
CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_BUILD_EXAMPLES=ON"
CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_BUILD_SERVER=ON"
CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_BUILD_TESTS=OFF"
CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_CURL=OFF"

# Backend configuration (match setup.sh)
if [ "$IS_MACOS" = "0" ]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DGGML_METAL=OFF"
fi

if [ "${GGML_METAL:-$METAL_DEFAULT}" = "1" ] && [ "$IS_MACOS" = "1" ]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DGGML_METAL=ON"
    echo "Enabling Metal backend"
fi

if [ "${GGML_CUDA:-0}" = "1" ]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DGGML_CUDA=ON"
    echo "Enabling CUDA backend"
fi

if [ "${GGML_VULKAN:-0}" = "1" ]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DGGML_VULKAN=ON"
    echo "Enabling Vulkan backend"
fi

if [ "${GGML_SYCL:-0}" = "1" ]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DGGML_SYCL=ON"
    echo "Enabling SYCL backend"
fi

if [ "${GGML_HIP:-0}" = "1" ]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DGGML_HIP=ON"
    echo "Enabling HIP/ROCm backend"
fi

if [ "${GGML_OPENCL:-0}" = "1" ]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DGGML_OPENCL=ON"
    echo "Enabling OpenCL backend"
fi

echo ""
echo "=== Building llama.cpp binaries ==="
echo "Source: ${BUILD_DIR}"
echo "Install to: ${BIN_DIR}"
echo ""

cd "${BUILD_DIR}"

# Use separate build directory for binaries to avoid conflicts
mkdir -p build-bins
cd build-bins

echo "Configuring with: cmake .. ${CMAKE_ARGS}"
cmake .. ${CMAKE_ARGS}

echo ""
echo "Building binaries..."
cmake --build . --config Release -j

echo ""
echo "Installing binaries to ${BIN_DIR}..."

# Copy binaries from bin directory
if [ -d "bin${CMAKE_BUILD_SUBDIR}" ]; then
    cp -f bin${CMAKE_BUILD_SUBDIR}/llama-* "${BIN_DIR}/" 2>/dev/null || true
fi

# Also check for binaries in root of build directory (some cmake versions)
for bin in llama-cli llama-server llama-quantize llama-perplexity llama-embedding \
           llama-simple llama-speculative llama-bench llama-run llama-gguf \
           llama-gguf-split llama-export-lora llama-cvector-generator \
           llama-imatrix llama-infill llama-lookup llama-minicpmv-cli \
           llama-parallel llama-passkey llama-retrieval llama-save-load-state \
           llama-tokenize; do
    if [ -f "${bin}${CMAKE_BUILD_SUBDIR:+.exe}" ]; then
        cp -f "${bin}${CMAKE_BUILD_SUBDIR:+.exe}" "${BIN_DIR}/" 2>/dev/null || true
    fi
done

cd "${CWD}"

# Count installed binaries
BIN_COUNT=$(ls -1 "${BIN_DIR}" 2>/dev/null | wc -l | tr -d ' ')

echo ""
echo "=== Build Complete ==="
echo "Installed ${BIN_COUNT} binaries to: ${BIN_DIR}"
echo ""
echo "Available binaries:"
ls -1 "${BIN_DIR}" 2>/dev/null | sed 's/^/  /'
echo ""
echo "Add to PATH: export PATH=\"${BIN_DIR}:\$PATH\""
