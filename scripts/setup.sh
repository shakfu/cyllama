#!/bin/sh
#
# scripts/setup.sh - Thin wrapper around manage.py
#
# This script delegates to manage.py for all build operations.
# It preserves the original interface for backwards compatibility.
#
# Usage:
#   setup2.sh             : Build all dependencies (llama.cpp, whisper.cpp, stable-diffusion.cpp)
#   setup2.sh --clean     : Clean build artifacts before building
#   setup2.sh --deps-only : Build dependencies only, skip editable install
#
# Backend configuration via environment variables:
#   GGML_METAL=1    : Enable Metal backend (macOS, default on macOS)
#   GGML_CUDA=1     : Enable CUDA backend (NVIDIA GPUs)
#   GGML_VULKAN=1   : Enable Vulkan backend (cross-platform)
#   GGML_SYCL=1     : Enable SYCL backend (Intel GPUs)
#   GGML_HIP=1      : Enable HIP/ROCm backend (AMD GPUs)
#   GGML_OPENCL=1   : Enable OpenCL backend
#   SD_METAL=1      : Enable Metal for stable-diffusion.cpp (experimental, off by default)
#

set -e

# Change to project root directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Parse arguments
CLEAN=0
DEPS_ONLY=""
for arg in "$@"; do
    case "$arg" in
        --clean)
            CLEAN=1
            ;;
        --deps-only)
            DEPS_ONLY="--deps-only"
            ;;
    esac
done

# Clean if requested
if [ "$CLEAN" = "1" ]; then
    echo "Cleaning build artifacts..."
    rm -rf build thirdparty
fi

# Delegate to manage.py
echo "Building all dependencies via manage.py..."
exec uv run python scripts/manage.py build --all $DEPS_ONLY
