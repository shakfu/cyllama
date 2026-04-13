# Building cyllama with Different Backends

cyllama supports multiple GPU acceleration backends through llama.cpp. This guide shows you how to build with different backends using either the Makefile or the Python build manager (`scripts/manage.py`).

## Quick Start

### Default Build (Metal on macOS, CPU-only on Linux)

**Using Makefile:**

```bash
make build
```

**Using manage.py:**

```bash
python3 scripts/manage.py build --llama-cpp
```

### Build with Specific Backend

Each backend has both static and dynamic build targets. Static builds compile all libraries into the extension. Dynamic builds use shared libraries (`.so`/`.dylib`) installed alongside the extension.

**Using Makefile (static):**

```bash
make build-cpu       # CPU-only (no GPU)
make build-metal     # Metal (macOS)
make build-cuda      # CUDA (NVIDIA GPUs)
make build-vulkan    # Vulkan (cross-platform GPU)
make build-sycl      # SYCL (Intel GPUs)
make build-hip       # HIP/ROCm (AMD GPUs)
make build-opencl    # OpenCL
```

**Using Makefile (dynamic -- shared libs):**

```bash
make build-cpu-dynamic
make build-metal-dynamic
make build-cuda-dynamic
make build-vulkan-dynamic
make build-sycl-dynamic
make build-hip-dynamic
make build-opencl-dynamic
```

**Using manage.py:**

```bash
# CUDA (NVIDIA GPUs)
python3 scripts/manage.py build --llama-cpp --cuda

# Vulkan (Cross-platform GPU)
python3 scripts/manage.py build --llama-cpp --vulkan

# CPU-only (no GPU)
python3 scripts/manage.py build --llama-cpp --cpu-only

# SYCL (Intel GPUs)
python3 scripts/manage.py build --llama-cpp --sycl

# HIP/ROCm (AMD GPUs)
python3 scripts/manage.py build --llama-cpp --hip

# Multiple backends (CUDA + Vulkan)
python3 scripts/manage.py build --llama-cpp --cuda --vulkan

# Metal on macOS (default, or explicit)
python3 scripts/manage.py build --llama-cpp --metal
```

## Environment Variable Control

You can fine-tune backend selection using environment variables (works with both Makefile and manage.py):

**Using Makefile:**

```bash
# Enable specific backends
export GGML_CUDA=1
export GGML_VULKAN=1
make build

# Disable Metal on macOS
export GGML_METAL=0
make build

# Build with multiple backends
export GGML_CUDA=1 GGML_VULKAN=1
make build
```

**Using manage.py:**

```bash
# Environment variables work the same way
export GGML_CUDA=1
export GGML_VULKAN=1
python3 scripts/manage.py build --llama-cpp

# Or combine with command-line flags (flags override env vars)
export GGML_METAL=1
python3 scripts/manage.py build --llama-cpp --cuda  # Enables both Metal and CUDA
```

### Available Backend Flags

These flags apply uniformly to all components (llama.cpp, whisper.cpp, stable-diffusion.cpp):

| Variable | Default | Description |
|----------|---------|-------------|
| `GGML_METAL` | `1` | Apple Metal (macOS GPU) |
| `GGML_CUDA` | `0` | NVIDIA CUDA |
| `GGML_VULKAN` | `0` | Vulkan (cross-platform GPU) |
| `GGML_SYCL` | `0` | Intel SYCL (oneAPI) |
| `GGML_HIP` | `0` | AMD ROCm/HIP |
| `GGML_OPENCL` | `0` | OpenCL (Adreno, mobile GPUs) |
| `SD_USE_VENDORED_GGML` | `0` | Link stable-diffusion against its own vendored ggml instead of llama.cpp's |

## Backend Requirements

### CUDA (NVIDIA GPUs)

**Requirements:**

- NVIDIA GPU with compute capability 6.0+
- CUDA Toolkit 11.0+ installed
- `nvcc` compiler in PATH

**Install CUDA:**

```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6

# Verify installation
nvcc --version
```

**Build:**

```bash
export GGML_CUDA=1
make build
```

### Vulkan (Cross-platform GPU)

**Requirements:**

- Vulkan-capable GPU (NVIDIA, AMD, Intel, or Apple)
- Vulkan SDK installed
- Vulkan headers in system include path

**Install Vulkan SDK:**

```bash
# Ubuntu/Debian
sudo apt-get install -y libvulkan-dev vulkan-tools

# macOS
brew install vulkan-headers vulkan-loader molten-vk

# Verify installation
vulkaninfo --summary
```

**Build:**

```bash
export GGML_VULKAN=1
make build
```

### Metal (Apple Silicon/macOS)

**Requirements:**

- macOS 11.0+ (Big Sur or later)
- Apple Silicon (M1/M2/M3) or Intel Mac with AMD GPU
- Xcode Command Line Tools

**Build (enabled by default on macOS):**

```bash
make build
# or explicitly:
export GGML_METAL=1
make build
```

### SYCL (Intel GPUs)

**Requirements:**

- Intel GPU (Iris Xe, Arc, or Flex)
- Intel oneAPI Base Toolkit installed

**Install Intel oneAPI:**

```bash
# Ubuntu/Debian
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt-get update
sudo apt-get install -y intel-basekit

# Setup environment
source /opt/intel/oneapi/setvars.sh
```

**Build:**

```bash
export GGML_SYCL=1
make build
```

### HIP/ROCm (AMD GPUs)

**Requirements:**

- AMD GPU with ROCm support (gfx90a, gfx942, gfx1100, or newer)
- ROCm 6.3+ installed

**Install ROCm:**

```bash
# Ubuntu 22.04+
sudo apt-get update
wget https://repo.radeon.com/amdgpu-install/6.3.3/ubuntu/jammy/amdgpu-install_6.3.60303-1_all.deb
sudo apt-get install -y ./amdgpu-install_6.3.60303-1_all.deb
sudo amdgpu-install --usecase=rocm

# Verify installation
rocm-smi
```

**Build:**

```bash
export GGML_HIP=1
make build
```

## Multi-Backend Builds

You can build with multiple backends simultaneously:

```bash
# CUDA + Vulkan + Metal (on macOS)
export GGML_METAL=1 GGML_CUDA=1 GGML_VULKAN=1
make build

# CUDA + Vulkan (on Linux)
export GGML_CUDA=1 GGML_VULKAN=1
make build
```

At runtime, llama.cpp will automatically select the best available backend, or you can specify one explicitly via the model configuration.

## Checking Your Build

After building, you can verify which backends were compiled:

```bash
# Show current backend configuration
make show-backends

# Check compiled libraries
ls -lh thirdparty/llama.cpp/lib/
```

### Querying at Runtime

The build generates `src/cyllama/_backend.py` with the enabled backends and their configuration. You can inspect this at runtime:

```python
from cyllama import _backend

print(_backend.cuda)    # True if built with CUDA
print(_backend.metal)   # True if built with Metal
print(_backend.vulkan)  # True if built with Vulkan

# Backend-specific options (None if not set)
print(_backend.cuda_architectures)   # e.g. "89" or None
print(_backend.blas_vendor)          # e.g. "OpenBLAS" or None
```

This is also used internally for platform-specific setup such as Windows CUDA DLL discovery (see [Troubleshooting](troubleshooting.md#cuda-dlls-not-found-windows)).

Expected libraries for each backend:

| Backend | Library |
|---------|---------|
| CPU | `libggml-cpu.a` (always built) |
| Metal | `libggml-metal.a` + `libggml-blas.a` |
| CUDA | `libggml-cuda.a` |
| Vulkan | `libggml-vulkan.a` |
| SYCL | `libggml-sycl.a` |
| HIP/ROCm | `libggml-hip.a` |
| OpenCL | `libggml-opencl.a` |

## Troubleshooting

### "nvcc: command not found" (CUDA)

Make sure CUDA toolkit is installed and in your PATH:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### "vulkan/vulkan.h: No such file" (Vulkan)

Install Vulkan SDK:

```bash
# Ubuntu/Debian
sudo apt-get install -y libvulkan-dev

# macOS
brew install vulkan-headers molten-vk
```

### "cannot find -lcuda" (CUDA linking error)

Add CUDA library path:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Metal not working on macOS

Ensure you have the latest Xcode Command Line Tools:

```bash
xcode-select --install
```

### Performance is slow with GPU backend

Make sure you're enabling GPU offloading in your model configuration:

```python
from cyllama import LLM

model = LLM(
    model_path="model.gguf",
    n_gpu_layers=99,  # Offload all layers to GPU
)
```

## Clean Rebuild

If you encounter issues after changing backends:

```bash
# Clean build artifacts and dynamic libs (keeps thirdparty sources and .venv)
make clean

# Full reset including thirdparty libs and .venv
make reset

# Then rebuild with your desired backend
make build-cuda           # static
make build-cuda-dynamic   # dynamic
```

The `build-*` targets automatically run `clean` as a prerequisite, so switching backends is safe without manual cleanup.

## Performance Comparison

Approximate relative performance (inference speed):

| Backend | Relative Speed | Notes |
|---------|----------------|-------|
| CPU only | 1x (baseline) | Good for small models |
| Metal (M1/M2/M3) | 5-15x | Best on Apple Silicon |
| CUDA (RTX 4090) | 10-30x | Best on NVIDIA GPUs |
| Vulkan | 5-20x | Good cross-platform option |
| SYCL (Arc A770) | 3-8x | Intel GPUs |
| HIP/ROCm | 8-25x | AMD GPUs |

*Performance varies greatly based on model size, quantization, and hardware.*

## Recommended Backends by Platform

| Platform | Recommended Backend | Alternative |
|----------|-------------------|-------------|
| **Apple Silicon Mac** | Metal | Vulkan |
| **Linux + NVIDIA GPU** | CUDA | Vulkan |
| **Linux + AMD GPU** | HIP/ROCm | Vulkan |
| **Linux + Intel GPU** | SYCL | Vulkan |
| **Windows + NVIDIA GPU** | CUDA | Vulkan |
| **Windows + AMD GPU** | Vulkan | HIP/ROCm |

## Advanced: Custom CMake Flags

Cyllama has a two-stage build process:

1. **Dependency build** (`scripts/manage.py`): Builds llama.cpp, whisper.cpp, stable-diffusion.cpp as static libraries
2. **Extension build** (`CMakeLists.txt`): Builds Cython extensions with scikit-build-core, linking against the static libraries

### Customizing Dependency Build

Use `scripts/manage.py` with environment variables or flags:

```bash
# Example: Build llama.cpp with CUDA for specific architectures
CMAKE_CUDA_ARCHITECTURES="86-real;89-real" python3 scripts/manage.py build --llama-cpp --cuda
```

### Customizing Extension Build

For the Cython extension build, pass CMake args via scikit-build-core:

```bash
# Pass CMake args during wheel build
CMAKE_ARGS="-DGGML_METAL=ON" uv build --wheel
```

## References

- [llama.cpp Build Documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- [llama.cpp GPU Acceleration Guide](https://www.ywian.com/blog/llama-cpp-gpu-acceleration-complete-guide)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [Vulkan SDK](https://www.lunarg.com/vulkan-sdk/)
- [Intel oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)
- [ROCm Documentation](https://rocm.docs.amd.com/)
