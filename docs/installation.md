# Installation

This guide covers installing cyllama on different platforms.

## Requirements

- Python 3.12 or later

    From v0.3.0, cyllama ships only abi3 (CPython stable ABI) wheels, which require Python 3.12+. For Python 3.10 / 3.11, install an older release (`pip install "cyllama<0.3.0"`) or build from source.

- C++ compiler (clang or gcc)

- CMake 3.21+

- Git

### Platform-Specific Requirements

**macOS:**

```bash
xcode-select --install  # Xcode Command Line Tools
```

**Ubuntu/Debian:**

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3-dev
```

**Fedora/RHEL:**

```bash
sudo dnf install -y gcc-c++ cmake git python3-devel
```

## Install from PyPI

```bash
pip install cyllama
```

### GPU-Accelerated Variants

GPU variants are available on PyPI as separate packages (dynamically linked, Linux x86_64 only):

```bash
pip install cyllama-cuda12   # NVIDIA GPU (CUDA 12.4)
pip install cyllama-rocm     # AMD GPU (ROCm 6.3, requires glibc >= 2.35)
pip install cyllama-sycl     # Intel GPU (oneAPI SYCL 2025.3)
pip install cyllama-vulkan   # Cross-platform GPU (Vulkan)
```

All GPU variants install the same `cyllama` Python package -- only the compiled backend differs. Install one at a time (they replace each other). GPU variants require the corresponding driver/runtime installed on your system.

#### `cyllama-sycl` host prerequisites

The SYCL wheel deliberately does not vendor the [Intel oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneapi-toolkit.html) runtime (`libsycl`, MKL, TBB, `libiomp5`, the Intel compiler runtimes). The full vendored stack exceeds PyPI's 100 MB per-file cap, and the GPU-coupled pieces must match the host driver anyway. The host therefore has to supply two things, and they live in separate registers.

**1. oneAPI userspace runtimes (required for `import cyllama` to succeed).**

These provide the libraries the wheel's `DT_NEEDED` entries point at -- `libsycl.so.8`, `libmkl_*.so*`, `libtbb.so.12`, `libiomp5.so`, `libsvml.so`, `libimf.so`, `libintlc.so.5`, `libirng.so`. On Debian/Ubuntu after adding the [Intel oneAPI APT repo](https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/current/apt-005.html):

```bash
sudo apt install \
  intel-oneapi-runtime-dpcpp-cpp \
  intel-oneapi-runtime-mkl \
  intel-oneapi-runtime-tbb \
  intel-oneapi-runtime-openmp
source /opt/intel/oneapi/setvars.sh
python -c "import cyllama"   # should succeed
```

For RPM-based distros, use Intel's [DNF/Yum repo](https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/current/yum-dnf-zypper.html) with the same `intel-oneapi-runtime-*` package names. If you already have the full [Intel oneAPI base toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneapi-toolkit.html) (`intel-basekit`) installed, these runtimes are included -- just source `setvars.sh`. Without these libraries on the loader path, import fails with `libsycl.so.8: cannot open shared object file` (or a similar message naming one of the other runtimes).

**2. A SYCL-visible compute device (required for actual GPU/CPU compute, not for import).**

`cyllama` needs at least one runtime device for SYCL to dispatch kernels onto. This is hardware-conditional and lives outside the oneAPI runtime layer -- pick one of:

- **Intel GPU via OpenCL**: install the Intel compute-runtime package providing `libOpenCL.so.1` and the Intel GPU ICD (`intel-opencl-icd` on recent Ubuntu, or the upstream `intel-compute-runtime` packages). Follow [Intel's GPU driver install guide](https://dgpu-docs.intel.com/driver/installation.html) for your distro and GPU family (Arc, Iris Xe, Data Center GPU Max/Flex).

- **Intel GPU via Level Zero**: install `level-zero` and `intel-level-zero-gpu`. Same install guide.

- **CPU fallback (no Intel GPU)**: install the Intel CPU runtime for OpenCL applications, packaged as `intel-oneapi-runtime-opencl` or the standalone CPU runtime. This is *not* a substitute for the oneAPI runtimes in step 1 -- it only adds the CPU as an OpenCL device.

Package names and recommended install paths drift across distro versions and Intel releases, so we link to Intel's authoritative install pages rather than hard-coding an `apt install` line we can't keep current. Without a device, import succeeds but SYCL device enumeration returns empty and any actual generation call fails.

You can verify which backend is active after installation:

```bash
cyllama info
```

## Build from Source

```bash
# Clone repository
git clone https://github.com/shakfu/cyllama.git
cd cyllama

# Build everything (downloads llama.cpp, whisper.cpp, builds cyllama)
make

# Download a test model
make download

# Verify installation
python -c "from cyllama import complete; print('OK')"
```

## Build Options

### Default Build

The default build enables GPU acceleration appropriate for your platform:

- **macOS**: Metal (Apple GPU)

- **Linux**: CPU-only (GPU backends optional)

```bash
make build
```

### GPU Backends

Build with specific GPU support (static or dynamic):

```bash
# Static builds (all libs compiled into the extension)
make build-cpu       # CPU only
make build-cuda      # NVIDIA CUDA
make build-vulkan    # Vulkan (cross-platform)

# Dynamic builds (shared libs installed alongside extension)
make build-cpu-dynamic
make build-cuda-dynamic
make build-vulkan-dynamic

# Multiple backends
GGML_CUDA=1 GGML_VULKAN=1 make build
```

See [Building with Different Backends](build_backends.md) for detailed GPU setup instructions.

### Optional Components

**Stable Diffusion support:**

```bash
WITH_STABLEDIFFUSION=1 make build

# Opt into sharing llama.cpp's ggml (not recommended for GPU backends)
SD_USE_VENDORED_GGML=0 make build
```

**Whisper support** (included by default):

```bash
make build  # Whisper is built automatically
```

## Build System

Cyllama uses **scikit-build-core** with CMake for building the Cython extensions. The build process:

1. **Dependencies**: `make` downloads and builds llama.cpp, whisper.cpp (and optionally stable-diffusion.cpp)

2. **Cython compilation**: CMake compiles `.pyx` files to C++ using Cython

3. **Extension linking**: C++ extensions are linked against the static libraries

4. **Installation**: Extensions are installed in editable mode

### Build Commands

| Command | Description |
|---------|-------------|
| `make` | Full build (dependencies + editable install) |
| `make build-<backend>` | Static build for a specific backend (e.g. `build-cuda`) |
| `make build-<backend>-dynamic` | Dynamic build for a specific backend (e.g. `build-cuda-dynamic`) |
| `make wheel` | Build wheel for distribution |
| `make wheel-<backend>` | Static wheel for a specific backend |
| `make wheel-<backend>-dynamic` | Dynamic wheel for a specific backend |
| `make clean` | Remove build artifacts and dynamic libs |
| `make reset` | Full reset including thirdparty and .venv |
| `make remake` | Clean rebuild with tests |
| `make leaks` | RSS-growth memory leak detection |

### Wheel Distribution

To build a distributable wheel:

```bash
make wheel
# Creates wheel in dist/
```

The wheel includes all compiled extensions and can be installed on systems with matching platform/Python version.

## Installing Models

### LLM Models (GGUF format)

Download the default test model:

```bash
make download
# Downloads: models/Llama-3.2-1B-Instruct-Q8_0.gguf
```

Or download manually from [Hugging Face](https://huggingface.co/models?search=gguf):

```bash
# Example: Download a model
curl -L -o models/llama.gguf \
  "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf"
```

### Whisper Models

Download from [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp):

```bash
curl -L -o models/ggml-base.en.bin \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
```

### Stable Diffusion Models

Download SDXL Turbo or other SD models in GGUF or safetensors format.

## Verification

### Test Installation

```bash
# Run test suite
make test

# Quick smoke test
python -c "
from cyllama import complete
print(complete('Hello', model_path='models/Llama-3.2-1B-Instruct-Q8_0.gguf', max_tokens=10))
"
```

### Check GPU Support

```python
from cyllama.llama.llama_cpp import ggml_backend_load_all

# Load all available backends
ggml_backend_load_all()

# Check what's available
from cyllama.llama.llama_cpp import LlamaModel, LlamaModelParams
params = LlamaModelParams()
params.n_gpu_layers = -1  # Offload all layers to GPU
# If GPU is available, layers will be offloaded
```

## Troubleshooting

### "No module named 'cyllama'"

Make sure you're in the project directory or have installed cyllama:

```bash
cd cyllama
make  # or: uv pip install -e .
```

### Build Errors

Clean and rebuild:

```bash
make reset  # Full clean
make build
```

### Metal Not Working (macOS)

Ensure Xcode Command Line Tools are installed:

```bash
xcode-select --install
```

### CUDA Not Found (Linux)

Add CUDA to your PATH:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Development Install

For development with editable install:

```bash
git clone https://github.com/shakfu/cyllama.git
cd cyllama
make  # Builds dependencies and installs in editable mode
```

For manual editable install (after dependencies are built):

```bash
uv pip install -e .
```

## Next Steps

- [User Guide](user_guide.md) - Learn the API

- [Cookbook](cookbook.md) - Common patterns and recipes

- [Building with Different Backends](build_backends.md) - GPU setup details
