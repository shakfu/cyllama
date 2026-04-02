# Advanced Build Options

This guide covers advanced build configuration for cyllama. For basic backend setup (CUDA, Metal, Vulkan, etc.), see [Building with Different Backends](build_backends.md).

## How the Build Works

Cyllama uses a two-phase build:

1. **Phase 1 -- Build dependencies**: `scripts/manage.py build --deps-only` clones and builds llama.cpp (and optionally whisper.cpp, stable-diffusion.cpp), producing static libraries in `thirdparty/`.
2. **Phase 2 -- Build cyllama**: `uv pip install .` (or `uv sync`) runs scikit-build-core, which links the pre-built libraries into Cython extension modules.

Most options in this guide affect Phase 1. Backend flags (`GGML_CUDA`, etc.) affect both phases.

## Static vs Dynamic Linking

By default, cyllama links statically -- all llama.cpp code is compiled into the extension. This produces a self-contained install with no runtime dependencies on shared libraries.

Dynamic linking is available via the `--dynamic` flag:

```bash
# Download pre-built llama.cpp shared libraries and link against them
python3 scripts/manage.py build --all --dynamic

# Or build shared libraries from source
python3 scripts/manage.py build --all --shared
```

In `--dynamic` mode, the build downloads pre-built release archives from the llama.cpp GitHub releases when available. If no pre-built asset exists for your platform/backend combination, it falls back to building from source with `BUILD_SHARED_LIBS=ON`.

### Pre-built Dynamic Releases

| Platform | Backend | Available? |
|----------|---------|------------|
| macOS arm64 | Metal | Yes |
| macOS x86_64 | Metal | Yes |
| Linux x86_64 | CPU | Yes |
| Linux x86_64 | Vulkan | Yes |
| Linux x86_64 | CUDA | No (builds from source) |
| Linux x86_64 | HIP/ROCm | No (builds from source) |
| Windows x64 | CPU | Yes |
| Windows x64 | CUDA | Yes (CUDA 12.4 by default) |

For Windows CUDA, the downloaded asset defaults to CUDA 12.4. To target a different version (if llama.cpp publishes a matching release):

```powershell
$env:LLAMACPP_CUDA_RELEASE = "13.1"
python scripts/manage.py build --all --dynamic --cuda
```

## CUDA Options

### Architecture Targeting

Control which GPU architectures to compile for:

```bash
# Target specific architectures (SASS -- native code, fastest but arch-specific)
CMAKE_CUDA_ARCHITECTURES="86-real;89-real" GGML_CUDA=1 make build

# Target PTX only (JIT-compiled at runtime, portable but slower first load)
CMAKE_CUDA_ARCHITECTURES="75" GGML_CUDA=1 make build

# Let llama.cpp auto-detect your GPU (default when GGML_NATIVE is ON)
GGML_CUDA=1 make build
```

The `-real` suffix produces SASS (native GPU code) for that architecture only. Without `-real`, PTX (portable intermediate code) is generated, which the CUDA driver JIT-compiles to your GPU on first use.

Common compute capabilities:

| Architecture | Compute Capability | Example GPUs |
|-------------|-------------------|-------------|
| Turing | 75 | RTX 2080, T4 |
| Ampere | 80, 86 | A100, RTX 3090 |
| Ada Lovelace | 89 | RTX 4090, L40 |
| Hopper | 90 | H100 |

### Custom CUDA Compiler

If you have multiple CUDA toolkit installations, specify which `nvcc` to use:

```bash
CMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc GGML_CUDA=1 make build
```

### Performance Tuning Flags

These compile-time flags affect CUDA kernel selection. Set them as environment variables before building:

```bash
# Force custom quantized matrix multiplication kernels
GGML_CUDA_FORCE_MMQ=ON GGML_CUDA=1 make build

# Force cuBLAS with FP16 instead of custom kernels
GGML_CUDA_FORCE_CUBLAS=ON GGML_CUDA=1 make build

# Compile flash attention for all KV cache quantization types
# (increases binary size but supports all quant formats)
GGML_CUDA_FA_ALL_QUANTS=ON GGML_CUDA=1 make build

# Set max batch size for peer GPU access in multi-GPU setups (default: 128)
GGML_CUDA_PEER_MAX_BATCH_SIZE=256 GGML_CUDA=1 make build
```

These flags are forwarded to CMake for all three components (llama.cpp, whisper.cpp, stable-diffusion.cpp).

## HIP/ROCm Options

### Architecture Targeting

```bash
# Target specific AMD GPU architectures
CMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100" GGML_HIP=1 make build
```

### rocWMMA Flash Attention

Enable rocWMMA-accelerated flash attention for supported AMD GPUs:

```bash
GGML_HIP_ROCWMMA_FATTN=1 GGML_HIP=1 make build
```

## BLAS Backend

Enable explicit BLAS acceleration for CPU-based matrix operations. On macOS with Metal, Apple Accelerate is used automatically. On Linux/Windows, you can select a BLAS vendor:

```bash
# Using manage.py CLI flag
python3 scripts/manage.py build --all --blas

# With a specific vendor (OpenBLAS, Intel MKL, etc.)
GGML_BLAS_VENDOR=OpenBLAS python3 scripts/manage.py build --all --blas

# Intel MKL
GGML_BLAS_VENDOR=Intel10_64lp python3 scripts/manage.py build --all --blas
```

Or using environment variables directly:

```bash
GGML_BLAS=1 GGML_BLAS_VENDOR=OpenBLAS make build
```

## OpenMP

OpenMP is enabled by default in llama.cpp for CPU parallelism. Disable it for Arm or embedded builds where OpenMP is unavailable or undesirable:

```bash
# Using manage.py CLI flag
python3 scripts/manage.py build --all --no-openmp

# Using environment variable
GGML_OPENMP=0 make build
```

## Portability: Native vs Cross-Platform Builds

By default, llama.cpp builds with `GGML_NATIVE=ON`, which optimizes for the CPU instruction set of the build machine (e.g., AVX2 on modern x86). This produces the fastest binaries for your machine but may crash (`SIGILL`) on machines with older CPUs.

For portable binaries (e.g., distributing to others):

```bash
GGML_NATIVE=OFF GGML_CUDA=1 make build
```

The CI wheel builds always set `GGML_NATIVE=OFF` for this reason.

## Windows Builds

On Windows, use PowerShell to set environment variables:

```powershell
# Build from source with CUDA
$env:GGML_CUDA = "1"
python scripts/manage.py build --all --deps-only --cuda
uv pip install . --force-reinstall --no-binary cyllama

# With specific CUDA architectures
$env:CMAKE_CUDA_ARCHITECTURES = "86;89"
$env:GGML_CUDA = "1"
python scripts/manage.py build --all --deps-only --cuda
uv pip install . --force-reinstall --no-binary cyllama

# Custom CUDA compiler path
$env:CMAKE_CUDA_COMPILER = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe"
$env:GGML_CUDA = "1"
python scripts/manage.py build --all --deps-only --cuda
```

Make sure `CUDA_PATH` or your `PATH` points to the correct CUDA toolkit installation so CMake can find it.

### Runtime DLL Discovery

On Windows, CUDA-linked extensions depend on toolkit DLLs (e.g. `cublas64_13.dll`). When cyllama is built with `GGML_CUDA=1`, it automatically registers CUDA DLL search paths via `os.add_dll_directory()` before loading native extensions. The discovery checks, in order:

1. `CUDA_PATH` / `CUDA_HOME` environment variables
2. `nvcc` on `PATH`
3. Standard install location (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\`)

This is a no-op on non-Windows platforms or non-CUDA builds, controlled by the build-time `_backend.py` config.

## Complete Environment Variable Reference

### Backend Selection

| Variable | Default | Description |
|----------|---------|-------------|
| `GGML_METAL` | `1` on macOS, `0` otherwise | Apple Metal backend |
| `GGML_CUDA` | `0` | NVIDIA CUDA backend |
| `GGML_VULKAN` | `0` | Vulkan backend |
| `GGML_SYCL` | `0` | Intel SYCL backend |
| `GGML_HIP` | `0` | AMD HIP/ROCm backend |
| `GGML_OPENCL` | `0` | OpenCL backend |
| `GGML_BLAS` | `0` | BLAS backend (CPU matrix ops) |

### Build Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GGML_NATIVE` | `ON` | Optimize for build machine CPU |
| `GGML_OPENMP` | `ON` | Enable OpenMP parallelism |
| `GGML_BLAS_VENDOR` | (auto) | BLAS vendor: `OpenBLAS`, `Intel10_64lp`, `Generic`, etc. |
| `CMAKE_CUDA_ARCHITECTURES` | (auto) | CUDA compute capabilities, e.g. `"75;86"` |
| `CMAKE_CUDA_COMPILER` | (auto) | Path to `nvcc` |
| `CMAKE_HIP_ARCHITECTURES` | (auto) | HIP GPU targets, e.g. `"gfx90a;gfx1100"` |
| `LLAMACPP_CUDA_RELEASE` | `12.4` | CUDA version for `--dynamic` pre-built downloads (Windows) |

### CUDA Tuning (compile-time)

| Variable | Default | Description |
|----------|---------|-------------|
| `GGML_CUDA_FORCE_MMQ` | `OFF` | Force custom quantized matrix kernels |
| `GGML_CUDA_FORCE_CUBLAS` | `OFF` | Force cuBLAS FP16 instead of custom kernels |
| `GGML_CUDA_FA_ALL_QUANTS` | `OFF` | Compile all KV cache quantization types for flash attention |
| `GGML_CUDA_PEER_MAX_BATCH_SIZE` | `128` | Max batch size for multi-GPU peer access |

### HIP Tuning (compile-time)

| Variable | Default | Description |
|----------|---------|-------------|
| `GGML_HIP_ROCWMMA_FATTN` | `OFF` | Enable rocWMMA flash attention |

## See Also

- [Building with Different Backends](build_backends.md) -- basic backend setup and troubleshooting
- [llama.cpp Build Documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) -- upstream build reference
