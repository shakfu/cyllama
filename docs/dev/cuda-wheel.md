# CUDA wheel size and architecture targets

Analysis behind the `CMAKE_CUDA_ARCHITECTURES` change in `_gpu-build-cuda.yml` and `_gpu-build-cuda-windows.yml` before 1.0.0. Date: 2026-11-17.

This is the reference for both cyllama and inferna. inferna sets the same value in `_gpu-build-cuda.yml`, `_gpu-build-cuda-windows.yml` and `_CI_LINUX_BUILD_ENV` in `scripts/manage.py`. Measurements below are for `cyllama-cuda12`.

## Summary

- The 1.0.0a0 `cyllama_cuda12` wheels exceed PyPI's 100 MB file limit: 113.1 MB (Linux), 109.6 MB (Windows).
- `libggml-cuda` is ~80% of each wheel. Its GPU code (fatbin) dominates.
- CI built with `"75"`. Ampere and newer GPUs ran Turing-level kernels, JIT-compiled at first load.
- New setting: `"75-real;80-virtual"`. About 2.5 MB smaller, 12-25% faster long-context prompt processing on Ada, Turing unchanged.
- PyPI raised the `cyllama-cuda12` file-size limit to 200 MB (granted 2026-11-21).

## Terms

- **SASS**: machine code for one GPU architecture (`sm_XY`). A cubin holds SASS.
- **PTX**: portable intermediate code (`compute_XY`). The driver JIT-compiles it to SASS for the installed GPU and caches the result in `~/.nv/ComputeCache`.
- **`XY-real`**: SASS only. **`XY-virtual`**: PTX only. Plain **`XY`**: both.
- SASS for `sm_XY` runs on `sm_XZ` with Z >= Y (same major version). PTX for `compute_XY` runs on any GPU with compute capability >= X.Y.

## 1.0.0a0 wheel composition (Linux)

| Member | Uncompressed | In wheel |
|-|-|-|
| `libggml-cuda.so` | 257 MB | 91.0 MB |
| `stable_diffusion.abi3.so` | 40 MB | 16.3 MB |
| 168 other files | ~19 MB | ~6 MB |

The Windows wheel matches: `ggml-cuda.dll` is 251 MB, 89.8 MB in the wheel.

Inside `libggml-cuda.so`:

- `.nv_fatbin` is 233 MB of the 257 MB. nvcc already compresses it (default mode).
- It holds 149 `sm_75` cubins (149 MB) and 149 `compute_75` PTX entries (362 MB) before compression. Gzipped: ~35 MB and ~59 MB.
- Flash-attention kernels: 160 MB PTX, 60 MB cubin. MMQ kernels: 122 MB PTX, 49 MB cubin. Together ~78% of the fatbin.
- No line info or other GPU debug sections.

`stable_diffusion.abi3.so` has 30 MB of `.rodata`, present in every wheel. The cause is assumed to be embedded tokenizer vocabularies; not verified.

## Why `"75"` was slow on newer GPUs

With `"75"`, only Turing runs SASS. Every Ampere, Ada, Hopper and Blackwell GPU JIT-compiles the `compute_75` PTX. That PTX is built with `__CUDA_ARCH__=750`, so ggml's compile-time Ampere+ paths are absent (`ggml/src/ggml-cuda/common.cuh`: `AMPERE_MMA_AVAILABLE`, `CP_ASYNC_AVAILABLE`).

The RTX 4060 used for 1.0.0a0 testing took this JIT path.

## Benchmark

### Method

- llama.cpp `b12359`, CUDA 12.0, `GGML_NATIVE=OFF`, Release, static `llama-bench`. Only `CMAKE_CUDA_ARCHITECTURES` varies.
- RTX 4060 laptop GPU (compute capability 8.9, 8 GB), driver 595.91.07.
- `llama-bench -ngl 99 -p 512 -n 128 -fa 0,1 -d 0,4096 -r 5`.
- Models: Llama-3.2-1B-Instruct Q8_0, Qwen3-4B Q8_0, gemma-4-E4B-it Q5_K_M.
- Each variant uses an empty, isolated `CUDA_CACHE_PATH`; `CUDA_CACHE_MAXSIZE=4294967296`.
- `llama-bench` prints `failed to restore kv cache` when `-fa` changes between tests. It then rebuilds the depth state, so results are unaffected.

### Throughput

Ratios of tokens/s. pp512 = 512-token prompt processing; tg128 = 128-token generation.

| Workload | Model | 80-virtual / 75-virtual | 89-real / 80-virtual | 75-real;80-real / 80-virtual |
|-|-|-|-|-|
| pp512, FA on, depth 4096 | Llama-3.2-1B | 1.25 | 0.97 | 0.83 |
| | Qwen3-4B | 1.16 | 0.97 | 0.83 |
| | gemma-4-E4B | 1.12 | 0.97 | 0.88 |
| pp512, FA on, depth 0 | all | 1.04-1.06 | 0.94-0.97 | 0.79-0.89 |
| pp512, FA off | all | 0.97-1.03 | 0.98-1.04 | 0.81-0.93 |
| tg128 | Llama, Qwen | 1.00 | 1.00 | 0.98 |
| tg128 | gemma | 0.96 | 0.98 | 0.99 |

Findings:

- `compute_80` PTX recovers the Ampere+ kernels. Native `sm_89` SASS adds nothing on top.
- `sm_80` SASS on Ada is 15-20% slower at pp512 than `compute_80` PTX JIT-compiled for `sm_89`. An interleaved rerun (Qwen3-4B, FA on, GPU at 52-74 C) reproduced it, so it is not thermal. Assumed cause: ptxas schedules for the target GPU, and `sm_80` SASS is scheduled for A100. RTX 30 (8.6) running `sm_80` SASS is untested.
- gemma tg128 is 4% slower with any Ampere-level build than with `compute_75`. Not investigated.
- Llama tg128 at FA off, depth 4096 had +-32 t/s spread on the PTX builds; treat that row as noise.

Absolute values, Qwen3-4B pp512 FA on, depth 0 / 4096 (t/s): `75-virtual` 4452 / 3054, `80-virtual` 4716 / 3533, `89-real` 4594 / 3442, `75-real;80-real` 3736 / 2916.

### First load

Llama-3.2-1B, `-p 32 -n 0 -r 1`, wall time:

| Setting | Empty cache | Warm cache | JIT cache written |
|-|-|-|-|
| `75-virtual` | 25.5 s | 0.66 s | 45 MB |
| `80-virtual` | 24.0 s | 0.66 s | 42 MB |
| `75-real;80-real` | 0.79 s | 0.61 s | - |
| `89-real` | 0.60 s | 0.60 s | 0 |

This is a lower bound. CUDA's lazy module loading compiles kernels on first use, so other models, quant types and FA settings add more JIT time.

The development machine's `~/.nv/ComputeCache` was at 1.0 GB with `CUDA_CACHE_MAXSIZE` unset. If that is the default cap (unverified), new builds evict older entries and the first-load cost recurs.

### Size

Static `llama-bench` builds. "GPU code" is the `.nv_fatbin` section.

| Setting | Binary raw | Binary gzip -6 | GPU code raw | GPU code gzip -6 |
|-|-|-|-|-|
| `75` (1.0.0a0) | 273.4 MB | 95.7 MB | 234.2 MB | 88.1 MB |
| `75-real;80-virtual` (new) | 269.0 MB | 93.2 MB | 229.8 MB | 85.5 MB |
| `75-virtual` | 123.2 MB | 60.1 MB | 84.0 MB | 52.4 MB |
| `80-virtual` | 118.8 MB | 57.6 MB | 79.6 MB | 49.9 MB |
| `89-real` | 185.0 MB | 41.4 MB | 145.7 MB | 33.7 MB |
| `75-real;80-real` | 336.5 MB | 77.8 MB | 297.2 MB | 70.1 MB |
| `75-real;80-real;90-real` | 488.9 MB | 113.1 MB | 449.6 MB | 105.5 MB |

- One SASS arch adds 34-36 MB gzipped. One PTX arch adds 50-52 MB.
- The rest of the 1.0.0a0 wheel is ~25 MB (113.1 - 88.1).
- Wheel estimates below assume wheel size = 25 MB + GPU code gzip size. Only `75-real;80-virtual` was later built; see [Built wheels](#built-wheels-75-real80-virtual).

## Options

| Setting | Est. wheel | Turing | Ampere / Ada | Hopper | Blackwell |
|-|-|-|-|-|-|
| `75` (1.0.0a0) | 113 MB | SASS | `compute_75` PTX (slow) | PTX (slow) | PTX (slow) |
| **`75-real;80-virtual`** | **~110 MB** | **SASS** | **`compute_80` PTX** | **PTX** | **PTX** |
| `75-virtual` | ~77 MB | PTX, JIT stall | `compute_75` PTX (slow) | PTX (slow) | PTX (slow) |
| `80-virtual` | ~75 MB | no code | `compute_80` PTX | PTX | PTX |
| `75-real;80-real` | ~95 MB | SASS | `sm_80` SASS (15-20% slower on Ada) | no code | no code |
| `75-real;80-real;90-real` | ~131 MB | SASS | `sm_80` SASS | SASS | no code |

"no code" fails at kernel load with a missing-kernel-image error.

### Decision: `75-real;80-virtual`

- Turing keeps native SASS: no JIT stall, and CUDA 12 minor-version compatibility keeps the driver floor at 525+.
- Every Ampere+ GPU gets `compute_80` kernels, JIT-compiled for its own architecture.
- Turing was kept because it is the oldest architecture CUDA 13 still targets ([CUDA features archive](https://docs.nvidia.com/cuda/cuda-features-archive/index.html)), and T4 instances are common in cloud and Colab use.
- Wheels are about 2.5 MB smaller. They fit under the 200 MB `cyllama-cuda12` limit.

### Rejected

- **`75-virtual`**: fits under 100 MB, but keeps the 12-25% long-context loss. Turing gains a JIT stall and a 550+ driver floor, because PTX JIT is not covered by minor-version compatibility.
- **`80-virtual`**: fits under 100 MB, drops Turing (RTX 20, GTX 16, T4). Turing users would need `cyllama-vulkan`.
- **SASS only**: larger, slower on Ada, and no code for newer architectures. Blackwell SASS needs CUDA 12.8+, which CI does not use.
- **CUDA 12.8+ toolkit**: ggml then passes `-compress-mode=size` ([llama.cpp #12029](https://cdn04132025.gitlink.org.cn/replica/llama.cpp/commit/80c41ddd8f11b8094018296d9820c2b6c119ac12)). bitsandbytes reported 69 to 45 MB with that flag ([PR #1704](https://github.yxlr123.workers.dev/bitsandbytes-foundation/bitsandbytes/pull/1704)). The saving in our wheel is unmeasured and likely smaller, because wheel deflate already compresses the fatbin. rapids-cmake lists driver 550.54.14+ for `size` mode on CUDA 12 ([docs](https://docs.nvidia.com/rapids-cmake/latest/command/rapids_cuda_enable_fatbin_compression/)).
- **Separate wheel for `libggml-cuda`**: 91 MB alone leaves ~9% headroom under 100 MB, and adds cross-package pinning and library-path work.
- **`GGML_CUDA_FA=OFF`**: removes ~40% of GPU code and CUDA flash attention with it.

## Built wheels: `75-real;80-virtual`

CI run [44034241213](https://github.com/shakfu/cyllama/actions/runs/44034241213) on `94b05ce`. Fatbin entries counted by walking the fatbin headers in each wheel's CUDA library (no `cuobjdump` on macOS).

| Wheel | Size | CUDA lib raw / in wheel | Fatbin entries |
|-|-|-|-|
| Linux | 110.6 MB | 255.7 / 88.5 MB | 149 `sm_75` ELF, 149 `compute_80` PTX |
| Windows | 107.2 MB | 246.2 / 87.4 MB | 149 `sm_75` ELF, 149 `compute_80` PTX |

- No `compute_75` PTX and no `sm_80` SASS, as the setting intends.
- The ~110 MB estimate held. Both fit the 200 MB `cyllama-cuda12` limit.

## Not measured

- Turing, Ampere, Hopper and Blackwell hardware.
- CUDA 12.4, the CI toolkit. Local builds used 12.0.
- `-compress-mode=size`: the sandbox blocked running a downloaded NVIDIA `fatbinary`.

## Reproduction

```bash
git clone --depth 1 --branch b12359 https://github.com/ggml-org/llama.cpp src
cmake -S src -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON -DGGML_NATIVE=OFF "-DCMAKE_CUDA_ARCHITECTURES=75-real;80-virtual" \
    -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build build --target llama-bench -j
CUDA_CACHE_PATH=$PWD/jitcache build/bin/llama-bench -m MODEL.gguf \
    -ngl 99 -p 512 -n 128 -fa 0,1 -d 0,4096 -r 5 -o csv
objcopy -O binary --only-section=.nv_fatbin build/bin/llama-bench fatbin && gzip -6 -c fatbin | wc -c
cuobjdump --list-elf build/bin/llama-bench; cuobjdump --list-ptx build/bin/llama-bench
```
