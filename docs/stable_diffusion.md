# Stable Diffusion Integration Plan

This document outlines the plan for wrapping `stable-diffusion.cpp` into cyllama, providing Python bindings for image and video generation.

## Current State

The `stable-diffusion.cpp` thirdparty library is already built and installed:

- **Headers**: `thirdparty/stable-diffusion.cpp/include/stable-diffusion.h` (main C API)
- **Library**: `thirdparty/stable-diffusion.cpp/lib/libstable-diffusion.a` (17MB static library)
- **Dependencies**: ggml libraries (shared with llama.cpp)

## Public C API Overview

### Core Enums

| Enum | Description |
|------|-------------|
| `rng_type_t` | Random number generator types (STD_DEFAULT, CUDA, CPU) |
| `sample_method_t` | Sampling methods (EULER, EULER_A, HEUN, DPM2, DPMPP2S_A, DPMPP2M, etc.) |
| `scheduler_t` | Schedulers (DISCRETE, KARRAS, EXPONENTIAL, AYS, GITS, SGM_UNIFORM, etc.) |
| `prediction_t` | Prediction types (DEFAULT, EPS, V, EDM_V, SD3_FLOW, FLUX_FLOW) |
| `sd_type_t` | Data types (F32, F16, Q4_0, Q4_1, Q5_0, Q8_0, etc.) |
| `sd_log_level_t` | Log levels (DEBUG, INFO, WARN, ERROR) |
| `preview_t` | Preview modes (NONE, PROJ, TAE, VAE) |
| `lora_apply_mode_t` | LoRA application modes (AUTO, IMMEDIATELY, AT_RUNTIME) |

### Core Structures

| Struct | Description |
|--------|-------------|
| `sd_ctx_params_t` | Context creation parameters (model paths, threads, types, etc.) |
| `sd_image_t` | Image data (width, height, channel, data pointer) |
| `sd_sample_params_t` | Sampling parameters (scheduler, method, steps, eta) |
| `sd_img_gen_params_t` | Image generation parameters (prompt, size, seed, etc.) |
| `sd_vid_gen_params_t` | Video generation parameters (for Wan/CogVideoX models) |
| `sd_tiling_params_t` | Tiling parameters for large images |
| `sd_guidance_params_t` | CFG guidance parameters |
| `sd_pm_params_t` | PhotoMaker parameters |
| `sd_easycache_params_t` | Cache optimization parameters |

### Core Functions

```c
// Context management
sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* sd_ctx_params);
void free_sd_ctx(sd_ctx_t* sd_ctx);

// Image generation
sd_image_t* generate_image(sd_ctx_t* sd_ctx, const sd_img_gen_params_t* params);

// Video generation
sd_image_t* generate_video(sd_ctx_t* sd_ctx, const sd_vid_gen_params_t* params, int* num_frames_out);

// Upscaling (ESRGAN)
upscaler_ctx_t* new_upscaler_ctx(const char* esrgan_path, ...);
sd_image_t upscale(upscaler_ctx_t* ctx, sd_image_t input, uint32_t factor);

// Model conversion
bool convert(const char* input_path, const char* vae_path, const char* output_path, ...);

// Preprocessing
bool preprocess_canny(sd_image_t image, float high_threshold, ...);

// Callbacks
void sd_set_log_callback(sd_log_cb_t cb, void* data);
void sd_set_progress_callback(sd_progress_cb_t cb, void* data);
void sd_set_preview_callback(sd_preview_cb_t cb, ...);

// Parameter initialization helpers
void sd_ctx_params_init(sd_ctx_params_t* params);
void sd_sample_params_init(sd_sample_params_t* params);
void sd_img_gen_params_init(sd_img_gen_params_t* params);
```

## Implementation Plan

### File Structure

```
src/cyllama/stablediffusion/
    __init__.py              # Module exports
    stable_diffusion.pxd     # Cython declarations (extern from "stable-diffusion.h")
    stable_diffusion.pyx     # Cython wrapper implementation
    cli.py                   # Command-line interface (optional)
```

### Cython Declaration File (stable_diffusion.pxd)

Contents:
- All enums (`rng_type_t`, `sample_method_t`, `scheduler_t`, etc.)
- All structs (`sd_ctx_params_t`, `sd_image_t`, `sd_img_gen_params_t`, etc.)
- Opaque types (`sd_ctx_t`, `upscaler_ctx_t`)
- Callback types (`sd_log_cb_t`, `sd_progress_cb_t`, `sd_preview_cb_t`)
- All public functions

### Cython Wrapper Classes (stable_diffusion.pyx)

```python
cdef class SDContextParams:
    """Wrapper for sd_ctx_params_t"""
    cdef sd_ctx_params_t _params

cdef class SDImage:
    """Wrapper for sd_image_t with numpy integration"""
    cdef sd_image_t _image

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array (H, W, C)"""
        ...

    def to_pil(self) -> PIL.Image:
        """Convert to PIL Image"""
        ...

    @staticmethod
    def from_numpy(arr: np.ndarray) -> SDImage:
        """Create from numpy array"""
        ...

cdef class SDContext:
    """Main stable diffusion context"""
    cdef sd_ctx_t* _ctx

    def generate_image(self, prompt: str, **kwargs) -> list[SDImage]:
        ...

    def generate_video(self, prompt: str, **kwargs) -> list[SDImage]:
        ...

cdef class Upscaler:
    """ESRGAN upscaler wrapper"""
    cdef upscaler_ctx_t* _ctx

    def upscale(self, image: SDImage, factor: int) -> SDImage:
        ...
```

### Build System Changes

**setup.py:**
- Add new extension module `cyllama.stablediffusion.stable_diffusion`
- Link against `libstable-diffusion.a` and ggml libraries
- Add include path for stable-diffusion headers

**MANIFEST.in:**
- Add stable-diffusion source files (`*.pxd`, `*.pyx`)

## Proposed Python API

### High-Level API

```python
from cyllama.stablediffusion import text_to_image, image_to_image, SDModel

# Simple text-to-image
images = text_to_image(
    model_path="sd-v1-5.safetensors",
    prompt="a photo of a cat",
    negative_prompt="blurry, low quality",
    width=512,
    height=512,
    steps=20,
    seed=42
)

# With model reuse for batch generation
model = SDModel("sd-v1-5.safetensors")
for prompt in prompts:
    images = model.generate(prompt)

# Image-to-image
result = image_to_image(
    model_path="sd-v1-5.safetensors",
    init_image=input_image,  # PIL Image or numpy array
    prompt="make it a painting",
    strength=0.7
)

# Video generation (for Wan/CogVideoX models)
frames = model.generate_video(
    prompt="a cat walking",
    video_frames=16
)

# Upscaling with ESRGAN
from cyllama.stablediffusion import Upscaler
upscaler = Upscaler("esrgan-x4.bin")
upscaled = upscaler.upscale(image, factor=4)
```

### Low-Level API

```python
from cyllama.stablediffusion import (
    SDContext, SDContextParams, SDImage,
    SDSampleParams, SDImageGenParams,
    SampleMethod, Scheduler, SDType
)

# Full control over context creation
params = SDContextParams()
params.model_path = "sd-v1-5.safetensors"
params.vae_path = "vae-ft-mse.safetensors"
params.n_threads = 4
params.wtype = SDType.F16

ctx = SDContext(params)

# Full control over generation
gen_params = SDImageGenParams()
gen_params.prompt = "a beautiful landscape"
gen_params.negative_prompt = "ugly, blurry"
gen_params.width = 768
gen_params.height = 512
gen_params.seed = 42

sample_params = SDSampleParams()
sample_params.sample_method = SampleMethod.EULER_A
sample_params.scheduler = Scheduler.KARRAS
sample_params.sample_steps = 25
gen_params.sample_params = sample_params

images = ctx.generate_image(gen_params)
```

## Implementation Phases

### Phase 1: Core API (3-4 hours)

- [ ] Create `stable_diffusion.pxd` with all declarations
- [ ] Implement `SDContext` class with context lifecycle management
- [ ] Implement `SDImage` class with numpy integration
- [ ] Implement `SDContextParams` with sensible defaults
- [ ] Basic `generate_image()` functionality
- [ ] Add to setup.py build configuration

### Phase 2: Extended Features (2-3 hours)

- [ ] Video generation (`generate_video()`)
- [ ] `Upscaler` wrapper class
- [ ] LoRA support in context params
- [ ] ControlNet support
- [ ] Tiling support for large images
- [ ] PhotoMaker support

### Phase 3: High-Level API (2-3 hours)

- [ ] `text_to_image()` convenience function
- [ ] `image_to_image()` convenience function
- [ ] `SDModel` class for model reuse
- [ ] Progress callbacks with Python integration
- [ ] Preview callbacks
- [ ] PIL Image integration
- [ ] CLI tool (`python -m cyllama.stablediffusion`)

### Phase 4: Testing & Documentation (2-3 hours)

- [ ] Unit tests for all wrapper classes
- [ ] Integration tests with real models
- [ ] Example scripts in `tests/examples/`
- [ ] API documentation
- [ ] Update CHANGELOG.md

## Key Challenges

### 1. Image Data Handling

Need efficient conversion between `sd_image_t` (raw uint8 buffer) and numpy arrays/PIL Images:

```python
cdef class SDImage:
    def to_numpy(self):
        cdef np.ndarray arr = np.empty(
            (self._image.height, self._image.width, self._image.channel),
            dtype=np.uint8
        )
        memcpy(arr.data, self._image.data,
               self._image.width * self._image.height * self._image.channel)
        return arr
```

### 2. Callback Threading

Progress and preview callbacks happen during generation - need proper GIL handling:

```python
cdef void progress_callback(int step, int steps, float time, void* data) noexcept with gil:
    py_callback = <object>data
    py_callback(step, steps, time)
```

### 3. Large Parameter Structs

`sd_ctx_params_t` has 30+ fields - need Python-friendly initialization:

```python
cdef class SDContextParams:
    def __cinit__(self):
        sd_ctx_params_init(&self._params)  # Initialize with defaults

    @property
    def model_path(self):
        return self._params.model_path.decode() if self._params.model_path else None

    @model_path.setter
    def model_path(self, value):
        self._model_path_bytes = value.encode()
        self._params.model_path = self._model_path_bytes
```

### 4. Memory Management

Image generation is memory-intensive - need proper cleanup:

```python
cdef class SDContext:
    def __dealloc__(self):
        if self._ctx != NULL:
            free_sd_ctx(self._ctx)
            self._ctx = NULL
```

## Complexity Assessment

| Aspect | Complexity | Notes |
|--------|------------|-------|
| API Surface | Medium | ~30 functions, ~15 types |
| Memory Management | Medium | Image buffers, context lifecycle |
| Callbacks | Medium | Progress/preview need GIL handling |
| Image I/O | Low | Simple numpy/PIL integration |
| Testing | Medium | Requires model files, GPU testing |

## Supported Models

The stable-diffusion.cpp library supports:

- **SD 1.x/2.x**: Standard Stable Diffusion models
- **SDXL**: Stable Diffusion XL
- **SD3/SD3.5**: Stable Diffusion 3.x
- **FLUX**: FLUX.1 models (dev, schnell)
- **Wan**: Video generation models
- **PhotoMaker**: Identity-preserving generation
- **ControlNet**: Conditional generation
- **LoRA**: Low-rank adaptation
- **ESRGAN**: Image upscaling

## Dependencies

Required (already in cyllama):
- ggml libraries (shared with llama.cpp)
- numpy

Optional:
- PIL/Pillow (for PIL Image conversion)
- tqdm (for progress bars in CLI)

## References

- [stable-diffusion.cpp repository](https://github.com/leejet/stable-diffusion.cpp)
- [stable-diffusion.h header](thirdparty/stable-diffusion.cpp/include/stable-diffusion.h)
- [cyllama whisper module](src/cyllama/whisper/) (implementation reference)
