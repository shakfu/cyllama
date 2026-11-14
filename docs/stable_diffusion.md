# Stable Diffusion Integration

Cyllama wraps [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) to provide image and video generation capabilities in Python.

**Note**: Build with `WITH_STABLEDIFFUSION=1` to enable this module. By default, stable-diffusion.cpp shares llama.cpp's ggml, so every extension runs the same ggml version. To link SD's own vendored ggml instead, set `SD_USE_VENDORED_GGML=1`; see [Stable Diffusion ggml selection](build_backends.md#stable-diffusion-ggml-selection).

## Overview

The stable diffusion module provides Python bindings to stable-diffusion.cpp, enabling:

- Text-to-image generation

- Image-to-image transformation

- Inpainting with masks

- ControlNet guided generation

- Video generation (with compatible models like Wan, CogVideoX)

- ESRGAN image upscaling

- Model format conversion

## Quick Start

### Text-to-Image

```python
from cyllama.sd import text_to_image

image = text_to_image(
    model_path="models/sd_xl_turbo_1.0.q8_0.gguf",
    prompt="a photo of a cute cat",
    width=512,
    height=512,
    sample_steps=4,
    cfg_scale=1.0
)

image.save("output.png")
```

### With Model Reuse

For generating multiple images, reuse the context:

```python
from cyllama.sd import SDContext, SDContextParams

params = SDContextParams()
params.model_path = "models/sd_xl_turbo_1.0.q8_0.gguf"

with SDContext(params) as ctx:
    for prompt in ["a cat", "a dog", "a bird"]:
        images = ctx.generate(
            prompt=prompt,
            sample_steps=4,
            cfg_scale=1.0
        )
        images[0].save(f"{prompt.replace(' ', '_')}.png")
```

## API Reference

### Convenience Functions

#### text_to_image()

Generate one image from a text prompt. `text_to_images()` takes the same arguments plus `batch_count: int = 1` and returns `List[SDImage]`.

```python
def text_to_image(
    model_path: str,
    prompt: str,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    seed: int = -1,
    sample_steps: int = 20,
    cfg_scale: float = 7.0,
    sample_method: SampleMethod = SampleMethod.COUNT,
    scheduler: Scheduler = Scheduler.COUNT,
    n_threads: int = -1,
    vae_path: str = None,
    taesd_path: str = None,
    clip_l_path: str = None,
    clip_g_path: str = None,
    t5xxl_path: str = None,
    control_net_path: str = None,
    clip_skip: int = -1,
    eta: float = float('inf'),
    slg_scale: float = 0.0,
    vae_tiling: bool = False,
    hires_fix: bool = False,
    hires_scale: float = 2.0,
    diffusion_flash_attn: bool = False,
    backend: str = None,
    params_backend: str = None
) -> SDImage
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | required | Path to model file |
| `prompt` | str | required | Text prompt |
| `negative_prompt` | str | "" | What to avoid |
| `width` | int | 512 | Output width |
| `height` | int | 512 | Output height |
| `seed` | int | -1 | Random seed (-1 = random) |
| `sample_steps` | int | 20 | Sampling steps |
| `cfg_scale` | float | 7.0 | CFG guidance scale |
| `sample_method` | SampleMethod | COUNT | Sampling method (COUNT = model default) |
| `scheduler` | Scheduler | COUNT | Noise scheduler (COUNT = model default) |
| `clip_skip` | int | -1 | CLIP layers to skip |
| `n_threads` | int | -1 | Thread count (-1 = auto) |
| `eta` | float | inf | Eta for samplers (inf = auto-resolve per method) |
| `slg_scale` | float | 0.0 | Skip layer guidance scale |
| `vae_tiling` | bool | False | Enable VAE tiling for large images |
| `hires_fix` | bool | False | Enable hires-fix two-pass generation (latent upscale) |
| `hires_scale` | float | 2.0 | Hires-fix upscale factor |
| `diffusion_flash_attn` | bool | False | Use flash attention |
| `backend` | str | None | Compute placement, e.g. `"diffusion=cuda0,te=cpu"` (see `SDContextParams`) |
| `params_backend` | str | None | Weight placement, e.g. `"te=cpu"` (see `SDContextParams`) |

#### image_to_image()

Transform an existing image with text guidance.

```python
def image_to_image(
    model_path: str,
    init_image: SDImage | str,
    prompt: str,
    negative_prompt: str = "",
    strength: float = 0.75,
    seed: int = -1,
    sample_steps: int = 20,
    cfg_scale: float = 7.0,
    ...
) -> List[SDImage]
```

The `strength` parameter (0.0-1.0) controls how much to transform the input image.

### SDContext

Main context class for model loading and generation.

```python
from cyllama.sd import SDContext, SDContextParams

params = SDContextParams()
params.model_path = "models/sd-v1-5.gguf"
params.n_threads = 4

ctx = SDContext(params)

# Check if loaded successfully
if ctx.is_valid:
    images = ctx.generate(
        prompt="a beautiful landscape",
        negative_prompt="blurry, ugly",
        width=512,
        height=512,
        sample_steps=20,
        cfg_scale=7.0,
        flow_shift=0.0
    )
```

**Methods:**

| Method | Description |
|--------|-------------|
| `generate(...)` | Generate images from text prompt |
| `generate_video(...)` | Generate video frames (requires video model) |
| `generate_with_params(params)` | Generate from a populated `SDImageGenParams` |
| `get_default_sample_method()` | Get model's default sampler |
| `get_default_scheduler(sample_method=None)` | Get model's default scheduler |
| `cancel(mode=CancelMode.ALL)` | Stop an in-flight generation from another thread (`ALL`, `NEW_LATENTS`, `RESET`) |
| `set_generation_progress_callback(cb)` | Per-context progress callback; `None` clears it (see below) |
| `load_control_net(path)` / `unload_control_net()` | Swap the ControlNet; not safe during a generation |
| `has_control_net` | `bool` -- a ControlNet is loaded |
| `is_valid` | Check if context is valid |
| `supports_image_generation` | `bool` — model can run `generate()` / txt2img (false for WAN-style video-only models) |
| `supports_video_generation` | `bool` — model can run `generate_video()` |

`generate()` also takes `vae_tile_size` (`(w, h)` in pixels) and `vae_tile_overlap`.

`set_generation_progress_callback()` passes an `SDGenerationProgress` namedtuple `(stage, detail, unit, sample_pass, completed, total)` to the callback. Import it from `cyllama.sd.stable_diffusion`. The call raises `RuntimeError` if the context is closed or busy.

### SDContextParams

Configuration for model loading.

```python
params = SDContextParams()

# Model paths
params.model_path = "model.gguf"              # Main model
params.diffusion_model_path = "unet.gguf"     # Diffusion model (for split models)
params.vae_path = "vae.safetensors"           # VAE model
params.clip_l_path = "clip_l.safetensors"     # CLIP-L (SDXL/SD3)
params.clip_g_path = "clip_g.safetensors"     # CLIP-G (SDXL/SD3)
params.clip_vision_path = "clip_vision.safetensors"  # CLIP vision
params.t5xxl_path = "t5xxl.safetensors"       # T5-XXL (SD3/FLUX)
params.llm_path = "qwen.gguf"                 # LLM encoder (FLUX2)
params.llm_vision_path = "qwen_vision.gguf"   # LLM vision encoder
params.taesd_path = "taesd.safetensors"       # TAESD for fast preview
params.control_net_path = "controlnet.gguf"   # ControlNet model
params.photo_maker_path = "photomaker.bin"    # PhotoMaker model
params.high_noise_diffusion_model_path = "..."  # High-noise model (Wan2.2 MoE)
params.tensor_type_rules = "^vae\\.=f16"      # Mixed precision rules

# Numeric/enum parameters
params.n_threads = 4                          # Thread count
params.wtype = SDType.COUNT                   # Weight type (COUNT = auto-detect)
params.rng_type = RngType.CUDA                # RNG type
params.sampler_rng_type = RngType.COUNT       # Sampler RNG type (COUNT = same as rng_type)
params.prediction = Prediction.COUNT          # Prediction type (COUNT = auto-detect)
params.lora_apply_mode = LoraApplyMode.AUTO   # LoRA application mode
params.vae_format = VaeFormat.AUTO            # VAE format (AUTO = detect from model)
params.conditioning_cache_size = 4            # Cached conditioning entries; 0 disables, negative raises ValueError
params.model_args = "chroma_use_dit_mask=0"   # Extra key=value loader args (Chroma, Qwen-Image)

# Memory placement (replaces the removed per-component CPU-offload flags)
params.max_vram = None                        # GiB per device: "N" cap, "-N" leave N free, "0"/None live free VRAM
params.backend = None                         # Compute placement, e.g. "diffusion=cuda0,te=cpu"
params.params_backend = None                  # Weight placement, e.g. "te=cpu"; "cpu"/"disk" valid
params.auto_fit = True                        # Place modules by free memory; non-empty params_backend disables it
params.split_mode = None                      # Multi-device split: "layer" (default), "row", or "diffusion=row"
params.eager_load = False                     # Load all params at load time, not lazily
params.pulid_weights_path = None              # Optional PuLID weights
params.rpc_servers = None                     # Optional RPC backends "host:port,..."

# Boolean flags
params.enable_mmap = True                     # Enable memory-mapped loading
params.diffusion_flash_attn = False           # Flash attention
params.diffusion_conv_direct = False          # Direct convolution
params.vae_conv_direct = False                # VAE direct convolution
params.tae_preview_only = False               # TAESD for preview only
params.batched_cfg = True                     # One batched forward for both CFG branches; no effect on Metal
```

`circular_x` / `circular_y` (tileable output) are `SDImageGenParams` properties.

### SDImage

Image wrapper with numpy and PIL integration, plus file I/O.

```python
from cyllama.sd import SDImage

# Load from file (PNG, JPEG, BMP, TGA, GIF, PSD, HDR, PIC supported)
img = SDImage.load("input.png")
img = SDImage.load("input.jpg", channels=3)  # Force RGB

# Properties
print(img.width, img.height, img.channels)
print(img.shape)   # (H, W, C)
print(img.is_valid)
print(img.generation_info)  # sdcpp.generation/v1 JSON str, or None for Python-built images

# Save to file (PNG, JPEG, BMP supported)
img.save("output.png")
img.save("output.jpg", quality=90)
img.save("output.bmp")

# Convert to numpy (requires numpy)
arr = img.to_numpy()  # Returns (H, W, C) uint8 array

# Create from numpy
img = SDImage.from_numpy(arr)

# Convert to PIL (requires Pillow)
pil_img = img.to_pil()
```

### SDImageGenParams

Detailed generation parameters for advanced control.

```python
from cyllama.sd import SDImageGenParams, SDImage

params = SDImageGenParams()
params.prompt = "a cute cat"
params.negative_prompt = "ugly, blurry"
params.width = 512
params.height = 512
params.seed = 42
params.batch_count = 1
params.strength = 0.75           # For img2img
params.clip_skip = -1
params.control_strength = 0.9    # ControlNet strength

# VAE tiling for large images
params.vae_tiling_enabled = True
params.vae_tile_size = (512, 512)   # image pixels; 0 = 256
params.vae_tile_overlap = 0.5

# EasyCache acceleration
params.easycache_enabled = True
params.easycache_threshold = 0.1
params.easycache_range = (0.0, 1.0)

# Hires-fix two-pass generation (one-shot helper)
from cyllama.sd import HiresUpscaler
params.set_hires_fix(
    enabled=True,
    upscaler=HiresUpscaler.LATENT,   # or LANCZOS, NEAREST, MODEL, ...
    scale=2.0,                        # ignored if target_width/height > 0
    denoising_strength=0.7,
    steps=0,                          # 0 = use base sample_steps
)
# ...or set fields individually:
# params.hires_enabled = True
# params.hires_upscaler = HiresUpscaler.LATENT_BICUBIC
# params.hires_target_size = (1024, 1024)
# params.hires_model_path = "/path/to/upscaler.gguf"  # required for HiresUpscaler.MODEL

# Set init image for img2img
init_img = SDImage.load("input.png")
params.set_init_image(init_img)

# Set mask for inpainting
mask_img = SDImage.load("mask.png")
params.set_mask_image(mask_img)

# Set control image for ControlNet
params.set_control_image(control_img, strength=0.8)

# Identity customization
# Photo Maker (requires SDContextParams.photo_maker_path):
params.set_pm_id_images([SDImage.load("face1.png"), SDImage.load("face2.png")])
params.pm_id_embed_path = "/path/to/id_embed.bin"  # optional precomputed embedding
params.pm_style_strength = 20.0
# PuLID (requires SDContextParams.pulid_weights_path):
params.pulid_id_embedding_path = "/path/to/id_embedding.bin"
params.pulid_id_weight = 1.0

# Access sample parameters
sample = params.sample_params
sample.sample_steps = 20
sample.cfg_scale = 7.0
sample.sample_method = SampleMethod.COUNT  # COUNT = auto-detect from model
sample.scheduler = Scheduler.KARRAS
sample.eta = float('inf')                  # inf = auto-resolve per method
sample.slg_scale = 2.5           # Skip layer guidance
sample.slg_layer_start = 0.01
sample.slg_layer_end = 0.2
sample.img_cfg_scale = 1.5       # Image CFG (inpaint)
sample.distilled_guidance = 3.5  # For FLUX
```

### SDSampleParams

Sampling configuration.

```python
from cyllama.sd import SDSampleParams, SampleMethod, Scheduler

params = SDSampleParams()
params.sample_method = SampleMethod.COUNT  # COUNT = auto-detect from model
params.scheduler = Scheduler.COUNT         # COUNT = auto-detect from model
params.sample_steps = 20
params.cfg_scale = 7.0
params.eta = float('inf')         # inf = auto-resolve per method
params.shifted_timestep = 0       # NitroFusion models
params.flow_shift = float('inf')  # inf = auto-detect (SD3.x/Wan)
params.img_cfg_scale = 1.5        # Image guidance
params.distilled_guidance = 3.5   # FLUX guidance
params.slg_scale = 0.0            # Skip layer guidance
params.slg_layer_start = 0.01
params.slg_layer_end = 0.2
```

### Upscaler

ESRGAN-based image upscaling.

```python
from cyllama.sd import Upscaler, SDImage

# Load upscaler model
upscaler = Upscaler(
    "models/esrgan-x4.bin",
    n_threads=4,
    direct=False
)

# Check upscale factor
print(f"Factor: {upscaler.upscale_factor}x")

# Upscale an image
img = SDImage.load("input.png")
upscaled = upscaler.upscale(img)
upscaled.save("upscaled.png")

# Multiple upscale passes
for _ in range(2):
    img = upscaler.upscale(img)  # 16x total
```

## Enums

### SampleMethod

Sampling methods for diffusion:

| Value | Description |
|-------|-------------|
| `EULER` | Euler method |
| `EULER_A` | Euler ancestral |
| `HEUN` | Heun's method |
| `DPM2` | DPM-2 |
| `DPMPP2S_A` | DPM++ 2S ancestral |
| `DPMPP2M` | DPM++ 2M |
| `DPMPP2Mv2` | DPM++ 2M v2 |
| `IPNDM` | IPNDM |
| `IPNDM_V` | IPNDM-V |
| `LCM` | Latent Consistency Model |
| `DDIM_TRAILING` | DDIM trailing |
| `TCD` | TCD |
| `RES_MULTISTEP` | Residual multistep |
| `RES_2S` | Residual 2S |
| `ER_SDE` | ER-SDE |
| `EULER_CFG_PP` | Euler CFG++ |
| `EULER_A_CFG_PP` | Euler ancestral CFG++ |
| `EULER_GE` | Euler GE |
| `DPMPP2M_SDE` | DPM++ 2M SDE |
| `DPMPP2M_SDE_BT` | DPM++ 2M SDE (upstream `dpm++2m_sde_bt`) |
| `LMS` | LMS |
| `COUNT` | Sentinel — auto-resolve per model (default) |

### Scheduler

Noise schedulers:

| Value | Description |
|-------|-------------|
| `DISCRETE` | Discrete scheduler |
| `KARRAS` | Karras scheduler |
| `EXPONENTIAL` | Exponential scheduler |
| `AYS` | AYS scheduler |
| `GITS` | GITS scheduler |
| `SGM_UNIFORM` | SGM uniform |
| `SIMPLE` | Simple scheduler |
| `SMOOTHSTEP` | Smoothstep scheduler |
| `KL_OPTIMAL` | KL-optimal scheduler |
| `LCM` | LCM scheduler |
| `BONG_TANGENT` | Bong-tangent scheduler |
| `LTX2` | LTX-2 scheduler |
| `LOGIT_NORMAL` | Logit-normal scheduler |
| `FLUX2` | FLUX.2 scheduler |
| `FLUX` | FLUX scheduler |
| `BETA` | Beta scheduler |
| `LLADA_IMAGE` | LLaDA image scheduler |
| `COUNT` | Sentinel — auto-resolve per model (default) |

### Prediction

Prediction types:

| Value | Description |
|-------|-------------|
| `EPS` | Epsilon prediction |
| `V` | V-prediction |
| `EDM_V` | EDM V-prediction |
| `FLOW` | SD3 flow matching (CLI `sd3_flow`) |
| `FLUX_FLOW` | FLUX flow matching |
| `SEFI_FLOW` | SEFI flow matching |
| `MINIT2I_FLOW` | MiniT2I flow matching |
| `SENSENOVA_U1_FLOW` | SenseNova U1 flow matching |
| `FLUX3_FLOW` | FLUX.3 flow matching |
| `H3_AV_FLOW` | H3 AV flow matching |
| `MING_IMAGE_FLOW` | Ming image flow matching |
| `COUNT` | Auto-detect from model (default) |

### SDType

Data types for quantization:

- Float: `F32`, `F16`, `BF16`

- 4-bit: `Q4_0`, `Q4_1`, `Q4_K`

- 5-bit: `Q5_0`, `Q5_1`, `Q5_K`

- 8-bit: `Q8_0`, `Q8_1`, `Q8_K`

- K-quants: `Q2_K`, `Q3_K`, `Q6_K`

- IQ quants: `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`

- Other: `TQ1_0`, `TQ2_0`, `MXFP4`, `NVFP4`, `Q1_0`, `Q2_0`, `F8_E4M3`, `F8_E5M2`

- Integer / F64: `I8`, `I16`, `I32`, `I64`, `F64`

- `COUNT`: auto-detect sentinel

### LoraApplyMode

LoRA application modes:

| Value | Description |
|-------|-------------|
| `AUTO` | Auto-detect best mode |
| `IMMEDIATELY` | Apply at load time |
| `AT_RUNTIME` | Apply during generation |

### HiresUpscaler

Upscaler modes for hires-fix two-pass generation. Set via `SDImageGenParams.hires_upscaler` or `set_hires_fix(upscaler=...)`.

| Value | Description |
|-------|-------------|
| `NONE` | Disabled |
| `LATENT` | Latent-space bilinear (default) |
| `LATENT_NEAREST` | Latent-space nearest |
| `LATENT_NEAREST_EXACT` | Latent-space nearest, exact |
| `LATENT_ANTIALIASED` | Latent-space bilinear, antialiased |
| `LATENT_BICUBIC` | Latent-space bicubic |
| `LATENT_BICUBIC_ANTIALIASED` | Latent-space bicubic, antialiased |
| `LANCZOS` | Pixel-space Lanczos |
| `NEAREST` | Pixel-space nearest |
| `MODEL` | External upscaler model (set `hires_model_path`) |

### PreviewMode

Preview modes during generation:

| Value | Description |
|-------|-------------|
| `NONE` | No preview |
| `PROJ` | Projection preview |
| `TAE` | TAESD preview |
| `VAE` | Full VAE preview |

### VaeFormat

VAE format for `SDContextParams.vae_format`: `AUTO` (detect from model, default), `FLUX`, `SD3`, `FLUX2`, `WAN`.

### CancelMode

Modes for `SDContext.cancel()`:

| Value | Description |
|-------|-------------|
| `ALL` | Stop as soon as possible (default) |
| `NEW_LATENTS` | Finish the current sample, skip the remaining batch latents |
| `RESET` | Clear a pending cancel request |

## Callbacks

Set callbacks for logging, progress, and preview during generation.

```python
from cyllama.sd import (
    set_log_callback,
    set_progress_callback,
    set_preview_callback,
    LogLevel,
    PreviewMode
)

# Log callback
def log_cb(level: LogLevel, text: str):
    level_names = {0: 'DEBUG', 1: 'VERBOSE', 2: 'INFO', 3: 'WARN', 4: 'ERROR'}
    print(f'[{level_names.get(level, level)}] {text}', end='')

set_log_callback(log_cb)

# Progress callback
def progress_cb(step: int, steps: int, time_ms: float):
    pct = (step / steps) * 100 if steps > 0 else 0
    print(f'Step {step}/{steps} ({pct:.1f}%) - {time_ms:.2f}s')

set_progress_callback(progress_cb)

# Preview callback
def preview_cb(step: int, frames: list, is_noisy: bool):
    if frames:
        frames[0].save(f"preview_{step}.png")

set_preview_callback(
    preview_cb,
    mode=PreviewMode.TAE,
    interval=5,
    denoised=True,
    noisy=False
)

# Clear callbacks
set_log_callback(None)
set_progress_callback(None)
set_preview_callback(None)
```

## Model Conversion

Convert models between formats with optional quantization.

```python
from cyllama.sd import convert_model, SDType

convert_model(
    input_path="sd-v1-5.safetensors",
    output_path="sd-v1-5-q4_0.gguf",
    output_type=SDType.Q4_0,
    vae_path="vae-ft-mse.safetensors",  # Optional
    tensor_type_rules="^vae\\.=f16"      # Optional mixed precision
)
```

## ControlNet Preprocessing

Apply Canny edge detection for ControlNet conditioning.

```python
from cyllama.sd import SDImage, canny_preprocess

img = SDImage.load("photo.png")

# Apply Canny preprocessing (modifies image in place)
success = canny_preprocess(
    img,
    high_threshold=0.8,
    low_threshold=0.1,
    weak=0.5,
    strong=1.0,
    inverse=False
)

img.save("edges.png")
```

## CLI Tool

Command-line interface with subcommands for all operations.

### txt2img - Text to Image

```bash
python -m cyllama.sd txt2img \
    --model models/sd_xl_turbo_1.0.q8_0.gguf \
    --prompt "a beautiful sunset" \
    --output sunset.png \
    --steps 4 --cfg-scale 1.0

# Using diffusion model directly (FLUX, etc.)
python -m cyllama.sd txt2img \
    --diffusion-model models/flux-dev.gguf \
    --vae models/ae.safetensors \
    --clip-l models/clip_l.safetensors \
    --t5xxl models/t5xxl.gguf \
    --prompt "a photo of a cat" \
    -W 1024 -H 1024

# With memory optimization
python -m cyllama.sd txt2img \
    --diffusion-model models/flux.gguf \
    --vae models/ae.safetensors \
    --llm models/qwen.gguf \
    --offload-to-cpu \
    --diffusion-fa \
    --prompt "a lovely cat" \
    -W 512 -H 1024
```

### img2img - Image to Image

```bash
python -m cyllama.sd img2img \
    --model models/sd-v1-5.gguf \
    --init-img input.png \
    --prompt "oil painting style" \
    --strength 0.7 \
    --output styled.png
```

### inpaint - Inpainting

```bash
python -m cyllama.sd inpaint \
    --model models/sd-inpaint.gguf \
    --init-img photo.png \
    --mask mask.png \
    --prompt "a red hat" \
    --output inpainted.png
```

### controlnet - ControlNet Guided Generation

```bash
python -m cyllama.sd controlnet \
    --model models/sd-v1-5.gguf \
    --control-net models/control_canny.gguf \
    --control-image edges.png \
    --prompt "a beautiful landscape" \
    --control-strength 0.9

# With automatic Canny preprocessing
python -m cyllama.sd controlnet \
    --model models/sd-v1-5.gguf \
    --control-net models/control_canny.gguf \
    --control-image photo.png \
    --canny \
    --prompt "anime style"
```

### video - Video Generation

```bash
# Text to video
python -m cyllama.sd video \
    --model models/wan2.1.gguf \
    --prompt "a cat walking" \
    --video-frames 16 \
    --fps 24

# Image to video
python -m cyllama.sd video \
    --model models/wan2.1.gguf \
    --init-img first_frame.png \
    --prompt "camera slowly zooms in" \
    --video-frames 24

# Frame interpolation
python -m cyllama.sd video \
    --model models/wan2.1.gguf \
    --init-img start.png \
    --end-img end.png \
    --video-frames 16
```

### upscale - Image Upscaling

```bash
python -m cyllama.sd upscale \
    --model models/esrgan-x4.bin \
    --input image.png \
    --output image_4x.png

# Multiple passes
python -m cyllama.sd upscale \
    --model models/esrgan-x4.bin \
    --input image.png \
    --output image_16x.png \
    --repeats 2
```

### convert - Model Conversion

```bash
python -m cyllama.sd convert \
    --input sd-v1-5.safetensors \
    --output sd-v1-5-q4_0.gguf \
    --type q4_0

# With VAE baking
python -m cyllama.sd convert \
    --input sdxl-base.safetensors \
    --output sdxl-q8_0.gguf \
    --type q8_0 \
    --vae sdxl-vae.safetensors
```

### info - System Information

```bash
python -m cyllama.sd info
```

### CLI Options Reference

**Model Options** (most subcommands):

| Option | Description |
|--------|-------------|
| `--model`, `-m` | Main model file |
| `--diffusion-model` | Diffusion model (for split architectures) |
| `--vae` | VAE model |
| `--taesd` | TAESD model (fast preview) |
| `--clip-l` | CLIP-L model (SDXL/SD3) |
| `--clip-g` | CLIP-G model (SDXL/SD3) |
| `--clip-vision` | CLIP vision model |
| `--t5xxl` | T5-XXL model (SD3/FLUX) |
| `--llm` | LLM text encoder (FLUX2) |
| `--llm-vision` | LLM vision encoder |
| `--high-noise-diffusion-model` | High-noise diffusion model (Wan2.2 MoE) |
| `--tensor-type-rules` | Per-tensor type rules, e.g. `"^vae\.=f16"` |
| `--tokenizer` | `tokenizer.json`; required for PiD and Lens |
| `--control-net` | ControlNet model (`controlnet` subcommand) |

**Generation Options**:

| Option | Description |
|--------|-------------|
| `--prompt`, `-p` | Text prompt |
| `--negative`, `-n` | Negative prompt |
| `--output`, `-o` | Output file path |
| `--width`, `-W` | Image width |
| `--height`, `-H` | Image height |
| `--steps` | Sampling steps |
| `--cfg-scale` | CFG guidance scale |
| `--seed`, `-s` | Random seed (-1 = random) |
| `--batch`, `-b` | Batch count |
| `--clip-skip` | CLIP layers to skip |

**Sampler Options**:

| Option | Description |
|--------|-------------|
| `--sampler` | Sampling method: upstream name (`dpm++2m`) or lowercase enum name (`dpmpp2m`) |
| `--scheduler` | Noise scheduler, same naming as `--sampler` |
| `--eta` | Eta for DDIM/TCD |
| `--rng` | RNG type (std_default, cuda, cpu) |
| `--sampler-rng` | Sampler RNG type (default: same as `--rng`) |
| `--prediction` | Prediction type override: `eps`, `v`, `edm_v`, `sd3_flow`, `flux_flow`, `sefi_flow`, `minit2i_flow`, `sensenova_u1_flow`, `flux3_flow`, `h3_av_flow`, `ming_image_flow` |
| `--flow-shift` | Flow shift (SD3.x/Wan) |

**Guidance Options**:

| Option | Description |
|--------|-------------|
| `--slg-scale` | Skip layer guidance scale |
| `--skip-layer-start` | SLG start point |
| `--skip-layer-end` | SLG end point |
| `--guidance` | Distilled guidance (FLUX) |
| `--img-cfg-scale` | Image CFG scale |

**Memory Options**:

| Option | Description |
|--------|-------------|
| `--threads`, `-t` | Thread count |
| `--max-vram` | Per-device GiB budget: `N` cap, `-N` leave N free, `0` live free VRAM |
| `--backend` | Compute placement, e.g. `diffusion=cuda0,te=cpu` |
| `--params-backend` | Weight placement; also accepts `cpu`/`disk` targets |
| `--auto-fit [on\|off]` | Place modules by free memory (default on) |
| `--eager-load` | Load all params at model-load time |
| `--disable-prefetch` | Disable async prefetch of the next segment's weights |
| `--disable-segmented-compute` | Force monolithic graphs |
| `--offload-to-cpu` | Alias for `--params-backend cpu` |
| `--clip-on-cpu` | Alias for `--params-backend te=cpu` |
| `--vae-on-cpu` | Alias for `--params-backend vae=cpu` |
| `--control-net-cpu` | Alias for `--params-backend control-net=cpu` |
| `--diffusion-fa` | Flash attention |
| `--diffusion-conv-direct` / `--vae-conv-direct` | Direct convolution |
| `--vae-tiling` | Enable VAE tiling |
| `--vae-tile-size WxH` | VAE tile size in pixels (default 256x256) |
| `--vae-tile-overlap` | VAE tile overlap fraction (default 0.5) |

**Other Options**:

| Option | Description |
|--------|-------------|
| `--verbose`, `-v` | Verbose output |
| `--progress` | Show progress |
| `--preview` | Preview mode (none, proj, tae, vae) |

## Supported Models

| Model Family | Examples | Notes |
|--------------|----------|-------|
| SD 1.x/2.x | sd-v1-5, sd-v2-1 | Standard models |
| SDXL | sdxl-base, sdxl-turbo | Use cfg_scale=1.0, steps=1-4 for turbo |
| SD3/SD3.5 | sd3-medium, sd3.5-large | May need T5-XXL encoder |
| FLUX | flux.1-dev, flux.1-schnell | Needs clip_l + t5xxl or llm |
| FLUX2 | flux2-* | Uses LLM encoder (Qwen) |
| Wan/CogVideoX | wan-2.1, cogvideox | Video generation |
| LoRA | *.safetensors | Load via `SDImageGenParams.set_loras()` |
| ControlNet | control_* | Use with control images |
| ESRGAN | esrgan-x4 | Upscaling only |

## Utility Functions

```python
from cyllama.sd import (
    get_num_cores,
    get_system_info,
    type_name,
    sample_method_name,
    scheduler_name
)

print(f"CPU cores: {get_num_cores()}")
print(get_system_info())
print(type_name(SDType.Q4_0))           # "q4_0"
print(sample_method_name(SampleMethod.EULER))  # "euler"
print(scheduler_name(Scheduler.KARRAS))  # "karras"
```

## Performance Tips

1. **Use turbo models** for fast generation (1-4 steps, cfg_scale=1.0)

2. **Quantize models** to Q4_0 or Q8_0 for memory efficiency

3. **Reuse SDContext** when generating multiple images

4. **Set n_threads** to match physical CPU cores

5. **Use `--offload-to-cpu`** for low VRAM GPUs

6. **Enable `--diffusion-fa`** (flash attention) for faster inference

7. **Use `--vae-tiling`** for generating large images

8. **Use progress callback** to track long generations

## Troubleshooting

### Model Loading Errors

```python
import os
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")
```

### Out of Memory

- Use smaller model (SD 1.5 vs SDXL)

- Use quantized model (Q4_0 vs F16)

- Reduce image dimensions

- Reduce batch_count

- Enable `--offload-to-cpu`

- Enable `--vae-tiling` for large images

### Slow Generation

- Use turbo/LCM models with fewer steps

- Enable flash attention (`--diffusion-fa`)

- Increase n_threads

- Use direct convolution (`--diffusion-conv-direct`)

### FLUX/SD3 Models Not Working

- Ensure you have the required encoders (clip_l, t5xxl)

- For FLUX2, use `--llm` instead of `--t5xxl`

- Check prediction type matches model

## See Also

- [stable-diffusion.cpp repository](https://github.com/leejet/stable-diffusion.cpp)

- [SDXL Turbo](https://huggingface.co/stabilityai/sdxl-turbo) - Fast generation model

- [API Reference](api_reference.md) - Detailed API documentation
