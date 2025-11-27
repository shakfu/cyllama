# cyllama Update - November 2025 (v0.1.12)

## What's New in cyllama

I'm excited to share the latest updates to [cyllama](https://github.com/shakfu/cyllama) - the comprehensive Python library for LLM inference built on llama.cpp. This release brings two major new capabilities: a zero-dependency **Agent Framework** and **Stable Diffusion** image generation support.

A quick reminder: cyllama is a performant, compiled Cython wrapper for llama.cpp that provides both low-level access and a high-level Pythonic API. It statically links the core libraries for simplicity and performance.

---

## Major New Features

### 1. Agent Framework (Zero Dependencies)

cyllama now includes a complete agent framework with three agent architectures, all with zero external dependencies:

**ReActAgent** - Reasoning + Acting agent with tool calling:

```python
from cyllama import LLM
from cyllama.agents import ReActAgent, tool

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

llm = LLM("model.gguf")
agent = ReActAgent(llm=llm, tools=[calculate, search])
result = agent.run("What is 25 * 4 + 10?")
print(result.answer)  # "The result is 110"
```

**ConstrainedAgent** - Grammar-enforced tool calling for 100% reliability:

```python
from cyllama.agents import ConstrainedAgent

# Uses GBNF grammars to guarantee valid JSON tool calls
agent = ConstrainedAgent(llm=llm, tools=[calculate])
result = agent.run("Calculate 100 / 4")  # Always produces valid tool calls
```

**ContractAgent** - Contract-based agent with pre/post conditions (C++26-inspired):

```python
from cyllama.agents import ContractAgent, tool, pre, post, ContractPolicy

@tool
@pre(lambda args: args['x'] != 0, "cannot divide by zero")
@post(lambda r: r is not None, "result must not be None")
def divide(a: float, x: float) -> float:
    """Divide a by x."""
    return a / x

agent = ContractAgent(
    llm=llm,
    tools=[divide],
    policy=ContractPolicy.ENFORCE,  # AUDIT, ENFORCE, or DISABLED
    task_precondition=lambda task: len(task) > 10,
    answer_postcondition=lambda ans: len(ans) > 0,
)
result = agent.run("What is 100 divided by 4?")
```

**Key Features:**
- Zero external dependencies - uses only Python stdlib
- Three agent architectures for different use cases
- `@tool` decorator for easy function registration
- Automatic JSON schema generation from type hints
- Grammar-constrained generation for reliable tool calls
- Contract-based validation with configurable policies
- Comprehensive audit logging

See [contract_agent.md](contract_agent.md) for detailed ContractAgent documentation.

---

### 2. Stable Diffusion Integration

Full integration of [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) for image and video generation:

**Simple Text-to-Image:**

```python
from cyllama.stablediffusion import text_to_image

images = text_to_image(
    model_path="models/sd_xl_turbo_1.0.q8_0.gguf",
    prompt="a photo of a cute cat sitting on a windowsill",
    width=512,
    height=512,
    sample_steps=4,  # Turbo models need fewer steps
    cfg_scale=1.0
)
images[0].save("output.png")
```

**Advanced Generation with SDContext:**

```python
from cyllama.stablediffusion import (
    SDContext, SDContextParams,
    SampleMethod, Scheduler,
    set_progress_callback
)

# Progress tracking
def progress_cb(step, steps, time_ms):
    pct = (step / steps) * 100
    print(f'Step {step}/{steps} ({pct:.1f}%)')

set_progress_callback(progress_cb)

# Create context with full control
params = SDContextParams()
params.model_path = "models/sd_xl_turbo_1.0.q8_0.gguf"
params.n_threads = 4
params.vae_path = "models/vae.safetensors"  # Optional

ctx = SDContext(params)
images = ctx.generate(
    prompt="a beautiful mountain landscape at sunset",
    negative_prompt="blurry, ugly, distorted",
    width=512,
    height=512,
    sample_method=SampleMethod.EULER,
    scheduler=Scheduler.DISCRETE,
    seed=42
)
```

**Image-to-Image:**

```python
from cyllama.stablediffusion import image_to_image, SDImage

init_img = SDImage.load("input.png")
images = image_to_image(
    model_path="models/sd_xl_turbo_1.0.q8_0.gguf",
    init_image=init_img,
    prompt="make it a watercolor painting",
    strength=0.75
)
```

**ESRGAN Upscaling:**

```python
from cyllama.stablediffusion import Upscaler, SDImage

upscaler = Upscaler("models/esrgan-x4.bin")
img = SDImage.load("small.png")
upscaled = upscaler.upscale(img)  # 4x resolution
upscaled.save("large.png")
```

**ControlNet with Canny Preprocessing:**

```python
from cyllama.stablediffusion import SDImage, canny_preprocess

img = SDImage.load("photo.png")
canny_preprocess(img, high_threshold=0.8, low_threshold=0.1)
# Use img as control image for ControlNet generation
```

**CLI Tool:**

```bash
# Generate image
python -m cyllama.stablediffusion generate \
    --model models/sd_xl_turbo_1.0.q8_0.gguf \
    --prompt "a beautiful sunset over mountains" \
    --output sunset.png \
    --steps 4 --cfg 1.0 --progress

# Upscale image
python -m cyllama.stablediffusion upscale \
    --model models/esrgan-x4.bin \
    --input image.png \
    --output image_4x.png

# Convert model format
python -m cyllama.stablediffusion convert \
    --input sd-v1-5.safetensors \
    --output sd-v1-5-q4_0.gguf \
    --type q4_0

# Show system info
python -m cyllama.stablediffusion info
```

**Supported Models:**
- SD 1.x/2.x - Standard Stable Diffusion
- SDXL/SDXL Turbo - High-quality generation (use cfg_scale=1.0, steps=1-4 for Turbo)
- SD3/SD3.5 - Latest Stable Diffusion 3.x
- FLUX - FLUX.1 models (dev, schnell)
- Wan/CogVideoX - Video generation (use `generate_video()`)
- LoRA - Low-rank adaptation files
- ControlNet - Conditional generation
- ESRGAN - Image upscaling

**Key Features:**
- Full numpy/PIL integration (`SDImage.to_numpy()`, `SDImage.to_pil()`)
- Progress, log, and preview callbacks
- All samplers (Euler, Euler_A, DPM2, DPMPP2M, LCM, etc.)
- All schedulers (Discrete, Karras, Exponential, AYS, etc.)
- Model conversion with quantization support
- Video generation for compatible models

---

### 3. Agent Client Protocol (ACP) Support

New ACP implementation for editor/IDE integration:

```python
from cyllama.agents import ACPAgent

# ACP agent for editor integration (Zed, Neovim, etc.)
agent = ACPAgent(model_path="model.gguf")
agent.run()  # Starts JSON-RPC server over stdio
```

**Features:**
- JSON-RPC 2.0 transport over stdio
- Session management (new, load, prompt, cancel)
- Tool permission flow for user approval
- File operations delegated to editor
- Terminal operations support

---

## Current Status

**Version:** 0.1.12 (November 2025)
**llama.cpp Version:** b7126 (tracking bleeding-edge)
**stable-diffusion.cpp:** Integrated and tested
**whisper.cpp:** Integrated and tested
**Tests:** 600+ passing
**Platform:** macOS (primary), Linux (tested)

**API Coverage - All Major Goals Met:**

| Component | Status |
|-----------|--------|
| Core llama.cpp wrapper | Complete |
| High-level Python API | Complete |
| Agent Framework | Complete |
| Stable Diffusion | Complete |
| Multimodal (LLAVA) | Complete |
| Whisper.cpp | Complete |
| TTS | Complete |
| HTTP Servers | Complete |
| Framework Integrations | Complete |

---

## Why This Update Matters

**Agents without dependencies:** Build tool-using AI agents with just cyllama - no LangChain, no AutoGen, no external frameworks required. Three architectures cover different reliability/flexibility tradeoffs.

**Image generation in Python:** Generate images with the same library you use for LLM inference. Full control over samplers, schedulers, and all generation parameters. Support for the latest models including SDXL Turbo, SD3, and FLUX.

**Production-ready:** 600+ tests, comprehensive documentation, proper error handling. Ready for both quick prototyping and production use.

---

## Quick Start Examples

```python
# Text generation
from cyllama import complete
response = complete("What is Python?", model_path="model.gguf")

# Agent with tools
from cyllama import LLM
from cyllama.agents import ReActAgent, tool

@tool
def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny, 72F"

agent = ReActAgent(llm=LLM("model.gguf"), tools=[get_weather])
result = agent.run("What's the weather in Paris?")

# Image generation
from cyllama.stablediffusion import text_to_image
images = text_to_image(
    model_path="sd_xl_turbo.gguf",
    prompt="a cyberpunk cityscape",
    sample_steps=4
)
images[0].save("cityscape.png")

# Speech transcription
from cyllama.whisper import WhisperContext
ctx = WhisperContext("whisper-base.bin")
result = ctx.transcribe("audio.wav")
print(result.text)
```

---

## Resources

- **Repo:** <https://github.com/shakfu/cyllama>
- **Docs:** See `docs/` directory
  - [User Guide](user_guide.md)
  - [API Reference](api_reference.md)
  - [Contract Agent Guide](contract_agent.md)
  - [Cookbook](cookbook.md)
- **Examples:** See `tests/examples/` directory
  - Agent examples (`agent_*.py`)
  - Stable Diffusion examples (`stablediffusion_*.py`)
  - Server implementations
  - Multimodal demos

---

## What's Next?

Potential future work:
- Async API support (`async def complete_async()`)
- Response caching for identical prompts
- RAG utilities
- Web UI for testing

---

## Feedback Welcome

As always, feedback is appreciated:
- Questions? Ask away!
- Bugs? Please report them!
- Features? Suggestions welcome!
- Contributions? Pull requests accepted!

The goal remains: stay lean, stay fast, stay current with llama.cpp, and make it easy to use from Python.
