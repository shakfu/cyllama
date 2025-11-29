# cyllama Update - November 2025 (v0.1.15)

## What's New in cyllama

This update covers versions 0.1.13 through 0.1.15 of [cyllama](https://github.com/shakfu/cyllama). These releases focus on robustness, security, and developer experience improvements.

A quick reminder: cyllama is a performant, compiled zer-dependency Cython wrapper for llama.cpp, whisper.cpp, and stable-diffusion.cpp that provides both low-level access and a high-level Pythonic API. It statically links the core libraries for simplicity and performance.

---

## Highlights

### Security & Stability (v0.1.15)

Critical input validation added to prevent crashes and security issues:

- **Buffer Overflow Prevention** - `get_state_seq_data()` and `get_state_seq_data_with_flags()` now dynamically allocate buffers based on actual required size instead of using fixed 512-byte stack buffers
- **File Path Validation** - LoRA adapter and state file functions now validate paths before passing to C code
- **NULL Pointer Protection** - `LlamaContext` validates model before initialization, preventing segfaults

### Zero-Dependency Image I/O (v0.1.13)

Native PNG/JPEG/BMP support without PIL:

```python
from cyllama.stablediffusion import SDImage

# Load any common format
img = SDImage.load("photo.jpg")  # PNG, JPEG, BMP, TGA, GIF, PSD, HDR, PIC

# Save without external dependencies
img.save_png("output.png")
img.save_jpg("output.jpg", quality=90)
img.save_bmp("output.bmp")
```

### Enhanced Resource Management (v0.1.14)

Improved context lifecycle and memory management:

```python
from cyllama import LLM

# Context manager for automatic cleanup
with LLM("model.gguf") as llm:
    response = llm("Hello!")
    # Context is cached and reused when size permits

# Or explicit cleanup
llm = LLM("model.gguf")
llm.close()  # Explicit resource cleanup
```

### Robust Agent Framework (v0.1.14)

Significant improvements to agent reliability:

**ReActAgent** now handles malformed LLM outputs gracefully:

```python
from cyllama.agents import ReActAgent, ActionParseError

# Multi-strategy parsing handles common LLM variations:
# - Trailing commas in JSON: {"key": "value",}
# - Single-quoted JSON: {'key': 'value'}
# - Escaped quotes within values
# - Key=value pairs and positional arguments

# ActionParseError provides helpful debugging info
try:
    result = agent.run("Some task")
except ActionParseError as e:
    print(e.message)     # Human-readable error
    print(e.suggestion)  # How to fix it
    print(e.details)     # All parsing attempts made
```

**Enhanced Tool Type System:**

```python
from typing import List, Dict, Optional, Literal
from cyllama.agents import tool

@tool
def process_data(
    items: List[Dict[str, int]],
    mode: Literal["fast", "accurate"],
    limit: Optional[int] = None
) -> Dict[str, List[str]]:
    """Process data items.

    Args:
        items: List of data dictionaries
        mode: Processing mode to use
        limit: Optional result limit
    """
    ...

# Full JSON schema generation for complex types:
# - List[T], Dict[K, V], Optional[T], Union[A, B]
# - Tuple[A, B], Set[T], Literal["a", "b"]
# - Nested generics like List[Dict[str, int]]
# - Docstring parsing: Google, NumPy, Sphinx, Epytext styles
```

---

## All Changes by Version

### v0.1.15 - Security

- **Cython Input Validation**
  - Fixed buffer overflow in `get_state_seq_data()` / `get_state_seq_data_with_flags()`
  - File path validation for `lora_adapter_init()`, `load_state_file()`, `save_state_file()`, `load_state_seq_file()`, `save_state_seq_file()`
  - NULL pointer check in `LlamaContext.__init__`

### v0.1.14 - Major Quality Release

**Fixed:**
- Ensure Python 3.8-3.9 compatibility (type hint syntax)
- Bare except clauses replaced with specific exceptions
- Silent Unicode errors now logged with warnings
- Progress callback crash on `LlamaModelParams`
- MCP race condition in `send_notification()`
- LLM destructor safety for partial initialization

**Added:**
- `GenerationConfig` parameter validation (11 tests)
- Sampler `add_logit_bias()` implementation
- MCP configurable timeouts (`request_timeout`, `shutdown_timeout`)
- Thread safety in `color.py` for global settings
- Session storage error handling improvements
- LLM context reuse and explicit `close()` method
- BatchGenerator resource management and validation
- ReActAgent robust parsing with `ActionParseError` (28 tests)
- Tool type system with full generic support (23 tests)
- ContractAgent documentation and tests (28 tests)
- Comprehensive test suite `test_comprehensive.py` (53 tests)
- Benchmark script (`scripts/benchmark.py`)
- Batch memory pooling integration

**Changed:**
- Centralized model path configuration via `conftest.py`
- Memory module improvements with documented magic numbers
- Stop sequence logic simplified and fixed

### v0.1.13 - Image I/O

**Added:**
- Zero-dependency image I/O via bundled stb library
- `SDImage.save_png()`, `save_jpg()`, `save_bmp()`, `save_ppm()`
- `SDImage.load()` for PNG, JPEG, BMP, TGA, GIF, PSD, HDR, PIC
- Channel conversion support on load

**Changed:**
- Build scripts now handle stb headers consistently
- Added version constants for whisper.cpp and stable-diffusion.cpp

---

## Current Status

**Version:** 0.1.15 (November 2025)
**llama.cpp Version:** b7126 (tracking bleeding-edge)
**stable-diffusion.cpp:** Integrated and tested
**whisper.cpp:** Integrated and tested
**Tests:** 800+ passing
**Platform:** macOS (primary), Linux (tested)

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

# Image generation (no PIL required)
from cyllama.stablediffusion import text_to_image
images = text_to_image(
    model_path="sd_xl_turbo_1.0.q8_0.gguf",
    prompt="a cyberpunk cityscape",
    sample_steps=4
)
images[0].save_png("cityscape.png")  # Native PNG support

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

## Feedback Welcome

As always, feedback is appreciated:
- Questions? Ask away!
- Bugs? Please report them!
- Features? Suggestions welcome!
- Contributions? Pull requests accepted!
