# cyllama overview

[cyllama](https://github.com/shakfu/cyllama) is a zero-dependency Python library for local LLM inference which uses cython to wrap the following high-performance inference engines:

- llama.cpp: text-to-text, text-to-speech and multimodel

- whisper.cpp: automatic speech recognition

- stable-diffusion.cpp: text-to-image and text-to-video

## Core Features

- **High-level API** - `complete()`, `chat()`, `LLM` class for quick prototyping

- **Low-level API** - Direct access to llama.cpp, whisper.cpp, and stable-diffusion.cpp internals

- **Streaming** - Token-by-token output with callbacks

- **Batch processing** - Process multiple prompts 3-10x faster

- **GPU acceleration** - Metal (macOS), CUDA (NVIDIA), ROCm (AMD), Vulkan (cross-platform) backends

- **Memory tools** - Estimate GPU layers and VRAM usage

- **OpenAI-compatible servers** - `EmbeddedServer` (C/Mongoose) and `PythonServer` implementations

## Agent Framework

- **ReActAgent** - Reasoning + Acting with tool calling

- **ConstrainedAgent** - Grammar-enforced tool calls (100% valid output)

- **ContractAgent** - Pre/post conditions on tools (C++26-inspired contracts)

## Additional Capabilities

- **Speculative decoding** - 2-3x speedup with draft models

- **GGUF utilities** - Read/write model metadata

- **JSON schema grammars** - Structured output generation

## Integrations

- **OpenAI-compatible API** - Drop-in client replacement

- **LangChain** - Full LLM interface implementation

- **ACP/MCP support** - Agent and Model Context Protocols

## Architecture

Cyllama is structured as a layered stack. At the bottom, three C/C++ inference engines handle the heavy computation. Cython bindings (`.pyx` files) expose these engines to Python with minimal overhead. On top of the bindings, a high-level API provides simple functions like `complete()` and `chat()`, while framework modules (agents, RAG, servers, integrations) compose these primitives into higher-level capabilities.

![Architecture Diagram](assets/architecture.svg)

### Layer Breakdown

| Layer | Components | Role |
|-------|-----------|------|
| **High-Level API** | `api.py`, `batching.py`, `memory.py` | Simple Python interface for generation, batch processing, and memory estimation |
| **Frameworks** | `agents/`, `rag/`, `integrations/`, `llama/server/` | ReAct/Constrained/Contract agents, RAG pipeline, OpenAI/LangChain compatibility, HTTP servers |
| **Cython Bindings** | `llama_cpp.pyx`, `whisper_cpp.pyx`, `stable_diffusion.pyx`, `mtmd.pxi` | Direct C++ bindings with `.pxd` declarations; includes speculative decoding and TTS extensions |
| **C/C++ Engines** | llama.cpp, whisper.cpp, stable-diffusion.cpp | Core inference: text generation, speech recognition, image generation |
| **Hardware Backends** | Metal, CUDA, Vulkan, CPU | GPU/CPU acceleration selected at build time |

### Data Flow

1. User calls a high-level function (e.g., `complete("prompt", model_path="model.gguf")`)
2. The API layer loads the model via Cython bindings, which allocate C++ context objects
3. Tokens are sampled in C++ and streamed back through Cython to Python callbacks
4. Framework modules (agents, RAG) orchestrate multiple calls to the API layer, adding tool use, retrieval, or structured output on top

### Key Design Decisions

- **Cython over ctypes/pybind11**: Cython provides near-zero overhead bindings while keeping build complexity manageable. The `.pxd` declaration files mirror C++ headers, and `.pxi` includes allow modular extension (speculative decoding, TTS, multimodal) without monolithic files.

- **Zero Python dependencies**: The core library has no runtime dependencies beyond Python itself. Optional integrations (LangChain, OpenAI compat) import lazily.

- **Dual server strategy**: `EmbeddedServer` wraps llama.cpp's built-in Mongoose-based HTTP server for maximum performance; `PythonServer` offers a pure-Python alternative for flexibility and debugging.

## Quick Example

```python
from cyllama import complete

response = complete(
    "Explain quantum computing in simple terms",
    model_path="models/llama.gguf",
    temperature=0.7
)
print(response)
```

## Requirements

- Python 3.10+

- macOS, Linux, or Windows

- GGUF model files (download from HuggingFace)

repo: <https://github.com/shakfu/cyllama>
