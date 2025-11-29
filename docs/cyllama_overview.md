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
- **GPU acceleration** - Metal (macOS), CUDA, Vulkan backends
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

- Python 3.8+
- macOS or Linux
- GGUF model files (download from HuggingFace)

repo: <https://github.com/shakfu/cyllama>
