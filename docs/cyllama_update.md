# Cyllama Update

## What's New Since December 4, 2024

This document summarizes the major features, improvements, and changes to cyllama since the last announcement.

---

## Major New Features

### 1. High-Level API (`LLM`, `complete()`, `chat()`)

A complete rewrite of the user-facing API for simplicity and clarity:

```python
from cyllama import complete, chat, LLM

# Simple one-line completion
response = complete("What is Python?", model_path="model.gguf")

# Multi-turn chat
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing"}
]
response = chat(messages, model_path="model.gguf")

# Reusable LLM instance (faster for multiple prompts)
llm = LLM("model.gguf")
response1 = llm("First question")
response2 = llm("Second question")  # Model stays loaded!
```

**Key Benefits:**
- Automatic model lifecycle management
- Streaming support with token-by-token callbacks
- Configurable generation parameters (temperature, top-k, top-p, etc.)
- Built-in performance statistics
- Clean, Pythonic interface

### 2. Batch Processing

Efficient parallel inference for multiple prompts (3-10x throughput improvement):

```python
from cyllama import batch_generate, BatchGenerator

# Simple batch processing
prompts = ["What is 2+2?", "What is 3+3?", "What is 4+4?"]
responses = batch_generate(prompts, model_path="model.gguf")

# Advanced batch processing with detailed stats
batch_gen = BatchGenerator("model.gguf", batch_size=8)
results = batch_gen.generate_batch_detailed(requests)
```

**Performance:**
- Utilizes llama.cpp's native batching
- Shared model and KV cache across sequences
- Per-sequence statistics and error handling

### 3. Framework Integrations

#### OpenAI-Compatible API

Drop-in replacement for OpenAI's Python client:

```python
from cyllama.integrations import OpenAIClient

client = OpenAIClient(model_path="model.gguf")

response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7
)
print(response.choices[0].message.content)
```

**Features:**
- Full chat completions API compatibility
- Streaming support with proper chunking
- Compatible message format and response objects
- Usage statistics (prompt tokens, completion tokens)

#### LangChain Integration

Seamless integration with the LangChain ecosystem:

```python
from cyllama.integrations import CyllamaLLM
from langchain.chains import LLMChain

llm = CyllamaLLM(model_path="model.gguf", temperature=0.7)
chain = LLMChain(llm=llm, prompt=prompt_template)
result = chain.run(topic="AI")
```

**Features:**
- Full LLM interface implementation
- Works with chains, agents, and tools
- Streaming support with callback managers
- Proper error handling

### 4. Speculative Decoding

2-3x inference speedup using draft models:

```python
from cyllama import Speculative, SpeculativeParams

spec = Speculative(ctx_target, ctx_draft)
params = SpeculativeParams(n_draft=16, p_min=0.75)
draft_tokens = spec.gen_draft(params, prompt_tokens, last_token)
```

### 5. N-gram Cache

Pattern-based acceleration for repetitive text (2-10x speedup):

```python
from cyllama import NgramCache

cache = NgramCache()
cache.update(tokens, ngram_min=2, ngram_max=4)
draft = cache.draft(input_tokens, n_draft=16)
```

**Use Cases:**
- Code completion with repeated patterns
- Template-based text generation
- Structured output generation

### 6. GGUF File Manipulation

Inspect and modify GGUF model files:

```python
from cyllama import GGUFContext

# Read model metadata
ctx = GGUFContext.from_file("model.gguf")
metadata = ctx.get_all_metadata()
print(f"Model: {metadata['general.name']}")

# Create new GGUF files
writer = GGUFContext()
writer.set_metadata("custom.key", "value")
writer.write_to_file("custom.gguf")
```

### 7. JSON Schema to Grammar

Generate structured output with type safety:

```python
from cyllama import json_schema_to_grammar

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    }
}
grammar = json_schema_to_grammar(schema)
# Use grammar in generation to ensure valid JSON output
```

### 8. Model Download Helper

Ollama-style convenience for downloading models:

```python
from cyllama import download_model, list_cached_models

# Download from HuggingFace
download_model("bartowski/Llama-3.2-1B-Instruct-GGUF:q4")

# List cached models
models = list_cached_models()
```

### 9. Memory Optimization

Smart GPU layer allocation based on available VRAM:

```python
from cyllama import estimate_gpu_layers

estimate = estimate_gpu_layers(
    model_path="model.gguf",
    available_vram_mb=8000
)
print(f"Recommended GPU layers: {estimate.n_gpu_layers}")
```

---

## API Improvements

### Renamed for Clarity (Latest Changes)

To improve consistency and clarity, the core API has been renamed:

| Old Name | New Name | Purpose |
|----------|----------|---------|
| `generate()` | `complete()` | Text completion function |
| `Generator` | `LLM` | Main LLM class |
| `generate.py` | `api.py` | Unified API module |

**Example:**
```python
# Old API (still works for compatibility)
from cyllama import generate, Generator

# New API (recommended)
from cyllama import complete, LLM
```

### Logging Improvements

Debug output is now disabled by default for a cleaner experience:

```python
# Quiet by default
llm = LLM("model.gguf")

# Enable detailed logging when needed
llm = LLM("model.gguf", verbose=True)
```

### Cleaner Integration Imports

```python
# OpenAI integration - shorter import path
from cyllama.integrations import OpenAIClient  # New!
# vs
from cyllama.integrations.openai_compat import OpenAICompatibleClient  # Old

# Both work for backward compatibility
```

---

## Documentation

Comprehensive documentation added:

1. **User Guide** (`docs/user_guide.md`) - 450+ lines covering all APIs
2. **API Reference** (`docs/api_reference.md`) - Complete API documentation
3. **Cookbook** (`docs/cookbook.md`) - 350+ lines of practical recipes
4. **Improvements Summary** (`docs/improvements_summary.md`) - Detailed feature overview

**Topics Covered:**
- Text generation patterns
- Chat application examples
- Structured output generation
- Performance optimization
- Framework integration examples
- FastAPI/Flask/Gradio servers
- Error handling and best practices

---

## Testing

Comprehensive test coverage ensures reliability:

- **253 tests passing** (32 skipped for optional features)
- Unit tests for all new APIs
- Integration tests with real models
- Edge case testing
- Performance validation

---

## What This Means For You

### For Quick Prototyping

```python
from cyllama import complete

# One line to get started
response = complete("Your prompt", model_path="model.gguf")
```

### For Production Applications

```python
from cyllama import LLM, GenerationConfig

# Efficient model reuse
llm = LLM("model.gguf")
config = GenerationConfig(temperature=0.7, max_tokens=100)

# Fast repeated inference
for prompt in prompts:
    response = llm(prompt, config=config)
```

### For Framework Integration

```python
# OpenAI compatibility
from cyllama.integrations import OpenAIClient

# LangChain integration
from cyllama.integrations import CyllamaLLM

# Works seamlessly with existing code
```

### For Performance

```python
# Batch processing for throughput
from cyllama import batch_generate

# Speculative decoding for speed
from cyllama import Speculative

# N-gram cache for patterns
from cyllama import NgramCache
```

---

## Migration Guide

If you were using the old API:

```python
# Old code
from cyllama import generate
response = generate(prompt, model_path="model.gguf")

# New code (recommended, but old still works)
from cyllama import complete
response = complete(prompt, model_path="model.gguf")
```

```python
# Old code
from cyllama import Generator
gen = Generator("model.gguf")

# New code (recommended, but old still works)
from cyllama import LLM
llm = LLM("model.gguf")
```

**Note:** All old APIs remain available for backward compatibility.

---

## Performance Highlights

- **Model Reuse:** 5-10s saved per generation by keeping model loaded
- **Batch Processing:** 3-10x throughput improvement for multiple prompts
- **Speculative Decoding:** 2-3x speedup with draft models
- **N-gram Cache:** 2-10x speedup for repetitive patterns
- **Memory Optimization:** Smart GPU layer allocation for available VRAM

---

## Technical Details

### llama.cpp Version
Currently tracking llama.cpp **b7126** (November 2054)

### Platform Support
- macOS (primary, fully tested)
- Linux (tested)
- GPU acceleration via Metal/CUDA/Vulkan

### Python Version
Python 3.13+ (tested on 3.13.2)

---

## Status

**Current Version:** 0.1.9 (November 21, 2025)

**API Coverage:**
- [x] Core llama.cpp wrapper (complete)
- [x] High-level API (complete)
- [x] Batch processing (complete)
- [x] Framework integrations (complete)
- [x] GGUF manipulation (complete)
- [x] JSON schema grammar (complete)
- [x] Model downloads (complete)
- [x] Speculative decoding (complete)
- [x] N-gram cache (complete)
- [x] Memory optimization (complete)
- [x] Multimodal (LLAVA) (complete)
- [x] Whisper.cpp (complete)

---

## Getting Started

1. Clone and build:
```bash
git clone https://github.com/shakfu/cyllama.git
cd cyllama
pip install -r requirements.txt
make
```

2. Download a model:
```bash
make download
```

3. Try it out:
```python
from cyllama import complete

response = complete(
    "What is Python?",
    model_path="models/Llama-3.2-1B-Instruct-Q8_0.gguf"
)
print(response)
```

---

## Learn More

- **Repository:** https://github.com/shakfu/cyllama
- **Documentation:** See `docs/` directory
- **Examples:** See `tests/examples/` directory

---

## What's Next

Potential future features:
- Async API support
- Response caching
- Built-in prompt templates
- RAG utilities
- Web UI for testing

---

**Questions or feedback?** Feel free to open an issue on GitHub!
