# cyllama - LLM Inference for Python

Fast, Pythonic LLM Inference -- Cyllama is a comprehensive no-dependencies Python library for LLM inference built on [llama.cpp](https://github.com/ggml-org/llama.cpp), the leading open-source C++ LLM inference engine. It combines the performance of compiled Cython wrappers with a simple, high-level Python API.

## Quick Start

```python
from cyllama import complete

# One line is all you need
response = complete(
    "Explain quantum computing in simple terms",
    model_path="models/llama.gguf",
    temperature=0.7,
    max_tokens=200
)
print(response)
```

## Key Features

### Simple by Default, Powerful When Needed

**High-Level API** - Get started in seconds:

```python
from cyllama import complete, chat, LLM

# One-shot completion
response = complete("What is Python?", model_path="model.gguf")

# Multi-turn chat
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"}
]
response = chat(messages, model_path="model.gguf")

# Reusable LLM instance (faster for multiple prompts)
llm = LLM("model.gguf")
response1 = llm("Question 1")
response2 = llm("Question 2")  # Model stays loaded!
```

**Streaming Support** - Real-time token-by-token output:

```python
for chunk in complete("Tell me a story", model_path="model.gguf", stream=True):
    print(chunk, end="", flush=True)
```

### Performance Optimized

**Batch Processing** - Process multiple prompts 3-10x faster:

```python
from cyllama import batch_generate

prompts = ["What is 2+2?", "What is 3+3?", "What is 4+4?"]
responses = batch_generate(prompts, model_path="model.gguf")
```

**Speculative Decoding** - 2-3x speedup with draft models:

```python
from cyllama import Speculative, SpeculativeParams

spec = Speculative(ctx_target, ctx_draft)
params = SpeculativeParams(n_draft=16, p_min=0.75)
draft_tokens = spec.gen_draft(params, prompt_tokens, last_token)
```

**Memory Optimization** - Smart GPU layer allocation:

```python
from cyllama import estimate_gpu_layers

estimate = estimate_gpu_layers(
    model_path="model.gguf",
    available_vram_mb=8000
)
print(f"Recommended GPU layers: {estimate.n_gpu_layers}")
```

### Framework Integrations

**OpenAI-Compatible API** - Drop-in replacement:

```python
from cyllama.integrations import OpenAIClient

client = OpenAIClient(model_path="model.gguf")

response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7
)
print(response.choices[0].message.content)
```

**LangChain Integration** - Seamless ecosystem access:

```python
from cyllama.integrations import CyllamaLLM
from langchain.chains import LLMChain

llm = CyllamaLLM(model_path="model.gguf", temperature=0.7)
chain = LLMChain(llm=llm, prompt=prompt_template)
result = chain.run(topic="AI")
```

### Advanced Features

**GGUF File Manipulation** - Inspect and modify model files:

```python
from cyllama import GGUFContext

ctx = GGUFContext.from_file("model.gguf")
metadata = ctx.get_all_metadata()
print(f"Model: {metadata['general.name']}")
```

**Structured Output** - JSON schema to grammar conversion:

```python
from cyllama import json_schema_to_grammar

schema = {"type": "object", "properties": {"name": {"type": "string"}}}
grammar = json_schema_to_grammar(schema)
```

**Model Downloads** - Ollama-style convenience:

```python
from cyllama import download_model, list_cached_models

download_model("bartowski/Llama-3.2-1B-Instruct-GGUF:q4")
models = list_cached_models()
```

**N-gram Cache** - 2-10x speedup for repetitive text:

```python
from cyllama import NgramCache

cache = NgramCache()
cache.update(tokens, ngram_min=2, ngram_max=4)
draft = cache.draft(input_tokens, n_draft=16)
```

## What's Inside

### Core Capabilities

- [x] **Full llama.cpp API** - Complete Cython wrapper with strong typing
- [x] **High-Level API** - Simple, Pythonic interface (`LLM`, `complete`, `chat`)
- [x] **Streaming Support** - Token-by-token generation with callbacks
- [x] **Batch Processing** - Efficient parallel inference
- [x] **GPU Acceleration** - Automatic Metal/CUDA/Vulkan backend support
- [x] **Memory Management** - Smart allocation and optimization tools

### Framework Support

- [x] **OpenAI API** - Compatible client for easy migration
- [x] **LangChain** - Full LLM interface implementation
- [x] **FastAPI/Flask** - Ready-to-use server examples
- [x] **Gradio** - Interactive UI integration

### Additional Features

- [x] **Multimodal** - LLAVA and vision-language models
- [x] **Whisper.cpp** - Speech-to-text transcription
- [x] **TTS** - Text-to-speech generation
- [x] **Speculative Decoding** - 2-3x inference speedup

## Why Cyllama?

**Performance**: Compiled Cython wrappers with minimal overhead

- Strong type checking at compile time
- Zero-copy data passing where possible
- Efficient memory management
- Native integration with llama.cpp optimizations

**Simplicity**: From 50 lines to 1 line for basic generation

- Intuitive, Pythonic API design
- Automatic resource management
- Sensible defaults, full control when needed

**Production-Ready**: Battle-tested and comprehensive

- 264 passing tests with extensive coverage
- Comprehensive documentation and examples
- Proper error handling and logging
- Framework integration for real applications

**Up-to-Date**: Tracks bleeding-edge llama.cpp (currently b6374)

- Regular updates with latest features
- All high-priority APIs wrapped
- Performance optimizations included

## Status

**Current Version**: 0.1.9 (November 2025)
**llama.cpp Version**: b6374
**Test Coverage**: 264 tests passing
**Platform**: macOS (primary), Linux (tested)

### API Coverage

| Component | Status | Description |
|-----------|--------|-------------|
| Core llama.cpp API | [x] Complete | Full wrapper with Cython classes |
| High-level generation | [x] Complete | Simple API with streaming |
| Batch processing | [x] Complete | Parallel inference utilities |
| GGUF manipulation | [x] Complete | Read/write model files |
| JSON schema grammar | [x] Complete | Structured output generation |
| Download helper | [x] Complete | HuggingFace/URL downloads |
| N-gram cache | [x] Complete | Pattern-based acceleration |
| Speculative decoding | [x] Complete | Draft model speedup |
| OpenAI compatibility | [x] Complete | Drop-in API replacement |
| LangChain integration | [x] Complete | Ecosystem access |
| Multimodal (LLAVA) | [x] Complete | Vision-language models |
| Whisper.cpp | [x] Complete | Speech-to-text |
| Memory optimization | [x] Complete | GPU layer estimation |
| Server implementations | [x] Complete | Embedded, Mongoose, Launcher |

### Recent Releases

**v0.1.9** (Nov 2025) - High-level APIs, integrations, batch processing, comprehensive documentation
**v0.1.8** (Nov 2025) - Speculative decoding API
**v0.1.7** (Nov 2025) - GGUF, JSON Schema, Downloads, N-gram Cache
**v0.1.6** (Nov 2025) - Multimodal test fixes
**v0.1.5** (Oct 2025) - Mongoose server, embedded server
**v0.1.4** (Oct 2025) - Memory estimation, performance optimizations

See [CHANGELOG.md](CHANGELOG.md) for complete release history.

## Setup

To build `cyllama`:

1. A recent version of `python3` (currently testing on python 3.13)

2. Git clone the latest version of `cyllama`:

    ```sh
    git clone https://github.com/shakfu/cyllama.git
    cd cyllama
    ```

3. Install dependencies of `cython`, `setuptools`, and `pytest` for testing:

    ```sh
    pip install -r requirements.txt
    ```

4. Type `make` in the terminal.

    This will:

    1. Download and build `llama.cpp`
    2. Install it into `bin`, `include`, and `lib` in the cloned `cyllama` folder
    3. Build `cyllama`

## Testing

The `tests` directory in this repo provides extensive examples of using cyllama.

However, as a first step, you should download a smallish llm in the `.gguf` model from [huggingface](https://huggingface.co/models?search=gguf). A good small model to start and which is assumed by tests is [Llama-3.2-1B-Instruct-Q8_0.gguf](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf). `cyllama` expects models to be stored in a `models` folder in the cloned `cyllama` directory. So to create the `models` directory if doesn't exist and download this model, you can just type:

```sh
make download
```

This basically just does:

```sh
cd cyllama
mkdir models && cd models
wget https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf
```

Now you can test it using `llama-cli` or `llama-simple`:

```sh
bin/llama-cli -c 512 -n 32 -m models/Llama-3.2-1B-Instruct-Q8_0.gguf \
 -p "Is mathematics discovered or invented?"
```

With 264 passing tests, the library is ready for both quick prototyping and production use:

```sh
make test  # Run full test suite
```

You can also explore interactively:

```python
python3 -i scripts/start.py

>>> from cyllama import complete
>>> response = complete("What is 2+2?", model_path="models/Llama-3.2-1B-Instruct-Q8_0.gguf")
>>> print(response)
```

## Documentation

- **[User Guide](docs/user_guide.md)** - Comprehensive guide covering all features
- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation
- **[Cookbook](docs/cookbook.md)** - Practical recipes and patterns
- **[Changelog](CHANGELOG.md)** - Complete release history
- **Examples** - See `tests/examples/` for working code samples

## Roadmap

### Completed

- [x] Full llama.cpp API wrapper with Cython
- [x] High-level API (`LLM`, `complete`, `chat`)
- [x] Batch processing utilities
- [x] OpenAI-compatible API client
- [x] LangChain integration
- [x] Speculative decoding
- [x] GGUF file manipulation
- [x] JSON schema to grammar conversion
- [x] Model download helper
- [x] N-gram cache
- [x] OpenAI-compatible servers (embedded, mongoose, launcher)
- [x] Whisper.cpp integration
- [x] Multimodal support (LLAVA)
- [x] Memory estimation utilities

### Future

- [ ] Async API support (`async def complete_async()`)
- [ ] Response caching for identical prompts
- [ ] Built-in prompt template system
- [ ] RAG utilities
- [ ] Web UI for testing
- [ ] Wrap [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)

## Contributing

Contributions are welcome! Please see the [User Guide](docs/user_guide.md) for development guidelines.

## License

This project wraps [llama.cpp](https://github.com/ggml-org/llama.cpp) and follows its licensing terms.
