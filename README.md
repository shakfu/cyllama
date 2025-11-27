# cyllama - Fast, Pythonic LLM Inference

cyllama is a comprehensive no-dependencies Python library for LLM inference built on [llama.cpp](https://github.com/ggml-org/llama.cpp), the leading open-source C++ LLM inference engine. It combines the performance of compiled Cython wrappers with a simple, high-level Python API.

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

### Agent Framework

Cyllama includes a zero-dependency agent framework with three agent architectures:

**ReActAgent** - Reasoning + Acting agent with tool calling:

```python
from cyllama import LLM
from cyllama.agents import ReActAgent, tool

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

llm = LLM("model.gguf")
agent = ReActAgent(llm=llm, tools=[calculate])
result = agent.run("What is 25 * 4?")
print(result.answer)
```

**ConstrainedAgent** - Grammar-enforced tool calling for 100% reliability:

```python
from cyllama.agents import ConstrainedAgent

agent = ConstrainedAgent(llm=llm, tools=[calculate])
result = agent.run("Calculate 100 / 4")  # Guaranteed valid tool calls
```

**ContractAgent** - Contract-based agent with C++26-inspired pre/post conditions:

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
    policy=ContractPolicy.ENFORCE,
    task_precondition=lambda task: len(task) > 10,
    answer_postcondition=lambda ans: len(ans) > 0,
)
result = agent.run("What is 100 divided by 4?")
```

See [contract_agent.md](docs/contract_agent.md) for detailed `ContractAgent` documentation.

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

**Huggingface Model Downloads**:

```python
from cyllama import download_model, list_cached_models, get_hf_file

# Download from HuggingFace (saves to ~/.cache/llama.cpp/)
download_model("bartowski/Llama-3.2-1B-Instruct-GGUF:latest")

# Or with explicit parameters
download_model(hf_repo="bartowski/Llama-3.2-1B-Instruct-GGUF:latest")

# Download specific file to custom path
download_model(
    hf_repo="bartowski/Llama-3.2-1B-Instruct-GGUF",
    hf_file="Llama-3.2-1B-Instruct-Q8_0.gguf",
    model_path="./models/my_model.gguf"
)

# Get file info without downloading
info = get_hf_file("bartowski/Llama-3.2-1B-Instruct-GGUF:latest")
print(info)  # {'repo': '...', 'gguf_file': '...', 'mmproj_file': '...'}

# List cached models
models = list_cached_models()
```

**N-gram Cache** - 2-10x speedup for repetitive text:

```python
from cyllama import NgramCache

cache = NgramCache()
cache.update(tokens, ngram_min=2, ngram_max=4)
draft = cache.draft(input_tokens, n_draft=16)
```

### Stable Diffusion

**Image Generation** - Generate images from text using stable-diffusion.cpp:

```python
from cyllama.stablediffusion import text_to_image

# Simple text-to-image
images = text_to_image(
    model_path="models/sd_xl_turbo_1.0.q8_0.gguf",
    prompt="a photo of a cute cat",
    width=512,
    height=512,
    sample_steps=4,
    cfg_scale=1.0
)
images[0].save("output.png")
```

**Advanced Generation** - Full control with SDContext:

```python
from cyllama.stablediffusion import SDContext, SDContextParams, SampleMethod, Scheduler

params = SDContextParams()
params.model_path = "models/sd_xl_turbo_1.0.q8_0.gguf"
params.n_threads = 4

ctx = SDContext(params)
images = ctx.generate(
    prompt="a beautiful mountain landscape",
    negative_prompt="blurry, ugly",
    width=512,
    height=512,
    sample_method=SampleMethod.EULER,
    scheduler=Scheduler.DISCRETE
)
```

**CLI Tool** - Command-line interface:

```bash
# Generate image
python -m cyllama.stablediffusion generate \
    --model models/sd_xl_turbo_1.0.q8_0.gguf \
    --prompt "a beautiful sunset" \
    --output sunset.png

# Show system info
python -m cyllama.stablediffusion info
```

Supports SD 1.x/2.x, SDXL, SD3, FLUX, video generation (Wan/CogVideoX), LoRA, ControlNet, and ESRGAN upscaling. See [API Reference](docs/api_reference.md#stable-diffusion-integration) for full documentation.

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

### Agent Framework

- [x] **ReActAgent** - Reasoning + Acting agent with tool calling
- [x] **ConstrainedAgent** - Grammar-enforced tool calling for 100% reliability
- [x] **ContractAgent** - Contract-based agent with pre/post conditions

### Additional Features

- [x] **Multimodal** - LLAVA and vision-language models
- [x] **Whisper.cpp** - Speech-to-text transcription
- [x] **Stable Diffusion** - Image generation with stable-diffusion.cpp
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

- 600+ passing tests with extensive coverage
- Comprehensive documentation and examples
- Proper error handling and logging
- Framework integration for real applications

**Up-to-Date**: Tracks bleeding-edge llama.cpp (currently b7126)

- Regular updates with latest features
- All high-priority APIs wrapped
- Performance optimizations included

## Status

**Current Version**: 0.1.12 (November 2025)
**llama.cpp Version**: b7126
**Test Coverage**: 600+ tests passing
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
| Agent Framework | [x] Complete | ReActAgent, ConstrainedAgent, ContractAgent |
| Multimodal (LLAVA) | [x] Complete | Vision-language models |
| Whisper.cpp | [x] Complete | Speech-to-text |
| Stable Diffusion | [x] Complete | Image generation with stable-diffusion.cpp |
| Memory optimization | [x] Complete | GPU layer estimation |
| Server implementations | [x] Complete | PythonServer, EmbeddedServer, LlamaServer |

### Recent Releases

- **v0.1.12** (Nov 2025) - Stable Diffusion integration with stable-diffusion.cpp
- **v0.1.11** (Nov 2025) - ACP support, build improvements
- **v0.1.10** (Nov 2025) - Agent Framework, bug fixes
- **v0.1.9** (Nov 2025) - High-level APIs, integrations, batch processing, comprehensive documentation
- **v0.1.8** (Nov 2025) - Speculative decoding API
- **v0.1.7** (Nov 2025) - GGUF, JSON Schema, Downloads, N-gram Cache
- **v0.1.6** (Nov 2025) - Multimodal test fixes
- **v0.1.5** (Oct 2025) - Mongoose server, embedded server
- **v0.1.4** (Oct 2025) - Memory estimation, performance optimizations

See [CHANGELOG.md](CHANGELOG.md) for complete release history.

## Setup

To build `cyllama`:

1. A recent version of `python3` (currently testing on python 3.13)

2. Git clone the latest version of `cyllama`:

    ```sh
    git clone https://github.com/shakfu/cyllama.git
    cd cyllama
    ```

3. We use [uv](https://github.com/astral-sh/uv) for package management:

    If you don't have it see the link above to install it, otherwise:

    ```sh
    uv sync
    ```

4. Type `make` in the terminal.

    This will:

    1. Download and build `llama.cpp`, `whisper.cpp` and `stablediffusion.cpp`
    2. Install them into the `thirdparty` folder
    3. Build `cyllama`

### GPU Acceleration

By default, cyllama builds with Metal support on macOS and CPU-only on Linux. To enable other GPU backends (CUDA, Vulkan, etc.):

```sh
# CUDA (NVIDIA GPUs)
make build-cuda

# Vulkan (Cross-platform GPU)
make build-vulkan

# Multiple backends
export GGML_CUDA=1 GGML_VULKAN=1
make build
```

See [docs/build_backends.md](docs/BUILD_BACKENDS.md) for comprehensive backend build instructions.

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

With 600+ passing tests, the library is ready for both quick prototyping and production use:

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
- [x] OpenAI-compatible servers (PythonServer, EmbeddedServer, LlamaServer)
- [x] Whisper.cpp integration
- [x] Multimodal support (LLAVA)
- [x] Memory estimation utilities
- [x] Agent Framework (ReActAgent, ConstrainedAgent, ContractAgent)
- [x] Stable Diffusion (stable-diffusion.cpp) - image/video generation

### Future

- [ ] Async API support (`async def complete_async()`)
- [ ] Response caching for identical prompts
- [ ] Built-in prompt template system
- [ ] RAG utilities
- [ ] Web UI for testing

## Contributing

Contributions are welcome! Please see the [User Guide](docs/user_guide.md) for development guidelines.

## License

This project wraps [llama.cpp](https://github.com/ggml-org/llama.cpp) and follows its licensing terms.
