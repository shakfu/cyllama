# cyllama - Fast, Pythonic AI Inference

cyllama is a comprehensive no-dependencies Python library for AI inference built on the `.cpp` ecosystem:

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** - LLM text generation
- **[whisper.cpp](https://github.com/ggerganov/whisper.cpp)** - Speech-to-text transcription
- **[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)** - Image and video generation

It combines the performance of compiled Cython wrappers with a simple, high-level Python API.

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
from cyllama.llama.llama_cpp import Speculative, SpeculativeParams

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

**N-gram Cache** - 2-10x speedup for repetitive text:

```python
from cyllama.llama.llama_cpp import NgramCache

cache = NgramCache()
cache.update(tokens, ngram_min=2, ngram_max=4)
draft = cache.draft(input_tokens, n_draft=16)
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
from simpleeval import simple_eval

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression safely."""
    return str(simple_eval(expression))

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

See [Agents Overview](docs/book/src/agents_overview.qmd) for detailed agent documentation.

### Speech Recognition

**Whisper Transcription** - Transcribe audio files with timestamps:

```python
from cyllama.whisper import WhisperContext, WhisperFullParams
import numpy as np

# Load model and audio
ctx = WhisperContext("models/ggml-base.en.bin")
samples = load_audio_as_16khz_float32("audio.wav")  # Your audio loading function

# Transcribe
params = WhisperFullParams()
ctx.full(samples, params)

# Get results
for i in range(ctx.full_n_segments()):
    start = ctx.full_get_segment_t0(i) / 100.0
    end = ctx.full_get_segment_t1(i) / 100.0
    text = ctx.full_get_segment_text(i)
    print(f"[{start:.2f}s - {end:.2f}s] {text}")
```

See [Whisper docs](docs/book/src/whisper.qmd) for full documentation.

### Stable Diffusion

**Image Generation** - Generate images from text using stable-diffusion.cpp:

```python
from cyllama.sd import text_to_image

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
from cyllama.sd import SDContext, SDContextParams, SampleMethod, Scheduler

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
# Text to image
python -m cyllama.sd txt2img \
    --model models/sd_xl_turbo_1.0.q8_0.gguf \
    --prompt "a beautiful sunset" \
    --output sunset.png

# Image to image
python -m cyllama.sd img2img \
    --model models/sd-v1-5.gguf \
    --init-img input.png \
    --prompt "oil painting style" \
    --strength 0.7

# Show system info
python -m cyllama.sd info
```

Supports SD 1.x/2.x, SDXL, SD3, FLUX, FLUX2, z-image-turbo, video generation (Wan/CogVideoX), LoRA, ControlNet, inpainting, and ESRGAN upscaling. See [Stable Diffusion docs](docs/book/src/stable_diffusion.qmd) for full documentation.

### RAG (Retrieval-Augmented Generation)

**Simple RAG** - Query your documents with LLMs:

```python
from cyllama.rag import RAG

# Create RAG instance with embedding and generation models
rag = RAG(
    embedding_model="models/bge-small-en-v1.5-q8_0.gguf",
    generation_model="models/llama.gguf"
)

# Add documents
rag.add_texts([
    "Python is a high-level programming language.",
    "Machine learning is a subset of artificial intelligence.",
    "Neural networks are inspired by biological neurons."
])

# Query
response = rag.query("What is Python?")
print(response.text)
```

**Load Documents** - Support for multiple file formats:

```python
from cyllama.rag import RAG, load_directory

rag = RAG(
    embedding_model="models/bge-small-en-v1.5-q8_0.gguf",
    generation_model="models/llama.gguf"
)

# Load all documents from a directory
documents = load_directory("docs/", glob="**/*.md")
rag.add_documents(documents)

response = rag.query("How do I configure the system?")
```

**Hybrid Search** - Combine vector and keyword search:

```python
from cyllama.rag import RAG, HybridStore, Embedder

embedder = Embedder("models/bge-small-en-v1.5-q8_0.gguf")
store = HybridStore("knowledge.db", embedder)

store.add_texts(["Document content..."])

# Hybrid search with configurable weights
results = store.search("query", k=5, vector_weight=0.7, fts_weight=0.3)
```

**Agent Integration** - Use RAG as an agent tool:

```python
from cyllama import LLM
from cyllama.agents import ReActAgent
from cyllama.rag import RAG, create_rag_tool

rag = RAG(
    embedding_model="models/bge-small-en-v1.5-q8_0.gguf",
    generation_model="models/llama.gguf"
)
rag.add_texts(["Your knowledge base..."])

# Create a tool from the RAG instance
search_tool = create_rag_tool(rag)

llm = LLM("models/llama.gguf")
agent = ReActAgent(llm=llm, tools=[search_tool])
result = agent.run("Find information about X in the knowledge base")
```

Supports text chunking, multiple embedding pooling strategies, async operations, reranking, and SQLite-vector for persistent storage.

### Common Utilities

**GGUF File Manipulation** - Inspect and modify model files:

```python
from cyllama.llama.llama_cpp import GGUFContext

ctx = GGUFContext.from_file("model.gguf")
metadata = ctx.get_all_metadata()
print(f"Model: {metadata['general.name']}")
```

**Structured Output** - JSON schema to grammar conversion:

```python
from cyllama.llama.llama_cpp import json_schema_to_grammar

schema = {"type": "object", "properties": {"name": {"type": "string"}}}
grammar = json_schema_to_grammar(schema)
```

**Huggingface Model Downloads**:

```python
from cyllama.llama.llama_cpp import download_model, list_cached_models, get_hf_file

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

## What's Inside

### Text Generation (llama.cpp)

- [x] **Full llama.cpp API** - Complete Cython wrapper with strong typing
- [x] **High-Level API** - Simple, Pythonic interface (`LLM`, `complete`, `chat`)
- [x] **Streaming Support** - Token-by-token generation with callbacks
- [x] **Batch Processing** - Efficient parallel inference
- [x] **Multimodal** - LLAVA and vision-language models
- [x] **Speculative Decoding** - 2-3x inference speedup with draft models

### Speech Recognition (whisper.cpp)

- [x] **Full whisper.cpp API** - Complete Cython wrapper
- [x] **High-Level API** - Simple `transcribe()` function
- [x] **Multiple Formats** - WAV, MP3, FLAC, and more
- [x] **Language Detection** - Automatic or specified language
- [x] **Timestamps** - Word and segment-level timing

### Image & Video Generation (stable-diffusion.cpp)

- [x] **Full stable-diffusion.cpp API** - Complete Cython wrapper
- [x] **Text-to-Image** - SD 1.x/2.x, SDXL, SD3, FLUX, FLUX2
- [x] **Image-to-Image** - Transform existing images
- [x] **Inpainting** - Mask-based editing
- [x] **ControlNet** - Guided generation with edge/pose/depth
- [x] **Video Generation** - Wan, CogVideoX models
- [x] **Upscaling** - ESRGAN 4x upscaling

### Cross-Cutting Features

- [x] **GPU Acceleration** - Metal, CUDA, Vulkan backends
- [x] **Memory Optimization** - Smart GPU layer allocation
- [x] **Agent Framework** - ReActAgent, ConstrainedAgent, ContractAgent
- [x] **Framework Integration** - OpenAI API, LangChain, FastAPI

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

- 863+ passing tests with extensive coverage
- Comprehensive documentation and examples
- Proper error handling and logging
- Framework integration for real applications

**Up-to-Date**: Tracks bleeding-edge llama.cpp (currently b7126)

- Regular updates with latest features
- All high-priority APIs wrapped
- Performance optimizations included

## Status

**Current Version**: 0.1.18 (December 2025)
**llama.cpp Version**: b7126
**Build System**: scikit-build-core + CMake
**Test Coverage**: 863+ tests passing
**Platform**: macOS (tested), Linux (tested), Windows (tested)

### Recent Releases

- **v0.1.18** (Dec 2025) - Remaining stable-diffusion.cpp wrapped
- **v0.1.16** (Dec 2025) - Response class, Async API, Chat templates
- **v0.1.12** (Nov 2025) - Initial wrapper of stable-diffusion.cpp
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

    1. Download and build `llama.cpp`, `whisper.cpp` and `stable-diffusion.cpp`
    2. Install them into the `thirdparty` folder
    3. Build `cyllama` using scikit-build-core + CMake

### Build Commands

```sh
# Full build (default)
make              # Build dependencies + editable install

# Build wheel for distribution
make wheel        # Creates wheel in dist/

# Backend-specific builds
make build-metal  # macOS Metal (default on macOS)
make build-cuda   # NVIDIA CUDA
make build-vulkan # Vulkan (cross-platform)
make build-cpu    # CPU only

# Clean and rebuild
make clean        # Remove build artifacts
make reset        # Full reset including thirdparty
make remake       # Clean rebuild with tests
```

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

See [Build Backends](docs/book/src/build_backends.qmd) for comprehensive backend build instructions.

### Multi-GPU Configuration

For systems with multiple GPUs, cyllama provides full control over GPU selection and model splitting:

```python
from cyllama import LLM, GenerationConfig

# Use a specific GPU (GPU index 1)
llm = LLM("model.gguf", main_gpu=1)

# Multi-GPU with layer splitting (default mode)
llm = LLM("model.gguf", split_mode=1, n_gpu_layers=99)

# Multi-GPU with tensor parallelism (row splitting)
llm = LLM("model.gguf", split_mode=2, n_gpu_layers=99)

# Custom tensor split: 30% GPU 0, 70% GPU 1
llm = LLM("model.gguf", tensor_split=[0.3, 0.7])

# Full configuration via GenerationConfig
config = GenerationConfig(
    main_gpu=0,
    split_mode=1,          # 0=NONE, 1=LAYER, 2=ROW
    tensor_split=[1, 2],   # 1/3 GPU0, 2/3 GPU1
    n_gpu_layers=99
)
llm = LLM("model.gguf", config=config)
```

**Split Modes:**
- `0` (NONE): Single GPU only, uses `main_gpu`
- `1` (LAYER): Split layers and KV cache across GPUs (default)
- `2` (ROW): Tensor parallelism - split layers with row-wise distribution

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

With 863+ passing tests, the library is ready for both quick prototyping and production use:

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

- **[User Guide](docs/book/src/user_guide.qmd)** - Comprehensive guide covering all features
- **[API Reference](docs/book/src/api_reference.qmd)** - Complete API documentation
- **[Cookbook](docs/book/src/cookbook.qmd)** - Practical recipes and patterns
- **[Changelog](CHANGELOG.md)** - Complete release history
- **Examples** - See `tests/examples/` for working code samples

## Roadmap

### Completed

- [x] Full llama.cpp API wrapper with Cython
- [x] High-level API (`LLM`, `complete`, `chat`)
- [x] Async API support (`AsyncLLM`, `complete_async`, `chat_async`)
- [x] Response class with stats and serialization
- [x] Built-in chat template system (llama.cpp templates)
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
- [x] RAG utilities (text chunking, document processing)

### Future

- [ ] Response caching for identical prompts
- [ ] Web UI for testing

## Contributing

Contributions are welcome! Please see the [User Guide](docs/book/src/user_guide.qmd) for development guidelines.

## License

This project wraps [llama.cpp](https://github.com/ggml-org/llama.cpp), [whisper.cpp](https://github.com/ggml-org/whisper.cpp), and [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) which all follow the MIT licensing terms, as does cyllama.
