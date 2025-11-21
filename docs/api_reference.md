# Cyllama API Reference

**Version**: 0.1.9
**Date**: November 2025

Complete API reference for cyllama, a high-performance Python library for LLM inference built on llama.cpp.

## Table of Contents

1. [High-Level Generation API](#high-level-generation-api)
2. [Batch Processing API](#batch-processing-api)
3. [Framework Integrations](#framework-integrations)
4. [Memory Utilities](#memory-utilities)
5. [Core llama.cpp API](#core-llamacpp-api)
6. [Advanced Features](#advanced-features)
7. [Server Implementations](#server-implementations)
8. [Multimodal Support](#multimodal-support)
9. [Whisper Integration](#whisper-integration)

---

## High-Level Generation API

The high-level API provides simple, Pythonic functions and classes for text generation.

### `complete()`

One-shot text generation function.

```python
def complete(
    prompt: str,
    model_path: str,
    config: Optional[GenerationConfig] = None,
    stream: bool = False,
    **kwargs
) -> str | Iterator[str]
```

**Parameters:**

- `prompt` (str): Input text prompt
- `model_path` (str): Path to GGUF model file
- `config` (GenerationConfig, optional): Generation configuration object
- `stream` (bool): If True, return iterator of text chunks
- `**kwargs`: Override config parameters (temperature, max_tokens, etc.)

**Returns:**

- `str`: Generated text (if stream=False)
- `Iterator[str]`: Iterator of text chunks (if stream=True)

**Example:**

```python
from cyllama import complete

response = complete(
    "What is Python?",
    model_path="models/llama.gguf",
    temperature=0.7,
    max_tokens=200
)

# Streaming
for chunk in complete("Tell me a story", model_path="models/llama.gguf", stream=True):
    print(chunk, end="", flush=True)
```

---

### `chat()`

Chat-style generation with message history.

```python
def chat(
    messages: List[Dict[str, str]],
    model_path: str,
    config: Optional[GenerationConfig] = None,
    stream: bool = False,
    **kwargs
) -> str | Iterator[str]
```

**Parameters:**

- `messages` (List[Dict]): List of message dicts with 'role' and 'content' keys
- `model_path` (str): Path to GGUF model file
- `config` (GenerationConfig, optional): Generation configuration
- `stream` (bool): Enable streaming output
- `**kwargs`: Override config parameters

**Returns:**

- `str` or `Iterator[str]`: Generated response

**Example:**

```python
from cyllama import chat

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"}
]

response = chat(messages, model_path="models/llama.gguf")
```

---

### `LLM` Class

Reusable generator with model caching for improved performance.

```python
class LLM:
    def __init__(
        self,
        model_path: str,
        config: Optional[GenerationConfig] = None,
        verbose: bool = False
    )
```

**Parameters:**

- `model_path` (str): Path to GGUF model file
- `config` (GenerationConfig, optional): Default generation configuration
- `verbose` (bool): Print detailed information during generation

**Methods:**

#### `__call__()`

Generate text from a prompt.

```python
def __call__(
    self,
    prompt: str,
    config: Optional[GenerationConfig] = None,
    stream: bool = False,
    on_token: Optional[Callable[[str], None]] = None
) -> str | Iterator[str]
```

**Parameters:**

- `prompt` (str): Input text
- `config` (GenerationConfig, optional): Override instance config
- `stream` (bool): Enable streaming
- `on_token` (Callable, optional): Callback for each token

#### `generate_with_stats()`

Generate text and return performance statistics.

```python
def generate_with_stats(
    self,
    prompt: str,
    config: Optional[GenerationConfig] = None
) -> Tuple[str, GenerationStats]
```

**Returns:**

- `Tuple[str, GenerationStats]`: Generated text and statistics

**Example:**

```python
from cyllama import LLM, GenerationConfig

gen = LLM("models/llama.gguf")

# Simple generation
response = gen("What is Python?")

# With custom config
config = GenerationConfig(temperature=0.9, max_tokens=100)
response = gen("Tell me a joke", config=config)

# With statistics
response, stats = gen.generate_with_stats("Question?")
print(f"Generated {stats.generated_tokens} tokens in {stats.total_time:.2f}s")
print(f"Speed: {stats.tokens_per_second:.2f} tokens/sec")
```

---

### `GenerationConfig` Dataclass

Configuration for text generation.

```python
@dataclass
class GenerationConfig:
    max_tokens: int = 512
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    repeat_penalty: float = 1.1
    n_gpu_layers: int = 99
    n_ctx: Optional[int] = None
    n_batch: int = 512
    seed: int = -1
    stop_sequences: List[str] = field(default_factory=list)
    add_bos: bool = True
    parse_special: bool = True
```

**Attributes:**

- `max_tokens`: Maximum tokens to generate (default: 512)
- `temperature`: Sampling temperature, 0.0 = greedy (default: 0.8)
- `top_k`: Top-k sampling parameter (default: 40)
- `top_p`: Top-p (nucleus) sampling (default: 0.95)
- `min_p`: Minimum probability threshold (default: 0.05)
- `repeat_penalty`: Penalty for repeating tokens (default: 1.1)
- `n_gpu_layers`: GPU layers to offload (default: 99 = all)
- `n_ctx`: Context window size, None = auto (default: None)
- `n_batch`: Batch size for processing (default: 512)
- `seed`: Random seed, -1 = random (default: -1)
- `stop_sequences`: Strings that stop generation (default: [])
- `add_bos`: Add beginning-of-sequence token (default: True)
- `parse_special`: Parse special tokens in prompt (default: True)

---

### `GenerationStats` Dataclass

Statistics from a generation run.

```python
@dataclass
class GenerationStats:
    prompt_tokens: int
    generated_tokens: int
    total_time: float
    tokens_per_second: float
    prompt_time: float = 0.0
    generation_time: float = 0.0
```

---

## Batch Processing API

Efficient parallel processing of multiple prompts.

### `batch_generate()`

Convenience function for batch processing.

```python
def batch_generate(
    prompts: List[str],
    model_path: str,
    config: Optional[GenerationConfig] = None,
    **kwargs
) -> List[str]
```

**Parameters:**

- `prompts` (List[str]): List of input prompts
- `model_path` (str): Path to GGUF model file
- `config` (GenerationConfig, optional): Generation configuration
- `**kwargs`: Override config parameters

**Returns:**

- `List[str]`: List of generated responses

**Example:**

```python
from cyllama import batch_generate

prompts = ["What is 2+2?", "What is 3+3?", "What is 4+4?"]
responses = batch_generate(prompts, model_path="models/llama.gguf")

for prompt, response in zip(prompts, responses):
    print(f"{prompt} -> {response}")
```

---

### `BatchGenerator` Class

Reusable batch processor for improved performance.

```python
class BatchGenerator:
    def __init__(
        self,
        model_path: str,
        batch_size: int = 512,
        n_ctx: int = 2048,
        n_gpu_layers: int = 99,
        verbose: bool = False
    )
```

**Parameters:**

- `model_path` (str): Path to GGUF model file
- `batch_size` (int): Maximum batch size (default: 512)
- `n_ctx` (int): Context window size (default: 2048)
- `n_gpu_layers` (int): GPU layers to offload (default: 99)
- `verbose` (bool): Print detailed information

**Methods:**

#### `generate_batch()`

Process multiple prompts in parallel.

```python
def generate_batch(
    self,
    prompts: List[str],
    config: Optional[GenerationConfig] = None
) -> List[str]
```

**Example:**

```python
from cyllama import BatchGenerator

batch_gen = BatchGenerator("models/llama.gguf", batch_size=8)
responses = batch_gen.generate_batch([
    "What is Python?",
    "What is Rust?",
    "What is Go?"
])
```

---

### `BatchRequest` / `BatchResponse` Dataclasses

Structured batch operations.

```python
@dataclass
class BatchRequest:
    id: int
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7

@dataclass
class BatchResponse:
    id: int
    prompt: str
    response: str
    tokens_generated: int
    time_taken: float
```

---

## Framework Integrations

### OpenAI-Compatible API

Drop-in replacement for OpenAI Python client.

#### `OpenAICompatibleClient` Class

```python
from cyllama.integrations.openai_compat import OpenAICompatibleClient

class OpenAICompatibleClient:
    def __init__(
        self,
        model_path: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        n_gpu_layers: int = 99
    )
```

**Attributes:**

- `chat`: Chat completions interface

**Example:**

```python
from cyllama.integrations.openai_compat import OpenAICompatibleClient

client = OpenAICompatibleClient(model_path="models/llama.gguf")

response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"}
    ],
    temperature=0.7,
    max_tokens=200
)

print(response.choices[0].message.content)

# Streaming
for chunk in client.chat.completions.create(
    messages=[{"role": "user", "content": "Count to 5"}],
    stream=True
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

### LangChain Integration

Full LangChain LLM interface implementation.

#### `CyllamaLLM` Class

```python
from cyllama.integrations import CyllamaLLM

class CyllamaLLM(LLM):
    model_path: str
    temperature: float = 0.7
    max_tokens: int = 512
    top_k: int = 40
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    n_gpu_layers: int = 99
```

**Example:**

```python
from cyllama.integrations import CyllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = CyllamaLLM(model_path="models/llama.gguf", temperature=0.7)

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms:"
)

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(topic="quantum computing")

# With streaming
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = CyllamaLLM(
    model_path="models/llama.gguf",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)
```

---

## Memory Utilities

Tools for estimating and optimizing GPU memory usage.

### `estimate_gpu_layers()`

Estimate optimal number of GPU layers for available VRAM.

```python
def estimate_gpu_layers(
    model_path: str,
    available_vram_mb: int,
    n_ctx: int = 2048,
    n_batch: int = 512
) -> MemoryEstimate
```

**Parameters:**

- `model_path` (str): Path to GGUF model file
- `available_vram_mb` (int): Available VRAM in megabytes
- `n_ctx` (int): Context window size
- `n_batch` (int): Batch size

**Returns:**

- `MemoryEstimate`: Object with recommended settings

**Example:**

```python
from cyllama import estimate_gpu_layers

estimate = estimate_gpu_layers(
    model_path="models/llama.gguf",
    available_vram_mb=8000,  # 8GB VRAM
    n_ctx=2048
)

print(f"Recommended GPU layers: {estimate.n_gpu_layers}")
print(f"Estimated VRAM usage: {estimate.vram / 1024 / 1024:.2f} MB")
```

---

### `estimate_memory_usage()`

Estimate total memory requirements for model loading.

```python
def estimate_memory_usage(
    model_path: str,
    n_ctx: int = 2048,
    n_batch: int = 512,
    n_gpu_layers: int = 0
) -> MemoryEstimate
```

---

### `MemoryEstimate` Dataclass

Memory estimation results.

```python
@dataclass
class MemoryEstimate:
    layers: int                          # Total layers
    graph_size: int                      # Computation graph size
    vram: int                            # VRAM usage (bytes)
    vram_kv: int                         # KV cache VRAM (bytes)
    total_size: int                      # Total memory (bytes)
    tensor_split: Optional[List[int]]    # Multi-GPU split
```

---

## Core llama.cpp API

Low-level Cython wrappers for direct llama.cpp access.

### Core Classes

#### `LlamaModel`

Represents a loaded GGUF model.

```python
from cyllama import LlamaModel, LlamaModelParams

params = LlamaModelParams()
params.n_gpu_layers = 99
params.use_mmap = True
params.use_mlock = False

model = LlamaModel("models/llama.gguf", params)

# Properties
print(model.n_params)      # Total parameters
print(model.n_layers)      # Number of layers
print(model.n_embd)        # Embedding dimension
print(model.n_vocab)       # Vocabulary size

# Methods
vocab = model.get_vocab()  # Get vocabulary
model.free()               # Free resources
```

---

#### `LlamaContext`

Inference context for model.

```python
from cyllama import LlamaContext, LlamaContextParams

ctx_params = LlamaContextParams()
ctx_params.n_ctx = 2048
ctx_params.n_batch = 512
ctx_params.n_threads = 4
ctx_params.n_threads_batch = 4

ctx = LlamaContext(model, ctx_params)

# Decode batch
from cyllama import llama_batch_get_one
batch = llama_batch_get_one(tokens)
ctx.decode(batch)

# KV cache management
ctx.kv_cache_clear()
ctx.kv_cache_seq_rm(seq_id, p0, p1)
ctx.kv_cache_seq_add(seq_id, p0, p1, delta)

# Performance
ctx.print_perf_data()
```

---

#### `LlamaSampler`

Sampling strategies for token generation.

```python
from cyllama import LlamaSampler, LlamaSamplerChainParams

sampler_params = LlamaSamplerChainParams()
sampler = LlamaSampler(sampler_params)

# Add sampling methods
sampler.add_top_k(40)
sampler.add_top_p(0.95, 1)
sampler.add_temp(0.7)
sampler.add_dist(seed)

# Sample token
token_id = sampler.sample(ctx, idx)

# Reset state
sampler.reset()
```

---

#### `LlamaVocab`

Vocabulary and tokenization.

```python
vocab = model.get_vocab()

# Tokenization
tokens = vocab.tokenize("Hello world", add_special=True, parse_special=True)

# Detokenization
text = vocab.detokenize(tokens)
piece = vocab.token_to_piece(token_id, special=True)

# Special tokens
print(vocab.bos)           # Begin-of-sequence token
print(vocab.eos)           # End-of-sequence token
print(vocab.eot)           # End-of-turn token
print(vocab.n_vocab)       # Vocabulary size

# Check token types
is_eog = vocab.is_eog(token_id)
is_control = vocab.is_control(token_id)
```

---

#### `LlamaBatch`

Efficient batch processing.

```python
from cyllama import LlamaBatch

# Create batch
batch = LlamaBatch(n_tokens=512, embd=0, n_seq_max=1)

# Add token
batch.add(token_id, pos, seq_ids=[0], logits=True)

# Clear batch
batch.clear()

# Convenience function
from cyllama import llama_batch_get_one
batch = llama_batch_get_one(tokens, pos_offset=0)
```

---

### Backend Management

```python
from cyllama import (
    ggml_backend_load_all,
    ggml_backend_offload_supported,
    ggml_backend_metal_set_n_cb
)

# Load all available backends (Metal, CUDA, etc.)
ggml_backend_load_all()

# Check GPU support
if ggml_backend_offload_supported():
    print("GPU offload supported")

# Configure Metal (macOS)
ggml_backend_metal_set_n_cb(2)  # Number of command buffers
```

---

## Advanced Features

### GGUF File Manipulation

Inspect and modify GGUF model files.

#### `GGUFContext` Class

```python
from cyllama import GGUFContext

# Read existing file
ctx = GGUFContext.from_file("model.gguf")

# Get metadata
metadata = ctx.get_all_metadata()
print(metadata['general.architecture'])
print(metadata['general.name'])

value = ctx.get_val_str("general.architecture")

# Create new file
ctx = GGUFContext.empty()
ctx.set_val_str("custom.key", "value")
ctx.set_val_u32("custom.number", 42)
ctx.write_to_file("custom.gguf", write_tensors=False)

# Modify existing
ctx = GGUFContext.from_file("model.gguf")
ctx.set_val_str("custom.metadata", "updated")
ctx.write_to_file("modified.gguf")
```

---

### JSON Schema to Grammar

Convert JSON schemas to llama.cpp grammar format for structured output.

```python
from cyllama import json_schema_to_grammar

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string"}
    },
    "required": ["name", "age"]
}

grammar = json_schema_to_grammar(schema)

# Use with generation
from cyllama import LlamaSampler
sampler = LlamaSampler()
sampler.add_grammar(grammar)
```

---

### Model Download

Download models from HuggingFace with Ollama-style tags.

```python
from cyllama import download_model, list_cached_models

# Download from HuggingFace
download_model(
    hf_repo="bartowski/Llama-3.2-1B-Instruct-GGUF:q4",
    cache_dir="~/.cache/cyllama/models"
)

# List cached models
models = list_cached_models()
for model in models:
    print(f"{model['user']}/{model['model']}:{model['tag']}")
    print(f"  Path: {model['path']}")
    print(f"  Size: {model['size'] / 1024 / 1024:.2f} MB")

# Direct URL download
download_model(
    url="https://example.com/model.gguf",
    output_path="models/custom.gguf"
)
```

---

### N-gram Cache

Pattern-based token prediction for 2-10x speedup on repetitive text.

```python
from cyllama import NgramCache

# Create cache
cache = NgramCache()

# Learn patterns from token sequences
tokens = [1, 2, 3, 4, 5, 6, 7, 8]
cache.update(tokens, ngram_min=2, ngram_max=4)

# Predict likely continuations
input_tokens = [1, 2, 3]
draft_tokens = cache.draft(input_tokens, n_draft=16)

# Save/load cache
cache.save("patterns.bin")
loaded_cache = NgramCache.from_file("patterns.bin")

# Clear cache
cache.clear()
```

---

### Speculative Decoding

Use draft model for 2-3x inference speedup.

```python
from cyllama import (
    LlamaModel, LlamaContext, LlamaModelParams, LlamaContextParams,
    Speculative, SpeculativeParams
)

# Load target and draft models
model_target = LlamaModel("models/large.gguf", LlamaModelParams())
model_draft = LlamaModel("models/small.gguf", LlamaModelParams())

ctx_params = LlamaContextParams()
ctx_params.n_ctx = 2048

ctx_target = LlamaContext(model_target, ctx_params)
ctx_draft = LlamaContext(model_draft, ctx_params)

# Check compatibility
spec = Speculative(ctx_target, ctx_draft)
if spec.are_compatible():
    print("Models are compatible for speculative decoding")

    # Configure speculative parameters
    params = SpeculativeParams(
        n_draft=16,      # Number of draft tokens
        n_reuse=8,       # Tokens to reuse
        p_min=0.75       # Minimum acceptance probability
    )

    # Generate draft tokens
    prompt_tokens = [1, 2, 3]
    last_token = prompt_tokens[-1]
    draft_tokens = spec.gen_draft(params, prompt_tokens, last_token)
```

**Parameters:**

- `n_draft`: Number of tokens to draft (default: 16)
- `n_reuse`: Number of tokens to reuse from previous draft (default: 8)
- `p_min`: Minimum acceptance probability (default: 0.75)

---

## Server Implementations

Three OpenAI-compatible server implementations.

### Embedded Server

Pure Python server implementation.

```python
from cyllama.llama.server.embedded import start_server

# Start server
start_server(
    model_path="models/llama.gguf",
    host="127.0.0.1",
    port=8000,
    n_ctx=2048,
    n_gpu_layers=99
)

# Use with OpenAI client
import openai
openai.api_base = "http://127.0.0.1:8000/v1"

response = openai.ChatCompletion.create(
    model="cyllama",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

### Mongoose Server

High-performance C server using Mongoose library.

```python
from cyllama.llama.server.mongoose_server import MongooseServer

server = MongooseServer(
    model_path="models/llama.gguf",
    host="127.0.0.1",
    port=8080,
    n_ctx=2048,
    n_threads=4
)

server.start()

# Server runs in background
# Access at http://127.0.0.1:8080

server.stop()
```

---

### Launcher

Wrapper around llama.cpp's binary server.

```python
from cyllama.llama.server.launcher import ServerLauncher

launcher = ServerLauncher(
    model_path="models/llama.gguf",
    host="127.0.0.1",
    port=8080,
    server_binary="bin/llama-server"
)

launcher.start()

# Check status
if launcher.is_running():
    print("Server is running")

launcher.stop()
```

---

## Multimodal Support

LLAVA and other vision-language models.

```python
from cyllama.llama.mtmd.multimodal import (
    LlavaImageEmbed,
    load_mmproj,
    process_image
)

# Load multimodal projector
mmproj = load_mmproj("models/mmproj.gguf")

# Process image
image_embed = process_image(
    ctx=ctx,
    image_path="image.jpg",
    mmproj=mmproj
)

# Use in generation
# Image embeddings are automatically integrated into context
```

---

## Whisper Integration

Speech-to-text transcription using whisper.cpp.

```python
from cyllama.whisper import WhisperContext, WhisperParams

# Initialize whisper
params = WhisperParams()
params.language = "en"
params.n_threads = 4

ctx = WhisperContext("models/whisper-base.bin")

# Transcribe audio
result = ctx.transcribe("audio.wav", params)

# Get segments
for segment in result.segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

# Full text
print(result.text)
```

---

## Error Handling

All cyllama functions raise appropriate Python exceptions:

```python
from cyllama import complete, Generator

try:
    response = complete("Hello", model_path="nonexistent.gguf")
except FileNotFoundError:
    print("Model file not found")
except RuntimeError as e:
    print(f"Runtime error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

# Generator with error handling
try:
    gen = LLM("models/llama.gguf")
    response = gen("What is Python?")
except Exception as e:
    print(f"Generation failed: {e}")
```

---

## Type Hints

All functions include comprehensive type hints for IDE support:

```python
from typing import List, Dict, Optional, Iterator, Callable, Tuple
from cyllama import (
    generate,           # str | Iterator[str]
    chat,              # str | Iterator[str]
    Generator,         # class
    GenerationConfig,  # @dataclass
    batch_generate,    # List[str]
)
```

---

## Performance Tips

### 1. Model Reuse

```python
# BAD: Reloads model each time (slow)
for prompt in prompts:
    response = complete(prompt, model_path="model.gguf")

# GOOD: Reuses loaded model (fast)
gen = LLM("model.gguf")
for prompt in prompts:
    response = gen(prompt)
```

### 2. Batch Processing

```python
# BAD: Sequential processing
responses = [generate(p, model_path="model.gguf") for p in prompts]

# GOOD: Parallel batch processing (3-10x faster)
responses = batch_generate(prompts, model_path="model.gguf")
```

### 3. GPU Offloading

```python
# Estimate optimal layers
from cyllama import estimate_gpu_layers

estimate = estimate_gpu_layers("model.gguf", available_vram_mb=8000)

# Use recommended settings
config = GenerationConfig(n_gpu_layers=estimate.n_gpu_layers)
gen = LLM("model.gguf", config=config)
```

### 4. Context Sizing

```python
# Auto-size context (recommended)
config = GenerationConfig(n_ctx=None, max_tokens=200)

# Manual sizing (for control)
config = GenerationConfig(n_ctx=2048, max_tokens=200)
```

### 5. Streaming for Long Outputs

```python
# Non-streaming: waits for complete response
response = complete("Write a long essay", model_path="model.gguf", max_tokens=2000)

# Streaming: see output as it generates
for chunk in complete("Write a long essay", model_path="model.gguf",
                     max_tokens=2000, stream=True):
    print(chunk, end="", flush=True)
```

---

## Version Compatibility

- **Python**: ≥3.8 (tested on 3.13)
- **llama.cpp**: b6374 (November 2025)
- **Platform**: macOS (primary), Linux (tested)

---

## See Also

- [User Guide](user_guide.md) - Comprehensive usage guide
- [Cookbook](cookbook.md) - Practical recipes and patterns
- [Changelog](../CHANGELOG.md) - Release history
- [llama.cpp Documentation](https://github.com/ggml-org/llama.cpp)

---

**Last Updated**: November 21, 2025
**Cyllama Version**: 0.1.9
