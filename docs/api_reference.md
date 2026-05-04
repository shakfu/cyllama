# Cyllama API Reference

Complete API reference for cyllama, a high-performance Python library for LLM inference built on llama.cpp.

## Table of Contents

1. [High-Level Generation API](#high-level-generation-api)
2. [Async API](#async-api)
3. [Framework Integrations](#framework-integrations)
4. [Memory Utilities](#memory-utilities)
5. [Core llama.cpp API](#core-llamacpp-api)
6. [Advanced Features](#advanced-features)
7. [Server Implementations](#server-implementations)
8. [Multimodal Support](#multimodal-support)
9. [Whisper Integration](#whisper-integration)
10. [Stable Diffusion Integration](#stable-diffusion-integration)

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
) -> Response | Iterator[str]
```

**Parameters:**

- `prompt` (str): Input text prompt

- `model_path` (str): Path to GGUF model file

- `config` (GenerationConfig, optional): Generation configuration object

- `stream` (bool): If True, return iterator of text chunks

- `**kwargs`: Override config parameters (temperature, max_tokens, etc.)

**Returns:**

- `Response`: Response object with text and stats (if stream=False)

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

Chat-style generation with message history. Automatically applies the model's built-in chat template.

```python
def chat(
    messages: List[Dict[str, str]],
    model_path: str,
    config: Optional[GenerationConfig] = None,
    stream: bool = False,
    template: Optional[str] = None,
    **kwargs
) -> str | Iterator[str]
```

**Parameters:**

- `messages` (List[Dict]): List of message dicts with 'role' and 'content' keys

- `model_path` (str): Path to GGUF model file

- `config` (GenerationConfig, optional): Generation configuration

- `stream` (bool): Enable streaming output

- `template` (str, optional): Chat template name to use. If None, uses model's default.

- `**kwargs`: Override config parameters

**Returns:**

- `Response`: Response object with text and stats (if stream=False)

- `Iterator[str]`: Iterator of text chunks (if stream=True)

**Example:**

```python
from cyllama import chat

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"}
]

response = chat(messages, model_path="models/llama.gguf")

# With explicit template
response = chat(messages, model_path="models/llama.gguf", template="chatml")
```

---

### `apply_chat_template()`

Apply a chat template to format messages into a prompt string.

```python
def apply_chat_template(
    messages: List[Dict[str, str]],
    model_path: str,
    template: Optional[str] = None,
    add_generation_prompt: bool = True,
    verbose: bool = False,
) -> str
```

**Parameters:**

- `messages` (List[Dict]): List of message dicts with 'role' and 'content' keys

- `model_path` (str): Path to GGUF model file

- `template` (str, optional): Template name or string. If None, uses model's default.

- `add_generation_prompt` (bool): Add assistant prompt prefix (default: True)

- `verbose` (bool): Enable detailed logging

**Returns:**

- `str`: Formatted prompt string

**Supported Templates:**

- llama2, llama3, llama4

- chatml (Qwen, Yi, etc.)

- mistral-v1, mistral-v3, mistral-v7

- phi3, phi4

- deepseek, deepseek2, deepseek3

- gemma, falcon3, command-r, vicuna, zephyr, and more

**Example:**

```python
from cyllama.api import apply_chat_template

messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
]

prompt = apply_chat_template(messages, "models/llama.gguf")
print(prompt)
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>
# Hello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

---

### `get_chat_template()`

Get the chat template string from a model.

```python
def get_chat_template(
    model_path: str,
    template_name: Optional[str] = None
) -> str
```

**Parameters:**

- `model_path` (str): Path to GGUF model file

- `template_name` (str, optional): Specific template name to retrieve

**Returns:**

- `str`: Template string (Jinja-style), or empty string if not found

**Example:**

```python
from cyllama.api import get_chat_template

template = get_chat_template("models/llama.gguf")
print(template)  # Shows the Jinja-style template
```

---

### `Response` Class

Structured response object returned by generation functions.

```python
@dataclass
class Response:
    text: str                           # Generated text content
    stats: Optional[GenerationStats]    # Generation statistics
    finish_reason: str = "stop"         # Why generation stopped
    model: str = ""                     # Model path used
```

**Attributes:**

- `text` (str): The generated text content

- `stats` (GenerationStats, optional): Statistics including timing and token counts

- `finish_reason` (str): Reason for completion ("stop", "length", etc.)

- `model` (str): Path to the model used

**String Compatibility:**

`Response` implements the string protocol for backward compatibility:

- `str(response)` returns `response.text`

- `response == "string"` compares with text

- `len(response)` returns text length

- `for char in response:` iterates over text characters

- `"substring" in response` checks text containment

- `response + " more"` concatenates text

**Methods:**

#### `to_dict()`

Convert response to dictionary.

```python
def to_dict(self) -> Dict[str, Any]
```

#### `to_json()`

Convert response to JSON string.

```python
def to_json(self, indent: Optional[int] = None) -> str
```

**Example:**

```python
from cyllama import complete

response = complete("What is Python?", model_path="model.gguf")

# Use as string (backward compatible)
print(response)  # Prints text
if "programming" in response:
    print("Mentioned programming!")

# Access structured data
print(f"Finish reason: {response.finish_reason}")
if response.stats:
    print(f"Tokens/sec: {response.stats.tokens_per_second:.1f}")

# Serialize
data = response.to_dict()
json_str = response.to_json(indent=2)
```

---

### `GenerationStats` Class

Statistics from a generation run.

```python
@dataclass
class GenerationStats:
    prompt_tokens: int       # Number of tokens in prompt
    generated_tokens: int    # Number of tokens generated
    total_time: float        # Total generation time (seconds)
    tokens_per_second: float # Generation speed
    prompt_time: float       # Time for prompt processing
    generation_time: float   # Time for token generation
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
) -> Response | Iterator[str]
```

**Parameters:**

- `prompt` (str): Input text

- `config` (GenerationConfig, optional): Override instance config

- `stream` (bool): Enable streaming

- `on_token` (Callable, optional): Callback for each token

**Returns:**

- `Response`: Response object with text and stats (if stream=False)

- `Iterator[str]`: Iterator of text chunks (if stream=True)

#### `chat()`

Generate a response from chat messages using the model's chat template.

```python
def chat(
    self,
    messages: List[Dict[str, str]],
    config: Optional[GenerationConfig] = None,
    stream: bool = False,
    template: Optional[str] = None
) -> str | Iterator[str]
```

**Parameters:**

- `messages` (List[Dict]): List of message dicts with 'role' and 'content' keys

- `config` (GenerationConfig, optional): Override instance config

- `stream` (bool): Enable streaming

- `template` (str, optional): Chat template name to use

#### `get_chat_template()`

Get the chat template string from the loaded model.

```python
def get_chat_template(
    self,
    template_name: Optional[str] = None
) -> str
```

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

# Chat with template
messages = [{"role": "user", "content": "Hello!"}]
response = gen.chat(messages)

# Get template
template = gen.get_chat_template()
```

#### MCP client methods

Since 0.2.11 `LLM` can attach to Model Context Protocol servers and drive a tool-calling loop against their tools:

```python
def add_mcp_server(
    self,
    name: str,
    *,
    command: Optional[str] = None,
    args: Optional[list[str]] = None,
    env: Optional[dict[str, str]] = None,
    cwd: Optional[str] = None,
    url: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
    transport: Optional["McpTransportType"] = None,
    request_timeout: Optional[float] = None,
    shutdown_timeout: Optional[float] = None,
) -> None
def remove_mcp_server(self, name: str) -> None
def list_mcp_tools(self) -> list["McpTool"]
def list_mcp_resources(self) -> list["McpResource"]
def call_mcp_tool(self, name: str, arguments: dict) -> Any
def read_mcp_resource(self, uri: str) -> str
def chat_with_tools(
    self,
    messages: list[dict],
    *,
    tools: Optional[list["Tool"]] = None,
    use_mcp: bool = True,
    max_iterations: int = 8,
    verbose: bool = False,
    system_prompt: Optional[str] = None,
    generation_config: Optional[GenerationConfig] = None,
) -> str
```

See [MCP Client](mcp.md) for stdio/HTTP quick-start, per-method semantics, and examples of mixing local `Tool`s with MCP tools.

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
    repeat_penalty: float = 1.0
    penalty_last_n: int = 64
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    mirostat: int = 0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    n_gpu_layers: int = -1
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

- `repeat_penalty`: Penalty for repeating tokens (default: 1.0, disabled)

- `penalty_last_n`: Number of recent tokens considered for penalties; 0 = disabled, -1 = full context (default: 64)

- `frequency_penalty`: Penalize tokens by frequency in the recent window, 0.0 = disabled (default: 0.0)

- `presence_penalty`: Penalize tokens already present in the recent window, 0.0 = disabled (default: 0.0)

- `mirostat`: Mirostat sampling mode -- 0 = off, 1 = v1, 2 = v2. When enabled, replaces top_k / top_p / min_p with the mirostat sampler (default: 0)

- `mirostat_tau`: Mirostat target entropy (default: 5.0)

- `mirostat_eta`: Mirostat learning rate (default: 0.1)

- `n_gpu_layers`: GPU layers to offload (default: -1 = all)

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

## Async API

The async API provides non-blocking generation for use in async applications (FastAPI, aiohttp, etc.).

### `AsyncLLM` Class

Async wrapper around the LLM class for non-blocking text generation.

```python
class AsyncLLM:
    def __init__(
        self,
        model_path: str,
        config: Optional[GenerationConfig] = None,
        verbose: bool = False,
        **kwargs
    )
```

**Parameters:**

- `model_path` (str): Path to GGUF model file

- `config` (GenerationConfig, optional): Generation configuration

- `verbose` (bool): Print detailed information during generation

- `**kwargs`: Generation parameters (temperature, max_tokens, etc.)

**Methods:**

#### `__call__()` / `generate()`

Generate text asynchronously.

```python
async def __call__(
    self,
    prompt: str,
    config: Optional[GenerationConfig] = None,
    **kwargs
) -> str
```

#### `stream()`

Stream generated text chunks asynchronously.

```python
async def stream(
    self,
    prompt: str,
    config: Optional[GenerationConfig] = None,
    **kwargs
) -> AsyncIterator[str]
```

#### `generate_with_stats()`

Generate text and return statistics.

```python
async def generate_with_stats(
    self,
    prompt: str,
    config: Optional[GenerationConfig] = None
) -> Tuple[str, GenerationStats]
```

**Example:**

```python
import asyncio
from cyllama import AsyncLLM

async def main():
    # Context manager ensures cleanup
    async with AsyncLLM("model.gguf", temperature=0.7) as llm:
        # Simple generation
        response = await llm("What is Python?")
        print(response)

        # Streaming
        async for chunk in llm.stream("Tell me a story"):
            print(chunk, end="", flush=True)

        # With stats
        text, stats = await llm.generate_with_stats("Question?")
        print(f"Generated {stats.generated_tokens} tokens")

asyncio.run(main())
```

---

### `complete_async()`

Async convenience function for one-off text completion.

```python
async def complete_async(
    prompt: str,
    model_path: str,
    config: Optional[GenerationConfig] = None,
    verbose: bool = False,
    **kwargs
) -> str
```

**Example:**

```python
response = await complete_async(
    "What is Python?",
    model_path="model.gguf",
    temperature=0.7
)
```

---

### `chat_async()`

Async convenience function for chat-style generation.

```python
async def chat_async(
    messages: List[Dict[str, str]],
    model_path: str,
    config: Optional[GenerationConfig] = None,
    verbose: bool = False,
    **kwargs
) -> str
```

**Example:**

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"}
]

response = await chat_async(messages, model_path="model.gguf")
```

---

### `stream_complete_async()`

Async streaming completion for one-off use.

```python
async def stream_complete_async(
    prompt: str,
    model_path: str,
    config: Optional[GenerationConfig] = None,
    verbose: bool = False,
    **kwargs
) -> AsyncIterator[str]
```

**Example:**

```python
async for chunk in stream_complete_async("Tell me a story", "model.gguf"):
    print(chunk, end="", flush=True)
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
        n_gpu_layers: int = -1
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
    repeat_penalty: float = 1.0
    n_gpu_layers: int = -1
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
from cyllama.llama.llama_cpp import LlamaModel, LlamaModelParams

params = LlamaModelParams()
params.n_gpu_layers = -1
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
from cyllama.llama.llama_cpp import LlamaContext, LlamaContextParams

ctx_params = LlamaContextParams()
ctx_params.n_ctx = 2048
ctx_params.n_batch = 512
ctx_params.n_threads = 4
ctx_params.n_threads_batch = 4

ctx = LlamaContext(model, ctx_params)

# Decode batch
from cyllama.llama.llama_cpp import llama_batch_get_one
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
from cyllama.llama.llama_cpp import LlamaSampler, LlamaSamplerChainParams

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
from cyllama.llama.llama_cpp import LlamaBatch

# Create batch
batch = LlamaBatch(n_tokens=512, embd=0, n_seq_max=1)

# Add token
batch.add(token_id, pos, seq_ids=[0], logits=True)

# Clear batch
batch.clear()

# Convenience function
from cyllama.llama.llama_cpp import llama_batch_get_one
batch = llama_batch_get_one(tokens, pos_offset=0)
```

---

### Backend Management

```python
from cyllama.llama.llama_cpp import (
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
from cyllama.llama.llama_cpp import GGUFContext

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

Convert JSON schemas to llama.cpp grammar format for structured output. This is implemented in pure Python (vendored from llama.cpp) with no C++ dependency.

```python
from cyllama.llama.llama_cpp import json_schema_to_grammar

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
from cyllama.llama.llama_cpp import LlamaSampler
sampler = LlamaSampler()
sampler.add_grammar(grammar)
```

---

### Model Download

Download models from HuggingFace with Ollama-style tags.

```python
from cyllama.llama.llama_cpp import download_model, list_cached_models

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
from cyllama.llama.llama_cpp import NgramCache

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
from cyllama.llama.llama_cpp import (
    LlamaModel, LlamaContext, LlamaModelParams, LlamaContextParams,
    Speculative, SpeculativeParams
)

# Load target and draft models
model_target = LlamaModel("models/large.gguf", LlamaModelParams())
model_draft = LlamaModel("models/small.gguf", LlamaModelParams())

ctx_params = LlamaContextParams()
ctx_params.n_ctx = 2048

ctx_target = LlamaContext(model_target, ctx_params)

# Configure speculative parameters
params = SpeculativeParams(
    n_max=16,        # Maximum number of draft tokens
    n_min=0,         # Minimum number of draft tokens
    p_split=0.1,     # Speculative decoding split probability
    p_min=0.75       # Minimum acceptance probability
)

# Check target-context compatibility (static method)
if Speculative.is_compat(ctx_target):
    print("Target context is compatible for speculative decoding")

    ctx_draft = LlamaContext(model_draft, ctx_params)

    # Create speculative decoding instance
    spec = Speculative(params, ctx_target, ctx_draft)

    # Begin a speculative decoding round
    prompt_tokens = [1, 2, 3]
    spec.begin(prompt_tokens)

    # Generate draft tokens
    last_token = prompt_tokens[-1]
    draft_tokens = spec.draft(params, prompt_tokens, last_token)

    # Accept verified tokens (n_accepted from target verification)
    spec.accept(n_accepted=len(draft_tokens))

    # Print performance statistics
    spec.print_stats()
```

**Parameters:**

- `n_max`: Maximum number of tokens to draft (default: 16)

- `n_min`: Minimum number of draft tokens (default: 0)

- `p_split`: Speculative decoding split probability (default: 0.1)

- `p_min`: Minimum acceptance probability (default: 0.75)

**Methods:**

| Method | Description |
|--------|-------------|
| `Speculative.is_compat(ctx_target)` | Static: check if target context supports speculative decoding |
| `begin(prompt_tokens)` | Begin a speculative decoding round |
| `draft(params, prompt_tokens, last_token_id)` | Generate draft tokens from the draft model |
| `accept(n_accepted)` | Accept the first `n_accepted` verified draft tokens |
| `print_stats()` | Print speculative decoding performance statistics |

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
    n_gpu_layers=-1
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
from cyllama.llama.server.mongoose_server import EmbeddedServer

server = EmbeddedServer(
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

### LlamaServer

Python wrapper around the llama.cpp server binary.

```python
from cyllama.llama.server import LlamaServer, LauncherServerConfig

config = LauncherServerConfig(
    model_path="models/llama.gguf",
    host="127.0.0.1",
    port=8080
)

server = LlamaServer(config, server_binary="bin/llama-server")
server.start()

# Check status
if server.is_running():
    print("Server is running")

server.stop()
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

Speech-to-text transcription using whisper.cpp. See [Whisper.cpp Integration](whisper.md) for complete documentation.

### Quick Start

```python
from cyllama.whisper import WhisperContext, WhisperFullParams
import numpy as np

# Load model
ctx = WhisperContext("models/ggml-base.en.bin")

# Audio must be 16kHz mono float32
samples = load_audio_as_float32("audio.wav")  # Your audio loading function

# Transcribe
params = WhisperFullParams()
params.language = "en"
ctx.full(samples, params)

# Get results
for i in range(ctx.full_n_segments()):
    t0 = ctx.full_get_segment_t0(i) / 100.0  # centiseconds to seconds
    t1 = ctx.full_get_segment_t1(i) / 100.0
    text = ctx.full_get_segment_text(i)
    print(f"[{t0:.2f}s - {t1:.2f}s] {text}")
```

### Key Classes

| Class | Description |
|-------|-------------|
| `WhisperContext` | Main context for model loading and inference |
| `WhisperContextParams` | Configuration for context creation |
| `WhisperFullParams` | Configuration for transcription |
| `WhisperVadParams` | Voice activity detection parameters |

### WhisperContext Methods

| Method | Description |
|--------|-------------|
| `full(samples, params)` | Run transcription on float32 audio samples |
| `full_n_segments()` | Get number of transcribed segments |
| `full_get_segment_text(i)` | Get text of segment i |
| `full_get_segment_t0(i)` | Get start time (centiseconds) |
| `full_get_segment_t1(i)` | Get end time (centiseconds) |
| `full_lang_id()` | Get detected language ID |
| `is_multilingual()` | Check if model supports multiple languages |

### Audio Requirements

- **Sample rate**: 16000 Hz

- **Channels**: Mono

- **Format**: Float32 normalized to [-1.0, 1.0]

---

## Stable Diffusion Integration

Image generation using stable-diffusion.cpp. Supports SD 1.x/2.x, SDXL, SD3, FLUX, video generation (Wan/CogVideoX), and ESRGAN upscaling.

**Note**: Build with `WITH_STABLEDIFFUSION=1` to enable this module.

The module is exposed as `cyllama.sd` (CLI: `python -m cyllama.sd`). For broader narrative documentation, see [`docs/stable_diffusion.md`](stable_diffusion.md); this section is the API reference.

### Quick Start

```python
from cyllama.sd import text_to_image

# Simple text-to-image generation
image = text_to_image(
    model_path="models/sd_xl_turbo_1.0.q8_0.gguf",
    prompt="a photo of a cute cat",
    width=512,
    height=512,
    sample_steps=4,
    cfg_scale=1.0
)

# text_to_image returns a single SDImage; text_to_images returns a List[SDImage]
image.save("output.png")
```

### `text_to_image()`

Convenience function that creates a context, generates one image, and tears the context down. Returns a single `SDImage`. For batches use `text_to_images()`.

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
    vae_path: Optional[str] = None,
    taesd_path: Optional[str] = None,
    clip_l_path: Optional[str] = None,
    clip_g_path: Optional[str] = None,
    t5xxl_path: Optional[str] = None,
    control_net_path: Optional[str] = None,
    clip_skip: int = -1,
    eta: float = float('inf'),
    slg_scale: float = 0.0,
    vae_tiling: bool = False,
    hires_fix: bool = False,
    hires_scale: float = 2.0,
    offload_to_cpu: bool = False,
    keep_clip_on_cpu: bool = False,
    keep_vae_on_cpu: bool = False,
    diffusion_flash_attn: bool = False
) -> SDImage
```

`SampleMethod.COUNT` and `Scheduler.COUNT` are auto-detect sentinels — the C library picks based on the loaded model. `eta=float('inf')` resolves to a method-specific default. `hires_fix=True` enables hires-fix two-pass generation with default latent upscale; for finer control use `SDImageGenParams.set_hires_fix(...)`.

### `text_to_images()`

Same as `text_to_image()` but returns `List[SDImage]` and accepts `batch_count: int = 1`. Each image in the batch uses an incremented seed, producing variants of the same prompt.

### `image_to_image()`

Img2img convenience function. Note: builds a context with `vae_decode_only=False` so the encoder is available.

```python
def image_to_image(
    model_path: str,
    init_image: Union[SDImage, str],
    prompt: str,
    negative_prompt: str = "",
    strength: float = 0.75,
    seed: int = -1,
    sample_steps: int = 20,
    cfg_scale: float = 7.0,
    sample_method: SampleMethod = SampleMethod.COUNT,
    scheduler: Scheduler = Scheduler.COUNT,
    n_threads: int = -1,
    vae_path: Optional[str] = None,
    clip_skip: int = -1
) -> List[SDImage]
```

`init_image` accepts either an `SDImage` or a filesystem path; output dimensions are taken from the init image.

### `SDContext`

Persistent generation context — load the model once, generate many times.

```python
from cyllama.sd import SDContext, SDContextParams, SampleMethod, Scheduler

params = SDContextParams()
params.model_path = "models/sd_xl_turbo_1.0.q8_0.gguf"
params.n_threads = 4

with SDContext(params) as ctx:
    images = ctx.generate(
        prompt="a beautiful landscape",
        negative_prompt="blurry, ugly",
        width=512, height=512,
        sample_steps=4, cfg_scale=1.0,
        sample_method=SampleMethod.EULER,  # or COUNT for auto-detect
        scheduler=Scheduler.DISCRETE,
        hires_fix=False,
    )
```

`SDContext.generate(...)` accepts the same kwargs as `text_to_image()` plus `batch_count`, `init_image`, `mask_image`, `control_image`, `control_strength`, `strength`, and `flow_shift`. Returns `List[SDImage]`.

**Properties:**

- `is_valid` (bool): Context loaded successfully.

- `supports_image_generation` (bool): Model can run `generate()` (false for video-only models).

- `supports_video_generation` (bool): Model can run `generate_video()`.

**Methods:**

- `generate(**kwargs) -> List[SDImage]`: Text/img2img/inpaint/ControlNet generation.

- `generate_with_params(params: SDImageGenParams) -> List[SDImage]`: Low-level entry point taking a fully populated params object — needed for advanced features (LoRAs, reference images, Photo Maker, hires-fix model upscalers, full cache configuration).

- `generate_video(**kwargs) -> List[SDImage]`: Video frame generation (requires video-capable model).

- `default_sample_method(sample_method=None) -> SampleMethod`: Model's preferred sampler.

- `default_scheduler(sample_method=None) -> Scheduler`: Model's preferred scheduler.

### `SDContextParams`

Configuration for model loading.

```python
params = SDContextParams()
params.model_path = "model.gguf"          # Main model
params.vae_path = "vae.safetensors"       # Optional VAE
params.taesd_path = "taesd.safetensors"   # Optional TAESD (fast previews)
params.clip_l_path = "clip_l.safetensors" # Optional CLIP-L (SDXL/SD3)
params.clip_g_path = "clip_g.safetensors" # Optional CLIP-G (SDXL/SD3)
params.t5xxl_path = "t5xxl.safetensors"   # Optional T5-XXL (SD3/FLUX)
params.control_net_path = "cn.safetensors" # Optional ControlNet
params.n_threads = 4
params.vae_decode_only = True             # Set False for img2img
params.diffusion_flash_attn = False
params.offload_params_to_cpu = False      # Low-VRAM mode
params.keep_clip_on_cpu = False
params.keep_vae_on_cpu = False
params.wtype = SDType.COUNT               # COUNT = auto-detect
params.rng_type = RngType.CUDA
```

### `SDImage`

Image wrapper with numpy and PIL integration.

```python
from cyllama.sd import SDImage
import numpy as np

arr = np.zeros((512, 512, 3), dtype=np.uint8)
img = SDImage.from_numpy(arr)

print(img.width, img.height, img.channels)

arr = img.to_numpy()       # (H, W, C) uint8
pil_img = img.to_pil()     # requires Pillow

img.save("output.png")
img = SDImage.load("input.png")
```

### `SDImageGenParams`

Full generation parameters; pass to `SDContext.generate_with_params()`. The `text_to_image()` convenience function only exposes a curated subset — drop down to this class for LoRAs, reference images, Photo Maker, full cache control, hires-fix model upscalers, etc.

```python
from cyllama.sd import SDImageGenParams, SDImage, HiresUpscaler

params = SDImageGenParams()
params.prompt = "a cute cat"
params.negative_prompt = "ugly, blurry"
params.width = 512
params.height = 512
params.seed = 42
params.batch_count = 1
params.strength = 0.75           # For img2img
params.clip_skip = -1

# VAE tiling
params.vae_tiling_enabled = True
params.vae_tile_size = (512, 512)
params.vae_tile_overlap = 0.5

# Cache acceleration (legacy easycache_* aliases also available)
params.cache_mode = 1            # 0=disabled, 1=easycache, 2=ucache, 3=dbcache, 4=taylorseer, 5=cache_dit
params.cache_threshold = 0.1
params.cache_range = (0.0, 1.0)

# Hires-fix two-pass generation
params.set_hires_fix(
    enabled=True,
    upscaler=HiresUpscaler.LATENT,   # or LANCZOS, NEAREST, MODEL, ...
    scale=2.0,
    denoising_strength=0.7,
)
# ...individual setters also work:
# params.hires_enabled = True
# params.hires_target_size = (1024, 1024)
# params.hires_model_path = "/path/to/upscaler.gguf"  # required for HiresUpscaler.MODEL

# img2img / inpaint / ControlNet
params.set_init_image(SDImage.load("input.png"))
params.set_mask_image(SDImage.load("mask.png"))
params.set_control_image(control_img, strength=0.8)

# LoRAs and reference images
params.set_loras([{"path": "lora.safetensors", "multiplier": 0.8}])
params.set_ref_images([ref_img1, ref_img2])

# Sample params (delegated to nested SDSampleParams)
sample = params.sample_params
sample.sample_steps = 20
sample.cfg_scale = 7.0
sample.sample_method = SampleMethod.COUNT
sample.scheduler = Scheduler.COUNT
```

See `docs/stable_diffusion.md` for the full property catalog (Photo Maker, ControlNet refs, full cache configuration, all hires-fix fields).

### `SDSampleParams`

Sampling configuration. Usually accessed as `gen_params.sample_params` rather than instantiated directly.

```python
from cyllama.sd import SDSampleParams, SampleMethod, Scheduler

params = SDSampleParams()
params.sample_method = SampleMethod.COUNT
params.scheduler = Scheduler.COUNT
params.sample_steps = 20
params.cfg_scale = 7.0
params.eta = float('inf')        # inf = method-specific default
params.slg_scale = 0.0           # Skip layer guidance
params.flow_shift = float('inf') # Flow shift (SD3.x / Wan)
```

### `Upscaler`

ESRGAN-based image upscaling.

```python
from cyllama.sd import Upscaler, SDImage

upscaler = Upscaler(
    "models/esrgan-x4.bin",
    n_threads=4,
    offload_to_cpu=False,
    direct=False,         # direct convolution
    tile_size=0,          # 0 = default
)

print(f"Factor: {upscaler.upscale_factor}x")

img = SDImage.load("input.png")
upscaled = upscaler.upscale(img)               # use model's native factor
upscaled = upscaler.upscale(img, factor=2)     # or override
upscaled.save("upscaled.png")
```

`Upscaler` is also usable as a context manager (`with Upscaler(...) as up:`).

### `convert_model()`

Convert models between formats / quantize.

```python
from cyllama.sd import convert_model, SDType

convert_model(
    input_path="sd-v1-5.safetensors",
    output_path="sd-v1-5-q4_0.gguf",
    output_type=SDType.Q4_0,
    vae_path="vae-ft-mse.safetensors",   # optional
    tensor_type_rules=None,              # optional per-tensor type rules
    convert_name=False,                  # convert tensor names
)
```

Raises `FileNotFoundError` if the input is missing, `RuntimeError` on conversion failure.

### `canny_preprocess()`

Canny edge detection for ControlNet conditioning. Modifies the image in place.

```python
from cyllama.sd import SDImage, canny_preprocess

img = SDImage.load("photo.png")
success = canny_preprocess(
    img,
    high_threshold=0.8,
    low_threshold=0.1,
    weak=0.5,
    strong=1.0,
    inverse=False,
)
```

### Callbacks

```python
from cyllama.sd import (
    set_log_callback,
    set_progress_callback,
    set_preview_callback,
    PreviewMode,
)

# Logging: callback receives (LogLevel, str)
def log_cb(level, text):
    print(f'[{level.name}] {text}', end='')
set_log_callback(log_cb)

# Progress: callback receives (step, total_steps, time_seconds)
def progress_cb(step, steps, time_s):
    pct = (step / steps) * 100 if steps > 0 else 0
    print(f'Step {step}/{steps} ({pct:.1f}%) - {time_s:.2f}s')
set_progress_callback(progress_cb)

# Preview: callback receives (step, frames: List[SDImage], is_noisy: bool)
def preview_cb(step, frames, is_noisy):
    for i, frame in enumerate(frames):
        frame.save(f"preview_{step}_{i}.png")
set_preview_callback(
    preview_cb,
    mode=PreviewMode.TAE,
    interval=5,
    denoised=True,
    noisy=False,
)

# Pass None to clear any of them.
set_log_callback(None)
set_progress_callback(None)
set_preview_callback(None)
```

### Enums

**`SampleMethod`**

- `EULER`, `EULER_A`, `HEUN`, `DPM2`, `DPMPP2S_A`, `DPMPP2M`, `DPMPP2Mv2`
- `IPNDM`, `IPNDM_V`, `LCM`, `DDIM_TRAILING`, `TCD`
- `RES_MULTISTEP`, `RES_2S`, `ER_SDE`
- `COUNT` (auto-detect sentinel)

**`Scheduler`**

- `DISCRETE`, `KARRAS`, `EXPONENTIAL`, `AYS`, `GITS`
- `SGM_UNIFORM`, `SIMPLE`, `SMOOTHSTEP`, `KL_OPTIMAL`, `LCM`, `BONG_TANGENT`
- `COUNT` (auto-detect sentinel)

**`Prediction`**

- `EPS`, `V`, `EDM_V`, `FLOW`, `FLUX_FLOW`, `FLUX2_FLOW`, `COUNT`

**`SDType`**: Data types for model weights / quantization

- `F32`, `F16`, `BF16`
- `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`
- `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`
- `COUNT` (auto-detect sentinel)

**`RngType`**: `STD_DEFAULT`, `CUDA`, `CPU`

**`LogLevel`**: `DEBUG`, `INFO`, `WARN`, `ERROR`

**`PreviewMode`**: `NONE`, `PROJ`, `TAE`, `VAE`

**`LoraApplyMode`**: `AUTO`, `IMMEDIATELY`, `AT_RUNTIME`

**`HiresUpscaler`**: hires-fix upscaler modes

- `NONE`
- `LATENT`, `LATENT_NEAREST`, `LATENT_NEAREST_EXACT`, `LATENT_ANTIALIASED`, `LATENT_BICUBIC`, `LATENT_BICUBIC_ANTIALIASED`
- `LANCZOS`, `NEAREST`
- `MODEL` (external upscaler model — set `hires_model_path`)

### Utility Functions

```python
from cyllama.sd import (
    get_num_cores,
    get_system_info,
    type_name,
    sample_method_name,
    scheduler_name,
    ggml_backend_load_all,
)

ggml_backend_load_all()  # call before get_system_info() so GPU backends register
print(f"CPU cores: {get_num_cores()}")
print(get_system_info())

print(type_name(SDType.Q4_0))                  # "q4_0"
print(sample_method_name(SampleMethod.EULER))  # "euler"
print(scheduler_name(Scheduler.KARRAS))        # "karras"
```

### CLI Tool

```bash
# txt2img (alias: generate)
python -m cyllama.sd txt2img \
    --model models/sd_xl_turbo_1.0.q8_0.gguf \
    --prompt "a beautiful sunset" \
    --output sunset.png \
    --steps 4 --cfg 1.0

# img2img / inpaint / ControlNet / video
python -m cyllama.sd img2img --model M --init INPUT --prompt "..." --output OUT
python -m cyllama.sd inpaint --model M --init INPUT --mask MASK --prompt "..." --output OUT
python -m cyllama.sd controlnet --model M --control-net CN --control-image C --prompt "..." --output OUT
python -m cyllama.sd video --model M --prompt "..." --output frames/

# Upscale image
python -m cyllama.sd upscale \
    --model models/esrgan-x4.bin \
    --input image.png \
    --output image_4x.png

# Convert model
python -m cyllama.sd convert \
    --input sd-v1-5.safetensors \
    --output sd-v1-5-q4_0.gguf \
    --type q4_0

# Show system info
python -m cyllama.sd info
```

### Supported Models

- **SD 1.x/2.x**: Standard Stable Diffusion models

- **SDXL/SDXL Turbo**: Stable Diffusion XL (use cfg_scale=1.0, steps=1-4 for Turbo)

- **SD3/SD3.5**: Stable Diffusion 3.x

- **FLUX**: FLUX.1 models (dev, schnell)

- **Wan/CogVideoX**: Video generation models (use `generate_video()`)

- **LoRA**: Low-rank adaptation files

- **ControlNet**: Conditional generation with control images

- **ESRGAN**: Image upscaling models

---

## Error Handling

All cyllama functions raise appropriate Python exceptions:

```python
from cyllama import complete, LLM

try:
    response = complete("Hello", model_path="nonexistent.gguf")
except FileNotFoundError:
    print("Model file not found")
except RuntimeError as e:
    print(f"Runtime error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

# LLM with error handling
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
    complete,          # str | Iterator[str]
    chat,              # str | Iterator[str]
    LLM,               # class
    GenerationConfig,  # @dataclass
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
from cyllama import batch_generate, GenerationConfig

# BAD: Sequential processing
responses = [generate(p, model_path="model.gguf") for p in prompts]

# GOOD: Parallel batch processing (3-10x faster)
prompts = ["What is 2+2?", "What is 3+3?", "What is 4+4?"]
responses = batch_generate(
    prompts,
    model_path="model.gguf",
    n_seq_max=8,  # Max parallel sequences
    config=GenerationConfig(max_tokens=50, temperature=0.7)
)
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

- **Python**: >=3.10 (tested on 3.13)

- **llama.cpp**: see `CHANGELOG.md` for the pinned revision

- **Platform**: macOS, Linux, Windows

---

## See Also

- [User Guide](user_guide.md) - Comprehensive usage guide

- [Cookbook](cookbook.md) - Practical recipes and patterns

- [Changelog](https://github.com/shakfu/cyllama/blob/main/CHANGELOG.md) - Release history

- [llama.cpp Documentation](https://github.com/ggml-org/llama.cpp)

---

See `pyproject.toml` for the current cyllama version.
