# Cyllama User Guide

Complete guide to using cyllama for LLM inference.

## Table of Contents

1. [Getting Started](#getting-started)
2. [High-Level API](#high-level-api)
3. [Streaming Generation](#streaming-generation)
4. [Framework Integrations](#framework-integrations)
5. [Advanced Features](#advanced-features)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

```bash
git clone https://github.com/shakfu/cyllama.git
cd cyllama
make  # Downloads llama.cpp, builds everything
make download  # Download default test model
```

### Quick Start

The simplest way to generate text:

```python
from cyllama import complete

response = complete(
    "What is Python?",
    model_path="models/Llama-3.2-1B-Instruct-Q8_0.gguf"
)
print(response)
```

## High-Level API

### Basic Generation

The `complete()` function provides the simplest interface:

```python
from cyllama import complete, GenerationConfig

# Simple generation
response = complete(
    "Explain quantum computing",
    model_path="models/llama.gguf",
    max_tokens=200,
    temperature=0.7
)

# With configuration object
config = GenerationConfig(
    max_tokens=500,
    temperature=0.8,
    top_p=0.95,
    top_k=40,
    repeat_penalty=1.1
)

response = complete(
    "Write a poem about AI",
    model_path="models/llama.gguf",
    config=config
)
```

### Chat Interface

For multi-turn conversations:

```python
from cyllama import chat

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is..."},
    {"role": "user", "content": "Can you give an example?"}
]

response = chat(
    messages,
    model_path="models/llama.gguf",
    temperature=0.7,
    max_tokens=300
)
```

### LLM Class

For repeated generations, use the `LLM` class for better performance:

```python
from cyllama import LLM, GenerationConfig

# Create generator (loads model once)
gen = LLM("models/llama.gguf")

# Generate multiple times
prompts = [
    "What is AI?",
    "What is ML?",
    "What is DL?"
]

for prompt in prompts:
    response = gen(prompt)
    print(f"Q: {prompt}")
    print(f"A: {response}\n")
```

## Streaming Generation

Stream responses token-by-token:

```python
from cyllama import LLM

gen = LLM("models/llama.gguf")

# Stream to console
for chunk in gen("Tell me a story", stream=True):
    print(chunk, end="", flush=True)
print()

# Collect chunks
chunks = []
for chunk in gen("Count to 10", stream=True):
    chunks.append(chunk)
full_response = "".join(chunks)
```

### Token Callbacks

Process each token as it's generated:

```python
from cyllama import LLM

gen = LLM("models/llama.gguf")

tokens_seen = []

def on_token(token: str):
    tokens_seen.append(token)
    print(f"Token: {repr(token)}")

response = gen(
    "Hello world",
    on_token=on_token
)

print(f"\nTotal tokens: {len(tokens_seen)}")
```

## Framework Integrations

### OpenAI-Compatible API

Drop-in replacement for OpenAI client:

```python
from cyllama.integrations.openai_compat import OpenAICompatibleClient

client = OpenAICompatibleClient(
    model_path="models/llama.gguf",
    temperature=0.7
)

# Chat completions
response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"}
    ],
    max_tokens=200
)

print(response.choices[0].message.content)

# Streaming
for chunk in client.chat.completions.create(
    messages=[{"role": "user", "content": "Count to 5"}],
    stream=True
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### LangChain Integration

Use with LangChain chains and agents:

```python
from cyllama.integrations import CyllamaLLM

# Note: Requires langchain to be installed
# pip install langchain

llm = CyllamaLLM(
    model_path="models/llama.gguf",
    temperature=0.7,
    max_tokens=500
)

# Use in chains
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

prompt = PromptTemplate.from_template(
    "Tell me about {topic} in {style} style"
)

chain = LLMChain(llm=llm, prompt=prompt)

result = chain.run(topic="AI", style="simple")
print(result)

# Streaming with callbacks
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm_streaming = CyllamaLLM(
    model_path="models/llama.gguf",
    temperature=0.7,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)
```

## Advanced Features

### Configuration Options

Complete `GenerationConfig` options:

```python
from cyllama import GenerationConfig

config = GenerationConfig(
    # Generation limits
    max_tokens=512,           # Maximum tokens to generate

    # Sampling parameters
    temperature=0.8,          # 0.0 = greedy, higher = more random
    top_k=40,                 # Top-k sampling
    top_p=0.95,               # Nucleus sampling
    min_p=0.05,               # Minimum probability threshold
    repeat_penalty=1.1,       # Penalize repetition

    # Model parameters
    n_gpu_layers=99,          # Layers to offload to GPU (-1 = all)
    n_ctx=2048,               # Context window size
    n_batch=512,              # Batch size for processing

    # Control
    seed=42,                  # Random seed (-1 = random)
    stop_sequences=["END"],   # Stop generation at these strings

    # Tokenization
    add_bos=True,             # Add beginning-of-sequence token
    parse_special=True        # Parse special tokens
)
```

### Speculative Decoding

2-3x speedup with compatible models:

```python
from cyllama import (
    LlamaModel, LlamaContext, LlamaModelParams, LlamaContextParams,
    Speculative, SpeculativeParams
)

# Load target (main) model
model_target = LlamaModel("models/llama-3b.gguf", LlamaModelParams())
ctx_target = LlamaContext(model_target, LlamaContextParams())

# Load draft (smaller, faster) model
model_draft = LlamaModel("models/llama-1b.gguf", LlamaModelParams())
ctx_draft = LlamaContext(model_draft, LlamaContextParams())

# Setup speculative decoding
spec = Speculative(ctx_target, ctx_draft)
params = SpeculativeParams(
    n_draft=16,    # Number of tokens to draft
    p_min=0.75     # Acceptance probability
)

# Generate draft tokens
draft_tokens = spec.gen_draft(
    params,
    prompt_tokens=[1, 2, 3, 4],
    last_token=5
)
```

### Memory Estimation

Estimate GPU memory requirements:

```python
from cyllama import estimate_gpu_layers, estimate_memory_usage

# Estimate optimal GPU layers
estimate = estimate_gpu_layers(
    model_path="models/llama.gguf",
    available_vram_mb=8000,
    n_ctx=2048
)

print(f"Recommended GPU layers: {estimate.n_gpu_layers}")
print(f"Est. GPU memory: {estimate.gpu_memory_mb:.0f} MB")
print(f"Est. CPU memory: {estimate.cpu_memory_mb:.0f} MB")

# Detailed memory analysis
memory_info = estimate_memory_usage(
    model_path="models/llama.gguf",
    n_ctx=2048,
    n_batch=512
)

print(f"Model size: {memory_info.model_size_mb:.0f} MB")
print(f"KV cache: {memory_info.kv_cache_mb:.0f} MB")
print(f"Total: {memory_info.total_mb:.0f} MB")
```

## Performance Optimization

### GPU Acceleration

```python
from cyllama import LLM, GenerationConfig

# Offload all layers to GPU
config = GenerationConfig(n_gpu_layers=-1)  # or 99
gen = LLM("models/llama.gguf", config=config)

# Partial GPU offloading (for large models)
config = GenerationConfig(n_gpu_layers=20)  # First 20 layers only
```

### Batch Size Tuning

```python
# Larger batch = more throughput, more memory
config = GenerationConfig(n_batch=1024)

# Smaller batch = less memory, potentially slower
config = GenerationConfig(n_batch=128)
```

### Context Window Management

```python
# Auto-size context (prompt + max_tokens)
config = GenerationConfig(n_ctx=None, max_tokens=512)

# Fixed context size
config = GenerationConfig(n_ctx=4096, max_tokens=512)
```

## Troubleshooting

### Out of Memory

```python
# Reduce GPU layers
config = GenerationConfig(n_gpu_layers=10)

# Reduce context size
config = GenerationConfig(n_ctx=1024)

# Reduce batch size
config = GenerationConfig(n_batch=128)
```

### Slow Generation

```python
# Maximize GPU usage
config = GenerationConfig(n_gpu_layers=-1)

# Increase batch size
config = GenerationConfig(n_batch=512)

# Use speculative decoding (if you have a draft model)
```

### Quality Issues

```python
# More deterministic (lower temperature)
config = GenerationConfig(temperature=0.1)

# More diverse (higher temperature)
config = GenerationConfig(temperature=1.2)

# Adjust top-p for nucleus sampling
config = GenerationConfig(top_p=0.9)

# Reduce repetition
config = GenerationConfig(repeat_penalty=1.2)
```

### Import Errors

```bash
# Rebuild after updates
make build

# Clean rebuild
make remake

# Check installation
python -c "import cyllama; print(cyllama.__file__)"
```

## Best Practices

1. **Reuse Generators**: Create once, generate many times
2. **Reuse LLM Instances**: Keep the model loaded for multiple requests
3. **Monitor Memory**: Use memory estimation tools
4. **Tune Temperature**: Start at 0.7, adjust based on needs
5. **Use Stop Sequences**: Prevent over-generation
6. **Stream Long Outputs**: Better UX for users
7. **Profile Performance**: Measure before optimizing

## Examples

See the `tests/examples/` directory for complete working examples:

- `generate_example.py` - Basic generation
- `speculative_example.py` - Speculative decoding
- `integration_example.py` - Framework integrations

## Next Steps

- Read the [Cookbook](cookbook.md) for common patterns
- Check [API Reference](api_reference.md) for detailed documentation
- See [Examples](../tests/examples/) for complete code

## Support

- GitHub Issues: <https://github.com/shakfu/cyllama/issues>
- Documentation: <https://github.com/shakfu/cyllama/docs>
