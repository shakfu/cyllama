# cyllama Update - November 2025

## Update on the cyllama Project

It's been nearly a year since my last announcement, and I wanted to share what's new with [cyllama](https://github.com/shakfu/cyllama) - the thin cython wrapper for llama.cpp.

A quick reminder: cyllama is a minimal, performant, compiled Python extension wrapping llama.cpp's core functionality. It statically links libllama.a and libggml.a for simplicity and performance (~1.2 MB wheel).

---

## What's Changed Since December 2024

Thanks to the targeted use of AI agents, the project has managed to keep up with the fast pace of changes at llama.cpp and is currently tracking release `b7126`. Here is a summary of some changes since the last post.

### 1. **High-Level Python API**

We now have a complete, Pythonic API layer that makes cyllama more pleasant to use:

```python
from cyllama import complete, chat, LLM

# Simple one-liner
response = complete("What is Python?", model_path="model.gguf")

# Reusable LLM instance (model stays loaded)
llm = LLM("model.gguf")
response = llm("Your question here")

# Multi-turn chat with proper message formatting
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing"}
]
response = chat(messages, model_path="model.gguf")
```

**Why this matters:** Previously, you had to manually manage models, contexts, samplers, and batches. Now it's automatic with sensible defaults, but full control is still available when needed.

### 2. **Chat Templates & Conversation Support**

Full support for chat templates and multi-turn conversations through the high-level API:

```python
from cyllama import chat

# Multi-turn conversation with automatic template formatting
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What is Python?"}
]
response = chat(messages, model_path="model.gguf")

# Or use the Chat class for interactive CLI
from cyllama.llama.chat import Chat

chat_session = Chat(model_path="model.gguf")
chat_session.chat_loop()  # Interactive chat with template auto-detection
```

**Features:**
- Automatic chat template detection from model metadata
- Supports built-in templates (ChatML, Llama-3, Mistral, etc.)
- Custom template support via `LlamaChatMessage` and `chat_apply_template()`
- Conversation history management

### 3. **Text-to-Speech (TTS) Support**

Full TTS integration for voice generation:

```python
from cyllama.llama import TTSGenerator

tts = TTSGenerator("models/outetts-0.2-500M-Q8_0.gguf")

# Generate speech from text
tts.generate(
    text="Hello, this is a test of the text to speech system.",
    output_file="output.wav"
)
```

**Features:**
- Supports OuteTTS and similar TTS models
- WAV file output with configurable sample rate
- Speaker voice cloning support
- Handles text preprocessing (numbers to words, etc.)
- Streaming audio generation

### 4. **Multimodal (LLAVA/Vision) Support**

Vision-language models for image understanding:

```python
from cyllama.llama.mtmd import MultimodalProcessor, VisionLanguageChat
from cyllama import LlamaModel, LlamaContext

# Load model and create processor
model = LlamaModel("models/llava-v1.6-mistral-7b.Q4_K_M.gguf")
ctx = LlamaContext(model)

# Initialize vision processor
processor = MultimodalProcessor("models/mmproj-model-f16.gguf", model)

# Or use high-level chat interface
vision_chat = VisionLanguageChat("models/mmproj-model-f16.gguf", model, ctx)
response = vision_chat.ask_about_image("What's in this image?", "image.jpg")
```

**Capabilities:**
- Image understanding and description
- Visual question answering
- Support for multiple images in conversation
- Works with LLAVA, BakLLaVA, and similar vision-language models
- Automatic vision capability detection

### 5. **Embedded HTTP Server (Mongoose)**

Production-ready [embedded HTTP server](https://github.com/cesanta/mongoose) with OpenAI-compatible API:

```python
from cyllama.llama.server import EmbeddedServer

# Create server with configuration
server = EmbeddedServer(
    model_path="model.gguf",
    host="127.0.0.1",
    port=8080
)

# Start server (runs in background thread)
server.start()

# Server provides OpenAI-compatible endpoints:
# POST /v1/chat/completions
# POST /v1/completions
# GET /v1/models
# GET /health
```

**Server Features:**
- OpenAI API compatibility (drop-in replacement)
- Streaming support (SSE)
- CORS support
- Graceful shutdown
- Thread-safe request handling
- Multiple server implementations:
  - `EmbeddedServer`: Python-based with threading
  - `MongooseServer`: High-performance C-based server
  - `LlamaServer`: Python wrapper around the llama.cpp server binary

**Example with curl:**
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7
  }'
```

### 6. **Framework Integrations**

**OpenAI-Compatible API:**
```python
from cyllama.integrations import OpenAIClient

client = OpenAIClient(model_path="model.gguf")
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7
)
```

**LangChain:**
```python
from cyllama.integrations import CyllamaLLM
from langchain.chains import LLMChain

llm = CyllamaLLM(model_path="model.gguf", temperature=0.7)
chain = LLMChain(llm=llm, prompt=prompt_template)
result = chain.run(topic="AI")
```

Both work seamlessly with existing code expecting OpenAI or LangChain interfaces.

### 7. **Performance Features**

- **Speculative Decoding**: 2-3x speedup using draft models
- **N-gram Cache**: 2-10x speedup for repetitive patterns (great for code completion)
- **Memory Optimization**: Automatic GPU layer estimation based on available VRAM

### 8. **Utility Features**

- **GGUF Manipulation**: Read/write model files, inspect/modify metadata
- **JSON Schema → Grammar**: Generate structured output with type safety
- **Model Downloads**: Ollama-style downloads from HuggingFace (`download_model("user/repo:quantization")`)

### 9. **Quality of Life**

- **Logging**: Debug output disabled by default (add `verbose=True` to enable)
- **Documentation**: Comprehensive user guide, API reference, and cookbook (1,200+ lines total)
- **Tests**: 276 passing tests for reliability
- **API Clarity**: Renamed `generate()` → `complete()` and `Generator` → `LLM` for better semantics

---

## Current Status

**Version:** 0.1.9 (November 21, 2025)
**llama.cpp Version:** b7126 (tracking bleeding-edge)
**whisper.cpp:** Integrated and tested
**Tests:** 276 passing
**Platform:** macOS (primary), Linux (tested)

**API Coverage - All Major Goals Met:**
- [x] Core llama.cpp wrapper (complete)
- [x] High-level Python API (complete)
- [x] llava-cli features (multimodal complete)
- [x] whisper.cpp integration (complete)
- [x] Chat templates and conversation support (complete)
- [x] TTS support (complete)
- [x] HTTP server with OpenAI API (complete)
- [ ] stable-diffusion.cpp (future)

---

## Why This Update Matters

**Before:** You needed 50+ lines of boilerplate to do basic inference, manually managing model lifecycle.

**Now:** One line for simple cases, with full power available when needed:

```python
# Text generation - one line!
response = complete("Your prompt", model_path="model.gguf")

# Chat conversations - easy!
response = chat(messages, model_path="model.gguf")

# TTS - simple!
tts.generate("Hello world", "output.wav")

# Vision - straightforward!
response = vision_chat.ask_about_image("What's in this?", "image.jpg")

# HTTP server - production-ready!
server = EmbeddedServer(model_path="model.gguf")
server.start()
```

The library is now genuinely production-ready for:
- Quick prototyping and experiments
- Chat applications with proper conversation handling
- Voice applications (TTS)
- Vision/multimodal applications (LLAVA)
- API servers (OpenAI-compatible)
- Integration into existing Python stacks (FastAPI, Flask, LangChain)
- Performance-critical applications (speculative decoding, n-gram caching)

---

## Use Cases Now Supported

1. **Text Generation**: Simple completions, structured output
2. **Chat Applications**: Multi-turn conversations with template support
3. **Voice Applications**: Text-to-speech with WAV output
4. **Vision Applications**: Image understanding and visual Q&A
5. **API Services**: Production HTTP servers with OpenAI compatibility
6. **Framework Integration**: Works with LangChain, OpenAI clients
7. **Performance**: Speculative decoding, n-gram caching

---

## Resources

- **Repo:** https://github.com/shakfu/cyllama
- **Docs:** See `docs/` directory (user guide, API reference, cookbook)
- **Examples:** See `tests/examples/` directory
  - Chat applications
  - TTS examples
  - Multimodal demos
  - Server implementations

---

## Feedback Welcome

As always, if you try it out:
- Questions? Ask away!
- Bugs? Please report them!
- Features? Suggestions welcome!
- Contributions? Pull requests accepted!

The goal remains: stay lean, stay fast, stay current with llama.cpp, and make it easy to use from Python.

---

## What's Next?

Potential future work:
- Async API support (`async def complete_async()`)
- Response caching
- RAG utilities
- stable-diffusion.cpp integration

But for now, the core feature set is comprehensive and ready to use.

---

**TL;DR:** Cyllama went from a thin wrapper requiring boilerplate to a batteries-included library with:
- High-level APIs for text, chat, TTS, and vision
- Production HTTP servers (Mongoose integration)
- Framework integrations (OpenAI, LangChain)
- Performance optimizations (speculative decoding, n-gram cache)

All while staying true to its minimal, compiled, performant roots (~1.2 MB wheel).

Give it a try and let me know what you think!
