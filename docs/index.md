# cyllama

This is the official documentation for **cyllama**, a high-performance Python library for local AI inference.

## About

cyllama provides high-performance Cython bindings to three C++ inference engines -- all from Python with zero runtime dependencies:

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** -- LLM text generation, chat, embeddings, and text-to-speech

- **[whisper.cpp](https://github.com/ggerganov/whisper.cpp)** -- Automatic speech recognition and translation

- **[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)** -- Image and video generation from text prompts

This documentation covers:

- **Installation and setup** across different platforms and GPU backends

- **Text generation** with llama.cpp for chat, completion, and embeddings

- **Speech recognition** with whisper.cpp for transcription and translation

- **Image generation** with stable-diffusion.cpp for text-to-image workflows

- **Agent framework** for building tool-using AI agents

- **RAG** for retrieval-augmented generation with local models

## Who This Is For

- **Python developers** who want to run LLMs locally without cloud dependencies

- **ML engineers** looking for a lightweight alternative to PyTorch-based inference

- **Application developers** building AI-powered features with predictable latency

- **Researchers** who need direct access to model internals and sampling parameters

## Prerequisites

- Python 3.10 or later

- Familiarity with command-line tools

- Understanding of what language models do (not how they work internally)

No machine learning expertise is required for basic usage.

## Conventions

Code examples use Python 3.10+ syntax:

```python
from cyllama import complete

response = complete("Hello!", model_path="models/llama.gguf")
```

Shell commands are shown with `bash` syntax:

```bash
make build
make test
```

## Source Code

cyllama is open source and available at:

<https://github.com/shakfu/cyllama>

Issues, contributions, and feedback are welcome.
