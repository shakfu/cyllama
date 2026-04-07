# RAG Support

Retrieval-Augmented Generation (RAG) enhances LLM responses by retrieving relevant context from a knowledge base before generation. cyllama provides a complete RAG solution using:

- **llama.cpp** for both embedding generation and text generation
- **sqlite-vector** for high-performance vector similarity search
- **SQLite FTS5** for hybrid keyword + semantic search

## Architecture

```text
                    +-----------------+
                    |   RAG Pipeline  |
                    +--------+--------+
                             |
         +-------------------+-------------------+
         |                   |                   |
+--------v--------+ +--------v--------+ +--------v--------+
|    Embedder     | |  VectorStore    | |   Generator     |
| (embedding LLM) | | (retrieval)     | | (generation LLM)|
+-----------------+ +-----------------+ +-----------------+
```

## Quick Start

The simplest way to use RAG is through the high-level `RAG` class:

```python
from cyllama.rag import RAG

# Initialize with embedding and generation models
rag = RAG(
    embedding_model="models/bge-small-en-v1.5-q8_0.gguf",
    generation_model="models/Llama-3.2-1B-Instruct-Q8_0.gguf"
)

# Add documents to the knowledge base
rag.add_texts([
    "Python is a high-level programming language known for its simplicity.",
    "Machine learning uses algorithms to learn patterns from data.",
    "Neural networks are inspired by biological brain structures."
])

# Or load from files
rag.add_documents(["docs/guide.md", "docs/api.txt"])

# Query the knowledge base
response = rag.query("What is Python?")
print(response.text)
print(f"Sources: {len(response.sources)}")

# Stream the response
for chunk in rag.stream("Explain machine learning"):
    print(chunk, end="", flush=True)

# Clean up
rag.close()
```

## Using Context Managers

For proper resource cleanup, use the context manager:

```python
from cyllama.rag import RAG

with RAG(
    embedding_model="models/bge-small.gguf",
    generation_model="models/llama.gguf"
) as rag:
    rag.add_texts(["Your documents here..."])
    response = rag.query("Your question?")
    print(response.text)
# Resources automatically cleaned up
```

## Components Overview

### Core Components

| Component | Description |
|-----------|-------------|
| `RAG` | High-level interface with sensible defaults |
| `AsyncRAG` | Async wrapper for non-blocking operations |
| `RAGPipeline` | Lower-level orchestration of retrieval + generation |
| `RAGConfig` | Configuration for retrieval and generation |

### Storage & Retrieval

| Component | Description |
|-----------|-------------|
| `Embedder` | Generate vector embeddings from text |
| `VectorStore` | SQLite-based vector storage with sqlite-vector |
| `HybridStore` | Combined FTS5 + vector search |

### Text Processing

| Component | Description |
|-----------|-------------|
| `TextSplitter` | Recursive character text splitting |
| `TokenTextSplitter` | Token-based splitting |
| `MarkdownSplitter` | Markdown-aware splitting |

### Document Loaders

| Component | Description |
|-----------|-------------|
| `TextLoader` | Plain text files |
| `MarkdownLoader` | Markdown with frontmatter |
| `JSONLoader` | JSON with configurable extraction |
| `JSONLLoader` | JSON Lines with lazy loading |
| `DirectoryLoader` | Batch loading from directories |
| `PDFLoader` | PDF files (requires `docling`) |

### Advanced Features

| Component | Description |
|-----------|-------------|
| `Reranker` | Cross-encoder reranking |
| `create_rag_tool` | Agent integration |
| `async_search_knowledge` | Async search helper |

## Embedding Models

cyllama uses llama.cpp embedding models in GGUF format. Recommended models:

| Model | Dimension | Size | Notes |
|-------|-----------|------|-------|
| bge-small-en-v1.5 | 384 | ~130MB | Good quality/size balance |
| bge-base-en-v1.5 | 768 | ~440MB | Higher quality |
| snowflake-arctic-embed-s | 384 | ~130MB | Fast, accurate |
| all-MiniLM-L6-v2 | 384 | ~90MB | Lightweight |
| nomic-embed-text-v1.5 | 768 | ~550MB | Long context (8192) |

### Downloading Models

```bash
# Using huggingface-cli
huggingface-cli download BAAI/bge-small-en-v1.5-gguf bge-small-en-v1.5-q8_0.gguf

# Or directly with wget
wget https://huggingface.co/BAAI/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-q8_0.gguf
```

## Serving Embeddings over HTTP

The Embedder can also be served via the built-in OpenAI-compatible server (`PythonServer` or `EmbeddedServer`). This lets lightweight clients generate embeddings over HTTP without installing cyllama or having GPU access locally:

```python
from cyllama.llama.server.python import ServerConfig, PythonServer

config = ServerConfig(
    model_path="models/Llama-3.2-1B-Instruct-Q8_0.gguf",
    embedding=True,
    embedding_model_path="models/bge-small-en-v1.5-q8_0.gguf",
)

with PythonServer(config) as server:
    # Serves /v1/chat/completions and /v1/embeddings
    import time
    while True:
        time.sleep(1)
```

```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "hello world"}'
```

See [Embedder docs](rag_embedder.md#serving-embeddings-over-http) and [Server Usage](server_usage_examples.md) for configuration details.

## Command-Line Interface

The `cyllama rag` command provides command-line RAG without writing any Python:

```bash
# Single query against a directory
cyllama rag -m models/llama.gguf -e models/bge-small.gguf \
    -d docs/ -p "How do I configure the system?"

# Index specific files and enter interactive mode (omit -p)
cyllama rag -m models/llama.gguf -e models/bge-small.gguf \
    -f guide.md -f faq.md

# Stream output and show source chunks
cyllama rag -m models/llama.gguf -e models/bge-small.gguf \
    -d docs/ -p "Summarize the architecture" --stream --sources

# Custom system instruction and retrieval settings
cyllama rag -m models/llama.gguf -e models/bge-small.gguf \
    -d docs/ -s "Answer in one paragraph" -k 3 --threshold 0.4
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --model` | Path to GGUF generation model | (required) |
| `-e, --embedding-model` | Path to GGUF embedding model | (required) |
| `-f, --file` | File to index (repeatable) | |
| `-d, --dir` | Directory to index (repeatable) | |
| `--glob` | Glob pattern for directory loading | `**/*` |
| `-p, --prompt` | Single query (omit for interactive mode) | |
| `-s, --system` | System instruction prepended to the prompt template | |
| `-n, --max-tokens` | Maximum tokens to generate | 200 |
| `--temperature` | Generation temperature | 0.7 |
| `-k, --top-k` | Number of chunks to retrieve | 5 |
| `--threshold` | Minimum similarity threshold | (none) |
| `-ngl, --n-gpu-layers` | GPU layers to offload | 99 |
| `--stream` | Stream output tokens | off |
| `--sources` | Show source chunks with similarity scores | off |

At least one document source (`-f` or `-d`) is required.

In interactive mode, type your questions at the `>` prompt. Press Ctrl+C or EOF to exit.

## Next Steps

- [Embedder](rag_embedder.md) - Generating embeddings
- [VectorStore](rag_vectorstore.md) - Vector storage and search
- [Text Processing](rag_text_processing.md) - Document splitting and loading
- [RAG Pipeline](rag_pipeline.md) - RAG pipeline configuration
- [Advanced RAG Features](rag_advanced.md) - Async, hybrid search, agent integration
