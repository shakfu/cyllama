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
|    Embedder     | |SqliteVectorStore| |   Generator     |
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

## Pluggable Backends

`RAG` and `RAGPipeline` accept an injected embedder and vector store via the `embedder=` and `store=` constructor parameters:

```python
from cyllama.rag import RAG, SqliteVectorStore

rag = RAG(
    embedding_model="",  # ignored when embedder= is supplied
    generation_model="models/Llama-3.2-1B-Instruct-Q8_0.gguf",
    embedder=my_embedder,                                  # any EmbedderProtocol
    store=SqliteVectorStore(dimension=1536, db_path="x.db"),  # any VectorStoreProtocol
)
```

Both slots are typed as structural protocols (`EmbedderProtocol`, `VectorStoreProtocol` in `cyllama.rag.types`). Alternative backends — OpenAI embeddings, Qdrant, Chroma, pgvector, an in-house service — only need to implement the handful of methods the RAG layer actually calls to become drop-in replacements. See:

- [Embedder — Pluggable Backends](rag_embedder.md#pluggable-backends--embedderprotocol)
- [SqliteVectorStore — Pluggable Backends](rag_vectorstore.md#pluggable-backends--vectorstoreprotocol)

Omit the argument to fall back to the defaults (`Embedder` over a local GGUF embedding model and `SqliteVectorStore`).

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
| `SqliteVectorStore` | SQLite-based vector storage with sqlite-vector (default backend; implements `VectorStoreProtocol`). `VectorStore` remains as a deprecated alias. |
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
| `--temperature` | Generation temperature | 0.8 |
| `-k, --top-k` | Number of chunks to retrieve | 5 |
| `--threshold` | Minimum similarity threshold | (none) |
| `-ngl, --n-gpu-layers` | GPU layers to offload | -1 |
| `--stream` | Stream output tokens | off |
| `--sources` | Show source chunks with similarity scores | off |
| `--db PATH` | Persist the vector index to a SQLite file (see below) | (in-memory) |
| `--rebuild` | Delete the `--db` file and re-index from `-f`/`-d` | off |
| `--no-chat-template` | Use raw-completion prompting instead of the model's chat template | off (chat template on) |
| `--show-think` | Keep `<think>...</think>` reasoning blocks in the output | off (stripped) |
| `--repetition-threshold N` | Stop generation after the same n-gram repeats N times. `0` disables. | 2 |
| `--repetition-ngram N` | Word-level n-gram length for repetition detection | 5 |
| `--repetition-window N` | Rolling word-window size for repetition detection | 300 |

At least one document source (`-f` or `-d`) is required on the first run. With `--db PATH`, subsequent runs may omit `-f`/`-d` to query the existing index.

In interactive mode, type your questions at the `>` prompt. Press Ctrl+C or EOF to exit.

### Persistent Vector Store (CLI)

By default `cyllama rag` builds the index in memory and rebuilds it on every invocation. With `--db PATH`, the index is persisted to a SQLite file and reused on subsequent runs, so the corpus is embedded only once:

```bash
# First run: index the corpus and persist it
cyllama rag -m models/llama.gguf -e models/bge-small.gguf \
    --db rag.db -f corpus.txt -p "What is in the corpus?"

# Subsequent runs: reuse the persisted index without re-embedding
cyllama rag -m models/llama.gguf -e models/bge-small.gguf \
    --db rag.db -p "Another question?"

# Re-running with the same -f is a true no-op on indexing — the
# file's content hash is already in the dedup table:
cyllama rag -m models/llama.gguf -e models/bge-small.gguf \
    --db rag.db -f corpus.txt -p "..."
# > reusing N chunks from rag.db (1 unchanged)

# Switched embedding models or chunking? Use --rebuild:
cyllama rag -m models/llama.gguf -e models/bge-base.gguf \
    --db rag.db --rebuild -f corpus.txt -p "..."
```

Decision matrix:

| Args | Behavior |
|------|----------|
| `--db PATH` only, DB exists | Reuse existing index, no indexing |
| `--db PATH` + `-f/-d`, DB missing | Create DB, index sources |
| `--db PATH` + `-f/-d`, DB exists | Reopen DB, append (dedup-skipping unchanged sources) |
| `--db PATH --rebuild` + `-f/-d` | Delete DB, recreate, index sources |
| `--db PATH --rebuild` without `-f/-d` | Error (rebuild needs sources) |
| `--db PATH` missing, no `-f/-d` | Error (nothing to query) |

If the embedding model basename, chunk size, or chunk overlap on a reopen does not match what's stored in the DB's metadata table, `cyllama rag` exits with a clear error pointing at `--rebuild`. See [SqliteVectorStore — Metadata Validation](rag_vectorstore.md#metadata-validation) for details.

### Generation Defaults Worth Knowing

The CLI flips three `RAGConfig` fields from their library defaults because they fix common failure modes for chat-tuned and reasoning-tuned models. See [RAG Pipeline — RAGConfig](rag_pipeline.md#ragconfig) for the underlying fields.

| Behavior | CLI default | Disable with |
|----------|-------------|--------------|
| Native chat-template prompting | on | `--no-chat-template` |
| `<think>` block stripping | on | `--show-think` |
| N-gram repetition guard | on (`threshold=2`) | `--repetition-threshold 0` |

## Next Steps

- [Embedder](rag_embedder.md) - Generating embeddings
- [SqliteVectorStore](rag_vectorstore.md) - Vector storage and search
- [Text Processing](rag_text_processing.md) - Document splitting and loading
- [RAG Pipeline](rag_pipeline.md) - RAG pipeline configuration
- [Advanced RAG Features](rag_advanced.md) - Async, hybrid search, agent integration
