# Embedder

The `Embedder` class generates vector embeddings from text using llama.cpp embedding models in GGUF format.

## Basic Usage

```python
from cyllama.rag import Embedder

# Initialize with an embedding model
embedder = Embedder("models/bge-small-en-v1.5-q8_0.gguf")

# Embed a single text
embedding = embedder.embed("What is machine learning?")
print(f"Dimension: {len(embedding)}")  # e.g., 384

# Embed multiple texts efficiently
texts = [
    "Python is a programming language.",
    "Machine learning uses neural networks.",
    "Data science involves statistics."
]
embeddings = embedder.embed_batch(texts)
print(f"Generated {len(embeddings)} embeddings")

# Clean up
embedder.close()
```

## Constructor Options

```python
embedder = Embedder(
    model_path="models/bge-small.gguf",
    n_ctx=512,           # Context size (match model training)
    n_batch=512,         # Batch size for processing
    n_gpu_layers=-1,     # GPU layers (-1 = all)
    pooling="mean",      # Pooling strategy
    normalize=True       # L2 normalize embeddings
)
```

### Pooling Strategies

| Strategy | Description |
|----------|-------------|
| `mean` | Average all token embeddings (default) |
| `cls` | Use first token embedding (CLS token) |
| `last` | Use last token embedding |
| `none` | Return all token embeddings |

```python
from cyllama.rag import Embedder, PoolingType

# Using enum
embedder = Embedder(
    "model.gguf",
    pooling=PoolingType.CLS
)

# Or string
embedder = Embedder(
    "model.gguf",
    pooling="cls"
)
```

## Methods

### embed()

Embed a single text string:

```python
embedding = embedder.embed("Your text here")
# Returns: list[float] of dimension n_embd
```

### embed_batch()

Embed multiple texts efficiently:

```python
embeddings = embedder.embed_batch([
    "First document",
    "Second document",
    "Third document"
])
# Returns: list[list[float]]
```

### embed_documents()

Embed documents with optional progress tracking:

```python
embeddings = embedder.embed_documents(
    ["doc1", "doc2", "doc3"],
    show_progress=True  # Display progress bar
)
```

### embed_with_info()

Get embedding with additional metadata:

```python
result = embedder.embed_with_info("Your text here")
print(f"Embedding: {result.embedding[:5]}...")
print(f"Token count: {result.token_count}")
print(f"Truncated: {result.truncated}")
```

### embed_iter()

Generator for memory-efficient batch embedding:

```python
for embedding in embedder.embed_iter(large_text_list, batch_size=32):
    # Process each embedding
    store.add_one(embedding, text)
```

## Properties

```python
# Get embedding dimension
print(f"Dimension: {embedder.dimension}")  # e.g., 384

# Check if normalized
print(f"Normalized: {embedder.normalize}")
```

## Context Manager

Use context manager for automatic cleanup:

```python
from cyllama.rag import Embedder

with Embedder("models/bge-small.gguf") as embedder:
    embeddings = embedder.embed_batch(texts)
# Resources automatically released
```

## Normalization

By default, embeddings are L2-normalized (unit vectors). This is important for cosine similarity:

```python
import math

embedder = Embedder("model.gguf", normalize=True)
embedding = embedder.embed("test")

# Verify normalization
norm = math.sqrt(sum(x*x for x in embedding))
print(f"Norm: {norm}")  # Should be ~1.0
```

To disable normalization:

```python
embedder = Embedder("model.gguf", normalize=False)
```

## Example: Semantic Search

```python
from cyllama.rag import Embedder, SqliteVectorStore

# Initialize
embedder = Embedder("models/bge-small.gguf")

# Documents to index
documents = [
    "Python is a versatile programming language.",
    "JavaScript runs in web browsers.",
    "Rust provides memory safety without garbage collection.",
    "Go was designed for concurrent programming.",
]

# Generate embeddings and store
embeddings = embedder.embed_batch(documents)

with SqliteVectorStore(dimension=embedder.dimension) as store:
    store.add(embeddings, documents)

    # Search
    query = "Which language is good for web development?"
    query_embedding = embedder.embed(query)

    results = store.search(query_embedding, k=2)
    for result in results:
        print(f"[{result.score:.3f}] {result.text}")

embedder.close()
```

## Serving Embeddings over HTTP

The Embedder can be served via the built-in OpenAI-compatible server, allowing lightweight clients to generate embeddings without cyllama or GPU access locally:

```python
from cyllama.llama.server.python import ServerConfig, PythonServer

config = ServerConfig(
    model_path="models/Llama-3.2-1B-Instruct-Q8_0.gguf",
    embedding=True,
    embedding_model_path="models/bge-small-en-v1.5-q8_0.gguf",
    embedding_pooling="mean",
    embedding_normalize=True,
)

with PythonServer(config) as server:
    import time
    while True:
        time.sleep(1)
```

Clients can then call the standard `/v1/embeddings` endpoint:

```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "hello world"}'
```

Or use the built-in client:

```python
from cyllama.llama.server.launcher import LlamaServerClient

client = LlamaServerClient("http://localhost:8080")
result = client.embedding("hello world")
print(result["data"][0]["embedding"][:5])
```

All Embedder options (pooling strategy, normalization, context size, GPU layers) are configurable via the `ServerConfig` `embedding_*` parameters. See [Server Usage](server_usage_examples.md) for the full configuration reference.

## Pluggable Backends — `EmbedderProtocol`

`Embedder` is the default, llama.cpp-backed embedding backend, but `RAG` and `RAGPipeline` accept *any* object satisfying the structural contract `EmbedderProtocol` (declared in `cyllama.rag.types`). The contract is intentionally narrow — it covers only the members the RAG layer actually calls:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class EmbedderProtocol(Protocol):
    @property
    def dimension(self) -> int: ...
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
    def close(self) -> None: ...
```

Anything honouring these four members — an OpenAI-embeddings wrapper, a `sentence-transformers` adapter, a remote HTTP service client — can be passed via `RAG(embedder=...)`:

```python
from cyllama.rag import RAG, SqliteVectorStore

class MyEmbedder:
    dimension = 1536
    def embed(self, text): ...
    def embed_batch(self, texts): ...
    def close(self): ...

rag = RAG(
    embedding_model="",  # ignored when embedder= is supplied
    generation_model="models/Llama-3.2-1B-Instruct-Q8_0.gguf",
    embedder=MyEmbedder(),
    store=SqliteVectorStore(dimension=1536, db_path="docs.db"),
)
```

Passing `embedder=` skips the default `Embedder` construction entirely, so callers using a remote embedding API don't need a local GGUF embedding model. The RAG layer never calls backend-specific extensions (caching introspection, `embed_with_info`, async APIs) — those remain on the concrete `Embedder` and aren't part of the contract.

## Performance Tips

1. **Batch Processing**: Use `embed_batch()` instead of multiple `embed()` calls
2. **GPU Acceleration**: Set `n_gpu_layers=-1` to use all GPU layers
3. **Context Size**: Match `n_ctx` to your model's training context
4. **Memory Efficiency**: Use `embed_iter()` for large datasets
