# SqliteVectorStore

The `SqliteVectorStore` class provides SQLite-based vector storage using the sqlite-vector extension for high-performance similarity search. It is the default backend behind `RAG.store` and implements `VectorStoreProtocol`, so drop-in replacements (Qdrant, Chroma, pgvector, …) can be passed via `RAG(store=...)`.

> **Note:** the old name `VectorStore` is kept as a deprecated alias and will be removed in a future release. New code should import `SqliteVectorStore` directly.

## Basic Usage

```python
from cyllama.rag import SqliteVectorStore, Embedder

# Create embedder
embedder = Embedder("models/bge-small.gguf")

# Create vector store (in-memory)
store = SqliteVectorStore(dimension=embedder.dimension)

# Add embeddings
texts = ["Document 1", "Document 2", "Document 3"]
embeddings = embedder.embed_batch(texts)
ids = store.add(embeddings, texts)
print(f"Added {len(ids)} documents")

# Search
query_embedding = embedder.embed("search query")
results = store.search(query_embedding, k=2)
for result in results:
    print(f"[{result.score:.3f}] {result.text}")

# Clean up
store.close()
embedder.close()
```

## Constructor Options

```python
store = SqliteVectorStore(
    dimension=384,                       # Embedding dimension (required)
    db_path=":memory:",                  # Database path (":memory:" or file path)
    table_name="embeddings",             # Table name for vectors
    metric="cosine",                     # Distance metric
    vector_type="float32",               # Vector storage type
    embedding_model_path="bge.gguf",     # Optional: recorded for compat checks
    chunk_size=512,                      # Optional: recorded for compat checks
    chunk_overlap=50,                    # Optional: recorded for compat checks
)
```

The `embedding_model_path`, `chunk_size`, and `chunk_overlap` arguments are optional. When provided, they are written to the `{table_name}_meta` table on first creation and verified against the caller's values on every reopen — see [Metadata Validation](#metadata-validation) below. `RAG.__init__` forwards them automatically.

### Distance Metrics

| Metric | Description |
|--------|-------------|
| `cosine` | Cosine similarity (default, recommended) |
| `l2` | Euclidean distance |
| `dot` | Dot product |
| `l1` | Manhattan distance |
| `squared_l2` | Squared Euclidean distance |

### Vector Types

| Type | Description |
|------|-------------|
| `float32` | Full precision (default) |
| `float16` | Half precision (smaller storage) |
| `int8` | 8-bit integer (quantized) |
| `uint8` | Unsigned 8-bit integer |

## Adding Vectors

### add()

Add multiple embeddings with texts and optional metadata:

```python
embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
texts = ["Doc 1", "Doc 2"]
metadata = [{"source": "file1.txt"}, {"source": "file2.txt"}]

ids = store.add(embeddings, texts, metadata)
print(f"IDs: {ids}")  # [1, 2]
```

### add_one()

Add a single embedding:

```python
id = store.add_one(
    embedding=[0.1, 0.2, 0.3],
    text="Single document",
    metadata={"key": "value"}
)
```

## Searching

### search()

Find similar vectors:

```python
results = store.search(
    query_embedding=[0.1, 0.2, 0.3],
    k=5,                    # Number of results
    threshold=0.5           # Minimum similarity (optional)
)

for result in results:
    print(f"ID: {result.id}")
    print(f"Text: {result.text}")
    print(f"Score: {result.score}")
    print(f"Metadata: {result.metadata}")
```

## Retrieving Stored Data

### get()

Get stored item by ID:

```python
item = store.get("1")
if item:
    print(f"Text: {item.text}")
    print(f"Metadata: {item.metadata}")
```

### get_vector()

Get the embedding vector:

```python
vector = store.get_vector("1")
print(f"Vector: {vector[:5]}...")
```

## Deleting Data

### delete()

Delete by IDs:

```python
deleted = store.delete(["1", "2", "3"])
print(f"Deleted {deleted} items")
```

### clear()

Remove all data:

```python
count = store.clear()
print(f"Cleared {count} items")
```

## Persistence

### File-based Storage

```python
# Create persistent store
store = SqliteVectorStore(
    dimension=384,
    db_path="vectors.db"  # Will create this file
)

# Add data...
store.add(embeddings, texts)
store.close()
```

### Opening Existing Store

```python
# Re-open existing database
store = SqliteVectorStore.open("vectors.db")
results = store.search(query_embedding, k=5)
store.close()
```

## Metadata Validation

A persistent `SqliteVectorStore` records its configuration in a `{table_name}_meta` SQLite table on first creation:

- **Hard fields** (always validated on reopen): `dimension`, `metric`, `vector_type`

- **Soft fields** (validated only when the caller passes the matching kwarg): `embedding_model_basename`, `embedding_model_size_bytes`, `chunk_size`, `chunk_overlap`

- **Informational**: `cyllama_version`, `created_at`

On reopen, any mismatch between a stored hard field and the caller's value raises `VectorStoreError` with a message naming the stored value, the attempted value, and the fix. Soft fields only fire when the caller actually passes the corresponding constructor argument, so callers that don't care about embedding-model fingerprinting can opt out by simply not passing it.

```python
from cyllama.rag import SqliteVectorStore, VectorStoreError

# First run: creates the DB with metadata
store = SqliteVectorStore(
    dimension=384,
    db_path="vectors.db",
    embedding_model_path="models/bge-small.gguf",
    chunk_size=512,
    chunk_overlap=50,
)
store.close()

# Later: reopening with a different chunk size raises immediately
try:
    store = SqliteVectorStore(
        dimension=384,
        db_path="vectors.db",
        embedding_model_path="models/bge-small.gguf",
        chunk_size=1024,   # mismatch!
        chunk_overlap=50,
    )
except VectorStoreError as e:
    print(e)
    # "vectors.db was indexed with chunk_size=512 but the caller is
    #  opening it with chunk_size=1024. ... Either use the original
    #  chunk_size or pass --rebuild to recreate the index."
```

This catches the silent-corruption case where mixing two embedding models or two chunk configurations into a single index would produce garbage retrieval. It is the mechanism behind the `cyllama rag --rebuild` flag (see [RAG Overview — Persistent Vector Store](rag_overview.md#persistent-vector-store-cli)).

## Source Deduplication

A `SqliteVectorStore` also tracks per-source content hashes in a `{table_name}_sources` table — `(content_hash, source_label, chunk_count, indexed_at)`. The `add()` method accepts optional `source_hash` and `source_label` kwargs, written atomically with the chunk inserts in a single SQLite transaction so a process death between writes can't leave the store with orphaned chunks.

Three read methods are available:

```python
store.is_source_indexed(content_hash)   # bool: has this hash been added?
store.get_source_by_label(source_label) # row dict or None
store.list_sources()                    # all source rows, oldest first
```

These power the dedup logic in `RAG.add_documents` / `RAG.add_texts` (see [RAG Pipeline — Corpus Deduplication](rag_pipeline.md#corpus-deduplication)). Most users won't call them directly.

## Quantization for Large Datasets

For datasets with >10k vectors, quantization provides 4-5x faster search:

```python
# Add many vectors
store.add(large_embeddings, large_texts)

# Quantize for faster search
count = store.quantize(max_memory="30MB")
print(f"Quantized {count} vectors")

# Preload into memory for additional speedup
store.preload_quantization()

# Search now uses quantized index
results = store.search(query, k=10)
```

## Context Manager

```python
with SqliteVectorStore(dimension=384, db_path="data.db") as store:
    store.add(embeddings, texts)
    results = store.search(query)
# Automatically closed
```

## Properties

```python
# Number of stored vectors
print(f"Count: {len(store)}")

# Or use count property
print(f"Count: {store.count}")
```

## Example: Document Search System

```python
from cyllama.rag import Embedder, SqliteVectorStore

# Initialize
embedder = Embedder("models/bge-small.gguf")

# Knowledge base
documents = [
    {"text": "Python is great for data science.", "source": "python.txt"},
    {"text": "JavaScript powers the modern web.", "source": "js.txt"},
    {"text": "Rust provides memory safety.", "source": "rust.txt"},
    {"text": "Go excels at concurrent programming.", "source": "go.txt"},
]

# Create persistent store
with SqliteVectorStore(dimension=embedder.dimension, db_path="docs.db") as store:
    # Index documents
    for doc in documents:
        embedding = embedder.embed(doc["text"])
        store.add_one(
            embedding=embedding,
            text=doc["text"],
            metadata={"source": doc["source"]}
        )

    # Search
    query = "What language is good for backend?"
    query_emb = embedder.embed(query)

    results = store.search(query_emb, k=2)
    print(f"\nQuery: {query}\n")
    for r in results:
        print(f"[{r.score:.3f}] {r.text}")
        print(f"  Source: {r.metadata['source']}\n")

embedder.close()
```

## Pluggable Backends — `VectorStoreProtocol`

`SqliteVectorStore` is the default backend, but `RAG` and `RAGPipeline` accept *any* object satisfying the structural contract `VectorStoreProtocol` (declared in `cyllama.rag.types`). The contract covers only what the RAG layer actually calls:

```python
from typing import Protocol, runtime_checkable
from cyllama.rag import SearchResult

@runtime_checkable
class VectorStoreProtocol(Protocol):
    def search(self, query_embedding, k=5, threshold=None) -> list[SearchResult]: ...
    def add(self, embeddings, texts, metadata=None,
            source_hash=None, source_label=None) -> list[int]: ...
    def is_source_indexed(self, content_hash: str) -> bool: ...
    def get_source_by_label(self, source_label: str) -> dict | None: ...
    def clear(self) -> int: ...
    def close(self) -> None: ...
    def __len__(self) -> int: ...
```

This makes the RAG stack open to Qdrant, Chroma, LanceDB, pgvector, or any in-house vector service without forking `cyllama`:

```python
from cyllama.rag import RAG

class QdrantStoreAdapter:
    def search(self, query_embedding, k=5, threshold=None): ...
    def add(self, embeddings, texts, metadata=None,
            source_hash=None, source_label=None): ...
    # ... remaining protocol methods

rag = RAG(
    embedding_model="models/bge-small-en-v1.5-q8_0.gguf",
    generation_model="models/Llama-3.2-1B-Instruct-Q8_0.gguf",
    store=QdrantStoreAdapter(...),
)
```

Sqlite-specific features (quantization, FTS5 `HybridStore`, raw `store.conn` access) stay on `SqliteVectorStore` and aren't part of the contract. Backends without a natural dedup mechanism may return `False` / `None` from `is_source_indexed` / `get_source_by_label` — the RAG layer treats that as "always re-index" and still behaves correctly, just less efficiently on repeated `add_documents` calls.

## Performance Characteristics

- **1M vectors, 768 dimensions**: Few milliseconds query time

- **Memory footprint**: 30-50MB regardless of dataset size

- **No preindexing required**: Works immediately with your data

- **SIMD acceleration**: SSE2, AVX2, NEON support
