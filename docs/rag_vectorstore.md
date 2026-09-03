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

This makes the RAG stack open to Qdrant, Chroma, LanceDB, pgvector, or any in-house vector service without forking `cyllama`.

Four adapters ship in `cyllama.rag.stores`, each lazy-imported so `import cyllama.rag` stays free of the optional dependency:

| Adapter | Install | Notes |
|---------|---------|-------|
| `QdrantVectorStore` | `pip install qdrant-client` | `:memory:`, on-disk, or remote server |
| `SqliteVecStore` | `pip install sqlite-vec` | MIT/Apache-2.0 licensed SQLite backend |
| `ChromaVectorStore` | `pip install chromadb` | Ephemeral, on-disk, or remote server |
| `PgVectorStore` | `pip install "psycopg[binary]" pgvector` | PostgreSQL; needs a running server |

Install the clients directly (`pip install qdrant-client chromadb sqlite-vec "psycopg[binary]" pgvector`) to un-skip `tests/test_rag_{qdrant,chroma,sqlite_vec}.py` locally — cyllama ships no extras or dependency groups for them and takes no position on their version pins. CI covers them through the `test-store-adapters` workflow, which runs those tests weekly against current releases to catch upstream drift.

### Qdrant

`QdrantVectorStore` ships in `cyllama.rag.stores.qdrant` as the first worked example of the protocol. Install the optional dependency (`pip install qdrant-client`) and pass it to `RAG`:

```python
from cyllama.rag import RAG
from cyllama.rag.stores import QdrantVectorStore

store = QdrantVectorStore(
    dimension=384,
    collection_name="cyllama_docs",
    url="http://localhost:6333",  # or path=..., location=":memory:", client=<pre-built>
)

rag = RAG(
    embedding_model="models/bge-small-en-v1.5-q8_0.gguf",
    generation_model="models/Llama-3.2-1B-Instruct-Q8_0.gguf",
    store=store,
)
```

Source dedup is implemented via per-point payload fields (`content_hash`, `source_label`, `indexed_at`) so `RAG.add_documents` skips unchanged files just like on the sqlite backend. See `src/cyllama/rag/stores/qdrant.py` for the full implementation.

### sqlite-vec

`SqliteVecStore` backs the same SQLite-file workflow as the default store, but with [sqlite-vec](https://github.com/asg017/sqlite-vec) — which is dual MIT/Apache-2.0 licensed, unlike the vendored `sqlite-vector` extension (Elastic License 2.0: free for open-source projects, paid for commercial use). If that licensing matters for your deployment, this is the drop-in.

```python
from cyllama.rag import RAG
from cyllama.rag.stores import SqliteVecStore

store = SqliteVecStore(
    dimension=384,
    db_path="vectors.db",          # ":memory:" for ephemeral
    metric="cosine",               # cosine | l2 | squared_l2 | l1
    vector_type="float32",         # float32 | int8
)
```

The extension comes from the `sqlite-vec` PyPI package by default; pass `extension_path=...` to use your own build. Vectors live in a `vec0` virtual-table sidecar (`{table_name}_vec`) keyed by the base table's `id`, which keeps the chunk rows in an ordinary table — so FTS5 triggers still work over them.

Differences from `SqliteVectorStore`:

- No `dot` metric and no `uint8` vector type — `vec0` offers cosine/L2/L1 and float32/int8. (`vec0` parses a `float16` column type but stores it as float32 as of sqlite-vec 0.1.9, so the adapter doesn't offer it.)
- No `quantize()` / `preload_quantization()`. A `vec0` table is either exhaustive or built with an ANN index at CREATE time; there is no runtime quantization step.
- The on-disk format is different, so an existing `SqliteVectorStore` database has to be re-indexed rather than opened.

For a full comparison — licensing, benchmarks, and what a default-backend swap would cost — see [`docs/dev/use-sqlite-vec.md`](dev/use-sqlite-vec.md).

### Chroma

`ChromaVectorStore` adapts [Chroma](https://github.com/chroma-core/chroma), with the same transport choice as the Qdrant adapter:

```python
from cyllama.rag.stores import ChromaVectorStore

store = ChromaVectorStore(dimension=384)                       # ephemeral, in-process
store = ChromaVectorStore(dimension=384, path="./chroma")      # local on-disk
store = ChromaVectorStore(dimension=384, host="localhost", port=8000)  # remote server
store = ChromaVectorStore(dimension=384, client=my_client)     # caller-owned client
```

Metrics are `cosine`, `l2` and `dot`. The collection is created without an embedding function — cyllama always supplies the vectors itself. Chroma only stores scalar metadata values, so the adapter JSON-encodes anything nested on the way in and decodes it on the way out; arbitrary JSON-serializable metadata round-trips unchanged. Source dedup uses `content_hash` / `source_label` / `indexed_at` metadata fields, mirroring the Qdrant adapter. Note that Chroma requires collection names of 3-512 characters from `[a-zA-Z0-9._-]`, starting and ending alphanumeric.

### pgvector

`PgVectorStore` adapts [pgvector](https://github.com/pgvector/pgvector). It is the most capable of the four adapters — it covers **every metric the default backend does**, including the `dot` that sqlite-vec cannot offer — but it is the only one with no in-process mode: a reachable PostgreSQL server is a hard requirement.

```python
from cyllama.rag.stores import PgVectorStore

store = PgVectorStore(dimension=384, dsn="postgresql://user:pw@localhost/rag")
store = PgVectorStore(dimension=384, conn=my_psycopg_connection)  # caller-owned
```

| `metric` | Operator | Notes |
|----------|----------|-------|
| `cosine` | `<=>` | default |
| `l2` | `<->` | euclidean |
| `squared_l2` | `<->` | squared in Python; ordering identical |
| `dot` | `<#>` | score is the plain inner product |
| `l1` | `<+>` | requires pgvector >= 0.7.0, checked at construction |

Metadata is stored as native `JSONB`, so nested values round-trip with no encoding. Chunks live in `{table_name}`, with `{table_name}_meta` and `{table_name}_sources` alongside; reopening with a different `dimension` or `metric` raises rather than corrupting the table, and the declared `vector(N)` column width is cross-checked against the catalog.

Two pgvector-specific extras sit outside `VectorStoreProtocol` — this is pgvector's answer to `SqliteVectorStore.quantize()`:

```python
store.create_index("hnsw", m=16, ef_construction=64)   # or "ivfflat", lists=100
store.drop_index("hnsw")
print(store.pgvector_version)
```

The operator class is derived from the store's `metric`, so the index actually serves the queries `search()` issues. Build it after bulk loading, not before.

For local development without installing PostgreSQL, `pip install pgserver` (Python <= 3.12) ships a PostgreSQL binary with pgvector bundled and needs no root:

```python
import pgserver
db = pgserver.get_server("/tmp/pgdata")
store = PgVectorStore(dimension=384, dsn=db.get_uri())
```


Sqlite-specific features (quantization, FTS5 `HybridStore`, raw `store.conn` access) stay on `SqliteVectorStore` and aren't part of the contract. Backends without a natural dedup mechanism may return `False` / `None` from `is_source_indexed` / `get_source_by_label` — the RAG layer treats that as "always re-index" and still behaves correctly, just less efficiently on repeated `add_documents` calls.

## Performance Characteristics

- **1M vectors, 768 dimensions**: Few milliseconds query time

- **Memory footprint**: 30-50MB regardless of dataset size

- **No preindexing required**: Works immediately with your data

- **SIMD acceleration**: SSE2, AVX2, NEON support
