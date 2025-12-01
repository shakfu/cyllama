# RAG Support Design Document

This document outlines the design for Retrieval-Augmented Generation (RAG) support in cyllama.

## Overview

RAG enhances LLM responses by retrieving relevant context from a knowledge base before generation. This design provides a minimal-dependency RAG implementation using:

1. **llama.cpp** - Native embedding support for vector generation
2. **sqlite-vector** - SQLite extension for high-performance vector similarity search

## Architecture

```
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

### Components

1. **Embedder** - Generates vector embeddings from text using llama.cpp
2. **VectorStore** - Stores and retrieves embeddings with similarity search
3. **TextSplitter** - Chunks documents for indexing
4. **RAGPipeline** - Orchestrates retrieval and generation

## API Design

### High-Level API

```python
from cyllama.rag import RAG, Embedder, VectorStore

# Simple one-liner for common use case
rag = RAG(
    embedding_model="models/bge-small-en-v1.5-q8_0.gguf",
    generation_model="models/Llama-3.2-1B-Instruct-Q8_0.gguf"
)

# Index documents
rag.add_documents([
    "doc1.txt",
    "doc2.pdf",
    "doc3.md"
])

# Or add text directly
rag.add_texts([
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Tokyo is the capital of Japan."
])

# Query
response = rag.query("What is the capital of France?")
print(response.text)
print(response.sources)  # Retrieved chunks used
```

### Embedder Class

```python
from cyllama.rag import Embedder

class Embedder:
    """Generate embeddings using llama.cpp embedding models."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 512,
        n_batch: int = 512,
        n_gpu_layers: int = -1,
        pooling: str = "mean",  # "mean", "cls", "last", "none"
        normalize: bool = True
    ):
        """
        Initialize embedder with an embedding model.

        Args:
            model_path: Path to GGUF embedding model (BGE, Snowflake, etc.)
            n_ctx: Context size (should match model's training)
            n_batch: Batch size for processing
            n_gpu_layers: GPU layers (-1 = all)
            pooling: Pooling strategy for sequence embeddings
            normalize: Whether to L2-normalize output vectors
        """
        ...

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts efficiently."""
        ...

    def embed_documents(
        self,
        documents: list[str],
        show_progress: bool = True
    ) -> list[list[float]]:
        """Embed documents with progress tracking."""
        ...

    @property
    def dimension(self) -> int:
        """Return embedding dimension (n_embd)."""
        ...
```

### VectorStore Class

The VectorStore uses **sqlite-vector**, a high-performance SQLite extension with SIMD-accelerated
distance functions. It supports multiple vector types (Float32, Float16, Int8, UInt8) and distance
metrics (L2, Cosine, Dot Product, L1).

```python
from cyllama.rag import VectorStore

class VectorStore:
    """SQLite-based vector store using sqlite-vector extension."""

    def __init__(
        self,
        dimension: int,
        db_path: str = ":memory:",
        table_name: str = "embeddings",
        metric: str = "cosine",  # "cosine", "l2", "dot", "l1"
        vector_type: str = "FLOAT32",  # "FLOAT32", "FLOAT16", "INT8", "UINT8"
        use_quantization: bool = False  # Enable for large datasets (>100k vectors)
    ):
        """
        Initialize vector store with sqlite-vector.

        Args:
            dimension: Embedding dimension
            db_path: SQLite database path (":memory:" for in-memory)
            table_name: Name of the embeddings table
            metric: Distance metric for similarity
            vector_type: Vector storage type
            use_quantization: Enable quantized search for large datasets
        """
        ...

    def add(
        self,
        embeddings: list[list[float]],
        texts: list[str],
        metadata: list[dict] | None = None
    ) -> list[str]:
        """
        Add embeddings with associated texts and metadata.

        Uses sqlite-vector's vector_as_f32() for proper BLOB encoding.

        Returns:
            List of generated IDs for the added items.
        """
        ...

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        threshold: float | None = None
    ) -> list[SearchResult]:
        """
        Find k most similar embeddings using sqlite-vector.

        Uses vector_full_scan() for small datasets or
        vector_quantize_scan() for quantized large datasets.

        Args:
            query_embedding: Query vector
            k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of SearchResult(id, text, score, metadata)
        """
        ...

    def quantize(self, max_memory: str = "30MB") -> int:
        """
        Quantize vectors for faster approximate search.

        Call this after bulk inserts for datasets >10k vectors.
        Returns the number of quantized rows.
        """
        ...

    def preload_quantization(self) -> None:
        """Load quantized data into memory for 4-5x speedup."""
        ...

    def delete(self, ids: list[str]) -> None:
        """Delete embeddings by ID."""
        ...

    @classmethod
    def open(cls, db_path: str, table_name: str = "embeddings") -> "VectorStore":
        """Open existing vector store from disk."""
        ...

    def close(self) -> None:
        """Close the database connection."""
        ...

    def __len__(self) -> int:
        """Return number of stored embeddings."""
        ...

    def __enter__(self) -> "VectorStore":
        return self

    def __exit__(self, *args) -> None:
        self.close()
```

### TextSplitter Class

```python
from cyllama.rag import TextSplitter

class TextSplitter:
    """Split text into chunks for embedding."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] | None = None
    ):
        """
        Initialize text splitter.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            separators: Hierarchy of separators to split on
                        Default: ["\n\n", "\n", ". ", " ", ""]
        """
        ...

    def split(self, text: str) -> list[str]:
        """Split text into chunks."""
        ...

    def split_documents(
        self,
        documents: list[Document]
    ) -> list[Document]:
        """Split documents, preserving metadata."""
        ...
```

### RAGPipeline Class

```python
from cyllama.rag import RAGPipeline, RAGConfig, RAGResponse

@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    # Retrieval settings
    top_k: int = 5
    similarity_threshold: float | None = None

    # Generation settings
    max_tokens: int = 512
    temperature: float = 0.7

    # Prompt template
    prompt_template: str = """Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""

@dataclass
class RAGResponse:
    """Response from RAG query."""
    text: str
    sources: list[SearchResult]
    stats: GenerationStats | None = None

class RAGPipeline:
    """Complete RAG pipeline combining retrieval and generation."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        generator: LLM,
        config: RAGConfig | None = None
    ):
        ...

    def query(
        self,
        question: str,
        config: RAGConfig | None = None
    ) -> RAGResponse:
        """
        Answer a question using RAG.

        1. Embed the question
        2. Retrieve relevant chunks
        3. Format prompt with context
        4. Generate response
        """
        ...

    def stream(
        self,
        question: str,
        config: RAGConfig | None = None
    ) -> Iterator[str]:
        """Stream response tokens."""
        ...
```

### Convenience RAG Class

```python
from cyllama.rag import RAG

class RAG:
    """High-level RAG interface with sensible defaults."""

    def __init__(
        self,
        embedding_model: str,
        generation_model: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        **kwargs
    ):
        """
        Initialize RAG with models.

        Creates Embedder, VectorStore, TextSplitter, and RAGPipeline
        with sensible defaults.
        """
        self.embedder = Embedder(embedding_model)
        self.store = VectorStore(dimension=self.embedder.dimension)
        self.splitter = TextSplitter(chunk_size, chunk_overlap)
        self.generator = LLM(generation_model, **kwargs)
        self.pipeline = RAGPipeline(
            self.embedder, self.store, self.generator
        )

    def add_texts(
        self,
        texts: list[str],
        metadata: list[dict] | None = None
    ) -> None:
        """Add text strings to the knowledge base."""
        chunks = []
        for text in texts:
            chunks.extend(self.splitter.split(text))
        embeddings = self.embedder.embed_documents(chunks)
        self.store.add(embeddings, chunks, metadata)

    def add_documents(
        self,
        paths: list[str],
        **loader_kwargs
    ) -> None:
        """Load and add documents from files."""
        for path in paths:
            text = self._load_document(path, **loader_kwargs)
            self.add_texts([text], metadata=[{"source": path}])

    def query(self, question: str, **kwargs) -> RAGResponse:
        """Query the knowledge base."""
        return self.pipeline.query(question, **kwargs)

    def save(self, path: str) -> None:
        """Save the vector store to disk."""
        self.store.save(path)

    def load(self, path: str) -> None:
        """Load a saved vector store."""
        self.store = VectorStore.load(path)
        self.pipeline.store = self.store
```

## Implementation Details

### Embedding Generation

Using cyllama's existing low-level API:

```python
def _generate_embedding(self, text: str) -> list[float]:
    """Generate embedding using llama.cpp."""
    # Tokenize
    tokens = self.vocab.tokenize(text, add_special=True)

    # Truncate if needed
    if len(tokens) > self.n_ctx:
        tokens = tokens[:self.n_ctx]

    # Create batch
    batch = LlamaBatch(n_tokens=len(tokens), embd=0, n_seq_max=1)
    for i, token in enumerate(tokens):
        common_batch_add(batch, token, i, [0], True)

    # Decode
    self.ctx.decode(batch)

    # Get embeddings
    embeddings = self.ctx.get_embeddings()

    # Pool if needed (for models that output per-token embeddings)
    if self.pooling == "mean":
        embeddings = self._mean_pool(embeddings, len(tokens))
    elif self.pooling == "cls":
        embeddings = embeddings[:self.n_embd]  # First token

    # Normalize
    if self.normalize:
        embeddings = self._l2_normalize(embeddings)

    return list(embeddings)
```

### Vector Store with sqlite-vector

The VectorStore uses the **sqlite-vector** extension for high-performance similarity search.
This provides SIMD-accelerated distance functions (SSE2, AVX2, NEON) with minimal memory footprint.

**Key Features:**
- No preindexing required - works immediately with your data
- SIMD-accelerated distance functions (L2, Cosine, Dot, L1)
- Quantization for large datasets (>10k vectors) with >0.95 recall
- Handles 1M vectors in milliseconds with <50MB RAM

```python
import sqlite3
import struct
import json
from pathlib import Path

class VectorStore:
    """SQLite-based vector store using sqlite-vector extension."""

    # Path to sqlite-vector extension (without file extension)
    # The extension is built to: src/cyllama/rag/vector.{dylib,so,dll}
    # At runtime, this resolves relative to the rag module location
    EXTENSION_PATH = Path(__file__).parent / "vector"

    def __init__(
        self,
        dimension: int,
        db_path: str = ":memory:",
        table_name: str = "embeddings",
        metric: str = "cosine",
        vector_type: str = "FLOAT32"
    ):
        self.dimension = dimension
        self.table_name = table_name
        self.metric = metric.upper()
        self.vector_type = vector_type.upper()
        self._quantized = False

        # Connect and load extension
        self.conn = sqlite3.connect(db_path)
        self.conn.enable_load_extension(True)
        self.conn.load_extension(str(self.EXTENSION_PATH))

        # Create table
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                metadata TEXT
            )
        """)

        # Initialize vector extension for this table
        distance = self.metric if self.metric != "EUCLIDEAN" else "L2"
        self.conn.execute(f"""
            SELECT vector_init('{table_name}', 'embedding',
                'dimension={dimension},type={vector_type},distance={distance}')
        """)
        self.conn.commit()

    def _encode_vector(self, vector: list[float]) -> bytes:
        """Encode vector as binary BLOB (Float32)."""
        return struct.pack(f'{len(vector)}f', *vector)

    def add(
        self,
        embeddings: list[list[float]],
        texts: list[str],
        metadata: list[dict] | None = None
    ) -> list[int]:
        """Add embeddings with associated texts."""
        if metadata is None:
            metadata = [{}] * len(embeddings)

        ids = []
        cursor = self.conn.cursor()
        for emb, text, meta in zip(embeddings, texts, metadata):
            blob = self._encode_vector(emb)
            cursor.execute(
                f"INSERT INTO {self.table_name} (text, embedding, metadata) VALUES (?, ?, ?)",
                (text, blob, json.dumps(meta))
            )
            ids.append(cursor.lastrowid)
        self.conn.commit()

        # Invalidate quantization on new data
        self._quantized = False
        return ids

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        threshold: float | None = None
    ) -> list["SearchResult"]:
        """Find k most similar embeddings."""
        query_blob = self._encode_vector(query_embedding)

        # Use quantized search if available, otherwise full scan
        if self._quantized:
            scan_fn = "vector_quantize_scan"
        else:
            scan_fn = "vector_full_scan"

        cursor = self.conn.execute(f"""
            SELECT e.id, e.text, e.metadata, v.distance
            FROM {self.table_name} AS e
            JOIN {scan_fn}('{self.table_name}', 'embedding', ?, ?) AS v
                ON e.id = v.rowid
        """, (query_blob, k))

        results = []
        for row in cursor:
            id_, text, meta_json, distance = row
            # Convert distance to similarity score for cosine
            if self.metric == "COSINE":
                score = 1.0 - distance  # cosine distance to similarity
            else:
                score = -distance  # Lower distance = higher similarity

            if threshold is not None and score < threshold:
                continue

            results.append(SearchResult(
                id=str(id_),
                text=text,
                score=score,
                metadata=json.loads(meta_json) if meta_json else {}
            ))
        return results

    def quantize(self, max_memory: str = "30MB") -> int:
        """Quantize vectors for faster approximate search."""
        cursor = self.conn.execute(f"""
            SELECT vector_quantize('{self.table_name}', 'embedding', 'max_memory={max_memory}')
        """)
        count = cursor.fetchone()[0]
        self._quantized = True
        return count

    def preload_quantization(self) -> None:
        """Load quantized data into memory for 4-5x speedup."""
        self.conn.execute(f"""
            SELECT vector_quantize_preload('{self.table_name}', 'embedding')
        """)

    def delete(self, ids: list[str]) -> None:
        """Delete embeddings by ID."""
        placeholders = ",".join("?" * len(ids))
        self.conn.execute(
            f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})",
            [int(id_) for id_ in ids]
        )
        self.conn.commit()
        self._quantized = False  # Invalidate quantization

    def __len__(self) -> int:
        cursor = self.conn.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        return cursor.fetchone()[0]

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "VectorStore":
        return self

    def __exit__(self, *args) -> None:
        self.close()


@dataclass
class SearchResult:
    """Result from vector similarity search."""
    id: str
    text: str
    score: float
    metadata: dict
```

### Database Schema

The VectorStore creates a standard SQLite table with BLOB storage for vectors:

```sql
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,  -- Float32 array as bytes
    metadata TEXT             -- JSON
);

-- sqlite-vector creates internal structures via vector_init()
-- No explicit index needed - the extension handles it
```

### sqlite-vector API Reference

Key functions used by VectorStore:

| Function | Description |
|----------|-------------|
| `vector_init(table, column, options)` | Initialize vector search for a table |
| `vector_full_scan(table, column, query, k)` | Brute-force k-NN search |
| `vector_quantize(table, column)` | Create quantized index for ANN search |
| `vector_quantize_scan(table, column, query, k)` | Fast approximate k-NN |
| `vector_quantize_preload(table, column)` | Load quantized data to memory |

Options for `vector_init`:
- `dimension`: Vector dimensionality (required)
- `type`: FLOAT32, FLOAT16, BFLOAT16, INT8, UINT8
- `distance`: L2, SQUARED_L2, COSINE, DOT, L1

## Embedding Models

### Recommended Models (GGUF)

| Model | Dimension | Size | Notes |
|-------|-----------|------|-------|
| [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | 384 | ~130MB | Good quality/size balance |
| [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | 768 | ~440MB | Higher quality |
| [snowflake-arctic-embed-s](https://huggingface.co/Snowflake/snowflake-arctic-embed-s) | 384 | ~130MB | Fast, accurate |
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 384 | ~90MB | Lightweight |
| [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | 768 | ~550MB | Long context (8192) |

### Model Conversion

Convert HuggingFace models to GGUF:

```bash
# Clone model
git clone https://huggingface.co/BAAI/bge-small-en-v1.5

# Convert (from llama.cpp)
python convert_hf_to_gguf.py bge-small-en-v1.5/ --outfile bge-small-en-v1.5-f16.gguf

# Quantize (optional)
./llama-quantize bge-small-en-v1.5-f16.gguf bge-small-en-v1.5-q8_0.gguf q8_0
```

## Document Loaders

Support common formats with minimal dependencies:

```python
class DocumentLoader:
    """Load documents from various formats."""

    @staticmethod
    def load(path: str) -> str:
        """Auto-detect format and load."""
        ext = Path(path).suffix.lower()

        if ext == ".txt":
            return Path(path).read_text()
        elif ext == ".md":
            return Path(path).read_text()
        elif ext == ".json":
            return json.loads(Path(path).read_text())
        elif ext == ".pdf":
            return DocumentLoader._load_pdf(path)
        elif ext in (".html", ".htm"):
            return DocumentLoader._load_html(path)
        else:
            raise ValueError(f"Unsupported format: {ext}")

    @staticmethod
    def _load_pdf(path: str) -> str:
        """Load PDF (requires pypdf)."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            return "\n".join(page.extract_text() for page in reader.pages)
        except ImportError:
            raise ImportError("PDF support requires: pip install pypdf")

    @staticmethod
    def _load_html(path: str) -> str:
        """Load HTML (basic text extraction)."""
        import re
        html = Path(path).read_text()
        # Simple tag stripping
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
```

## File Structure

```
src/cyllama/rag/
    __init__.py          # Public API exports
    embedder.py          # Embedder class
    store.py             # VectorStore class (sqlite-vector backend)
    splitter.py          # TextSplitter class
    pipeline.py          # RAGPipeline class
    rag.py               # High-level RAG class
    loaders.py           # Document loaders
    types.py             # Data classes (SearchResult, RAGResponse, etc.)
    vector.dylib         # macOS sqlite-vector extension (built)
    vector.so            # Linux sqlite-vector extension (built)
    vector.dll           # Windows sqlite-vector extension (built)
```

## Integration with Existing cyllama

### With LLM Class

```python
from cyllama import LLM
from cyllama.rag import RAG

# RAG uses LLM internally
rag = RAG(
    embedding_model="models/bge-small.gguf",
    generation_model="models/llama.gguf",
    temperature=0.7,  # Passed to LLM
    max_tokens=512
)
```

### With Agents

```python
from cyllama.agents import ReActAgent, tool
from cyllama.rag import RAG

rag = RAG(...)

@tool
def search_knowledge(query: str) -> str:
    """Search the knowledge base for relevant information."""
    response = rag.query(query)
    return "\n".join(r.text for r in response.sources)

agent = ReActAgent(llm=llm, tools=[search_knowledge])
```

### With Async API

```python
from cyllama.rag import AsyncRAG

async def main():
    rag = AsyncRAG(
        embedding_model="models/bge-small.gguf",
        generation_model="models/llama.gguf"
    )

    await rag.add_documents(["doc.txt"])

    response = await rag.query("What is X?")
    print(response.text)
```

## Performance Considerations

### Embedding Batching

Process multiple texts in batches to maximize GPU utilization:

```python
def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed texts in batches for efficiency."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Process batch...
        embeddings.extend(batch_embeddings)
    return embeddings
```

### Vector Search Optimization with sqlite-vector

sqlite-vector provides two search modes:

1. **Full Scan** (`vector_full_scan`) - Brute-force exact search
   - Best for datasets <100k vectors
   - No preprocessing required
   - SIMD-accelerated for high performance

2. **Quantized Scan** (`vector_quantize_scan`) - Approximate search
   - Best for datasets >100k vectors
   - Requires calling `quantize()` after bulk inserts
   - >0.95 recall with 4-5x speedup
   - Uses ~30MB RAM by default (configurable)

**Performance Characteristics:**
- 1M vectors, 768 dimensions: few milliseconds query time
- Memory footprint: 30-50MB regardless of dataset size
- No preindexing wait time - works immediately

### Memory Management

- Vectors are stored in SQLite (file or in-memory)
- sqlite-vector handles memory efficiently via quantization
- Use `db_path=":memory:"` for fastest access, file path for persistence
- Generator LLM can be shared across queries
- Use context managers for proper cleanup:

```python
with VectorStore(dimension=384, db_path="vectors.db") as store:
    store.add(embeddings, texts)
    results = store.search(query)
# Connection automatically closed
```

## Implementation Phases

### Phase 1: Core Embedding API (COMPLETED)
- [x] `Embedder` class with llama.cpp integration
- [x] Basic `embed()` and `embed_batch()` methods
- [x] Pooling strategies (mean, cls, last, none)
- [x] L2 normalization
- [x] Unit tests (22 tests in `tests/test_rag_embedder.py`)

### Phase 2: Vector Store with sqlite-vector (COMPLETED)
- [x] `VectorStore` class using sqlite-vector extension
- [x] Extension loading and error handling
- [x] Add/delete/search operations with BLOB encoding
- [x] Similarity search (cosine, L2, dot, L1)
- [x] Quantization support for large datasets
- [x] Context manager for proper resource cleanup
- [x] Unit tests (49 tests in `tests/test_rag_store.py`)

### Phase 3: Text Processing (COMPLETED)
- [x] `TextSplitter` class with recursive character splitting
- [x] `TokenTextSplitter` for token-based splitting
- [x] `MarkdownSplitter` for markdown-aware splitting
- [x] Document loaders (txt, md, json, jsonl)
- [x] `DirectoryLoader` for batch file loading
- [x] Optional PDF support via docling (`pdf` dependency group)
- [x] Unit tests (36 splitter tests + 36 loader tests in `tests/test_rag_splitter.py` and `tests/test_rag_loaders.py`)

### Phase 4: RAG Pipeline (COMPLETED)
- [x] `RAGConfig` dataclass with validation
- [x] `RAGResponse` dataclass with serialization
- [x] `RAGPipeline` class with query, stream, retrieve methods
- [x] Customizable prompt templates
- [x] High-level `RAG` class with sensible defaults
- [x] Document loading and chunking integration
- [x] Unit tests (25 tests in `tests/test_rag_pipeline.py`)

### Phase 5: Advanced Features (COMPLETED)
- [x] `AsyncRAG` class - Async wrapper for non-blocking RAG operations
- [x] `create_rag_tool()` - Create agent tools from RAG instances
- [x] `Reranker` class - Cross-encoder reranking for improved quality
- [x] `HybridStore` class - Combined FTS5 + vector search with reciprocal rank fusion
- [x] `async_search_knowledge()` - Async helper for knowledge base search
- [x] Unit tests (37 tests in `tests/test_rag_advanced.py`)

## Testing Strategy

### Unit Tests

```python
def test_embedder_dimension():
    embedder = Embedder("models/bge-small.gguf")
    assert embedder.dimension == 384

def test_embedder_normalize():
    embedder = Embedder("models/bge-small.gguf", normalize=True)
    emb = embedder.embed("test")
    norm = math.sqrt(sum(x*x for x in emb))
    assert abs(norm - 1.0) < 1e-6

def test_vector_store_search():
    with VectorStore(dimension=3, metric="cosine") as store:
        store.add([[1, 0, 0], [0, 1, 0]], ["doc1", "doc2"])
        results = store.search([1, 0, 0], k=1)
        assert results[0].text == "doc1"

def test_vector_store_quantization():
    with VectorStore(dimension=384, db_path="test.db") as store:
        # Add many vectors
        embeddings = [[i * 0.001] * 384 for i in range(1000)]
        texts = [f"doc{i}" for i in range(1000)]
        store.add(embeddings, texts)

        # Quantize for faster search
        count = store.quantize()
        assert count == 1000

        # Search with quantized index
        results = store.search([0.5] * 384, k=5)
        assert len(results) == 5
```

### Integration Tests

```python
@pytest.mark.slow
def test_rag_end_to_end():
    rag = RAG(
        embedding_model="models/bge-small.gguf",
        generation_model="models/llama.gguf"
    )
    rag.add_texts(["Paris is the capital of France."])
    response = rag.query("What is the capital of France?")
    assert "Paris" in response.text
```

## References

### llama.cpp Embedding
- [llama.cpp Embedding Tutorial](https://github.com/ggml-org/llama.cpp/discussions/7712)
- [Building RAG Pipeline with llama.cpp](https://machinelearningmastery.com/building-a-rag-pipeline-with-llama-cpp-in-python/)
- [RAG with llama.cpp and external APIs](https://neuml.hashnode.dev/rag-with-llamacpp-and-external-api-services)
- [llama-cpp-agent RAG](https://llama-cpp-agent.readthedocs.io/en/latest/rag/)
- [LangChain LlamaCppEmbeddings](https://python.langchain.com/docs/integrations/text_embedding/llamacpp/)
- [Snowflake Arctic Embed](https://huggingface.co/Snowflake/snowflake-arctic-embed-s)

### sqlite-vector
- [sqlite-vector GitHub](https://github.com/sqliteai/sqlite-vector) - Source and documentation
- [sqlite-vector API Reference](https://github.com/sqliteai/sqlite-vector/blob/main/API.md) - Full API docs
- [sqlite-vector Quantization](https://github.com/sqliteai/sqlite-vector/blob/main/QUANTIZATION.md) - Details on ANN search
- Local extension: `src/cyllama/rag/vector` (NEON on ARM, AVX2/SSE2 on x86)
