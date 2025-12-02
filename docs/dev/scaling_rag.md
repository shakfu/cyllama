# Scaling RAG in cyllama

This document analyzes the current RAG implementation's scalability characteristics and provides recommendations for incremental improvements to support larger workloads.

## Current Architecture Overview

### Components

| Component | Implementation | Scalability |
|-----------|---------------|-------------|
| Embedder | llama.cpp BERT models | Good (GPU-accelerated) |
| VectorStore | SQLite + sqlite-vec extension | Good (has quantization) |
| HybridStore | VectorStore + FTS5 | Good (both components scale) |
| RAG Pipeline | Sequential query processing | Limited (no batching) |
| Document Loaders | File-based, synchronous | Good (streaming available) |

### sqlite-vec Integration

cyllama uses the [sqlite-vec](https://github.com/asg017/sqlite-vec) extension for vector operations. This provides:

- **`vector_init()`** - Initialize vector search on a table/column
- **`vector_full_scan()`** - Brute-force exact search
- **`vector_quantize()`** - Build quantized index for approximate search
- **`vector_quantize_scan()`** - Fast approximate search using quantization
- **`vector_quantize_preload()`** - Preload quantized data into memory

### Data Flow

```
Documents -> TextSplitter -> Chunks -> Embedder -> Vectors -> VectorStore
                                                                    |
                                                        [sqlite-vec extension]
                                                                    |
Query -> Embedder -> Query Vector ---> vector_full_scan() or vector_quantize_scan()
                                                                    |
                                                                    v
                                                              Top-K Results
```

## Scalability Analysis

### 1. Vector Search

**Current Implementation** (`store.py`):
```python
def search(self, query_embedding, k=5, threshold=None):
    query_blob = self._encode_vector(query_embedding)

    # Use quantized search if available, otherwise full scan
    scan_fn = "vector_quantize_scan" if self._quantized else "vector_full_scan"

    cursor = self.conn.execute(f"""
        SELECT e.id, e.text, e.metadata, v.distance
        FROM {self.table_name} AS e
        JOIN {scan_fn}('{self.table_name}', 'embedding', ?, ?) AS v
            ON e.id = v.rowid
    """, (query_blob, k))
```

**Two Search Modes**:

| Mode | Function | Complexity | Use Case |
|------|----------|------------|----------|
| Full Scan | `vector_full_scan()` | O(n) | Small datasets, exact results |
| Quantized | `vector_quantize_scan()` | O(log n) | Large datasets, approximate |

**Quantization API**:
```python
# After bulk inserts, quantize for faster search
store.quantize(max_memory="30MB")  # Build quantized index
store.preload_quantization()        # Load into memory for 4-5x speedup
```

**Benchmark Estimates** (384-dim embeddings):

| Vectors | Full Scan | Quantized Scan | Notes |
|---------|-----------|----------------|-------|
| 1,000 | ~5-10 ms | ~2-5 ms | Quantization optional |
| 10,000 | ~50-100 ms | ~5-15 ms | Quantization recommended |
| 100,000 | ~500-1000 ms | ~10-30 ms | Quantization required |
| 1,000,000 | ~5-10 sec | ~30-100 ms | Quantization essential |

**sqlite-vec Quantization Details**:
- Uses scalar quantization (float32 -> int8)
- Maintains >0.95 recall for typical workloads
- Memory-mapped for efficient I/O
- Configurable memory budget

### 2. Embedding Generation

**Current Implementation** (`embedder.py`):
```python
def embed_batch(self, texts: list[str], ...) -> list[list[float]]:
    return [self.embed(text, ...) for text in texts]
```

**Issues**:
- Sequential processing (no true batching at llama.cpp level)
- No async support for I/O-bound scenarios
- GPU utilization may be suboptimal for small texts

**Realistic Throughput** (GTE-small on M1 Mac):
- ~50-100 embeddings/second for short texts
- 10,000 documents with 5 chunks each = 50,000 embeddings
- Indexing time: ~8-15 minutes

### 3. HybridStore FTS5

**Current Implementation** (`advanced.py`):
```python
def _fts_search(self, query: str, k: int = 10) -> list[SearchResult]:
    cursor = self._vector_store.conn.execute(f"""
        SELECT ... FROM {fts_table}
        WHERE {fts_table} MATCH ? ORDER BY bm25(...) LIMIT ?
    """, (escaped_query, k))
```

**Scalability**: Excellent - FTS5 uses inverted indices with O(log n) lookup

**Hybrid Search**: Combines vector similarity with keyword matching using Reciprocal Rank Fusion (RRF). Both components scale well when quantization is enabled.

### 4. RAG Pipeline

**Current Implementation** (`pipeline.py`):
- Single query at a time
- No result caching
- Synchronous LLM calls

**Issues**:
- No query batching
- Repeated identical queries hit full search path
- No streaming for intermediate results

### 5. Document Loading

**Current Implementation**: Generally good
- `JSONLLoader.load_lazy()` provides streaming
- `DirectoryLoader` processes files sequentially

**Issues**:
- No parallel file loading
- Large files loaded entirely into memory before chunking

## Scaling Recommendations

### Phase 1: Leverage Existing Quantization (No Code Changes)

The most important optimization is already implemented - **use quantization**:

```python
# After bulk inserts
store = VectorStore(dimension=384, db_path="vectors.db")
store.add(embeddings, texts)

# Enable quantization for large datasets
if len(store) > 5000:
    store.quantize(max_memory="50MB")
    store.preload_quantization()

# Searches now use vector_quantize_scan automatically
results = store.search(query_embedding, k=10)
```

**Guidelines**:
- < 5,000 vectors: Full scan is fine
- 5,000 - 50,000 vectors: Quantize with 30-50MB memory
- 50,000+ vectors: Quantize with 100MB+ memory

### Phase 2: Quick Wins (Minor Code Changes)

#### 2.1 Embedding Cache

Cache computed embeddings for repeated queries:

```python
from functools import lru_cache

class Embedder:
    @lru_cache(maxsize=1000)
    def embed_cached(self, text: str) -> tuple[float, ...]:
        return tuple(self.embed(text))
```

**Expected Improvement**: Near-instant for repeated queries

#### 2.2 Auto-Quantization

Automatically quantize after bulk inserts:

```python
def add(self, embeddings, texts, metadata=None, auto_quantize_threshold=10000):
    ids = self._add_impl(embeddings, texts, metadata)

    if len(self) >= auto_quantize_threshold and not self._quantized:
        self.quantize()
        self.preload_quantization()

    return ids
```

#### 2.3 Persistent Quantization State

Track quantization state in database metadata:

```python
def _init_table(self):
    # ... existing table creation ...

    # Store metadata including quantization state
    self.conn.execute("""
        CREATE TABLE IF NOT EXISTS _vector_metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

def quantize(self, max_memory="30MB"):
    count = self._quantize_impl(max_memory)
    self.conn.execute(
        "INSERT OR REPLACE INTO _vector_metadata VALUES ('quantized', 'true')"
    )
    self.conn.commit()
    return count
```

### Phase 3: Async and Parallel Processing

#### 3.1 Async Embedding

```python
class AsyncEmbedder:
    async def embed_batch_async(
        self,
        texts: list[str],
        concurrency: int = 4
    ) -> list[list[float]]:
        semaphore = asyncio.Semaphore(concurrency)

        async def embed_one(text):
            async with semaphore:
                return await asyncio.to_thread(self.embed, text)

        return await asyncio.gather(*[embed_one(t) for t in texts])
```

#### 3.2 Parallel Document Loading

```python
from concurrent.futures import ThreadPoolExecutor

class DirectoryLoader:
    def load_parallel(self, path: str, max_workers: int = 4) -> list[Document]:
        files = list(Path(path).glob(self.glob))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            docs_lists = list(executor.map(self._load_single, files))

        return [doc for docs in docs_lists for doc in docs]
```

#### 3.3 Batch Query Processing

```python
class RAG:
    def query_batch(
        self,
        queries: list[str],
        top_k: int = 5
    ) -> list[RAGResponse]:
        # Embed all queries at once
        query_embeddings = self.embedder.embed_batch(queries)

        # Search for all (could parallelize)
        all_results = [
            self.store.search(emb, k=top_k)
            for emb in query_embeddings
        ]

        # Generate responses
        return [
            self._generate_response(q, results)
            for q, results in zip(queries, all_results)
        ]
```

### Phase 4: Advanced Features

#### 4.1 Metadata Pre-filtering

Filter candidates before vector search to reduce search space:

```python
def search(
    self,
    query: list[float],
    k: int = 5,
    filter: dict = None,  # {"source": "docs/*.md", "date_gte": "2024-01-01"}
) -> list[SearchResult]:
    if filter:
        # Build WHERE clause from filter
        where_clause, params = self._build_filter(filter)
        # Apply filter before vector search
        cursor = self.conn.execute(f"""
            SELECT e.id, e.text, e.metadata, v.distance
            FROM {self.table_name} AS e
            JOIN {scan_fn}('{self.table_name}', 'embedding', ?, ?) AS v
                ON e.id = v.rowid
            WHERE {where_clause}
        """, (query_blob, k * 2) + params)  # Over-fetch then filter
```

#### 4.2 Sharding for Very Large Stores

For 1M+ vectors, partition across multiple stores:

```python
class ShardedVectorStore:
    def __init__(self, num_shards: int = 4, **kwargs):
        self.shards = [
            VectorStore(db_path=f"shard_{i}.db", **kwargs)
            for i in range(num_shards)
        ]

    def add(self, embeddings, texts, ...):
        # Hash-based sharding
        for i, (emb, text) in enumerate(zip(embeddings, texts)):
            shard_id = hash(text) % len(self.shards)
            self.shards[shard_id].add([emb], [text], ...)

    def search(self, query, k=5, ...):
        # Search all shards in parallel, merge results
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor() as executor:
            shard_results = list(executor.map(
                lambda s: s.search(query, k=k),
                self.shards
            ))

        # Merge and re-rank
        all_results = [r for results in shard_results for r in results]
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:k]
```

#### 4.3 Incremental Quantization

Re-quantize periodically after incremental updates:

```python
class VectorStore:
    def __init__(self, ..., requantize_threshold=1000):
        self._adds_since_quantize = 0
        self._requantize_threshold = requantize_threshold

    def add(self, embeddings, texts, ...):
        ids = self._add_impl(embeddings, texts)
        self._adds_since_quantize += len(ids)

        if self._quantized and self._adds_since_quantize >= self._requantize_threshold:
            self.quantize()  # Rebuild quantized index
            self._adds_since_quantize = 0

        return ids
```

## Implementation Priority

| Phase | Feature | Effort | Impact | Priority |
|-------|---------|--------|--------|----------|
| 1 | Use existing quantization | None | High | Critical |
| 2.1 | Embedding cache | Low | Medium | High |
| 2.2 | Auto-quantization | Low | Medium | High |
| 3.1 | Async embedding | Medium | Medium | Medium |
| 3.2 | Parallel loading | Low | Low | Low |
| 4.1 | Metadata filtering | Medium | Medium | Medium |
| 4.2 | Sharding | High | High | Low (only for 1M+) |

## Recommended Scaling Path

### For 100s of Documents (Current)
- Current implementation is sufficient
- No quantization needed
- Full scan search is fast enough

### For 1,000s of Documents
- Enable quantization after bulk insert
- Add embedding cache for repeated queries
- Expected: <20ms search

### For 10,000s of Documents
- Quantization required
- Preload quantized data for best performance
- Consider auto-quantization threshold
- Expected: <30ms search

### For 100,000+ Documents
- Quantization with larger memory budget (100MB+)
- Add metadata pre-filtering to reduce candidate set
- Implement async embedding for ingestion
- Expected: <50ms search with filtering

### For 1,000,000+ Documents
- Consider sharding across multiple database files
- Use metadata filtering aggressively
- Periodic re-quantization for optimal performance
- Expected: <100ms search with sharding

## sqlite-vec Configuration

### Memory Budget for Quantization

The `max_memory` parameter controls the quantization index size:

| Dataset Size | Recommended Memory | Notes |
|--------------|-------------------|-------|
| 10k vectors | 30MB | Default, good for most cases |
| 50k vectors | 50-100MB | Larger index improves recall |
| 100k vectors | 100-200MB | Balance memory vs performance |
| 500k+ vectors | 500MB+ | May need chunked quantization |

### Distance Metrics

sqlite-vec supports multiple distance metrics:

```python
store = VectorStore(
    dimension=384,
    metric="cosine",  # Default, normalized similarity
    # metric="l2",    # Euclidean distance
    # metric="dot",   # Dot product (unnormalized)
)
```

**Recommendations**:
- `cosine`: Best for normalized embeddings (most embedding models)
- `l2`: When absolute distances matter
- `dot`: For maximum inner product search

### Vector Types

sqlite-vec supports quantized storage types:

```python
store = VectorStore(
    dimension=384,
    vector_type="float32",  # Default, full precision
    # vector_type="float16",  # Half precision, 2x storage savings
    # vector_type="int8",     # 4x storage savings, slight quality loss
)
```

**Trade-offs**:
- `float32`: Full precision, largest storage
- `float16`: 2x smaller, minimal quality loss
- `int8`: 4x smaller, ~1-2% recall reduction

## Benchmarking

Establish baselines before optimizing:

```python
import time
from cyllama.rag import Embedder, VectorStore

def benchmark_search(store, query_embedding, n_queries=100):
    times = []
    for _ in range(n_queries):
        start = time.perf_counter()
        store.search(query_embedding, k=10)
        times.append(time.perf_counter() - start)

    return {
        "mean_ms": sum(times) / len(times) * 1000,
        "p50_ms": sorted(times)[len(times)//2] * 1000,
        "p99_ms": sorted(times)[int(len(times)*0.99)] * 1000,
        "quantized": store.is_quantized,
    }

def benchmark_quantization(store):
    start = time.perf_counter()
    count = store.quantize(max_memory="50MB")
    quantize_time = time.perf_counter() - start

    start = time.perf_counter()
    store.preload_quantization()
    preload_time = time.perf_counter() - start

    return {
        "vectors_quantized": count,
        "quantize_seconds": quantize_time,
        "preload_seconds": preload_time,
    }

# Example usage
embedder = Embedder("model.gguf")
with VectorStore(dimension=384, db_path="test.db") as store:
    # Add test data
    # ...

    # Benchmark without quantization
    query_emb = embedder.embed("test query")
    before = benchmark_search(store, query_emb)
    print(f"Before quantization: {before['mean_ms']:.1f}ms mean")

    # Quantize and benchmark
    quant_stats = benchmark_quantization(store)
    print(f"Quantization took: {quant_stats['quantize_seconds']:.1f}s")

    after = benchmark_search(store, query_emb)
    print(f"After quantization: {after['mean_ms']:.1f}ms mean")
    print(f"Speedup: {before['mean_ms'] / after['mean_ms']:.1f}x")
```

## Conclusion

The cyllama RAG implementation leverages `sqlite-vec` for efficient vector operations, providing a solid foundation for scaling:

**Already Available**:
- Quantized approximate search via `vector_quantize_scan()`
- Configurable memory budgets
- Multiple distance metrics
- Storage-efficient vector types

**Key Insight**: The most important optimization is **enabling quantization** for datasets >5,000 vectors. This is already implemented and provides 4-5x speedup with >95% recall.

**Recommended Approach**:
1. **Use quantization** - It's built-in and highly effective
2. **Add caching** for repeated queries
3. **Implement async processing** for large ingestion jobs
4. **Consider sharding** only for 1M+ vector workloads

The architecture is well-suited for production workloads up to hundreds of thousands of documents when quantization is properly utilized.
