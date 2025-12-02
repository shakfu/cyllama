# TODO

## High Priority

- [ ] Response caching for identical prompts (decorator-based with TTL)
- [ ] Structured logging system (JSON output option, agent decision flow logging)

## Medium Priority

- [ ] Performance benchmarking suite (token generation speed, memory profiling, regression detection)
- [ ] Enhanced error context (custom exception classes with context dict)
- [ ] Document server implementations (PythonServer, EmbeddedServer, LlamaServer usage)

## Lower Priority

- [ ] Web UI for testing
- [ ] Add PDF loader tests with sample PDF files
- [ ] TokenTextSplitter with llama.cpp tokenizer integration (use model's tokenizer for accurate token counts)

## RAG Scaling (see docs/dev/scaling_rag.md)

### Phase 2: Quick Wins

- [ ] Embedding cache for repeated queries (`@lru_cache` on `Embedder.embed()`)
- [ ] Auto-quantization after bulk inserts (threshold-based)
- [ ] Persistent quantization state in database metadata

### Phase 3: Async/Parallel

- [ ] Async embedding generation (`embed_batch_async()`)
- [ ] Parallel document loading in DirectoryLoader
- [ ] Batch query processing in RAG pipeline

### Phase 4: Advanced

- [ ] Metadata pre-filtering in vector search (filter by source, date, etc.)
- [ ] Reranking support (cross-encoder for improved precision)
- [ ] Sharding for 1M+ vector workloads

## Completed

- [x] Enable CI/CD automation
- [x] Add code coverage reporting to CI
- [x] Add mypy type checking to CI
- [x] Async API support
- [x] Built-in prompt template system
- [x] Populate `docs/book/` with structured documentation
- [x] Improve test fixtures in `conftest.py`
- [x] Response class for complete(), chat(), LLM(), batch_generate()
- [x] RAG Support (Embedder, VectorStore, RAG pipeline, HybridStore, document loaders, text splitters)
- [x] RAG documentation in docs/book/
- [x] RAG examples in tests/examples/
- [x] sqlite-vec integration with quantization support (`quantize()`, `preload_quantization()`)
