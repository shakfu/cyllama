# TODO

## High Priority

- [ ] Thread safety audit (concurrent-access stress tests for shared C++ objects after GIL release)
- [ ] RAG response repetition — models (e.g. Qwen3-4B) repeat/paraphrase their answer in a loop when not given explicit system instructions. Repetition penalty (default 1.1) helps but doesn't fully prevent it. Current workaround: lower `max_tokens` and/or use `-s` system instruction. Next steps: n-gram repetition detection at the streaming level, or chat-template-based prompting instead of raw completion

## Medium Priority

- [ ] Performance benchmarking suite (token generation speed, memory profiling, regression detection)
- [ ] Response caching for identical prompts (decorator-based with TTL)
- [ ] Structured logging system (JSON output option, agent decision flow logging)

## Wheel / Packaging

- [ ] stable-diffusion.cpp uses compile-time `#ifdef SD_USE_CUDA` for backend selection instead of dynamic `ggml_backend_load_all()` like llama.cpp and whisper.cpp — propose dynamic backend discovery upstream or patch locally for consistency
- [ ] Investigate using versioned dylibs (e.g. `libllama.4.dylib`) instead of `.0.dylib` in dynamic wheels

## Lower Priority

- [ ] Web UI for testing

## RAG Scaling (see docs/dev/scaling_rag.md)

### Phase 2: Quick Wins

- [ ] Persistent quantization state in database metadata (quantize() exists but state is in-memory only)

### Phase 3: Async/Parallel

- [ ] Async embedding generation (`embed_batch_async()`)
- [ ] Parallel document loading in DirectoryLoader
- [ ] Batch query processing in RAG pipeline

### Phase 4: Advanced

- [ ] Metadata pre-filtering in vector search (filter by source, date, etc.)
- [ ] Sharding for 1M+ vector workloads

## Completed

- [x] Memory leak tests (loop create/destroy of LLM, SDContext, WhisperContext objects, assert RSS stays bounded)
- [x] Error message audit (bad model path, corrupt GGUF, OOM context -- clear errors, not segfaults or raw C++ assertions)
- [x] Wheel smoke tests in CI
- [x] Signal/interrupt handling (Ctrl-C safe)
- [x] PDF loader tests (PDFLoader + TestPDFLoader)
- [x] TokenTextSplitter with llama.cpp tokenizer integration (rag/splitter.py)
- [x] Document server implementations (docs/server_usage_examples.md)
- [x] Enhanced error context (ActionParseError, VectorStoreError, LoaderError with context dicts)
- [x] Auto-quantization after bulk inserts (VectorStore.quantize())
- [x] Reranking support (Reranker class in rag/advanced.py)

