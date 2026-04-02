# TODO

## High Priority

- [ ] Response caching for identical prompts (decorator-based with TTL)
- [ ] Structured logging system (JSON output option, agent decision flow logging)
- [x] Wheel smoke tests in CI (build wheel, install in clean venv, run minimal inference to catch packaging regressions)
- [ ] Memory leak tests (loop create/destroy of LLM, SDContext, WhisperContext objects, assert RSS stays bounded)
- [ ] Signal/interrupt handling (verify Ctrl-C during long inference doesn't segfault or leave resources dangling)

## Wheel / Packaging

- [ ] stable-diffusion.cpp uses compile-time `#ifdef SD_USE_CUDA` for backend selection instead of dynamic `ggml_backend_load_all()` like llama.cpp and whisper.cpp — propose dynamic backend discovery upstream or patch locally for consistency

## Medium Priority

- [ ] Performance benchmarking suite (token generation speed, memory profiling, regression detection)
- [ ] Thread safety audit (concurrent-access stress tests for shared C++ objects after GIL release)
- [ ] Error message audit (bad model path, corrupt GGUF, OOM context -- ensure clear errors, not segfaults or raw C++ assertions)
- [ ] Enhanced error context (custom exception classes with context dict)
- [ ] Document server implementations (PythonServer, EmbeddedServer, LlamaServer usage)

## Lower Priority

- [ ] Web UI for testing
- [ ] Add PDF loader tests with sample PDF files
- [ ] TokenTextSplitter with llama.cpp tokenizer integration (use model's tokenizer for accurate token counts)
- [ ] Investigate using versioned dylibs (e.g. `libllama.4.dylib`) instead of `.0.dylib` in dynamic wheels

## RAG Scaling (see docs/dev/scaling_rag.md)

### Phase 2: Quick Wins

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

