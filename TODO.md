# TODO

## Medium Priority

- [ ] Performance benchmarking suite (token generation speed, memory profiling, regression detection)
- [ ] Structured logging system (JSON output option, agent decision flow logging)

## Wheel / Packaging

- [ ] stable-diffusion.cpp uses compile-time `#ifdef SD_USE_CUDA` for backend selection instead of dynamic `ggml_backend_load_all()` like llama.cpp and whisper.cpp -- propose dynamic backend discovery upstream or patch locally for consistency
- [ ] Investigate using versioned dylibs (e.g. `libllama.4.dylib`) instead of `.0.dylib` in dynamic wheels

## RAG Scaling (see docs/dev/scaling_rag.md)

- [ ] Persistent quantization state in database metadata (quantize() exists but state is in-memory only)
- [ ] Metadata pre-filtering in vector search (filter by source, date, etc.)
- [ ] Async embedding generation (`embed_batch_async()`)
- [ ] Parallel document loading in DirectoryLoader
- [ ] Batch query processing in RAG pipeline
- [ ] Sharding for 1M+ vector workloads