# TODO

## High Priority

(no high-priority items currently open)

## Medium Priority

- [ ] Performance benchmarking suite (token generation speed, memory profiling, regression detection)
- [ ] Response caching for identical prompts (decorator-based with TTL)
- [ ] Structured logging system (JSON output option, agent decision flow logging)
- [ ] Migrate the 5 `SDContext`-creating tests in `tests/test_sd.py` to the new `sd_ctx_factory` fixture in `tests/conftest.py` so the cleanup pattern (`del ctx; gc.collect()`) is centralized instead of duplicated inline. Affected tests: `TestSDContextIntegration::test_context_creation`, `TestSDContextIntegration::test_generate_image`, `TestSDContextConcurrencyGuard::test_concurrent_generate_raises`, `TestSDContextConcurrencyGuard::test_concurrent_generate_with_params_raises`, `TestSDContextConcurrencyGuard::test_lock_release_allows_subsequent_acquire`. Rationale and the 5-cycle crash reproducer are in `docs/dev/test-cleanup.md`

## Wheel / Packaging

- [ ] stable-diffusion.cpp uses compile-time `#ifdef SD_USE_CUDA` for backend selection instead of dynamic `ggml_backend_load_all()` like llama.cpp and whisper.cpp — propose dynamic backend discovery upstream or patch locally for consistency

- [ ] Investigate using versioned dylibs (e.g. `libllama.4.dylib`) instead of `.0.dylib` in dynamic wheels

## Wrapper API Surface

- [ ] Expose missing `WhisperFullParams` fields as Python properties — only ~15 of ~40+ fields are accessible. Notable gaps: `initial_prompt` (commonly used to guide transcription with domain vocabulary), `detect_language`, `suppress_regex`, `no_speech_thold`, `temperature_inc` (fallback temperature increment), `greedy.best_of`, `beam_search.beam_size`, `beam_search.patience`, `audio_ctx`, `max_len`, `max_tokens` (per-segment), `prompt_tokens`/`prompt_n_tokens`, `vad`/`vad_model_path`/`vad_params`. All fields retain correct C defaults via `whisper_full_default_params()` but cannot be changed from Python. Identified during the v0.2.8 defaults audit

- [ ] Expose missing `WhisperContextParams` fields — `dtw_aheads_preset`, `dtw_n_top`, `dtw_aheads`, `dtw_mem_size` (DTW timestamp alignment). Low priority unless users request word-level timestamps

- [ ] Expose missing `LlamaContextParams` fields — `op_offload`, `swa_full`, `kv_unified`, `embeddings` (settable at creation time, not just post-creation via `set_embeddings_mode()`). All retain correct C defaults but cannot be overridden from Python. Identified during the v0.2.8 defaults audit

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

- [x] Audit SD wrapper `__init__` defaults against C `sd_*_init()` defaults — found and fixed 4 mismatches: `wtype` (F16 vs COUNT), `eta` (0.0 vs INFINITY), `sample_method` (EULER_A vs SAMPLE_METHOD_COUNT), `scheduler` (DISCRETE vs SCHEDULER_COUNT). Added COUNT sentinel values to `SDType`, `SampleMethod`, `Scheduler`, and `Prediction` enums. All defaults now match the C library exactly
- [x] RAG response repetition — Qwen3-4B paragraph-loop bug fixed and pinned by regression test against the actual model. Two opt-in fixes: streaming-level n-gram repetition detector (`RAGConfig.repetition_threshold`) and chat-template prompting path (`RAGConfig.use_chat_template`). CLI enables the detector by default
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
- [x] Concurrent-use runtime guard on `LLM` / `Embedder` / `WhisperContext` / `SDContext` — non-blocking `threading.Lock` around every native-touching public method, raises `RuntimeError` on actual two-thread contention while allowing sequential cross-thread handoff (asyncio.to_thread, ThreadPoolExecutor). 17 regression tests across `TestLLMConcurrencyGuard` (5), `TestEmbedderConcurrencyGuard` (6), `TestSDContextConcurrencyGuard` (3), `TestWhisperContextConcurrencyGuard` (3), all passing end-to-end with the standard project model fixtures. Maintainer rationale in `docs/dev/runtime-guard.md`; user-facing companion in `docs/threading.md`

