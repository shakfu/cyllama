# TODO

## High Priority

- [ ] Response caching for identical prompts (decorator-based with TTL)
- [ ] Structured logging system (JSON output option, agent decision flow logging)

## Code Quality (from code review 2026-03-30)

### High

- [x] Validate `LLAMACPP_DYLIB_DIR` exists in CMakeLists.txt when `WITH_DYLIB=ON` — currently defaults to `thirdparty/llama.cpp/dynamic` without checking contents
- [x] Log warning for broken symlinks in `build_shared()` (`scripts/manage.py:1116-1123`) — missing targets are silently skipped, producing incomplete wheels
- [x] Validate `sed` package-rename commands in `build-gpu-wheels.yml` — if pattern doesn't match, sed succeeds silently and wheels ship with wrong package name
- [x] Validate `LD_LIBRARY_PATH` directory exists in `CIBW_REPAIR_WHEEL_COMMAND_LINUX` (`build-gpu-wheels.yml`)
- [x] Make CI Python path configurable instead of hardcoding `/opt/python/cp310-cp310/bin/python` in `build-gpu-wheels.yml`
- [x] Improve Windows `get_lib_path()` fallback (`scripts/manage.py:781-809`) — returns `Release/` path on failure regardless of actual config
- [x] Validate chat message dicts in `api.py:1020-1025` — `msg.get("role", "user")` silently accepts empty/malformed messages
- [x] Fix incomplete exception handling in `AsyncLLM.stream()` (`api.py:1588-1601`) — producer_task may not be properly awaited if consumer raises before producer starts
- [x] Make server sampler parameters configurable (`llama/server/python.py:113-115`) — seed 1337, temp 0.8, min_p 0.05 are hardcoded with no per-request override

### Medium

- [ ] Release GIL during `whisper_full()` (`whisper_cpp.pyx:508-510`) — blocks all Python threads during inference
- [ ] Release GIL during `llama_encode()` (`llama_cpp.pyx:2323`) — blocks all Python threads during encoding
- [ ] Log exceptions in sd callbacks instead of `except Exception: pass` (`stable_diffusion.pyx:200-218`)
- [ ] Add overflow-safe integer arithmetic for image size calculations (`stable_diffusion.pyx:575-588,787-793`)
- [ ] Make RPATH conditional on `WITH_WHISPER`/`WITH_STABLEDIFFUSION` (`CMakeLists.txt:577-592`) — currently includes whisper/sd lib paths unconditionally
- [ ] Validate Cython include paths exist before passing to `cython_transpile()` (`CMakeLists.txt:469-481`)
- [ ] Exclude `*.a` files from sdist — `wheel.exclude` removes them but sdist includes `thirdparty/*/lib` dirs
- [ ] Log when `GGML_*` environment variables override build defaults (`scripts/manage.py:97-120`)
- [ ] Guard `_release_url()` against `None` from `_release_asset_name()` (`scripts/manage.py:1137-1172`)
- [ ] Fail early on invalid chat template name (`api.py:1013-1025`) — currently silently treated as a Jinja template string
- [ ] Make TTS token IDs configurable (`llama/tts.py:385-394`) — IDs 198, 151672-155772 hardcoded for specific model
- [ ] Wrap `close()` in try-except in `LLM.__del__` (`api.py:569-572`)
- [ ] Add timeout support to `AsyncLLM.stream()` thread operations (`api.py:1574-1601`)
- [ ] Extract config dict building into `GenerationConfig.to_dict()` (`api.py:500-517`) — duplicated in `LLM.__init__` and `AsyncLLM._build_config()`
- [ ] Fix `metadata = [{}] * len(embeddings)` shared dict reference (`rag/store.py:226`) — use list comprehension instead
- [ ] Validate metadata values are JSON-serializable before `json.dumps()` (`rag/store.py:236`)
- [ ] Add type checking for loader metadata inclusion (`rag/loaders.py:350-353`) — all non-text keys included without validation
- [ ] Add `_check_closed()` to `VectorStore.__repr__()` (`rag/store.py`)
- [ ] Document that HybridStore FTS triggers require exclusive access (`rag/advanced.py:566-591`)
- [ ] Preserve exception type distinction in `DirectoryLoader` error handling (`rag/loaders.py:612-618`)

### Low

- [ ] Standardize UnicodeDecodeError handling between `api.py` and `batching.py`
- [ ] Consider returning `Response` from `Response.__add__()` instead of `str` (`api.py:263-275`)
- [ ] Audit stop buffer flush at EOS in streaming (`api.py:920-924`)
- [ ] Remove unused instance variables in `LlamaCLI` (`llama/cli.py:26-27`)
- [ ] Consider memory-aware LRU for embedding cache (`rag/embedder.py:29-74`)
- [ ] Monitor llama.cpp pooling reliability — manual pooling workaround may become unnecessary (`rag/embedder.py:151-157`)
- [ ] Validate `errors` parameter in `TextLoader.__init__()` (`rag/loaders.py:102-114`)
- [ ] Add concurrent VectorStore access tests
- [ ] Add symlink handling tests for DirectoryLoader
- [ ] Add server request validation tests with malformed JSON
- [ ] Non-deterministic CI artifact naming — `strategy.job-index` changes if matrix changes (`build-cibw.yaml:39`)

## Wheel / Packaging

- [ ] Investigate using versioned dylibs (e.g. `libllama.4.dylib`) instead of `.0.dylib` in dynamic wheels
- [ ] Fix absolute homebrew OpenSSL paths in wheel (`/opt/homebrew/opt/openssl@3/lib/libssl.3.dylib`) -- either bundle with `@rpath` or skip OpenSSL for redistributable wheels
- [ ] stable-diffusion.cpp uses compile-time `#ifdef SD_USE_CUDA` for backend selection instead of dynamic `ggml_backend_load_all()` like llama.cpp and whisper.cpp — propose dynamic backend discovery upstream or patch locally for consistency

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

- [x] Embedding cache for repeated queries (LRU cache on `Embedder.embed()` with `cache_size` parameter)
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
