# TODO

## Bugs

- [x] `--stats` silently skipped in streaming mode (`cyllama generate --stream --stats`) -- the streaming path doesn't return a `Response` object so `response` is `None` and the stats table is never printed

## Medium Priority

- [x] Expose `LlamaContext.get_perf_data()` -- the `llama_perf_context_data` struct is declared in `llama.pxd` but the method is commented out in `llama_cpp.pyx` (line 2670). Would provide C-level prompt eval time and generation time, more accurate than wall-clock timing for `--stats`
- [x] Expose `MtmdContextParams.warmup` property -- the C struct field is declared in `mtmd.pxd` but has no Python getter/setter. Defaults to `true`, which adds latency during context creation that callers can't opt out of
- [x] Replace deprecated `mtmd_image_tokens_get_nx/ny` with `mtmd_image_tokens_get_decoder_pos` + `mtmd_decoder_pos` struct in `.pxd`/`.pxi`. Upstream wrapped the old functions in `DEPRECATED()` in b8802; they still link but will be removed
- [x] Performance benchmarking: token generation speed tracking (`GenerationStats`) and memory profiling (`test_memory_leaks.py`, `scripts/leak_check.py`)
- [ ] Performance regression detection -- CI-integrated baseline capture/comparison to catch speed or memory regressions across commits
- [ ] Structured logging system (JSON output option, agent decision flow logging)

## Wheel / Packaging

- [ ] stable-diffusion.cpp uses compile-time `#ifdef SD_USE_CUDA` for backend selection instead of dynamic `ggml_backend_load_all()` like llama.cpp and whisper.cpp -- propose dynamic backend discovery upstream or patch locally for consistency

## Explore

- [ ] MCP server (`cyllama/mcp/`) -- expose local inference capabilities (complete, chat, embed, transcribe, generate_image) as MCP tools, model listing as resources. Would let MCP clients (Claude Code, Claude Desktop, etc.) use local GGUF models. Thin wrapper over existing high-level API, `mcp` SDK as optional dependency. Protocol is still evolving -- evaluate when a concrete client integration need arises

## RAG Scaling (see docs/dev/scaling_rag.md)

- [ ] Persistent quantization state in database metadata (quantize() exists but state is in-memory only)
- [ ] Metadata pre-filtering in vector search (filter by source, date, etc.)
- [ ] Async embedding generation (`embed_batch_async()`)
- [ ] Parallel document loading in DirectoryLoader
- [ ] Batch query processing in RAG pipeline
- [ ] Sharding for 1M+ vector workloads