# TODO

## High Priority

- [x] Enable CI/CD automation (uncomment and fix `.github/workflows/ci.yml` for push/PR triggers)
- [x] Add code coverage reporting to CI (`pytest-cov` with minimum threshold)
- [x] Add mypy type checking to CI
- [x] Async API support (`async def complete_async()`, `AsyncLLM` class, async agent execution)

## Medium Priority

- [ ] Built-in prompt template system (Llama, Mistral, ChatML formats)
- [ ] Response caching for identical prompts (decorator-based with TTL)
- [ ] Populate `docs/book/` with structured documentation (quickstart, API reference, agents guide, troubleshooting)
- [ ] Improve test fixtures in `conftest.py` (LLM instance with cleanup, pre-configured agents)
- [ ] Structured logging system (JSON output option, agent decision flow logging)

## Lower Priority

- [ ] Performance benchmarking suite (token generation speed, memory profiling, regression detection)
- [ ] Enhanced error context (custom exception classes with context dict)
- [ ] RAG utilities
- [ ] Web UI for testing
- [ ] Document server implementations (PythonServer, EmbeddedServer, LlamaServer usage)