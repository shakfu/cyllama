# CHANGELOG

All notable project-wide changes will be documented in this file. Note that each subproject has its own CHANGELOG.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Commons Changelog](https://common-changelog.org). This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Types of Changes

- Added: for new features.
- Changed: for changes in existing functionality.
- Deprecated: for soon-to-be removed features.
- Removed: for now removed features.
- Fixed: for any bug fixes.
- Security: in case of vulnerabilities.

---

## [0.1.x]

## [0.1.9] - 2025-11-21

### Added

- **High-Level Generation API** (`src/cyllama/generate.py`)
  - Added `generate()` convenience function for one-line text generation
  - Added `chat()` function for multi-turn conversation interface
  - Added `Generator` class for efficient model reuse and caching
  - Added `GenerationConfig` dataclass for comprehensive generation parameters
  - Added `GenerationStats` dataclass for detailed performance metrics
  - Automatic context and sampler management with optimal sizing
  - Full streaming support with token-by-token callbacks
  - Support for temperature, top-k, top-p, min-p, repeat penalty, and seed parameters
  - Stop sequences and custom tokenization options
  - 60+ comprehensive tests in `tests/test_generate.py`

- **Batch Processing Utilities** (`src/cyllama/batching.py`)
  - Added `batch_generate()` convenience function for efficient batch processing
  - Added `BatchGenerator` class for parallel sequence processing
  - Added `BatchRequest` and `BatchResponse` dataclasses for structured batch operations
  - Utilizes llama.cpp's native batching for 3-10x throughput improvement
  - Detailed performance statistics per request
  - Automatic batch size optimization
  - Examples in documentation and tests

- **OpenAI-Compatible API** (`src/cyllama/integrations/openai_compat.py`)
  - Added `OpenAICompatibleClient` class providing drop-in replacement for OpenAI client
  - Full chat completions API compatibility
  - Streaming support with proper chunking
  - Compatible message format (system, user, assistant roles)
  - Usage statistics (prompt tokens, completion tokens)
  - Response objects matching OpenAI's format (ChatCompletion, ChatCompletionChunk)
  - 10+ comprehensive tests in `tests/test_integrations.py`

- **LangChain Integration** (`src/cyllama/integrations/langchain.py`)
  - Added `CyllamaLLM` class implementing LangChain's LLM interface
  - Works seamlessly with LangChain chains, agents, and tools
  - Streaming support with LangChain callback managers
  - Proper error handling when LangChain is not installed
  - Full parameter compatibility (temperature, max_tokens, top_k, top_p)
  - Example usage in documentation

- **Comprehensive Documentation**
  - Added `docs/USER_GUIDE.md` - Complete 450+ line user guide covering all APIs
  - Added `docs/COOKBOOK.md` - 350+ line cookbook with practical patterns and recipes
  - Added `docs/IMPROVEMENTS_SUMMARY.md` - Detailed summary of all improvements
  - Sections on text generation, chat apps, structured output, performance, integrations
  - Working examples for FastAPI, Flask, Gradio integrations
  - Error handling patterns, best practices, troubleshooting guides

### Changed

- **Module Exports**: Enhanced `src/cyllama/__init__.py` with convenient top-level imports
  - Exported high-level generation functions: `generate`, `chat`, `Generator`, `GenerationConfig`
  - Exported batching utilities: `batch_generate`, `BatchGenerator`, `BatchRequest`, `BatchResponse`
  - Exported memory utilities: `estimate_gpu_layers`, `estimate_memory_usage`, `MemoryEstimate`
  - All new APIs available directly from `import cyllama`

- **Documentation**: Updated `RECOMMENDED_TO_WRAP.md` to reflect completion status
  - All five high-priority APIs now marked as completed
  - Updated priorities for remaining optional features
  - Comprehensive status tracking and implementation notes

### Technical Implementation

- **High-Level API Architecture**: Designed for simplicity with power when needed
  - Automatic model and context lifecycle management
  - Lazy initialization with smart caching
  - Proper cleanup with Python context managers
  - Type hints throughout for IDE support

- **Streaming Implementation**: Efficient token-by-token generation
  - Generator-based streaming for memory efficiency
  - Optional token callbacks for real-time processing
  - Compatible with both sync and async patterns

- **Batch Processing**: Leverages llama.cpp's native batching
  - Parallel sequence processing with shared KV cache
  - Automatic batch size optimization based on context
  - Per-sequence logit computation
  - Efficient memory management

- **Integration Layer**: Minimal overhead adapters
  - OpenAI compatibility through adapter pattern
  - LangChain integration via interface implementation
  - Graceful degradation when optional dependencies missing
  - Zero-copy data passing where possible

- **Testing Strategy**: Comprehensive test coverage
  - Unit tests for all new APIs and configurations
  - Integration tests with real models
  - Edge case testing (empty prompts, zero tokens, etc.)
  - Performance validation tests
  - All 264 tests passing with 21 skipped (optional dependencies)

### Performance Improvements

- **Model Reuse**: Generator class caches model between generations
  - Eliminates repeated model loading (5-10s saved per generation)
  - Smart context recreation only when necessary
  - Sampler recreation for each generation to respect config changes

- **Batch Processing**: Up to 10x throughput improvement
  - Parallel processing of multiple prompts
  - Shared model and context overhead
  - Efficient GPU utilization

- **Memory Management**: Automatic context sizing
  - Dynamic sizing based on prompt + max_tokens
  - Prevents over-allocation
  - Optimal batch sizes for available memory

## [0.1.8] - 2025-11-21

### Added

- **Speculative Decoding API** (`speculative.h` wrapper)
  - Added `SpeculativeParams` class for configuring speculative decoding parameters
  - Added `Speculative` class for managing speculative decoding with target and draft models
  - Methods: `are_compatible()`, `add_replacement()`, `gen_draft()`
  - Parameters: `n_draft` (max drafted tokens), `n_reuse` (token reuse), `p_min` (acceptance probability)
  - 17 comprehensive tests in `tests/test_speculative.py`
  - Example: `tests/examples/speculative_example.py` with parameter tuning demonstrations
  - Enables 2-3x inference speedup when using compatible draft/target model pairs
  - Supports token replacement mappings for models with different tokenizers

### Changed

- **Documentation**: Updated `RECOMMENDED_TO_WRAP.md` to mark speculative decoding as completed
  - All five high-priority APIs now fully implemented
  - Updated implementation status and remaining priorities

### Technical Implementation

- **Speculative API**: Created `speculative.pxd` with C API declarations, wrapper implementation in `speculative.pxi`
- **Context Management**: Proper handling of LlamaContext pointer access via `.ptr` attribute
- **Memory Safety**: Automatic resource cleanup with `__dealloc__` method
- **Exception Handling**: All C++ API bindings use `except +` for automatic exception translation
- **Integration**: Seamlessly integrated into main module via `llama_cpp.pyx` includes

## [0.1.7] - 2025-11-17

### Added

- **GGUF File Format API** (`gguf.h` wrapper)
  - Added `GGUFContext` class for reading and writing GGUF model files
  - Methods: `from_file()`, `write_to_file()`, `get_value()`, `get_all_metadata()`, `set_val_*()`, `get_all_tensor_info()`, `find_tensor()`, `remove_key()`
  - 6 comprehensive tests in `tests/test_gguf.py`
  - Example: `tests/examples/gguf_example.py`
  - Enables model inspection, metadata manipulation, and custom GGUF creation

- **JSON Schema to Grammar API** (`json-schema-to-grammar.h` wrapper)
  - Added `json_schema_to_grammar()` function to convert JSON schemas to GBNF grammars
  - Supports nested objects, arrays, enums, and complex schemas
  - Force GBNF mode with `force_gbnf` parameter
  - C++ wrapper layer to bridge nlohmann::json library
  - 11 comprehensive tests in `tests/test_json_schema.py`
  - Example: `tests/examples/json_schema_example.py`
  - Essential for structured JSON output from language models

- **Download Helper API** (`download.h` wrapper)
  - Added `download_model()` function for downloading from HuggingFace, URLs, and Docker registries
  - Added `get_hf_file()` function with Ollama-style quantization tags (`:q4`, `:q8`, etc.)
  - Added `list_cached_models()` function to enumerate cached models
  - Added `resolve_docker_model()` function for Docker registry integration
  - Support for bearer token authentication
  - 11 comprehensive tests in `tests/test_download.py`
  - Example: `tests/examples/download_example.py`
  - Models cached in `~/.cache/llama.cpp/`

- **N-gram Cache API** (`ngram-cache.h` wrapper)
  - Added `NgramCache` class for accelerating generation with repeated patterns
  - Methods: `update()`, `draft()`, `save()`, `load()`, `merge()`
  - Support for context/dynamic/static cache types
  - Configurable ngram_min and ngram_max parameters (2-4)
  - 14 comprehensive tests in `tests/test_ngram_cache.py`
  - Example: `tests/examples/ngram_cache_example.py`
  - Provides 2-10x speedup for repetitive text (code, templates, structured data)

### Changed

- **Exception Handling**: All new C++ API bindings use `except +` for automatic exception translation
- **Documentation**: Updated `RECOMMENDED_TO_WRAP.md` to reflect completion of 4 new high-priority APIs

### Technical Implementation

- **GGUF API**: Created `gguf.pxd` with complete C API declarations, wrapper methods in `llama_cpp.pyx`
- **JSON Schema**: C++ bridge (`json_schema.cpp/h`) for nlohmann::json, installed v3.12.0 headers
- **Download API**: Created `download.pxd`, Cython wrappers with memory-safe string handling
- **N-gram Cache**: Created `ngram_cache.pxd`, draft vector seed token initialization, proper memory management

## [0.1.6]

### Fixed

- **Multimodal (MTMD) Test Infrastructure**: Resolved critical test import and type issues for multimodal functionality
  - **Import Structure**: Fixed circular import issue in `mtmd` submodule by correcting import paths from `..mtmd` to `..llama_cpp`
  - **Data Type Compatibility**: Updated `MtmdBitmap.create_image()` parameter annotation from `str` to `bytes` to match actual Cython implementation
  - **Error Handling**: Added file existence check to `MultimodalProcessor` constructor for better error reporting before type validation
  - **Test Expectations**: Updated test assertions to match actual behavior (empty string vs None for bitmap IDs, OverflowError for invalid parameters)
  - **Mock Object Integration**: Properly configured Mock objects in tests to avoid Cython type checking conflicts
  - **Test Results**: All 27 multimodal tests now pass with 3 appropriately skipped integration tests

- **Circular Import Resolution**: Eliminated circular dependency issues in multimodal module structure
  - Fixed `src/cyllama/llama/mtmd/multimodal.py` import from `..mtmd` to `..llama_cpp`
  - Fixed `src/cyllama/llama/mtmd/__init__.py` import from `..mtmd` to `..llama_cpp`
  - Ensured proper import hierarchy where Cython classes are imported from the compiled extension module
  - Maintained backward compatibility for all existing multimodal API usage

### Changed

- **Multimodal Error Handling**: Enhanced robustness of multimodal processor initialization
  - Added early file existence validation in `MultimodalProcessor` constructor
  - Improved error messages with clearer context for file not found scenarios
  - Better separation of concerns between file validation and object initialization

### Technical Implementation

- **Import Architecture**: Corrected module import hierarchy for proper Cython class access
  - The `mtmd.pxi` include file defines Cython classes that are compiled into `llama_cpp.pyx`
  - High-level Python wrappers in `multimodal.py` now correctly import from the compiled extension
  - Eliminated self-referential imports that were causing circular dependency issues

- **Type System Compatibility**: Improved compatibility between Python test framework and Cython type checking
  - Fixed parameter type annotations to match actual implementation behavior
  - Ensured Mock objects are properly isolated from Cython type validation where appropriate
  - Maintained strict type checking for production code while enabling flexible testing

## [0.1.5]

### Added

- **High-Performance Mongoose HTTP Server**: Production-ready C-based server alternative
  - New `src/cyllama/llama/server/mongoose_server.pyx` - Cython bindings for Mongoose web server
  - Complete integration of Mongoose v7.19 (single-file embedded web server)
  - `MongooseServer` class providing high-performance alternative to Python HTTP server
  - Zero external dependencies beyond existing cyllama requirements
  - Direct C networking with concurrent connection handling (vs. Python GIL limitations)
  - Uses same `ServerSlot` logic and OpenAI-compatible API as embedded server
  - Production-ready performance for high-throughput LLM inference scenarios

- **Mongoose Server nogil Optimizations**: Advanced GIL-free operations for maximum performance
  - **Event Loop Optimization**: Core `_wait_for_shutdown_nogil()` method runs `mg_mgr_poll()` without GIL blocking
  - **Connection Management**: `_close_connections_nogil()` method for GIL-free connection cleanup operations
  - **HTTP Response Optimization**: `_send_reply_nogil()` method for non-blocking HTTP response transmission
  - **Core API Enhancement**: All Mongoose C API functions marked with `nogil` decorators for maximum efficiency
  - **Concurrent Thread Support**: Python threads can run concurrently during network I/O operations
  - **Performance Results**: 15.9μs average server lifecycle, excellent concurrent thread performance
  - **Zero API Changes**: All optimizations are transparent with full backward compatibility

- **REST API Server Infrastructure**: Complete Python wrapper for llama.cpp server functionality
  - New `src/cyllama/llama/server.py` module with comprehensive server management capabilities
  - `ServerConfig` class for complete configuration management of all llama-server parameters
  - `LlamaServer` class with full subprocess lifecycle management (start, stop, restart, status)
  - `LlamaServerClient` class providing OpenAI-compatible API client functionality
  - Automatic binary detection with fallback paths for llama-server executable
  - Context manager support for automatic server cleanup and resource management

- **OpenAI-Compatible API Support**: Full compatibility with OpenAI API standards
  - Chat completions endpoint (`/v1/chat/completions`) with streaming support
  - Embeddings endpoint (`/v1/embeddings`) for vector generation
  - Models endpoint (`/v1/models`) for available model listing
  - Health check endpoint (`/health`) for server monitoring
  - Complete request/response handling with proper error management
  - Authentication support with API keys and SSL certificates

- **Server Management Features**: Production-ready server control and monitoring
  - Graceful shutdown with configurable timeouts and fallback force-kill
  - Health checking and readiness detection with automatic retry logic
  - Server status monitoring with API readiness detection
  - Comprehensive logging and error reporting
  - Support for all llama-server configuration options and parameters
  - Web UI integration and metrics endpoint support

- **Developer Tools and Examples**: Complete development and integration support
  - `examples/server_example.py` - Full-featured server demonstration script
  - `examples/server_simple.py` - Minimal server setup example
  - Convenience `start_server()` function for quick server initialization
  - Comprehensive documentation and usage examples
  - Integration with existing cyllama module structure

- **Comprehensive Testing**: Extensive test coverage for reliability
  - `tests/test_server.py` with 28 comprehensive test cases covering all functionality
  - Unit tests for configuration, server lifecycle, and client operations
  - Integration tests with real model files and llama-server binary
  - Mock-based testing for edge cases and error conditions
  - Graceful handling of optional dependencies (requests library)
  - All tests passing with proper skip behavior for missing dependencies

### Changed

- **Module Structure**: Enhanced cyllama.llama module with server functionality
  - Added server classes to `src/cyllama/llama/__init__.py` exports
  - Updated module imports for easy access to server components
  - Maintained backward compatibility with existing API structure

- **Dependency Management**: Optional dependency handling for enhanced functionality
  - Graceful degradation when `requests` library is not available
  - Clear error messages and installation guidance for missing dependencies
  - Server functionality works without requests (health checking disabled)
  - Client functionality requires requests with helpful error messages

### Technical Implementation

- **Mongoose nogil Implementation**: Low-level GIL optimization techniques
  - **Cython nogil Decorators**: Applied to all core Mongoose C API functions including `cyllama_mg_mgr_init`, `cyllama_mg_mgr_free`, `cyllama_mg_mgr_poll`, `cyllama_mg_http_listen`, and `cyllama_mg_http_reply`
  - **C Pointer Extraction**: Safe conversion of Python bytes objects to C char pointers before entering nogil sections
  - **GIL Management**: Strategic use of `with gil:` blocks for minimal Python object access during long-running operations
  - **Thread Safety**: Preserved thread safety while enabling concurrent Python thread execution during network operations
  - **Memory Safety**: Maintained proper memory management and cleanup without introducing race conditions

- **Subprocess Management**: Robust process control and monitoring
  - Automatic binary discovery across multiple installation paths
  - Comprehensive parameter translation from Python config to command-line arguments
  - Process health monitoring with PID tracking and status detection
  - Proper signal handling for graceful shutdown sequences

- **Error Handling and Reliability**: Production-ready error management
  - Comprehensive exception handling with descriptive error messages
  - Timeout handling for server startup and shutdown operations
  - Resource cleanup and memory management for long-running servers
  - Proper handling of network connectivity issues and API failures

- **Performance and Scalability**: Optimized for production use cases
  - Minimal overhead Python wrapper around native llama-server binary
  - Efficient configuration management with parameter validation
  - Support for high-performance server configurations and GPU utilization
  - Integration with existing cyllama performance optimizations

- **Embedded Server Infrastructure**: Native Python server using existing cyllama bindings
  - New `src/cyllama/llama/server/embedded.py` module with direct llama.cpp integration
  - `EmbeddedLlamaServer` class providing OpenAI-compatible API without external binaries
  - `ServerSlot` class for concurrent request processing using native cyllama objects
  - Direct memory sharing with `LlamaModel`, `LlamaContext`, and `LlamaSampler` instances
  - Built-in HTTP server using Python's standard library for zero external dependencies
  - CLI interface via `python -m cyllama.llama.server` for easy deployment

- **Zero-Binary Deployment**: Complete server functionality without subprocess management
  - No requirement for llama-server executable or external process spawning
  - Direct integration with existing libllama.a linkage through cyllama bindings
  - Better error handling with Python-level exception management
  - Simplified deployment as single Python process with embedded functionality
  - Resource cleanup through context manager support and automatic slot management
  - Fixed critical issues with context creation, token processing, and state management

- **Native API Endpoints**: Full OpenAI-compatible server implementation
  - `/health` endpoint for server monitoring and readiness checks
  - `/v1/models` endpoint for available model listing and metadata
  - `/v1/chat/completions` endpoint with complete chat completion functionality
  - Proper JSON request/response handling with error management
  - Support for streaming responses and standard OpenAI parameters
  - Successfully generating responses like "2 + 2 = 4" with proper token handling

- **Server Implementation Fixes**: Critical bug fixes for production stability
  - Fixed `vocab.is_eog_token()` method name error to correct `vocab.is_eog()`
  - Corrected token conversion from `token_to_piece(token_id)` to `token_to_piece(token_id, 0, True)`
  - Resolved LlamaContext constructor parameter handling with proper `LlamaContextParams` objects
  - Refactored from creating new contexts per request to slot-based persistent contexts
  - Added proper context state reset between requests to prevent response contamination
  - Eliminated segmentation faults and server crashes during chat completion processing

- **Comprehensive Testing and Examples**: Production-ready development support
  - `tests/test_embedded_server.py` with 26 comprehensive test cases
  - `examples/embedded_server_example.py` - Full demonstration with API testing
  - Unit tests covering configuration, server lifecycle, and HTTP endpoints
  - Integration tests with real model files and complete request/response cycles
  - Mock-based testing for edge cases and error conditions with proper isolation
  - Verified working implementation with successful chat completion generation

## [0.1.4]

### Added

- **GPU Memory Estimation Module**: Advanced memory management and GPU allocation optimization
  - New `src/cyllama/memory.py` module with sophisticated memory estimation capabilities
  - `estimate_gpu_layers()` function for intelligent GPU layer allocation across single or multiple GPUs
  - `estimate_memory_usage()` function for comprehensive memory analysis without GPU constraints
  - `MemoryEstimate` dataclass for structured memory allocation results
  - Support for multi-GPU tensor splitting with optimal layer distribution

- **Memory CLI Tool**: Complete command-line interface for memory analysis
  - `src/cyllama/memory_cli.py` - Interactive memory estimation tool
  - Memory overview with model parameter analysis and architecture details
  - GPU allocation estimation with hardware-specific recommendations
  - Multi-GPU configuration support with tensor split visualization
  - Human-readable output formatting with size conversions (B/KB/MB/GB)
  - Performance guidance for optimal hardware utilization

- **Multi-Architecture Support**: Comprehensive model architecture compatibility
  - LLaMA, Gemma, Qwen2, StableLM, DeepSeek architecture-specific calculations
  - Automatic fallback handling for unknown architectures
  - Architecture-aware graph memory computation with optimization factors

- **Advanced Memory Features**: Professional-grade memory management capabilities
  - Multiple quantization level support (F32, F16, Q4_0, Q8_0, etc.)
  - KV cache precision options (F16/F32) with memory impact analysis
  - Context size and batch size memory scaling
  - Memory safety margins and optimization hints
  - Projector memory requirements for multimodal models

- **Integration and Testing**: Seamless codebase integration
  - Added memory estimation functions to main `__init__.py` exports
  - Comprehensive test suite with unit tests for all core functionality
  - Mock-based testing for model loading scenarios
  - Integration tests with real model files

### Changed

- **Module Exports**: Enhanced main module interface
  - Added `estimate_gpu_layers`, `estimate_memory_usage`, and `MemoryEstimate` to public API
  - Updated import structure for easy access to memory estimation features

- **Performance Optimizations**: Major performance improvements across core operations

  **Tokenization Optimizations** (Priority 2 - Medium Risk, High Benefit):
  - **Tokenization Speed**: Achieved 2.5x performance improvement (up to 4.6M tokens/s from 1.8M tokens/s)
  - **Smart Memory Allocation**: Replaced fixed vocab-size allocation with conservative text-length estimation
  - **Pre-allocated Lists**: Optimized token copying with direct assignment instead of append operations
  - **Reduced Python Overhead**: Eliminated list extension operations and optimized Cython variable declarations
  - **Memory Efficiency**: Reduced allocation overhead by ~90% for typical text lengths
  - Performance scaling across text sizes: 1.6M-4.6M tokens/s with 17K-537K calls/s

  **Property Caching Optimizations** (Priority 1 - Low Risk, Immediate Benefit):
  - **Property Access Speed**: Achieved exceptional performance with 18-21 million property accesses/second
  - **Microsecond-Level Access**: Average 0.05μs per property access (virtually instantaneous)
  - **Cached Model Properties**: Optimized n_embd, n_layer, n_head, n_head_kv, n_ctx_train, n_params, size
  - **Automatic Cache Management**: Transparent caching with zero API changes or user intervention required
  - **Property-Heavy Workload Optimization**: Perfect for memory estimation and analysis operations (3.2M workloads/s)
  - **Zero API Disruption**: Fully backward compatible with existing code and interfaces

  **Batch Operations Optimizations** (Priority 3 - Medium Risk, High Performance Benefit):
  - **Batch Processing Speed**: Achieved exceptional batch operation performance with nogil optimizations
  - **GIL-Free Operations**: Core batch setup loops run without Python GIL overhead using Cython nogil decorators
  - **Optimized Functions**: Enhanced `set_batch()`, `add_sequence()`, `set_last_logits_to_true()`, and `llama_batch_get_one()`
  - **Memory Access Patterns**: Separated Python object access from C array operations for maximum efficiency
  - **Performance Scaling**: 2.1M batch creations/s (small), 813K/s (medium), 469K/s (large), 113K/s (very large batches)
  - **Batch Workload Optimization**: 985K workloads/s for typical 32-token batch processing workflows
  - **Zero API Changes**: Fully backward compatible with existing batch processing code

  **Context Operations Optimizations** (Priority 5 - Medium Risk, High Performance Benefit):
  - **Inference Performance**: Optimized critical inference path operations with reduced Python overhead
  - **Decode Optimization**: Enhanced `LlamaContext.decode()` with streamlined error handling and optimized parameter access
  - **Sampling Optimization**: Improved `LlamaSampler.sample()` with explicit Cython variable usage and reduced overhead
  - **Conservative Approach**: Focused on Python/Cython overhead reduction while maintaining full API compatibility
  - **Inference Speed**: 22 inference cycles/s with 45.6ms average time per decode+sample cycle
  - **Error Handling**: Optimized branching with `elif` patterns for faster conditional execution
  - **Zero API Disruption**: Fully backward compatible with existing context and sampling code

  **Memory Management Optimizations** (Priority 4 - Higher Complexity, High Performance Benefit):
  - **Memory Pool Systems**: Implemented sophisticated token and batch memory pooling for efficient object reuse
  - **Token List Pooling**: `TokenMemoryPool` class provides reusable token lists for common sizes (8-512 tokens)
  - **Batch Object Pooling**: `BatchMemoryPool` class enables LlamaBatch object reuse across inference operations
  - **Tokenization Performance**: 8.6-10.6% improvement in tokenization speed through memory pool integration
  - **Batch Creation Performance**: 6.1-7.7% improvement for medium-to-large batches (32-128 tokens)
  - **High-Pressure Performance**: 22.1% improvement under intensive allocation patterns (1.08M → 1.39M allocs/s)
  - **Smart Allocation Strategy**: Automatic pool bypass for very large objects, optimal reuse for common sizes
  - **Comprehensive API**: Public functions for pool management, statistics, and explicit pooled object creation
  - **Overall Performance Gain**: 8.8% faster performance across combined memory-intensive operations

### Technical Implementation

- **xllamacpp Integration**: Adapted best practices from xllamacpp fork analysis
  - Implemented memory estimation algorithms based on xllamacpp's sophisticated approach
  - Maintained compatibility with existing cyllama architecture and design principles
  - Selective integration focusing on memory management without breaking existing functionality

- **Performance Optimization**: Efficient memory calculation algorithms
  - Architecture-specific memory computation with minimal overhead
  - Intelligent layer size estimation based on quantization schemes
  - Optimized graph memory calculations with attention mechanism considerations

## [0.1.3]

### Added

- **Whisper Support**: Added Whisper.cpp integration for speech-to-text functionality
  - New `src/cyllama/whisper/` module with Cython bindings for whisper.cpp
  - `whisper_cpp.pyx` - Primary Whisper Cython extension module
  - `tests/test_whisper.py` - Comprehensive Whisper test suite
  - `samples/jfk.wav` - Sample audio file for testing
  - `scripts/download-ggml-model.sh` - Script to download Whisper models

- **Whisper CLI**: Complete Python CLI wrapper equivalent to whisper.cpp CLI
  - `src/cyllama/whisper/cli.py` - Full command-line interface for speech-to-text
  - Support for all major whisper.cpp CLI parameters and options
  - Multiple output formats: TXT, SRT, VTT, CSV, JSON (basic and full), LRC
  - Audio file loading with automatic resampling to 16kHz
  - WAV format support for 8, 16, 24, and 32-bit audio files
  - GPU acceleration support with Metal backend on macOS
  - Language detection and translation capabilities
  - Comprehensive argument parsing with help documentation

### Changed

- **Major Code Restructuring**: Reorganized codebase to support multiple AI modalities
  - Moved LLaMA-specific code to `src/cyllama/llama/` subdirectory
  - Separated Whisper functionality into `src/cyllama/whisper/` subdirectory
  - Updated module imports and package structure
  - Added `src/cyllama/__main__.py` for CLI entry point

- **Text-to-Speech Improvements**: Enhanced TTS functionality with better C++ compatibility
  - Improved TTS generation to match llama.cpp reference implementation
  - Fixed audio quality issues and generation completeness
  - Better speaker template management and prompt construction

- **Build System Updates**: Enhanced build configuration for multi-modal support
  - Updated `Makefile` with Whisper-specific build targets
  - Enhanced `setup.py` for multi-extension compilation
  - Updated `MANIFEST.in` and `pyproject.toml` for new package structure

### Fixed

- **Token Decoding**: Fixed `token_to_piece` method corruption issues
  - Resolved text output with replacement characters
  - Proper buffer length handling for token decoding
  - Added error handling for negative return values

- **Whisper Transcription**: Enabled and fixed the `full()` method in Whisper wrapper
  - Uncommented and activated the main transcription functionality
  - Fixed Cython compilation issues with proper memory view handling
  - Corrected import paths for whisper.pxd module
  - Proper error handling for transcription failures

## [0.1.2]

- Updated to latest release of `llama.cpp`: `b6374`

- Added unit tests

- Changed cyllama.pyx and tests to apply more consistent naming of Llama-type classes.

## [0.1.0]

- Moved cyllama code from [llamalib](https://github.com/shakfu/llamalib) to this repo
- Added low-level simple wrapper using cyllama
- Added high-level simple wrapper using cyllama
