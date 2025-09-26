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
