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


## [0.1.3]

### Added
- **Whisper Support**: Added Whisper.cpp integration for speech-to-text functionality
  - New `src/cyllama/whisper/` module with Cython bindings for whisper.cpp
  - `whisper_cpp.pyx` - Primary Whisper Cython extension module
  - `tests/test_whisper.py` - Comprehensive Whisper test suite
  - `samples/jfk.wav` - Sample audio file for testing
  - `scripts/download-ggml-model.sh` - Script to download Whisper models

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

## [0.1.2]

- Updated to latest release of `llama.cpp`: `b6374`

- Added unit tests

- Changed cyllama.pyx and tests to apply more consistent naming of Llama-type classes.

## [0.1.0]

- Moved cyllama code from [llamalib](https://github.com/shakfu/llamalib) to this repo
- Added low-level simple wrapper using cyllama
- Added high-level simple wrapper using cyllama
