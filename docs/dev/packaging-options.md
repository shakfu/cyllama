# Packaging Options for cyllama

This document outlines packaging strategies for cyllama, which provides Cython wrapper support for three ggml-based libraries: llama.cpp, whisper.cpp, and stable-diffusion.cpp.

## Current Situation

Currently, the underlying C++ libraries are statically linked, with each library potentially using a different version of ggml. This creates massive builds, especially for CUDA backends.

## The Shared Library Question

**Question**: What about compiling the dependencies as shared libraries and then linking them to the extensions?

**Answer**: If you compile ggml as a single shared library and link all three projects against it, they would all need to use the **same ggml version**. This is problematic because:

- llama.cpp, whisper.cpp, and stable-diffusion.cpp often pin to different ggml commits
- ggml's API/ABI is not stable between versions
- You'd be forced to synchronize updates across all three upstream projects

## Options to Consider

### Option 1: Shared libs with ggml statically linked (symbol visibility hidden)

Each of llama.cpp, whisper.cpp, and stable-diffusion.cpp becomes a shared library with ggml statically linked inside, but with internal symbols hidden (`-fvisibility=hidden`). This:

- Allows different ggml versions per library
- Reduces some duplication (shared C++ runtime, etc.)
- Avoids symbol conflicts at runtime

### Option 2: Split into separate Python packages

Instead of one monolithic `cyllama`, ship three separate packages:

- `cyllama` (llama.cpp wrapper)
- `cywhisper` (whisper.cpp wrapper)
- `cystable-diffusion` (stable-diffusion.cpp wrapper)

Users install only what they need. Doesn't reduce individual build size but reduces what users download.

### Option 3: Modular CUDA backend

The real size bloat comes from CUDA. Consider:

- Building CUDA kernels as a separate shared library
- ggml already has some support for this pattern
- One shared CUDA backend could potentially serve all three (if ggml versions align)

### Option 4: Runtime loading / plugin architecture

Ship CPU-only by default, with CUDA as an optional runtime-loadable plugin. This is how PyTorch and others handle it.

## Recommended Approach

A hybrid approach:

1. **Split into separate packages** - most users don't need all three
2. **Within each package**, use static linking with hidden visibility (current approach but optimized)
3. **Factor out CUDA** as a separate optional dependency if the ggml versions can be aligned for that specific component

## Trade-offs Summary

| Approach | Build Time | Wheel Size | Install Size | Version Flexibility |
|----------|------------|------------|--------------|---------------------|
| Current (static) | High | Large | Large | High |
| Single shared ggml | Medium | Small | Small | Low |
| Shared libs + hidden ggml | Medium | Medium | Medium | High |
| Separate packages | High | Large per-pkg | User choice | High |
| Modular CUDA | Medium | Small base | User choice | Medium |
