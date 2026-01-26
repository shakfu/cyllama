# Packaging Options for cyllama

This document outlines packaging strategies for cyllama, which provides Cython wrapper support for three ggml-based libraries: llama.cpp, whisper.cpp, and stable-diffusion.cpp.

## Current Situation

Currently, the underlying C++ libraries are statically linked, with each library potentially using a different version of ggml. This creates massive builds, especially for CUDA backends.

### Current Build Characteristics

- **Static linking without hidden visibility**: All symbols from the static libraries are exported
- **`--whole-archive` on Linux**: Used to ensure function-pointer symbols (like `ggml_repeat_4d`) are included, but this exports all symbols
- **Separate ggml per project**: stable-diffusion.cpp links against its own ggml libraries, while llama.cpp and whisper.cpp share theirs
- **No symbol isolation**: Symbols from all ggml copies are exported, which could cause conflicts if multiple extensions are loaded in the same process (works now because extensions are typically loaded separately)

## The Shared Library Question

**Question**: What about compiling the dependencies as shared libraries and then linking them to the extensions?

**Answer**: If you compile ggml as a single shared library and link all three projects against it, they would all need to use the **same ggml version**. This is problematic because:

- llama.cpp, whisper.cpp, and stable-diffusion.cpp often pin to different ggml commits
- ggml's API/ABI is not stable between versions
- You'd be forced to synchronize updates across all three upstream projects

## Options to Consider

### Option 1: Static linking with hidden visibility (optimized current approach)

Keep static linking but properly hide internal symbols. This requires changes in two places:

**1. When building upstream static libraries (llama.cpp, whisper.cpp, stable-diffusion.cpp):**

```cmake
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
```

Or via compiler flags:
```
-fvisibility=hidden -fvisibility-inlines-hidden
```

The public API symbols need explicit `__attribute__((visibility("default")))` marking.

**2. When building cyllama extensions:**

```cmake
if(NOT WIN32)
    set(CXX_COMPILE_OPTIONS -std=c++17 -fvisibility=hidden)
endif()
```

**Benefits:**
- Allows different ggml versions per library (no symbol conflicts)
- Smaller binaries (linker can discard unused hidden symbols)
- Avoids symbol conflicts at runtime
- Removes need for `--whole-archive` (only public API symbols are exported)

**Challenges:**
- Requires patching upstream projects or maintaining forks
- Need to ensure all necessary symbols are marked as visible in upstream

### Option 2: Shared libs with ggml statically linked (symbol visibility hidden)

Each of llama.cpp, whisper.cpp, and stable-diffusion.cpp becomes a shared library (`.so`/`.dylib`) with ggml statically linked inside, but with internal symbols hidden. This:

- Allows different ggml versions per library
- Reduces some duplication (shared C++ runtime, etc.)
- Avoids symbol conflicts at runtime
- Similar to Option 1 but produces shared libraries instead of static

### Option 3: Split into separate Python packages

Instead of one monolithic `cyllama`, ship three separate packages:

- `cyllama` (llama.cpp wrapper)
- `cywhisper` (whisper.cpp wrapper)
- `cystable-diffusion` (stable-diffusion.cpp wrapper)

Users install only what they need. Doesn't reduce individual build size but reduces what users download.

### Option 4: Modular CUDA backend

The real size bloat comes from CUDA. Consider:

- Building CUDA kernels as a separate shared library
- ggml already has some support for this pattern
- One shared CUDA backend could potentially serve all three (if ggml versions align)

### Option 5: Runtime loading / plugin architecture

Ship CPU-only by default, with CUDA as an optional runtime-loadable plugin. This is how PyTorch and others handle it.

## Recommended Approach

A hybrid approach:

1. **Split into separate packages** - most users don't need all three
2. **Within each package**, use static linking with hidden visibility (current approach but optimized)
3. **Factor out CUDA** as a separate optional dependency if the ggml versions can be aligned for that specific component

## Trade-offs Summary

| Approach | Build Time | Wheel Size | Install Size | Version Flexibility | Symbol Safety |
|----------|------------|------------|--------------|---------------------|---------------|
| **Current** (static, no hidden vis) | High | Large | Large | High | Poor |
| Static + hidden visibility | High | Medium | Medium | High | Good |
| Shared libs + hidden ggml | Medium | Medium | Medium | High | Good |
| Single shared ggml | Medium | Small | Small | Low | Good |
| Separate packages | High | Large per-pkg | User choice | High | Depends |
| Modular CUDA | Medium | Small base | User choice | Medium | Depends |

### Notes on Symbol Safety

- **Poor**: Multiple ggml symbol sets exported; potential conflicts if multiple extensions loaded
- **Good**: Symbols properly isolated; safe to load multiple extensions
- **Depends**: Inherits from underlying approach chosen
