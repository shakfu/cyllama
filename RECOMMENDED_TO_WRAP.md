# Recommended API Additions for cyllama

This document outlines additional llama.cpp APIs that would be valuable to wrap in the cyllama Python library.

## High Value - Recommended to Implement

### 1. **GGUF File Format API** (`gguf.h`)
**Priority: HIGH** - Currently completely missing

**Why**: Direct GGUF file manipulation is useful for:
- Model inspection and metadata reading
- Model conversion/modification tools
- Custom model creation

**What to wrap**:
```python
# Read model metadata without loading full model
gguf_info = GGUFReader("model.gguf")
print(gguf_info.get_metadata())
print(gguf_info.architecture)
print(gguf_info.tensor_info)
```

**Files needed**: `gguf.pxd`, possibly `gguf.py`

---

### 2. **Speculative Decoding API** (`speculative.h`)
**Priority: MEDIUM-HIGH** - Partially present in common_params but not exposed

**Why**: 2-3x inference speedup for compatible models

**Current state**: Parameters exist but no dedicated wrapper

**What to add**:
```python
# Speculative decoding with draft model
result = model.generate(
    prompt="...",
    speculative_model="draft-model.gguf",
    speculative_n_max=16
)
```

**Files needed**: `speculative.pxd`, wrapper in `llama_cpp.pyx`

---

### 3. **N-gram Cache API** (`ngram-cache.h`)
**Priority: MEDIUM** - For repeated pattern acceleration

**Why**: Speeds up inference when text has repeated patterns

**What to wrap**:
```python
# Enable n-gram cache for repetitive text
context.enable_ngram_cache(n_gram_min=4, n_gram_max=1024)
```

**Files needed**: `ngram-cache.pxd`

---

### 4. **JSON Schema to Grammar** (`json-schema-to-grammar.h`)
**Priority: MEDIUM-HIGH** - For structured output

**Why**: Essential for constrained generation with JSON output

**Current state**: Likely used internally but not exposed to Python

**What to expose**:
```python
# Generate grammar from JSON schema
schema = {"type": "object", "properties": {...}}
grammar = json_schema_to_grammar(schema)
result = model.generate(prompt="...", grammar=grammar)
```

**Files needed**: `json-schema-to-grammar.pxd`

---

### 5. **Download Helper** (`download.h`)
**Priority: MEDIUM** - New in recent llama.cpp

**Why**: Built-in model downloading from HuggingFace/URLs

**What to wrap**:
```python
# Download models directly
from cyllama.llama import download_model

path = download_model(
    "https://huggingface.co/user/model/resolve/main/model.gguf",
    cache_dir="~/.cache/cyllama"
)
```

**Files needed**: `download.pxd`

---

## Medium Value - Consider Implementing

### 6. **Console Utilities** (`console.h`)
**Priority: LOW-MEDIUM** - Better terminal output

**Why**: Better progress bars, color output, formatting

**What to wrap**: Terminal width detection, color support, progress rendering

---

### 7. **HTTP Utilities** (`http.h`)
**Priority: LOW** - Already have server implementations

**Note**: You already have embedded/mongoose servers, this is lower priority

---

### 8. **Regex/JSON Partial Parsers** (`regex-partial.h`, `json-partial.h`)
**Priority: LOW** - Streaming structured output

**Why**: Parse incomplete JSON/regex during streaming generation

**Use case**: Real-time structured output parsing

---

## Low Value - Skip or Low Priority

### 9. **Backend-Specific APIs**
- `ggml-metal.h`, `ggml-cuda.h`, `ggml-vulkan.h`, etc.

**Why skip**: Backend selection is automatic, rarely needs manual control

**Exception**: Metal API might be useful on macOS for advanced GPU control

---

### 10. **Low-Level GGML Operations**
- `ggml-alloc.h`, `ggml-opt.h`, `ggml-cpu.h`

**Why skip**: Too low-level, used for model training/custom operators

**Exception**: Could be useful if you want to build custom layers

---

### 11. **Argument Parser** (`arg.h`)

**Why skip**: Python has better CLI libraries (argparse, click, typer)

---

## Recommended Implementation Order

1. **GGUF Reader/Writer** - Most immediately useful for model inspection
2. **JSON Schema to Grammar** - High demand for structured output
3. **Download Helper** - Improves UX significantly
4. **Speculative Decoding** - Performance boost for users
5. **N-gram Cache** - Performance optimization
6. **Console Utilities** - Nice-to-have for better UX

---

## Quick Wins

For immediate value with minimal work:

### 1. **Expose existing but hidden features**
- Speculative decoding parameters are in `common_params` but may not be documented
- Check if `json-schema-to-grammar` is already linked but not exposed

### 2. **Add convenience wrappers**
```python
# Example: High-level GGUF inspection
def inspect_model(path: str) -> dict:
    """Quick model info without full load"""
    # Wrap gguf_* functions
```

---

## Current Wrapper Status

### Fully Wrapped
- ✅ Core llama.cpp API (`llama.h`)
- ✅ GGML core (`ggml.h`)
- ✅ Common utilities (`common.h`)
- ✅ Sampling API (`sampling.h`)
- ✅ Chat templates (`chat.h`)
- ✅ Multimodal (MTMD) (`mtmd.h`, `mtmd-helper.h`)
- ✅ Logging (`log.h`)
- ✅ TTS helpers
- ✅ Server implementations (embedded, mongoose, launcher)
- ✅ Whisper.cpp integration

### Partially Wrapped
- ⚠️ Backend management (`ggml-backend.h`) - basic functions only
- ⚠️ Speculative decoding - parameters exist but no high-level API

### Not Wrapped
- ❌ GGUF file format (`gguf.h`)
- ❌ JSON schema to grammar (`json-schema-to-grammar.h`)
- ❌ Download utilities (`download.h`)
- ❌ N-gram cache (`ngram-cache.h`)
- ❌ Console utilities (`console.h`)
- ❌ Regex/JSON partial parsers
- ❌ Backend-specific APIs (metal, cuda, vulkan, etc.)
- ❌ Low-level GGML operations

---

## Analysis Date
November 17, 2025

## Notes
This analysis is based on llama.cpp tag b6374 (or current bleeding edge). As llama.cpp evolves, new APIs may be added that should be considered for wrapping.
