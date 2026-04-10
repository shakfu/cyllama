# Chat Templates: Cython Binding Upgrade Analysis

This document captures the design analysis for upgrading cyllama's chat-template handling from llama.cpp's basic C API (`llama_chat_apply_template`) to the full Jinja-aware path (`common_chat_templates_apply` from `common/chat.cpp`).

The current implementation uses the basic C API, which only handles a hardcoded set of templates detected via substring heuristics on the embedded template string. Models whose embedded Jinja templates don't match any heuristic (Gemma 4 in particular, plus any future model llama.cpp's hardcoded list doesn't recognize) cause `RuntimeError: Failed to apply chat template` regardless of message shape. The pipeline currently has a three-tier fallback (`canonical chat -> merged-user chat -> raw completion`) that handles the symptom, but the proper fix is at the binding layer.

## Current state

cyllama's Cython binding (`src/cyllama/llama/llama_cpp.pyx:1948`) calls `llama.llama_chat_apply_template`, which is the basic C API exposed in `llama.h`:

```c
int32_t llama_chat_apply_template(
    const char * tmpl,
    const struct llama_chat_message * chat,
    size_t n_msg,
    bool add_ass,
    char * buf,
    int32_t length);
```

Inside llama.cpp this function calls `llm_chat_detect_template(curr_tmpl)` (`build/llama.cpp/src/llama-chat.cpp:88`), which:

1. Tries an exact-name lookup against the `LLM_CHAT_TEMPLATES` map (line 30-82) — this only matches when the template is one of llama.cpp's known short names like `"chatml"`, `"llama3"`, `"gemma"`.
2. If that fails, runs a series of substring heuristics looking for distinctive markers in the embedded Jinja template (e.g., `<start_of_turn>` -> `LLM_CHAT_TEMPLATE_GEMMA`, `<|start_header_id|>` -> `LLM_CHAT_TEMPLATE_LLAMA_3`, etc.).
3. If neither matches, returns `LLM_CHAT_TEMPLATE_UNKNOWN`, which causes `llama_chat_apply_template` to return -1.

The hardcoded path **does** have a graceful Gemma handler at line 375-396 that merges system messages into the user prompt — but only when the substring heuristic actually matches `LLM_CHAT_TEMPLATE_GEMMA`. For Gemma 4, the embedded Jinja template apparently doesn't trigger the substring match, and detection falls through to UNKNOWN.

llama.cpp **also** has a Jinja-aware path in `common/chat.cpp` that uses `minja` (a Jinja interpreter) to evaluate the embedded template directly. The relevant entry points (declared in `common/chat.h`):

```cpp
common_chat_templates_ptr common_chat_templates_init(
    const struct llama_model * model,
    const std::string & chat_template_override,
    const std::string & bos_token_override = "",
    const std::string & eos_token_override = "");

struct common_chat_params common_chat_templates_apply(
    const struct common_chat_templates * tmpls,
    const struct common_chat_templates_inputs & inputs);

void common_chat_templates_free(struct common_chat_templates * tmpls);
```

`common_chat_templates_apply` evaluates the embedded Jinja template via `minja` and returns a `common_chat_params` whose `prompt` field is the formatted text. This handles **any** Jinja template the model declares, regardless of whether llama.cpp's substring heuristics recognize it.

`libcommon.a` is already built and installed by `manage.py` at `thirdparty/llama.cpp/lib/libcommon.a`. The relevant headers (`chat.h`, `common.h`, `jinja/*.h`, `nlohmann/*.hpp`) are already exposed at `thirdparty/llama.cpp/include/`. The required symbols (`common_chat_templates_init`, `common_chat_templates_apply`, `common_chat_templates_free`) are present in `libcommon.a`. **The build system just doesn't link against it.**

## Existing pipeline-level fallback

`RAGPipeline._generate_chunks` already has a three-tier fallback chain that handles the user-visible symptom of this bug:

1. **Canonical** `[system, user]` chat-template call.
2. **Merged** `[user]` chat-template call (system content prepended) — for models like the hardcoded Gemma path that reject system role.
3. **Raw-completion path** with the legacy `Question:/Context:/Answer:` template — fires when both chat shapes raise `RuntimeError("...template...")`. Caches the decision on `_chat_template_unusable` so subsequent queries skip the failed chat attempts. Emits a one-time `RuntimeWarning`.

This works for Gemma 4 today (verified end-to-end with `scripts/case/rag-chat2.sh`) but produces lower-quality answers because the model is being prompted in a format it wasn't instruction-tuned on.

The binding upgrade would make tier 1 succeed for any model with an embedded Jinja template, demoting tiers 2 and 3 to defensive code that almost never fires.

## Two implementation options

### Option A: Replace

Rip out the existing `chat_apply_template` Cython method and use only `common_chat_templates_apply`.

- **Pros:** Single code path. Simpler binding. Removes the substring-heuristic limitation entirely. Smaller diff in `api.py`.
- **Cons:** If `minja` ever fails on a template (corrupt GGUF, future llama.cpp API churn, edge-case Jinja syntax it doesn't yet support), there's no fallback inside the binding — failures propagate up to the pipeline's three-tier fallback, which still works but sends straight to raw completion.
- **Risk:** Higher. One bug in the new path breaks chat for everyone.
- **Code size:** ~150-200 lines of new Cython, ~30 lines of CMakeLists, 0 lines of `api.py` changes (the existing call site just routes to the new method).

### Option B: Add alongside, fall back on failure

Keep `chat_apply_template` (basic) and add `chat_apply_jinja_template` (full Jinja). `api.py` tries Jinja first, falls back to basic on `RuntimeError`. Pipeline's three-tier fallback stays as the outermost safety net.

- **Pros:** Belt-and-suspenders. Any bug in the new path is masked by the legacy path. Can ship incrementally — if the Jinja binding has problems on some platform, users still get the existing behavior.
- **Cons:** Two paths to maintain. The legacy path becomes mostly dead code (only fires on Jinja failures), which is some maintenance overhead. Diff is ~30% larger.
- **Risk:** Much lower. The new path is purely additive.
- **Code size:** ~180-230 lines of new Cython, ~30 lines of CMakeLists, ~20 lines of `api.py` changes.

### Option C: Defer

Don't do this now. The three-tier pipeline fallback already handles Gemma 4 (with a `RuntimeWarning` and degraded answer quality). Document this as a known follow-up and address it in a focused PR later, ideally with end-to-end validation against multiple model families.

- **Pros:** Zero risk to current working state. Lets you validate the broader fix in `rag-chat2.sh` first to confirm the user-facing problem is genuinely solved before taking on the binding work. Decouples binding work from the bigger refactor.
- **Cons:** Gemma 4 (and similar models) get the degraded raw-completion path with a warning, which is suboptimal but functional.
- **Risk:** None. Status quo.
- **Code size:** 0.

### Recommendation

**Option B**, when ready to commit to a build cycle. The extra ~30% code is cheap insurance against subtle bugs in the new path, and the legacy fallback is genuinely useful for the cases (corrupted GGUFs, future API churn) where minja itself can't evaluate a template. If the binding eventually proves stable across many model families, the legacy path can be removed in a follow-up.

## Option B in detail

### Architecture

```text
LLM.chat(messages)
  +-> api.py:_apply_template
        +-> NEW: model.chat_apply_jinja_template(messages)        <-- full Jinja
        |     +- common_chat_templates_init (cached per model)
        |     +- common_chat_templates_apply
        +-> on RuntimeError, fall back to:
            model.chat_apply_template(tmpl, chat_messages, ...)   <-- legacy hardcoded
              +- llama_chat_apply_template
                 +- on -1, returns RuntimeError to pipeline,
                    which uses raw-completion path
```

### Files touched (6 files, ~280-320 lines added)

#### 1. `CMakeLists.txt` — ~6 lines added

Add `libcommon.a` to the static library list:

```cmake
static_lib(LIB_COMMON "${LLAMACPP_LIB}" "common")
list(APPEND STATIC_LIBS "${LIB_COMMON}")
```

Critical detail: `LIB_COMMON` must come **before** `LIB_LLAMA` in the link order on Linux because `libcommon.a` references symbols from `libllama`. macOS doesn't care about order. The current `STATIC_LIBS` order would need adjusting.

#### 2. `src/cyllama/llama/chat.pxd` — new file, ~50 lines

Cython declarations mirroring `chat.h`. Only the fields we actually touch — Cython doesn't need the full struct, just enough to construct locals and read the result. Approximate shape:

```cython
from libcpp.string cimport string as std_string
from libcpp.vector cimport vector as std_vector

cdef extern from "llama.h":
    cdef cppclass llama_model

cdef extern from "chat.h":
    cdef cppclass common_chat_msg:
        std_string role
        std_string content

    cdef cppclass common_chat_templates_inputs:
        common_chat_templates_inputs() except +
        std_vector[common_chat_msg] messages
        bint add_generation_prompt
        bint use_jinja
        bint add_bos
        bint add_eos

    cdef cppclass common_chat_params:
        std_string prompt

    cdef cppclass common_chat_templates  # opaque

    common_chat_templates * common_chat_templates_init(
        const llama_model * model,
        const std_string & chat_template_override,
        const std_string & bos_token_override,
        const std_string & eos_token_override,
    ) except +

    common_chat_params common_chat_templates_apply(
        const common_chat_templates * tmpls,
        const common_chat_templates_inputs & inputs,
    ) except +

    void common_chat_templates_free(common_chat_templates * tmpls)
```

The `except +` clauses are load-bearing — they're how Cython converts `std::runtime_error` (raised by `minja` on bad templates) into Python `RuntimeError`. Without them, exceptions would terminate the process.

#### 3. `src/cyllama/llama/llama_cpp.pyx` — ~80 lines added

Two changes to the existing `LlamaModel` class:

**a) Add a cached `common_chat_templates *` field**:

```cython
cdef common_chat_templates * _chat_templates  # NULL until first use
```

Initialized to `NULL` in `__cinit__`, freed in `__dealloc__` if non-NULL. The free must happen *before* the underlying `llama_model *` is freed, since the templates hold a reference to it.

**b) New method**:

```cython
def chat_apply_jinja_template(
    self,
    list msgs,                        # list[dict[str, str]]
    bint add_generation_prompt=True,
) -> str:
    """Apply the model's embedded chat template via llama.cpp's Jinja-
    aware common_chat_templates_apply, returning the formatted prompt.

    Lazily initializes the common_chat_templates handle on first call
    and caches it for the model's lifetime.
    """
    cdef std_string empty
    cdef common_chat_templates_inputs inputs
    cdef common_chat_msg msg
    cdef common_chat_params params

    if self._chat_templates is NULL:
        self._chat_templates = common_chat_templates_init(
            self.ptr, empty, empty, empty,
        )
        if self._chat_templates is NULL:
            raise RuntimeError("common_chat_templates_init returned NULL")

    inputs.add_generation_prompt = add_generation_prompt
    inputs.use_jinja = True
    for m in msgs:
        msg.role = (<str>m["role"]).encode("utf-8")
        msg.content = (<str>m["content"]).encode("utf-8")
        inputs.messages.push_back(msg)

    params = common_chat_templates_apply(self._chat_templates, inputs)
    return params.prompt.decode("utf-8")
```

Subtleties:

- `common_chat_templates_init` is called once per `LlamaModel` instance and the result is cached. This avoids re-parsing the Jinja template on every chat call.
- The `common_chat_msg` struct has more fields than just `role`/`content` (`tool_calls`, `reasoning_content`, etc.), but they're all default-constructed and we ignore them.
- `params` is returned by value and copy-constructed. `common_chat_params` has multiple `std::string` and `std::vector` fields — copy is O(prompt length) but unavoidable. Could be optimized later with `std::move`.

#### 4. `src/cyllama/api.py:_apply_template` — ~15 lines changed

```python
def _apply_template(self, messages, template=None, add_generation_prompt=True):
    # When the caller hasn't requested a specific template, prefer the
    # Jinja-capable common_chat_templates_apply path. This handles any
    # GGUF whose embedded chat template uses Jinja syntax (Gemma 2/3/4,
    # newer Mistral variants, etc.) -- which the legacy
    # llama_chat_apply_template substring heuristic cannot.
    if template is None:
        try:
            return self.model.chat_apply_jinja_template(
                messages, add_generation_prompt
            )
        except RuntimeError:
            # Jinja path failed -- the model has no embedded template,
            # or minja can't parse it. Fall through to the hardcoded
            # substring-heuristic path which still works for the
            # well-known templates (Llama, Qwen, ChatML, etc.).
            pass

    # --- existing legacy code path stays unchanged below ---
    if template:
        tmpl = self.model.get_default_chat_template_by_name(template)
        # ...
```

The change is purely additive — the legacy path is unchanged. If the new path raises, we fall through; if it returns, we return the result.

#### 5. `tests/test_llama_cpp_chat.py` (or similar) — ~60-80 lines added

Real-model tests because the binding can't be meaningfully mocked. Use the existing test fixture (`models/Llama-3.2-1B-Instruct-Q8_0.gguf`):

```python
def test_chat_apply_jinja_template_basic(model_path):
    """Verify the new Jinja path produces a Llama-3-shaped prompt."""
    model = LlamaModel(model_path)
    prompt = model.chat_apply_jinja_template([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi."},
    ])
    # Llama-3 template emits these tokens
    assert "<|start_header_id|>system<|end_header_id|>" in prompt
    assert "You are helpful." in prompt
    assert "<|start_header_id|>user<|end_header_id|>" in prompt
    assert "Hi." in prompt
    # add_generation_prompt=True by default
    assert "<|start_header_id|>assistant<|end_header_id|>" in prompt

def test_chat_apply_jinja_template_caching(model_path):
    """Multiple calls on the same model must reuse the cached templates."""
    model = LlamaModel(model_path)
    p1 = model.chat_apply_jinja_template([{"role": "user", "content": "a"}])
    p2 = model.chat_apply_jinja_template([{"role": "user", "content": "b"}])
    assert "a" in p1 and "b" in p2

def test_chat_apply_jinja_template_no_generation_prompt(model_path):
    """add_generation_prompt=False should omit the trailing assistant header."""
    model = LlamaModel(model_path)
    prompt = model.chat_apply_jinja_template(
        [{"role": "user", "content": "Hi."}],
        add_generation_prompt=False,
    )
    assert "Hi." in prompt
    # No trailing "<|start_header_id|>assistant<|end_header_id|>"
    assert not prompt.rstrip().endswith("assistant<|end_header_id|>")
```

Plus a Gemma-only test marked `@pytest.mark.skipif(not GEMMA_PATH.exists())` that verifies the binding handles a system message correctly on Gemma (where the legacy path raises).

#### 6. `CHANGELOG.md` — one entry under `[Unreleased] / Changed`

Document the new Jinja path as the default, the legacy path as the fallback, and note that the pipeline's three-tier fallback is now mostly dead code (kept for safety).

### Build & test cycle

Each iteration is approximately:

| Step | Time | What it does |
|---|---|---|
| 1. Edit Cython/C++ code | — | |
| 2. `make` (or rebuild via scikit-build) | ~3-5 min | Recompiles `llama_cpp.pyx` -> `.cpp` -> `.so` |
| 3. `uv run pytest tests/test_main.py tests/test_rag_*.py -q` | ~10-20s | Existing tests |
| 4. `uv run pytest tests/test_llama_cpp_chat.py` | ~5-15s per test | New binding tests (each loads a model) |
| 5. `./scripts/case/rag-chat2.sh` | manual | End-to-end Gemma 4 verification |

Plan for **2-4 build cycles** to get the binding compiling and the tests passing. Per cycle, the build is the slow step.

### Concrete failure modes to plan for

1. **Cython can't compile the struct declaration.**
    - Symptom: `error: 'common_chat_templates_inputs' is not a class template`
    - Cause: Declared a field with a type Cython doesn't understand (e.g., `std::chrono::time_point`)
    - Fix: omit that field from the Cython declaration; the C++ compiler still sees the full type via `chat.h`

2. **Linker can't find `common_chat_templates_init`.**
    - Symptom: `undefined symbol: common_chat_templates_init`
    - Cause: `libcommon.a` not in the link order, or in the wrong position
    - Fix: place `LIB_COMMON` *before* `LIB_LLAMA` in `STATIC_LIBS`

3. **Linker pulls in symbols that conflict with libllama.**
    - Symptom: duplicate symbol errors
    - Cause: `libcommon.a` and `libllama` both define some symbol
    - Fix: should not happen because `libcommon.a` is designed to extend `libllama`, but if it does, may need `--allow-multiple-definition` (Linux) — undesirable, but a known escape hatch

4. **`common_chat_templates_init` returns NULL on a model without an embedded template.**
    - Symptom: dereferencing NULL -> segfault
    - Fix: NULL-check after init, raise `RuntimeError("model has no embedded chat template")`, let the legacy fallback take over

5. **`common_chat_templates_apply` throws on a Jinja template that uses unsupported features.**
    - Symptom: `RuntimeError` with a minja error message
    - Fix: caught by `except +` in the Cython declaration, propagated to Python, caught by `_apply_template`'s `except RuntimeError`, falls back to legacy. Already handled.

6. **`common_chat_templates_free` crashes during `__dealloc__`.**
    - Symptom: segfault on Python interpreter shutdown
    - Cause: order of destruction — if `llama_model *` is freed before `common_chat_templates *`, the templates hold a dangling pointer
    - Fix: free `_chat_templates` *before* freeing the model in `__dealloc__`

7. **macOS-only: Metal symbols not pulled in transitively.**
    - Unlikely, but possible if `libcommon.a` somehow needs ggml symbols that the linker decides are dead-stripped
    - Fix: ensure `-Wl,--whole-archive` (Linux) or `-force_load` (macOS) is applied if needed. Currently CMakeLists does this for ggml libs only.

### What cannot easily be tested in a tool-call-only environment

- **Build success on Linux/Windows.** macOS-Metal is verifiable locally; the CMakeLists changes affect every backend permutation. The CI configs would catch issues, but a full CI cycle is much slower than local iteration.
- **Cross-model behavior.** The new binding can be run against Llama-3.2-1B (the test fixture). Verifying Qwen3, Gemma 4, Mistral, etc. requires those models on disk. Gemma 4 verification is the main motivation, so this needs to be validated manually after the build.
- **Memory safety under stress.** A full leak-check run (`tests/test_memory_leaks.py::TestLLMLeaks::test_create_destroy_loop`) would catch any leak in the new `_chat_templates` lifecycle, and is worth running once after the binding is in place.

### Incremental commit structure

Three logical chunks so each is independently verifiable:

1. **CMakeLists + chat.pxd + empty Cython method that just initializes/frees the templates handle.** Verify the build links and the model loads/unloads cleanly. No new behavior yet.
2. **Implement `chat_apply_jinja_template` body, add binding tests against the test model.** Verify the new method produces correct prompts for Llama-3.
3. **Wire the new method into `_apply_template` with the fallback, update CHANGELOG.** Verify existing tests still pass and that `rag-chat2.sh` against Gemma 4 now uses the Jinja path.

If step 2 fails, step 1 still gives a working build with no behavior change — easy to roll back.

### Effort estimate

- Realistic, focused work, no surprises: **1.5-2 hours**.
- Realistic with one or two debug cycles on the binding: **2-3 hours**.
- If something deeply weird happens (Cython refuses to compile the struct, linker conflicts on a specific platform): escalate rather than burning time.

## Option A in detail

The structural changes for Option A are a strict subset of Option B. Specifically:

### Files touched (5 files, ~210-250 lines)

1. **`CMakeLists.txt`** — same as Option B (~6 lines).
2. **`src/cyllama/llama/chat.pxd`** — same as Option B (~50 lines, new file).
3. **`src/cyllama/llama/llama_cpp.pyx`** — same Jinja method as Option B, **plus** removal of the existing `chat_apply_template` method (or its body, replaced with a delegation to the Jinja method). Net: ~60-80 lines added, ~25 lines removed.
4. **`src/cyllama/api.py:_apply_template`** — simpler than Option B because there's no fallback. The whole method becomes a thin wrapper around `chat_apply_jinja_template`. ~30 lines removed, ~5 lines added.
5. **`tests/test_llama_cpp_chat.py`** — same as Option B (~60-80 lines).

Plus the CHANGELOG entry.

### What gets removed

The existing `_apply_template` method has logic for:

- Looking up named templates via `get_default_chat_template_by_name`
- Falling back to a literal template string when the name lookup fails
- Building `LlamaChatMessage` objects to pass to the legacy C API

All of this becomes dead code under Option A. The simpler replacement is approximately:

```python
def _apply_template(self, messages, template=None, add_generation_prompt=True):
    if template is not None:
        # Caller explicitly requested a named template -- not currently
        # supported on the Jinja path. Either re-implement via
        # chat_template_override or document as a removed feature.
        raise NotImplementedError("named template selection requires the legacy path")
    return self.model.chat_apply_jinja_template(messages, add_generation_prompt)
```

Note that the `template=` parameter on `LLM.chat()` (which lets callers force a specific template like `"chatml"` or `"llama3"`) becomes unsupported, or has to be re-implemented by passing the template name as `chat_template_override` to `common_chat_templates_init`. This is a small but real API regression that Option B avoids.

### What stays simpler

- One code path for chat templates: easier to reason about, easier to debug.
- No `try/except RuntimeError` in `_apply_template`, which removes a potential source of confusion (the existing pattern of "swallow runtime errors and try something else" can mask genuine bugs).
- The pipeline-level three-tier fallback in `RAGPipeline._chat_with_fallback` becomes mostly defensive code, but unlike Option B it's the *only* fallback layer (Option B has the binding-level fallback inside `_apply_template` *plus* the pipeline-level fallback).

### When Option A is the right call

- If `minja` proves stable across all the model families that matter (verified by manual end-to-end testing against Llama-3, Qwen, Gemma 2/3/4, Mistral, Phi).
- If the maintenance overhead of two parallel code paths is judged worse than the rare edge cases where the legacy path would have helped.
- If the named-template feature (`LLM.chat(messages, template="chatml")`) is documented as removed or re-implemented via `chat_template_override`.

### When Option B is the right call

- If there's any doubt about minja's stability for a model family that doesn't have a dedicated test in the suite.
- If the `template=` parameter on `LLM.chat()` is in active use (worth grepping the codebase before deciding).
- If the maintenance overhead of dead code is judged less costly than a regression that breaks chat for some users.

## Open questions

1. **`template=` parameter on `LLM.chat()`** — is this used anywhere outside `LLM.chat`'s own signature? If yes, Option A needs a re-implementation strategy. `grep -rn "template=" src/cyllama/ tests/` would answer this.
2. **Build linkage on non-macOS platforms** — `libcommon.a` linking has only been verified on macOS-Metal. Linux (with `whole-archive` linking) and Windows (MSVC) need separate validation. The CI matrix would catch this, but a focused test run on at least one Linux backend before merging is worth the time.
3. **`minja` feature coverage** — `minja` is a Jinja subset, not a full Jinja3 implementation. Are any GGUF chat templates using features `minja` doesn't support? Worth a quick survey of HuggingFace tokenizer configs for the top 10 chat-tuned models.
4. **`common_chat_templates_inputs` fields beyond messages** — the struct has many fields (`tools`, `tool_choice`, `enable_thinking`, `chat_template_kwargs`, etc.). Should the Cython binding expose these? For a first cut, no — leave them at defaults. But `enable_thinking=False` would be a much cleaner way to suppress Qwen3's `<think>` blocks than the current text-stripping approach. Worth considering as a follow-up.

## References

- llama.cpp basic chat-template C API: `build/llama.cpp/src/llama.cpp:1109` (`llama_chat_apply_template`)
- llama.cpp template detection: `build/llama.cpp/src/llama-chat.cpp:88` (`llm_chat_detect_template`) and lines 30-82 for the name table, 95-200+ for the substring heuristics
- llama.cpp Jinja-aware path: `build/llama.cpp/common/chat.h:158-190` (`common_chat_templates_inputs`, `common_chat_params`, function declarations starting at line 221)
- cyllama's current binding: `src/cyllama/llama/llama_cpp.pyx:1948` (`chat_apply_template`)
- cyllama's current Python wrapper: `src/cyllama/api.py:1020` (`_apply_template`)
- cyllama's pipeline-level three-tier fallback: `src/cyllama/rag/pipeline.py:497` (`_generate_chunks`)
- HuggingFace's official Gemma chat template (illustrative — the literal source of the `raise_exception('System role not supported')` clause): the embedded `chat_template` field in the tokenizer config of `google/gemma-3-*-it` and similar releases on HuggingFace
