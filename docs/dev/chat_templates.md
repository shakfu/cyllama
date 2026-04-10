# Chat Templates in cyllama

This document is a guide to how cyllama handles chat templates: the user-facing API, the layered architecture under it, the vendored Jinja interpreter, the extension points for customising rendering, debugging tools, and the operational procedures (re-vendoring, testing).

A historical-context section at the end preserves the design analysis that led to the current implementation, for maintainers who need to understand *why* the system is shaped this way (e.g. before considering a change to it).

## What this is for

When you call `LLM.chat(messages)` (or its CLI equivalents `cyllama chat -p ...` and `cyllama rag`), cyllama needs to convert the list of `{"role": ..., "content": ...}` dicts into a single prompt string in the exact format the model was instruction-tuned on. Different model families use different formats:

- **Llama-3-Instruct**: `<|start_header_id|>system<|end_header_id|>\n\n...<|eot_id|>...`
- **Qwen2.5/3-Chat**: ChatML-style `<|im_start|>system\n...\n<|im_end|>...`
- **Mistral-Instruct**: `[INST] ... [/INST]`
- **Gemma 2/3**: `<start_of_turn>user\n...\n<end_of_turn>`
- **Gemma 4**: `<|turn>user\n...\n<turn|>` (different from Gemma 2/3, this caused a real bug)
- **Phi-3/4-Instruct**: `<|user|>\n...<|end|>`
- and many others, each with their own quirks

Each GGUF file embeds its own chat template as a Jinja string in its metadata (this is the same template the model's HuggingFace tokenizer config ships with). cyllama's job is to evaluate that template against the user's messages and produce the prompt the model expects.

## Quick reference: how to use it

The Python API surface is small. The most common entry point is `LLM.chat()`:

```python
from cyllama import LLM

llm = LLM("models/Llama-3.2-1B-Instruct-Q8_0.gguf")
response = llm.chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
])
print(response)  # "The capital of France is Paris."
```

`LLM.chat()` takes the standard OpenAI-style message list and handles the rest internally: applies the model's embedded chat template, generates tokens, and returns a `Response` object that prints as a string and exposes `response.stats`.

For lower-level access, `LLM._apply_template()` returns the rendered prompt string without generating anything:

```python
prompt = llm._apply_template([
    {"role": "user", "content": "Hi."},
])
print(prompt)
# <|begin_of_text|><|start_header_id|>user<|end_header_id|>
#
# Hi.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

This is useful for debugging, for inspecting what cyllama is actually sending to the model, or for routing the prompt through a non-standard generation path.

A standalone `apply_chat_template()` function is also exposed at the package level for callers that don't want to construct an `LLM` instance just to render a prompt:

```python
from cyllama.api import apply_chat_template
prompt = apply_chat_template(messages, "models/llama-3.gguf")
```

For RAG usage, the same flow happens automatically inside `RAGPipeline` when `RAGConfig.use_chat_template=True` (which is the default for the `cyllama rag` CLI). See `docs/rag_pipeline.md` for the RAG-specific surface.

## Architecture: the four code paths

When you call `LLM.chat()`, the rendering goes through up to four code paths in priority order. Each path is a fallback for the one above it. Most of the time only the first path runs.

### Path 1: vendored jinja2 (the default)

`LLM._apply_template()` calls `LLM._apply_jinja_template()` first whenever the caller hasn't explicitly requested a named template. This path:

1. Pulls the embedded chat template string from the GGUF metadata via `LlamaModel.get_default_chat_template()` (a one-line wrapper around llama.cpp's stable public C API `llama_model_chat_template`).
2. Resolves the model's `bos_token` and `eos_token` strings via `vocab.token_to_piece(vocab.token_bos(), special=True)` and the same for EOS.
3. Constructs an `ImmutableSandboxedEnvironment` from the vendored `cyllama._vendor.jinja2`, with the same configuration HuggingFace's `transformers.PreTrainedTokenizerBase.apply_chat_template` uses:
    - `trim_blocks=True`, `lstrip_blocks=True`
    - `loopcontrols` extension enabled (some templates use `{% break %}`)
    - `tojson` filter (used by tool-calling templates)
    - `raise_exception` global (used by templates that abort on bad input)
    - `strftime_now` global (used by Llama-3's `Today Date:` line and similar)
4. Renders the template with `messages`, `bos_token`, `eos_token`, and `add_generation_prompt` in scope.
5. Returns the rendered string.

This path handles **any** GGUF whose embedded template uses Jinja syntax, regardless of whether llama.cpp's hardcoded substring heuristics recognise it. Gemma 4 was the canonical case that motivated this — its template uses `<|turn>` markers, which llama.cpp's `llm_chat_detect_template` doesn't match, so the legacy basic-C-API path returned -1 with no diagnostic. The vendored jinja2 just evaluates the actual Jinja and produces the correct prompt.

The implementation is in `src/cyllama/api.py:_apply_jinja_template`. About 80 lines of Python.

### Path 2: legacy substring-heuristic (named templates)

If the caller passes `template="llama3"` (or any named template), `_apply_template` skips the Jinja path entirely and uses `LlamaModel.chat_apply_template(...)`, the Cython binding around llama.cpp's basic `llama_chat_apply_template` C API. This path:

1. Looks up the named template in llama.cpp's `LLM_CHAT_TEMPLATES` map (`build/llama.cpp/src/llama-chat.cpp:30-82`), or treats the string as a raw Jinja template if no name match.
2. If detection fails (the substring heuristics in `llm_chat_detect_template` at lines 88-200+ of the same file), returns -1.
3. cyllama's wrapper raises `RuntimeError("Failed to apply chat template")` on -1.

This path is **only** used when the caller explicitly requests a named template. It's preserved as a fallback for the rare case where `jinja2` itself can't evaluate a template (`TemplateSyntaxError` on a malformed embedded template, etc.) — `_apply_template` will catch the `_JinjaTemplateError` from path 1 and fall through to this path.

The implementation is in `src/cyllama/api.py:_apply_template` (the part after the `try: return self._apply_jinja_template(...)` block).

### Path 3: pipeline-level system-into-user merge

This path only fires inside `RAGPipeline._chat_with_fallback`, not in `LLM._apply_template`. It exists because some chat templates explicitly raise on a `system` role (Gemma 2/3 with the hardcoded path, Orion, some older instruct templates). The flow:

1. RAG pipeline calls `LLM.chat([{"role": "system", ...}, {"role": "user", ...}])`.
2. The chat call propagates a `RuntimeError` whose message contains "template" (from path 1's fallback into path 2 returning -1, or from path 1 raising `TemplateError("System role not supported")`).
3. `_chat_with_fallback` catches that error, merges the system content into the first user message, and retries with `[{"role": "user", "content": "{system}\n\n{user}"}]`.
4. Caches the decision on `self._system_role_supported` so subsequent queries skip the failed first attempt.

The implementation is in `src/cyllama/rag/pipeline.py:_chat_with_fallback`. With the vendored jinja2 path in place, this path almost never fires for currently-supported models (jinja2 evaluates Gemma 4's template directly without raising), but it's preserved for templates that genuinely call `raise_exception('System role not supported')` at the Jinja level.

### Path 4: pipeline-level raw-completion fallback (final safety net)

If both path 1 and path 3 fail, `RAGPipeline._generate_chunks` permanently degrades to the raw-completion path for the rest of the pipeline's lifetime:

1. Catches the `RuntimeError("...template...")` from `_chat_with_fallback`.
2. Sets `self._chat_template_unusable = True` so future queries skip the chat attempts.
3. Rebuilds `gen_config` to include the `Question:/Answer:` stop sequences that the legacy raw-completion path needs (the original `gen_config` had no stop sequences because it was built for the chat path).
4. Renders a `Question:/Context:/Answer:` style prompt and calls the generator directly.
5. Emits a one-time `RuntimeWarning` so the silent quality degradation isn't invisible.

This path is the last-resort safety net. With the vendored jinja2 path in place, it should fire only on truly malformed templates or corrupted GGUFs.

The implementation is in `src/cyllama/rag/pipeline.py:_generate_chunks` (the `except RuntimeError` block).

### Visual summary

```
LLM.chat(messages) ──────────────► LLM._apply_template(messages, template=None)
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │ Path 1: vendored jinja2      │
                       │ (cyllama._vendor.jinja2)     │
                       │ -> embedded GGUF template    │
                       │ -> ImmutableSandboxedEnv     │
                       │ -> rendered prompt string    │
                       └──────────────────────────────┘
                                      │ on TemplateError
                                      ▼
                       ┌──────────────────────────────┐
                       │ Path 2: legacy substring     │
                       │ heuristic (basic C API)      │
                       │ -> llama_chat_apply_template │
                       │ -> rendered prompt string    │
                       └──────────────────────────────┘
                                      │ on RuntimeError("...template...")
                                      │ (only inside RAG pipeline)
                                      ▼
                       ┌──────────────────────────────┐
                       │ Path 3: system->user merge   │
                       │ -> retry path 1 with merged  │
                       │    [{"role": "user", ...}]   │
                       │ -> cached on _system_role_   │
                       │    supported                 │
                       └──────────────────────────────┘
                                      │ on RuntimeError("...template...")
                                      ▼
                       ┌──────────────────────────────┐
                       │ Path 4: raw-completion       │
                       │ -> Question:/Answer: prompt  │
                       │ -> cached on _chat_template_ │
                       │    unusable                  │
                       │ -> one-time RuntimeWarning   │
                       └──────────────────────────────┘
```

Most calls only execute path 1. Paths 2-4 exist as defense-in-depth so an unexpected template never crashes the user's REPL.

## The vendored jinja2 setup

cyllama ships pure-Python copies of `jinja2 3.1.6` and `markupsafe 3.0.3` under `src/cyllama/_vendor/`. They are loaded under the `cyllama._vendor.*` namespace so they cannot collide with whatever versions the user has installed in their own environment.

### What's in `src/cyllama/_vendor/`

```
src/cyllama/_vendor/
├── __init__.py              # package marker, vendoring policy docstring
├── README.md                # operational docs (versions, rewrites, do-not-modify)
├── LICENSE.jinja2           # BSD-3 from jinja2
├── LICENSE.markupsafe       # BSD-3 from markupsafe
├── jinja2/                  # 26 .py files, ~14,425 lines
│   ├── __init__.py
│   ├── compiler.py          # *** has one critical patch (see below) ***
│   ├── environment.py
│   ├── runtime.py
│   ├── sandbox.py           # ImmutableSandboxedEnvironment lives here
│   └── ... (22 more)
└── markupsafe/              # 3 files, ~404 lines (no _speedups.c)
    ├── __init__.py
    ├── _native.py           # pure-Python escape fallback
    └── py.typed
```

Total: ~14,800 lines of pure Python, ~616 KB on disk. Negligible relative to cyllama's 30-90 MB compiled wheels.

### Why vendored rather than a runtime dependency

cyllama is distributed as binary wheels with bundled native libraries (libllama, libggml, libwhisper, libsd). The package philosophy is "everything you need is in the wheel." Adding ~600 KB of vendored Python to fit that model is a rounding error, and the alternative — declaring the deps in `pyproject.toml` and letting pip resolve them — would:

1. **Introduce a transitive dependency** that can conflict with the user's environment.
2. **Weaken reproducibility**: different `jinja2` versions can render the same chat template differently, so two users running the same cyllama version could see different outputs.
3. **Break the self-contained-wheel invariant** that cyllama otherwise maintains.

The trade-off is that security patches for `jinja2` require a cyllama release rather than a `pip install --upgrade jinja2`. Mitigating factors: `jinja2`'s security history is short and almost entirely about evaluating *untrusted* templates, which cyllama doesn't do. cyllama evaluates GGUF metadata templates inside `ImmutableSandboxedEnvironment`, which is the safe-evaluation mode explicitly designed for untrusted input. The realistic security exposure is low.

The full trade-off analysis (including the alternatives that were considered and rejected) is in the historical context section at the end of this document.

### The two import-path rewrites

When jinja2 is installed via pip into a normal `site-packages` location, all of its internal cross-module imports work because they find each other through the standard Python import machinery. When you copy the same files into `src/cyllama/_vendor/jinja2/`, two specific import patterns break:

#### Rewrite 1: `from markupsafe import` → `from cyllama._vendor.markupsafe import`

`jinja2` does `from markupsafe import escape, Markup, soft_str` and `import markupsafe` across 9 of its source files. Without rewriting, those imports would resolve to whatever `markupsafe` is on the user's `sys.path` (or fail with `ImportError` if none is installed). The rewrite changes them to point at `cyllama._vendor.markupsafe`.

This is the standard `pip._vendor` / `setuptools._vendor` pattern.

#### Rewrite 2: `from jinja2.runtime import` → `from cyllama._vendor.jinja2.runtime import`

This one is more subtle and is the rewrite that took longest to discover.

`jinja2`'s `compiler.py` emits a literal `from jinja2.runtime import ...` line into the Python source of every *compiled template*. This is not at module load time — it's at template-compile time, written into the compiled bytecode. Every template the vendored jinja2 compiles contains this line.

Without the rewrite, every compiled template tries to import from the top-level `jinja2.runtime` module:

- **If the user has `jinja2` installed in their environment**, compiled templates from the vendored jinja2 silently pull runtime symbols from the *user's* jinja2, mixing two versions in undefined ways. The observable failure mode is `_MissingType` sentinels leaking into rendered output as the literal string `"missing"` instead of being converted to `Undefined` — because the cross-module identity check `if rv is missing:` between the two `missing` sentinels fails, and the sentinel propagates as a string.
- **If the user has no jinja2 installed**, the import fails entirely and template rendering raises `ImportError`.

The rewrite changes the emit string to `from cyllama._vendor.jinja2.runtime import ...` so compiled templates always pull from the vendored runtime. The patch lives in `src/cyllama/_vendor/jinja2/compiler.py` at the line that calls `self.writeline(...)` to emit the import block.

Both rewrites are scripted in `scripts/vendor_jinja2.sh` and applied automatically on every re-vendor. The script also verifies that no stray non-rewritten imports remain after sed runs, and fails loudly if it finds any. See "Re-vendoring" below for the full procedure.

### Re-vendoring jinja2 / markupsafe

To update the vendored libraries to a newer version:

```bash
./scripts/vendor_jinja2.sh [JINJA2_VERSION] [MARKUPSAFE_VERSION]
```

Both arguments are optional; defaults are pinned in the script. The script:

1. Downloads the requested wheels via `pip download --no-deps`.
2. Extracts the `.py` source files into `src/cyllama/_vendor/`.
3. Removes `markupsafe`'s optional `_speedups.c` C extension and any compiled `.so` artifacts (we use the pure-Python `_native.py` path; HTML escaping is irrelevant for chat-template rendering and vendoring the C extension would reintroduce build complexity).
4. Re-runs both import-path rewrites against the new sources.
5. Copies fresh `LICENSE.txt` files from each wheel's `dist-info/licenses/` into the vendor directory.
6. Verifies no stray un-rewritten imports remain, failing loudly if any are found.

After running it:

1. Run the chat-template tests: `uv run pytest tests/test_jinja_chat.py tests/test_chat.py -v`. The 15 tests in `test_jinja_chat.py` are designed to catch most regressions (the `_MissingType` leak, the `set` inside `if not x is defined` pattern, the loopcontrols extension, etc.).
2. Update the version table in `src/cyllama/_vendor/README.md`.
3. Inspect the diff. Expected changes: only files inside `_vendor/` should be touched. The two rewrites should appear as small line-level changes; everything else should be the upstream content.
4. Commit.

You should re-vendor maybe 2-3 times a year — `jinja2` ships at that pace and is very stable. Outside of security advisories, there is rarely an urgent reason to update.

## Extending the renderer

The Jinja environment in `LLM._apply_jinja_template` is configured with the same globals and filters HuggingFace's `apply_chat_template` provides. If a model's embedded template references something that isn't currently provided, you have two extension points.

### Adding a custom global

A "global" is a function or value the template can reference directly, like `raise_exception('...')` or `strftime_now('%Y')`. To add one, edit `LLM._apply_jinja_template` in `src/cyllama/api.py`:

```python
def _apply_jinja_template(self, messages, add_generation_prompt=True):
    # ... existing setup ...

    def my_custom_function(arg):
        return some_computed_value

    env.globals["my_custom_function"] = my_custom_function

    # ... existing render call ...
```

The function is then accessible inside the template as `{{ my_custom_function('arg') }}`. Keep the function pure-Python and side-effect-free where possible — templates run inside `ImmutableSandboxedEnvironment`, which restricts attribute access and method calls for security, but globals run unrestricted.

### Adding a custom filter

A "filter" is applied to a value with the pipe operator, like `{{ messages | tojson }}` or `{{ name | trim }}`. The current renderer provides `tojson` (used by tool-calling templates). To add another:

```python
def my_filter(value, *args, **kwargs):
    return some_transformation(value)

env.filters["my_filter"] = my_filter
```

Then templates can do `{{ value | my_filter }}` or `{{ value | my_filter('arg') }}`.

### Suppressing reasoning blocks at the template level

Some reasoning-tuned models (Qwen3, DeepSeek-R1) emit a `<think>...</think>` block before their actual answer. The current cyllama implementation strips these blocks at the *streaming output* level via `cyllama.rag.repetition.ThinkBlockStripper`, but a cleaner approach is to suppress them at the *template* level by passing `enable_thinking=False` as a render kwarg.

Qwen3's chat template (and some others) accepts an `enable_thinking` template variable that, when False, skips the `<think>` block generation entirely. This is much more efficient than letting the model generate the block and then discarding it — it saves both tokens *and* generation time.

To wire this up, add a render kwarg to `_apply_jinja_template`:

```python
return compiled.render(
    messages=messages,
    bos_token=bos_token,
    eos_token=eos_token,
    add_generation_prompt=add_generation_prompt,
    enable_thinking=False,  # NEW
)
```

Templates that don't use `enable_thinking` will silently ignore the kwarg (Jinja2 doesn't error on extra context variables). Templates that do use it will skip the reasoning block.

This is documented as an open follow-up: a survey of how widely the `enable_thinking` convention is adopted across reasoning-tuned models would tell us whether to make this the default, expose it via `RAGConfig`, or leave it manual. Until then, the streaming `ThinkBlockStripper` remains the cyllama default for reasoning suppression.

## Debugging

### Inspecting the rendered prompt

The fastest way to see what cyllama is sending to the model is to call `_apply_template` directly and print the result:

```python
from cyllama import LLM

llm = LLM("models/your-model.gguf", verbose=False)
prompt = llm._apply_template([
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hi."},
])
print(repr(prompt))
```

`repr()` shows escape characters explicitly (`\n`, `\t`) so you can see the exact byte sequence. The prompt should:

1. Begin with the model's BOS token (or include it after a Jinja-rendered preamble).
2. Have the system message in the model's expected role marker.
3. Have each turn properly delimited.
4. End with the assistant role marker and `add_generation_prompt=True`'s trailing prefix.

### Inspecting the embedded template itself

If the rendered prompt looks wrong, dump the raw template to see what Jinja source cyllama is evaluating:

```python
from cyllama import LLM

llm = LLM("models/your-model.gguf", verbose=False)
template_str = llm.model.get_default_chat_template()
for i, line in enumerate(template_str.splitlines(), 1):
    print(f"{i:3d}: {line}")
```

This is the literal string from the GGUF metadata. If it's empty, the model has no embedded template and `_apply_jinja_template` will raise `TemplateError("Model has no embedded chat template")`, falling through to the legacy path which will use a built-in default if the model name matches one.

### Checking which path was taken

Currently there's no explicit per-call logging of which code path rendered the template. If you need this for diagnostics, the cleanest way is to instrument `_apply_template` temporarily:

```python
def _apply_template(self, messages, template=None, add_generation_prompt=True):
    if template is None:
        try:
            result = self._apply_jinja_template(messages, add_generation_prompt)
            print("[chat template] used vendored jinja2 path", file=sys.stderr)
            return result
        except _JinjaTemplateError:
            print("[chat template] jinja2 path raised, falling back", file=sys.stderr)
            pass
    # ... legacy path ...
    print("[chat template] used legacy substring-heuristic path", file=sys.stderr)
    return self.model.chat_apply_template(...)
```

For the RAG pipeline's three-tier fallback, set a breakpoint or logging in `RAGPipeline._chat_with_fallback` and `RAGPipeline._generate_chunks`. The `_system_role_supported` and `_chat_template_unusable` instance attributes record which fallbacks have fired:

```python
print(f"system role supported: {pipeline._system_role_supported}")
print(f"chat template unusable: {pipeline._chat_template_unusable}")
```

`None` means "haven't tried yet"; `True`/`False` means "the corresponding probe has fired and cached its result".

### Common failure modes

1. **`RuntimeError: Failed to apply chat template`** at the binding level (`llama_cpp.pyx:1948`). This means the legacy substring-heuristic path was reached and llama.cpp's `llama_chat_apply_template` returned -1. Either:
    - The Jinja path raised first and the wrapper fell through (look for the cause in the Jinja path)
    - The caller passed a `template=` parameter that doesn't match anything in `LLM_CHAT_TEMPLATES` and isn't a recognisable Jinja string

2. **`TemplateError: Model has no embedded chat template`** from `_apply_jinja_template`. The GGUF metadata doesn't contain a chat template at all. This is normal for embedding-only models; for chat-tuned models it usually means the GGUF was built without preserving the tokenizer config. The wrapper will fall through to the legacy path, which has its own `_format_messages_simple` fallback.

3. **`TemplateError` with a model-specific message** like "System role not supported" (Gemma 2/3 with their `raise_exception` calls). The Jinja path is correctly evaluating the template and the template is intentionally raising. In `LLM.chat()` direct usage this propagates to the caller; in `RAGPipeline` usage the system-into-user merge fallback (path 3) catches it and retries.

4. **`_MissingType` sentinel literal `"missing"` appearing in rendered output.** This means the vendored jinja2's `compiler.py` is emitting compiled templates with `from jinja2.runtime import ...` instead of `from cyllama._vendor.jinja2.runtime import ...`. The rewrite was either skipped during re-vendoring or got reverted. Re-run `scripts/vendor_jinja2.sh` to fix.

5. **`{% set %}` inside an `if` block doesn't escape the block** for templates that use the `{% if not x is defined %}{% set x = ... %}{% endif %}` pattern (real Llama-3 templates do this). This is a known Jinja2 quirk that depends on the variable being checked vs. the variable being set. There's a regression test for it in `tests/test_jinja_chat.py::TestVendoredJinja2::test_set_inside_is_defined_if_escapes_block`. If this test starts failing after a re-vendor, the upstream Jinja2 changed its scope behaviour and the templates may need a workaround.

### Testing against a specific model

`tests/test_jinja_chat.py` is the canonical test suite for the chat-template renderer. To run it against a model that isn't in the standard fixtures:

```python
# tests/test_jinja_chat.py already has a TestGemma4Regression class
# that follows this pattern; copy it for new models.

import pytest
from pathlib import Path

MY_MODEL = Path(__file__).parent.parent / "models" / "your-model.gguf"

@pytest.mark.skipif(not MY_MODEL.exists(), reason=f"model not found at {MY_MODEL}")
class TestMyModel:
    def test_jinja_path_renders_template(self):
        from cyllama.api import LLM
        llm = LLM(str(MY_MODEL), verbose=False)
        try:
            prompt = llm._apply_jinja_template([
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi."},
            ])
            assert "expected_marker" in prompt
        finally:
            llm.close()

    def test_chat_method_works_end_to_end(self):
        from cyllama.api import LLM, GenerationConfig
        llm = LLM(
            str(MY_MODEL),
            config=GenerationConfig(max_tokens=50, temperature=0.0),
            verbose=False,
        )
        try:
            response = llm.chat([
                {"role": "user", "content": "What is 2+2?"},
            ])
            assert "4" in str(response)
        finally:
            llm.close()
```

The `try: ... finally: llm.close()` pattern is important — it prevents two large model contexts from briefly overlapping on the GPU between tests, which can cause flaky `RuntimeError: llama_decode failed` errors with large models.

## Reference

| File | Purpose |
|---|---|
| `src/cyllama/api.py:_apply_template` | Main wrapper, tries jinja2 path then legacy fallback |
| `src/cyllama/api.py:_apply_jinja_template` | The vendored-jinja2 rendering implementation |
| `src/cyllama/_vendor/jinja2/` | Vendored jinja2 3.1.6, with two import-path rewrites applied |
| `src/cyllama/_vendor/markupsafe/` | Vendored markupsafe 3.0.3, no C speedups |
| `src/cyllama/_vendor/README.md` | Operational docs for the vendor directory |
| `src/cyllama/_vendor/__init__.py` | Vendoring policy docstring |
| `src/cyllama/_vendor/jinja2/compiler.py:843` | The hardcoded `from cyllama._vendor.jinja2.runtime import` line (rewrite target) |
| `scripts/vendor_jinja2.sh` | Re-vendoring script with the two sed rewrites |
| `src/cyllama/llama/llama_cpp.pyx:get_default_chat_template` | Cython binding for `llama_model_chat_template` (extracts embedded template string) |
| `src/cyllama/llama/llama_cpp.pyx:chat_apply_template` | Cython binding for the legacy `llama_chat_apply_template` C API |
| `src/cyllama/rag/pipeline.py:_chat_with_fallback` | RAG pipeline's path-3 system-merge fallback |
| `src/cyllama/rag/pipeline.py:_generate_chunks` | RAG pipeline's path-4 raw-completion final safety net |
| `tests/test_jinja_chat.py` | 15 tests covering the vendored renderer end-to-end |
| `tests/test_chat.py` | Existing chat template tests against the legacy path |
| `tests/test_rag_pipeline.py` | Pipeline-level fallback tests (paths 3 and 4) |

llama.cpp internals (for understanding *why* the legacy path is limited):

| File | Purpose |
|---|---|
| `build/llama.cpp/src/llama-chat.cpp:30-82` | `LLM_CHAT_TEMPLATES` name -> enum map |
| `build/llama.cpp/src/llama-chat.cpp:88` | `llm_chat_detect_template` substring heuristics (lines 95-200+) |
| `build/llama.cpp/src/llama.cpp:1109` | `llama_chat_apply_template` public C API |
| `build/llama.cpp/common/chat.h` | The Jinja-aware path in `libcommon.a` (NOT linked by cyllama — see historical context) |

## Historical context

This section preserves the design analysis that led to the current implementation. It exists for maintainers who need to understand *why* the system is shaped this way before considering a change to it. None of this is operationally necessary for using or extending the current renderer.

### The basic C API limitation

llama.cpp exposes two chat-template APIs:

1. `llama_chat_apply_template` in the public C API (`include/llama.h`). This is what cyllama's Cython binding originally called. It only handles a hardcoded set of templates detected via substring heuristics on the embedded template string. If the substring detection fails, it returns -1 with no diagnostic.

2. `common_chat_templates_apply` in `common/chat.cpp`. This uses llama.cpp's internal Jinja interpreter (`common/jinja/`) to evaluate the template directly, regardless of substring matching.

The basic C API was the original cyllama binding because it's part of llama.cpp's stable public API surface. The problem was that any model whose embedded Jinja template doesn't match one of the hardcoded substring heuristics fails entirely. **Gemma 4** was the canonical case: its template uses `<|turn>` markers, but llama.cpp's heuristic looks for `<start_of_turn>` (which Gemma 2/3 use). The heuristic returned `LLM_CHAT_TEMPLATE_UNKNOWN` and the function returned -1 regardless of message shape.

### Why we don't link `libcommon.a`

The obvious fix would be to call `common_chat_templates_apply` from cyllama's Cython binding. The headers and the static library are already installed by `manage.py` at build time. Linking would be a few lines of CMake.

The reason we don't is that **`libcommon.a` is not part of llama.cpp's stable API**. It's the example/utility library that backs `llama-server`, `llama-cli`, etc. Function signatures, struct layouts (especially `common_chat_templates_inputs` and `common_chat_params`, which gain fields per release), and even header file paths change between llama.cpp versions without notice. Linking against it would mean:

- The cyllama build breaks on most llama.cpp upgrades
- Cython `.pxd` declarations have to be re-audited against `chat.h` after every upgrade
- Subtle ABI mismatches could produce silent corruption: Cython trusts the declaration, so if a struct field has shifted, reads return garbage without a compile error
- The Cython binding work has to be redone whenever upstream restructures `chat.cpp`

This is a large recurring maintenance burden for a behaviour we can get a different way.

### Why we vendor jinja2 instead

Once `libcommon.a` was off the table, the next question was how to evaluate Jinja templates in a way that works regardless of llama.cpp's hardcoded substring detection. The options were:

- **Add `jinja2` as a runtime dependency** (Option D in the original analysis): standard Python idiom, smallest implementation, but introduces a transitive dep that conflicts with the self-contained-wheel philosophy and weakens reproducibility (different `jinja2` versions can render the same template differently).
- **Vendor a C++ Jinja interpreter** (Option E): copy `minja` or llama.cpp's `common/jinja/` into cyllama's source tree and compile it as part of the extension. Eliminates the dependency, gives byte-for-byte parity with `llama-server`, but requires substantial Cython binding work plus a build matrix that has to compile vendored C++ across macOS, Linux, and Windows with multiple GPU backends. Estimated 4-8 hours of focused work plus per-iteration build cycles.
- **Vendor `jinja2` itself as pure Python** (Option F, what we shipped): copy `jinja2 3.1.6` and `markupsafe 3.0.3` under `cyllama._vendor.*`. ~616 KB pure Python, no Cython binding, no build matrix risk, no transitive deps. Same Jinja semantics as HuggingFace's `apply_chat_template` because it IS the same library. ~1.5 hours of focused work.

Option F won because the cost-benefit profile dominated the alternatives for cyllama specifically:

- cyllama already distributes binary wheels with bundled native libraries (50+ MB), so adding 600 KB of vendored Python is a rounding error
- Reproducibility matters more for chat-template rendering than for typical Python deps — the entire job of this code path is "produce a deterministic prompt string from a model's metadata"
- jinja2 is mature, BSD-3-licensed, and ships 2-3 minor releases per year, so the manual re-vendoring burden is small
- The security exposure is low because we evaluate trusted GGUF metadata inside `ImmutableSandboxedEnvironment`

### Why we kept the legacy substring path as a fallback

After Option F shipped, the legacy `chat_apply_template` Cython binding became mostly dead code — it only fires now if `_apply_jinja_template` raises (`TemplateError`, malformed template, no embedded template) or if the caller explicitly requests a named template via `LLM.chat(messages, template="llama3")`.

It's preserved as a safety net rather than removed because:

1. It costs almost nothing to keep (it's already in the Cython binding)
2. It handles the named-template case that the Jinja path doesn't currently support (the Jinja path requires an embedded template, not a name)
3. It's a defensive layer for the rare cases where the upstream `jinja2` version we vendored has a bug evaluating some specific template
4. If the binding ever breaks (e.g. a llama.cpp upgrade changes `chat_apply_template`'s signature), it would be caught by `tests/test_chat.py` rather than silently leaving the named-template feature broken

If the legacy path is empirically dead-code over many model families and many releases, it can be removed in a follow-up. Until then, it's cheap insurance.

### Why the pipeline-level fallback layers exist

The three pipeline-level fallback layers (`_chat_with_fallback` for system-role merging, `_generate_chunks` for raw-completion degradation) were added before the vendored-jinja2 path existed, when Gemma 4 was crashing the RAG pipeline immediately on the first query. They were the user-facing fix that shipped first; the vendored-jinja2 path is the proper underlying fix that came later.

With the vendored jinja2 path in place, paths 3 and 4 fire much less often:

- **Path 3 (system-into-user merge)** still fires on Gemma 2/3 templates that explicitly call `raise_exception('System role not supported')` at the Jinja level. Both the legacy path AND the vendored Jinja path correctly propagate this exception, so the merge fallback is still load-bearing for those models.
- **Path 4 (raw-completion degradation)** fires only if BOTH chat shapes (system+user, merged user) raise template errors. With the vendored Jinja path correctly evaluating any well-formed Jinja template, this should now only fire on truly malformed embedded templates or corrupted GGUFs.

Both paths are preserved as defense-in-depth. The maintenance cost is low (a handful of lines of Python plus the cached state on `RAGPipeline`) and the correctness benefit is real for the long tail of models we don't have in our test fixtures.

### Open follow-ups

These are documented for future maintainers who may want to pick them up. None are urgent.

1. **`enable_thinking=False` for Qwen3 and similar reasoning models.** Qwen3's HuggingFace template supports an `enable_thinking` template kwarg that, when False, suppresses the `<think>` block at the *template* level — much cleaner than the current text-stripping `ThinkBlockStripper` because it saves both tokens *and* generation time. A survey of how widely the convention is adopted across reasoning-tuned models would tell us whether to make this the default, expose it via `RAGConfig`, or leave it manual. If adoption is broad enough, the `ThinkBlockStripper` could potentially be deleted.

2. **Removing the legacy substring-heuristic Cython binding.** Once we have empirical confidence that the vendored Jinja path covers every model family in cyllama's user base, the legacy `chat_apply_template` method on `LlamaModel` and the `_apply_template` fallback to it become dead code. Removing them would simplify the codebase by ~50 lines and eliminate one path that has to be maintained.

3. **Removing the pipeline-level path-4 raw-completion fallback.** Same reasoning. With Jinja support in place, this should never fire in practice. Removing it would simplify `RAGPipeline._generate_chunks` and the warning message that ships with it.

4. **Public API for `_apply_jinja_template`.** Currently it's a private method with the underscore prefix. If users want to render a chat template without going through `LLM.chat()` or the standalone `apply_chat_template()` function, they'd benefit from a public method. This is purely a naming/documentation change.

5. **Linux/Windows verification of the vendored path.** All testing has been done on macOS-Metal. The vendored path is pure Python so it should work identically on Linux and Windows, but a CI run against at least one Linux backend would confirm.

6. **Survey of which embedded templates use which Jinja features.** Some templates use advanced features (`{% include %}`, `{% extends %}`, custom tests) that we haven't exercised. A scripted survey across the top-N chat-tuned GGUFs on HuggingFace would tell us whether the current renderer covers everything in practice or whether some templates need additional extensions.
