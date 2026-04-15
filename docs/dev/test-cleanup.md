# Test cleanup requirements for native-backed contexts

> **The rule:** When a test creates an `SDContext`, `LLM`, or `WhisperContext`, explicitly delete it and run `gc.collect()` at the end of the test body. Do not rely on Python's reference-scoping to release native resources between tests.
>
> This applies most critically to `SDContext` (where violating the rule crashes the process on macOS Metal) and as a precaution to `LLM` and `WhisperContext` (where the same class of bug is theoretically possible but has not been observed in practice).

This document is a maintainer guide. It explains:

- The symptom (what goes wrong)
- The root cause (why pytest's default scoping is not enough)
- The fix (what every SD test needs)
- The evidence (5-cycle reproducer)
- Scope (SD is proven, LLM/Whisper are precautionary)
- Related mechanisms and precedent

The design analysis at the end records *why* we chose Python-side forced cleanup over alternatives like pytest-forked or upstream fixes, so future maintainers can decide whether this is still the right tradeoff.

## The symptom

Without explicit cleanup, a full run of `tests/test_sd.py` fails on macOS Metal with one of two manifestations:

1. **`RuntimeError: Image generation failed`** raised from `stable_diffusion.pyx:2064` — SD.cpp's `generate_image()` returned NULL, caught by the v0.2.3 validation guardrail.
2. **`Fatal Python error: Aborted`** — a native `abort()` or `SIGABRT` from inside Metal/ggml before any Python-level validation can run. The screen visibly shakes during the preceding test (macOS window-server compositor stuttering under GPU pressure) before the abort fires.

Both come from the same root cause. The `RuntimeError` is the friendly surface; the abort is what happens when the GPU fails hard enough that the NULL-check never gets to run.

Neither manifestation reproduces when the failing test runs in isolation. Both reproduce deterministically when `tests/test_sd.py` runs in full.

## The root cause

`tests/test_sd.py` has five tests (before `TestConvenienceFunctionsIntegration`) that create an `SDContext`:

1. `TestSDContextIntegration::test_context_creation`
2. `TestSDContextIntegration::test_generate_image`
3. `TestSDContextConcurrencyGuard::test_concurrent_generate_raises`
4. `TestSDContextConcurrencyGuard::test_concurrent_generate_with_params_raises`
5. `TestSDContextConcurrencyGuard::test_lock_release_allows_subsequent_acquire`

Each of them assigns the context to a local variable (`ctx = SDContext(params)`) and lets it fall out of function scope when the test returns. Naive reading: "Python's refcount drops to zero at `return`, `__dealloc__` fires, `free_sd_ctx()` is called, Metal buffers are released."

The reality on pytest:

1. pytest retains the test's stack frame after the test body returns, for use in failure reporting and fixture teardown. Local variables in that frame are not dropped immediately.
2. Python's garbage collector runs on its own heuristic, not synchronously after every function return.
3. On macOS Metal, SD.cpp allocates ~4–6 GB of unified memory per `SDContext`. `SDContext.__dealloc__` calls upstream `free_sd_ctx()`, which calls `~StableDiffusionGGML()`, which calls `ggml_backend_free(backend)`. Metal object lifetime is then handled by ARC — buffers are released *eventually*, not deterministically.

Put it together: during the ~80 non-SD tests that run between `test_lock_release_allows_subsequent_acquire` and `TestConvenienceFunctionsIntegration::test_text_to_image`, up to 5 `SDContext` instances can be simultaneously alive in the Python process, each holding a full set of Metal allocations that haven't been released yet. By the time a fresh `SDContext` is constructed in `test_text_to_image`, the Metal working-set limit is exceeded, SD.cpp's GPU allocator fails to acquire buffers, and `generate_image()` returns NULL (or aborts outright if Metal rejects a command buffer).

The 5-cycle threshold matches the precedent already encoded in `tests/test_memory_leaks.py:164-192`, whose `TestSDContextLeaks` uses `WARMUP_CYCLES = 2 + MEASURE_CYCLES = 3 = 5` total cycles with a 250 MB tolerance window. The existing code comment on that test says:

> SD's allocator pools are large (~4 GB model) and take more than one cycle to stabilize — a single warmup leaves a ~270 MB jump on the first measured cycle that then flattens out completely. Two warmups absorb that settling so the measurement window reflects steady-state.

That test passes because it calls `gc.collect()` between cycles. The crash happens in `test_sd.py` because the 5 tests there did not.

## The fix

Every test that creates a native-backed context must explicitly release it:

```python
def test_something_with_sdcontext(self):
    params = SDContextParams()
    params.model_path = MODEL_PATH
    ctx = SDContext(params)

    # ... test body ...

    # Force immediate SD.cpp cleanup so Metal/ggml state doesn't
    # accumulate across successive SDContext lifecycles in the same
    # process. See tests/test_memory_leaks.py for the same pattern.
    del ctx
    gc.collect()
```

If the test uses `try`/`finally` (e.g. for the concurrency guard tests that hold the busy-lock), put the cleanup *after* the `finally` block:

```python
def test_concurrent_generate_raises(self):
    ctx = self._make_ctx()
    assert ctx._busy_lock.acquire(blocking=False) is True
    try:
        # ... test body that may raise ...
    finally:
        ctx._busy_lock.release()
    del ctx
    gc.collect()
```

Not inside `finally`, because `del` + `gc.collect()` should not run if the test itself failed (it would mask the original exception).

Do not skip `gc.collect()`. `del ctx` alone drops the refcount to zero, but pytest's frame retention can keep *other* references alive (test-class `self`, fixture caches, etc.). `gc.collect()` forces cycle collection and breaks any unintended keepalives.

## Scope: proven for SD, precautionary for LLM/Whisper

**Proven necessary: `SDContext`.** The 5-cycle crash is deterministic on macOS Metal with `sd_xl_turbo_1.0.q8_0.gguf`. This is what the rule was introduced for.

**Precautionary: `LLM` and `WhisperContext`.** The same class of bug is theoretically possible — all three wrappers use the same "native resource held until `__dealloc__` fires" model, and llama.cpp / whisper.cpp allocate GPU buffers on Metal the same way SD.cpp does. We have not observed the crash on LLM or Whisper, likely because:

1. The test LLM (`Llama-3.2-1B-Instruct-Q8_0.gguf`) is ~1.3 GB vs SD's ~6 GB. Even 5 concurrent instances fit comfortably in Metal's working set.
2. `tests/test_memory_leaks.py::TestLLMLeaks` and `TestWhisperContextLeaks` already use the explicit-cleanup pattern, proving it works under stress.

New tests that create multiple `LLM` or `WhisperContext` instances in a single process **should** follow the same `del + gc.collect()` pattern. Existing tests that create only one are fine as-is.

## When you can skip the cleanup

You can skip the forced cleanup when the test only constructs `SDContextParams` / `GenerationConfig` / similar *parameter* objects without instantiating the actual context. Those are pure-Python and don't touch Metal. For example:

```python
def test_vae_path(self):
    params = SDContextParams()
    params.vae_path = "foo.safetensors"
    assert params.vae_path == "foo.safetensors"
    # No cleanup needed — no SDContext was created
```

The rule applies only to tests that instantiate the context class itself.

## Related mechanisms

### `tests/test_memory_leaks.py`

The `TestSDContextLeaks`, `TestLLMLeaks`, and `TestWhisperContextLeaks` classes all use explicit `gc.collect()` between cycles. Their docstrings document the stabilization behavior. They are the canonical precedent for this pattern and existed before the `test_sd.py` crash was diagnosed — that they worked is actually how we knew the pattern was correct.

### `__dealloc__` in `stable_diffusion.pyx:1844-1847`

cyllama's side of the cleanup is minimal and correct:

```cython
def __dealloc__(self):
    if self._ctx != NULL:
        free_sd_ctx(self._ctx)
        self._ctx = NULL
```

The issue is not that `__dealloc__` is broken — it's that Python doesn't call it synchronously at function return under pytest.

### `free_sd_ctx` and `~StableDiffusionGGML` (upstream)

```cpp
// build/stable-diffusion.cpp/src/stable-diffusion.cpp:2394
void free_sd_ctx(sd_ctx_t* sd_ctx) {
    if (sd_ctx->sd != nullptr) {
        delete sd_ctx->sd;  // triggers ~StableDiffusionGGML
        sd_ctx->sd = nullptr;
    }
    free(sd_ctx);
}

// build/stable-diffusion.cpp/src/stable-diffusion.cpp:157-168
~StableDiffusionGGML() {
    if (clip_backend != backend) ggml_backend_free(clip_backend);
    if (control_net_backend != backend) ggml_backend_free(control_net_backend);
    if (vae_backend != backend) ggml_backend_free(vae_backend);
    ggml_backend_free(backend);
}
```

The destructor body frees the raw `ggml_backend_t` pointers first, then C++ member-destructor ordering takes over and destroys the `std::shared_ptr`-managed model wrappers (`cond_stage_model`, `diffusion_model`, etc.). Whether ggml-metal tolerates tensor release through an already-freed backend is not verified — my suspicion is that it either silently orphans Metal buffers or handles it via ARC deferral. Either way, the resource reclamation is not synchronous on the native side even once `__dealloc__` has been called, which compounds the Python-side keepalive problem.

This is potentially a latent upstream issue but we have not confirmed it. The Python-side fix sidesteps the question.

## Design analysis: alternatives considered

### Alternative 1: `pytest-forked` / subprocess per test

Each SD test runs in its own subprocess. No cross-test state possible. Zero risk of recurrence.

**Rejected because:**

- Adds a plugin dependency for one class of tests.
- Subprocess startup is ~1 second per test; `tests/test_sd.py` has 8 tests that touch `SDContext`, so ~8 seconds of overhead on every run.
- Masks the problem rather than addressing it — a future test that needs to create 5 contexts *within* one process would still crash.

### Alternative 2: Skip the affected tests (the short-lived first fix)

Mark `TestSDContextIntegration` and `TestSDContextConcurrencyGuard` with `@pytest.mark.skip`, matching the existing precedent for `test_deterministic_seed:512`.

**Rejected because:**

- Loses coverage for the v0.2.5 `SDContext` concurrent-use guard — the feature whose whole purpose is to prevent silent state corruption on shared contexts. Shipping the guard without its regression tests is worse than the crash.
- Loses the only end-to-end generation smoke test (`test_generate_image`).

### Alternative 3: Reorder tests to put integration tests before the guard tests

Move `TestConvenienceFunctionsIntegration` earlier in source order. The guard tests then run *after* the integration tests and only crash things that don't exist yet.

**Rejected because:**

- Fragile: any future test added between them silently re-breaks the ordering.
- Obscures intent: source-order dependencies are invisible in PR review.
- Doesn't actually fix the bug — it just hides it behind a precarious sequence.

### Alternative 4: Fix SD.cpp's destructor ordering upstream

File an issue on `leejet/stable-diffusion.cpp` explaining the 5-cycle leak and the member-destructor ordering suspicion. Wait for upstream to fix.

**Rejected because:**

- Not verified to be the root cause — the fix might land and not help.
- Uncertain timeline. Multiple SD.cpp Metal bugs have already taken multiple releases to resolve (see v0.2.3 and v0.2.4 CHANGELOG entries).
- Doesn't unblock the current test suite regardless of outcome.

We should still file the upstream issue as follow-up, but not as the primary fix.

### Alternative 5 (chosen): Python-side forced cleanup

Add `del ctx; gc.collect()` to the 5 offending tests. Zero new dependencies, no reordering, no lost coverage, matches existing `test_memory_leaks.py` precedent.

**Known weakness:** it's opt-in at the test author's discretion. A future contributor adding a new SD test will not know to add the cleanup and will re-introduce the crash. This document and the `CLAUDE.md` note exist specifically to mitigate that.

**Alternative mitigation worth considering later:** a pytest fixture that auto-cleans contexts on test teardown. Something like:

```python
@pytest.fixture
def sd_ctx_factory():
    created: list[SDContext] = []
    def make(params):
        ctx = SDContext(params)
        created.append(ctx)
        return ctx
    yield make
    created.clear()
    gc.collect()
```

Not implemented yet. If this pattern keeps biting, the fixture is the next step.

## Known weaknesses of the current fix

1. **Opt-in discipline.** A test author who forgets the cleanup breaks the suite for anyone running the file in full. The only mitigations are documentation (this file + `CLAUDE.md`) and code review.

2. **Platform-specific.** The crash has only been reproduced on macOS Metal. The fix is applied unconditionally because there's no cost on other backends and the pattern is defensible even without the crash (deterministic cleanup is good hygiene). But the *urgency* is Metal-specific; on CUDA/CPU builds the suite may have been silently correct without it.

3. **Does not cover the `generate()`-twice crash.** The existing `@pytest.mark.skip` on `tests/test_sd.py::TestSDContextIntegration::test_deterministic_seed:512` ("Multiple generations on same context causes segfault") is a different failure mode — two `generate()` calls on one context, rather than five context lifecycles. Adding `gc.collect()` between the two generates may or may not help; it's worth trying as a follow-up but is not the scope of this document.

4. **No CI guard.** Nothing fails loudly if a contributor writes a new SD test without the cleanup. A lint rule or a conftest hook could be added, but neither exists today.

## Citable upstream references

- [stable-diffusion.cpp source](https://github.com/leejet/stable-diffusion.cpp/blob/master/stable-diffusion.cpp) — `free_sd_ctx` and `~StableDiffusionGGML`
- [ggml-metal.m](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal.m) — `ggml_backend_metal_free` and Metal buffer lifetime
- cyllama CHANGELOG v0.2.3: "Stable Diffusion generation silently continued on GPU OOM" — the v0.2.3 validation guardrail that turned silent garbage into the `RuntimeError` we now see
- cyllama CHANGELOG v0.2.4: "Metal backend was previously buggy on stable-diffusion.cpp" — the acknowledgment that SD.cpp Metal is historically fragile
- cyllama CHANGELOG v0.2.5: "Memory-leak regression tests" — the `test_memory_leaks.py` file this document cites as precedent

## Revision history

- **2026-04-11**: Initial version. Pattern discovered while diagnosing `pytest tests/test_sd.py` crash on macOS Metal. Fix applied to `test_context_creation`, `test_generate_image`, and the three `TestSDContextConcurrencyGuard` tests.
