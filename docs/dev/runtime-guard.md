# Concurrent-use runtime guard

> **The rule:** Always use one context per thread. Do not share the same context across threads.
>
> This applies to every `LLM`, `WhisperContext`, and `SDContext` instance. The underlying llama.cpp / whisper.cpp / stable-diffusion.cpp contexts are not designed for concurrent access — sharing one across threads corrupts internal state silently. cyllama enforces the rule at runtime with a non-blocking lock; the rest of this document explains the hazard, the implementation, the alternatives we rejected, and how to extend the guard when adding new native-touching methods.

This document is a maintainer guide. It covers:

- The underlying hazard and why it exists
- The current implementation
- The alternative designs that were considered and rejected
- Known weaknesses of the current design
- When and how to extend the guard if a new native-touching method is added

The design analysis at the end is preserved so future maintainers can decide whether to keep, replace, or extend the guard based on first principles, not by reading the code and guessing.

## What this is for

llama.cpp's `llama_context`, whisper.cpp's `whisper_context`, and stable-diffusion.cpp's `sd_ctx_t` are **not safe to use from multiple threads simultaneously**. The data structures rely on single-owner access: the KV cache, sampler state, batch buffers, and RNG state are all touched without locking on every token. Two concurrent native calls into the same context produce silent corruption (garbage tokens, scrambled state) or segfaults under load.

How well each upstream documents this varies — and the variation itself is part of why cyllama needs a Python-level guard rather than relying on users to read C headers:

**whisper.cpp** is explicit and unambiguous. Three places in `thirdparty/whisper.cpp/include/whisper.h`:

```c
// thirdparty/whisper.cpp/include/whisper.h:45-46
// The following interface is thread-safe as long as the same whisper_context is not used by multiple threads
// concurrently.
```

```c
// thirdparty/whisper.cpp/include/whisper.h:600-603
// Run the entire model: PCM -> log mel spectrogram -> encoder -> decoder -> text
// Not thread safe for same context
// Uses the specified decoding strategy to obtain the text.
WHISPER_API int whisper_full(...);
```

```c
// thirdparty/whisper.cpp/include/whisper.h:616-621
// Split the input audio in chunks and process each chunk separately using whisper_full_with_state()
// Result is stored in the default state of the context
// Not thread safe if executed in parallel on the same context.
WHISPER_API int whisper_full_parallel(...);
```

So for whisper, the rule is stated in the header: do not call `whisper_full`/`whisper_full_parallel` on the same `whisper_context` from multiple threads. cyllama's `WhisperContext._busy_lock` enforces exactly this.

**llama.cpp** is **not** explicit about `llama_context` thread safety in the C header. `thirdparty/llama.cpp/include/llama.h` has thread-safety annotations only on two scoped subsystems:

- The Tokenization section (line 1120) is marked safe: `// The API is thread-safe.` — but the comment is positioned just above the `llama_tokenize` family and applies only to that section, not to the whole API.
- Global logger state is marked unsafe at lines 521 and 1511: `// this function is NOT thread safe because it modifies the global llama logger state` and similar.

The actually-dangerous functions — `llama_decode`, `llama_encode`, `llama_sampler_sample`, `llama_set_*` — have **no thread-safety annotation either way**. Their docstrings document return codes and memory-state behaviour but not thread safety. The `ggml.h` "Multi-threading" section that you'd hope would clarify things is literally `// TODO`:

```c
// thirdparty/llama.cpp/include/ggml.h:155-158
// ## Multi-threading
//
// TODO
```

The headers are silent, but the upstream issue tracker is **explicit** about the rule, and the rule exactly matches what cyllama enforces. Three references that together establish the contract:

1. **[ggml-org/llama.cpp#499](https://github.com/ggml-org/llama.cpp/issues/499)** — *"Thread safety"* (SpeedyCraftah, March 2023). The original question: *"Is llama.cpp thread safe? I have encountered some problems and weird issues when creating a CTX on another thread and then using it in another."* Answered by Ronsor with the canonical *"not yet, but it's a priority and parallel inference is on the roadmap."* Crucially, **slaren** (Diego Devesa, llama.cpp maintainer) clarified that the design goal was thread-safe usage of **different `llama_context` objects**, not concurrent use of the same one — *"llama.cpp shouldn't be too far from being thread safe over different `llama_context` objects."* And **ggerganov** noted that the supported parallelism story is "multiple models, multiple contexts, and parallel decoding [via batching]" — *not* sharing a context across threads. So as of March 2023, the rule was already on record: one context per thread is the supported pattern, sharing one context across threads is not.

2. **[ggml-org/llama.cpp#3960](https://github.com/ggml-org/llama.cpp/issues/3960)** — *"ggml : become thread-safe"* (opened by ggerganov himself, November 2023). The fact that the project's lead author opened this issue is itself the most authoritative statement possible that ggml wasn't thread-safe at the time. The body: *"We should be able to run inference on multiple graphs, backends and devices in parallel. Currently, there are CUDA singletons that break this requirement and possibly there could be other problems."* Note the framing: the goal is parallel inference across **multiple graphs** (i.e. multiple contexts), not concurrent access to one graph. After the fix landed, the same issue thread states the post-fix contract in plain terms:

   > *"For example, using multiple llama contexts simultaneously each with a different CUDA GPU on different threads should now be possible. CPU and Metal also should be thread-safe, other backends probably not."*

   Three things to note about this quote, because it is the closest thing to an explicit, citable upstream statement of the rule:

   - The unit of parallelism is **"multiple llama contexts"** (plural). The supported pattern is one context per worker, never one context shared between workers.
   - Backend safety varies. **CPU and Metal** are thread-safe (across contexts). **CUDA** is thread-safe in the multi-GPU/multi-context case. **"Other backends probably not"** — Vulkan, SYCL, HIP, OpenCL are explicitly *not* claimed to be thread-safe, even across contexts. cyllama users on those backends should treat the runtime guard as the only safety net.
   - The hedge ("should now be possible", "probably not") is the maintainer's own. Even within the supported pattern, upstream is not making a strong guarantee.

3. **[ggml-org/llama.cpp#6170](https://github.com/ggml-org/llama.cpp/pull/6170)** — *"cuda : refactor to remove global resources"* (slaren, merged March 2024). This is the PR that closed #3960. Its scope is narrower than the issue title implies: it fixes the CUDA backend singletons that prevented even the per-context case from working. The PR description: *"Pools and other resources are tied to the `ggml_backend` instance and are freed along with it. It should also be thread-safe."* Note the hedge ("should") and the scope (CUDA backend resources, not `llama_context` state).

The crucial inference: **closing #3960 does not mean "you can now share one `llama_context` across threads."** It means "the CUDA singletons that broke per-context isolation are gone, so one-context-per-thread actually works now." The KV cache, sampler state, and batch buffers inside a single `llama_context` are still single-owner data structures touched on every token without locks — there's no PR claiming to have changed that, because it was never the goal.

So the upstream design contract for llama.cpp, established across these three threads, is exactly the rule cyllama's runtime guard enforces:

> **Always use one context per thread. Do not share the same context across threads.**
>
> Concretely: one `llama_context` is safe to use from one thread at a time. Multiple `llama_context` objects (one per worker thread) can run in parallel, subject to backend caveats noted above. Sharing one `llama_context` across multiple concurrent native calls is not supported and never has been.

The headers don't say this — but the issue tracker does, and ggerganov / slaren are the authoritative sources for "what llama.cpp supports." cyllama's `LLM._busy_lock` prevents the unsupported pattern (one context, multiple concurrent users) while allowing the legitimate sequential-handoff pattern (one context, one in-flight call, possibly bounced between threads via `asyncio.to_thread`).

**stable-diffusion.cpp** is also **not** explicit. `thirdparty/stable-diffusion.cpp/include/stable-diffusion.h` has no thread-safety statements about `sd_ctx_t`, `generate_image`, or `generate_video`. The only "thread-safe" mentions in the include directory are in unrelated bundled libraries (`stb_image.h`, `ggml-cann.h`). The non-thread-safety is again conventional knowledge from the implementation rather than a documented contract.

Net: of the three upstreams, only whisper.cpp documents the rule. For llama.cpp and sd.cpp, "concurrent use of one context corrupts state" is true but not stated where users would find it. The runtime guard is doing real informational work, not just enforcing something the user could have looked up.

cyllama makes this slightly worse than vanilla Python because **it releases the GIL around native calls** to allow other Python threads to run during inference. This is a deliberate, documented win for throughput in async server contexts (`AsyncLLM`, `PythonServer`, `AsyncReActAgent` all rely on it). The downside: two Python threads holding references to the same `LLM` can now actually execute native code in parallel, instead of being serialized by the GIL by accident. So:

```python
llm = LLM("model.gguf")

def worker():
    llm("hello")  # native call, GIL released

t1 = threading.Thread(target=worker)
t2 = threading.Thread(target=worker)
t1.start(); t2.start()   # ← two native calls into one context, in parallel
```

What you got from this before the guard was **silent corruption**: garbage tokens, scrambled KV cache, occasional segfaults under load, or — worst case — a clean run that returned nonsense the user trusted. None of these surface as a clear "you did it wrong" error. Users blamed the model.

The runtime guard converts this silent UB into a loud `RuntimeError` on the *first* concurrent call.

## Quick reference: how it presents to users

> **Always use one context per thread. Do not share the same context across threads.**

That is the entire rule for users. The runtime guard exists to enforce it loudly when it's violated, not to enable any clever pattern around it. If you find yourself wanting to share an `LLM` (or `WhisperContext`, or `SDContext`) across worker threads, the answer is always *create one per worker* — even if it costs extra memory.

When two threads try to use the same instance concurrently, the second one raises:

```text
RuntimeError: LLM is currently being used by another thread. llama.cpp
contexts are not thread-safe — create one LLM per thread instead of
sharing a single instance across threads.
```

Equivalent messages exist for `WhisperContext` and `SDContext`. The fix is always the same: create one instance per thread (or per worker), don't share.

### One narrow exception: sequential ownership transfer

The rule "one context per thread" is about *concurrent* use. There is one pattern that *looks* like cross-thread use but isn't, and it must keep working:

```python
# asyncio.to_thread / executor handoff: LLM created on the main thread,
# called sequentially on a worker thread. No contention, no error.
llm = LLM("model.gguf")
result = await asyncio.to_thread(llm, "hello")
```

Here the LLM was created on the main asyncio thread but the actual call happens on a worker thread. There is **no concurrent access** — the main thread isn't using the LLM while the worker is. This is *sequential ownership transfer*, not sharing, and it's the pattern `AsyncReActAgent.run` relies on. The guard catches **contention**, not thread identity, so this pattern works.

Why call this out? Because the very first version of this guard used strict thread-id matching and broke `AsyncReActAgent` immediately. The contention-based design is what lets the rule "one context per thread" coexist with the legitimate `asyncio.to_thread` pattern. See "Alternative 1: Strict thread-id matching" in the design analysis section for the full story.

## Implementation

### Core mechanism

Each guarded class holds a `threading.Lock` set in `__init__` after the underlying native context is successfully created:

```python
# src/cyllama/api.py             — LLM
# src/cyllama/rag/embedder.py    — Embedder
self._busy_lock = threading.Lock()
```

```cython
# src/cyllama/whisper/whisper_cpp.pyx — WhisperContext
# src/cyllama/sd/stable_diffusion.pyx     — SDContext
cdef readonly object _busy_lock

def __init__(self, ...):
    ...
    import threading
    self._busy_lock = threading.Lock()
```

Note the `cdef readonly object` declaration on the Cython classes: this exposes the lock as a Python-readable attribute (so the regression tests can simulate contention by holding it directly) while preventing rebinding from Python. The `LLM` and `Embedder` classes are pure Python so the attribute is visible by default.

### The acquire helper

A small private method on each class implements the non-blocking acquire:

```python
def _try_acquire_busy(self) -> None:
    if not self._busy_lock.acquire(blocking=False):
        raise RuntimeError(
            "LLM is currently being used by another thread. llama.cpp "
            "contexts are not thread-safe — create one LLM per thread "
            "instead of sharing a single instance across threads."
        )
```

`blocking=False` is the load-bearing detail: a second concurrent caller fails fast instead of waiting in line. The reason for failing fast over serializing is discussed in the design-analysis section below.

### Where the guard is called

Each native-touching public method acquires the lock, runs its native call inside `try`, and releases in `finally`:

```python
def __call__(self, prompt, config=None, stream=False, on_token=None):
    self._try_acquire_busy()
    if stream:
        return self._stream_with_busy_release(
            self._generate_stream(prompt, config, on_token)
        )
    try:
        return self._generate(prompt, config, on_token)
    finally:
        self._busy_lock.release()
```

The guarded methods, by class:

| Class | Guarded methods | Notes |
|---|---|---|
| `LLM` | `__call__`, `generate_with_stats`, `reset_context` | `chat()` is unguarded because it delegates to `__call__`, which already holds the lock. Acquiring twice would deadlock since `Lock` is non-reentrant. |
| `Embedder` | `embed`, `embed_with_info` | `embed_batch`, `embed_documents`, `embed_iter` are unguarded — they delegate to `embed()` per item, which acquires/releases per item. This means a parallel `embed_batch` from another thread will see interleaved per-item locking, not exclusive batch ownership; that's correct because each `embed()` call is self-contained. |
| `WhisperContext` | `encode`, `full` | `full()` releases the GIL inside `with nogil:` while holding the lock. |
| `SDContext` | `generate_with_params`, `generate_video` | `generate()` is unguarded because it delegates to `generate_with_params()`. |

`close()`, `__del__`, and `__dealloc__` are **deliberately unguarded** because the garbage collector may run them on any thread, and a guard there would convert benign cross-thread cleanup into a hard failure.

### The streaming wrapper

`LLM.__call__(stream=True)` returns a generator that may live longer than the `__call__` invocation itself. We can't release the lock when `__call__` returns — we have to hold it until the generator is exhausted, closed, or garbage-collected.

This is handled by wrapping the underlying stream generator:

```python
def _stream_with_busy_release(self, gen):
    try:
        yield from gen
    finally:
        try:
            self._busy_lock.release()
        except RuntimeError:
            # Lock already released — defensive, should not happen.
            pass
```

Generator `finally` blocks run on every termination path (StopIteration, exception, `.close()`, gc), so the lock is guaranteed to be released even if the caller drops the iterator without consuming it.

### Cython layout notes

The `cdef` class layouts deserve a callout because they're slightly subtle:

1. **`cdef readonly object _busy_lock`** — `readonly` is essential. Plain `cdef object` makes the attribute invisible from Python (so the test couldn't hold it). Without `readonly`, the attribute would be writable from Python (`ctx._busy_lock = something_else`), which we don't want — only the class itself should bind it during `__init__`. `readonly` is the minimum exposure needed.

2. **`cdef` declarations at the top of `def` methods.** Cython requires all `cdef` declarations in a function body to be hoisted to the top of the function — they can't appear inside `try` blocks or after Python statements. The `WhisperContext.full()` method had to be restructured to declare `cdef float[::1] samples_view` and friends at the top before the `_try_acquire_busy()` call.

3. **`__init__` import inside the function.** The `import threading` is local to `__init__` rather than at module top. This is just a minor optimization to avoid pulling in `threading` at module load time when it's only needed once per instance.

## Test pattern

The same pattern appears in all three regression test classes:

1. Create an instance with a real model.
2. Manually `acquire(blocking=False)` the busy-lock from the **test thread** to simulate "thread A is currently in flight."
3. Spawn a worker thread that calls the actual public method.
4. Assert the worker raised `RuntimeError` containing `"another thread"` and `"not thread-safe"`.
5. Release the lock in `finally`.

```python
def test_concurrent_generate_raises(self):
    ctx = self._make_ctx()
    assert ctx._busy_lock.acquire(blocking=False) is True
    try:
        errors: list[Exception] = []
        def worker():
            try:
                ctx.generate(prompt="x", width=64, height=64,
                             seed=0, sample_steps=1, cfg_scale=1.0)
            except RuntimeError as e:
                errors.append(e)

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=10)

        assert not t.is_alive(), "worker did not return — guard may be missing"
        assert len(errors) == 1
        assert "another thread" in str(errors[0])
        assert "not thread-safe" in str(errors[0])
    finally:
        ctx._busy_lock.release()
```

This pattern is intentional for two reasons:

- **It exercises the real public-API entry points.** The worker calls `ctx.generate()` (or `llm()`, or `ctx.full()`), proving the guard is wired up to the user-visible method, not just to some internal helper.
- **The worker never reaches native code.** It hits `_try_acquire_busy()` and raises before any native call runs. This is critical for `SDContext` because `test_deterministic_seed` is currently skipped with the comment *"Multiple generations on same context causes segfault — needs investigation"*. By ensuring the worker never reaches native code, our test avoids that unrelated bug.

The `LLM` test (`tests/test_comprehensive.py::TestLLMConcurrencyGuard::test_concurrent_calls_from_two_threads_raises`) uses a slightly different pattern because `LLM` exposes an `on_token` callback that gives us a deterministic way to pause inside a real call: thread A starts a real generation, the `on_token` callback blocks on a `threading.Event`, thread B then attempts a concurrent call and is rejected. This is a stronger end-to-end test than the lock-holding shortcut because it actually runs native code in thread A. Use whichever pattern makes sense for the API surface being tested.

The test classes:

| Class | File | Tests | Model fixture |
|---|---|---|---|
| `TestLLMConcurrencyGuard` | `tests/test_comprehensive.py` | 5 | `Llama-3.2-1B-Instruct-Q8_0.gguf` |
| `TestEmbedderConcurrencyGuard` | `tests/test_rag_embedder.py` | 6 | `Llama-3.2-1B-Instruct-Q8_0.gguf` |
| `TestSDContextConcurrencyGuard` | `tests/test_sd.py` | 3 | `sd_xl_turbo_1.0.q8_0.gguf` |
| `TestWhisperContextConcurrencyGuard` | `tests/test_whisper.py` | 3 | `ggml-base.en.bin` |

Each class skips cleanly when its model is missing, so the suite stays runnable on machines without all three.

## Extending the guard

If you add a new method to `LLM`, `WhisperContext`, or `SDContext` that touches native state, **you must add the acquire/release pair**. There is no static check that catches this — the discipline is purely manual.

Checklist for a new guarded method:

1. Does the method touch `self._ctx`, `self._sampler`, `self.model`, `self._c_ctx`, `self.ptr`, or any other attribute that wraps a C++ object's mutable state?
2. Does the method trigger a native call that releases the GIL (`with nogil:`, or any binding that internally does so)?
3. If yes to either: wrap the body in `_try_acquire_busy()` / `try` / `finally: self._busy_lock.release()`.
4. If the method is a thin delegator that only calls another already-guarded public method: do **not** acquire (the inner method will). `Lock` is non-reentrant — double acquire from the same thread would deadlock. Document the delegation in a comment.
5. If the method returns a generator that holds native state across yields (streaming): use the `_stream_with_busy_release()` wrapper pattern.
6. Add a regression test that follows the pattern in the table above. The test should hit your new method specifically, even if other guarded methods on the same class are already covered — bug-of-omission tests need to be per-method, not per-class.

Methods that **don't** need the guard:

- Pure Python helpers that don't touch C++ state (cache management, config validation, message-list parsing).
- Read-only metadata accessors that go through llama.cpp's thread-safe `llama_model_*` family (model name, n_params, vocab queries) — these are documented as safe.
- `close()`, `__del__`, `__exit__`, `__dealloc__` — see the rationale above.
- Methods that already delegate to a guarded method.

When in doubt: read the upstream comment for the underlying C function. llama.cpp documents thread safety per-function in `llama.h`.

## Design analysis

### What we picked

A non-blocking `threading.Lock` acquired around every native-touching public method, raising `RuntimeError` on contention.

### Alternatives considered

#### Alternative 1: Strict thread-id matching

Record the creating thread's id in `__init__`, check on every method that the current thread matches. This was the **first** design we tried; it shipped briefly and broke immediately.

```python
# DON'T DO THIS — it false-positives on legitimate patterns.
def __init__(self, ...):
    self._owning_thread_id = threading.get_ident()

def _check_thread(self):
    if threading.get_ident() != self._owning_thread_id:
        raise RuntimeError("created on thread X, called from thread Y")
```

**Why rejected:** false-positives on `asyncio.to_thread`, `ThreadPoolExecutor.submit`, and any executor handoff. The `AsyncReActAgent` integration tests broke instantly because `agent.run` does `await asyncio.to_thread(self._agent.run, task)` — perfectly safe sequential ownership transfer, but the LLM was created on the main asyncio thread and used on a worker. Strict thread-id matching can't tell the difference between safe handoff and unsafe sharing.

The lesson: the rule we want to enforce is "no two native calls overlap in time," not "all native calls happen on the creating thread." Those are different rules, and only the first one matches the actual hazard.

#### Alternative 2: Blocking lock (serialize instead of raise)

Use `acquire(blocking=True)` so concurrent callers wait their turn instead of raising. This is what most stdlib Python objects in this position do: `sqlite3` (with `check_same_thread=False`), `requests.Session`, `http.client.HTTPConnection`, etc.

**Pros:**

- Code that worked accidentally (relying on GIL serialization or coincidental timing) keeps working.
- No surprise behavior change for existing users.
- Matches Python ecosystem norms.

**Cons:**

- **Hides architectural mistakes.** A user who accidentally shares one `LLM` across 8 worker threads gets 1/8th the throughput they expected, with no signal telling them why. They discover the problem months later staring at a flame graph.
- **Unbounded latency in async contexts.** A blocking acquire inside an `asyncio.to_thread` call can deadlock the event loop or stall it for arbitrary periods if the lock-holder is slow.
- **Loses the "fail loud" signal.** cyllama generally takes the position that bad inputs and unsafe usage should raise immediately, not degrade silently. Other parts of the codebase do the same: `validate_gguf_file`, `VectorStoreError` on metadata mismatch, the typed exceptions on model loaders.

**When to revisit this:** If cyllama's positioning ever shifts toward "high-level convenience library where users don't think about threading," serialize would become the right call. The conversion is a one-line change (`blocking=False` → `blocking=True`) — the lock infrastructure stays the same, so this decision isn't permanently locked in.

#### Alternative 3: Per-thread context pool

Maintain a pool of contexts inside each `LLM`, allocate one per thread on first use, multiplex calls automatically.

**Pros:**

- Actually allows concurrent inference instead of just preventing the bad case.

**Cons:**

- **Memory blowup.** A `llama_context` with default `n_ctx=2048` consumes hundreds of MB to a few GB depending on the model. Allocating one per worker thread is prohibitive — most production setups can barely fit *one* context per GPU.
- **Wrong layer.** If a user wants concurrent inference, they should make that decision explicitly by creating multiple `LLM` instances (or by using multiple GPUs). Hiding the cost inside the wrapper makes it look free when it isn't.
- **Doesn't help the asyncio.to_thread case.** That pattern uses one context shared across many short calls, which works fine without a pool — adding pooling would just waste memory.

#### Alternative 4: Documentation only, no runtime check

Just say "don't share `LLM` across threads" in the docs and rely on users reading them.

**Pros:**

- Zero overhead.
- No false positives.

**Cons:**

- **Silent UB when the rule is violated.** The failure mode is corrupt KV cache → garbage tokens → user blames the model and files a confused bug report. Extremely hard to debug.
- **Contradicts cyllama's "fail loud" philosophy.** We have typed exceptions on model loaders, validation errors on `VectorStore` metadata, friendly errors on missing files, etc. Letting this one class of bug fall through silently would be inconsistent.
- **The runtime check is cheap.** One atomic int compare per call, which is negligible compared to a token generation step.

#### Alternative 5: Fix it upstream

Make llama.cpp / whisper.cpp / sd.cpp actually thread-safe.

**Why rejected:** not in our control. Upstream has explicitly chosen not to lock the inner generation loop because it would be expensive — the sampler state and KV cache are touched on every token, and a per-token mutex would dominate runtime. Patching it locally would mean re-syncing on every upstream version bump, which is the same maintenance burden as the current guard but harder.

Worth noting: even if upstream became thread-safe, we'd probably want to keep some form of the guard, because "thread-safe" in upstream-speak typically means "won't crash" — not "concurrent calls produce sensible results."

### Known weaknesses of the current design

1. **Maintenance discipline.** "Every method that touches native state must call `_try_acquire_busy()` first." There is no static check for this — a future PR adding a new method that touches `self._ctx` directly without the guard would silently bypass it. The only mitigation is the test discipline described above.

2. **Streaming holds the lock for the entire stream.** A user who wants to "stream the answer to the user while embedding the next query" can't do it on one `LLM`. This is *correct* (concurrent native calls would corrupt state), but it's a behavioral surprise. The right answer is "use two `LLM` instances, one for generation and one for embedding," which is what most production setups do anyway.

3. **No support for reentrancy from callbacks.** If a user's `on_token` callback calls `llm()` again from the same thread, they get a `RuntimeError` immediately because the lock isn't reentrant (`Lock`, not `RLock`). This is exotic and probably correct — recursion into the same llama.cpp context is upstream UB regardless — but it's not signposted in user-facing docs. If someone needs to do this, they should create a second `LLM` for the inner call.

4. **Contention check, not correctness check.** What we actually want to express is "no two native calls into the same context overlap in time." The lock gives us that *in practice*, but only because every Python entry into native code goes through one of the guarded methods. If a future binding exposes raw `llama_decode()` to Python (or any other native call) without going through the wrapper, the guard wouldn't help.

5. **Doesn't address the upstream root cause.** We're papering over an upstream limitation. That's fine — sometimes papering over is the correct response — but it means we'll need this guard for as long as upstream remains non-thread-safe (which is likely indefinitely), and any new upstream API surface will need the same treatment.

6. **Lock release across threads.** `threading.Lock` allows release from any thread, which we rely on for the streaming wrapper case (a generator created on thread A might be consumed and finally-ed on thread B). This is not a bug, but it's worth being aware of: we cannot use `threading.RLock` (which enforces same-thread release) without breaking the streaming case. If you ever want reentrancy, you'll need a custom recursion-counter implementation, not a stdlib `RLock`.

### Why "raise" is the right call for cyllama specifically

Putting it all together: cyllama is positioned as a low-level Python wrapper around llama.cpp. Users who pick it up are typically building their own inference architecture — async servers, agent frameworks, RAG pipelines, batch processors. They need to know about thread-safety constraints so they can design their architecture correctly.

A serializing lock would be friendlier in the moment but would hide architectural mistakes that show up as unexplained performance problems weeks later, in production, under load, where they're the hardest to debug. A loud `RuntimeError` on the first concurrent call is annoying but actionable. The annoyance happens during development, when the user is in a position to fix it.

If cyllama were positioned higher up the stack — "give me an LLM, I don't care about threading" — the calculus would be different and serialize would win. But that's not the niche cyllama occupies, and changing the runtime guard wouldn't change the niche.

### Open follow-ups

- **`docs/` user-facing threading model page.** Tracked in `TODO.md` under High Priority. Should describe the "one in-flight call per instance" rule, the `asyncio.to_thread` pattern that's safe, the recommendation to create one `LLM` per worker, and the `RuntimeError` users will see if they get it wrong. This document is for maintainers; a separate user-facing one is still needed.
- **Static check for missing guards.** No good story here — Cython doesn't lend itself to lint-time analysis, and the discipline is per-method. Could be done at code-review time via a checklist, or via a runtime check that wraps every public method on the class with an assertion that `_busy_lock` was touched. Both feel overengineered.
