# Threading and concurrency in cyllama

> **The rule:** Always use one context per thread. Do not share the same context across threads.
>
> Applies to every `LLM`, `WhisperContext`, `SDContext`, and `Embedder` instance. cyllama enforces this at runtime for all four — sharing one across threads raises a clear `RuntimeError` instead of silently corrupting state.

This page is for users writing multi-threaded or async code with cyllama. It covers:

- What's safe to share between threads and what isn't
- Patterns that work, with copy-pasteable examples
- Patterns to avoid, and the error you'll see if you trip them
- Async-specific guidance (`AsyncLLM`, `asyncio.to_thread`)
- Where to look if you want the design rationale (spoiler: [`docs/dev/runtime-guard.md`](dev/runtime-guard.md))

## What's safe to share

| Type | Safe to share across threads? | Notes |
|---|---|---|
| `LLM` | **No** — one per thread | Holds a `llama_context` (KV cache, sampler, batch buffers) which is single-owner. Concurrent calls raise `RuntimeError`. |
| `WhisperContext` | **No** — one per thread | Same situation. Upstream documents the rule explicitly in `whisper.h`. |
| `SDContext` | **No** — one per thread | Same situation. The cyllama runtime guard is your only safety net here. |
| `Embedder` | **No** — one per thread | Holds a `LlamaContext` internally. Concurrent calls raise `RuntimeError`. Same guard pattern as `LLM`. |
| `AsyncLLM` | One instance per LLM, **shared across coroutines is fine** | Wraps an `LLM` plus an internal `asyncio.Lock` that serializes concurrent `await llm(...)` calls cleanly. |
| `VectorStore` | **No** — rejects cross-thread use | Backed by stdlib `sqlite3` with `check_same_thread=True`. The rejection is automatic and surfaces as `sqlite3.ProgrammingError`. For concurrent access, open a separate `VectorStore` instance per thread on the same DB file (SQLite handles the cross-process locking). |
| `GenerationConfig` | **Yes** | Plain dataclass of generation parameters. No mutable state worth racing on. |
| `RAGConfig` | **Yes** | Same — plain dataclass. |
| `LlamaModel` | Yes for read-only metadata, no for inference | The model weights themselves are immutable after load, so reading model metadata (vocab size, n_params, chat template) is safe. The `LlamaContext` derived from it is the dangerous part. |
| `Embedder` cache | Yes (internal lock) | The `Embedder`'s LRU cache uses an internal lock for its own bookkeeping. The Embedder *as a whole* still isn't safe to share because of the underlying context. |

The short version: **anything that wraps a llama.cpp / whisper.cpp / sd.cpp context is one-per-thread**. Pure dataclasses and metadata accessors are fine to share.

## What raises and how to fix it

When two threads try to use the same `LLM` (or `WhisperContext` / `SDContext`) concurrently, the second one raises:

```text
RuntimeError: LLM is currently being used by another thread. llama.cpp
contexts are not thread-safe — create one LLM per thread instead of
sharing a single instance across threads.
```

The fix is always the same: **create one instance per thread or per worker, don't share**. Below are the three most common ways this error shows up and the corresponding fix.

### Mistake: module-global LLM shared across handlers

```python
# ❌ Don't do this
from cyllama import LLM

llm = LLM("model.gguf")  # module global

@app.route("/generate")
def handler(request):
    return llm(request.text)  # multiple workers race here
```

If your web framework runs handlers in a thread pool (Flask, Django sync mode, FastAPI sync routes), two simultaneous requests will both try to use the same `llm` and the second will raise.

**Fix:** one LLM per worker. The simplest version is a `threading.local()`:

```python
# ✅ One LLM per worker thread
import threading
from cyllama import LLM

_local = threading.local()
MODEL_PATH = "model.gguf"

def get_llm() -> LLM:
    if not hasattr(_local, "llm"):
        _local.llm = LLM(MODEL_PATH)
    return _local.llm

@app.route("/generate")
def handler(request):
    return get_llm()(request.text)
```

Each worker thread gets its own `LLM` lazily on first request. Memory cost: one model context per worker. If that's too expensive, switch to `AsyncLLM` (next section) which shares one context behind a serialization lock.

### Mistake: ThreadPoolExecutor with one shared LLM

```python
# ❌ Don't do this
from concurrent.futures import ThreadPoolExecutor
from cyllama import LLM

llm = LLM("model.gguf")

with ThreadPoolExecutor(max_workers=4) as ex:
    results = list(ex.map(llm, prompts))  # racing, second worker raises
```

**Fix:** one LLM per worker (matching `max_workers`):

```python
# ✅ One LLM per worker
from concurrent.futures import ThreadPoolExecutor
from cyllama import LLM

MAX_WORKERS = 4
llms = [LLM("model.gguf") for _ in range(MAX_WORKERS)]

def worker(args):
    idx, prompt = args
    return llms[idx % MAX_WORKERS](prompt)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    results = list(ex.map(worker, enumerate(prompts)))
```

Or use `threading.local()` as in the previous example.

### Mistake: streaming + non-streaming on the same LLM concurrently

```python
# ❌ Don't do this
llm = LLM("model.gguf")

# Thread A starts a stream
def consume_stream():
    for chunk in llm("tell a long story", stream=True):
        time.sleep(0.1)  # slow consumer
        print(chunk, end="")

threading.Thread(target=consume_stream).start()

# Thread B tries to call the same LLM while the stream is in flight
result = llm("quick question")  # raises
```

The streaming `__call__` holds the busy lock until the generator is exhausted, closed, or garbage-collected. Any concurrent call into the same `LLM` while the stream is alive will raise. This is **correct** behaviour — the underlying `llama_context` is mid-decode and concurrent native calls would corrupt KV cache state.

**Fix:** use two separate `LLM` instances if you genuinely need to interleave streaming and non-streaming inference. In practice this is rare; most production setups stream the answer to one user at a time per worker.

## Patterns that work

### Pattern 1: one LLM per worker (recommended for most production)

This is the canonical pattern for sync web servers, batch jobs, and worker pools. See the `threading.local()` example above. Tradeoff: linear memory in worker count, but full parallel inference and zero contention.

### Pattern 2: `AsyncLLM` (recommended for async servers)

`AsyncLLM` wraps a single `LLM` plus an internal `asyncio.Lock`. Multiple coroutines can `await llm(...)` concurrently — they're serialized by the lock, not raised:

```python
# ✅ Async server with one shared LLM behind AsyncLLM
import asyncio
from cyllama import AsyncLLM
from fastapi import FastAPI

app = FastAPI()
llm = AsyncLLM("model.gguf")

@app.get("/generate")
async def generate(prompt: str):
    response = await llm(prompt)
    return {"text": str(response)}

@app.on_event("shutdown")
async def shutdown():
    await llm.close()
```

Key properties:

- One LLM, shared across all request coroutines.
- Concurrent `await llm(...)` calls **serialize** (run one after the other), they don't raise. The internal `asyncio.Lock` is what makes this safe.
- Total throughput is single-LLM throughput. If you need real parallelism, run multiple `AsyncLLM` instances (one per worker process, or one per GPU).
- Inference still runs on a thread pool internally (via `asyncio.to_thread`), so the event loop stays responsive.

### Pattern 3: `asyncio.to_thread` with sequential ownership transfer

If you have a synchronous LLM and want to call it from an async function without blocking the event loop, `asyncio.to_thread` is the right tool:

```python
# ✅ Sync LLM, used from async via asyncio.to_thread
import asyncio
from cyllama import LLM

llm = LLM("model.gguf")  # created on the main asyncio thread

async def handler():
    # The actual call runs on a worker thread, but only one at a time.
    return await asyncio.to_thread(llm, "hello")
```

The LLM was created on the main asyncio thread but the actual call runs on a worker. **This works** even though it crosses thread boundaries — there's no concurrent access, just sequential ownership transfer. The runtime guard catches contention, not thread identity.

This is the pattern `AsyncReActAgent.run` uses internally.

**Caveat:** if you call this from multiple coroutines simultaneously, they will race — `asyncio.to_thread` does not serialize them. Either wrap with your own `asyncio.Lock`, or just use `AsyncLLM` which does this for you.

### Pattern 4: process-level parallelism

For CPU-bound or GPU-bound parallelism beyond what one process can deliver, use `multiprocessing` or a process pool. Each process gets its own LLM:

```python
# ✅ Process pool, one LLM per worker process
from multiprocessing import Pool
from cyllama import LLM

# Each worker process loads its own LLM in initializer.
_llm = None

def init():
    global _llm
    _llm = LLM("model.gguf")

def worker(prompt):
    return str(_llm(prompt))

if __name__ == "__main__":
    with Pool(processes=4, initializer=init) as p:
        results = p.map(worker, prompts)
```

Tradeoffs vs. thread-per-LLM: higher memory (each process has its own copy of model weights — can be mitigated with `mmap`), higher startup cost, but bypasses the GIL entirely and isolates crashes.

## VectorStore: a different rule, same outcome

`VectorStore` is the one cyllama type with cross-thread protection that does NOT come from cyllama itself — it's inherited from stdlib `sqlite3`, which defaults to `check_same_thread=True`:

```python
# ❌ Don't do this
from cyllama.rag import VectorStore

store = VectorStore(dimension=384, db_path="vectors.db")

def worker():
    store.search([0.1] * 384, k=5)  # raises sqlite3.ProgrammingError

threading.Thread(target=worker).start()
```

You'll get something like:

```text
sqlite3.ProgrammingError: SQLite objects created in a thread can only
be used in that same thread.
```

**Fix:** open a separate `VectorStore` instance per thread on the same DB file. SQLite handles the cross-process file locking for you:

```python
# ✅ One VectorStore instance per thread on the same DB file
from cyllama.rag import VectorStore

DB = "vectors.db"

def worker():
    with VectorStore.open(DB) as store:
        results = store.search([0.1] * 384, k=5)
        # ...

threading.Thread(target=worker).start()
```

Or, if all your workers are reading and you want a thread-pool pattern, give each worker its own long-lived `VectorStore` instance (analogous to the LLM `threading.local()` pattern above).

## Backend-specific caveats

Per the upstream llama.cpp maintainers (see [`docs/dev/runtime-guard.md`](dev/runtime-guard.md) for the citations), thread safety across *separate* contexts varies by backend:

- **CPU, Metal, CUDA**: thread-safe across separate contexts. The "one LLM per worker" pattern works as expected.
- **Vulkan, SYCL, HIP, OpenCL**: "probably not" thread-safe even across separate contexts, per the upstream maintainers. If you're running multiple LLMs in parallel on these backends, you may need additional process-level isolation. The cyllama runtime guard prevents the most common mistake (sharing one context) but doesn't help with the rarer mistake (multiple contexts on a backend that can't handle them).

## Want the rationale?

This page is "what to do." For "why we did it this way" — the design analysis, the alternatives we rejected (strict thread-id matching, blocking lock / serialize, per-thread context pool, docs-only, fix-upstream), the upstream issue references that establish the contract, and the implementation details — see [`docs/dev/runtime-guard.md`](dev/runtime-guard.md).
