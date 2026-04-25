# Cancelling generation

`LLM` supports thread-safe cancellation of an in-flight generation at two
layers:

- **Between tokens** — a `threading.Event` polled in the per-token loop.
  Sub-millisecond latency in steady-state generation.
- **Mid-decode** — a nogil `ggml_abort_callback` reads a C-level flag and
  aborts the in-progress `llama_decode` from inside ggml's compute graph.
  This is what makes cancellation responsive during long prompt prefill,
  where a single `decode` call may run for seconds.

Both layers are wired by a single call: `llm.cancel()`.

## What "abort" means

`ggml_abort_callback` is cooperative: when it returns non-zero, ggml stops
scheduling further ops in the current graph and `llama_decode` returns
early. **The process is not killed.** Control returns to Python normally,
the partially-produced tokens are yielded, and the `LLM` object remains
reusable for the next call. Only the in-progress batch is discarded.

The cancel flag auto-clears at the start of each generation, so a stale
`cancel()` does not leak into the next call.

## API

- `LLM.cancel()` — request cancellation. Safe from any thread.
- `LLM.cancel_requested` — read-only `bool` property.
- `LLM.install_sigint_handler()` — opt-in Ctrl-C handler. Returns a
  context manager / handle with `.restore()`.
- `LlamaContext.cancel` — read/write `bool` mirror of the C-level flag,
  for direct lower-level use.

## Examples

### 1. Cancel from another thread

```python
import threading
from cyllama import LLM, GenerationConfig

llm = LLM("models/Llama-3.2-1B-Instruct-Q8_0.gguf")
config = GenerationConfig(max_tokens=512, temperature=0.0)

threading.Timer(0.1, llm.cancel).start()

chunks = list(llm("Write a long essay about cats.", config=config, stream=True))
print(f"got {len(''.join(chunks))} chars before cancel")

# The LLM is still usable.
followup = llm("Say hi.", config=GenerationConfig(max_tokens=10))
print(followup)
```

### 2. Ctrl-C handler — interrupts even mid-prefill

```python
from cyllama import LLM, GenerationConfig

llm = LLM("models/Llama-3.2-1B-Instruct-Q8_0.gguf")
huge_prompt = "..." * 10_000  # forces a long prefill

with llm.install_sigint_handler():
    for chunk in llm(huge_prompt, config=GenerationConfig(max_tokens=200), stream=True):
        print(chunk, end="", flush=True)

# After Ctrl-C: prior SIGINT handler is restored, llm still usable.
print("\n-- back to normal --")
print(llm("ok?", config=GenerationConfig(max_tokens=5)))
```

`install_sigint_handler()` is opt-in by design; cyllama does not touch
signal handlers otherwise. The previous handler is saved and restored
on `.restore()` / `__exit__`, so it composes with Click, Jupyter,
asyncio, etc. Must be called from the main thread (`signal.signal`
restriction).

### 3. Cancel-on-disconnect in a FastAPI / SSE sidecar

The motivating use case: a streaming HTTP server should free the GPU
when the client closes the connection, instead of running to
`max_tokens`.

```python
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from cyllama import LLM, GenerationConfig

app = FastAPI()
llm = LLM("models/Llama-3.2-1B-Instruct-Q8_0.gguf")

@app.get("/stream")
async def stream(request: Request, prompt: str):
    async def gen():
        loop = asyncio.get_running_loop()
        it = iter(llm(prompt, config=GenerationConfig(max_tokens=2048), stream=True))
        try:
            while True:
                if await request.is_disconnected():
                    llm.cancel()           # aborts mid-decode
                    break
                chunk = await loop.run_in_executor(None, next, it, None)
                if chunk is None:
                    break
                yield f"data: {chunk}\n\n"
        finally:
            llm.cancel()                   # idempotent; safe on normal exit too

    return StreamingResponse(gen(), media_type="text/event-stream")
```

### 4. Direct use of `LlamaContext.cancel`

For callers working below the `LLM` API:

```python
from cyllama import LLM

llm = LLM("models/Llama-3.2-1B-Instruct-Q8_0.gguf")
list(llm("warm up", stream=True))   # forces _ensure_context()
ctx = llm._ctx

ctx.cancel = True                   # sets the C bint
assert ctx.cancel is True
ctx.cancel = False                  # clear before next call
```

## Notes and caveats

- **Performance.** The between-token check is one `Event.is_set()` per
  token (sub-microsecond). The mid-decode callback is `noexcept nogil`
  and does a single indirect load per ggml op poll. Overhead is not
  measurable against decode time.
- **Memory model.** The C flag is a plain `bint`, not a C11 atomic.
  Aligned word writes are atomic on every CPU cyllama targets; a stale
  read just delays cancellation by one op poll. This is acceptable for
  a one-shot "abort now" signal.
- **Custom abort callbacks.** `LLM` auto-installs the cancel callback
  on every context creation. Calling `LlamaContext.set_abort_callback()`
  with a Python callable overrides it. To combine user logic with
  cancellation, consult `ctx.cancel` (or your own state) inside that
  Python callback.
- **Stable Diffusion.** `cyllama.sd` does **not** currently support
  cancellation; `generate_image()` is a single blocking C call with no
  abort path. Tracked against upstream
  [leejet/stable-diffusion.cpp#1124](https://github.com/leejet/stable-diffusion.cpp/pull/1124).
