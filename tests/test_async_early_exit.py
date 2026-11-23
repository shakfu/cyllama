"""Async streams stopped early must stop and join their worker thread before
releasing their lock (iterate_in_thread)."""

import asyncio
import gc
import threading
import time

import pytest

from cyllama.utils.cancellation import iterate_in_thread


def _slow_counter(state, n=1000, delay=0.01):
    try:
        for i in range(n):
            time.sleep(delay)
            yield i
    finally:
        state["closed_on"] = threading.current_thread().name
        state["closed_at"] = time.monotonic()


async def test_yields_all_items():
    assert [x async for x in iterate_in_thread(lambda: iter(range(5)))] == [0, 1, 2, 3, 4]


async def test_error_propagates():
    def boom():
        yield 1
        raise ValueError("boom")

    got = []
    with pytest.raises(ValueError, match="boom"):
        async for x in iterate_in_thread(boom):
            got.append(x)
    assert got == [1]


async def test_early_exit_cancels_and_joins():
    state = {}
    cancelled = threading.Event()
    gen = iterate_in_thread(lambda: _slow_counter(state), cancelled.set)
    async for x in gen:
        if x == 2:
            break
    await gen.aclose()
    assert cancelled.is_set()
    # the iterator was closed on its worker thread before aclose() returned
    assert "closed_at" in state and state["closed_on"] != threading.current_thread().name


async def test_lock_held_until_thread_exits():
    state = {}
    lock = asyncio.Lock()

    async def consume():
        async with lock:
            gen = iterate_in_thread(lambda: _slow_counter(state))
            try:
                async for _ in gen:
                    break
            finally:
                await gen.aclose()

    await consume()
    released_at = time.monotonic()
    assert state["closed_at"] <= released_at


async def test_timeout_stops_worker():
    state = {}
    gen = iterate_in_thread(lambda: _slow_counter(state, delay=0.5))
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(gen.__anext__(), timeout=0.05)
    await gen.aclose()
    assert "closed_at" in state


async def test_async_llm_break_cancels_generation(model_path):
    from cyllama.api import AsyncLLM

    llm = AsyncLLM(model_path, max_tokens=256, temperature=0.0, n_ctx=512)
    inner = llm._llm._generate_stream
    pulled = []

    def counting(*args, **kwargs):
        for chunk in inner(*args, **kwargs):
            pulled.append(chunk)
            yield chunk

    llm._llm._generate_stream = counting
    async for _ in llm.stream("Count from one to one hundred:"):
        break
    # stream() returned only after the worker stopped, a few tokens in
    n = len(pulled)
    assert 1 <= n < 16
    await asyncio.sleep(0.2)
    assert len(pulled) == n
    llm._llm._generate_stream = inner
    assert (await llm("Hi", max_tokens=2)).text is not None
    await llm.close()
    del llm
    gc.collect()
