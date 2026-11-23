"""Shared SIGINT-to-cancel plumbing for long-running native calls.

cyllama subsystems that run a long native call (``LLM`` over llama.cpp,
``WhisperContext`` over whisper.cpp) expose a uniform cancellation surface:
``cancel()`` / ``cancel_requested`` plus ``install_sigint_handler()``. The
signal-handling half is identical across them and lives here so each subsystem
delegates rather than reimplements it.

The handler is intentionally opt-in: importing or using these subsystems never
touches process-global signal state on its own. A caller (typically a CLI)
installs the handler explicitly, and Ctrl-C then routes to the subsystem's
``cancel()`` -- which aborts even a long native call (e.g. ``llama_decode``
prefill or ``whisper_full`` encode) where Python's default ``KeyboardInterrupt``
would otherwise be deferred until the call returns.
"""

from __future__ import annotations

import asyncio
import signal
import threading
from typing import Any, AsyncGenerator, Callable, Iterable, Optional, Tuple, TypeVar

T = TypeVar("T")


class SigintHandle:
    """Restorer returned by ``install_sigint_handler()``.

    Acts as a context manager (``__exit__`` restores the prior handler) and as
    an imperative handle (call ``.restore()`` directly). Idempotent -- a second
    restore is a no-op.

    Standard last-installed-first-restored caveat applies: if multiple handlers
    are stacked, restore them in reverse order.
    """

    __slots__ = ("_previous", "_restored")

    def __init__(self, previous: object) -> None:
        self._previous = previous
        self._restored = False

    def restore(self) -> None:
        if self._restored:
            return
        try:
            signal.signal(signal.SIGINT, self._previous)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            # Off-main-thread or unrestorable handler; best-effort.
            pass
        self._restored = True

    def __enter__(self) -> "SigintHandle":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.restore()


def install_sigint_handler(cancel: Callable[[], None]) -> SigintHandle:
    """Install a SIGINT (Ctrl-C) handler that invokes *cancel*.

    Must be called from the main Python thread (the ``signal.signal``
    restriction). The previous SIGINT handler is saved and restored via the
    returned :class:`SigintHandle` (use it as a context manager or call
    ``.restore()``). The handler swallows the signal -- no ``KeyboardInterrupt``
    propagates from it; cancellation is delivered through *cancel* instead.

    Args:
        cancel: zero-arg callable that requests cancellation (e.g. a
            subsystem's ``cancel`` method).

    Returns:
        A :class:`SigintHandle` that restores the previous handler.
    """
    previous = signal.signal(signal.SIGINT, lambda *_: cancel())
    return SigintHandle(previous)


async def iterate_in_thread(
    make_iter: Callable[[], Iterable[T]], cancel: Optional[Callable[[], None]] = None
) -> AsyncGenerator[T, None]:
    """Yield the items of ``make_iter()``, which runs on a worker thread.

    When the consumer stops early (break, exception, ``aclose()``, timeout), the
    iterator is stopped at its next item and ``cancel`` is called to abort a long
    native call. The generator does not finish until the thread has exited, so a
    lock held around it is not released while the thread still uses shared state.
    A worker thread cannot be interrupted, so waiting is the only way to know that.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Tuple[bool, Any]] = asyncio.Queue()
    stop = threading.Event()

    def run() -> None:
        it = iter(make_iter())
        try:
            for item in it:
                if stop.is_set():
                    break
                loop.call_soon_threadsafe(queue.put_nowait, (False, item))
        except BaseException as e:
            loop.call_soon_threadsafe(queue.put_nowait, (True, e))
            return
        finally:
            close = getattr(it, "close", None)
            if close is not None:
                close()  # runs the iterator's own cleanup on this thread
        loop.call_soon_threadsafe(queue.put_nowait, (True, None))

    worker = asyncio.ensure_future(asyncio.to_thread(run))
    finished = False
    try:
        while True:
            finished, value = await queue.get()
            if finished:
                if value is not None:
                    raise value
                return
            yield value
    finally:
        if not finished:
            stop.set()
            if cancel is not None:
                cancel()
        try:
            await asyncio.shield(worker)
        except Exception:
            pass  # an error raised while closing the iterator; the consumer has already left
