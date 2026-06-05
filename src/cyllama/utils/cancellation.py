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

import signal
from typing import Callable


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
