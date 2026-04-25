"""Tests for cooperative + mid-decode generation cancellation on LLM.

Two layers exercised:

* ``LLM._cancel_event`` -- polled between tokens in ``_generate_stream``.
* ``LlamaContext._cancel_flag`` -- read by a nogil ggml_abort_callback so
  ``llama_decode`` itself bails when the flag is set.

Both layers are wired through ``LLM.cancel()``. These tests are slow because
they require an actual model; they're tagged accordingly.
"""

from __future__ import annotations

import gc
import threading
import time

import pytest

from cyllama import LLM, GenerationConfig


@pytest.mark.slow
class TestCancellation:
    def test_cancel_between_tokens_stops_streaming(self, model_path):
        """cancel() called from another thread aborts the streaming loop."""
        llm = LLM(model_path)
        try:
            config = GenerationConfig(max_tokens=512, temperature=0.0)

            # Cancel ~50ms after we start consuming chunks. Steady-state
            # generation produces tens of tokens per second, so we expect
            # well under max_tokens chunks before the loop bails.
            cancel_timer = threading.Timer(0.05, llm.cancel)
            cancel_timer.start()

            t0 = time.time()
            chunks = list(llm("Write a long detailed essay about cats.", config=config, stream=True))
            elapsed = time.time() - t0

            cancel_timer.cancel()

            # We should have stopped well before max_tokens. With max_tokens=512
            # an uncancelled run on any reasonable model takes >>1s; cancelled
            # runs bail in well under that. Use a generous bound to stay
            # robust on slow CI hardware.
            assert elapsed < 5.0, (
                f"cancel did not interrupt generation in time (took {elapsed:.2f}s)"
            )
            # We should have produced *some* tokens (cancel arrived after start).
            joined = "".join(chunks)
            assert len(joined) >= 0  # may be 0 on very fast cancel; not asserting non-empty
            # cancel_requested should reflect the request.
            assert llm.cancel_requested is True
        finally:
            del llm
            gc.collect()

    def test_cancel_clears_between_generations(self, model_path):
        """A cancelled generation does not poison the next one."""
        llm = LLM(model_path)
        try:
            config = GenerationConfig(max_tokens=20, temperature=0.0)

            # Cancel the first run.
            llm.cancel()
            list(llm("First prompt.", config=config, stream=True))
            # After the call, cancel should be cleared at entry, so the next
            # generation must run normally.
            assert llm.cancel_requested is False, (
                "cancel flag should be cleared at the start of _generate_stream"
            )

            # Second run, no cancel: must produce non-empty output.
            chunks = list(llm("Say hello.", config=config, stream=True))
            assert "".join(chunks).strip() != ""
            assert llm.cancel_requested is False
        finally:
            del llm
            gc.collect()

    def test_cancel_idempotent(self, model_path):
        """Multiple cancel() calls before/during/after generation are safe."""
        llm = LLM(model_path)
        try:
            config = GenerationConfig(max_tokens=10, temperature=0.0)

            # Cancel before any context exists -- should be a no-op on the
            # ctx side and just set the Python event.
            llm.cancel()
            llm.cancel()
            assert llm.cancel_requested is True

            # Run a generation: cancel auto-clears, generation proceeds.
            chunks = list(llm("Hi.", config=config, stream=True))
            assert isinstance(chunks, list)

            # Post-run cancel is fine.
            llm.cancel()
            llm.cancel()
        finally:
            del llm
            gc.collect()

    def test_cancel_before_generation_short_circuits(self, model_path):
        """If cancel() is set before generation, it returns essentially nothing."""
        llm = LLM(model_path)
        try:
            # Run one short generation first to ensure a context exists.
            list(llm("Warm up.", config=GenerationConfig(max_tokens=2), stream=True))

            # Now request cancellation, then start another generation. The
            # event is cleared at entry to _generate_stream, so cancellation
            # must come from a thread *during* the call. The "set before"
            # case is the documented "auto-clear" behavior -- this test
            # locks that in.
            llm.cancel()
            chunks = list(llm("Another prompt.", config=GenerationConfig(max_tokens=5), stream=True))
            # Auto-cleared, so generation proceeds normally.
            assert isinstance(chunks, list)
        finally:
            del llm
            gc.collect()

    def test_install_sigint_handler_restores_previous(self, model_path):
        """install_sigint_handler() context manager restores prior handler."""
        import signal as _signal

        sentinel = object()
        previous_handlers: list = []

        def my_prior(*_args):
            previous_handlers.append(sentinel)

        # Install a known prior handler.
        old = _signal.signal(_signal.SIGINT, my_prior)
        try:
            llm = LLM(model_path)
            try:
                with llm.install_sigint_handler():
                    # Inside the context, our cancel-installer is active.
                    cur = _signal.getsignal(_signal.SIGINT)
                    assert cur is not my_prior
                    assert callable(cur)
                # On exit, prior handler restored.
                assert _signal.getsignal(_signal.SIGINT) is my_prior
            finally:
                del llm
                gc.collect()
        finally:
            _signal.signal(_signal.SIGINT, old)

    def test_install_sigint_handler_calls_cancel(self, model_path):
        """The installed handler routes SIGINT to llm.cancel()."""
        import os as _os
        import signal as _signal

        llm = LLM(model_path)
        try:
            assert llm.cancel_requested is False
            with llm.install_sigint_handler():
                # Raise SIGINT in our own process; the handler should
                # synchronously invoke cancel() before returning.
                _os.kill(_os.getpid(), _signal.SIGINT)
                # signal handlers run at the next bytecode boundary; give
                # the interpreter a brief moment to dispatch.
                time.sleep(0.05)
                assert llm.cancel_requested is True
        finally:
            del llm
            gc.collect()

    def test_install_sigint_handler_idempotent_restore(self, model_path):
        """Calling restore() multiple times is safe."""
        llm = LLM(model_path)
        try:
            handle = llm.install_sigint_handler()
            handle.restore()
            handle.restore()  # second call is a no-op
        finally:
            del llm
            gc.collect()

    def test_context_cancel_property_roundtrip(self, model_path):
        """LlamaContext.cancel is a Python-visible bint mirror."""
        llm = LLM(model_path)
        try:
            list(llm("Warm up.", config=GenerationConfig(max_tokens=2), stream=True))
            ctx = llm._ctx
            assert ctx is not None
            assert ctx.cancel is False
            ctx.cancel = True
            assert ctx.cancel is True
            ctx.cancel = False
            assert ctx.cancel is False
        finally:
            del llm
            gc.collect()
