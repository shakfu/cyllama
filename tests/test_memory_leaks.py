"""Memory-leak regression tests.

Loops create/destroy of the three high-level model wrappers (LLM,
SDContext, WhisperContext), measures RSS after each cycle, and asserts
that growth stays within a tolerance window. The goal is to catch
native-side leaks (e.g. a missing free() in a Cython destructor) that
would not show up in normal Python-level test runs.

Implementation notes:

- We use ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` to avoid adding
  psutil as a dependency. This is the same primitive that
  ``scripts/leak_check.py`` already uses. ``ru_maxrss`` reports the
  *peak* RSS the process has seen since startup -- it never decreases.
  This is actually fine for leak detection: in a leak-free loop the
  peak is established by the first iteration and stays flat; in a
  leaking loop each cycle pushes the peak higher.

- All tests are marked ``slow`` because each iteration reloads a model
  from disk (the SD model in particular is multi-GB).

- Tests skip cleanly when the corresponding model file is not present
  -- the LLM test reuses the existing ``model_path`` fixture from
  conftest.py (which auto-skips when the file is missing); SD and
  whisper tests do their own ``skipif``.

- Windows is excluded because ``resource.getrusage`` is POSIX-only,
  matching the gating already used for ``pytest-memray`` in
  ``pyproject.toml``.
"""

from __future__ import annotations

import gc
import resource
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WHISPER_MODEL = ROOT / "models" / "ggml-base.en.bin"
SD_MODEL = ROOT / "models" / "sd_xl_turbo_1.0.q8_0.gguf"


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="resource.getrusage is POSIX only",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_rss_mb() -> float:
    """Return the process's peak RSS in megabytes.

    Linux ``ru_maxrss`` is reported in kilobytes; macOS reports in bytes.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def assert_rss_bounded(
    baseline_mb: float,
    samples_mb: list[float],
    *,
    tolerance_mb: float,
    label: str,
) -> None:
    """Assert that the peak RSS across all post-baseline samples does not
    exceed ``baseline_mb + tolerance_mb``.

    Reports the per-cycle samples in the failure message so a real leak
    is easy to diagnose (you see the monotonic growth).
    """
    peak = max(samples_mb)
    delta = peak - baseline_mb
    assert delta <= tolerance_mb, (
        f"{label}: peak RSS grew by {delta:.1f} MB across "
        f"{len(samples_mb)} cycles (baseline={baseline_mb:.1f} MB, "
        f"peak={peak:.1f} MB, tolerance={tolerance_mb} MB). "
        f"Per-cycle samples: {[round(x, 1) for x in samples_mb]}"
    )


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


class TestLLMLeaks:
    """Loop create/destroy LLM, including a tiny generation per cycle so
    the sampler/context allocators are exercised in addition to the
    bare model load."""

    WARMUP_CYCLES = 2
    MEASURE_CYCLES = 5
    TOLERANCE_MB = 80

    def _cycle(self, model_path: str) -> None:
        from cyllama import LLM, GenerationConfig

        config = GenerationConfig(
            max_tokens=4,
            n_ctx=256,
            n_batch=256,
            n_gpu_layers=0,
        )
        llm = LLM(model_path, config=config)
        llm("Hi")
        llm.close()
        del llm

    def test_create_destroy_loop(self, model_path: str):
        # Warm up so allocator pools / GPU shaders / etc. are stable.
        for _ in range(self.WARMUP_CYCLES):
            self._cycle(model_path)
        gc.collect()
        baseline = get_rss_mb()

        samples: list[float] = []
        for _ in range(self.MEASURE_CYCLES):
            self._cycle(model_path)
            gc.collect()
            samples.append(get_rss_mb())

        assert_rss_bounded(baseline, samples, tolerance_mb=self.TOLERANCE_MB, label="LLM")


# ---------------------------------------------------------------------------
# SDContext
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not SD_MODEL.exists(),
    reason=f"Stable Diffusion model not found at {SD_MODEL}",
)
class TestSDContextLeaks:
    """Loop create/destroy of SDContext. We do *not* call .generate() per
    cycle because the SD model is multi-GB and even a single-step
    generation is slow; the leak we're guarding against is in
    SDContext.__init__/__dealloc__, which is the dominant allocation
    path."""

    # SD's allocator pools are large (~4 GB model) and take more than one
    # cycle to stabilize -- a single warmup leaves a ~270 MB jump on the
    # first measured cycle that then flattens out completely. Two warmups
    # absorb that settling so the measurement window reflects steady-state.
    WARMUP_CYCLES = 2
    MEASURE_CYCLES = 3
    TOLERANCE_MB = 250  # SD model is ~4 GB; allocator slack is larger

    def _cycle(self) -> None:
        from cyllama.sd import SDContext, SDContextParams

        params = SDContextParams()
        params.model_path = str(SD_MODEL)
        params.n_threads = 4
        ctx = SDContext(params)
        assert ctx.is_valid
        del ctx
        del params

    def test_create_destroy_loop(self):
        for _ in range(self.WARMUP_CYCLES):
            self._cycle()
        gc.collect()
        baseline = get_rss_mb()

        samples: list[float] = []
        for _ in range(self.MEASURE_CYCLES):
            self._cycle()
            gc.collect()
            samples.append(get_rss_mb())

        assert_rss_bounded(baseline, samples, tolerance_mb=self.TOLERANCE_MB, label="SDContext")


# ---------------------------------------------------------------------------
# WhisperContext
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not WHISPER_MODEL.exists(),
    reason=f"Whisper model not found at {WHISPER_MODEL}",
)
class TestWhisperContextLeaks:
    """Loop create/destroy of WhisperContext."""

    WARMUP_CYCLES = 2
    MEASURE_CYCLES = 5
    TOLERANCE_MB = 60

    def _cycle(self) -> None:
        from cyllama.whisper import whisper_cpp as wh

        ctx = wh.WhisperContext(str(WHISPER_MODEL))
        del ctx

    def test_create_destroy_loop(self):
        for _ in range(self.WARMUP_CYCLES):
            self._cycle()
        gc.collect()
        baseline = get_rss_mb()

        samples: list[float] = []
        for _ in range(self.MEASURE_CYCLES):
            self._cycle()
            gc.collect()
            samples.append(get_rss_mb())

        assert_rss_bounded(
            baseline,
            samples,
            tolerance_mb=self.TOLERANCE_MB,
            label="WhisperContext",
        )
