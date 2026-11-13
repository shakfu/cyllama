"""Tests for generation_info, the per-context progress callback, batched_cfg
and conditioning_cache_size (stable-diffusion.cpp master-1094)."""

import gc
import json
import os

import numpy as np
import pytest

pytest.importorskip("cyllama.sd")

from cyllama.sd import SDContext, SDContextParams, SDImage
from cyllama.sd.stable_diffusion import SDGenerationProgress

MODEL_PATH = "models/sd_xl_turbo_1.0.q8_0.gguf"


class TestParams:
    def test_defaults_match_upstream(self):
        params = SDContextParams()
        assert params.batched_cfg is True
        assert params.conditioning_cache_size == 4

    def test_roundtrip(self):
        params = SDContextParams()
        params.batched_cfg = False
        params.conditioning_cache_size = 0
        assert params.batched_cfg is False
        assert params.conditioning_cache_size == 0

    def test_negative_cache_size_rejected(self):
        params = SDContextParams()
        with pytest.raises(ValueError):
            params.conditioning_cache_size = -1
        assert params.conditioning_cache_size == 4


def test_python_image_has_no_generation_info():
    img = SDImage.from_numpy(np.zeros((8, 8, 3), dtype=np.uint8))
    assert img.generation_info is None


def _generate(ctx, **kw):
    args = dict(prompt="a red cube", width=128, height=128, seed=42, sample_steps=2, cfg_scale=1.0)
    args.update(kw)
    return ctx.generate(**args)


@pytest.fixture(scope="class")
def sd_ctx():
    params = SDContextParams()
    params.model_path = MODEL_PATH
    params.n_threads = 4
    params.conditioning_cache_size = 0
    with SDContext(params) as ctx:
        yield ctx
    del ctx
    gc.collect()


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason=f"Model not found at {MODEL_PATH}")
class TestWithModel:
    def test_generation_info_json(self, sd_ctx):
        (img,) = _generate(sd_ctx)
        info = json.loads(img.generation_info)
        assert info["schema"] == "sdcpp.generation/v1"
        assert info["prompt"] == "a red cube"
        assert (info["width"], info["height"]) == (128, 128)
        assert info["seed"] == "42"

    def test_progress_callback_reports_stages(self, sd_ctx):
        events = []
        sd_ctx.set_generation_progress_callback(events.append)
        try:
            _generate(sd_ctx, sample_steps=2)
        finally:
            sd_ctx.set_generation_progress_callback(None)
        assert events and all(isinstance(e, SDGenerationProgress) for e in events)
        stages = {e.stage for e in events}
        assert {"sampling", "vae_decode"} <= stages
        sampling = [e for e in events if e.stage == "sampling"]
        assert all(e.unit == "steps" and e.total == 2 for e in sampling)
        assert sampling[-1].completed == 2

    def test_progress_callback_cleared(self, sd_ctx):
        events = []
        sd_ctx.set_generation_progress_callback(events.append)
        sd_ctx.set_generation_progress_callback(None)
        _generate(sd_ctx)
        assert events == []

    def test_progress_callback_exception_does_not_abort(self, sd_ctx, capfd):
        def boom(_):
            raise ValueError("callback failure")

        sd_ctx.set_generation_progress_callback(boom)
        try:
            (img,) = _generate(sd_ctx)
        finally:
            sd_ctx.set_generation_progress_callback(None)
        assert img.is_valid
        assert "callback failure" in capfd.readouterr().err


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason=f"Model not found at {MODEL_PATH}")
def test_batched_cfg_matches_sequential():
    """Batched and sequential CFG give the same image up to numerical noise.

    Runs on CPU: upstream disables batched CFG on Metal (sd.cpp #2149), so
    there the flag is a no-op. One SDXL-turbo step scales eps error by
    sigma ~14.6; measured CPU batched-vs-sequential mean diff is 4.4/255,
    below the CPU-vs-Metal sequential floor of 5.6.
    """
    results = []
    for batched in (True, False):
        params = SDContextParams()
        params.model_path = MODEL_PATH
        params.n_threads = 4
        params.backend = "cpu"
        params.batched_cfg = batched
        with SDContext(params) as ctx:
            (img,) = _generate(ctx, cfg_scale=2.0, negative_prompt="blurry", sample_steps=1)
            results.append(img.to_numpy().astype(np.float64))
        del ctx
        gc.collect()
    a, b = results
    assert np.abs(a - b).mean() < 8
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.98


def test_set_callback_on_closed_context_raises():
    ctx = SDContext.__new__(SDContext)  # __cinit__ only: no native context
    with pytest.raises(RuntimeError, match="closed"):
        ctx.set_generation_progress_callback(print)
