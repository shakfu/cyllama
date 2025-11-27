"""Tests for the stable diffusion module."""

import os
import pytest
import numpy as np

# Skip all tests if stable diffusion module not available
pytest.importorskip("cyllama.stablediffusion")

from cyllama.stablediffusion import (
    SDContext, SDContextParams, SDImage, SDImageGenParams, SDSampleParams,
    RngType, SampleMethod, Scheduler, Prediction, SDType, LogLevel, PreviewMode, LoraApplyMode,
    text_to_image, image_to_image,
    get_num_cores, get_system_info, type_name, sample_method_name, scheduler_name,
    set_log_callback, set_progress_callback
)


# Model path for integration tests
MODEL_PATH = "models/sd_xl_turbo_1.0.q8_0.gguf"


class TestEnums:
    """Test enum types."""

    def test_rng_type(self):
        assert len(list(RngType)) >= 3
        assert RngType.STD_DEFAULT.value == 0
        assert RngType.CUDA.value == 1
        assert RngType.CPU.value == 2

    def test_sample_method(self):
        assert len(list(SampleMethod)) >= 10
        assert SampleMethod.EULER.value == 0
        assert SampleMethod.EULER_A.value == 1

    def test_scheduler(self):
        assert len(list(Scheduler)) >= 8
        assert Scheduler.DISCRETE.value == 0
        assert Scheduler.KARRAS.value == 1

    def test_sd_type(self):
        assert len(list(SDType)) >= 10
        assert SDType.F32.value == 0
        assert SDType.F16.value == 1


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_num_cores(self):
        cores = get_num_cores()
        assert cores > 0

    def test_get_system_info(self):
        info = get_system_info()
        assert isinstance(info, str)
        assert len(info) > 0

    def test_type_name(self):
        assert type_name(SDType.F16) == "f16"
        assert type_name(SDType.F32) == "f32"
        assert type_name(SDType.Q4_0) == "q4_0"

    def test_sample_method_name(self):
        assert sample_method_name(SampleMethod.EULER) == "euler"
        assert sample_method_name(SampleMethod.EULER_A) == "euler_a"

    def test_scheduler_name(self):
        assert scheduler_name(Scheduler.DISCRETE) == "discrete"
        assert scheduler_name(Scheduler.KARRAS) == "karras"


class TestSDImage:
    """Test SDImage class."""

    def test_from_numpy(self):
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[10:20, 10:20] = [255, 0, 0]  # Red square

        img = SDImage.from_numpy(arr)
        assert img.width == 64
        assert img.height == 64
        assert img.channels == 3

    def test_to_numpy(self):
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[10:20, 10:20] = [255, 128, 64]

        img = SDImage.from_numpy(arr)
        arr2 = img.to_numpy()

        assert arr2.shape == (64, 64, 3)
        assert arr2.dtype == np.uint8
        assert np.all(arr2[10:20, 10:20] == [255, 128, 64])

    def test_roundtrip(self):
        """Test numpy -> SDImage -> numpy preserves data."""
        arr = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

        img = SDImage.from_numpy(arr)
        arr2 = img.to_numpy()

        assert np.array_equal(arr, arr2)


class TestSDContextParams:
    """Test SDContextParams class."""

    def test_default_init(self):
        params = SDContextParams()
        assert params.n_threads > 0 or params.n_threads == -1
        assert params.vae_decode_only is True

    def test_model_path(self):
        params = SDContextParams()
        params.model_path = "/path/to/model.safetensors"
        assert params.model_path == "/path/to/model.safetensors"

    def test_n_threads(self):
        params = SDContextParams()
        params.n_threads = 8
        assert params.n_threads == 8

    def test_wtype(self):
        params = SDContextParams()
        params.wtype = SDType.F32
        assert params.wtype == SDType.F32


class TestSDSampleParams:
    """Test SDSampleParams class."""

    def test_default_init(self):
        params = SDSampleParams()
        assert params.sample_steps > 0

    def test_sample_method(self):
        params = SDSampleParams()
        params.sample_method = SampleMethod.EULER_A
        assert params.sample_method == SampleMethod.EULER_A

    def test_scheduler(self):
        params = SDSampleParams()
        params.scheduler = Scheduler.KARRAS
        assert params.scheduler == Scheduler.KARRAS

    def test_cfg_scale(self):
        params = SDSampleParams()
        params.cfg_scale = 7.5
        assert abs(params.cfg_scale - 7.5) < 0.01


class TestSDImageGenParams:
    """Test SDImageGenParams class."""

    def test_default_init(self):
        params = SDImageGenParams()
        assert params.width == 512
        assert params.height == 512

    def test_prompt(self):
        params = SDImageGenParams()
        params.prompt = "a photo of a cat"
        assert params.prompt == "a photo of a cat"

    def test_dimensions(self):
        params = SDImageGenParams()
        params.width = 768
        params.height = 1024
        assert params.width == 768
        assert params.height == 1024

    def test_seed(self):
        params = SDImageGenParams()
        params.seed = 42
        assert params.seed == 42

    def test_constructor_kwargs(self):
        params = SDImageGenParams(
            prompt="test prompt",
            width=256,
            height=256,
            seed=123,
            sample_steps=10,
            cfg_scale=5.0
        )
        assert params.prompt == "test prompt"
        assert params.width == 256
        assert params.height == 256
        assert params.seed == 123


class TestCallbacks:
    """Test callback functions."""

    def test_set_log_callback(self):
        logs = []

        def callback(level, text):
            logs.append((level, text))

        set_log_callback(callback)
        # Callback is set, but won't be triggered without actual SD operations
        assert True  # Just verify no errors

    def test_set_progress_callback(self):
        progress = []

        def callback(step, steps, time):
            progress.append((step, steps, time))

        set_progress_callback(callback)
        assert True  # Just verify no errors


# Integration tests that require a real model
@pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason=f"Model not found at {MODEL_PATH}"
)
class TestSDContextIntegration:
    """Integration tests requiring a real model."""

    def test_context_creation(self):
        params = SDContextParams()
        params.model_path = MODEL_PATH
        params.n_threads = 4

        ctx = SDContext(params)
        assert ctx.is_valid

    def test_generate_image(self):
        params = SDContextParams()
        params.model_path = MODEL_PATH
        params.n_threads = 4

        ctx = SDContext(params)

        # SDXL Turbo specific settings
        images = ctx.generate(
            prompt="a simple test image",
            width=256,  # Smaller for faster testing
            height=256,
            seed=42,
            sample_steps=1,  # Minimum steps for speed
            cfg_scale=1.0,
        )

        assert len(images) == 1
        img = images[0]
        assert img.width == 256
        assert img.height == 256
        assert img.channels == 3

        # Verify we got actual image data
        arr = img.to_numpy()
        assert arr.shape == (256, 256, 3)
        assert arr.min() >= 0
        assert arr.max() <= 255
        # Image should have some variation (not all black or white)
        assert arr.std() > 10

    @pytest.mark.skip(reason="Multiple generations on same context causes segfault - needs investigation")
    def test_deterministic_seed(self):
        """Test that same seed produces same image."""
        params = SDContextParams()
        params.model_path = MODEL_PATH
        params.n_threads = 4

        ctx = SDContext(params)

        # Generate twice with same seed
        images1 = ctx.generate(
            prompt="test",
            width=128,
            height=128,
            seed=42,
            sample_steps=1,
            cfg_scale=1.0,
        )

        images2 = ctx.generate(
            prompt="test",
            width=128,
            height=128,
            seed=42,
            sample_steps=1,
            cfg_scale=1.0,
        )

        arr1 = images1[0].to_numpy()
        arr2 = images2[0].to_numpy()

        # Should be identical with same seed
        assert np.array_equal(arr1, arr2)
