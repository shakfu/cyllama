"""Tests for the stable diffusion module."""

import os
import tempfile
import pytest
import numpy as np

# Skip all tests if stable diffusion module not available
pytest.importorskip("cyllama.sd")

from cyllama.sd import (
    SDContext,
    SDContextParams,
    SDImage,
    SDImageGenParams,
    SDSampleParams,
    Upscaler,
    RngType,
    SampleMethod,
    Scheduler,
    Prediction,
    SDType,
    LogLevel,
    PreviewMode,
    LoraApplyMode,
    HiresUpscaler,
    text_to_image,
    text_to_images,
    image_to_image,
    convert_model,
    canny_preprocess,
    get_num_cores,
    get_system_info,
    type_name,
    sample_method_name,
    scheduler_name,
    set_log_callback,
    set_progress_callback,
    set_preview_callback,
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

    def test_save_ppm(self):
        """Test saving image as PPM format (no PIL required)."""
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[10:20, 10:20] = [255, 0, 0]  # Red square
        img = SDImage.from_numpy(arr)

        with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as f:
            ppm_path = f.name

        try:
            img.save_ppm(ppm_path)
            assert os.path.exists(ppm_path)
            assert os.path.getsize(ppm_path) > 0

            # Verify PPM header
            with open(ppm_path, "rb") as f:
                header = f.read(20)
                assert header.startswith(b"P6\n")
        finally:
            os.unlink(ppm_path)

    def test_save_bmp(self):
        """Test saving image as BMP format (no PIL required)."""
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[10:20, 10:20] = [0, 255, 0]  # Green square
        img = SDImage.from_numpy(arr)

        with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as f:
            bmp_path = f.name

        try:
            img.save_bmp(bmp_path)
            assert os.path.exists(bmp_path)
            assert os.path.getsize(bmp_path) > 0

            # Verify BMP header
            with open(bmp_path, "rb") as f:
                header = f.read(2)
                assert header == b"BM"
        finally:
            os.unlink(bmp_path)

    def test_ppm_roundtrip(self):
        """Test save/load PPM preserves image data."""
        arr = np.random.randint(0, 256, (32, 48, 3), dtype=np.uint8)
        img = SDImage.from_numpy(arr)

        with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as f:
            ppm_path = f.name

        try:
            img.save_ppm(ppm_path)
            img2 = SDImage.load_ppm(ppm_path)

            arr2 = img2.to_numpy()
            assert arr2.shape == arr.shape
            assert np.array_equal(arr, arr2)
        finally:
            os.unlink(ppm_path)

    def test_bmp_roundtrip(self):
        """Test save/load BMP preserves image data."""
        arr = np.random.randint(0, 256, (32, 48, 3), dtype=np.uint8)
        img = SDImage.from_numpy(arr)

        with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as f:
            bmp_path = f.name

        try:
            img.save_bmp(bmp_path)
            img2 = SDImage.load_bmp(bmp_path)

            arr2 = img2.to_numpy()
            assert arr2.shape == arr.shape
            assert np.array_equal(arr, arr2)
        finally:
            os.unlink(bmp_path)

    def test_save_auto_format(self):
        """Test save() auto-detects format from extension."""
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        arr[:] = [128, 64, 32]
        img = SDImage.from_numpy(arr)

        with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as f:
            ppm_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as f:
            bmp_path = f.name

        try:
            # save() should use built-in writers for ppm/bmp
            img.save(ppm_path)
            img.save(bmp_path)

            assert os.path.exists(ppm_path)
            assert os.path.exists(bmp_path)

            # Verify formats
            with open(ppm_path, "rb") as f:
                assert f.read(2) == b"P6"
            with open(bmp_path, "rb") as f:
                assert f.read(2) == b"BM"
        finally:
            os.unlink(ppm_path)
            os.unlink(bmp_path)

    def test_grayscale_to_ppm(self):
        """Test grayscale image saves to PPM correctly."""
        arr = np.zeros((32, 32, 1), dtype=np.uint8)
        arr[10:20, 10:20] = 200
        img = SDImage.from_numpy(arr)

        with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as f:
            ppm_path = f.name

        try:
            img.save_ppm(ppm_path)
            assert os.path.exists(ppm_path)

            # Load and verify grayscale was expanded to RGB
            img2 = SDImage.load_ppm(ppm_path)
            arr2 = img2.to_numpy()
            assert arr2.shape == (32, 32, 3)
            # Grayscale values should be replicated to RGB
            assert arr2[15, 15, 0] == 200
            assert arr2[15, 15, 1] == 200
            assert arr2[15, 15, 2] == 200
        finally:
            os.unlink(ppm_path)

    def test_save_png(self):
        """Test saving image as PNG format (stb_image_write, no PIL required)."""
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[10:20, 10:20] = [255, 0, 0]  # Red square
        img = SDImage.from_numpy(arr)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            png_path = f.name

        try:
            img.save_png(png_path)
            assert os.path.exists(png_path)
            assert os.path.getsize(png_path) > 0

            # Verify PNG header (magic bytes)
            with open(png_path, "rb") as f:
                header = f.read(8)
                assert header[:4] == b"\x89PNG"
        finally:
            os.unlink(png_path)

    def test_save_jpg(self):
        """Test saving image as JPEG format (stb_image_write, no PIL required)."""
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[10:20, 10:20] = [0, 255, 0]  # Green square
        img = SDImage.from_numpy(arr)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            jpg_path = f.name

        try:
            img.save_jpg(jpg_path, quality=85)
            assert os.path.exists(jpg_path)
            assert os.path.getsize(jpg_path) > 0

            # Verify JPEG header (magic bytes)
            with open(jpg_path, "rb") as f:
                header = f.read(2)
                assert header == b"\xff\xd8"  # JPEG SOI marker
        finally:
            os.unlink(jpg_path)

    def test_png_roundtrip(self):
        """Test save/load PNG preserves image data (via stb)."""
        arr = np.random.randint(0, 256, (32, 48, 3), dtype=np.uint8)
        img = SDImage.from_numpy(arr)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            png_path = f.name

        try:
            img.save_png(png_path)
            img2 = SDImage.load(png_path)

            arr2 = img2.to_numpy()
            assert arr2.shape == arr.shape
            # PNG is lossless, so should be exact
            assert np.array_equal(arr, arr2)
        finally:
            os.unlink(png_path)

    def test_jpg_roundtrip(self):
        """Test save/load JPEG (via stb). Note: JPEG is lossy."""
        # Use a simple solid color to minimize compression artifacts
        arr = np.full((32, 48, 3), [128, 128, 128], dtype=np.uint8)
        img = SDImage.from_numpy(arr)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            jpg_path = f.name

        try:
            img.save_jpg(jpg_path, quality=100)  # Max quality
            img2 = SDImage.load(jpg_path)

            arr2 = img2.to_numpy()
            assert arr2.shape == arr.shape
            # JPEG is lossy, so check values are close (within 5)
            assert np.allclose(arr, arr2, atol=5)
        finally:
            os.unlink(jpg_path)

    def test_load_with_channel_conversion(self):
        """Test loading image with channel conversion (stb feature)."""
        # Create and save an RGB image
        arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        img = SDImage.from_numpy(arr)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            png_path = f.name

        try:
            img.save_png(png_path)

            # Load as RGBA (4 channels)
            img2 = SDImage.load(png_path, channels=4)
            arr2 = img2.to_numpy()
            assert arr2.shape == (32, 32, 4)

            # Load as grayscale (1 channel)
            img3 = SDImage.load(png_path, channels=1)
            arr3 = img3.to_numpy()
            assert arr3.shape == (32, 32, 1)
        finally:
            os.unlink(png_path)


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
        params = SDImageGenParams(prompt="test prompt", width=256, height=256, seed=123, sample_steps=10, cfg_scale=5.0)
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

        # Idempotency: multiple register/unregister cycles should not raise
        # or leak native state. If the setter mishandled re-registration the
        # second iteration would crash.
        for _ in range(3):
            set_log_callback(callback)
            set_log_callback(None)
        # No SD operations ran, so no log events should have fired into the
        # callback -- catches accidental synchronous invocation during setup.
        assert logs == []

    def test_set_progress_callback(self):
        progress = []

        def callback(step, steps, time):
            progress.append((step, steps, time))

        for _ in range(3):
            set_progress_callback(callback)
            set_progress_callback(None)
        assert progress == []


# Integration tests that require a real model
@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason=f"Model not found at {MODEL_PATH}")
class TestSDContextIntegration:
    """Integration tests requiring a real model."""

    def test_context_creation(self, sd_ctx_factory):
        params = SDContextParams()
        params.model_path = MODEL_PATH
        params.n_threads = 4

        ctx = sd_ctx_factory(params)
        assert ctx.is_valid

    def test_generate_image(self, sd_ctx_factory):
        params = SDContextParams()
        params.model_path = MODEL_PATH
        params.n_threads = 4

        ctx = sd_ctx_factory(params)

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
        assert arr.std() > 5

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


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason=f"Model not found at {MODEL_PATH}")
class TestSDContextConcurrencyGuard:
    """Tests for the concurrent-use guard on SDContext.

    sd_ctx_t is not thread-safe under concurrent native calls. The
    guard catches actual contention via a non-blocking lock acquired
    around each native-touching public method (`generate`,
    `generate_with_params`, `generate_video`). A second concurrent
    caller hits the guard and raises a clear RuntimeError before any
    native code runs — so the test is safe even though
    multi-generation on the same context is a known segfault
    (TestSDContextIntegration::test_deterministic_seed is
    skipped for that reason).

    The test pattern: hold the busy-lock from the test thread to
    simulate "another thread is currently inside generate()", then
    invoke the real public method on a worker thread and assert it
    raises RuntimeError. The worker never reaches native code because
    `_try_acquire_busy()` rejects it first.
    """

    def _make_ctx(self, sd_ctx_factory) -> SDContext:
        params = SDContextParams()
        params.model_path = MODEL_PATH
        params.n_threads = 4
        return sd_ctx_factory(params)

    def test_concurrent_generate_raises(self, sd_ctx_factory):
        """A worker thread calling generate() while the busy-lock is
        already held must raise RuntimeError without entering native
        code."""
        import threading

        ctx = self._make_ctx(sd_ctx_factory)
        # Simulate "thread A is currently inside generate()" by holding
        # the lock from the test thread.
        assert ctx._busy_lock.acquire(blocking=False) is True
        try:
            errors: list[Exception] = []

            def worker():
                try:
                    ctx.generate(
                        prompt="x",
                        width=64,
                        height=64,
                        seed=0,
                        sample_steps=1,
                        cfg_scale=1.0,
                    )
                except RuntimeError as e:
                    errors.append(e)

            t = threading.Thread(target=worker)
            t.start()
            t.join(timeout=10)

            assert not t.is_alive(), "worker did not return — guard may be missing"
            assert len(errors) == 1, f"Expected concurrent generate() to raise, got: {errors}"
            msg = str(errors[0])
            assert "another thread" in msg
            assert "not thread-safe" in msg
        finally:
            ctx._busy_lock.release()

    def test_concurrent_generate_with_params_raises(self, sd_ctx_factory):
        """Same as above but exercises the lower-level
        generate_with_params() entry point directly."""
        import threading

        ctx = self._make_ctx(sd_ctx_factory)
        gen_params = SDImageGenParams(
            prompt="x",
            negative_prompt="",
            width=64,
            height=64,
            seed=0,
            batch_count=1,
            sample_steps=1,
            cfg_scale=1.0,
            sample_method=ctx.get_default_sample_method(),
            scheduler=ctx.get_default_scheduler(),
        )

        assert ctx._busy_lock.acquire(blocking=False) is True
        try:
            errors: list[Exception] = []

            def worker():
                try:
                    ctx.generate_with_params(gen_params)
                except RuntimeError as e:
                    errors.append(e)

            t = threading.Thread(target=worker)
            t.start()
            t.join(timeout=10)

            assert not t.is_alive(), "worker did not return — guard may be missing"
            assert len(errors) == 1, f"Expected concurrent generate_with_params() to raise, got: {errors}"
            assert "another thread" in str(errors[0])
        finally:
            ctx._busy_lock.release()

    def test_lock_release_allows_subsequent_acquire(self, sd_ctx_factory):
        """Sanity check: after releasing the busy-lock,
        `_try_acquire_busy()` succeeds again so a normal call would
        proceed. We don't actually call generate() to avoid the
        unrelated multi-generation segfault."""
        ctx = self._make_ctx(sd_ctx_factory)

        assert ctx._busy_lock.acquire(blocking=False) is True
        ctx._busy_lock.release()

        # Should not raise
        ctx._try_acquire_busy()
        ctx._busy_lock.release()


class TestSDImageExtended:
    """Extended SDImage tests."""

    def test_grayscale_image(self):
        """Test creating a grayscale (1 channel) image."""
        arr = np.zeros((64, 64, 1), dtype=np.uint8)
        arr[10:20, 10:20] = 128

        img = SDImage.from_numpy(arr)
        assert img.width == 64
        assert img.height == 64
        assert img.channels == 1

        arr2 = img.to_numpy()
        assert arr2.shape == (64, 64, 1)
        assert arr2[15, 15, 0] == 128

    def test_rgba_image(self):
        """Test creating an RGBA (4 channel) image."""
        arr = np.zeros((64, 64, 4), dtype=np.uint8)
        arr[10:20, 10:20] = [255, 128, 64, 255]

        img = SDImage.from_numpy(arr)
        assert img.width == 64
        assert img.height == 64
        assert img.channels == 4

        arr2 = img.to_numpy()
        assert arr2.shape == (64, 64, 4)
        assert np.all(arr2[15, 15] == [255, 128, 64, 255])

    def test_large_image(self):
        """Test creating a larger image."""
        arr = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)

        img = SDImage.from_numpy(arr)
        assert img.width == 1024
        assert img.height == 1024
        assert img.channels == 3

        arr2 = img.to_numpy()
        assert np.array_equal(arr, arr2)

    def test_non_square_image(self):
        """Test creating non-square images."""
        arr = np.random.randint(0, 256, (256, 512, 3), dtype=np.uint8)

        img = SDImage.from_numpy(arr)
        assert img.width == 512
        assert img.height == 256

        arr2 = img.to_numpy()
        assert arr2.shape == (256, 512, 3)


class TestSDContextParamsExtended:
    """Extended SDContextParams tests."""

    def test_vae_path(self):
        params = SDContextParams()
        params.vae_path = "/path/to/vae.safetensors"
        assert params.vae_path == "/path/to/vae.safetensors"

    def test_clip_l_path(self):
        params = SDContextParams()
        params.clip_l_path = "/path/to/clip_l.safetensors"
        assert params.clip_l_path == "/path/to/clip_l.safetensors"

    def test_clip_g_path(self):
        params = SDContextParams()
        params.clip_g_path = "/path/to/clip_g.safetensors"
        assert params.clip_g_path == "/path/to/clip_g.safetensors"

    def test_t5xxl_path(self):
        params = SDContextParams()
        params.t5xxl_path = "/path/to/t5xxl.safetensors"
        assert params.t5xxl_path == "/path/to/t5xxl.safetensors"

    def test_vae_decode_only(self):
        params = SDContextParams()
        params.vae_decode_only = False
        assert params.vae_decode_only is False
        params.vae_decode_only = True
        assert params.vae_decode_only is True

    def test_diffusion_flash_attn(self):
        params = SDContextParams()
        params.diffusion_flash_attn = True
        assert params.diffusion_flash_attn is True

    def test_rng_type(self):
        params = SDContextParams()
        params.rng_type = RngType.CUDA
        assert params.rng_type == RngType.CUDA

    def test_diffusion_model_path(self):
        params = SDContextParams()
        params.diffusion_model_path = "/path/to/diffusion.safetensors"
        assert params.diffusion_model_path == "/path/to/diffusion.safetensors"

    def test_offload_params_to_cpu(self):
        params = SDContextParams()
        params.offload_params_to_cpu = True
        assert params.offload_params_to_cpu is True

    def test_clip_vision_path(self):
        params = SDContextParams()
        params.clip_vision_path = "/path/to/clip_vision.safetensors"
        assert params.clip_vision_path == "/path/to/clip_vision.safetensors"

    def test_llm_path(self):
        params = SDContextParams()
        params.llm_path = "/path/to/qwen.gguf"
        assert params.llm_path == "/path/to/qwen.gguf"

    def test_llm_vision_path(self):
        params = SDContextParams()
        params.llm_vision_path = "/path/to/qwen_vision.gguf"
        assert params.llm_vision_path == "/path/to/qwen_vision.gguf"

    def test_taesd_path(self):
        params = SDContextParams()
        params.taesd_path = "/path/to/taesd.safetensors"
        assert params.taesd_path == "/path/to/taesd.safetensors"

    def test_control_net_path(self):
        params = SDContextParams()
        params.control_net_path = "/path/to/controlnet.gguf"
        assert params.control_net_path == "/path/to/controlnet.gguf"

    def test_photo_maker_path(self):
        params = SDContextParams()
        params.photo_maker_path = "/path/to/photomaker.bin"
        assert params.photo_maker_path == "/path/to/photomaker.bin"

    def test_tensor_type_rules(self):
        params = SDContextParams()
        params.tensor_type_rules = "^vae\\.=f16,model\\.=q8_0"
        assert params.tensor_type_rules == "^vae\\.=f16,model\\.=q8_0"

    def test_sampler_rng_type(self):
        params = SDContextParams()
        params.sampler_rng_type = RngType.CPU
        assert params.sampler_rng_type == RngType.CPU

    def test_prediction(self):
        params = SDContextParams()
        params.prediction = Prediction.V
        assert params.prediction == Prediction.V

    def test_lora_apply_mode(self):
        params = SDContextParams()
        params.lora_apply_mode = LoraApplyMode.IMMEDIATELY
        assert params.lora_apply_mode == LoraApplyMode.IMMEDIATELY

    def test_keep_clip_on_cpu(self):
        params = SDContextParams()
        params.keep_clip_on_cpu = True
        assert params.keep_clip_on_cpu is True

    def test_keep_vae_on_cpu(self):
        params = SDContextParams()
        params.keep_vae_on_cpu = True
        assert params.keep_vae_on_cpu is True

    def test_keep_control_net_on_cpu(self):
        params = SDContextParams()
        params.keep_control_net_on_cpu = True
        assert params.keep_control_net_on_cpu is True

    def test_diffusion_conv_direct(self):
        params = SDContextParams()
        params.diffusion_conv_direct = True
        assert params.diffusion_conv_direct is True

    def test_vae_conv_direct(self):
        params = SDContextParams()
        params.vae_conv_direct = True
        assert params.vae_conv_direct is True

    def test_tae_preview_only(self):
        params = SDContextParams()
        params.tae_preview_only = True
        assert params.tae_preview_only is True

    def test_flow_shift(self):
        params = SDSampleParams()
        params.flow_shift = 1.5
        assert abs(params.flow_shift - 1.5) < 0.001

    def test_chroma_use_dit_mask(self):
        params = SDContextParams()
        params.chroma_use_dit_mask = False
        assert params.chroma_use_dit_mask is False

    def test_chroma_use_t5_mask(self):
        params = SDContextParams()
        params.chroma_use_t5_mask = True
        assert params.chroma_use_t5_mask is True

    def test_chroma_t5_mask_pad(self):
        params = SDContextParams()
        params.chroma_t5_mask_pad = 10
        assert params.chroma_t5_mask_pad == 10


class TestSDSampleParamsExtended:
    """Extended SDSampleParams tests."""

    def test_sample_steps(self):
        params = SDSampleParams()
        params.sample_steps = 25
        assert params.sample_steps == 25

    def test_eta(self):
        params = SDSampleParams()
        params.eta = 0.5
        assert abs(params.eta - 0.5) < 0.001

    def test_cfg_scale_range(self):
        """Test various CFG scale values."""
        params = SDSampleParams()
        # Low CFG
        params.cfg_scale = 1.0
        assert abs(params.cfg_scale - 1.0) < 0.001
        # High CFG
        params.cfg_scale = 15.0
        assert abs(params.cfg_scale - 15.0) < 0.001

    def test_sample_method_all(self):
        """Test setting different sample methods."""
        params = SDSampleParams()
        for method in [SampleMethod.EULER, SampleMethod.EULER_A, SampleMethod.DPM2]:
            params.sample_method = method
            assert params.sample_method == method

    def test_scheduler_all(self):
        """Test setting different schedulers."""
        params = SDSampleParams()
        for sched in [Scheduler.DISCRETE, Scheduler.KARRAS, Scheduler.EXPONENTIAL]:
            params.scheduler = sched
            assert params.scheduler == sched

    def test_slg_scale(self):
        params = SDSampleParams()
        params.slg_scale = 2.5
        assert abs(params.slg_scale - 2.5) < 0.001

    def test_slg_layer_start(self):
        params = SDSampleParams()
        params.slg_layer_start = 0.01
        assert abs(params.slg_layer_start - 0.01) < 0.001

    def test_slg_layer_end(self):
        params = SDSampleParams()
        params.slg_layer_end = 0.2
        assert abs(params.slg_layer_end - 0.2) < 0.001

    def test_img_cfg_scale(self):
        params = SDSampleParams()
        params.img_cfg_scale = 1.5
        assert abs(params.img_cfg_scale - 1.5) < 0.001

    def test_distilled_guidance(self):
        params = SDSampleParams()
        params.distilled_guidance = 3.5
        assert abs(params.distilled_guidance - 3.5) < 0.001

    def test_shifted_timestep(self):
        params = SDSampleParams()
        params.shifted_timestep = 100
        assert params.shifted_timestep == 100


class TestSDImageGenParamsExtended:
    """Extended SDImageGenParams tests."""

    def test_negative_prompt(self):
        params = SDImageGenParams()
        params.negative_prompt = "blurry, ugly"
        assert params.negative_prompt == "blurry, ugly"

    def test_batch_count(self):
        params = SDImageGenParams()
        params.batch_count = 4
        assert params.batch_count == 4

    def test_strength(self):
        params = SDImageGenParams()
        params.strength = 0.75
        assert abs(params.strength - 0.75) < 0.001

    def test_clip_skip(self):
        params = SDImageGenParams()
        params.clip_skip = 2
        assert params.clip_skip == 2

    def test_set_init_image(self):
        """Test setting init image."""
        params = SDImageGenParams()
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        img = SDImage.from_numpy(arr)
        params.set_init_image(img)
        # Re-setting should replace, not leak or corrupt the struct.
        params.set_init_image(img)
        # The params object should still be readable after the setter ran;
        # a corrupted Cython struct would crash on attribute access.
        assert params.sample_params is not None

    def test_set_control_image(self):
        """Test setting control image."""
        params = SDImageGenParams()
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        img = SDImage.from_numpy(arr)
        params.set_control_image(img, strength=0.8)
        # set_control_image also stores the strength, which is readable.
        assert abs(params.control_strength - 0.8) < 1e-6

    def test_sample_params(self):
        """Test accessing sample_params."""
        params = SDImageGenParams()
        sample_params = params.sample_params
        assert sample_params is not None
        # Verify we can modify sample params through gen params
        sample_params.sample_steps = 30
        assert params.sample_params.sample_steps == 30

    def test_control_strength(self):
        params = SDImageGenParams()
        params.control_strength = 0.9
        assert abs(params.control_strength - 0.9) < 0.001

    def test_vae_tiling_enabled(self):
        params = SDImageGenParams()
        params.vae_tiling_enabled = True
        assert params.vae_tiling_enabled is True

    def test_vae_tile_size(self):
        params = SDImageGenParams()
        params.vae_tile_size = (256, 256)
        assert params.vae_tile_size == (256, 256)

    def test_vae_tile_overlap(self):
        params = SDImageGenParams()
        params.vae_tile_overlap = 0.25
        assert abs(params.vae_tile_overlap - 0.25) < 0.001

    def test_easycache_enabled(self):
        params = SDImageGenParams()
        params.easycache_enabled = True
        assert params.easycache_enabled is True

    def test_easycache_threshold(self):
        params = SDImageGenParams()
        params.easycache_threshold = 0.15
        assert abs(params.easycache_threshold - 0.15) < 0.001

    def test_easycache_range(self):
        params = SDImageGenParams()
        params.easycache_range = (0.1, 0.9)
        start, end = params.easycache_range
        assert abs(start - 0.1) < 0.001
        assert abs(end - 0.9) < 0.001

    def test_auto_resize_ref_image(self):
        params = SDImageGenParams()
        params.auto_resize_ref_image = True
        assert params.auto_resize_ref_image is True

    def test_hires_defaults(self):
        params = SDImageGenParams()
        assert params.hires_enabled is False
        assert params.hires_upscaler == HiresUpscaler.NONE
        assert abs(params.hires_scale - 0.0) < 1e-6
        assert params.hires_target_size == (0, 0)
        assert params.hires_steps == 0
        assert abs(params.hires_denoising_strength - 0.0) < 1e-6
        assert params.hires_tile_size == 0
        assert params.hires_model_path is None

    def test_hires_setters(self):
        params = SDImageGenParams()
        params.hires_enabled = True
        params.hires_upscaler = HiresUpscaler.LANCZOS
        params.hires_scale = 1.5
        params.hires_target_size = (1024, 768)
        params.hires_steps = 15
        params.hires_denoising_strength = 0.55
        params.hires_tile_size = 256
        params.hires_model_path = "/tmp/upscaler.gguf"
        assert params.hires_enabled is True
        assert params.hires_upscaler == HiresUpscaler.LANCZOS
        assert abs(params.hires_scale - 1.5) < 1e-6
        assert params.hires_target_size == (1024, 768)
        assert params.hires_steps == 15
        assert abs(params.hires_denoising_strength - 0.55) < 1e-6
        assert params.hires_tile_size == 256
        assert params.hires_model_path == "/tmp/upscaler.gguf"
        # clearing the path
        params.hires_model_path = None
        assert params.hires_model_path is None

    def test_hires_set_hires_fix(self):
        params = SDImageGenParams()
        params.set_hires_fix(
            enabled=True,
            upscaler=HiresUpscaler.LATENT_BICUBIC,
            scale=1.75,
            target_width=512,
            target_height=512,
            steps=10,
            denoising_strength=0.6,
            tile_size=192,
        )
        assert params.hires_enabled is True
        assert params.hires_upscaler == HiresUpscaler.LATENT_BICUBIC
        assert abs(params.hires_scale - 1.75) < 1e-6
        assert params.hires_target_size == (512, 512)
        assert params.hires_steps == 10
        assert abs(params.hires_denoising_strength - 0.6) < 1e-6
        assert params.hires_tile_size == 192

    def test_hires_upscaler_enum_values(self):
        # The enum must mirror the C-side ordering exactly: NONE first, then
        # LATENT, ..., MODEL last. Drift here means the C struct field will
        # be set to the wrong upscaler.
        assert HiresUpscaler.NONE == 0
        assert HiresUpscaler.LATENT == 1
        assert HiresUpscaler.MODEL == 9

    def test_set_mask_image(self):
        """Test setting mask image for inpainting."""
        params = SDImageGenParams()
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[20:40, 20:40] = 255  # White area = inpaint region
        mask = SDImage.from_numpy(arr)
        params.set_mask_image(mask)
        # Re-setting should replace, not leak or corrupt the struct.
        params.set_mask_image(mask)
        # The params object should still be readable after the setter ran.
        assert params.sample_params is not None


class TestCannyPreprocess:
    """Test Canny edge detection preprocessing."""

    def test_canny_basic(self):
        """Test basic Canny preprocessing on an image."""
        # Create a simple image with edges
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[10:30, 10:30] = 255  # White square on black background

        img = SDImage.from_numpy(arr)

        # Apply Canny preprocessing (modifies in place)
        result = canny_preprocess(img)
        assert result is True

        # The image should now contain edge data
        arr2 = img.to_numpy()
        assert arr2.shape == (64, 64, 3)

    def test_canny_thresholds(self):
        """Test Canny with custom thresholds."""
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[20:40, 20:40] = 128

        img = SDImage.from_numpy(arr)

        result = canny_preprocess(img, high_threshold=0.9, low_threshold=0.2, weak=0.3, strong=1.0)
        assert result is True

    def test_canny_inverse(self):
        """Test Canny with inverse option."""
        arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

        img = SDImage.from_numpy(arr)
        result = canny_preprocess(img, inverse=True)
        assert result is True


class TestPreviewCallback:
    """Test preview callback functionality."""

    def test_set_preview_callback(self):
        """Test setting a preview callback."""
        previews = []

        def callback(step, frames, is_noisy):
            previews.append({"step": step, "frame_count": len(frames), "is_noisy": is_noisy})

        # Idempotent register/unregister cycles should not raise.
        for _ in range(3):
            set_preview_callback(callback)
            set_preview_callback(None)
        # No SD operations ran, so no preview events should have fired.
        assert previews == []

    def test_clear_preview_callback(self):
        """Test clearing the preview callback."""
        # Clearing with None is valid even without a prior set.
        set_preview_callback(None)
        # Repeatable clear is a no-op; should not raise.
        set_preview_callback(None)
        # Establish the followup set/clear cycle still works after the
        # idempotent clear.
        set_preview_callback(lambda *a, **kw: None)
        set_preview_callback(None)
        # The setter should still be callable after all those cycles.
        assert callable(set_preview_callback)


class TestEnumsExtended:
    """Extended enum tests."""

    def test_prediction_enum(self):
        """Test Prediction enum values."""
        assert Prediction.EPS.value == 0
        assert len(list(Prediction)) >= 6  # Includes FLUX2_FLOW
        # Verify key prediction types exist
        assert hasattr(Prediction, "EPS")
        assert hasattr(Prediction, "V")
        assert hasattr(Prediction, "FLUX_FLOW")
        assert hasattr(Prediction, "FLUX2_FLOW")

    def test_log_level_enum(self):
        """Test LogLevel enum values."""
        assert LogLevel.DEBUG.value == 0
        assert LogLevel.INFO.value == 1
        assert LogLevel.WARN.value == 2
        assert LogLevel.ERROR.value == 3

    def test_preview_mode_enum(self):
        """Test PreviewMode enum values."""
        assert PreviewMode.NONE.value == 0
        assert len(list(PreviewMode)) >= 3

    def test_lora_apply_mode_enum(self):
        """Test LoraApplyMode enum values."""
        assert LoraApplyMode.AUTO.value == 0
        assert len(list(LoraApplyMode)) >= 2

    def test_all_sample_methods(self):
        """Test all sample method names."""
        methods = [
            SampleMethod.EULER,
            SampleMethod.EULER_A,
            SampleMethod.HEUN,
            SampleMethod.DPM2,
            SampleMethod.DPMPP2S_A,
            SampleMethod.DPMPP2M,
            SampleMethod.DPMPP2Mv2,
            SampleMethod.IPNDM,
            SampleMethod.IPNDM_V,
            SampleMethod.LCM,
            SampleMethod.DDIM_TRAILING,
            SampleMethod.TCD,
        ]
        for m in methods:
            name = sample_method_name(m)
            assert isinstance(name, str)
            assert len(name) > 0

    def test_all_schedulers(self):
        """Test all scheduler names."""
        schedulers = [
            Scheduler.DISCRETE,
            Scheduler.KARRAS,
            Scheduler.EXPONENTIAL,
            Scheduler.AYS,
            Scheduler.GITS,
            Scheduler.SGM_UNIFORM,
            Scheduler.SIMPLE,
            Scheduler.SMOOTHSTEP,
            Scheduler.LCM,
        ]
        for s in schedulers:
            name = scheduler_name(s)
            assert isinstance(name, str)
            assert len(name) > 0

    def test_all_sd_types(self):
        """Test all SDType names."""
        types = [
            SDType.F32,
            SDType.F16,
            SDType.Q4_0,
            SDType.Q4_1,
            SDType.Q5_0,
            SDType.Q5_1,
            SDType.Q8_0,
            SDType.Q8_1,
            SDType.Q2_K,
            SDType.Q3_K,
            SDType.Q4_K,
            SDType.Q5_K,
            SDType.Q6_K,
            SDType.Q8_K,
            SDType.BF16,
        ]
        for t in types:
            name = type_name(t)
            assert isinstance(name, str)
            assert len(name) > 0


class TestUpscaler:
    """Test Upscaler class (unit tests without model)."""

    def test_upscaler_import(self):
        """Test that Upscaler can be imported."""
        assert Upscaler is not None

    def test_upscaler_invalid_model(self):
        """Test that Upscaler raises error for invalid model path."""
        with pytest.raises(FileNotFoundError):
            Upscaler("/nonexistent/model.bin")


class TestConvertModel:
    """Test model conversion function (unit tests)."""

    def test_convert_model_import(self):
        """Test that convert_model can be imported."""
        assert convert_model is not None

    def test_convert_model_invalid_input(self):
        """Test that convert_model raises error for invalid input."""
        with pytest.raises(Exception):
            convert_model(
                input_path="/nonexistent/model.safetensors", output_path="/tmp/output.gguf", output_type=SDType.F16
            )


class TestConvenienceFunctions:
    """Test convenience functions (unit tests without model)."""

    def test_text_to_image_import(self):
        """Test that text_to_image can be imported."""
        assert text_to_image is not None

    def test_text_to_images_import(self):
        """Test that text_to_images can be imported."""
        assert text_to_images is not None

    def test_image_to_image_import(self):
        """Test that image_to_image can be imported."""
        assert image_to_image is not None

    def test_text_to_image_invalid_model(self):
        """Test that text_to_image raises error for invalid model."""
        with pytest.raises(FileNotFoundError):
            text_to_image(model_path="/nonexistent/model.safetensors", prompt="test")

    def test_text_to_images_invalid_model(self):
        """Test that text_to_images raises error for invalid model."""
        with pytest.raises(FileNotFoundError):
            text_to_images(model_path="/nonexistent/model.safetensors", prompt="test")

    def test_image_to_image_invalid_model(self):
        """Test that image_to_image raises error for invalid model."""
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        init_img = SDImage.from_numpy(arr)

        with pytest.raises(FileNotFoundError):
            image_to_image(model_path="/nonexistent/model.safetensors", init_image=init_img, prompt="test")


# Integration tests for convenience functions
@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason=f"Model not found at {MODEL_PATH}")
class TestConvenienceFunctionsIntegration:
    """Integration tests for convenience functions."""

    def test_text_to_image(self):
        """Test text_to_image returns a single SDImage."""
        image = text_to_image(
            model_path=MODEL_PATH,
            prompt="a simple test",
            width=256,
            height=256,
            seed=42,
            sample_steps=1,
            cfg_scale=1.0,
            n_threads=4,
        )

        assert isinstance(image, SDImage)
        assert image.width == 256
        assert image.height == 256

    def test_text_to_images(self):
        """Test text_to_images returns a list of SDImage variants."""
        images = text_to_images(
            model_path=MODEL_PATH,
            prompt="a simple test",
            width=256,
            height=256,
            seed=42,
            batch_count=2,
            sample_steps=1,
            cfg_scale=1.0,
            n_threads=4,
        )

        assert isinstance(images, list)
        assert len(images) == 2
        assert all(isinstance(img, SDImage) for img in images)
        assert all(img.width == 256 and img.height == 256 for img in images)

    @pytest.mark.skip(reason="image_to_image after text_to_image in same session causes segfault - needs investigation")
    def test_image_to_image(self):
        """Test image_to_image convenience function."""
        # Create an init image
        arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        init_img = SDImage.from_numpy(arr)

        images = image_to_image(
            model_path=MODEL_PATH,
            init_image=init_img,
            prompt="a simple test",
            strength=0.5,
            seed=42,
            sample_steps=1,
            cfg_scale=1.0,
            n_threads=4,
        )

        assert len(images) == 1
        assert images[0].width == 256
        assert images[0].height == 256
