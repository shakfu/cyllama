"""
Stable Diffusion module for cyllama.

Provides Python bindings for stable-diffusion.cpp image generation.

Example:
    from cyllama.sd import text_to_image, text_to_images, SDContext, SDContextParams

    # Simple usage - single image
    image = text_to_image(
        model_path="sd-v1-5.safetensors",
        prompt="a photo of a cat",
        width=512,
        height=512
    )

    # Save in common formats (no dependencies - uses bundled stb library)
    image.save("output.png")
    image.save("output.jpg", quality=90)
    image.save("output.bmp")

    # Batch generation - multiple variants
    images = text_to_images(
        model_path="sd-v1-5.safetensors",
        prompt="a photo of a cat",
        batch_count=4,
    )

    # Load images (PNG, JPEG, BMP, TGA, GIF, PSD, HDR, PIC supported)
    img = SDImage.load("input.png")
    img = SDImage.load("input.jpg", channels=3)  # Force RGB

    # With model reuse
    params = SDContextParams(model_path="sd-v1-5.safetensors")
    with SDContext(params) as ctx:
        for prompt in prompts:
            images = ctx.generate(prompt)

    # Video generation (requires video-capable model like Wan)
    frames = ctx.generate_video(
        prompt="a cat walking",
        video_frames=16
    )

    # Upscaling with ESRGAN
    from cyllama.sd import Upscaler
    upscaler = Upscaler("esrgan-x4.bin")
    upscaled = upscaler.upscale(image)

CLI Usage:
    python -m cyllama.sd generate --model MODEL --prompt "..."
    python -m cyllama.sd upscale --model MODEL --input IMAGE
    python -m cyllama.sd convert --input MODEL --output MODEL
    python -m cyllama.sd info
"""

from ..utils.platform import ensure_native_deps

ensure_native_deps()

from .stable_diffusion import (
    # Main classes
    SDContext,
    SDContextParams,
    SDImage,
    SDImageGenParams,
    SDSampleParams,
    Upscaler,
    # Enums
    RngType,
    SampleMethod,
    Scheduler,
    Prediction,
    SDType,
    LogLevel,
    PreviewMode,
    LoraApplyMode,
    # Convenience functions
    text_to_image,
    text_to_images,
    image_to_image,
    # Model utilities
    convert_model,
    canny_preprocess,
    # Backend loading
    ggml_backend_load_all,
    # Utility functions
    get_num_cores,
    get_system_info,
    type_name,
    sample_method_name,
    scheduler_name,
    # Callback setters
    set_log_callback,
    set_progress_callback,
    set_preview_callback,
)

__all__ = [
    # Main classes
    "SDContext",
    "SDContextParams",
    "SDImage",
    "SDImageGenParams",
    "SDSampleParams",
    "Upscaler",
    # Enums
    "RngType",
    "SampleMethod",
    "Scheduler",
    "Prediction",
    "SDType",
    "LogLevel",
    "PreviewMode",
    "LoraApplyMode",
    # Convenience functions
    "text_to_image",
    "text_to_images",
    "image_to_image",
    # Model utilities
    "convert_model",
    "canny_preprocess",
    # Backend loading
    "ggml_backend_load_all",
    # Utility functions
    "get_num_cores",
    "get_system_info",
    "type_name",
    "sample_method_name",
    "scheduler_name",
    # Callback setters
    "set_log_callback",
    "set_progress_callback",
    "set_preview_callback",
]
