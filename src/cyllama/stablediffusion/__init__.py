"""
Stable Diffusion module for cyllama.

Provides Python bindings for stable-diffusion.cpp image generation.

Example:
    from cyllama.stablediffusion import text_to_image, SDContext, SDContextParams

    # Simple usage
    images = text_to_image(
        model_path="sd-v1-5.safetensors",
        prompt="a photo of a cat",
        width=512,
        height=512
    )
    images[0].save("output.png")

    # With model reuse
    params = SDContextParams(model_path="sd-v1-5.safetensors")
    with SDContext(params) as ctx:
        for prompt in prompts:
            images = ctx.generate(prompt)
"""

from .stable_diffusion import (
    # Main classes
    SDContext,
    SDContextParams,
    SDImage,
    SDImageGenParams,
    SDSampleParams,

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
    image_to_image,

    # Utility functions
    get_num_cores,
    get_system_info,
    type_name,
    sample_method_name,
    scheduler_name,

    # Callback setters
    set_log_callback,
    set_progress_callback,
)

__all__ = [
    # Main classes
    "SDContext",
    "SDContextParams",
    "SDImage",
    "SDImageGenParams",
    "SDSampleParams",

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
    "image_to_image",

    # Utility functions
    "get_num_cores",
    "get_system_info",
    "type_name",
    "sample_method_name",
    "scheduler_name",

    # Callback setters
    "set_log_callback",
    "set_progress_callback",
]
