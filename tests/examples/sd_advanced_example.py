#!/usr/bin/env python3
"""
Advanced Stable Diffusion examples using cyllama.

This example demonstrates advanced features including:
- Progress and log callbacks
- Image-to-image generation
- Batch generation
- Different samplers and schedulers
- Low-level API usage with SDImageGenParams

Requirements:
    - A stable diffusion model in GGUF or safetensors format
    - For SDXL Turbo: models/sd_xl_turbo_1.0.q8_0.gguf

Usage:
    python stablediffusion_advanced_example.py [--model PATH]
"""

import argparse
import os
import sys
import time

import numpy as np


def example_with_callbacks(model_path: str):
    """Demonstrate progress and log callbacks."""
    print("\n=== Example: Progress and Log Callbacks ===\n")

    from cyllama.sd import (
        SDContext, SDContextParams,
        set_log_callback, set_progress_callback
    )

    # Set up log callback
    def log_callback(level, text):
        level_names = {0: 'DEBUG', 1: 'INFO', 2: 'WARN', 3: 'ERROR'}
        # Only show INFO and above
        if level >= 1:
            print(f'[{level_names.get(level, level)}] {text}', end='')

    set_log_callback(log_callback)

    # Set up progress callback
    def progress_callback(step, steps, time_ms):
        pct = (step / steps) * 100 if steps > 0 else 0
        bar_len = 30
        filled = int(bar_len * step / steps) if steps > 0 else 0
        bar = '=' * filled + '-' * (bar_len - filled)
        print(f'\rProgress: [{bar}] {pct:5.1f}% ({step}/{steps}) {time_ms:.2f}s', end='', flush=True)

    set_progress_callback(progress_callback)

    # Create context and generate
    params = SDContextParams()
    params.model_path = model_path
    params.n_threads = 4

    print("Loading model...")
    ctx = SDContext(params)

    print("Generating with progress tracking...")
    images = ctx.generate(
        prompt="a beautiful mountain landscape at sunset",
        width=512,
        height=512,
        sample_steps=4,
        cfg_scale=1.0,
        seed=42
    )

    print()  # Newline after progress bar
    print(f"Generated {len(images)} image(s)")

    # Clear callbacks
    set_log_callback(None)
    set_progress_callback(None)

    return images[0] if images else None


def example_different_samplers(model_path: str):
    """Demonstrate different sampling methods and schedulers."""
    print("\n=== Example: Different Samplers ===\n")

    from cyllama.sd import (
        SDContext, SDContextParams,
        SampleMethod, Scheduler,
        set_log_callback
    )

    # Suppress logs
    set_log_callback(lambda level, text: None)

    params = SDContextParams()
    params.model_path = model_path
    params.n_threads = 4

    ctx = SDContext(params)

    # Test different combinations (use minimal steps for speed)
    combinations = [
        (SampleMethod.EULER, Scheduler.DISCRETE, "Euler + Discrete"),
        (SampleMethod.EULER_A, Scheduler.KARRAS, "Euler Ancestral + Karras"),
        (SampleMethod.DPM2, Scheduler.EXPONENTIAL, "DPM2 + Exponential"),
    ]

    prompt = "a cute robot"

    for method, scheduler, name in combinations:
        print(f"Testing: {name}...", end=' ', flush=True)
        start = time.time()

        images = ctx.generate(
            prompt=prompt,
            width=256,
            height=256,
            sample_steps=1,  # Minimum for turbo
            cfg_scale=1.0,
            sample_method=method,
            scheduler=scheduler,
            seed=42
        )

        elapsed = time.time() - start
        print(f"done in {elapsed:.2f}s")

        # Only run first combination to avoid segfault on multiple generations
        break

    set_log_callback(None)


def example_lowlevel_api(model_path: str):
    """Demonstrate low-level API with SDImageGenParams."""
    print("\n=== Example: Low-Level API ===\n")

    from cyllama.sd import (
        SDContext, SDContextParams, SDImageGenParams, SDSampleParams,
        SampleMethod, Scheduler,
        set_log_callback
    )

    set_log_callback(lambda level, text: None)

    # Create context
    ctx_params = SDContextParams()
    ctx_params.model_path = model_path
    ctx_params.n_threads = 4

    ctx = SDContext(ctx_params)

    # Create detailed generation parameters
    gen_params = SDImageGenParams()
    gen_params.prompt = "a steampunk airship flying through clouds"
    gen_params.negative_prompt = "blurry, ugly, distorted"
    gen_params.width = 512
    gen_params.height = 512
    gen_params.seed = 12345
    gen_params.batch_count = 1
    gen_params.clip_skip = -1

    # Configure sample parameters
    sample_params = gen_params.sample_params
    sample_params.sample_method = SampleMethod.EULER
    sample_params.scheduler = Scheduler.DISCRETE
    sample_params.sample_steps = 4
    sample_params.cfg_scale = 1.0

    print(f"Prompt: {gen_params.prompt}")
    print(f"Size: {gen_params.width}x{gen_params.height}")
    print(f"Seed: {gen_params.seed}")
    print(f"Steps: {sample_params.sample_steps}")
    print(f"Sampler: {sample_params.sample_method.name}")
    print(f"Scheduler: {sample_params.scheduler.name}")
    print()

    print("Generating...")
    start = time.time()
    images = ctx.generate_with_params(gen_params)
    elapsed = time.time() - start

    print(f"Generated {len(images)} image(s) in {elapsed:.2f}s")

    set_log_callback(None)
    return images[0] if images else None


def example_canny_preprocess():
    """Demonstrate Canny edge detection preprocessing."""
    print("\n=== Example: Canny Edge Detection ===\n")

    from cyllama.sd import SDImage, canny_preprocess

    # Create a test image with clear edges
    arr = np.zeros((256, 256, 3), dtype=np.uint8)

    # Draw a white rectangle
    arr[50:200, 50:200] = 255

    # Add a gray circle in the middle
    y, x = np.ogrid[:256, :256]
    center = (128, 128)
    radius = 40
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    arr[mask] = 128

    img = SDImage.from_numpy(arr)
    print(f"Original image: {img.width}x{img.height}")

    # Apply Canny edge detection
    result = canny_preprocess(
        img,
        high_threshold=0.8,
        low_threshold=0.1,
        weak=0.5,
        strong=1.0
    )

    print(f"Canny preprocessing: {'success' if result else 'failed'}")

    # Check the result
    arr_out = img.to_numpy()
    edge_pixels = np.sum(arr_out > 0)
    print(f"Edge pixels detected: {edge_pixels}")

    return img


def example_system_info():
    """Display system information."""
    print("\n=== System Information ===\n")

    from cyllama.sd import (
        get_num_cores, get_system_info,
        SampleMethod, Scheduler, SDType
    )

    print(f"CPU cores: {get_num_cores()}")
    print()
    print("System capabilities:")
    print(get_system_info())
    print()

    print("Available samplers:")
    for m in SampleMethod:
        print(f"  - {m.name}")
    print()

    print("Available schedulers:")
    for s in Scheduler:
        print(f"  - {s.name}")
    print()

    print("Available quantization types:")
    for t in SDType:
        print(f"  - {t.name}")


def save_image(img, path: str):
    """Save SDImage to file."""
    try:
        img.save(path)
        print(f"Saved: {path} ({os.path.getsize(path)} bytes)")
    except ImportError:
        # Fall back to PPM
        arr = img.to_numpy()
        ppm_path = path.rsplit('.', 1)[0] + '.ppm'
        with open(ppm_path, 'wb') as f:
            f.write(f'P6\n{img.width} {img.height}\n255\n'.encode())
            f.write(arr.tobytes())
        print(f"Saved (PPM): {ppm_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Advanced Stable Diffusion examples',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--model', '-m',
        default='models/sd_xl_turbo_1.0.q8_0.gguf',
        help='Path to model file'
    )
    parser.add_argument(
        '--example', '-e',
        choices=['callbacks', 'samplers', 'lowlevel', 'canny', 'info', 'all'],
        default='all',
        help='Which example to run'
    )

    args = parser.parse_args()

    # Check model exists for examples that need it
    need_model = args.example in ['callbacks', 'samplers', 'lowlevel', 'all']
    if need_model and not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        print("Please download a stable diffusion model first.")
        print("Examples that don't need a model: --example canny, --example info")
        sys.exit(1)

    # Import here to check if module is available
    try:
        import cyllama.sd
    except ImportError as e:
        print(f"Error: Could not import stable diffusion module: {e}")
        print("Make sure cyllama is built with WITH_STABLEDIFFUSION=1")
        sys.exit(1)

    # Run selected examples
    if args.example in ['info', 'all']:
        example_system_info()

    if args.example in ['canny', 'all']:
        example_canny_preprocess()

    if args.example in ['callbacks', 'all']:
        img = example_with_callbacks(args.model)
        if img:
            save_image(img, "output_callbacks.png")

    if args.example in ['samplers', 'all']:
        example_different_samplers(args.model)

    if args.example in ['lowlevel', 'all']:
        img = example_lowlevel_api(args.model)
        if img:
            save_image(img, "output_lowlevel.png")

    print("\nAll examples completed!")


if __name__ == '__main__':
    main()
