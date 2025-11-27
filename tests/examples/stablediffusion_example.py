#!/usr/bin/env python3
"""
Stable Diffusion example using cyllama.

This example demonstrates how to use the stable diffusion module for image generation.

Requirements:
    - A stable diffusion model in GGUF or safetensors format
    - For SDXL Turbo: models/sd_xl_turbo_1.0.q8_0.gguf (or similar)

Usage:
    python stablediffusion_example.py [--model PATH] [--prompt TEXT] [--output PATH]

Examples:
    # Basic usage with default model
    python stablediffusion_example.py

    # Custom prompt
    python stablediffusion_example.py --prompt "a beautiful sunset over mountains"

    # Different model
    python stablediffusion_example.py --model models/sd-v1-5.safetensors --steps 20
"""

import argparse
import os
import sys
import time

import numpy as np


def save_ppm(arr: np.ndarray, path: str) -> None:
    """Save numpy array as PPM image (no PIL required)."""
    height, width, channels = arr.shape
    with open(path, 'wb') as f:
        f.write(f'P6\n{width} {height}\n255\n'.encode())
        f.write(arr.tobytes())
    print(f"Saved: {path} ({os.path.getsize(path)} bytes)")


def save_image(img, path: str) -> None:
    """Save SDImage to file, using PIL if available, otherwise PPM."""
    try:
        # Try PIL first for PNG support
        img.save(path)
        print(f"Saved: {path} ({os.path.getsize(path)} bytes)")
    except ImportError:
        # Fall back to PPM format
        arr = img.to_numpy()
        ppm_path = path.rsplit('.', 1)[0] + '.ppm'
        save_ppm(arr, ppm_path)


def main():
    parser = argparse.ArgumentParser(
        description='Generate images using Stable Diffusion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--model', '-m',
        default='models/sd_xl_turbo_1.0.q8_0.gguf',
        help='Path to model file (default: models/sd_xl_turbo_1.0.q8_0.gguf)'
    )
    parser.add_argument(
        '--prompt', '-p',
        default='a photo of a cute cat sitting on a windowsill, sunlight, highly detailed',
        help='Text prompt for image generation'
    )
    parser.add_argument(
        '--negative', '-n',
        default='blurry, low quality, distorted, ugly',
        help='Negative prompt'
    )
    parser.add_argument(
        '--output', '-o',
        default='output.png',
        help='Output image path (default: output.png)'
    )
    parser.add_argument(
        '--width', '-W',
        type=int, default=512,
        help='Image width (default: 512)'
    )
    parser.add_argument(
        '--height', '-H',
        type=int, default=512,
        help='Image height (default: 512)'
    )
    parser.add_argument(
        '--steps', '-s',
        type=int, default=4,
        help='Number of sampling steps (default: 4 for turbo models, use 20+ for others)'
    )
    parser.add_argument(
        '--cfg', '-c',
        type=float, default=1.0,
        help='CFG scale (default: 1.0 for turbo, use 7.0 for others)'
    )
    parser.add_argument(
        '--seed',
        type=int, default=-1,
        help='Random seed (-1 for random)'
    )
    parser.add_argument(
        '--threads', '-t',
        type=int, default=4,
        help='Number of threads (default: 4)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Check model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        print("\nPlease download a stable diffusion model, for example:")
        print("  - SDXL Turbo (recommended): sd_xl_turbo_1.0.q8_0.gguf")
        print("  - SD 1.5: sd-v1-5.safetensors")
        print("\nSee: https://huggingface.co/stabilityai")
        sys.exit(1)

    # Import stable diffusion module
    try:
        from cyllama.stablediffusion import (
            SDContext, SDContextParams,
            SampleMethod, Scheduler,
            set_log_callback
        )
    except ImportError as e:
        print(f"Error: Could not import stable diffusion module: {e}")
        print("Make sure cyllama is built with WITH_STABLEDIFFUSION=1")
        sys.exit(1)

    # Set up logging
    if args.verbose:
        def log_callback(level, text):
            level_names = {0: 'DEBUG', 1: 'INFO', 2: 'WARN', 3: 'ERROR'}
            print(f'[{level_names.get(level, level)}] {text}', end='')
        set_log_callback(log_callback)
    else:
        # Only show warnings and errors
        def log_callback(level, text):
            if level >= 2:
                level_names = {2: 'WARN', 3: 'ERROR'}
                print(f'[{level_names.get(level, level)}] {text}', end='')
        set_log_callback(log_callback)

    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print(f"Size: {args.width}x{args.height}")
    print(f"Steps: {args.steps}, CFG: {args.cfg}")
    print()

    # Create context
    print("Loading model...")
    start = time.time()

    params = SDContextParams()
    params.model_path = args.model
    params.n_threads = args.threads

    try:
        ctx = SDContext(params)
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")
    print()

    # Generate image
    print("Generating image...")
    start = time.time()

    try:
        images = ctx.generate(
            prompt=args.prompt,
            negative_prompt=args.negative,
            width=args.width,
            height=args.height,
            seed=args.seed,
            sample_steps=args.steps,
            cfg_scale=args.cfg,
            sample_method=SampleMethod.EULER,
            scheduler=Scheduler.DISCRETE
        )
    except RuntimeError as e:
        print(f"Error generating image: {e}")
        sys.exit(1)

    gen_time = time.time() - start
    print(f"Generated {len(images)} image(s) in {gen_time:.2f}s")
    print()

    # Save output
    if images:
        img = images[0]
        print(f"Image size: {img.width}x{img.height}x{img.channels}")

        # Get numpy array for stats
        arr = img.to_numpy()
        print(f"Pixel stats: min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}")

        # Save
        save_image(img, args.output)

    print()
    print(f"Total time: {load_time + gen_time:.2f}s")


if __name__ == '__main__':
    main()
