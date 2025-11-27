#!/usr/bin/env python3
"""
CLI tool for stable diffusion image generation.

Usage:
    python -m cyllama.stablediffusion generate --model MODEL --prompt "..." [options]
    python -m cyllama.stablediffusion upscale --model MODEL --input IMAGE [options]
    python -m cyllama.stablediffusion convert --input MODEL --output MODEL [options]
    python -m cyllama.stablediffusion info

Examples:
    # Generate an image
    python -m cyllama.stablediffusion generate \\
        --model models/sd_xl_turbo_1.0.q8_0.gguf \\
        --prompt "a photo of a cat" \\
        --output cat.png

    # Upscale an image
    python -m cyllama.stablediffusion upscale \\
        --model models/esrgan-x4.bin \\
        --input cat.png \\
        --output cat_upscaled.png

    # Convert model to different quantization
    python -m cyllama.stablediffusion convert \\
        --input sd-v1-5.safetensors \\
        --output sd-v1-5-q4_0.gguf \\
        --type q4_0
"""

import argparse
import os
import sys
import time


def save_image(img, path: str) -> None:
    """Save SDImage to file."""
    try:
        img.save(path)
    except ImportError:
        # Fall back to PPM if PIL not available
        import numpy as np
        arr = img.to_numpy()
        ppm_path = path.rsplit('.', 1)[0] + '.ppm'
        with open(ppm_path, 'wb') as f:
            f.write(f'P6\n{img.width} {img.height}\n255\n'.encode())
            f.write(arr.tobytes())
        print(f"Note: PIL not available, saved as PPM: {ppm_path}")
        return
    print(f"Saved: {path}")


def cmd_generate(args):
    """Generate images from text prompt."""
    from .stable_diffusion import (
        SDContext, SDContextParams, SampleMethod, Scheduler,
        set_log_callback, set_progress_callback
    )

    # Setup logging
    if args.verbose:
        def log_cb(level, text):
            level_names = {0: 'DEBUG', 1: 'INFO', 2: 'WARN', 3: 'ERROR'}
            print(f'[{level_names.get(level, level)}] {text}', end='')
        set_log_callback(log_cb)
    else:
        def log_cb(level, text):
            if level >= 2:
                print(f'[{"WARN" if level == 2 else "ERROR"}] {text}', end='')
        set_log_callback(log_cb)

    # Setup progress callback
    if args.progress:
        def progress_cb(step, steps, time_ms):
            pct = (step / steps) * 100 if steps > 0 else 0
            print(f'\rStep {step}/{steps} ({pct:.1f}%) - {time_ms:.2f}s', end='', flush=True)
        set_progress_callback(progress_cb)

    # Load model
    print(f"Loading model: {args.model}")
    start = time.time()

    params = SDContextParams()
    params.model_path = args.model
    params.n_threads = args.threads
    if args.vae:
        params.vae_path = args.vae
    if args.clip_l:
        params.clip_l_path = args.clip_l
    if args.clip_g:
        params.clip_g_path = args.clip_g
    if args.t5xxl:
        params.t5xxl_path = args.t5xxl
    if args.lora_dir:
        params.lora_model_dir = args.lora_dir

    try:
        ctx = SDContext(params)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")

    # Parse sample method and scheduler
    sample_method = None
    if args.sampler:
        try:
            sample_method = SampleMethod[args.sampler.upper()]
        except KeyError:
            print(f"Unknown sampler: {args.sampler}", file=sys.stderr)
            print(f"Available: {[m.name for m in SampleMethod]}")
            return 1

    scheduler = None
    if args.scheduler:
        try:
            scheduler = Scheduler[args.scheduler.upper()]
        except KeyError:
            print(f"Unknown scheduler: {args.scheduler}", file=sys.stderr)
            print(f"Available: {[s.name for s in Scheduler]}")
            return 1

    # Generate
    print(f"Generating {args.batch} image(s)...")
    print(f"  Prompt: {args.prompt}")
    print(f"  Size: {args.width}x{args.height}")
    print(f"  Steps: {args.steps}, CFG: {args.cfg}")

    start = time.time()
    try:
        images = ctx.generate(
            prompt=args.prompt,
            negative_prompt=args.negative or "",
            width=args.width,
            height=args.height,
            seed=args.seed,
            batch_count=args.batch,
            sample_steps=args.steps,
            cfg_scale=args.cfg,
            sample_method=sample_method,
            scheduler=scheduler,
            clip_skip=args.clip_skip
        )
    except RuntimeError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1

    gen_time = time.time() - start
    if args.progress:
        print()  # Newline after progress
    print(f"Generated {len(images)} image(s) in {gen_time:.2f}s")

    # Save images
    for i, img in enumerate(images):
        if args.batch == 1:
            output_path = args.output
        else:
            base, ext = os.path.splitext(args.output)
            output_path = f"{base}_{i+1}{ext}"
        save_image(img, output_path)

    return 0


def cmd_upscale(args):
    """Upscale an image using ESRGAN."""
    from .stable_diffusion import Upscaler, SDImage, set_log_callback

    if args.verbose:
        def log_cb(level, text):
            level_names = {0: 'DEBUG', 1: 'INFO', 2: 'WARN', 3: 'ERROR'}
            print(f'[{level_names.get(level, level)}] {text}', end='')
        set_log_callback(log_cb)

    # Load input image
    print(f"Loading image: {args.input}")
    try:
        input_img = SDImage.load(args.input)
    except Exception as e:
        print(f"Error loading image: {e}", file=sys.stderr)
        return 1

    print(f"Input size: {input_img.width}x{input_img.height}")

    # Load upscaler
    print(f"Loading upscaler: {args.model}")
    try:
        upscaler = Upscaler(args.model, n_threads=args.threads)
    except Exception as e:
        print(f"Error loading upscaler: {e}", file=sys.stderr)
        return 1

    print(f"Upscale factor: {upscaler.upscale_factor}x")

    # Upscale
    print("Upscaling...")
    start = time.time()
    try:
        output_img = upscaler.upscale(input_img, factor=args.factor or 0)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    elapsed = time.time() - start
    print(f"Upscaled to {output_img.width}x{output_img.height} in {elapsed:.2f}s")

    # Save
    save_image(output_img, args.output)
    return 0


def cmd_convert(args):
    """Convert model to different format/quantization."""
    from .stable_diffusion import convert_model, SDType, set_log_callback

    if args.verbose:
        def log_cb(level, text):
            level_names = {0: 'DEBUG', 1: 'INFO', 2: 'WARN', 3: 'ERROR'}
            print(f'[{level_names.get(level, level)}] {text}', end='')
        set_log_callback(log_cb)

    # Parse output type
    try:
        output_type = SDType[args.type.upper()]
    except KeyError:
        print(f"Unknown type: {args.type}", file=sys.stderr)
        print(f"Available: {[t.name for t in SDType]}")
        return 1

    print(f"Converting: {args.input} -> {args.output}")
    print(f"Output type: {args.type}")

    start = time.time()
    try:
        convert_model(
            input_path=args.input,
            output_path=args.output,
            output_type=output_type,
            vae_path=args.vae
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    elapsed = time.time() - start
    print(f"Conversion completed in {elapsed:.2f}s")
    print(f"Output: {args.output} ({os.path.getsize(args.output) / 1e6:.1f} MB)")
    return 0


def cmd_info(args):
    """Show system info and available features."""
    from .stable_diffusion import get_num_cores, get_system_info

    print("Stable Diffusion Module Info")
    print("=" * 40)
    print(f"CPU cores: {get_num_cores()}")
    print()
    print("System info:")
    print(get_system_info())
    print()

    # Show available samplers and schedulers
    from .stable_diffusion import SampleMethod, Scheduler, SDType

    print("Available samplers:")
    for m in SampleMethod:
        print(f"  - {m.name.lower()}")
    print()

    print("Available schedulers:")
    for s in Scheduler:
        print(f"  - {s.name.lower()}")
    print()

    print("Available quantization types:")
    for t in SDType:
        print(f"  - {t.name.lower()}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Stable Diffusion CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate images from text')
    gen_parser.add_argument('--model', '-m', required=True, help='Path to model file')
    gen_parser.add_argument('--prompt', '-p', required=True, help='Text prompt')
    gen_parser.add_argument('--negative', '-n', help='Negative prompt')
    gen_parser.add_argument('--output', '-o', default='output.png', help='Output path')
    gen_parser.add_argument('--width', '-W', type=int, default=512, help='Image width')
    gen_parser.add_argument('--height', '-H', type=int, default=512, help='Image height')
    gen_parser.add_argument('--steps', '-s', type=int, default=20, help='Sampling steps')
    gen_parser.add_argument('--cfg', '-c', type=float, default=7.0, help='CFG scale')
    gen_parser.add_argument('--seed', type=int, default=-1, help='Random seed')
    gen_parser.add_argument('--batch', '-b', type=int, default=1, help='Batch count')
    gen_parser.add_argument('--sampler', help='Sampling method (euler, euler_a, etc.)')
    gen_parser.add_argument('--scheduler', help='Scheduler (discrete, karras, etc.)')
    gen_parser.add_argument('--threads', '-t', type=int, default=-1, help='Number of threads')
    gen_parser.add_argument('--vae', help='Path to VAE model')
    gen_parser.add_argument('--clip-l', help='Path to CLIP-L model')
    gen_parser.add_argument('--clip-g', help='Path to CLIP-G model')
    gen_parser.add_argument('--t5xxl', help='Path to T5-XXL model')
    gen_parser.add_argument('--lora-dir', help='Directory containing LoRA files')
    gen_parser.add_argument('--clip-skip', type=int, default=-1, help='CLIP skip layers')
    gen_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    gen_parser.add_argument('--progress', action='store_true', help='Show progress')

    # Upscale command
    up_parser = subparsers.add_parser('upscale', help='Upscale an image')
    up_parser.add_argument('--model', '-m', required=True, help='Path to ESRGAN model')
    up_parser.add_argument('--input', '-i', required=True, help='Input image path')
    up_parser.add_argument('--output', '-o', required=True, help='Output image path')
    up_parser.add_argument('--factor', '-f', type=int, help='Upscale factor (default: model default)')
    up_parser.add_argument('--threads', '-t', type=int, default=-1, help='Number of threads')
    up_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    # Convert command
    conv_parser = subparsers.add_parser('convert', help='Convert model format')
    conv_parser.add_argument('--input', '-i', required=True, help='Input model path')
    conv_parser.add_argument('--output', '-o', required=True, help='Output model path')
    conv_parser.add_argument('--type', '-t', default='f16', help='Output type (f16, q4_0, q8_0, etc.)')
    conv_parser.add_argument('--vae', help='Path to VAE model')
    conv_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show system info')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == 'generate':
        return cmd_generate(args)
    elif args.command == 'upscale':
        return cmd_upscale(args)
    elif args.command == 'convert':
        return cmd_convert(args)
    elif args.command == 'info':
        return cmd_info(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
