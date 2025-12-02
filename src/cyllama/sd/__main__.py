#!/usr/bin/env python3
"""
CLI tool for stable diffusion image generation.

Usage:
    python -m cyllama.sd txt2img --model MODEL --prompt "..." [options]
    python -m cyllama.sd img2img --model MODEL --init-img IMAGE --prompt "..." [options]
    python -m cyllama.sd inpaint --model MODEL --init-img IMAGE --mask MASK --prompt "..." [options]
    python -m cyllama.sd controlnet --model MODEL --control-net CN --control-image IMAGE --prompt "..." [options]
    python -m cyllama.sd video --model MODEL --prompt "..." [options]
    python -m cyllama.sd upscale --model MODEL --input IMAGE [options]
    python -m cyllama.sd convert --input MODEL --output MODEL [options]
    python -m cyllama.sd info

Examples:
    # Text to image
    python -m cyllama.sd txt2img \\
        --model models/sd_xl_turbo_1.0.q8_0.gguf \\
        --prompt "a photo of a cat" \\
        --output cat.png

    # Image to image
    python -m cyllama.sd img2img \\
        --model models/sd-v1-5.gguf \\
        --init-img input.png \\
        --prompt "oil painting style" \\
        --strength 0.7

    # Inpainting
    python -m cyllama.sd inpaint \\
        --model models/sd-inpaint.gguf \\
        --init-img photo.png \\
        --mask mask.png \\
        --prompt "a red hat"

    # ControlNet
    python -m cyllama.sd controlnet \\
        --model models/sd-v1-5.gguf \\
        --control-net models/control_canny.gguf \\
        --control-image edges.png \\
        --prompt "a beautiful landscape"

    # Video generation
    python -m cyllama.sd video \\
        --model models/wan2.1.gguf \\
        --prompt "a cat walking" \\
        --video-frames 16

    # Upscale
    python -m cyllama.sd upscale \\
        --model models/esrgan-x4.bin \\
        --input cat.png \\
        --output cat_4x.png

    # Convert model
    python -m cyllama.sd convert \\
        --input sd-v1-5.safetensors \\
        --output sd-v1-5-q8_0.gguf \\
        --type q8_0
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
        arr = img.to_numpy()
        ppm_path = path.rsplit('.', 1)[0] + '.ppm'
        with open(ppm_path, 'wb') as f:
            f.write(f'P6\n{img.width} {img.height}\n255\n'.encode())
            f.write(arr.tobytes())
        print(f"Note: PIL not available, saved as PPM: {ppm_path}")
        return
    print(f"Saved: {path}")


def save_video_frames(frames, output_path: str, fps: int = 24) -> None:
    """Save video frames to files or video."""
    base, ext = os.path.splitext(output_path)

    # Save as individual frames
    for i, frame in enumerate(frames):
        frame_path = f"{base}_{i:04d}.png"
        save_image(frame, frame_path)

    print(f"Saved {len(frames)} frames to {base}_*.png")


def setup_logging(args):
    """Setup logging and progress callbacks."""
    from .stable_diffusion import set_log_callback, set_progress_callback

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

    if args.progress:
        def progress_cb(step, steps, time_ms):
            pct = (step / steps) * 100 if steps > 0 else 0
            print(f'\rStep {step}/{steps} ({pct:.1f}%) - {time_ms:.2f}s', end='', flush=True)
        set_progress_callback(progress_cb)


def setup_preview(args):
    """Setup preview callback if requested."""
    if not hasattr(args, 'preview') or not args.preview or args.preview == 'none':
        return

    from .stable_diffusion import set_preview_callback, PreviewMode

    mode_map = {
        'proj': PreviewMode.PROJ,
        'tae': PreviewMode.TAE,
        'vae': PreviewMode.VAE,
    }
    mode = mode_map.get(args.preview, PreviewMode.NONE)

    preview_path = getattr(args, 'preview_path', './preview.png')
    interval = getattr(args, 'preview_interval', 1)

    def preview_cb(step, frames, is_noisy):
        if frames:
            frames[0].save(preview_path)
            print(f"\rPreview saved: {preview_path} (step {step})", end='', flush=True)

    set_preview_callback(
        preview_cb,
        mode=mode,
        interval=interval,
        denoised=not getattr(args, 'preview_noisy', False),
        noisy=getattr(args, 'preview_noisy', False)
    )


def validate_model_args(args):
    """Validate that at least one model path is provided."""
    model = getattr(args, 'model', None)
    diffusion_model = getattr(args, 'diffusion_model', None)
    if not model and not diffusion_model:
        print("Error: Either --model or --diffusion-model is required", file=sys.stderr)
        sys.exit(1)


def create_context_params(args):
    """Create SDContextParams from CLI args."""
    from .stable_diffusion import (
        SDContextParams, RngType, Prediction, LoraApplyMode
    )

    params = SDContextParams()
    if hasattr(args, 'model') and args.model:
        params.model_path = args.model
    params.n_threads = args.threads

    # Model paths
    if hasattr(args, 'vae') and args.vae:
        params.vae_path = args.vae
    if hasattr(args, 'taesd') and args.taesd:
        params.taesd_path = args.taesd
    if hasattr(args, 'clip_l') and args.clip_l:
        params.clip_l_path = args.clip_l
    if hasattr(args, 'clip_g') and args.clip_g:
        params.clip_g_path = args.clip_g
    if hasattr(args, 'clip_vision') and args.clip_vision:
        params.clip_vision_path = args.clip_vision
    if hasattr(args, 't5xxl') and args.t5xxl:
        params.t5xxl_path = args.t5xxl
    if hasattr(args, 'llm') and args.llm:
        params.llm_path = args.llm
    if hasattr(args, 'llm_vision') and args.llm_vision:
        params.llm_vision_path = args.llm_vision
    if hasattr(args, 'diffusion_model') and args.diffusion_model:
        params.diffusion_model_path = args.diffusion_model
    if hasattr(args, 'high_noise_diffusion_model') and args.high_noise_diffusion_model:
        params.high_noise_diffusion_model_path = args.high_noise_diffusion_model
    if hasattr(args, 'control_net') and args.control_net:
        params.control_net_path = args.control_net
    if hasattr(args, 'lora_dir') and args.lora_dir:
        params.lora_model_dir = args.lora_dir
    if hasattr(args, 'embd_dir') and args.embd_dir:
        params.embedding_dir = args.embd_dir
    if hasattr(args, 'photo_maker') and args.photo_maker:
        params.photo_maker_path = args.photo_maker
    if hasattr(args, 'tensor_type_rules') and args.tensor_type_rules:
        params.tensor_type_rules = args.tensor_type_rules

    # Memory/performance options
    if hasattr(args, 'offload_to_cpu') and args.offload_to_cpu:
        params.offload_params_to_cpu = True
    if hasattr(args, 'clip_on_cpu') and args.clip_on_cpu:
        params.keep_clip_on_cpu = True
    if hasattr(args, 'vae_on_cpu') and args.vae_on_cpu:
        params.keep_vae_on_cpu = True
    if hasattr(args, 'control_net_cpu') and args.control_net_cpu:
        params.keep_control_net_on_cpu = True
    if hasattr(args, 'diffusion_fa') and args.diffusion_fa:
        params.diffusion_flash_attn = True
    if hasattr(args, 'diffusion_conv_direct') and args.diffusion_conv_direct:
        params.diffusion_conv_direct = True
    if hasattr(args, 'vae_conv_direct') and args.vae_conv_direct:
        params.vae_conv_direct = True

    # RNG and prediction
    if hasattr(args, 'rng') and args.rng:
        rng_map = {'std_default': RngType.STD_DEFAULT, 'cuda': RngType.CUDA, 'cpu': RngType.CPU}
        params.rng_type = rng_map.get(args.rng, RngType.CUDA)
    if hasattr(args, 'sampler_rng') and args.sampler_rng:
        rng_map = {'std_default': RngType.STD_DEFAULT, 'cuda': RngType.CUDA, 'cpu': RngType.CPU}
        params.sampler_rng_type = rng_map.get(args.sampler_rng, RngType.CUDA)
    if hasattr(args, 'prediction') and args.prediction:
        pred_map = {
            'eps': Prediction.EPS, 'v': Prediction.V, 'edm_v': Prediction.EDM_V,
            'sd3_flow': Prediction.SD3_FLOW, 'flux_flow': Prediction.FLUX_FLOW,
            'flux2_flow': Prediction.FLUX2_FLOW
        }
        params.prediction = pred_map.get(args.prediction, Prediction.DEFAULT)
    if hasattr(args, 'lora_apply_mode') and args.lora_apply_mode:
        mode_map = {
            'auto': LoraApplyMode.AUTO, 'immediately': LoraApplyMode.IMMEDIATELY,
            'at_runtime': LoraApplyMode.AT_RUNTIME
        }
        params.lora_apply_mode = mode_map.get(args.lora_apply_mode, LoraApplyMode.AUTO)

    # Flow/chroma options
    if hasattr(args, 'flow_shift') and args.flow_shift is not None:
        params.flow_shift = args.flow_shift
    if hasattr(args, 'chroma_disable_dit_mask') and args.chroma_disable_dit_mask:
        params.chroma_use_dit_mask = False
    if hasattr(args, 'chroma_enable_t5_mask') and args.chroma_enable_t5_mask:
        params.chroma_use_t5_mask = True
    if hasattr(args, 'chroma_t5_mask_pad') and args.chroma_t5_mask_pad:
        params.chroma_t5_mask_pad = args.chroma_t5_mask_pad

    # TAESD options
    if hasattr(args, 'taesd_preview_only') and args.taesd_preview_only:
        params.tae_preview_only = True

    return params


def parse_sampler_scheduler(args):
    """Parse sampler and scheduler from args."""
    from .stable_diffusion import SampleMethod, Scheduler

    sample_method = None
    if hasattr(args, 'sampler') and args.sampler:
        try:
            sample_method = SampleMethod[args.sampler.upper()]
        except KeyError:
            print(f"Unknown sampler: {args.sampler}", file=sys.stderr)
            print(f"Available: {[m.name.lower() for m in SampleMethod]}")
            sys.exit(1)

    scheduler = None
    if hasattr(args, 'scheduler') and args.scheduler:
        try:
            scheduler = Scheduler[args.scheduler.upper()]
        except KeyError:
            print(f"Unknown scheduler: {args.scheduler}", file=sys.stderr)
            print(f"Available: {[s.name.lower() for s in Scheduler]}")
            sys.exit(1)

    return sample_method, scheduler


def save_outputs(images, args):
    """Save generated images."""
    batch = len(images)
    for i, img in enumerate(images):
        if batch == 1:
            output_path = args.output
        else:
            base, ext = os.path.splitext(args.output)
            output_path = f"{base}_{i+1}{ext}"
        save_image(img, output_path)


# =============================================================================
# Command handlers
# =============================================================================

def cmd_txt2img(args):
    """Generate images from text prompt."""
    from .stable_diffusion import SDContext

    validate_model_args(args)
    setup_logging(args)
    setup_preview(args)

    model_name = args.model or args.diffusion_model
    print(f"Loading model: {model_name}")
    start = time.time()

    params = create_context_params(args)

    try:
        ctx = SDContext(params)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")

    sample_method, scheduler = parse_sampler_scheduler(args)

    print(f"Generating {args.batch} image(s)...")
    print(f"  Prompt: {args.prompt}")
    print(f"  Size: {args.width}x{args.height}")
    print(f"  Steps: {args.steps}, CFG: {args.cfg_scale}")

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
            cfg_scale=args.cfg_scale,
            sample_method=sample_method,
            scheduler=scheduler,
            clip_skip=args.clip_skip,
            eta=args.eta,
            slg_scale=args.slg_scale,
            vae_tiling=args.vae_tiling,
        )
    except RuntimeError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1

    gen_time = time.time() - start
    if args.progress:
        print()
    print(f"Generated {len(images)} image(s) in {gen_time:.2f}s")

    save_outputs(images, args)
    return 0


def cmd_img2img(args):
    """Generate images from init image + prompt."""
    from .stable_diffusion import SDContext, SDImage

    validate_model_args(args)
    setup_logging(args)
    setup_preview(args)

    # Load init image
    print(f"Loading init image: {args.init_img}")
    try:
        init_image = SDImage.load(args.init_img)
    except Exception as e:
        print(f"Error loading init image: {e}", file=sys.stderr)
        return 1
    print(f"  Size: {init_image.width}x{init_image.height}")

    model_name = args.model or args.diffusion_model
    print(f"Loading model: {model_name}")
    start = time.time()

    params = create_context_params(args)
    params.vae_decode_only = False  # Need encoder for img2img

    try:
        ctx = SDContext(params)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")

    sample_method, scheduler = parse_sampler_scheduler(args)

    # Use init image dimensions if not specified
    width = args.width if args.width > 0 else init_image.width
    height = args.height if args.height > 0 else init_image.height

    print(f"Generating {args.batch} image(s)...")
    print(f"  Prompt: {args.prompt}")
    print(f"  Size: {width}x{height}")
    print(f"  Strength: {args.strength}")

    start = time.time()
    try:
        images = ctx.generate(
            prompt=args.prompt,
            negative_prompt=args.negative or "",
            width=width,
            height=height,
            seed=args.seed,
            batch_count=args.batch,
            sample_steps=args.steps,
            cfg_scale=args.cfg_scale,
            sample_method=sample_method,
            scheduler=scheduler,
            init_image=init_image,
            strength=args.strength,
            clip_skip=args.clip_skip,
            eta=args.eta,
            slg_scale=args.slg_scale,
            vae_tiling=args.vae_tiling,
        )
    except RuntimeError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1

    gen_time = time.time() - start
    if args.progress:
        print()
    print(f"Generated {len(images)} image(s) in {gen_time:.2f}s")

    save_outputs(images, args)
    return 0


def cmd_inpaint(args):
    """Inpaint image with mask."""
    from .stable_diffusion import SDContext, SDImage

    validate_model_args(args)
    setup_logging(args)
    setup_preview(args)

    # Load init image
    print(f"Loading init image: {args.init_img}")
    try:
        init_image = SDImage.load(args.init_img)
    except Exception as e:
        print(f"Error loading init image: {e}", file=sys.stderr)
        return 1

    # Load mask
    print(f"Loading mask: {args.mask}")
    try:
        mask_image = SDImage.load(args.mask)
    except Exception as e:
        print(f"Error loading mask: {e}", file=sys.stderr)
        return 1

    model_name = args.model or args.diffusion_model
    print(f"Loading model: {model_name}")
    start = time.time()

    params = create_context_params(args)
    params.vae_decode_only = False

    try:
        ctx = SDContext(params)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")

    sample_method, scheduler = parse_sampler_scheduler(args)

    width = args.width if args.width > 0 else init_image.width
    height = args.height if args.height > 0 else init_image.height

    print(f"Inpainting...")
    print(f"  Prompt: {args.prompt}")
    print(f"  Size: {width}x{height}")

    start = time.time()
    try:
        images = ctx.generate(
            prompt=args.prompt,
            negative_prompt=args.negative or "",
            width=width,
            height=height,
            seed=args.seed,
            batch_count=args.batch,
            sample_steps=args.steps,
            cfg_scale=args.cfg_scale,
            sample_method=sample_method,
            scheduler=scheduler,
            init_image=init_image,
            mask_image=mask_image,
            strength=args.strength,
            clip_skip=args.clip_skip,
            eta=args.eta,
            vae_tiling=args.vae_tiling,
        )
    except RuntimeError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1

    gen_time = time.time() - start
    if args.progress:
        print()
    print(f"Generated {len(images)} image(s) in {gen_time:.2f}s")

    save_outputs(images, args)
    return 0


def cmd_controlnet(args):
    """Generate with ControlNet guidance."""
    from .stable_diffusion import SDContext, SDImage, canny_preprocess

    validate_model_args(args)
    setup_logging(args)
    setup_preview(args)

    # Load control image
    print(f"Loading control image: {args.control_image}")
    try:
        control_image = SDImage.load(args.control_image)
    except Exception as e:
        print(f"Error loading control image: {e}", file=sys.stderr)
        return 1

    # Apply canny preprocessing if requested
    if args.canny:
        print("Applying Canny edge detection...")
        canny_preprocess(control_image)

    model_name = args.model or args.diffusion_model
    print(f"Loading model: {model_name}")
    start = time.time()

    params = create_context_params(args)

    try:
        ctx = SDContext(params)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")

    sample_method, scheduler = parse_sampler_scheduler(args)

    width = args.width if args.width > 0 else control_image.width
    height = args.height if args.height > 0 else control_image.height

    print(f"Generating with ControlNet...")
    print(f"  Prompt: {args.prompt}")
    print(f"  Size: {width}x{height}")
    print(f"  Control strength: {args.control_strength}")

    start = time.time()
    try:
        images = ctx.generate(
            prompt=args.prompt,
            negative_prompt=args.negative or "",
            width=width,
            height=height,
            seed=args.seed,
            batch_count=args.batch,
            sample_steps=args.steps,
            cfg_scale=args.cfg_scale,
            sample_method=sample_method,
            scheduler=scheduler,
            control_image=control_image,
            control_strength=args.control_strength,
            clip_skip=args.clip_skip,
            eta=args.eta,
            slg_scale=args.slg_scale,
            vae_tiling=args.vae_tiling,
        )
    except RuntimeError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1

    gen_time = time.time() - start
    if args.progress:
        print()
    print(f"Generated {len(images)} image(s) in {gen_time:.2f}s")

    save_outputs(images, args)
    return 0


def cmd_video(args):
    """Generate video frames."""
    from .stable_diffusion import SDContext, SDImage

    validate_model_args(args)
    setup_logging(args)

    model_name = args.model or args.diffusion_model
    print(f"Loading model: {model_name}")
    start = time.time()

    params = create_context_params(args)

    try:
        ctx = SDContext(params)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")

    sample_method, scheduler = parse_sampler_scheduler(args)

    # Load init image if provided
    init_image = None
    if args.init_img:
        print(f"Loading init image: {args.init_img}")
        try:
            init_image = SDImage.load(args.init_img)
        except Exception as e:
            print(f"Error loading init image: {e}", file=sys.stderr)
            return 1

    # Load end image if provided
    end_image = None
    if args.end_img:
        print(f"Loading end image: {args.end_img}")
        try:
            end_image = SDImage.load(args.end_img)
        except Exception as e:
            print(f"Error loading end image: {e}", file=sys.stderr)
            return 1

    print(f"Generating {args.video_frames} video frames...")
    print(f"  Prompt: {args.prompt}")
    print(f"  Size: {args.width}x{args.height}")
    print(f"  FPS: {args.fps}")

    start = time.time()
    try:
        frames = ctx.generate_video(
            prompt=args.prompt,
            negative_prompt=args.negative or "",
            width=args.width,
            height=args.height,
            seed=args.seed,
            sample_steps=args.steps,
            cfg_scale=args.cfg_scale,
            video_frames=args.video_frames,
            sample_method=sample_method,
            scheduler=scheduler,
        )
    except RuntimeError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1

    gen_time = time.time() - start
    if args.progress:
        print()
    print(f"Generated {len(frames)} frames in {gen_time:.2f}s")

    save_video_frames(frames, args.output, args.fps)
    return 0


def cmd_upscale(args):
    """Upscale an image using ESRGAN."""
    from .stable_diffusion import Upscaler, SDImage, set_log_callback

    if args.verbose:
        def log_cb(level, text):
            level_names = {0: 'DEBUG', 1: 'INFO', 2: 'WARN', 3: 'ERROR'}
            print(f'[{level_names.get(level, level)}] {text}', end='')
        set_log_callback(log_cb)

    print(f"Loading image: {args.input}")
    try:
        input_img = SDImage.load(args.input)
    except Exception as e:
        print(f"Error loading image: {e}", file=sys.stderr)
        return 1

    print(f"Input size: {input_img.width}x{input_img.height}")

    print(f"Loading upscaler: {args.model}")
    try:
        upscaler = Upscaler(
            args.model,
            n_threads=args.threads,
            offload_to_cpu=args.offload_to_cpu
        )
    except Exception as e:
        print(f"Error loading upscaler: {e}", file=sys.stderr)
        return 1

    print(f"Upscale factor: {upscaler.upscale_factor}x")

    # Run upscale repeats times
    current_img = input_img
    for i in range(args.repeats):
        if args.repeats > 1:
            print(f"Upscaling pass {i+1}/{args.repeats}...")
        else:
            print("Upscaling...")

        start = time.time()
        try:
            current_img = upscaler.upscale(current_img, factor=args.factor or 0)
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        elapsed = time.time() - start
        print(f"  -> {current_img.width}x{current_img.height} in {elapsed:.2f}s")

    save_image(current_img, args.output)
    return 0


def cmd_convert(args):
    """Convert model to different format/quantization."""
    from .stable_diffusion import convert_model, SDType, set_log_callback

    if args.verbose:
        def log_cb(level, text):
            level_names = {0: 'DEBUG', 1: 'INFO', 2: 'WARN', 3: 'ERROR'}
            print(f'[{level_names.get(level, level)}] {text}', end='')
        set_log_callback(log_cb)

    try:
        output_type = SDType[args.type.upper()]
    except KeyError:
        print(f"Unknown type: {args.type}", file=sys.stderr)
        print(f"Available: {[t.name.lower() for t in SDType]}")
        return 1

    print(f"Converting: {args.input} -> {args.output}")
    print(f"Output type: {args.type}")

    start = time.time()
    try:
        convert_model(
            input_path=args.input,
            output_path=args.output,
            output_type=output_type,
            vae_path=args.vae,
            tensor_type_rules=args.tensor_type_rules
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
    from .stable_diffusion import (
        get_num_cores, get_system_info,
        SampleMethod, Scheduler, SDType, Prediction, RngType
    )

    print("Stable Diffusion Module Info")
    print("=" * 50)
    print(f"CPU cores: {get_num_cores()}")
    print()
    print("System info:")
    print(get_system_info())
    print()

    print("Available samplers:")
    for m in SampleMethod:
        print(f"  {m.name.lower()}")
    print()

    print("Available schedulers:")
    for s in Scheduler:
        print(f"  {s.name.lower()}")
    print()

    print("Available quantization types:")
    for t in SDType:
        print(f"  {t.name.lower()}")
    print()

    print("Available prediction types:")
    for p in Prediction:
        print(f"  {p.name.lower()}")
    print()

    print("Available RNG types:")
    for r in RngType:
        print(f"  {r.name.lower()}")

    return 0


# =============================================================================
# Argument parser helpers
# =============================================================================

def add_common_model_args(parser):
    """Add common model path arguments."""
    parser.add_argument('--model', '-m', help='Path to model file (or use --diffusion-model)')
    parser.add_argument('--vae', help='Path to VAE model')
    parser.add_argument('--taesd', help='Path to TAESD model (fast preview)')
    parser.add_argument('--clip-l', dest='clip_l', help='Path to CLIP-L model')
    parser.add_argument('--clip-g', dest='clip_g', help='Path to CLIP-G model')
    parser.add_argument('--clip-vision', dest='clip_vision', help='Path to CLIP vision model')
    parser.add_argument('--t5xxl', help='Path to T5-XXL model')
    parser.add_argument('--llm', help='Path to LLM text encoder')
    parser.add_argument('--llm-vision', dest='llm_vision', help='Path to LLM vision encoder')
    parser.add_argument('--diffusion-model', dest='diffusion_model', help='Path to diffusion model')
    parser.add_argument('--high-noise-diffusion-model', dest='high_noise_diffusion_model',
                        help='Path to high-noise diffusion model')
    parser.add_argument('--lora-dir', dest='lora_dir', help='LoRA model directory')
    parser.add_argument('--embd-dir', dest='embd_dir', help='Embeddings directory')
    parser.add_argument('--tensor-type-rules', dest='tensor_type_rules',
                        help='Tensor type rules (e.g., "^vae\\.=f16,model\\.=q8_0")')


def add_common_gen_args(parser):
    """Add common generation arguments."""
    parser.add_argument('--prompt', '-p', required=True, help='Text prompt')
    parser.add_argument('--negative', '-n', help='Negative prompt')
    parser.add_argument('--output', '-o', default='output.png', help='Output path')
    parser.add_argument('--width', '-W', type=int, default=512, help='Image width')
    parser.add_argument('--height', '-H', type=int, default=512, help='Image height')
    parser.add_argument('--steps', type=int, default=20, help='Sampling steps')
    parser.add_argument('--cfg-scale', dest='cfg_scale', type=float, default=7.0, help='CFG scale')
    parser.add_argument('--seed', '-s', type=int, default=-1, help='Random seed')
    parser.add_argument('--batch', '-b', type=int, default=1, help='Batch count')
    parser.add_argument('--clip-skip', dest='clip_skip', type=int, default=-1, help='CLIP skip layers')


def add_common_sampler_args(parser):
    """Add sampler/scheduler arguments."""
    parser.add_argument('--sampler', help='Sampling method (euler, euler_a, heun, dpm2, etc.)')
    parser.add_argument('--scheduler', help='Scheduler (discrete, karras, exponential, ays, etc.)')
    parser.add_argument('--eta', type=float, default=0.0, help='Eta for DDIM/TCD samplers')
    parser.add_argument('--rng', choices=['std_default', 'cuda', 'cpu'], help='RNG type')
    parser.add_argument('--sampler-rng', dest='sampler_rng', choices=['std_default', 'cuda', 'cpu'],
                        help='Sampler RNG type')
    parser.add_argument('--prediction', choices=['eps', 'v', 'edm_v', 'sd3_flow', 'flux_flow', 'flux2_flow'],
                        help='Prediction type override')


def add_common_guidance_args(parser):
    """Add guidance arguments."""
    parser.add_argument('--slg-scale', dest='slg_scale', type=float, default=0.0,
                        help='Skip layer guidance scale (0=disabled, 2.5 good for SD3.5)')
    parser.add_argument('--skip-layer-start', dest='skip_layer_start', type=float, default=0.01,
                        help='SLG enabling point')
    parser.add_argument('--skip-layer-end', dest='skip_layer_end', type=float, default=0.2,
                        help='SLG disabling point')
    parser.add_argument('--guidance', type=float, help='Distilled guidance scale (for FLUX)')
    parser.add_argument('--img-cfg-scale', dest='img_cfg_scale', type=float,
                        help='Image CFG scale for inpaint/instruct-pix2pix')


def add_common_memory_args(parser):
    """Add memory/performance arguments."""
    parser.add_argument('--threads', '-t', type=int, default=-1, help='Number of threads')
    parser.add_argument('--offload-to-cpu', dest='offload_to_cpu', action='store_true',
                        help='Offload weights to CPU (low VRAM)')
    parser.add_argument('--clip-on-cpu', dest='clip_on_cpu', action='store_true',
                        help='Keep CLIP on CPU')
    parser.add_argument('--vae-on-cpu', dest='vae_on_cpu', action='store_true',
                        help='Keep VAE on CPU')
    parser.add_argument('--control-net-cpu', dest='control_net_cpu', action='store_true',
                        help='Keep ControlNet on CPU')
    parser.add_argument('--diffusion-fa', dest='diffusion_fa', action='store_true',
                        help='Use flash attention in diffusion')
    parser.add_argument('--diffusion-conv-direct', dest='diffusion_conv_direct', action='store_true',
                        help='Use direct convolution in diffusion')
    parser.add_argument('--vae-conv-direct', dest='vae_conv_direct', action='store_true',
                        help='Use direct convolution in VAE')


def add_common_vae_tiling_args(parser):
    """Add VAE tiling arguments."""
    parser.add_argument('--vae-tiling', dest='vae_tiling', action='store_true',
                        help='Enable VAE tiling for large images')
    parser.add_argument('--vae-tile-size', dest='vae_tile_size', default='512x512',
                        help='VAE tile size (default: 512x512)')
    parser.add_argument('--vae-tile-overlap', dest='vae_tile_overlap', type=float, default=0.5,
                        help='VAE tile overlap fraction (default: 0.5)')


def add_common_preview_args(parser):
    """Add preview arguments."""
    parser.add_argument('--preview', choices=['none', 'proj', 'tae', 'vae'], default='none',
                        help='Preview mode')
    parser.add_argument('--preview-path', dest='preview_path', default='./preview.png',
                        help='Preview output path')
    parser.add_argument('--preview-interval', dest='preview_interval', type=int, default=1,
                        help='Preview interval (steps)')
    parser.add_argument('--preview-noisy', dest='preview_noisy', action='store_true',
                        help='Preview noisy instead of denoised')
    parser.add_argument('--taesd-preview-only', dest='taesd_preview_only', action='store_true',
                        help='Use TAESD only for preview, not final decode')


def add_common_misc_args(parser):
    """Add misc arguments."""
    parser.add_argument('--lora-apply-mode', dest='lora_apply_mode',
                        choices=['auto', 'immediately', 'at_runtime'],
                        help='LoRA application mode')
    parser.add_argument('--flow-shift', dest='flow_shift', type=float,
                        help='Flow shift for SD3.x/Wan models')
    parser.add_argument('--chroma-disable-dit-mask', dest='chroma_disable_dit_mask', action='store_true',
                        help='Disable DiT mask for Chroma')
    parser.add_argument('--chroma-enable-t5-mask', dest='chroma_enable_t5_mask', action='store_true',
                        help='Enable T5 mask for Chroma')
    parser.add_argument('--chroma-t5-mask-pad', dest='chroma_t5_mask_pad', type=int,
                        help='T5 mask pad for Chroma')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--progress', action='store_true', help='Show progress')


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Stable Diffusion CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # -------------------------------------------------------------------------
    # txt2img command
    # -------------------------------------------------------------------------
    txt2img_parser = subparsers.add_parser('txt2img', help='Generate images from text',
                                            aliases=['generate'])
    add_common_model_args(txt2img_parser)
    add_common_gen_args(txt2img_parser)
    add_common_sampler_args(txt2img_parser)
    add_common_guidance_args(txt2img_parser)
    add_common_memory_args(txt2img_parser)
    add_common_vae_tiling_args(txt2img_parser)
    add_common_preview_args(txt2img_parser)
    add_common_misc_args(txt2img_parser)

    # -------------------------------------------------------------------------
    # img2img command
    # -------------------------------------------------------------------------
    img2img_parser = subparsers.add_parser('img2img', help='Image to image generation')
    add_common_model_args(img2img_parser)
    add_common_gen_args(img2img_parser)
    img2img_parser.add_argument('--init-img', '-i', dest='init_img', required=True,
                                 help='Path to init image')
    img2img_parser.add_argument('--strength', type=float, default=0.75,
                                 help='Denoising strength (0.0-1.0)')
    add_common_sampler_args(img2img_parser)
    add_common_guidance_args(img2img_parser)
    add_common_memory_args(img2img_parser)
    add_common_vae_tiling_args(img2img_parser)
    add_common_preview_args(img2img_parser)
    add_common_misc_args(img2img_parser)

    # -------------------------------------------------------------------------
    # inpaint command
    # -------------------------------------------------------------------------
    inpaint_parser = subparsers.add_parser('inpaint', help='Inpainting with mask')
    add_common_model_args(inpaint_parser)
    add_common_gen_args(inpaint_parser)
    inpaint_parser.add_argument('--init-img', '-i', dest='init_img', required=True,
                                 help='Path to init image')
    inpaint_parser.add_argument('--mask', required=True, help='Path to mask image (white=inpaint)')
    inpaint_parser.add_argument('--strength', type=float, default=1.0,
                                 help='Denoising strength (0.0-1.0)')
    add_common_sampler_args(inpaint_parser)
    add_common_guidance_args(inpaint_parser)
    add_common_memory_args(inpaint_parser)
    add_common_vae_tiling_args(inpaint_parser)
    add_common_preview_args(inpaint_parser)
    add_common_misc_args(inpaint_parser)

    # -------------------------------------------------------------------------
    # controlnet command
    # -------------------------------------------------------------------------
    cn_parser = subparsers.add_parser('controlnet', help='ControlNet guided generation')
    add_common_model_args(cn_parser)
    cn_parser.add_argument('--control-net', dest='control_net', required=True,
                            help='Path to ControlNet model')
    cn_parser.add_argument('--control-image', dest='control_image', required=True,
                            help='Path to control image')
    cn_parser.add_argument('--control-strength', dest='control_strength', type=float, default=0.9,
                            help='Control strength (0.0-1.0+)')
    cn_parser.add_argument('--canny', action='store_true',
                            help='Apply Canny edge detection to control image')
    add_common_gen_args(cn_parser)
    add_common_sampler_args(cn_parser)
    add_common_guidance_args(cn_parser)
    add_common_memory_args(cn_parser)
    add_common_vae_tiling_args(cn_parser)
    add_common_preview_args(cn_parser)
    add_common_misc_args(cn_parser)

    # -------------------------------------------------------------------------
    # video command
    # -------------------------------------------------------------------------
    video_parser = subparsers.add_parser('video', help='Generate video frames')
    add_common_model_args(video_parser)
    video_parser.add_argument('--prompt', '-p', required=True, help='Text prompt')
    video_parser.add_argument('--negative', '-n', help='Negative prompt')
    video_parser.add_argument('--output', '-o', default='output.png', help='Output path prefix')
    video_parser.add_argument('--width', '-W', type=int, default=512, help='Frame width')
    video_parser.add_argument('--height', '-H', type=int, default=512, help='Frame height')
    video_parser.add_argument('--steps', type=int, default=20, help='Sampling steps')
    video_parser.add_argument('--cfg-scale', dest='cfg_scale', type=float, default=7.0, help='CFG scale')
    video_parser.add_argument('--seed', '-s', type=int, default=-1, help='Random seed')
    video_parser.add_argument('--video-frames', dest='video_frames', type=int, default=16,
                               help='Number of video frames')
    video_parser.add_argument('--fps', type=int, default=24, help='Frames per second')
    video_parser.add_argument('--init-img', '-i', dest='init_img', help='Path to init image')
    video_parser.add_argument('--end-img', dest='end_img', help='Path to end image (for flf2v)')
    video_parser.add_argument('--moe-boundary', dest='moe_boundary', type=float, default=0.875,
                               help='MoE boundary for Wan2.2 (default: 0.875)')
    add_common_sampler_args(video_parser)
    add_common_memory_args(video_parser)
    add_common_misc_args(video_parser)

    # -------------------------------------------------------------------------
    # upscale command
    # -------------------------------------------------------------------------
    up_parser = subparsers.add_parser('upscale', help='Upscale image with ESRGAN')
    up_parser.add_argument('--model', '-m', required=True, help='Path to ESRGAN model')
    up_parser.add_argument('--input', '-i', required=True, help='Input image path')
    up_parser.add_argument('--output', '-o', required=True, help='Output image path')
    up_parser.add_argument('--factor', '-f', type=int, help='Upscale factor (default: model default)')
    up_parser.add_argument('--repeats', '-r', type=int, default=1, help='Upscale repeats')
    up_parser.add_argument('--threads', '-t', type=int, default=-1, help='Number of threads')
    up_parser.add_argument('--offload-to-cpu', dest='offload_to_cpu', action='store_true',
                            help='Offload to CPU')
    up_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    # -------------------------------------------------------------------------
    # convert command
    # -------------------------------------------------------------------------
    conv_parser = subparsers.add_parser('convert', help='Convert model format')
    conv_parser.add_argument('--input', '-i', required=True, help='Input model path')
    conv_parser.add_argument('--output', '-o', required=True, help='Output model path')
    conv_parser.add_argument('--type', '-t', default='f16',
                              help='Output type (f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0, etc.)')
    conv_parser.add_argument('--vae', help='Path to VAE model')
    conv_parser.add_argument('--tensor-type-rules', dest='tensor_type_rules',
                              help='Tensor type rules')
    conv_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    # -------------------------------------------------------------------------
    # info command
    # -------------------------------------------------------------------------
    subparsers.add_parser('info', help='Show system info and available options')

    # -------------------------------------------------------------------------
    # Parse and dispatch
    # -------------------------------------------------------------------------
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    cmd_map = {
        'txt2img': cmd_txt2img,
        'generate': cmd_txt2img,  # alias
        'img2img': cmd_img2img,
        'inpaint': cmd_inpaint,
        'controlnet': cmd_controlnet,
        'video': cmd_video,
        'upscale': cmd_upscale,
        'convert': cmd_convert,
        'info': cmd_info,
    }

    handler = cmd_map.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
