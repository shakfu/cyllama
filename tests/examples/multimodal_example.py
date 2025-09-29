#!/usr/bin/env python3
"""
Example usage of cyllama's multimodal (mtmd) capabilities.

This example demonstrates how to use the mtmd integration for vision-language
processing and audio analysis.

Usage:
    python examples/multimodal_example.py --model path/to/model.gguf --mmproj path/to/vision.mmproj --image path/to/image.jpg
"""

import argparse
import sys
from pathlib import Path

try:
    import cyllama
    from cyllama.llama.mtmd import (
        MultimodalProcessor,
        VisionLanguageChat,
        ImageAnalyzer,
        AudioProcessor,
        UnsupportedModalityError,
        MultimodalError,
    )
    HAS_MULTIMODAL = True
except ImportError as e:
    print(f"Error: Multimodal support not available: {e}")
    print("Please ensure cyllama is compiled with mtmd support.")
    sys.exit(1)


def vision_example(model_path: str, mmproj_path: str, image_path: str):
    """Demonstrate vision-language processing."""
    print("=== Vision-Language Processing Example ===")

    try:
        # Load model
        print(f"Loading model: {model_path}")
        model = cyllama.LlamaModel(model_path)

        # Create multimodal processor
        print(f"Loading multimodal projector: {mmproj_path}")
        processor = MultimodalProcessor(mmproj_path, model)

        # Check capabilities
        print(f"Vision support: {processor.supports_vision}")
        print(f"Audio support: {processor.supports_audio}")

        if not processor.supports_vision:
            print("Error: This model does not support vision processing")
            return

        # Process image with question
        print(f"Processing image: {image_path}")
        question = "What's in this image? Describe it in detail."

        try:
            chunks = processor.process_image(question, image_path)
            print(f"Successfully tokenized input:")
            print(f"  Total chunks: {len(chunks)}")
            print(f"  Total tokens: {chunks.total_tokens}")
            print(f"  Total positions: {chunks.total_positions}")

            # Display chunk information
            for i, chunk in enumerate(chunks):
                print(f"  Chunk {i}: type={chunk.type.name}, tokens={chunk.n_tokens}, positions={chunk.n_pos}")
                if chunk.id:
                    print(f"    ID: {chunk.id}")

        except Exception as e:
            print(f"Error processing image: {e}")

    except Exception as e:
        print(f"Error: {e}")


def analyzer_example(model_path: str, mmproj_path: str, image_path: str):
    """Demonstrate image analyzer functionality."""
    print("\n=== Image Analyzer Example ===")

    try:
        # Load model
        model = cyllama.LlamaModel(model_path)

        # Create image analyzer
        analyzer = ImageAnalyzer(mmproj_path, model)

        print("Testing different analysis tasks:")

        # Test different description levels
        for level in ["brief", "medium", "detailed"]:
            print(f"  {level.capitalize()} description: [would generate response here]")

        # Test object detection
        print("  Object detection: [would list detected objects here]")

        # Test question answering
        print("  Question answering: [would answer specific questions here]")

        print("Note: Actual text generation requires implementation of sampling loop")

    except UnsupportedModalityError:
        print("Error: Vision not supported by this model")
    except Exception as e:
        print(f"Error: {e}")


def chat_example(model_path: str, mmproj_path: str, image_path: str):
    """Demonstrate vision-language chat."""
    print("\n=== Vision-Language Chat Example ===")

    try:
        # Load model and context
        model = cyllama.LlamaModel(model_path)
        context = cyllama.LlamaContext(model)

        # Create chat interface
        chat = VisionLanguageChat(mmproj_path, model, context)

        print("Starting vision-language conversation...")

        # Ask about image
        response = chat.ask_about_image("What do you see in this image?", image_path)
        print(f"Assistant: {response}")

        # Continue conversation
        follow_up = chat.continue_conversation("Can you tell me more about the colors?")
        print(f"Assistant: {follow_up}")

        print(f"Conversation history: {len(chat.conversation_history)} exchanges")

        print("Note: Actual text generation requires implementation of sampling loop")

    except UnsupportedModalityError:
        print("Error: Vision not supported by this model")
    except Exception as e:
        print(f"Error: {e}")


def audio_example(model_path: str, mmproj_path: str, audio_path: str):
    """Demonstrate audio processing."""
    print("\n=== Audio Processing Example ===")

    try:
        # Load model
        model = cyllama.LlamaModel(model_path)

        # Create multimodal processor
        processor = MultimodalProcessor(mmproj_path, model)

        if not processor.supports_audio:
            print("Audio not supported by this model")
            return

        print(f"Audio bitrate: {processor.audio_bitrate} Hz")

        # Create audio processor
        audio_proc = AudioProcessor(mmproj_path, model)

        print("Testing audio analysis tasks:")
        print("  Transcription: [would transcribe audio here]")
        print("  Analysis: [would analyze audio content here]")

        print("Note: Actual text generation requires implementation of sampling loop")

    except UnsupportedModalityError:
        print("Error: Audio not supported by this model")
    except Exception as e:
        print(f"Error: {e}")


def low_level_example(model_path: str, mmproj_path: str, image_path: str):
    """Demonstrate low-level mtmd API usage."""
    print("\n=== Low-Level API Example ===")

    try:
        from cyllama.llama.mtmd import (
            MtmdContext,
            MtmdContextParams,
            MtmdBitmap,
            get_default_media_marker,
        )

        # Load model
        model = cyllama.LlamaModel(model_path)

        # Create context with custom parameters
        params = MtmdContextParams(use_gpu=True, n_threads=4, verbosity=2)
        mtmd_ctx = MtmdContext(mmproj_path, model, params)

        print("Context capabilities:")
        print(f"  Vision: {mtmd_ctx.supports_vision}")
        print(f"  Audio: {mtmd_ctx.supports_audio}")
        print(f"  Uses non-causal: {mtmd_ctx.uses_non_causal}")
        print(f"  Uses M-RoPE: {mtmd_ctx.uses_mrope}")

        if mtmd_ctx.supports_vision:
            # Load image as bitmap
            bitmap = MtmdBitmap.from_file(mtmd_ctx, image_path)
            print(f"Loaded image: {bitmap.width}x{bitmap.height}, audio={bitmap.is_audio}")

            # Set bitmap ID
            bitmap.id = f"image_{Path(image_path).stem}"
            print(f"Bitmap ID: {bitmap.id}")

            # Create input text with marker
            marker = get_default_media_marker()
            text = f"Describe this image: {marker}"

            # Tokenize
            chunks = mtmd_ctx.tokenize(text, [bitmap])
            print(f"Tokenized: {len(chunks)} chunks, {chunks.total_tokens} tokens")

    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Cyllama multimodal examples")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--mmproj", required=True, help="Path to multimodal projector (.mmproj) file")
    parser.add_argument("--image", help="Path to test image file")
    parser.add_argument("--audio", help="Path to test audio file")
    parser.add_argument("--example", choices=["all", "vision", "analyzer", "chat", "audio", "lowlevel"],
                       default="all", help="Which example to run")

    args = parser.parse_args()

    # Validate files
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    if not Path(args.mmproj).exists():
        print(f"Error: Multimodal projector file not found: {args.mmproj}")
        sys.exit(1)

    print(f"Cyllama multimodal examples")
    print(f"Model: {args.model}")
    print(f"Multimodal projector: {args.mmproj}")

    # Run examples
    try:
        if args.example in ["all", "vision"] and args.image:
            vision_example(args.model, args.mmproj, args.image)

        if args.example in ["all", "analyzer"] and args.image:
            analyzer_example(args.model, args.mmproj, args.image)

        if args.example in ["all", "chat"] and args.image:
            chat_example(args.model, args.mmproj, args.image)

        if args.example in ["all", "audio"] and args.audio:
            audio_example(args.model, args.mmproj, args.audio)

        if args.example in ["all", "lowlevel"] and args.image:
            low_level_example(args.model, args.mmproj, args.image)

        if not args.image and not args.audio:
            print("\nNote: Provide --image and/or --audio files to run full examples")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()