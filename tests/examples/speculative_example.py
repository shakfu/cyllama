#!/usr/bin/env python3
"""
Speculative Decoding Example

This example demonstrates how to use speculative decoding with cyllama to achieve
2-3x inference speedup. Speculative decoding works by using a smaller, faster "draft"
model to generate candidate tokens, which are then verified by a larger "target" model.

Requirements:
- A target model (larger, more accurate)
- A draft model (smaller, faster, compatible vocabulary)

For this example to work with actual speedup, you need two models:
1. Target model: e.g., Llama-3.2-3B-Instruct-Q8_0.gguf
2. Draft model: e.g., Llama-3.2-1B-Instruct-Q8_0.gguf

The models must have compatible tokenizers (usually from the same model family).

Usage:
    python speculative_example.py -m models/Llama-3.2-1B-Instruct-Q8_0.gguf
    python speculative_example.py --target models/large.gguf --draft models/small.gguf
"""

import sys
import os
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from cyllama.llama.llama_cpp import (
    LlamaModel,
    LlamaContext,
    LlamaContextParams,
    LlamaModelParams,
    Speculative,
    SpeculativeParams,
)


def load_model_and_context(model_path: str, n_ctx: int = 2048, n_gpu_layers: int = 0):
    """
    Load a model and create a context.

    Args:
        model_path: Path to the GGUF model file
        n_ctx: Context size (default: 2048)
        n_gpu_layers: Number of layers to offload to GPU (default: 0 for CPU)

    Returns:
        Tuple of (model, context)
    """
    print(f"Loading model: {model_path}")

    # Configure model parameters
    model_params = LlamaModelParams()
    model_params.n_gpu_layers = n_gpu_layers

    # Load model
    model = LlamaModel(model_path, model_params)

    # Configure context parameters
    ctx_params = LlamaContextParams()
    ctx_params.n_ctx = n_ctx
    ctx_params.n_batch = 512

    # Create context
    context = LlamaContext(model, ctx_params)

    print(f"  Loaded with {model.n_params} parameters")
    print(f"  Context size: {n_ctx}")

    return model, context


def demonstrate_basic_usage(target_model_path, draft_model_path):
    """
    Demonstrate basic speculative decoding setup.

    Note: This uses the same model for both target and draft for demonstration.
    In practice, you would use a larger model as target and smaller as draft.
    """
    print("=" * 70)
    print("BASIC SPECULATIVE DECODING DEMONSTRATION")
    print("=" * 70)

    # Check if models exist
    if not os.path.exists(target_model_path):
        print(f"Error: Target model not found at {target_model_path}")
        print("Please download a model first using 'make download'")
        return

    print("\n1. Loading Target Model")
    print("-" * 70)
    target_model, target_ctx = load_model_and_context(target_model_path, n_ctx=1024)

    print("\n2. Loading Draft Model")
    print("-" * 70)
    draft_model, draft_ctx = load_model_and_context(draft_model_path, n_ctx=1024)

    print("\n3. Checking Compatibility")
    print("-" * 70)
    compatible = Speculative.are_compatible(target_ctx, draft_ctx)
    print(f"Target and draft contexts compatible: {compatible}")

    if not compatible:
        print("ERROR: Models are not compatible for speculative decoding!")
        print("Ensure both models use the same tokenizer/vocabulary.")
        return

    print("\n4. Initializing Speculative Decoding")
    print("-" * 70)
    spec = Speculative(target_ctx, draft_ctx)
    print(f"Created: {spec}")

    print("\n5. Configuring Parameters")
    print("-" * 70)
    params = SpeculativeParams(
        n_draft=16,   # Generate up to 16 draft tokens
        n_reuse=256,  # Reuse up to 256 tokens from previous draft
        p_min=0.75    # Minimum probability to accept draft token
    )
    print(f"Parameters: {params}")
    print(f"  - n_draft: Maximum {params.n_draft} tokens to draft per iteration")
    print(f"  - n_reuse: Reuse up to {params.n_reuse} tokens from previous draft")
    print(f"  - p_min: Accept draft tokens with probability >= {params.p_min}")

    print("\n6. Testing Draft Generation")
    print("-" * 70)
    # Example: Generate draft tokens for a simple prompt
    # Note: Token IDs are model-specific. These are just examples.
    prompt_tokens = [1, 791, 2232, 374]  # Example token sequence
    last_token = 374

    print(f"Input prompt tokens: {prompt_tokens}")
    print(f"Last token ID: {last_token}")

    draft_tokens = spec.gen_draft(params, prompt_tokens, last_token)

    print(f"Generated draft tokens: {draft_tokens}")
    print(f"Number of draft tokens: {len(draft_tokens)}")

    print("\n7. Token Replacement Mapping (Optional)")
    print("-" * 70)
    print("You can add token replacements for models with different tokenizers:")
    spec.add_replacement("hello", "hi")
    spec.add_replacement("world", "earth")
    print("Added replacements: 'hello'->'hi', 'world'->'earth'")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


def demonstrate_parameter_tuning(target_model_path):
    """
    Demonstrate how different parameters affect draft generation.
    """
    print("\n" + "=" * 70)
    print("PARAMETER TUNING DEMONSTRATION")
    print("=" * 70)

    if not os.path.exists(target_model_path):
        print(f"Error: Model not found at {target_model_path}")
        return

    print("\nLoading model...")
    target_model, target_ctx = load_model_and_context(target_model_path, n_ctx=1024)
    draft_model, draft_ctx = load_model_and_context(target_model_path, n_ctx=1024)

    spec = Speculative(target_ctx, draft_ctx)
    prompt_tokens = [1, 791, 2232, 374]
    last_token = 374

    print("\nTesting different n_draft values:")
    print("-" * 70)
    for n_draft in [4, 8, 16, 32]:
        params = SpeculativeParams(n_draft=n_draft, p_min=0.75)
        draft = spec.gen_draft(params, prompt_tokens, last_token)
        print(f"n_draft={n_draft:2d}: Generated {len(draft):2d} tokens")

    print("\nTesting different p_min values:")
    print("-" * 70)
    for p_min in [0.5, 0.6, 0.7, 0.8, 0.9]:
        params = SpeculativeParams(n_draft=16, p_min=p_min)
        draft = spec.gen_draft(params, prompt_tokens, last_token)
        print(f"p_min={p_min:.1f}: Generated {len(draft):2d} tokens")


def print_usage_tips():
    """Print tips for effective speculative decoding."""
    print("\n" + "=" * 70)
    print("TIPS FOR EFFECTIVE SPECULATIVE DECODING")
    print("=" * 70)
    print("""
1. MODEL SELECTION:
   - Use models from the same family (e.g., Llama 3.2 1B and 3B)
   - Draft model should be 2-4x smaller than target model
   - Both must have compatible tokenizers

2. PARAMETER TUNING:
   - n_draft: Start with 16, increase for more speedup (but less accuracy)
   - p_min: Start with 0.75, increase for better quality (less speedup)
   - n_reuse: Keep at 256 for most cases

3. PERFORMANCE EXPECTATIONS:
   - Expect 1.5-3x speedup depending on models and parameters
   - Best results with simple, predictable text
   - Less effective for very creative/diverse outputs

4. GPU USAGE:
   - Offload both models to GPU for maximum speedup
   - Draft model can use fewer layers if VRAM is limited

5. WHEN TO USE:
   - Long-form content generation
   - Code generation
   - Structured output
   - High throughput scenarios

6. WHEN NOT TO USE:
   - Very short responses
   - Extremely creative tasks
   - When maximum quality is critical
    """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speculative Decoding Example")
    parser.add_argument("-m", "--model", help="Path to model file (used for both target and draft)")
    parser.add_argument("--target", help="Path to target model file")
    parser.add_argument("--draft", help="Path to draft model file")
    args = parser.parse_args()

    # Determine model paths
    if args.target and args.draft:
        target_model_path = args.target
        draft_model_path = args.draft
    elif args.model:
        target_model_path = args.model
        draft_model_path = args.model
    else:
        parser.error("Either -m/--model or both --target and --draft are required")

    try:
        # Run basic demonstration
        demonstrate_basic_usage(target_model_path, draft_model_path)

        # Run parameter tuning demonstration
        demonstrate_parameter_tuning(target_model_path)

        # Print usage tips
        print_usage_tips()

        print("\n" + "=" * 70)
        print("SUCCESS: All demonstrations completed!")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure you have downloaded the required models:")
        print("  make download")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
