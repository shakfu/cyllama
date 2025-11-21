"""Example: Using Download API to fetch models from HuggingFace and other sources.

This example demonstrates:
1. Downloading models from HuggingFace with Ollama-style tags
2. Listing cached models
3. Getting HuggingFace file information
4. Docker registry integration
5. Direct URL downloads
"""

from cyllama.llama.llama_cpp import (
    download_model,
    get_hf_file,
    list_cached_models,
    resolve_docker_model
)
import os


def list_cached_models_example():
    """List all cached models."""
    print("\n=== Listing Cached Models ===\n")

    models = list_cached_models()

    if not models:
        print("No cached models found")
        print("Models are cached in ~/.cache/llama.cpp/")
        return

    print(f"Found {len(models)} cached model(s):\n")

    for i, model in enumerate(models, 1):
        print(f"{i}. {model['user']}/{model['model']}:{model['tag']}")
        print(f"   Manifest: {model['manifest_path']}")
        print(f"   Size: {model['size']:,} bytes ({model['size'] / (1024**3):.2f} GB)")
        print()


def get_hf_file_info_example():
    """Get information about HuggingFace files."""
    print("\n=== Getting HuggingFace File Info ===\n")

    # Example repos with different tag formats
    repos = [
        "bartowski/Llama-3.2-1B-Instruct-GGUF:Q8_0",
        "bartowski/Llama-3.2-1B-Instruct-GGUF:q4",  # Short form
        "bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M",  # Specific quant
    ]

    for repo in repos:
        print(f"Repository: {repo}")

        try:
            # Try offline first (will only work if already cached)
            info = get_hf_file(repo, offline=True)

            print(f"  Repo: {info['repo']}")
            print(f"  GGUF file: {info['gguf_file']}")
            print(f"  MMProj file: {info['mmproj_file']}")
            print()

        except RuntimeError as e:
            print(f"  Offline lookup failed: {e}")
            print("  (Run with offline=False to download manifest)")
            print()


def download_from_huggingface_example(offline=True):
    """Download a model from HuggingFace."""
    print("\n=== Downloading from HuggingFace ===\n")

    # Use a small model for the example
    hf_repo = "bartowski/TinyLlama-1.1B-Chat-v1.0-GGUF:Q4_K_M"

    print(f"Repository: {hf_repo}")
    print(f"Offline mode: {offline}")

    if offline:
        print("\nNote: Running in offline mode (won't actually download)")
        print("To download, call with offline=False and provide optional model_path")

    try:
        # Download (or check cache in offline mode)
        success = download_model(
            hf_repo=hf_repo,
            offline=offline
        )

        if success:
            print(f"\nSuccess! Model is available")
        else:
            print(f"\nModel not available in cache (offline mode)")
            print("To download: download_model(hf_repo='{hf_repo}', offline=False)")

    except RuntimeError as e:
        print(f"\nError: {e}")


def download_to_specific_path_example():
    """Download to a specific path."""
    print("\n=== Download to Specific Path ===\n")

    import tempfile

    # Create temporary path
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
        model_path = tmp.name

    print(f"Target path: {model_path}")
    print("Repository: bartowski/TinyLlama-1.1B-Chat-v1.0-GGUF:Q4_K_M")
    print("\nExample code:")
    print(f"""
    success = download_model(
        model_path="{model_path}",
        hf_repo="bartowski/TinyLlama-1.1B-Chat-v1.0-GGUF:Q4_K_M",
        offline=False  # Set to False to actually download
    )
    """)

    # Clean up temp file
    os.unlink(model_path)

    print("\nNote: Set offline=False to perform actual download")


def download_from_url_example():
    """Download from direct URL."""
    print("\n=== Download from Direct URL ===\n")

    url = "https://huggingface.co/user/repo/resolve/main/model.gguf"

    print(f"URL: {url}")
    print("\nExample code:")
    print(f"""
    success = download_model(
        url="{url}",
        model_path="my_model.gguf",
        offline=False
    )
    """)

    print("\nNote: This is for demonstration - URL may not be valid")


def docker_registry_example():
    """Download from Docker registry."""
    print("\n=== Docker Registry Example ===\n")

    docker_repo = "registry.ollama.ai/library/llama3.2:latest"

    print(f"Docker repo: {docker_repo}")
    print("\nExample code:")
    print(f"""
    try:
        path = resolve_docker_model("{docker_repo}")
        print(f"Model downloaded to: {{path}}")
    except RuntimeError as e:
        print(f"Error: {{e}}")
    """)

    print("\nNote: Requires valid Docker registry credentials")


def ollama_style_tags_example():
    """Demonstrate Ollama-style quantization tags."""
    print("\n=== Ollama-Style Quantization Tags ===\n")

    print("The download API supports Ollama-style short tags:\n")

    examples = [
        ("q2", "Q2_K quantization (smallest, lowest quality)"),
        ("q3", "Q3_K_S or Q3_K_M quantization"),
        ("q4", "Q4_K_M quantization (recommended balance)"),
        ("q5", "Q5_K_M quantization"),
        ("q6", "Q6_K quantization"),
        ("q8", "Q8_0 quantization (largest, highest quality)"),
    ]

    print("Tag   -> Quantization")
    print("-" * 40)
    for tag, desc in examples:
        print(f":{tag:<4} -> {desc}")

    print("\nExamples:")
    print("  bartowski/Llama-3.2-1B-Instruct-GGUF:q4")
    print("  bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M")
    print("  bartowski/Llama-3.2-1B-Instruct-GGUF:Q8_0")

    print("\nBoth short (:q4) and explicit (:Q4_K_M) tags are supported")


def authentication_example():
    """Using bearer tokens for private repos."""
    print("\n=== Authentication with Bearer Tokens ===\n")

    print("For private HuggingFace repositories, use a bearer token:\n")

    print("Example code:")
    print("""
    # Get your token from https://huggingface.co/settings/tokens
    bearer_token = "hf_..."

    success = download_model(
        hf_repo="private-user/private-model:q4",
        bearer_token=bearer_token,
        offline=False
    )
    """)

    print("\nNote: Never commit tokens to version control!")
    print("Use environment variables: os.getenv('HF_TOKEN')")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Download API Examples")
    print("=" * 60)

    # 1. List cached models
    list_cached_models_example()

    # 2. Get HF file info (offline - safe to run)
    get_hf_file_info_example()

    # 3. Download from HuggingFace (offline mode - won't download)
    download_from_huggingface_example(offline=True)

    # 4. Download to specific path
    download_to_specific_path_example()

    # 5. Download from URL
    download_from_url_example()

    # 6. Docker registry
    docker_registry_example()

    # 7. Ollama-style tags
    ollama_style_tags_example()

    # 8. Authentication
    authentication_example()

    print("\n" + "=" * 60)
    print("Download API Examples Complete")
    print("=" * 60)

    print("\nKey Features:")
    print("- Download from HuggingFace with Ollama-style tags (:q4, :q8, etc.)")
    print("- List all cached models")
    print("- Get file information without downloading")
    print("- Support for direct URLs and Docker registries")
    print("- Bearer token authentication for private repos")
    print("- Models cached in ~/.cache/llama.cpp/")

    print("\nTo actually download a model:")
    print("  download_model(hf_repo='user/repo:q4', offline=False)")


if __name__ == "__main__":
    main()
