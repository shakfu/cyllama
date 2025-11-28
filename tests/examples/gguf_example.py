"""Example: Using GGUF API to inspect and manipulate GGUF model files.

This example demonstrates:
1. Reading GGUF model metadata
2. Inspecting tensor information
3. Creating custom GGUF files
4. Modifying metadata
"""

from cyllama.llama.llama_cpp import GGUFContext


def inspect_model(model_path):
    """Inspect a GGUF model file and display metadata."""
    print(f"\n=== Inspecting GGUF Model: {model_path} ===\n")

    # Load GGUF file
    ctx = GGUFContext.from_file(model_path)

    # Display basic info
    print(f"GGUF Version: {ctx.version}")
    print(f"Number of tensors: {ctx.n_tensors}")
    print(f"Number of metadata KV pairs: {ctx.n_kv}")
    print(f"Alignment: {ctx.alignment} bytes")
    print(f"Data offset: {ctx.data_offset} bytes")

    # Get all metadata
    print("\n--- Model Metadata ---")
    metadata = ctx.get_all_metadata()

    # Display important metadata
    important_keys = [
        'general.architecture',
        'general.name',
        'general.file_type',
        'llama.context_length',
        'llama.embedding_length',
        'llama.block_count',
        'llama.vocab_size',
    ]

    for key in important_keys:
        if key in metadata:
            print(f"  {key}: {metadata[key]}")

    # Display tensor info
    print("\n--- Tensor Information ---")
    tensors = ctx.get_all_tensor_info()
    print(f"Total tensors: {len(tensors)}")

    if tensors:
        print("\nFirst 5 tensors:")
        for tensor in tensors[:5]:
            print(f"  {tensor['name']}:")
            print(f"    Type: {tensor['type']}")
            print(f"    Size: {tensor['size']} bytes")
            print(f"    Offset: {tensor['offset']} bytes")

    return ctx


def create_custom_gguf(output_path):
    """Create a custom GGUF file with metadata."""
    print(f"\n=== Creating Custom GGUF: {output_path} ===\n")

    # Create empty GGUF context
    ctx = GGUFContext.empty()

    # Add metadata
    ctx.set_val_str("general.architecture", "custom")
    ctx.set_val_str("general.name", "Custom Model")
    ctx.set_val_u32("custom.version", 1)
    ctx.set_val_f32("custom.scale", 0.5)
    ctx.set_val_bool("custom.enabled", True)
    ctx.set_val_str("custom.description", "Example GGUF file")

    # Write to file
    ctx.write_to_file(output_path)
    print(f"Created custom GGUF file: {output_path}")

    # Verify by reading back
    verify_ctx = GGUFContext.from_file(output_path)
    metadata = verify_ctx.get_all_metadata()

    print("\nVerifying metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")


def modify_metadata(model_path, output_path):
    """Load a GGUF file, modify metadata, and save."""
    print(f"\n=== Modifying GGUF Metadata ===\n")

    # Load existing GGUF
    ctx = GGUFContext.from_file(model_path)

    # Add custom metadata
    ctx.set_val_str("custom.modified_by", "cyllama")
    ctx.set_val_str("custom.date", "2025-11-17")
    ctx.set_val_bool("custom.quantized", True)

    # Save modified version
    ctx.write_to_file(output_path)
    print(f"Saved modified GGUF to: {output_path}")

    # Verify changes
    verify_ctx = GGUFContext.from_file(output_path)
    metadata = verify_ctx.get_all_metadata()

    print("\nCustom metadata added:")
    for key in ["custom.modified_by", "custom.date", "custom.quantized"]:
        if key in metadata:
            print(f"  {key}: {metadata[key]}")


def search_tensor(model_path, tensor_name):
    """Search for a specific tensor in the model."""
    print(f"\n=== Searching for tensor: {tensor_name} ===\n")

    ctx = GGUFContext.from_file(model_path)

    # Find tensor
    idx = ctx.find_tensor(tensor_name)

    if idx >= 0:
        print(f"Found tensor '{tensor_name}' at index {idx}")

        # Get all tensor info to display details
        tensors = ctx.get_all_tensor_info()
        # Find the tensor in the list
        for tensor in tensors:
            if tensor['name'] == tensor_name:
                print(f"  Type: {tensor['type']}")
                print(f"  Size: {tensor['size']} bytes")
                print(f"  Offset: {tensor['offset']} bytes")
                break
    else:
        print(f"Tensor '{tensor_name}' not found")


def main(model_path):
    """Run all examples."""
    import os

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please run 'make download' to get the test model")
        return

    # 1. Inspect existing model
    ctx = inspect_model(model_path)

    # 2. Search for specific tensors
    search_tensor(model_path, "token_embd.weight")
    search_tensor(model_path, "output.weight")

    # Note: Custom GGUF creation and modification examples are available
    # but disabled due to current limitations with empty GGUF contexts
    print("\n=== Additional Capabilities ===\n")
    print("The GGUF API also supports:")
    print("- Creating custom GGUF files with GGUFContext.empty()")
    print("- Adding metadata with set_val_str(), set_val_u32(), etc.")
    print("- Writing custom GGUF files with write_to_file()")
    print("- Modifying existing GGUF metadata")
    print("\nSee tests/test_gguf.py for working examples")

    print("\n=== GGUF Example Complete ===\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GGUF API Examples")
    parser.add_argument("-m", "--model", required=True, help="Path to model file")
    args = parser.parse_args()
    main(args.model)
