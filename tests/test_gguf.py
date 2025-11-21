"""Tests for GGUF file format API."""

import pytest
from cyllama.llama.llama_cpp import GGUFContext


def test_gguf_read_model():
    """Test reading GGUF file metadata."""
    model_path = "models/Llama-3.2-1B-Instruct-Q8_0.gguf"

    # Load GGUF file
    ctx = GGUFContext.from_file(model_path)

    # Basic properties
    assert ctx.version > 0
    assert ctx.n_tensors > 0
    assert ctx.n_kv > 0
    assert ctx.alignment > 0
    assert ctx.data_offset > 0

    # String representation
    repr_str = repr(ctx)
    assert "GGUFContext" in repr_str
    assert "version" in repr_str

    print(f"\nGGUF Info:")
    print(f"  Version: {ctx.version}")
    print(f"  Tensors: {ctx.n_tensors}")
    print(f"  KV pairs: {ctx.n_kv}")
    print(f"  Alignment: {ctx.alignment}")
    print(f"  Data offset: {ctx.data_offset}")


def test_gguf_metadata():
    """Test reading metadata from GGUF file."""
    model_path = "models/Llama-3.2-1B-Instruct-Q8_0.gguf"
    ctx = GGUFContext.from_file(model_path)

    # Get all metadata
    metadata = ctx.get_all_metadata()
    assert isinstance(metadata, dict)
    assert len(metadata) > 0

    # Common metadata keys
    expected_keys = [
        "general.architecture",
        "general.name",
    ]

    for key in expected_keys:
        if key in metadata:
            print(f"{key}: {metadata[key]}")

    # Test specific key access
    if "general.architecture" in metadata:
        arch = ctx.get_value("general.architecture")
        print(f"\nArchitecture: {arch}")
        assert isinstance(arch, str)

    # Test find_key
    key_id = ctx.find_key("general.architecture")
    if key_id >= 0:
        key_name = ctx.get_key(key_id)
        assert key_name == "general.architecture"


def test_gguf_tensor_info():
    """Test reading tensor information."""
    model_path = "models/Llama-3.2-1B-Instruct-Q8_0.gguf"
    ctx = GGUFContext.from_file(model_path)

    # Get tensor info
    tensors = ctx.get_all_tensor_info()
    assert isinstance(tensors, list)
    assert len(tensors) == ctx.n_tensors

    # Check first tensor
    if len(tensors) > 0:
        tensor = tensors[0]
        assert "name" in tensor
        assert "type" in tensor
        assert "offset" in tensor
        assert "size" in tensor
        print(f"\nFirst tensor: {tensor['name']}")
        print(f"  Type: {tensor['type']}")
        print(f"  Size: {tensor['size']} bytes")

    # Test find_tensor
    if len(tensors) > 0:
        tensor_name = tensors[0]["name"]
        tensor_id = ctx.find_tensor(tensor_name)
        assert tensor_id >= 0

        found_name = ctx.get_tensor_name(tensor_id)
        assert found_name == tensor_name


def test_gguf_create_empty():
    """Test creating empty GGUF context."""
    ctx = GGUFContext.empty()

    assert ctx.version > 0
    assert ctx.n_tensors == 0
    assert ctx.n_kv == 0

    # Add metadata
    ctx.set_val_str("test.string", "hello")
    ctx.set_val_u32("test.number", 42)
    ctx.set_val_bool("test.bool", True)
    ctx.set_val_f32("test.float", 3.14)

    # Verify values
    assert ctx.get_value("test.string") == "hello"
    assert ctx.get_value("test.number") == 42
    assert ctx.get_value("test.bool") == True
    assert abs(ctx.get_value("test.float") - 3.14) < 0.01

    # Check KV count increased
    assert ctx.n_kv == 4

    print("\nCreated empty GGUF with metadata:")
    for key, value in ctx.get_all_metadata().items():
        print(f"  {key}: {value}")


def test_gguf_remove_key():
    """Test removing keys from GGUF context."""
    ctx = GGUFContext.empty()

    ctx.set_val_str("test.key1", "value1")
    ctx.set_val_str("test.key2", "value2")
    assert ctx.n_kv == 2

    # Remove key
    removed_id = ctx.remove_key("test.key1")
    assert removed_id >= 0
    assert ctx.n_kv == 1

    # Try to get removed key
    with pytest.raises(KeyError):
        ctx.get_value("test.key1")

    # Other key should still exist
    assert ctx.get_value("test.key2") == "value2"

    # Remove non-existent key
    removed_id = ctx.remove_key("nonexistent")
    assert removed_id == -1


def test_gguf_key_not_found():
    """Test error handling for non-existent keys."""
    ctx = GGUFContext.empty()

    # find_key should return -1
    key_id = ctx.find_key("nonexistent")
    assert key_id == -1

    # get_value should raise KeyError
    with pytest.raises(KeyError):
        ctx.get_value("nonexistent")


if __name__ == "__main__":
    # Run tests manually
    test_gguf_read_model()
    test_gguf_metadata()
    test_gguf_tensor_info()
    test_gguf_create_empty()
    test_gguf_remove_key()
    test_gguf_key_not_found()
    print("\nAll GGUF tests passed!")
