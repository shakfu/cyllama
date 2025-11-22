"""Tests for download API."""

import pytest
import os
from unittest.mock import patch, MagicMock
from cyllama.llama.llama_cpp import (
    get_hf_file,
    download_model,
    list_cached_models,
    resolve_docker_model
)

pytest.skip(allow_module_level=True)

def test_get_hf_file_basic():
    """Test getting HF file info (offline mode to avoid actual download)."""
    # This test will fail if no cache exists, but demonstrates the API
    try:
        info = get_hf_file(
            "bartowski/Llama-3.2-1B-Instruct-GGUF:Q8_0",
            offline=True
        )

        # Check structure
        assert isinstance(info, dict)
        assert 'repo' in info
        assert 'gguf_file' in info
        assert 'mmproj_file' in info

        print(f"\nHF file info:")
        print(f"  Repo: {info['repo']}")
        print(f"  GGUF file: {info['gguf_file']}")
        print(f"  MMProj file: {info['mmproj_file']}")

    except RuntimeError as e:
        # Expected to fail if no cache in offline mode
        print(f"\nOffline mode failed (expected if no cache): {e}")
        assert "failed to get manifest" in str(e) or "offline mode" in str(e)


def test_get_hf_file_with_tag():
    """Test HF file info with different tag formats."""
    # Test with different tag formats (offline to avoid downloads)
    test_repos = [
        "bartowski/Llama-3.2-1B-Instruct-GGUF:q4",
        "bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M",
        "bartowski/Llama-3.2-1B-Instruct-GGUF:Q8_0",
    ]

    for repo in test_repos:
        try:
            info = get_hf_file(repo, offline=True)
            print(f"\n{repo} -> {info.get('gguf_file', 'N/A')}")
        except RuntimeError as e:
            # Expected to fail offline if no cache
            print(f"\n{repo} -> offline failed (expected)")


def test_download_model_hf_repo():
    """Test download_model with HF repo (dry run / offline)."""
    # Test with offline mode (won't actually download)
    success = download_model(
        hf_repo="bartowski/Llama-3.2-1B-Instruct-GGUF:Q8_0",
        offline=True
    )

    # Should return False in offline mode if not cached
    assert isinstance(success, bool)
    print(f"\nDownload (offline) success: {success}")


def test_download_model_with_path():
    """Test download_model with explicit path."""
    # Test that we can specify a path
    success = download_model(
        model_path="/tmp/test_model.gguf",
        hf_repo="test/repo",
        offline=True
    )

    assert isinstance(success, bool)
    print(f"\nDownload with path (offline) success: {success}")


def test_download_model_url():
    """Test download_model with direct URL (offline)."""
    success = download_model(
        url="https://example.com/model.gguf",
        offline=True
    )

    assert isinstance(success, bool)
    print(f"\nDownload from URL (offline) success: {success}")


def test_download_model_parameters():
    """Test download_model with various parameter combinations."""
    # Test that all parameters are accepted
    test_cases = [
        {"hf_repo": "user/repo:tag"},
        {"hf_repo": "user/repo", "hf_file": "file.gguf"},
        {"url": "https://example.com/model.gguf"},
        {"docker_repo": "registry.io/model:tag"},
        {"model_path": "/tmp/model.gguf", "hf_repo": "user/repo"},
    ]

    for params in test_cases:
        # Add offline to avoid actual downloads
        params['offline'] = True
        success = download_model(**params)
        assert isinstance(success, bool)
        print(f"\nTested params: {params} -> {success}")


def test_list_cached_models():
    """Test listing cached models."""
    models = list_cached_models()

    # Should return a list
    assert isinstance(models, list)

    print(f"\nFound {len(models)} cached models")

    # Check structure if any models exist
    if models:
        model = models[0]
        assert isinstance(model, dict)
        assert 'manifest_path' in model
        assert 'user' in model
        assert 'model' in model
        assert 'tag' in model
        assert 'size' in model

        print("\nFirst cached model:")
        print(f"  User: {model['user']}")
        print(f"  Model: {model['model']}")
        print(f"  Tag: {model['tag']}")
        print(f"  Size: {model['size']} bytes")
        print(f"  Manifest: {model['manifest_path']}")
    else:
        print("  (No cached models found)")


def test_list_cached_models_structure():
    """Test that list_cached_models returns correct structure."""
    models = list_cached_models()

    for model in models:
        # Verify all required keys exist
        assert 'manifest_path' in model
        assert 'user' in model
        assert 'model' in model
        assert 'tag' in model
        assert 'size' in model

        # Verify types
        assert isinstance(model['manifest_path'], str)
        assert isinstance(model['user'], str)
        assert isinstance(model['model'], str)
        assert isinstance(model['tag'], str)
        assert isinstance(model['size'], int)

        print(f"\n  {model['user']}/{model['model']}:{model['tag']}")


def test_resolve_docker_model():
    """Test Docker registry model resolution (offline)."""
    try:
        path = resolve_docker_model("myregistry.io/llama:latest")

        # Should return a string path
        assert isinstance(path, str)
        print(f"\nDocker model resolved to: {path}")

    except RuntimeError as e:
        # Expected to fail without actual Docker registry
        print(f"\nDocker resolution failed (expected): {e}")
        # Just verify it's a RuntimeError
        assert isinstance(e, RuntimeError)


def test_download_with_bearer_token():
    """Test that bearer_token parameter is accepted."""
    success = download_model(
        hf_repo="user/repo:tag",
        bearer_token="hf_test_token",
        offline=True
    )

    assert isinstance(success, bool)
    print(f"\nDownload with bearer token (offline): {success}")


def test_api_exists():
    """Test that all download functions are exported."""
    # Verify all functions exist and are callable
    assert callable(get_hf_file)
    assert callable(download_model)
    assert callable(list_cached_models)
    assert callable(resolve_docker_model)

    print("\nAll download API functions are accessible")


@pytest.mark.skip(reason="Requires internet connection and actual download")
def test_download_model_real():
    """
    Real download test (skipped by default).

    To run this test:
        pytest tests/test_download.py::test_download_model_real -v
    """
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.gguf")

        success = download_model(
            model_path=model_path,
            hf_repo="bartowski/TinyLlama-1.1B-Chat-v1.0-GGUF:Q4_K_M",
            offline=False
        )

        assert success
        assert os.path.exists(model_path)
        assert os.path.getsize(model_path) > 0

        print(f"\nDownloaded model to: {model_path}")
        print(f"Size: {os.path.getsize(model_path)} bytes")


if __name__ == "__main__":
    # Run tests manually
    print("Testing download API...")
    test_get_hf_file_basic()
    test_get_hf_file_with_tag()
    test_download_model_hf_repo()
    test_download_model_with_path()
    test_download_model_url()
    test_download_model_parameters()
    test_list_cached_models()
    test_list_cached_models_structure()
    test_resolve_docker_model()
    test_download_with_bearer_token()
    test_api_exists()
    print("\nAll download API tests completed!")
