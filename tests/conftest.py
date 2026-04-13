"""
Pytest configuration and shared fixtures for cyllama tests.

This module provides:
- Model path fixture (already heavily used)
- LLM instances with proper resource cleanup
- Generation config presets
- Custom pytest markers
"""

from pathlib import Path

import pytest

# Project root
ROOT = Path.cwd()

# Default model path constant (for use in subprocesses where fixtures aren't available)
DEFAULT_MODEL = "models/Llama-3.2-1B-Instruct-Q8_0.gguf"


# =============================================================================
# Core Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def model_path() -> str:
    """
    Provide the path to the default test model.

    This fixture is module-scoped to avoid repeated path construction.
    Heavily used across test files.

    Returns:
        Absolute path to the test model file.
    """
    return str(ROOT / DEFAULT_MODEL)


@pytest.fixture(scope="module")
def model_exists(model_path: str) -> bool:
    """Check if the test model exists."""
    return Path(model_path).exists()


# =============================================================================
# LLM Instance Fixtures with Cleanup
# =============================================================================


@pytest.fixture(scope="function")
def llm(model_path: str):
    """
    Provide an LLM instance with automatic cleanup.

    Creates a new LLM instance for each test function and ensures
    proper cleanup of resources (GPU memory, contexts) after the
    test completes.

    Yields:
        An LLM instance configured for testing (low max_tokens for speed).

    Example:
        def test_generation(llm):
            response = llm("What is 2+2?")
            assert "4" in response.text
    """
    from cyllama import LLM, GenerationConfig

    config = GenerationConfig(max_tokens=64, temperature=0.7)
    gen = LLM(model_path, config=config)
    yield gen
    gen.close()


@pytest.fixture(scope="function")
def llm_deterministic(model_path: str):
    """
    Provide an LLM instance with deterministic settings.

    Uses temperature=0 for greedy sampling - useful for tests
    that need reproducible output.

    Yields:
        An LLM instance with greedy sampling.
    """
    from cyllama import LLM, GenerationConfig

    config = GenerationConfig(
        max_tokens=64,
        temperature=0.0,
    )
    gen = LLM(model_path, config=config)
    yield gen
    gen.close()


@pytest.fixture(scope="module")
def llm_shared(model_path: str):
    """
    Provide a shared LLM instance for module-level reuse.

    Loads the model once per test module, improving performance
    when multiple tests need the same model. Use when tests don't
    need isolated LLM state.

    Yields:
        A shared LLM instance.
    """
    from cyllama import LLM, GenerationConfig

    config = GenerationConfig(max_tokens=128, temperature=0.7)
    gen = LLM(model_path, config=config)
    yield gen
    gen.close()


# =============================================================================
# SD Context Fixtures with Cleanup
# =============================================================================


@pytest.fixture
def sd_ctx_factory():
    """
    Factory fixture that constructs SDContext instances with automatic cleanup.

    SDContext holds ~4-6GB of Metal/ggml state per instance. Letting multiple
    instances survive simultaneously in the same test process crashes macOS
    Metal after ~5 cycles because cumulative Metal working-set pressure
    exceeds the GPU's recommended working set. The crash manifests as either
    a clean RuntimeError("Image generation failed") from the v0.2.3 validation
    guardrail, or a hard Fatal Python error abort from native code, depending
    on which failure path fires first.

    This fixture centralizes the forced-cleanup pattern. Tests that use it
    get automatic `del + gc.collect()` at teardown and do not need to
    remember the manual pattern in their test body.

    Usage:
        def test_something(sd_ctx_factory):
            from cyllama.sd import SDContext, SDContextParams

            params = SDContextParams()
            params.model_path = "models/sd_xl_turbo_1.0.q8_0.gguf"
            ctx = sd_ctx_factory(params)
            assert ctx.is_valid
            # No manual cleanup needed; teardown releases ctx + gc.collect()

    The fixture is function-scoped so every test gets a fresh cleanup cycle.
    If a test needs multiple contexts (e.g. to exercise a regression that
    requires two), call the factory multiple times -- all of them will be
    released on teardown.

    See docs/dev/test-cleanup.md for the full diagnosis and the 5-cycle
    reproducer that motivated this fixture.

    Yields:
        A callable `make(params) -> SDContext` that constructs and tracks
        SDContext instances for automatic teardown.
    """
    import gc

    created: list = []

    def make(params):
        from cyllama.sd import SDContext

        ctx = SDContext(params)
        created.append(ctx)
        return ctx

    yield make

    # Drop the factory's references, then force a collection cycle so
    # native cleanup runs deterministically before the next test starts.
    # Order matters: clear() first so the only remaining refs (if any) are
    # in the test frame, which pytest releases on return; gc.collect() then
    # finalizes everything.
    created.clear()
    gc.collect()


# =============================================================================
# Generation Config Fixtures
# =============================================================================


@pytest.fixture
def fast_config():
    """GenerationConfig optimized for fast testing (low max_tokens)."""
    from cyllama import GenerationConfig

    return GenerationConfig(max_tokens=32, temperature=0.7)


@pytest.fixture
def deterministic_config():
    """GenerationConfig for deterministic output (temperature=0)."""
    from cyllama import GenerationConfig

    return GenerationConfig(max_tokens=64, temperature=0.0)


# =============================================================================
# Helper Fixtures
# =============================================================================


@pytest.fixture
def chat_messages():
    """Sample chat messages for testing chat() function."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]


@pytest.fixture
def multi_turn_messages():
    """Multi-turn chat messages for conversation testing."""
    return [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "What is 5 + 3?"},
        {"role": "assistant", "content": "5 + 3 equals 8."},
        {"role": "user", "content": "And if I add 2 more?"},
    ]


# =============================================================================
# Pytest Markers and Hooks
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "requires_model: marks tests that require the test model")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU acceleration")


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests that require the model if it doesn't exist.

    This automatically detects tests using model-related fixtures:
    - model_path
    - llm
    - llm_deterministic
    - llm_shared

    Tests can also be explicitly marked with @pytest.mark.requires_model.
    """
    model_file = ROOT / DEFAULT_MODEL

    # Fixtures that require the model to exist
    model_fixtures = {"model_path", "llm", "llm_deterministic", "llm_shared"}

    if not model_file.exists():
        skip_no_model = pytest.mark.skip(reason="Model file not found")
        for item in items:
            # Skip if explicitly marked
            if "requires_model" in item.keywords:
                item.add_marker(skip_no_model)
                continue

            # Skip if test uses any model-related fixture
            if hasattr(item, "fixturenames"):
                if model_fixtures & set(item.fixturenames):
                    item.add_marker(skip_no_model)
