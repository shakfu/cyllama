"""Tests for N-gram cache API."""

import pytest
import tempfile
import os
from cyllama.llama.llama_cpp import NgramCache


def test_ngram_cache_create():
    """Test creating an N-gram cache."""
    cache = NgramCache()
    assert cache is not None
    assert isinstance(cache, NgramCache)
    print(f"\nCreated: {cache}")


def test_ngram_cache_update():
    """Test updating cache with tokens."""
    cache = NgramCache()

    # Simple token sequence
    tokens = [1, 2, 3, 4, 5]
    cache.update(tokens, ngram_min=2, ngram_max=4)

    # Should not raise
    print("\nSuccessfully updated cache with tokens")


def test_ngram_cache_update_with_repetition():
    """Test updating cache with repeated patterns."""
    cache = NgramCache()

    # Sequence with repeated pattern [2, 3, 4]
    tokens = [1, 2, 3, 4, 5, 2, 3, 4, 6, 2, 3, 4, 7]
    cache.update(tokens, ngram_min=2, ngram_max=4)

    print(f"\nUpdated cache with {len(tokens)} tokens including repetitions")


def test_ngram_cache_incremental_update():
    """Test incremental updates with nnew parameter."""
    cache = NgramCache()

    # Initial tokens
    tokens = [1, 2, 3, 4, 5]
    cache.update(tokens, ngram_min=2, ngram_max=3)

    # Add more tokens
    tokens_extended = [1, 2, 3, 4, 5, 6, 7, 8]
    cache.update(tokens_extended, ngram_min=2, ngram_max=3, nnew=3)

    print("\nSuccessfully performed incremental update")


def test_ngram_cache_draft():
    """Test drafting tokens from cache."""
    cache = NgramCache()

    # Build cache with repeated pattern
    tokens = [1, 2, 3, 4, 5, 1, 2, 3, 4, 6, 1, 2, 3, 4, 7]
    cache.update(tokens, ngram_min=2, ngram_max=4)

    # Try to draft continuation of [1, 2]
    inp = [1, 2]
    draft = cache.draft(inp, n_draft=5, ngram_min=2, ngram_max=4)

    assert isinstance(draft, list)
    print(f"\nInput: {inp}")
    print(f"Drafted: {draft}")
    print(f"Draft length: {len(draft)}")


def test_ngram_cache_draft_empty():
    """Test drafting with empty cache."""
    cache = NgramCache()

    # Empty cache should return empty draft
    inp = [1, 2, 3]
    draft = cache.draft(inp, n_draft=5)

    assert isinstance(draft, list)
    assert len(draft) == 0
    print("\nEmpty cache produces empty draft (expected)")


def test_ngram_cache_draft_no_match():
    """Test drafting when no pattern matches."""
    cache = NgramCache()

    # Cache has one pattern
    tokens = [1, 2, 3, 4, 5]
    cache.update(tokens, ngram_min=2, ngram_max=4)

    # Try to draft from different pattern
    inp = [10, 11, 12]
    draft = cache.draft(inp, n_draft=5)

    assert isinstance(draft, list)
    print(f"\nNo matching pattern, draft: {draft}")


def test_ngram_cache_save_load():
    """Test saving and loading cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = os.path.join(tmpdir, "ngram_cache.bin")

        # Create and update cache
        cache1 = NgramCache()
        tokens = [1, 2, 3, 4, 5, 2, 3, 4, 6]
        cache1.update(tokens, ngram_min=2, ngram_max=4)

        # Save cache
        cache1.save(cache_file)
        assert os.path.exists(cache_file)
        print(f"\nSaved cache to {cache_file}")
        print(f"File size: {os.path.getsize(cache_file)} bytes")

        # Load cache
        cache2 = NgramCache.load(cache_file)
        assert cache2 is not None

        # Test that loaded cache works
        inp = [2, 3]
        draft = cache2.draft(inp, n_draft=3)
        print(f"Loaded cache draft for {inp}: {draft}")


def test_ngram_cache_merge():
    """Test merging two caches."""
    cache1 = NgramCache()
    cache1.update([1, 2, 3, 4, 5], ngram_min=2, ngram_max=3)

    cache2 = NgramCache()
    cache2.update([10, 11, 12, 13, 14], ngram_min=2, ngram_max=3)

    # Merge cache2 into cache1
    cache1.merge(cache2)

    # Both patterns should work
    draft1 = cache1.draft([1, 2], n_draft=3)
    draft2 = cache1.draft([10, 11], n_draft=3)

    print(f"\nAfter merge:")
    print(f"  Pattern [1,2] drafts: {draft1}")
    print(f"  Pattern [10,11] drafts: {draft2}")


def test_ngram_cache_merge_type_error():
    """Test that merging non-NgramCache raises error."""
    cache = NgramCache()

    with pytest.raises(TypeError, match="Can only merge with another NgramCache"):
        cache.merge("not a cache")

    with pytest.raises(TypeError):
        cache.merge([1, 2, 3])


def test_ngram_cache_parameters():
    """Test various parameter combinations."""
    cache = NgramCache()

    # Test different ngram_min/max values
    test_cases = [
        (1, 2),
        (2, 4),
        (3, 4),
        (1, 4),
    ]

    tokens = [1, 2, 3, 4, 5, 6, 7, 8]

    for ngram_min, ngram_max in test_cases:
        cache_test = NgramCache()
        cache_test.update(tokens, ngram_min=ngram_min, ngram_max=ngram_max)
        draft = cache_test.draft([1, 2], n_draft=3, ngram_min=ngram_min, ngram_max=ngram_max)
        print(f"\nngram_min={ngram_min}, ngram_max={ngram_max}: draft={draft}")


def test_ngram_cache_large_sequence():
    """Test with larger token sequence."""
    cache = NgramCache()

    # Generate larger sequence with patterns
    tokens = []
    for i in range(10):
        tokens.extend([1, 2, 3, 4, 5])
        tokens.extend([10, 11, 12, 13, 14])

    cache.update(tokens, ngram_min=2, ngram_max=4)

    # Should predict well due to strong patterns
    inp = [1, 2, 3]
    draft = cache.draft(inp, n_draft=10)

    print(f"\nLarge sequence test:")
    print(f"  Total tokens: {len(tokens)}")
    print(f"  Input: {inp}")
    print(f"  Draft: {draft}")
    print(f"  Draft length: {len(draft)}")


def test_ngram_cache_with_context_caches():
    """Test draft with separate context/dynamic/static caches."""
    context_cache = NgramCache()
    context_cache.update([1, 2, 3, 4, 5], ngram_min=2, ngram_max=3)

    dynamic_cache = NgramCache()
    dynamic_cache.update([1, 2, 3, 6, 7], ngram_min=2, ngram_max=3)

    static_cache = NgramCache()
    static_cache.update([1, 2, 3, 8, 9], ngram_min=2, ngram_max=3)

    # Draft using all three caches
    inp = [1, 2]
    draft = context_cache.draft(
        inp,
        n_draft=5,
        context_cache=context_cache,
        dynamic_cache=dynamic_cache,
        static_cache=static_cache
    )

    print(f"\nMulti-cache draft for {inp}: {draft}")


def test_ngram_cache_repr():
    """Test string representation."""
    cache = NgramCache()
    repr_str = repr(cache)

    assert "NgramCache" in repr_str
    assert "0x" in repr_str  # hex address
    print(f"\nRepr: {repr_str}")


if __name__ == "__main__":
    # Run tests manually
    print("Testing N-gram cache API...")
    test_ngram_cache_create()
    test_ngram_cache_update()
    test_ngram_cache_update_with_repetition()
    test_ngram_cache_incremental_update()
    test_ngram_cache_draft()
    test_ngram_cache_draft_empty()
    test_ngram_cache_draft_no_match()
    test_ngram_cache_save_load()
    test_ngram_cache_merge()
    test_ngram_cache_merge_type_error()
    test_ngram_cache_parameters()
    test_ngram_cache_large_sequence()
    test_ngram_cache_with_context_caches()
    test_ngram_cache_repr()
    print("\nAll N-gram cache tests completed!")
