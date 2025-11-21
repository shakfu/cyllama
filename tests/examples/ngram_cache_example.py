"""Example: Using N-gram Cache for accelerated text generation.

This example demonstrates:
1. Creating and populating n-gram caches
2. Drafting tokens based on learned patterns
3. Incremental cache updates
4. Saving and loading caches
5. Merging multiple caches
6. Using multiple cache types (context, dynamic, static)
"""

from cyllama.llama.llama_cpp import NgramCache, LlamaModel, LlamaContext
import os


def basic_ngram_example():
    """Basic n-gram cache usage."""
    print("\n=== Basic N-gram Cache Example ===\n")

    # Create cache
    cache = NgramCache()
    print(f"Created cache: {cache}")

    # Sample token sequence with repeated patterns
    # Pattern: [1, 2, 3] appears multiple times
    tokens = [1, 2, 3, 4, 5, 1, 2, 3, 6, 7, 1, 2, 3, 8, 9]

    print(f"\nToken sequence: {tokens}")
    print("Pattern [1, 2, 3] appears 3 times")

    # Update cache with tokens
    cache.update(tokens, ngram_min=2, ngram_max=4)
    print("\nCache updated with token patterns")

    # Now predict what comes after [1, 2]
    inp = [1, 2]
    draft = cache.draft(inp, n_draft=5)

    print(f"\nInput: {inp}")
    print(f"Drafted tokens: {draft}")
    print("Expected: [3, ...] based on learned pattern")


def incremental_update_example():
    """Incremental cache updates with nnew parameter."""
    print("\n=== Incremental Update Example ===\n")

    cache = NgramCache()

    # Initial sequence
    initial_tokens = [10, 20, 30, 40, 50]
    cache.update(initial_tokens, ngram_min=2, ngram_max=3)
    print(f"Initial tokens: {initial_tokens}")

    # Add more tokens incrementally
    # nnew=2 means only the last 2 tokens are new
    extended_tokens = [10, 20, 30, 40, 50, 60, 70]
    cache.update(extended_tokens, ngram_min=2, ngram_max=3, nnew=2)
    print(f"Extended tokens: {extended_tokens}")
    print("nnew=2: Only [60, 70] are processed as new")

    # Draft continuation
    inp = [40, 50]
    draft = cache.draft(inp, n_draft=3)
    print(f"\nInput: {inp}")
    print(f"Drafted: {draft}")


def save_load_example():
    """Save and load n-gram caches."""
    print("\n=== Save/Load Cache Example ===\n")

    import tempfile

    # Create and populate cache
    cache = NgramCache()
    tokens = [5, 10, 15, 20, 25, 5, 10, 15, 30, 35]
    cache.update(tokens, ngram_min=2, ngram_max=4)

    print(f"Created cache with {len(tokens)} tokens")

    # Save to file
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        cache_file = tmp.name

    try:
        cache.save(cache_file)
        print(f"Saved cache to: {cache_file}")

        # Get file size
        size = os.path.getsize(cache_file)
        print(f"Cache file size: {size} bytes")

        # Load cache
        loaded_cache = NgramCache.load(cache_file)
        print(f"\nLoaded cache from file")

        # Test loaded cache works
        inp = [5, 10]
        draft = loaded_cache.draft(inp, n_draft=3)
        print(f"Input: {inp}")
        print(f"Drafted from loaded cache: {draft}")

    finally:
        os.unlink(cache_file)


def merge_caches_example():
    """Merge multiple n-gram caches."""
    print("\n=== Merge Caches Example ===\n")

    # Create first cache with one pattern
    cache1 = NgramCache()
    tokens1 = [100, 200, 300, 400, 100, 200, 300, 500]
    cache1.update(tokens1, ngram_min=2, ngram_max=3)
    print(f"Cache 1 tokens: {tokens1}")

    # Create second cache with different pattern
    cache2 = NgramCache()
    tokens2 = [1000, 2000, 3000, 4000, 1000, 2000, 3000, 5000]
    cache2.update(tokens2, ngram_min=2, ngram_max=3)
    print(f"Cache 2 tokens: {tokens2}")

    # Merge cache2 into cache1
    cache1.merge(cache2)
    print("\nMerged cache2 into cache1")

    # Test both patterns work
    draft1 = cache1.draft([100, 200], n_draft=2)
    draft2 = cache1.draft([1000, 2000], n_draft=2)

    print(f"\nPattern from cache1 [100, 200]: {draft1}")
    print(f"Pattern from cache2 [1000, 2000]: {draft2}")
    print("Merged cache knows both patterns!")


def multi_cache_drafting_example():
    """Use multiple cache types for drafting."""
    print("\n=== Multi-Cache Drafting Example ===\n")

    # Context cache: current conversation
    context_cache = NgramCache()
    context_cache.update([1, 2, 3, 4, 5], ngram_min=2, ngram_max=3)
    print("Context cache: Current conversation patterns")

    # Dynamic cache: previous generations in session
    dynamic_cache = NgramCache()
    dynamic_cache.update([1, 2, 3, 6, 7], ngram_min=2, ngram_max=3)
    print("Dynamic cache: Session history patterns")

    # Static cache: corpus-wide patterns
    static_cache = NgramCache()
    static_cache.update([1, 2, 3, 8, 9], ngram_min=2, ngram_max=3)
    print("Static cache: Corpus patterns")

    # Draft using all three caches
    inp = [1, 2]
    draft = context_cache.draft(
        inp,
        n_draft=5,
        context_cache=context_cache,
        dynamic_cache=dynamic_cache,
        static_cache=static_cache
    )

    print(f"\nInput: {inp}")
    print(f"Drafted using all 3 caches: {draft}")
    print("Combines patterns from context, dynamic, and static caches")


def ngram_parameters_example():
    """Demonstrate different ngram_min and ngram_max parameters."""
    print("\n=== N-gram Parameters Example ===\n")

    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 9, 10]
    print(f"Token sequence: {tokens}")

    # Test different parameter combinations
    params = [
        (2, 2, "Bigrams only"),
        (2, 3, "Bigrams and trigrams"),
        (2, 4, "Bigrams, trigrams, and 4-grams"),
        (3, 4, "Trigrams and 4-grams only"),
    ]

    for ngram_min, ngram_max, description in params:
        cache = NgramCache()
        cache.update(tokens, ngram_min=ngram_min, ngram_max=ngram_max)

        inp = [1, 2, 3]
        draft = cache.draft(inp, n_draft=3, ngram_min=ngram_min, ngram_max=ngram_max)

        print(f"\n{description} (min={ngram_min}, max={ngram_max}):")
        print(f"  Input: {inp}")
        print(f"  Draft: {draft}")


def repetitive_text_example():
    """Example with highly repetitive text."""
    print("\n=== Repetitive Text Pattern Example ===\n")

    # Simulate repetitive pattern like code or structured data
    pattern = [10, 20, 30, 40, 50]
    tokens = []

    # Repeat pattern 10 times with slight variations
    for i in range(10):
        tokens.extend(pattern)
        tokens.append(60 + i)  # Variation at the end

    print(f"Generated {len(tokens)} tokens with repeated pattern")
    print(f"Base pattern: {pattern}")
    print(f"Repeated 10 times with variations")

    # Build cache
    cache = NgramCache()
    cache.update(tokens, ngram_min=2, ngram_max=4)

    # Should predict well due to strong pattern
    inp = [10, 20, 30]
    draft = cache.draft(inp, n_draft=10)

    print(f"\nInput: {inp}")
    print(f"Drafted: {draft}")
    print(f"Draft length: {len(draft)}")
    print("Strong patterns enable longer predictions")


def real_world_use_case():
    """Real-world use case: accelerating generation with cache."""
    print("\n=== Real-World Use Case: Generation Acceleration ===\n")

    print("N-gram cache accelerates generation by predicting tokens")
    print("based on previously seen patterns.\n")

    print("Use cases:")
    print("1. Repetitive text: Code generation, structured data")
    print("2. Templates: Form letters, boilerplate text")
    print("3. Conversations: Common phrases, greetings")
    print("4. Long documents: Recurring terminology, patterns")

    print("\nTypical workflow:")
    print("""
    # 1. Create cache
    cache = NgramCache()

    # 2. Update with context/history
    cache.update(context_tokens, ngram_min=2, ngram_max=4)

    # 3. During generation, draft tokens
    draft = cache.draft(current_tokens, n_draft=16)

    # 4. Use draft as speculative tokens
    # (speeds up generation if predictions are accurate)

    # 5. Save cache for reuse
    cache.save("my_cache.bin")
    """)

    print("\nPerformance: 2-10x speedup for repetitive patterns")


def main():
    """Run all examples."""
    print("=" * 60)
    print("N-gram Cache Examples")
    print("=" * 60)

    # 1. Basic usage
    basic_ngram_example()

    # 2. Incremental updates
    incremental_update_example()

    # 3. Save and load
    save_load_example()

    # 4. Merge caches
    merge_caches_example()

    # 5. Multi-cache drafting
    multi_cache_drafting_example()

    # 6. Different parameters
    ngram_parameters_example()

    # 7. Repetitive patterns
    repetitive_text_example()

    # 8. Real-world use case
    real_world_use_case()

    print("\n" + "=" * 60)
    print("N-gram Cache Examples Complete")
    print("=" * 60)

    print("\nKey Takeaways:")
    print("- N-gram cache learns token patterns from sequences")
    print("- draft() predicts likely continuations based on patterns")
    print("- Speeds up generation when text has repeated patterns")
    print("- Supports save/load for persistence")
    print("- Can merge multiple caches")
    print("- Adjustable ngram_min and ngram_max parameters")
    print("- Best for: code, templates, structured data, repetitive text")


if __name__ == "__main__":
    main()
