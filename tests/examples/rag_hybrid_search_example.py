"""
RAG Hybrid Search example.

This example demonstrates hybrid search combining vector similarity with
keyword search (FTS5) for improved retrieval:
- HybridStore combines vector embeddings with full-text search
- Reciprocal Rank Fusion (RRF) merges results from both methods
- Alpha parameter controls the balance between vector and keyword search

Usage:
    python rag_hybrid_search_example.py -e <embedding_model>

Example:
    python rag_hybrid_search_example.py -e models/gte-small-q8_0.gguf
"""

import argparse
import tempfile
from pathlib import Path

from cyllama.rag import Embedder, HybridStore
from cyllama.utils.color import header, section, info, success, bullet, kv, warning, error


def main():
    """Run the hybrid search example."""
    parser = argparse.ArgumentParser(
        description="RAG Hybrid Search Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python rag_hybrid_search_example.py -e models/gte-small-q8_0.gguf
        """,
    )
    parser.add_argument(
        "-e", "--embedding-model",
        type=str,
        required=True,
        help="Path to embedding model (GGUF file)",
    )

    args = parser.parse_args()

    embedding_model = Path(args.embedding_model)

    if not embedding_model.exists():
        error(f"Embedding model not found: {embedding_model}")
        return 1

    header("cyllama RAG Hybrid Search Example")

    info(f"Embedding model: {embedding_model.name}")

    # Technical documentation with specific terms
    documents = [
        "Python PEP-8 defines the style guide for Python code. It recommends "
        "using 4 spaces for indentation and limiting lines to 79 characters.",

        "The HTTP/2 protocol improves web performance through multiplexing, "
        "header compression, and server push capabilities.",

        "PostgreSQL JSONB type provides efficient storage and querying of "
        "JSON data with indexing support using GIN indexes.",

        "Docker containers use cgroups and namespaces for isolation. "
        "Images are built from Dockerfiles with layered filesystems.",

        "REST API design follows principles like statelessness, uniform "
        "interface, and resource-based URLs for web services.",

        "GraphQL queries allow clients to request exactly the data they need. "
        "It uses a strongly typed schema and introspection.",

        "Redis supports data structures like strings, hashes, lists, sets, "
        "and sorted sets with optional persistence and clustering.",

        "Kubernetes pods are the smallest deployable units containing one or "
        "more containers that share storage and network resources.",
    ]

    # Initialize embedder
    section("Initializing embedder...")
    embedder = Embedder(
        str(embedding_model),
        n_ctx=512,
        pooling="mean",
        normalize=True,
    )
    info(f"Embedding dimension: {embedder.dimension}")

    # Generate embeddings
    section("Generating embeddings...")
    embeddings = embedder.embed_batch(documents)
    info(f"Generated {len(embeddings)} embeddings")

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "hybrid.db"

        # Create HybridStore with default alpha
        section("Creating HybridStore...")
        with HybridStore(
            dimension=embedder.dimension,
            db_path=str(db_path),
            alpha=0.5,  # Equal weight between vector and FTS
        ) as store:
            # Add documents
            ids = store.add(embeddings, documents)
            info(f"Indexed {len(ids)} documents")
            kv("Alpha (default)", "0.5 (balanced)")

            # Compare different search modes
            section("Comparing search modes...")

            query = "container deployment Kubernetes"
            print()
            bullet(f"Query: '{query}'")

            # 1. Pure vector search (alpha=1.0)
            print()
            info("Pure Vector Search (alpha=1.0):")
            info("  Uses semantic similarity only")
            query_emb = embedder.embed(query)
            results = store.search(query_emb, query_text=query, k=3, alpha=1.0)
            for r in results:
                preview = r.text[:70].replace('\n', ' ')
                print(f"    [{r.score:.3f}] {preview}...")

            # 2. Pure keyword search (alpha=0.0)
            print()
            info("Pure Keyword Search (alpha=0.0):")
            info("  Uses FTS5 full-text search only")
            results = store.search(query_emb, query_text=query, k=3, alpha=0.0)
            for r in results:
                preview = r.text[:70].replace('\n', ' ')
                print(f"    [{r.score:.3f}] {preview}...")

            # 3. Hybrid search (alpha=0.5)
            print()
            info("Hybrid Search (alpha=0.5):")
            info("  Combines both methods with Reciprocal Rank Fusion")
            results = store.search(query_emb, query_text=query, k=3, alpha=0.5)
            for r in results:
                preview = r.text[:70].replace('\n', ' ')
                print(f"    [{r.score:.3f}] {preview}...")

            # Demonstrate when hybrid search helps
            section("When hybrid search excels...")

            # Case 1: Specific technical term
            print()
            bullet("Case 1: Specific technical term (PEP-8)")
            query = "PEP-8 Python style"
            query_emb = embedder.embed(query)

            print("  Vector only (might miss exact term):")
            results = store.search(query_emb, query_text=query, k=2, alpha=1.0)
            for r in results:
                preview = r.text[:60].replace('\n', ' ')
                is_pep8 = "PEP-8" in r.text
                marker = "[*]" if is_pep8 else "[ ]"
                print(f"    {marker} [{r.score:.3f}] {preview}...")

            print("  Hybrid (boosted by keyword match):")
            results = store.search(query_emb, query_text=query, k=2, alpha=0.5)
            for r in results:
                preview = r.text[:60].replace('\n', ' ')
                is_pep8 = "PEP-8" in r.text
                marker = "[*]" if is_pep8 else "[ ]"
                print(f"    {marker} [{r.score:.3f}] {preview}...")

            # Case 2: Semantic query without keywords
            print()
            bullet("Case 2: Semantic query (no exact keywords)")
            query = "how to make web services faster"
            query_emb = embedder.embed(query)

            print("  Keyword only (may miss semantic intent):")
            results = store.search(query_emb, query_text=query, k=2, alpha=0.0)
            if results:
                for r in results:
                    preview = r.text[:60].replace('\n', ' ')
                    print(f"    [{r.score:.3f}] {preview}...")
            else:
                print("    No keyword matches found")

            print("  Hybrid (finds semantically relevant docs):")
            results = store.search(query_emb, query_text=query, k=2, alpha=0.5)
            for r in results:
                preview = r.text[:60].replace('\n', ' ')
                print(f"    [{r.score:.3f}] {preview}...")

            # Case 3: Mixed query
            print()
            bullet("Case 3: Mixed query (JSONB + concept)")
            query = "JSONB efficient JSON storage"
            query_emb = embedder.embed(query)

            print("  Different alpha values:")
            for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
                results = store.search(query_emb, query_text=query, k=1, alpha=alpha)
                if results:
                    preview = results[0].text[:50].replace('\n', ' ')
                    contains_jsonb = "JSONB" in results[0].text
                    marker = "[*]" if contains_jsonb else "[ ]"
                    print(f"    alpha={alpha}: {marker} {preview}...")

            # Recommendations
            section("Recommendations...")
            bullet("alpha=0.5: Balanced approach (default, good for most cases)")
            bullet("alpha=0.7: Favor semantic when queries are natural language")
            bullet("alpha=0.3: Favor keywords for technical terms, codes, IDs")
            bullet("alpha=1.0: Pure vector when exact terms don't matter")
            bullet("alpha=0.0: Pure FTS when exact matches are required")

        embedder.close()

    success("Hybrid search example completed!")
    return 0


if __name__ == "__main__":
    exit(main())
