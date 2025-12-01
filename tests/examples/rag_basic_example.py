"""
Basic RAG (Retrieval-Augmented Generation) example.

This example demonstrates the simplest way to use cyllama's RAG functionality:
- Initialize RAG with embedding and generation models
- Add documents to a knowledge base
- Query the knowledge base and get AI-generated answers

Usage:
    python rag_basic_example.py -e <embedding_model> -m <generation_model>

Example:
    python rag_basic_example.py -e models/gte-small-q8_0.gguf -m models/Llama-3.2-1B-Instruct-Q8_0.gguf
"""

import argparse
from pathlib import Path

from cyllama.rag import RAG, RAGConfig
from cyllama.utils.color import header, section, info, success, bullet, kv, error


def main():
    """Run the basic RAG example."""
    parser = argparse.ArgumentParser(
        description="Basic RAG Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python rag_basic_example.py -e models/gte-small-q8_0.gguf -m models/Llama-3.2-1B-Instruct-Q8_0.gguf
        """,
    )
    parser.add_argument(
        "-e", "--embedding-model",
        type=str,
        required=True,
        help="Path to embedding model (GGUF file)",
    )
    parser.add_argument(
        "-m", "--generation-model",
        type=str,
        required=True,
        help="Path to generation model (GGUF file)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    embedding_model = Path(args.embedding_model)
    generation_model = Path(args.generation_model)

    if not embedding_model.exists():
        error(f"Embedding model not found: {embedding_model}")
        return 1

    if not generation_model.exists():
        error(f"Generation model not found: {generation_model}")
        return 1

    header("cyllama RAG Basic Example")

    info(f"Embedding model: {embedding_model.name}")
    info(f"Generation model: {generation_model.name}")

    # Configure RAG
    config = RAGConfig(
        top_k=3,                    # Retrieve top 3 documents
        similarity_threshold=0.3,   # Minimum similarity score
        max_tokens=256,             # Maximum response length
        temperature=0.7,            # Generation temperature
    )

    # Knowledge base - sample documents about programming languages
    documents = [
        "Python was created by Guido van Rossum and first released in 1991. "
        "It emphasizes code readability and uses significant indentation.",

        "JavaScript was created by Brendan Eich in 1995 at Netscape. "
        "It is the primary language for web browsers and enables interactive web pages.",

        "Rust was first released in 2010 and developed by Mozilla. "
        "It focuses on memory safety without garbage collection.",

        "Go (Golang) was designed at Google and released in 2009. "
        "It was created by Robert Griesemer, Rob Pike, and Ken Thompson.",

        "TypeScript is a superset of JavaScript developed by Microsoft. "
        "It adds optional static typing and class-based programming.",

        "C++ was designed by Bjarne Stroustrup starting in 1979. "
        "It provides object-oriented features to the C language.",
    ]

    section("Initializing RAG system...")

    with RAG(
        embedding_model=str(embedding_model),
        generation_model=str(generation_model),
        chunk_size=512,
        chunk_overlap=50,
        config=config,
    ) as rag:
        # Add documents to knowledge base
        section("Adding documents to knowledge base...")
        rag.add_texts(documents)
        info(f"Added {len(documents)} documents")

        # Example queries
        queries = [
            "Who created Python and when?",
            "Which language focuses on memory safety?",
            "What language was designed at Google?",
        ]

        section("Querying the knowledge base...")

        for query in queries:
            print()
            bullet(f"Query: {query}")

            # Get response from RAG
            response = rag.query(query)

            # Display answer
            success(f"Answer: {response.text}")

            # Display sources
            if response.sources:
                kv("Sources", f"{len(response.sources)} document(s)")
                for i, source in enumerate(response.sources, 1):
                    print(f"    [{i}] (score: {source.score:.3f}) {source.text[:80]}...")

        # Demonstrate retrieve-only (no generation)
        section("Retrieve-only example (no generation)...")
        sources = rag.retrieve("memory safety programming")
        print()
        bullet("Query: 'memory safety programming'")
        for i, source in enumerate(sources, 1):
            print(f"  [{source.score:.3f}] {source.text[:100]}...")

    success("RAG example completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
