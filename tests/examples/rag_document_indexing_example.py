"""
RAG Document Indexing example.

This example demonstrates document loading and indexing with cyllama's RAG:
- Load documents from files (text, markdown, JSON)
- Split documents into chunks
- Generate embeddings and store in vector database
- Persist the vector store to disk for later use

Usage:
    python rag_document_indexing_example.py -e <embedding_model>

Example:
    python rag_document_indexing_example.py -e models/gte-small-q8_0.gguf
"""

import argparse
import tempfile
from pathlib import Path

from cyllama.rag import (
    Embedder,
    VectorStore,
    TextSplitter,
    MarkdownSplitter,
    TextLoader,
    MarkdownLoader,
    DirectoryLoader,
    load_document,
)
from cyllama.utils.color import header, section, info, success, bullet, kv, error


def create_sample_files(temp_dir: Path) -> list[Path]:
    """Create sample files for demonstration."""
    files = []

    # Create a sample text file
    txt_file = temp_dir / "python_guide.txt"
    txt_file.write_text("""
Python Programming Guide

Python is a high-level, interpreted programming language known for its
simplicity and readability. It was created by Guido van Rossum and
first released in 1991.

Key Features:
- Easy to learn and use
- Dynamic typing
- Interpreted language
- Object-oriented and functional programming support
- Large standard library

Python is widely used in:
- Web development (Django, Flask)
- Data science and machine learning (NumPy, Pandas, scikit-learn)
- Automation and scripting
- Scientific computing
- Artificial intelligence
""")
    files.append(txt_file)

    # Create a sample markdown file
    md_file = temp_dir / "rust_intro.md"
    md_file.write_text("""# Introduction to Rust

Rust is a systems programming language focused on safety, speed, and concurrency.

## Key Features

### Memory Safety
Rust guarantees memory safety without needing a garbage collector. It uses a
unique ownership system with rules that the compiler checks at compile time.

### Zero-Cost Abstractions
You don't pay a runtime cost for abstractions in Rust. High-level constructs
compile down to efficient low-level code.

### Concurrency
Rust's ownership system prevents data races at compile time, making concurrent
programming safer and easier.

## When to Use Rust

- System programming
- WebAssembly applications
- Command-line tools
- Network services
- Embedded systems
""")
    files.append(md_file)

    # Create a sample JSON file
    json_file = temp_dir / "languages.json"
    json_file.write_text("""[
    {
        "name": "Go",
        "description": "Go is a statically typed, compiled language designed at Google. It features garbage collection, structural typing, and CSP-style concurrency.",
        "creator": "Robert Griesemer, Rob Pike, Ken Thompson",
        "year": 2009
    },
    {
        "name": "TypeScript",
        "description": "TypeScript is a typed superset of JavaScript that compiles to plain JavaScript. It adds optional static typing and class-based OOP.",
        "creator": "Microsoft",
        "year": 2012
    }
]""")
    files.append(json_file)

    return files


def main():
    """Run the document indexing example."""
    parser = argparse.ArgumentParser(
        description="RAG Document Indexing Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python rag_document_indexing_example.py -e models/gte-small-q8_0.gguf
        """,
    )
    parser.add_argument(
        "-e", "--embedding-model",
        type=str,
        required=True,
        help="Path to embedding model (GGUF file)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    embedding_model = Path(args.embedding_model)

    if not embedding_model.exists():
        error(f"Embedding model not found: {embedding_model}")
        return 1

    header("cyllama RAG Document Indexing Example")

    info(f"Embedding model: {embedding_model.name}")

    # Create temporary directory for sample files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create sample files
        section("Creating sample documents...")
        sample_files = create_sample_files(temp_path)
        for f in sample_files:
            bullet(f"Created: {f.name}")

        # Initialize embedder
        section("Initializing embedder...")
        embedder = Embedder(
            str(embedding_model),
            n_ctx=512,
            pooling="mean",
            normalize=True,
        )
        info(f"Embedding dimension: {embedder.dimension}")

        # Demonstrate different loaders
        section("Loading documents with different loaders...")

        # 1. TextLoader
        print()
        bullet("TextLoader (plain text files)")
        loader = TextLoader()
        txt_docs = loader.load(str(sample_files[0]))
        info(f"  Loaded {len(txt_docs)} document(s) from {sample_files[0].name}")

        # 2. MarkdownLoader
        print()
        bullet("MarkdownLoader (preserves structure)")
        md_loader = MarkdownLoader(parse_frontmatter=True)
        md_docs = md_loader.load(str(sample_files[1]))
        info(f"  Loaded {len(md_docs)} document(s) from {sample_files[1].name}")

        # 3. load_document (auto-detect format)
        print()
        bullet("load_document() (auto-detect format)")
        json_docs = load_document(str(sample_files[2]), text_key="description")
        info(f"  Loaded {len(json_docs)} document(s) from {sample_files[2].name}")

        # 4. DirectoryLoader (batch load)
        print()
        bullet("DirectoryLoader (batch loading)")
        dir_loader = DirectoryLoader(glob="*.txt")
        dir_docs = dir_loader.load(str(temp_path))
        info(f"  Loaded {len(dir_docs)} document(s) from directory")

        # Demonstrate text splitting
        section("Splitting documents into chunks...")

        # TextSplitter
        splitter = TextSplitter(
            chunk_size=256,
            chunk_overlap=50,
        )

        all_chunks = []
        for doc in txt_docs + md_docs:
            chunks = splitter.split(doc.text)
            all_chunks.extend(chunks)

        # For JSON docs (already separate items)
        for doc in json_docs:
            all_chunks.append(doc.text)

        info(f"Total chunks: {len(all_chunks)}")

        # Display sample chunks
        print()
        bullet("Sample chunks:")
        for i, chunk in enumerate(all_chunks[:3], 1):
            preview = chunk[:80].replace('\n', ' ').strip()
            print(f"    [{i}] {preview}...")

        # Create vector store and index documents
        section("Creating vector store and indexing...")

        # Use a temporary database file
        db_path = temp_path / "vectors.db"

        with VectorStore(
            dimension=embedder.dimension,
            db_path=str(db_path),
            metric="cosine",
        ) as store:
            # Generate embeddings
            info("Generating embeddings...")
            embeddings = embedder.embed_batch(all_chunks)
            info(f"Generated {len(embeddings)} embeddings")

            # Add to store with metadata
            metadata_list = [{"chunk_index": i} for i in range(len(all_chunks))]
            ids = store.add(embeddings, all_chunks, metadata_list)
            info(f"Indexed {len(ids)} chunks")
            kv("Database file", str(db_path))
            kv("Database size", f"{db_path.stat().st_size / 1024:.1f} KB")

            # Demonstrate search
            section("Searching the vector store...")

            queries = [
                "memory safety without garbage collector",
                "web development frameworks",
                "Google programming language",
            ]

            for query in queries:
                print()
                bullet(f"Query: '{query}'")
                query_emb = embedder.embed(query)
                results = store.search(query_emb, k=2, threshold=0.3)

                if results:
                    for r in results:
                        preview = r.text[:80].replace('\n', ' ').strip()
                        print(f"    [{r.score:.3f}] {preview}...")
                else:
                    print("    No results found above threshold")

            # Demonstrate persistence
            section("Demonstrating persistence...")
            info(f"Vector count before closing: {len(store)}")

        # Reopen the database to verify persistence
        print()
        bullet("Reopening database to verify persistence...")
        with VectorStore.open(str(db_path)) as store:
            info(f"Vector count after reopening: {len(store)}")
            success("Data persisted successfully!")

        embedder.close()

    success("Document indexing example completed!")
    return 0


if __name__ == "__main__":
    exit(main())
