"""
RAG Streaming Response example.

This example demonstrates streaming responses with cyllama's RAG:
- Stream generated tokens as they're produced
- Real-time response display
- Handle streaming with context managers

Usage:
    python rag_streaming_example.py -e <embedding_model> -m <generation_model>

Example:
    python rag_streaming_example.py -e models/gte-small-q8_0.gguf -m models/Llama-3.2-1B-Instruct-Q8_0.gguf
"""

import argparse
import sys
import time
from pathlib import Path

from cyllama.rag import RAG, RAGConfig
from cyllama.utils.color import header, section, info, success, bullet, error


def main():
    """Run the streaming RAG example."""
    parser = argparse.ArgumentParser(
        description="RAG Streaming Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python rag_streaming_example.py -e models/gte-small-q8_0.gguf -m models/Llama-3.2-1B-Instruct-Q8_0.gguf
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

    args = parser.parse_args()

    embedding_model = Path(args.embedding_model)
    generation_model = Path(args.generation_model)

    if not embedding_model.exists():
        error(f"Embedding model not found: {embedding_model}")
        return 1

    if not generation_model.exists():
        error(f"Generation model not found: {generation_model}")
        return 1

    header("cyllama RAG Streaming Example")

    info(f"Embedding model: {embedding_model.name}")
    info(f"Generation model: {generation_model.name}")

    # Knowledge base about space exploration
    documents = [
        "The Apollo 11 mission successfully landed the first humans on the Moon "
        "on July 20, 1969. Neil Armstrong and Buzz Aldrin walked on the lunar surface "
        "while Michael Collins orbited above in the command module.",

        "The International Space Station (ISS) is a modular space station in low Earth "
        "orbit. It is a multinational collaborative project involving NASA, Roscosmos, "
        "JAXA, ESA, and CSA. The first module was launched in 1998.",

        "SpaceX's Falcon 9 is a reusable rocket designed by SpaceX. It was first "
        "launched in 2010 and has become the world's most frequently launched rocket. "
        "The first stage can land vertically for reuse.",

        "The Hubble Space Telescope was launched in 1990 and remains one of the most "
        "important astronomical instruments ever built. It orbits Earth at about "
        "547 km altitude and has made over 1.5 million observations.",

        "Mars rovers include Sojourner (1997), Spirit and Opportunity (2004), "
        "Curiosity (2012), and Perseverance (2021). These robots have explored "
        "the Martian surface and searched for signs of past microbial life.",
    ]

    config = RAGConfig(
        top_k=3,
        max_tokens=256,
        temperature=0.7,
        prompt_template="""Use the following context to answer the question.
Be concise but informative.

Context:
{context}

Question: {question}

Answer:""",
    )

    section("Initializing RAG system...")

    with RAG(
        embedding_model=str(embedding_model),
        generation_model=str(generation_model),
        config=config,
    ) as rag:
        # Add documents
        section("Building knowledge base...")
        rag.add_texts(documents)
        info(f"Added {len(documents)} documents")

        # Streaming queries
        queries = [
            "When did humans first land on the Moon?",
            "What is the International Space Station?",
            "Tell me about reusable rockets.",
        ]

        section("Streaming responses...")

        for query in queries:
            print()
            bullet(f"Question: {query}")
            print()
            print("    Answer: ", end="")
            sys.stdout.flush()

            # Track timing
            start_time = time.time()
            token_count = 0
            first_token_time = None

            # Stream the response
            for chunk in rag.stream(query):
                if first_token_time is None:
                    first_token_time = time.time()

                print(chunk, end="")
                sys.stdout.flush()
                token_count += 1

            end_time = time.time()

            # Print statistics
            print()
            print()
            total_time = end_time - start_time
            ttft = first_token_time - start_time if first_token_time else 0
            tokens_per_sec = token_count / total_time if total_time > 0 else 0

            info(f"    Time to first token: {ttft:.2f}s")
            info(f"    Total time: {total_time:.2f}s")
            info(f"    Tokens: {token_count}")
            info(f"    Speed: {tokens_per_sec:.1f} tokens/sec")

        # Compare streaming vs non-streaming
        section("Comparing streaming vs non-streaming...")

        query = "What rovers have explored Mars?"
        print()
        bullet(f"Query: {query}")

        # Non-streaming
        print()
        info("Non-streaming mode:")
        start = time.time()
        response = rag.query(query)
        elapsed = time.time() - start
        print(f"    Answer: {response.text[:150]}...")
        info(f"    Total time: {elapsed:.2f}s (response not available until complete)")

        # Streaming
        print()
        info("Streaming mode:")
        start = time.time()
        first_chunk_time = None
        full_response = []

        print("    Answer: ", end="")
        for chunk in rag.stream(query):
            if first_chunk_time is None:
                first_chunk_time = time.time()
            full_response.append(chunk)
            print(chunk, end="")
            sys.stdout.flush()

        print()
        elapsed = time.time() - start
        ttft = first_chunk_time - start if first_chunk_time else 0
        info(f"    Total time: {elapsed:.2f}s")
        info(f"    Time to first token: {ttft:.2f}s (response starts appearing immediately)")

    success("Streaming example completed!")
    return 0


if __name__ == "__main__":
    exit(main())
