"""
RAG Agent Integration example.

This example demonstrates integrating RAG with cyllama agents:
- Create a RAG-powered tool for agents
- Use ReActAgent to answer questions using knowledge retrieval
- Combine RAG with other tools for complex tasks

Usage:
    python rag_agent_example.py -e <embedding_model> -m <generation_model>

Example:
    python rag_agent_example.py -e models/gte-small-q8_0.gguf -m models/Llama-3.2-1B-Instruct-Q8_0.gguf
"""

import argparse
from datetime import datetime
from pathlib import Path

from cyllama import LLM, GenerationConfig
from cyllama.agents import ReActAgent, tool
from cyllama.rag import RAG, create_rag_tool
from cyllama.utils.color import header, section, info, success, bullet, kv, error


# Additional tools for the agent
@tool
def get_current_date() -> str:
    """
    Get the current date.

    Returns:
        Current date in YYYY-MM-DD format
    """
    return datetime.now().strftime("%Y-%m-%d")


@tool
def calculate(expression: str) -> float:
    """
    Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression (e.g., "2 + 2", "100 / 4")

    Returns:
        The result of the calculation
    """
    try:
        # Safe eval for basic math
        result = eval(expression, {"__builtins__": {}}, {})
        return float(result)
    except Exception as e:
        return f"Error: {str(e)}"


def main():
    """Run the RAG agent integration example."""
    parser = argparse.ArgumentParser(
        description="RAG Agent Integration Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python rag_agent_example.py -e models/gte-small-q8_0.gguf -m models/Llama-3.2-1B-Instruct-Q8_0.gguf
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
        help="Enable verbose agent output",
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

    header("cyllama RAG Agent Integration Example")

    info(f"Embedding model: {embedding_model.name}")
    info(f"Generation model: {generation_model.name}")

    # Company knowledge base
    company_docs = [
        "Acme Corp was founded in 2015 by Jane Smith and John Doe. "
        "The company is headquartered in San Francisco, California.",

        "Acme Corp's main product is CloudSync, a cloud storage solution "
        "launched in 2017. It currently has over 5 million users worldwide.",

        "The company went public in 2021 with an IPO price of $45 per share. "
        "Current revenue is approximately $250 million annually.",

        "Acme Corp employs 850 people across 12 offices globally. "
        "The engineering team consists of 300 software engineers.",

        "The company's mission is to make cloud storage accessible and secure "
        "for everyone. Core values include innovation, security, and simplicity.",

        "CloudSync offers three tiers: Free (5GB), Pro ($9.99/month, 100GB), "
        "and Enterprise (custom pricing, unlimited storage).",

        "Recent acquisitions include DataVault (2022) for $50 million and "
        "SecureFiles (2023) for $75 million to enhance security features.",

        "The company's tech stack includes Python, Go, and Kubernetes. "
        "All data is encrypted using AES-256 encryption.",
    ]

    section("Setting up RAG system...")

    # Initialize RAG
    rag = RAG(
        embedding_model=str(embedding_model),
        generation_model=str(generation_model),
    )

    # Add knowledge base
    rag.add_texts(company_docs)
    info(f"Added {len(company_docs)} documents to knowledge base")

    # Create RAG tool for the agent
    section("Creating RAG-powered tool...")

    knowledge_tool = create_rag_tool(
        rag,
        name="search_company_knowledge",
        description="Search the company knowledge base for information about "
                    "Acme Corp, its products, employees, financials, and history.",
        top_k=3,
        include_scores=True,
    )

    info(f"Created tool: {knowledge_tool.name}")

    # Initialize the LLM for the agent
    section("Initializing agent...")

    config = GenerationConfig(n_ctx=4096, n_batch=512)
    llm = LLM(str(generation_model), config=config)

    # Create agent with RAG tool and other utilities
    agent = ReActAgent(
        llm=llm,
        tools=[knowledge_tool, get_current_date, calculate],
        max_iterations=5,
        verbose=args.verbose,
    )

    info("Agent initialized with tools:")
    for t in agent.list_tools():
        bullet(f"{t.name}: {t.description[:60]}...")

    # Example queries
    section("Running agent queries...")

    queries = [
        "When was Acme Corp founded and who founded it?",
        "How many employees does Acme Corp have and what's the engineering team size?",
        "What is the Pro tier pricing for CloudSync?",
    ]

    for query in queries:
        print()
        bullet(f"Question: {query}")

        result = agent.run(query)

        if result.success:
            success(f"Answer: {result.answer}")
            kv("Iterations", str(result.iterations))
        else:
            error(f"Failed: {result.error}")

    # Demonstrate combining RAG with other tools
    section("Combining RAG with other tools...")

    print()
    query = "What is the company's annual revenue, and what is that amount divided by the number of employees?"
    bullet(f"Question: {query}")
    info("(This requires using RAG to get revenue and employee count, then calculate)")

    result = agent.run(query)
    if result.success:
        success(f"Answer: {result.answer}")
        kv("Iterations", str(result.iterations))
    else:
        error(f"Failed: {result.error}")

    # Cleanup
    section("Cleaning up...")
    llm.close()
    rag.close()

    success("RAG agent integration example completed!")
    return 0


if __name__ == "__main__":
    exit(main())
