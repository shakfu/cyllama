"""
Research Assistant Agent Example

Demonstrates using a cyllama agent for multi-step research tasks.
Shows how to build an agent that can:
- Search for information
- Synthesize findings
- Take notes
- Generate reports

Usage:
    python agent_researcher.py [model_path]

    If no model_path is provided, will look for models in ./models/ directory.
"""

import argparse
from pathlib import Path
from cyllama import LLM, GenerationConfig
from cyllama.agents import ReActAgent, ConstrainedAgent, tool
from cyllama.utils.color import (
    header, section, subheader, success, error, info,
    bullet, numbered, kv, divider
)
from datetime import datetime


# Mock knowledge base
KNOWLEDGE_BASE = {
    "machine_learning": {
        "supervised": "Learning from labeled data",
        "unsupervised": "Finding patterns in unlabeled data",
        "reinforcement": "Learning through trial and error with rewards"
    },
    "neural_networks": {
        "CNN": "Convolutional Neural Networks - good for images",
        "RNN": "Recurrent Neural Networks - good for sequences",
        "Transformer": "Attention-based architecture - state of the art"
    },
    "python": {
        "history": "Created by Guido van Rossum in 1991",
        "philosophy": "Readability counts - The Zen of Python",
        "use_cases": "Web dev, data science, automation, AI/ML"
    }
}

# Research notes storage
research_notes = []


# Define research tools

@tool
def search_topic(query: str) -> str:
    """
    Search for information about a topic.

    Args:
        query: Search query

    Returns:
        Search results
    """
    # Normalize query: lowercase, replace spaces/underscores, split into words
    query_lower = query.lower().replace("_", " ")
    query_words = set(query_lower.split())

    # Search through knowledge base with fuzzy matching
    results = []
    for category, items in KNOWLEDGE_BASE.items():
        category_normalized = category.lower().replace("_", " ")
        category_words = set(category_normalized.split())

        # Match if any query word matches category or category word
        category_match = (
            query_lower in category_normalized or
            category_normalized in query_lower or
            bool(query_words & category_words)  # Word intersection
        )

        if category_match:
            results.append(f"Category: {category}")
            for key, value in items.items():
                results.append(f"  - {key}: {value}")

        # Also check individual items
        for key, value in items.items():
            key_lower = key.lower()
            value_lower = value.lower()

            # Match if query word appears in key or value
            item_match = (
                any(word in key_lower or word in value_lower for word in query_words) or
                query_lower in key_lower or
                query_lower in value_lower
            )

            if item_match and f"Category: {category}" not in results:
                results.append(f"Category: {category}")
                results.append(f"  - {key}: {value}")
            elif item_match:
                # Avoid duplicates
                item_line = f"  - {key}: {value}"
                if item_line not in results:
                    results.append(item_line)

    if results:
        return "\n".join(results)
    else:
        return f"No results found for '{query}'. Try broader terms like: machine_learning, neural_networks, python"


@tool
def search_specific(category: str, subtopic: str) -> str:
    """
    Search for specific information within a category.

    Args:
        category: Main category to search in
        subtopic: Specific subtopic

    Returns:
        Detailed information
    """
    if category in KNOWLEDGE_BASE:
        if subtopic in KNOWLEDGE_BASE[category]:
            return f"{subtopic}: {KNOWLEDGE_BASE[category][subtopic]}"
        else:
            available = ", ".join(KNOWLEDGE_BASE[category].keys())
            return f"Subtopic '{subtopic}' not found in {category}. Available: {available}"
    else:
        available = ", ".join(KNOWLEDGE_BASE.keys())
        return f"Category '{category}' not found. Available: {available}"


@tool
def take_note(note: str) -> str:
    """
    Take a research note.

    Args:
        note: Note content

    Returns:
        Confirmation message
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    research_notes.append({"time": timestamp, "note": note})
    return f"Note saved (total notes: {len(research_notes)})"


@tool
def list_notes() -> str:
    """
    List all research notes.

    Returns:
        All notes formatted
    """
    if not research_notes:
        return "No notes yet"

    formatted = []
    for i, note in enumerate(research_notes, 1):
        formatted.append(f"{i}. [{note['time']}] {note['note']}")

    return "\n".join(formatted)


@tool
def summarize_notes() -> str:
    """
    Get a summary of all notes.

    Returns:
        Summary of research notes
    """
    if not research_notes:
        return "No notes to summarize"

    summary = f"Research Summary ({len(research_notes)} notes):\n\n"
    for note in research_notes:
        summary += f"- {note['note']}\n"

    return summary


@tool
def compare_topics(topic1: str, topic2: str) -> str:
    """
    Compare two topics.

    Args:
        topic1: First topic
        topic2: Second topic

    Returns:
        Comparison results
    """
    result1 = search_topic(topic1)
    result2 = search_topic(topic2)

    return f"Comparison of '{topic1}' vs '{topic2}':\n\n{topic1}:\n{result1}\n\n{topic2}:\n{result2}"


@tool
def generate_bibliography(topics: str) -> str:
    """
    Generate a bibliography for research topics.

    Args:
        topics: Comma-separated list of topics

    Returns:
        Mock bibliography
    """
    topic_list = [t.strip() for t in topics.split(',')]

    bib = "Bibliography:\n\n"
    for i, topic in enumerate(topic_list, 1):
        bib += f"{i}. '{topic}' - Internal Knowledge Base, {datetime.now().year}\n"

    return bib


def find_model() -> Path:
    """Find a model in the default locations."""
    ROOT = Path.cwd()

    # Preferred models in order
    candidates = [
        ROOT / 'models' / 'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf',
        ROOT / 'models' / 'Llama-3.2-1B-Instruct-Q8_0.gguf',
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def example_basic_research(llm: LLM):
    """Demonstrate basic research workflow."""
    section("BASIC RESEARCH EXAMPLE")

    agent = ReActAgent(
        llm=llm,
        tools=[search_topic, take_note],
        max_iterations=8,
        verbose=True
    )

    section("Task: Research machine learning and take notes", color='yellow')

    global research_notes
    research_notes = []  # Reset notes

    result = agent.run(
        "Search for information about machine learning and take notes on the main types"
    )

    section("RESULT", color='bright_green')
    kv("Success", str(result.success), value_color='green' if result.success else 'red')
    kv("Answer", result.answer)


def example_multi_step_research(llm: LLM):
    """Demonstrate complex multi-step research."""
    section("MULTI-STEP RESEARCH EXAMPLE")

    # Use ConstrainedAgent for reliability
    agent = ConstrainedAgent(
        llm=llm,
        tools=[
            search_topic,
            search_specific,
            take_note,
            list_notes,
            compare_topics
        ],
        max_iterations=15,
        verbose=True,
        format="json"
    )

    section("Task: Compare supervised vs unsupervised learning", color='yellow')

    global research_notes
    research_notes = []  # Reset notes

    result = agent.run(
        "Research and compare supervised learning vs unsupervised learning. "
        "Take notes on key differences."
    )

    section("RESULT", color='bright_green')
    kv("Answer", result.answer)


def example_research_report(llm: LLM):
    """Demonstrate generating a research report."""
    section("RESEARCH REPORT GENERATION EXAMPLE")

    agent = ReActAgent(
        llm=llm,
        tools=[
            search_topic,
            take_note,
            list_notes,
            summarize_notes,
            generate_bibliography
        ],
        max_iterations=12,
        verbose=True
    )

    section("Task: Create a report on neural networks", color='yellow')

    global research_notes
    research_notes = []  # Reset notes

    result = agent.run(
        "Research neural networks, take notes on different types (CNN, RNN, Transformer), "
        "and then summarize your findings"
    )

    section("RESULT", color='bright_green')
    kv("Answer", result.answer)


def show_research_workflow():
    """Show typical research workflow."""
    section("TYPICAL RESEARCH WORKFLOW")

    subheader("1. EXPLORATION PHASE", color='cyan')
    bullet("Broad search for general information")
    bullet("Take initial notes on interesting findings")
    bullet("Identify key subtopics")

    print()
    subheader("2. DEEP DIVE PHASE", color='cyan')
    bullet("Search specific subtopics")
    bullet("Compare different approaches")
    bullet("Take detailed notes")

    print()
    subheader("3. SYNTHESIS PHASE", color='cyan')
    bullet("Review all notes")
    bullet("Generate summary")
    bullet("Create bibliography")

    print()
    subheader("4. REPORTING PHASE", color='cyan')
    bullet("Organize findings")
    bullet("Generate final report")
    bullet("Include citations")


def main():
    """Run all research assistant examples."""
    parser = argparse.ArgumentParser(
        description="Research Assistant Agent Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python agent_researcher.py
    python agent_researcher.py /path/to/model.gguf
    python agent_researcher.py models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
        """
    )
    parser.add_argument(
        'model_path',
        nargs='?',
        type=str,
        help='Path to GGUF model file (optional, will auto-detect if not provided)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose model output'
    )

    args = parser.parse_args()

    # Resolve model path
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            error(f"Model not found: {model_path}")
            return 1
    else:
        model_path = find_model()
        if model_path is None:
            error("No model found. Please provide a model path.")
            info("Usage: python agent_researcher.py /path/to/model.gguf")
            return 1

    header("RESEARCH ASSISTANT AGENT EXAMPLES")

    print("\nThis example demonstrates:")
    numbered([
        "Information search and retrieval",
        "Note-taking during research",
        "Multi-step research workflows",
        "Comparing topics and concepts",
        "Synthesizing findings",
        "Generating research reports"
    ])

    # Show workflow
    show_research_workflow()

    # Initialize LLM once and share across examples
    info(f"Loading model: {model_path.name}...")
    config = GenerationConfig(n_batch=4096, n_ctx=8192)
    llm = LLM(str(model_path), config=config, verbose=args.verbose)
    success("Model loaded successfully.")

    # Run examples
    example_basic_research(llm)
    example_multi_step_research(llm)
    example_research_report(llm)

    section("EXAMPLES COMPLETE", color='bright_green')

    subheader("Key Takeaways")
    bullet("Agents can manage complex research workflows")
    bullet("Natural language queries for research tasks")
    bullet("Automatic note-taking and organization")
    bullet("Can synthesize information from multiple sources")

    print()
    subheader("Production Extensions")
    bullet("Add real web search (Google, Bing APIs)")
    bullet("Add document retrieval (PDF, arXiv, PubMed)")
    bullet("Add citation management")
    bullet("Add fact-checking tools")
    bullet("Add knowledge graph integration")
    bullet("Export to markdown, LaTeX, or Word")

    return 0


if __name__ == "__main__":
    exit(main())
