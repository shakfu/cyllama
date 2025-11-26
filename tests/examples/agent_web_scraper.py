"""
Web Scraping Agent Example

Demonstrates using a cyllama agent to extract information from web pages.
This example shows how to build a practical agent that can:
- Fetch web pages
- Extract specific information
- Parse and structure data
- Handle errors gracefully

Usage:
    python agent_web_scraper.py [model_path]

    If no model_path is provided, will look for models in ./models/ directory.
"""

import argparse
from pathlib import Path
from cyllama import LLM, GenerationConfig
from cyllama.agents import ReActAgent, tool
from cyllama.utils.color import (
    header, section, subsection, subheader, success, error, info,
    bullet, numbered, kv
)
import re


# Define web scraping tools

@tool
def fetch_url(url: str) -> str:
    """
    Fetch the content of a web page.

    Args:
        url: URL to fetch

    Returns:
        Page content (first 2000 characters for demo)
    """
    try:
        # In production, use requests library
        # For this demo, we simulate fetching
        mock_content = {
            "https://example.com": """
            <html>
            <head><title>Example Domain</title></head>
            <body>
                <h1>Example Domain</h1>
                <p>This domain is for use in illustrative examples in documents.</p>
                <p>Contact: info@example.com</p>
                <p>Phone: +1-555-0123</p>
            </body>
            </html>
            """,
            "https://news.example.com": """
            <html>
            <head><title>Tech News</title></head>
            <body>
                <h1>Latest Tech News</h1>
                <article>
                    <h2>AI Advances in 2024</h2>
                    <p>Published: 2024-01-15</p>
                    <p>Artificial intelligence continues to advance...</p>
                </article>
                <article>
                    <h2>Quantum Computing Breakthrough</h2>
                    <p>Published: 2024-01-10</p>
                    <p>Scientists achieve new quantum milestone...</p>
                </article>
            </body>
            </html>
            """
        }

        content = mock_content.get(url, f"<html><body>Page not found: {url}</body></html>")
        return content[:2000]  # Limit for demo

    except Exception as e:
        return f"Error fetching {url}: {str(e)}"


@tool
def extract_emails(text: str) -> list:
    """
    Extract email addresses from text.

    Args:
        text: Text to search

    Returns:
        List of email addresses found
    """
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    return list(set(emails))  # Remove duplicates


@tool
def extract_phone_numbers(text: str) -> list:
    """
    Extract phone numbers from text.

    Args:
        text: Text to search

    Returns:
        List of phone numbers found
    """
    # Simple pattern for demo
    phone_pattern = r'\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}'
    phones = re.findall(phone_pattern, text)
    return list(set(phones))


@tool
def extract_between_tags(text: str, tag: str) -> list:
    """
    Extract content between HTML tags.

    Args:
        text: HTML text
        tag: Tag name (e.g., 'h1', 'p', 'title')

    Returns:
        List of content found between tags
    """
    pattern = f'<{tag}[^>]*>(.*?)</{tag}>'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    return [m.strip() for m in matches]


@tool
def count_occurrences(text: str, search_term: str) -> int:
    """
    Count occurrences of a term in text.

    Args:
        text: Text to search
        search_term: Term to count

    Returns:
        Number of occurrences
    """
    return text.lower().count(search_term.lower())


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


def example_basic_scraping(llm: LLM):
    """Demonstrate basic web scraping with agent."""
    section("BASIC WEB SCRAPING EXAMPLE")

    agent = ReActAgent(
        llm=llm,
        tools=[fetch_url, extract_emails, extract_phone_numbers, extract_between_tags],
        max_iterations=10,
        verbose=True
    )

    subsection("Task: Extract contact information from example.com", color='yellow')

    result = agent.run(
        "Fetch https://example.com and extract all email addresses and phone numbers"
    )

    subsection("RESULT", color='bright_green')
    kv("Success", str(result.success), value_color='green' if result.success else 'red')
    kv("Answer", result.answer)
    kv("Steps taken", str(result.iterations))


def example_structured_extraction(llm: LLM):
    """Demonstrate structured data extraction."""
    section("STRUCTURED DATA EXTRACTION EXAMPLE")

    agent = ReActAgent(
        llm=llm,
        tools=[fetch_url, extract_between_tags, count_occurrences],
        max_iterations=10,
        verbose=True
    )

    subsection("Task: Extract article titles from news page", color='yellow')

    result = agent.run(
        "Fetch https://news.example.com and extract all article titles (h2 tags)"
    )

    subsection("RESULT", color='bright_green')
    kv("Answer", result.answer)


def example_without_agent():
    """Show how complex it is without an agent."""
    section("COMPARISON: WITHOUT AGENT")

    subheader("Manual approach (what you'd write without agents)", color='cyan')
    print("""
    # Fetch page
    content = fetch_url("https://example.com")

    # Try to extract emails
    emails = extract_emails(content)

    # Try to extract phones
    phones = extract_phone_numbers(content)

    # Format result
    result = f"Found emails: {emails}, phones: {phones}"
    """)

    subheader("With agent approach", color='cyan')
    print("""
    agent = ReActAgent(llm=llm, tools=[fetch_url, extract_emails, extract_phone_numbers])
    result = agent.run("Extract contact info from https://example.com")

    # Agent decides:
    # 1. First fetch the URL
    # 2. Then extract emails from content
    # 3. Then extract phones from content
    # 4. Synthesize final answer
    """)

    subheader("Agent benefits", color='green')
    bullet("Automatic tool sequencing")
    bullet("Natural language interface")
    bullet("Error handling and retries")
    bullet("Context-aware decision making")


def main():
    """Run all web scraping examples."""
    parser = argparse.ArgumentParser(
        description="Web Scraping Agent Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python agent_web_scraper.py
    python agent_web_scraper.py /path/to/model.gguf
    python agent_web_scraper.py models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
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
            info("Usage: python agent_web_scraper.py /path/to/model.gguf")
            return 1

    header("WEB SCRAPING AGENT EXAMPLES")

    print("\nThis example demonstrates:")
    numbered([
        "Fetching web pages",
        "Extracting specific information",
        "Pattern matching and parsing",
        "Multi-step information gathering"
    ])

    # Show comparison first
    example_without_agent()

    # Initialize LLM once and share across examples
    info(f"Loading model: {model_path.name}...")
    config = GenerationConfig(n_batch=4096, n_ctx=8192)
    llm = LLM(str(model_path), config=config, verbose=args.verbose)
    success("Model loaded successfully.")

    # Then show agent in action
    example_basic_scraping(llm)
    example_structured_extraction(llm)

    return 0


if __name__ == "__main__":
    exit(main())
