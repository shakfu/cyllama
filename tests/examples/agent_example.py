"""
Example demonstrating the ReAct agent functionality.

This example shows how to create a simple agent with custom tools.

Usage:
    python agent_example.py [model_path]

    If no model_path is provided, will look for models in ./models/ directory.
"""

import argparse
from cyllama import LLM, GenerationConfig
from cyllama.agents import ReActAgent, tool
from cyllama.utils.color import (
    header, section, success, error, info, bullet, kv
)
from pathlib import Path

# Define custom tools


@tool
def calculator(expression: str) -> float:
    """
    Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2", "5 * 3")

    Returns:
        The result of the calculation
    """
    try:
        # Use eval safely for simple math expressions
        # In production, use a proper math parser
        result = eval(expression, {"__builtins__": {}}, {})
        return float(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_file_size(filepath: str) -> str:
    """
    Get the size of a file in bytes.

    Args:
        filepath: Path to the file

    Returns:
        File size in bytes
    """
    try:
        path = Path(filepath)
        if path.exists():
            size = path.stat().st_size
            return f"{size} bytes"
        else:
            return f"File not found: {filepath}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def string_reverse(text: str) -> str:
    """
    Reverse a string.

    Args:
        text: String to reverse

    Returns:
        Reversed string
    """
    return text[::-1]


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


def main():
    """Run the agent example."""
    parser = argparse.ArgumentParser(
        description="ReAct Agent Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python agent_example.py
    python agent_example.py /path/to/model.gguf
    python agent_example.py models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
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
            info("Usage: python agent_example.py /path/to/model.gguf")
            return 1

    info(f"Initializing LLM with {model_path.name}...")
    config = GenerationConfig(n_batch=4096, n_ctx=8192)
    llm = LLM(str(model_path), config=config, verbose=args.verbose)

    info("Creating agent with tools...")
    agent = ReActAgent(
        llm=llm,
        tools=[calculator, get_file_size, string_reverse],
        max_iterations=5,
        verbose=True  # Show agent reasoning
    )

    section("Agent initialized with tools:")
    for t in agent.list_tools():
        bullet(f"{t.name}: {t.description}")

    # Example tasks
    tasks = [
        "What is 15 multiplied by 23?",
        "Reverse the string 'hello world'",
        "What is the result of (100 + 50) / 3?",
    ]

    for i, task in enumerate(tasks, 1):
        section(f"Task {i}: {task}", color='yellow')

        result = agent.run(task)

        if result.success:
            success(f"Answer: {result.answer}")
            kv("Iterations", str(result.iterations))
        else:
            error(f"Failed: {result.error}")

    return 0


if __name__ == "__main__":
    exit(main())
