"""
Code Assistant Agent Example

Demonstrates using a cyllama agent as a coding assistant.
Shows how to build an agent that can:
- Read and analyze code files
- Generate new code
- Refactor existing code
- Run tests and fix bugs

Usage:
    python agent_code_assistant.py [model_path]

    If no model_path is provided, will look for models in ./models/ directory.
"""

import argparse
from pathlib import Path
from cyllama import LLM, GenerationConfig
from cyllama.agents import ReActAgent, tool
from cyllama.utils.color import (
    header, section, subsection, subheader, success, error, info,
    bullet, numbered, kv, divider
)
import subprocess
import tempfile
import os


# Define code assistant tools

@tool
def read_file(filepath: str) -> str:
    """
    Read the contents of a file.

    Args:
        filepath: Path to the file

    Returns:
        File contents
    """
    try:
        # For demo, we'll use mock files
        mock_files = {
            "calculator.py": '''def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
''',
            "test_calculator.py": '''from calculator import add, subtract, multiply, divide

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

def test_subtract():
    assert subtract(5, 3) == 2

def test_multiply():
    assert multiply(3, 4) == 12

def test_divide():
    assert divide(10, 2) == 5
    try:
        divide(1, 0)
        assert False, "Should raise ValueError"
    except ValueError:
        pass
'''
        }

        if filepath in mock_files:
            return mock_files[filepath]
        else:
            return f"File '{filepath}' not found. Available: {list(mock_files.keys())}"

    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def write_file(filepath: str, content: str) -> str:
    """
    Write content to a file.

    Args:
        filepath: Path where to write
        content: Content to write

    Returns:
        Success message
    """
    try:
        # For demo, we just confirm the write
        lines = content.count('\n') + 1
        chars = len(content)
        return f"Successfully wrote {lines} lines ({chars} characters) to {filepath}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


@tool
def run_python_code(code: str) -> str:
    """
    Execute Python code and return output.

    Args:
        code: Python code to execute

    Returns:
        Output from execution or error message
    """
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Run the code
            result = subprocess.run(
                ['python3', temp_file],
                capture_output=True,
                text=True,
                timeout=5
            )

            output = result.stdout if result.stdout else result.stderr
            return output if output else "Code executed successfully (no output)"

        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out (>5 seconds)"
    except Exception as e:
        return f"Error running code: {str(e)}"


@tool
def run_tests(test_file: str) -> str:
    """
    Run pytest tests from a file.

    Args:
        test_file: Path to test file

    Returns:
        Test results
    """
    # For demo, simulate test results
    mock_results = {
        "test_calculator.py": """
test_calculator.py::test_add PASSED
test_calculator.py::test_subtract PASSED
test_calculator.py::test_multiply PASSED
test_calculator.py::test_divide PASSED

==== 4 passed in 0.05s ====
""",
        "test_broken.py": """
test_broken.py::test_failing FAILED

FAILED test_broken.py::test_failing - AssertionError: expected 5, got 4

==== 1 failed in 0.03s ====
"""
    }

    return mock_results.get(test_file, f"No tests found in {test_file}")


@tool
def lint_code(code: str) -> str:
    """
    Check code for style issues.

    Args:
        code: Python code to lint

    Returns:
        Lint warnings/errors
    """
    issues = []

    # Simple lint checks
    lines = code.split('\n')

    for i, line in enumerate(lines):
        line_num = i + 1  # Human-readable line number

        # Check line length
        if len(line) > 100:
            issues.append(f"Line {line_num}: Line too long ({len(line)} > 100 characters)")

        # Check for missing docstrings in functions
        if line.strip().startswith('def ') and ':' in line:
            # Check if next line starts a docstring
            next_idx = i + 1
            if next_idx >= len(lines) or not lines[next_idx].strip().startswith(('"""', "'''")):
                func_name = line.split('def ')[1].split('(')[0]
                issues.append(f"Line {line_num}: Function '{func_name}' missing docstring")

    if not issues:
        return "No linting issues found. Code looks good!"
    else:
        return "\n".join(issues)


@tool
def generate_function(name: str, description: str, parameters: str) -> str:
    """
    Generate a Python function template.

    Args:
        name: Function name
        description: What the function should do
        parameters: Parameter list (e.g., "x: int, y: int")

    Returns:
        Generated function code
    """
    return f'''def {name}({parameters}):
    """
    {description}

    Args:
        {parameters.replace(',', '\\n        ')}

    Returns:
        TODO: Add return type description
    """
    # TODO: Implement {name}
    pass
'''


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


def example_code_reading(llm: LLM):
    """Demonstrate reading and analyzing code."""
    section("CODE READING AND ANALYSIS EXAMPLE")

    agent = ReActAgent(
        llm=llm,
        tools=[read_file, lint_code],
        max_iterations=5,
        verbose=True
    )

    subsection("Task: Read and analyze calculator.py", color='yellow')

    result = agent.run(
        "Read calculator.py and check it for code style issues using the linter"
    )

    subsection("RESULT", color='bright_green')
    kv("Answer", result.answer)


def example_code_generation(llm: LLM):
    """Demonstrate generating new code."""
    section("CODE GENERATION EXAMPLE")

    agent = ReActAgent(
        llm=llm,
        tools=[generate_function, write_file],
        max_iterations=5,
        verbose=True
    )

    subsection("Task: Generate a power function", color='yellow')

    result = agent.run(
        "Generate a Python function called 'power' that raises a number to an exponent, "
        "with parameters base: float and exponent: int"
    )

    subsection("RESULT", color='bright_green')
    kv("Answer", result.answer)


def example_test_running(llm: LLM):
    """Demonstrate running tests."""
    section("TEST EXECUTION EXAMPLE")

    agent = ReActAgent(
        llm=llm,
        tools=[run_tests, read_file],
        max_iterations=5,
        verbose=True
    )

    subsection("Task: Run tests for calculator", color='yellow')

    result = agent.run(
        "Run the tests in test_calculator.py and tell me if they all pass"
    )

    subsection("RESULT", color='bright_green')
    kv("Answer", result.answer)


def example_code_execution(llm: LLM):
    """Demonstrate executing code."""
    section("CODE EXECUTION EXAMPLE")

    agent = ReActAgent(
        llm=llm,
        tools=[run_python_code],
        max_iterations=5,
        verbose=True
    )

    subsection("Task: Execute a simple Python script", color='yellow')

    result = agent.run(
        "Write and execute Python code that prints the first 10 Fibonacci numbers. "
        "Use the run_python_code tool with the 'code' parameter containing the Python code."
    )

    subsection("RESULT", color='bright_green')
    kv("Answer", result.answer)


def show_use_cases():
    """Show practical use cases."""
    section("PRACTICAL USE CASES")

    subheader("1. CODE REVIEW ASSISTANT", color='cyan')
    bullet("Agent reads code, runs linter, suggests improvements")

    print()
    subheader("2. TEST GENERATOR", color='cyan')
    bullet("Agent reads function, generates comprehensive test cases")

    print()
    subheader("3. BUG FIXER", color='cyan')
    bullet("Agent reads code, runs tests, identifies failures, suggests fixes")

    print()
    subheader("4. DOCUMENTATION WRITER", color='cyan')
    bullet("Agent reads code, generates docstrings and README")

    print()
    subheader("5. REFACTORING ASSISTANT", color='cyan')
    bullet("Agent analyzes code structure, suggests refactorings")

    print()
    subheader("6. CODE EXPLAINER", color='cyan')
    bullet("Agent reads complex code, explains what it does")


def main():
    """Run all code assistant examples."""
    parser = argparse.ArgumentParser(
        description="Code Assistant Agent Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python agent_code_assistant.py
    python agent_code_assistant.py /path/to/model.gguf
    python agent_code_assistant.py models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
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
            info("Usage: python agent_code_assistant.py /path/to/model.gguf")
            return 1

    header("CODE ASSISTANT AGENT EXAMPLES")

    print("\nThis example demonstrates:")
    numbered([
        "Reading and analyzing code files",
        "Generating new code from descriptions",
        "Running tests and interpreting results",
        "Executing code and checking output",
        "Code quality checking (linting)"
    ])

    # Show use cases
    show_use_cases()

    # Initialize LLM once and share across examples
    info(f"Loading model: {model_path.name}...")
    config = GenerationConfig(n_batch=4096, n_ctx=8192)
    llm = LLM(str(model_path), config=config, verbose=args.verbose)
    success("Model loaded successfully.")

    # Run examples
    example_code_reading(llm)
    example_code_generation(llm)
    example_test_running(llm)
    example_code_execution(llm)

    return 0


if __name__ == "__main__":
    exit(main())
