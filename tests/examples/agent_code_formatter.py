"""
Code Formatting and Syntax Highlighting Agent Example

Demonstrates using a cyllama agent to format and highlight code.
Shows how to build an agent that can:
- Syntax highlight code in various languages
- Format code with proper indentation
- Convert code to different output formats (terminal, HTML, markdown)
- Pretty-print data structures

Usage:
    python agent_code_formatter.py [model_path]

    If no model_path is provided, will look for models in ./models/ directory.
"""

import argparse
from pathlib import Path
from cyllama import LLM, GenerationConfig
from cyllama.agents import ReActAgent, tool
from cyllama.utils.color import (
    header, section, subheader, success, error, info,
    bullet, numbered, kv
)
import json
import re


# ANSI color codes for terminal output
class Colors:
    """ANSI escape codes for colored terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'

    # Foreground colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'


# Define syntax highlighting tools

@tool
def highlight_python(code: str) -> str:
    """
    Apply syntax highlighting to Python code for terminal output.

    Args:
        code: Python code to highlight

    Returns:
        Code with ANSI color codes
    """
    # Keywords
    keywords = ['def', 'class', 'if', 'else', 'elif', 'for', 'while', 'return',
                'import', 'from', 'as', 'try', 'except', 'finally', 'with',
                'lambda', 'yield', 'raise', 'assert', 'pass', 'break', 'continue',
                'and', 'or', 'not', 'in', 'is', 'None', 'True', 'False']

    # Built-in functions
    builtins = ['print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict',
                'set', 'tuple', 'open', 'enumerate', 'zip', 'map', 'filter']

    lines = code.split('\n')
    highlighted = []

    for line in lines:
        # Highlight comments
        if '#' in line:
            comment_pos = line.find('#')
            before = line[:comment_pos]
            comment = line[comment_pos:]
            line = before + Colors.BRIGHT_BLACK + comment + Colors.RESET

        # Highlight strings
        # Simple approach - handle single and double quotes
        line = re.sub(r'(".*?")', Colors.YELLOW + r'\1' + Colors.RESET, line)
        line = re.sub(r"('.*?')", Colors.YELLOW + r'\1' + Colors.RESET, line)

        # Highlight keywords
        for keyword in keywords:
            pattern = r'\b' + keyword + r'\b'
            line = re.sub(pattern, Colors.MAGENTA + keyword + Colors.RESET, line)

        # Highlight built-ins
        for builtin in builtins:
            pattern = r'\b' + builtin + r'\('
            line = re.sub(pattern, Colors.CYAN + builtin + Colors.RESET + '(', line)

        # Highlight function definitions
        line = re.sub(r'\bdef\b\s+(\w+)',
                     Colors.MAGENTA + 'def' + Colors.RESET + ' ' + Colors.BLUE + r'\1' + Colors.RESET,
                     line)

        # Highlight numbers
        line = re.sub(r'\b(\d+\.?\d*)\b', Colors.GREEN + r'\1' + Colors.RESET, line)

        highlighted.append(line)

    return '\n'.join(highlighted)


@tool
def highlight_json(data: str) -> str:
    """
    Pretty-print and highlight JSON data.

    Args:
        data: JSON string to highlight

    Returns:
        Formatted and highlighted JSON
    """
    try:
        # Parse and re-format with indentation
        parsed = json.loads(data)
        formatted = json.dumps(parsed, indent=2)

        lines = formatted.split('\n')
        highlighted = []

        for line in lines:
            # Simple approach: apply highlighting in order, using lambda to avoid regex conflicts
            result = line

            # Highlight booleans and null first
            result = re.sub(r'\b(true|false|null)\b',
                           lambda m: Colors.MAGENTA + m.group(1) + Colors.RESET,
                           result)

            # Highlight numbers (avoid already colored text by checking for ANSI codes nearby)
            result = re.sub(r':\s*(\d+\.?\d*)(?=\s*[,}\]])',
                           lambda m: ': ' + Colors.GREEN + m.group(1) + Colors.RESET,
                           result)

            # Highlight string values before keys
            result = re.sub(r':\s*("(?:[^"\\]|\\.)*")',
                           lambda m: ': ' + Colors.YELLOW + m.group(1) + Colors.RESET,
                           result)

            # Highlight keys (strings followed by colon)
            result = re.sub(r'("(?:[^"\\]|\\.)*")\s*:',
                           lambda m: Colors.CYAN + m.group(1) + Colors.RESET + ':',
                           result)

            # Highlight brackets (but not the [ in ANSI codes like \033[)
            result = re.sub(r'([{}])', Colors.BRIGHT_BLACK + r'\1' + Colors.RESET, result)
            # For brackets, only highlight if not preceded by escape sequence
            result = re.sub(r'(?<!\033)([\[\]])', Colors.BRIGHT_BLACK + r'\1' + Colors.RESET, result)

            highlighted.append(result)

        return '\n'.join(highlighted)

    except json.JSONDecodeError as e:
        return f"Error parsing JSON: {e}"


@tool
def format_code_block(code: str, language: str) -> str:
    """
    Format code as a markdown code block.

    Args:
        code: Code to format
        language: Language identifier (python, javascript, etc.)

    Returns:
        Markdown-formatted code block
    """
    return f"```{language}\n{code}\n```"


@tool
def create_html_highlighted(code: str, language: str) -> str:
    """
    Create HTML with syntax highlighting.

    Args:
        code: Code to highlight
        language: Programming language

    Returns:
        HTML with inline styles for highlighting
    """
    if language.lower() == 'python':
        # Simple HTML highlighting for Python
        html = '<pre style="background: #2e3440; color: #d8dee9; padding: 1em; border-radius: 5px;">\n'

        lines = code.split('\n')
        for line in lines:
            # Escape HTML
            line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

            # Keywords
            line = re.sub(r'\b(def|class|if|else|return|import|for|while)\b',
                         r'<span style="color: #81a1c1;">\1</span>', line)

            # Strings
            line = re.sub(r'(".*?"|\'.*?\')',
                         r'<span style="color: #a3be8c;">\1</span>', line)

            # Comments
            line = re.sub(r'(#.*$)',
                         r'<span style="color: #616e88;">\1</span>', line)

            # Numbers
            line = re.sub(r'\b(\d+\.?\d*)\b',
                         r'<span style="color: #b48ead;">\1</span>', line)

            html += line + '\n'

        html += '</pre>'
        return html

    else:
        return f'<pre><code class="{language}">{code}</code></pre>'


@tool
def strip_colors(text: str) -> str:
    """
    Remove ANSI color codes from text.

    Args:
        text: Text with ANSI codes

    Returns:
        Plain text without colors
    """
    # Remove ANSI escape sequences
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


@tool
def add_line_numbers(code: str, start: int = 1) -> str:
    """
    Add line numbers to code.

    Args:
        code: Code to number
        start: Starting line number

    Returns:
        Code with line numbers
    """
    lines = code.split('\n')
    width = len(str(start + len(lines)))

    numbered = []
    for i, line in enumerate(lines, start):
        numbered.append(f"{i:>{width}} | {line}")

    return '\n'.join(numbered)


@tool
def create_diff(old_code: str, new_code: str) -> str:
    """
    Create a colored diff between two code snippets.

    Args:
        old_code: Original code
        new_code: Modified code

    Returns:
        Colored diff output
    """
    old_lines = old_code.split('\n')
    new_lines = new_code.split('\n')

    diff = []
    diff.append(Colors.BRIGHT_BLACK + "--- old" + Colors.RESET)
    diff.append(Colors.BRIGHT_BLACK + "+++ new" + Colors.RESET)

    # Simple line-by-line diff
    max_len = max(len(old_lines), len(new_lines))

    for i in range(max_len):
        old_line = old_lines[i] if i < len(old_lines) else ""
        new_line = new_lines[i] if i < len(new_lines) else ""

        if old_line == new_line:
            diff.append(f"  {old_line}")
        else:
            if old_line:
                diff.append(Colors.RED + f"- {old_line}" + Colors.RESET)
            if new_line:
                diff.append(Colors.GREEN + f"+ {new_line}" + Colors.RESET)

    return '\n'.join(diff)


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


def example_python_highlighting():
    """Demonstrate Python syntax highlighting."""
    section("PYTHON SYNTAX HIGHLIGHTING")

    sample_code = '''def fibonacci(n):
    """Calculate Fibonacci number."""
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# Test the function
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")
'''

    print("\nOriginal code:")
    print(sample_code)

    print("\nHighlighted code:")
    highlighted = highlight_python(sample_code)
    print(highlighted)


def example_json_highlighting():
    """Demonstrate JSON highlighting."""
    section("JSON SYNTAX HIGHLIGHTING")

    sample_json = '{"name": "John Doe", "age": 30, "active": true, "skills": ["Python", "JavaScript"], "score": 95.5}'

    print("\nOriginal JSON:")
    print(sample_json)

    print("\nFormatted and highlighted JSON:")
    highlighted = highlight_json(sample_json)
    print(highlighted)


def example_code_with_line_numbers():
    """Demonstrate adding line numbers."""
    section("CODE WITH LINE NUMBERS")

    code = '''def greet(name):
    return f"Hello, {name}!"

print(greet("World"))'''

    # First highlight, then add line numbers
    highlighted = highlight_python(code)
    numbered = add_line_numbers(highlighted)

    print("\nCode with line numbers and syntax highlighting:")
    print(numbered)


def example_diff_display():
    """Demonstrate diff highlighting."""
    section("CODE DIFF WITH HIGHLIGHTING")

    old_code = '''def calculate(x, y):
    result = x + y
    return result'''

    new_code = '''def calculate(x, y):
    """Add two numbers."""
    result = x + y
    print(f"Result: {result}")
    return result'''

    print("\nDiff:")
    diff = create_diff(old_code, new_code)
    print(diff)


def example_markdown_export():
    """Demonstrate markdown code block creation."""
    section("MARKDOWN CODE BLOCK EXPORT")

    code = '''def hello():
    print("Hello, World!")'''

    markdown = format_code_block(code, "python")

    print("\nMarkdown output:")
    print(markdown)


def example_html_export():
    """Demonstrate HTML highlighting."""
    section("HTML SYNTAX HIGHLIGHTING")

    code = '''def add(a, b):
    # Add two numbers
    return a + b

result = add(5, 3)
print(result)'''

    html = create_html_highlighted(code, "python")

    print("\nHTML output (first 200 chars):")
    print(html[:200] + "...")

    print("\n(View in browser to see highlighting)")


def example_with_agent(llm: LLM):
    """Demonstrate using agent for code highlighting."""
    section("AGENT-BASED CODE HIGHLIGHTING")

    agent = ReActAgent(
        llm=llm,
        tools=[
            highlight_python,
            highlight_json,
            add_line_numbers,
            format_code_block,
            create_diff
        ],
        max_iterations=5,
        verbose=True
    )

    section("Task: Highlight Python code and add line numbers", color='yellow')

    result = agent.run(
        "Highlight this Python code and add line numbers: def test(): return 42"
    )

    section("RESULT", color='bright_green')
    kv("Answer", result.answer)


def show_color_reference():
    """Display available colors."""
    section("ANSI COLOR REFERENCE")

    colors = [
        ("BLACK", Colors.BLACK),
        ("RED", Colors.RED),
        ("GREEN", Colors.GREEN),
        ("YELLOW", Colors.YELLOW),
        ("BLUE", Colors.BLUE),
        ("MAGENTA", Colors.MAGENTA),
        ("CYAN", Colors.CYAN),
        ("WHITE", Colors.WHITE),
        ("BRIGHT_RED", Colors.BRIGHT_RED),
        ("BRIGHT_GREEN", Colors.BRIGHT_GREEN),
        ("BRIGHT_YELLOW", Colors.BRIGHT_YELLOW),
        ("BRIGHT_BLUE", Colors.BRIGHT_BLUE),
        ("BRIGHT_MAGENTA", Colors.BRIGHT_MAGENTA),
        ("BRIGHT_CYAN", Colors.BRIGHT_CYAN),
    ]

    print("\nAvailable colors for syntax highlighting:")
    for name, code in colors:
        print(f"{code}{name:20}{Colors.RESET} - Sample text in {name}")


def main():
    """Run all code formatting examples."""
    parser = argparse.ArgumentParser(
        description="Code Formatting and Syntax Highlighting Agent Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python agent_code_formatter.py
    python agent_code_formatter.py /path/to/model.gguf
    python agent_code_formatter.py models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
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
            info("Usage: python agent_code_formatter.py /path/to/model.gguf")
            return 1

    header("CODE FORMATTING & HIGHLIGHTING EXAMPLES")

    print("\nThis example demonstrates:")
    numbered([
        "Python syntax highlighting with ANSI colors",
        "JSON pretty-printing and highlighting",
        "Adding line numbers to code",
        "Creating colored diffs",
        "Exporting to markdown and HTML",
        "Using agents to format code"
    ])

    # Show color reference
    show_color_reference()

    # Run non-agent examples first (don't need model)
    example_python_highlighting()
    example_json_highlighting()
    example_code_with_line_numbers()
    example_diff_display()
    example_markdown_export()
    example_html_export()

    # Initialize LLM for agent example
    info(f"Loading model: {model_path.name}...")
    config = GenerationConfig(n_batch=4096, n_ctx=8192)
    llm = LLM(str(model_path), config=config, verbose=args.verbose)
    success("Model loaded successfully.")

    # Run agent example
    example_with_agent(llm)

    return 0


if __name__ == "__main__":
    exit(main())
