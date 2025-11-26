"""
Contract Agent Example

Demonstrates using a ContractAgent with C++26-inspired contract assertions.
Shows how to:
- Define tools with @pre and @post contracts
- Use contract_assert for runtime invariants
- Configure violation handling policies
- Handle contract violations gracefully

Usage:
    python agent_contract_example.py [model_path]

    If no model_path is provided, will look for models in ./models/ directory.
"""

import argparse
from pathlib import Path
from cyllama import LLM, GenerationConfig
from cyllama.agents import (
    ContractAgent,
    ContractPolicy,
    ContractViolation,
    tool,
    pre,
    post,
    contract_assert,
    EventType,
)
from cyllama.utils.color import (
    header, section, subsection, subheader, success, error, info,
    bullet, numbered, kv, divider
)


# =============================================================================
# Define Tools with Contracts
# =============================================================================

@tool
@pre(lambda args: args['a'] is not None, "first operand required")
@pre(lambda args: args['b'] is not None, "second operand required")
@post(lambda r: isinstance(r, (int, float)), "result must be numeric")
def add(a: float, b: float) -> float:
    """
    Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b
    """
    return a + b


@tool
@pre(lambda args: args['b'] != 0, "cannot divide by zero")
@post(lambda r: r is not None, "result must not be None")
@post(lambda r, args: abs(r * args['b'] - args['a']) < 0.0001, "division check failed")
def divide(a: float, b: float) -> float:
    """
    Divide a by b.

    Args:
        a: Dividend
        b: Divisor (must not be zero)

    Returns:
        Result of a / b
    """
    return a / b


@tool
@pre(lambda args: args['items'], "items list must not be empty")
@post(lambda r: r >= 0, "average must be non-negative for non-negative inputs")
def calculate_average(items: str) -> float:
    """
    Calculate the average of a list of numbers.

    Args:
        items: Comma-separated list of numbers (e.g., "1,2,3,4,5")

    Returns:
        Average of the numbers
    """
    numbers = [float(x.strip()) for x in items.split(',')]
    contract_assert(len(numbers) > 0, "must have at least one number")
    return sum(numbers) / len(numbers)


@tool
@pre(lambda args: args['text'], "text must not be empty")
@post(lambda r: r >= 0, "word count must be non-negative")
def count_words(text: str) -> int:
    """
    Count the number of words in text.

    Args:
        text: Text to count words in

    Returns:
        Number of words
    """
    words = text.split()
    contract_assert(isinstance(words, list), "split must return a list")
    return len(words)


@tool
@pre(lambda args: 0 <= args['n'] <= 30, "n must be between 0 and 30")
@post(lambda r: r >= 0, "fibonacci result must be non-negative")
def fibonacci(n: int) -> int:
    """
    Calculate the nth Fibonacci number.

    Args:
        n: Index in Fibonacci sequence (0-30)

    Returns:
        The nth Fibonacci number
    """
    contract_assert(n >= 0, "n must be non-negative")
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


# =============================================================================
# Custom Violation Handler
# =============================================================================

violation_log = []


def logging_handler(violation: ContractViolation) -> None:
    """Custom handler that logs violations."""
    violation_log.append(violation)
    error(f"CONTRACT VIOLATION [{violation.kind.upper()}]")
    kv("Location", violation.location, value_color='yellow')
    kv("Predicate", violation.predicate, value_color='cyan')
    kv("Message", violation.message, value_color='red')
    if violation.context:
        kv("Context", str(violation.context)[:100], value_color='white')


# =============================================================================
# Helper Functions
# =============================================================================

def find_model() -> Path:
    """Find a model in the default locations."""
    ROOT = Path.cwd()
    candidates = [
        ROOT / 'models' / 'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf',
        ROOT / 'models' / 'Llama-3.2-1B-Instruct-Q8_0.gguf',
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


# =============================================================================
# Examples
# =============================================================================

def example_basic_contracts(llm: LLM):
    """Demonstrate basic contract checking."""
    section("BASIC CONTRACT EXAMPLE")

    agent = ContractAgent(
        llm=llm,
        tools=[add, divide, count_words],
        policy=ContractPolicy.ENFORCE,
        violation_handler=logging_handler,
        max_iterations=5,
        verbose=True
    )

    subsection("Task: Calculate 100 divided by 4", color='yellow')
    info("This task should succeed - contracts will pass")

    global violation_log
    violation_log = []

    result = agent.run("What is 100 divided by 4?")

    subsection("RESULT", color='bright_green')
    kv("Success", str(result.success), value_color='green' if result.success else 'red')
    kv("Answer", result.answer)
    kv("Violations", str(len(violation_log)))

    stats = agent.get_contract_stats()
    kv("Contract checks", str(stats['checks']))
    kv("Contract violations", str(stats['violations']))


def example_precondition_violation(llm: LLM):
    """Demonstrate precondition violation handling."""
    section("PRECONDITION VIOLATION EXAMPLE")

    agent = ContractAgent(
        llm=llm,
        tools=[divide],
        policy=ContractPolicy.OBSERVE,  # Continue after violation
        violation_handler=logging_handler,
        max_iterations=5,
        verbose=True
    )

    subsection("Task: Divide by zero (will violate precondition)", color='yellow')
    info("Using OBSERVE policy - will log violation but continue")

    global violation_log
    violation_log = []

    # The LLM might try to divide by zero if we ask it to
    result = agent.run("Calculate 10 divided by 0")

    subsection("RESULT", color='bright_green')
    kv("Success", str(result.success), value_color='green' if result.success else 'red')
    kv("Answer", result.answer[:200] if result.answer else "No answer")
    kv("Violations logged", str(len(violation_log)))


def example_task_precondition(llm: LLM):
    """Demonstrate task-level precondition."""
    section("TASK PRECONDITION EXAMPLE")

    def task_not_empty(task: str) -> bool:
        """Task must have at least 10 characters."""
        return len(task) >= 10

    agent = ContractAgent(
        llm=llm,
        tools=[add],
        policy=ContractPolicy.ENFORCE,
        violation_handler=logging_handler,
        task_precondition=task_not_empty,
        max_iterations=5,
        verbose=True
    )

    subsection("Task 1: Valid task (>= 10 chars)", color='yellow')

    global violation_log
    violation_log = []

    result = agent.run("What is 5 plus 3?")  # 16 chars - passes

    kv("Success", str(result.success), value_color='green' if result.success else 'red')
    kv("Violations", str(len(violation_log)))

    print()
    subsection("Task 2: Invalid task (< 10 chars)", color='yellow')

    violation_log = []

    result = agent.run("Hi")  # 2 chars - fails precondition

    kv("Success", str(result.success), value_color='green' if result.success else 'red')
    kv("Violations", str(len(violation_log)))
    if violation_log:
        kv("Violation type", violation_log[0].kind)


def example_answer_postcondition(llm: LLM):
    """Demonstrate answer-level postcondition."""
    section("ANSWER POSTCONDITION EXAMPLE")

    def answer_is_numeric(answer: str) -> bool:
        """Answer must contain a number."""
        import re
        return bool(re.search(r'\d', answer))

    agent = ContractAgent(
        llm=llm,
        tools=[add, fibonacci],
        policy=ContractPolicy.OBSERVE,
        violation_handler=logging_handler,
        answer_postcondition=answer_is_numeric,
        max_iterations=5,
        verbose=True
    )

    subsection("Task: Calculate fibonacci(10)", color='yellow')
    info("Answer postcondition requires numeric content")

    global violation_log
    violation_log = []

    result = agent.run("What is the 10th Fibonacci number?")

    subsection("RESULT", color='bright_green')
    kv("Success", str(result.success), value_color='green' if result.success else 'red')
    kv("Answer", result.answer[:100] if result.answer else "No answer")
    kv("Answer has number", str(bool(__import__('re').search(r'\d', result.answer or ''))))


def example_iteration_invariant(llm: LLM):
    """Demonstrate iteration invariant."""
    section("ITERATION INVARIANT EXAMPLE")

    def max_3_iterations(state) -> bool:
        """Limit to 3 iterations."""
        return state.iterations <= 3

    agent = ContractAgent(
        llm=llm,
        tools=[add, divide, count_words],
        policy=ContractPolicy.ENFORCE,
        violation_handler=logging_handler,
        iteration_invariant=max_3_iterations,
        max_iterations=10,  # Higher than invariant allows
        verbose=True
    )

    subsection("Task with iteration limit invariant", color='yellow')
    info("Invariant limits to 3 iterations even though max_iterations=10")

    global violation_log
    violation_log = []

    result = agent.run(
        "First add 5 and 3, then divide the result by 2, "
        "then count words in 'hello world test'"
    )

    subsection("RESULT", color='bright_green')
    kv("Success", str(result.success), value_color='green' if result.success else 'red')
    kv("Iterations", str(result.iterations))
    kv("Violations", str(len(violation_log)))


def example_policy_comparison(llm: LLM):
    """Compare different contract policies."""
    section("POLICY COMPARISON")

    policies = [
        (ContractPolicy.IGNORE, "Skips all checks"),
        (ContractPolicy.OBSERVE, "Logs violations, continues"),
        (ContractPolicy.ENFORCE, "Logs violations, terminates"),
        (ContractPolicy.QUICK_ENFORCE, "Terminates immediately"),
    ]

    for policy, description in policies:
        print()
        subheader(f"{policy.value.upper()}", color='cyan')
        bullet(description)


def show_contract_features():
    """Show available contract features."""
    section("CONTRACT FEATURES")

    subheader("1. TOOL PRECONDITIONS (@pre)", color='cyan')
    bullet("Checked before tool execution")
    bullet("Access to tool arguments via args dict")
    bullet("Multiple preconditions per tool supported")

    print()
    subheader("2. TOOL POSTCONDITIONS (@post)", color='cyan')
    bullet("Checked after tool execution")
    bullet("Access to result and optionally original args")
    bullet("Validate output format, ranges, invariants")

    print()
    subheader("3. RUNTIME ASSERTIONS (contract_assert)", color='cyan')
    bullet("Called within tool implementations")
    bullet("Participates in same violation handling")
    bullet("Useful for internal invariants")

    print()
    subheader("4. AGENT-LEVEL CONTRACTS", color='cyan')
    bullet("task_precondition: Validate input task")
    bullet("answer_postcondition: Validate final answer")
    bullet("iteration_invariant: Check at each iteration")

    print()
    subheader("5. POLICIES", color='cyan')
    bullet("IGNORE: Skip checking (production performance)")
    bullet("OBSERVE: Check and log, continue execution")
    bullet("ENFORCE: Check, handle, terminate")
    bullet("QUICK_ENFORCE: Check, terminate immediately")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all contract agent examples."""
    parser = argparse.ArgumentParser(
        description="Contract Agent Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python agent_contract_example.py
    python agent_contract_example.py /path/to/model.gguf
    python agent_contract_example.py models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
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
            info("Usage: python agent_contract_example.py /path/to/model.gguf")
            return 1

    header("CONTRACT AGENT EXAMPLES")

    print("\nThis example demonstrates C++26-inspired contracts:")
    numbered([
        "Tool preconditions (@pre) - validate inputs",
        "Tool postconditions (@post) - validate outputs",
        "Runtime assertions (contract_assert) - internal invariants",
        "Task/answer contracts - agent-level validation",
        "Configurable violation policies"
    ])

    # Show features
    show_contract_features()

    # Show policy comparison
    example_policy_comparison(None)

    # Initialize LLM
    print()
    info(f"Loading model: {model_path.name}...")
    config = GenerationConfig(n_batch=4096, n_ctx=8192)
    llm = LLM(str(model_path), config=config, verbose=args.verbose)
    success("Model loaded successfully.")

    # Run examples
    example_basic_contracts(llm)
    example_task_precondition(llm)
    example_answer_postcondition(llm)

    return 0


if __name__ == "__main__":
    exit(main())
