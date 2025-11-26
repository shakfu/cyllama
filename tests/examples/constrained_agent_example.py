"""
Example demonstrating the ConstrainedAgent with grammar-enforced tool calling.

This example shows how grammar constraints guarantee valid JSON tool calls,
eliminating parsing failures even with smaller models.
"""

from cyllama import LLM
from cyllama.agents import ConstrainedAgent, tool
from cyllama.agents.grammar import (
    generate_tool_call_schema,
    generate_tool_call_grammar,
    GrammarFormat
)
from cyllama.utils.color import (
    header, section, subsection, subheader, success, error, info,
    bullet, kv, divider
)
from pathlib import Path
import json


# Define tools with type hints


@tool
def calculator(expression: str) -> float:
    """
    Evaluate a mathematical expression safely.

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2", "10 * 5")

    Returns:
        Result of the calculation
    """
    try:
        # Simple eval for demo - use proper math parser in production
        result = eval(expression, {"__builtins__": {}}, {})
        return float(result)
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")


@tool
def get_weather(city: str, units: str = "celsius") -> str:
    """
    Get weather information for a city (mock implementation).

    Args:
        city: Name of the city
        units: Temperature units (celsius or fahrenheit)

    Returns:
        Weather information
    """
    # Mock weather data
    weather_data = {
        "paris": {"temp": 18, "condition": "Partly cloudy"},
        "london": {"temp": 15, "condition": "Rainy"},
        "tokyo": {"temp": 22, "condition": "Sunny"},
        "new york": {"temp": 20, "condition": "Clear"},
    }

    city_lower = city.lower()
    if city_lower in weather_data:
        data = weather_data[city_lower]
        temp = data["temp"]
        if units == "fahrenheit":
            temp = (temp * 9/5) + 32
        return f"{city}: {temp}°{units[0].upper()}, {data['condition']}"
    else:
        return f"Weather data not available for {city}"


@tool
def string_length(text: str) -> int:
    """
    Calculate the length of a string.

    Args:
        text: String to measure

    Returns:
        Length of the string
    """
    return len(text)


def demonstrate_schema_generation():
    """Demonstrate schema and grammar generation."""
    section("SCHEMA AND GRAMMAR GENERATION")

    tools = [calculator, get_weather, string_length]

    # Generate JSON schema
    subheader("1. Generated JSON Schema", color='cyan')
    schema = generate_tool_call_schema(
        tools,
        allow_reasoning=True,
        format=GrammarFormat.JSON
    )
    print(json.dumps(schema, indent=2))

    # Generate GBNF grammar
    print()
    subheader("2. Generated GBNF Grammar (first 500 chars)", color='cyan')
    grammar = generate_tool_call_grammar(
        tools,
        allow_reasoning=True,
        format=GrammarFormat.JSON
    )
    print(grammar[:500] + "...")
    kv("Total grammar length", f"{len(grammar)} characters")


def demonstrate_constrained_agent():
    """Demonstrate constrained agent with real model."""
    ROOT = Path.cwd()
    model_path = ROOT / 'models' / 'Llama-3.2-1B-Instruct-Q8_0.gguf'

    if not model_path.exists():
        error(f"Model not found: {model_path}")
        info("Please download a model first.")
        return

    section("CONSTRAINED AGENT DEMONSTRATION")

    info("Initializing LLM...")
    llm = LLM(str(model_path), verbose=False)

    info("Creating ConstrainedAgent...")
    agent = ConstrainedAgent(
        llm=llm,
        tools=[calculator, get_weather, string_length],
        max_iterations=5,
        verbose=True,  # Show reasoning
        format="json",
        allow_reasoning=False  # Simpler for demo
    )

    subsection("Agent initialized with tools:")
    for tool_obj in agent.list_tools():
        bullet(f"{tool_obj.name}: {tool_obj.description}")

    # Example tasks
    tasks = [
        "What is 25 multiplied by 4?",
        "What is the weather in Paris?",
        "Calculate (100 + 50) / 3",
    ]

    for i, task in enumerate(tasks, 1):
        subsection(f"Task {i}: {task}", color='yellow')

        try:
            result = agent.run(task)

            if result.success:
                success(f"Answer: {result.answer}")
                kv("Tool calls made", str(result.iterations))

                # Show detailed steps
                subheader("Execution trace", color='cyan')
                for j, event in enumerate(result.steps, 1):
                    content = event.content[:80] + ('...' if len(event.content) > 80 else '')
                    bullet(f"{event.type.value}: {content}", indent=1)
            else:
                error(f"Failed: {result.error}")

        except Exception as e:
            error(f"Error: {str(e)}")

        print()


def demonstrate_format_comparison():
    """Compare different output formats."""
    section("FORMAT COMPARISON")

    tools = [calculator]

    formats = [
        (GrammarFormat.JSON, "Simple JSON"),
        (GrammarFormat.FUNCTION_CALL, "OpenAI-style Function Call"),
    ]

    for fmt, description in formats:
        print()
        subheader(f"{description} ({fmt.value})", color='cyan')
        schema = generate_tool_call_schema(tools, allow_reasoning=False, format=fmt)
        print(json.dumps(schema, indent=2))


def demonstrate_caching():
    """Demonstrate grammar caching for performance."""
    section("GRAMMAR CACHING DEMONSTRATION")

    from cyllama.agents.grammar import get_cached_tool_grammar, clear_grammar_cache
    import time

    tools = [calculator, get_weather, string_length]

    # Clear cache
    clear_grammar_cache()

    # First generation (not cached)
    subheader("1. First generation (not cached)", color='cyan')
    start = time.time()
    grammar1 = get_cached_tool_grammar(tools)
    elapsed1 = time.time() - start
    kv("Time", f"{elapsed1*1000:.2f}ms")
    kv("Grammar length", f"{len(grammar1)} chars")

    # Second generation (cached)
    print()
    subheader("2. Second generation (cached)", color='cyan')
    start = time.time()
    grammar2 = get_cached_tool_grammar(tools)
    elapsed2 = time.time() - start
    kv("Time", f"{elapsed2*1000:.2f}ms")
    kv("Grammar length", f"{len(grammar2)} chars")

    print()
    kv("Speedup", f"{elapsed1/elapsed2:.1f}x faster", value_color='green')
    kv("Grammars identical", str(grammar1 == grammar2), value_color='green')


def main():
    """Run all demonstrations."""
    header("CONSTRAINED AGENT EXAMPLES")

    # 1. Show schema and grammar generation
    demonstrate_schema_generation()

    # 2. Compare formats
    demonstrate_format_comparison()

    # 3. Show grammar caching
    demonstrate_caching()

    # 4. Run actual agent (if model available)
    demonstrate_constrained_agent()


if __name__ == "__main__":
    main()
