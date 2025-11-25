"""
Framework Integration Examples

Demonstrates how to use cyllama agents with popular frameworks:
- LangChain
- OpenAI-compatible function calling
"""

from pathlib import Path


# Define some example tools
from cyllama.agents import tool


@tool
def calculator(expression: str) -> float:
    """
    Evaluate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        Result of the calculation
    """
    try:
        return float(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")


@tool
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """
    Get the current weather for a location (mock implementation).

    Args:
        location: City name
        unit: Temperature unit (celsius or fahrenheit)

    Returns:
        Weather information
    """
    # Mock weather data
    weather = {
        "san francisco": {"temp": 18, "condition": "Foggy"},
        "tokyo": {"temp": 22, "condition": "Sunny"},
        "paris": {"temp": 15, "condition": "Rainy"},
    }

    loc_lower = location.lower()
    if loc_lower in weather:
        data = weather[loc_lower]
        temp = data["temp"]
        if unit == "fahrenheit":
            temp = (temp * 9/5) + 32
        return f"{location}: {temp}°{unit[0].upper()}, {data['condition']}"
    else:
        return f"Weather data not available for {location}"


def example_openai_function_calling():
    """Demonstrate OpenAI-compatible function calling."""
    print("\n" + "="*60)
    print("OPENAI FUNCTION CALLING EXAMPLE")
    print("="*60)

    from cyllama.integrations.openai_agents import (
        OpenAIFunctionCallingClient,
        cyllama_tools_to_openai_tools,
    )

    ROOT = Path.cwd()
    model_path = ROOT / 'models' / 'Llama-3.2-1B-Instruct-Q8_0.gguf'

    if not model_path.exists():
        print(f"\nModel not found: {model_path}")
        print("Skipping OpenAI example")
        return

    print("\n1. Creating OpenAI Function Calling Client...")
    client = OpenAIFunctionCallingClient(
        model_path=str(model_path),
        tools=[calculator, get_current_weather],
        verbose=False
    )

    print("\n2. Listing available functions:")
    functions = client.list_functions()
    for func in functions:
        print(f"   - {func['name']}: {func['description']}")

    print("\n3. Functions in OpenAI tools format:")
    tools_format = client.list_tools()
    import json
    print(json.dumps(tools_format[0], indent=2))

    print("\n4. Example chat completion with function calling:")
    print("   (Note: This would require model inference)")
    print("   messages = [{'role': 'user', 'content': 'What is 25 * 4?'}]")
    print("   response = client.chat_completion_with_functions(messages)")


def example_langchain_tool_conversion():
    """Demonstrate LangChain tool conversion."""
    print("\n" + "="*60)
    print("LANGCHAIN TOOL CONVERSION EXAMPLE")
    print("="*60)

    try:
        from cyllama.integrations.langchain_agents import (
            cyllama_tool_to_langchain,
            LANGCHAIN_AVAILABLE,
        )

        if not LANGCHAIN_AVAILABLE:
            print("\nLangChain is not installed. Skipping example.")
            print("Install with: pip install langchain langchain-core")
            return

        print("\n1. Converting cyllama tool to LangChain tool...")
        lc_calculator = cyllama_tool_to_langchain(calculator)

        print(f"   Tool name: {lc_calculator.name}")
        print(f"   Tool description: {lc_calculator.description}")

        print("\n2. Using the LangChain tool:")
        result = lc_calculator.func(expression="10 + 15")
        print(f"   calculator(expression='10 + 15') = {result}")

    except ImportError as e:
        print(f"\nLangChain import error: {e}")
        print("Install with: pip install langchain langchain-core")


def example_cyllama_agent_with_langchain():
    """Demonstrate using cyllama agent with LangChain."""
    print("\n" + "="*60)
    print("CYLLAMA AGENT WITH LANGCHAIN EXAMPLE")
    print("="*60)

    try:
        from cyllama.integrations.langchain_agents import (
            CyllamaAgentLangChainAdapter,
            create_cyllama_react_agent,
            LANGCHAIN_AVAILABLE,
        )

        if not LANGCHAIN_AVAILABLE:
            print("\nLangChain is not installed. Skipping example.")
            return

        ROOT = Path.cwd()
        model_path = ROOT / 'models' / 'Llama-3.2-1B-Instruct-Q8_0.gguf'

        if not model_path.exists():
            print(f"\nModel not found: {model_path}")
            print("Skipping example")
            return

        print("\n1. Creating cyllama ReAct agent...")
        agent = create_cyllama_react_agent(
            model_path=str(model_path),
            tools=[calculator, get_current_weather],
            verbose=False
        )

        print("\n2. Wrapping with LangChain adapter...")
        lc_adapter = CyllamaAgentLangChainAdapter(agent)

        print("\n3. Using LangChain-style interface:")
        print("   inputs = {'input': 'What is 12 * 8?'}")
        print("   output = lc_adapter(inputs)")
        print("   # Returns: {'output': '96', 'intermediate_steps': [...]}")

    except ImportError as e:
        print(f"\nImport error: {e}")


def example_framework_comparison():
    """Compare different framework approaches."""
    print("\n" + "="*60)
    print("FRAMEWORK COMPARISON")
    print("="*60)

    print("\n1. Pure cyllama agent:")
    print("   from cyllama import LLM")
    print("   from cyllama.agents import ReActAgent, tool")
    print("   ")
    print("   llm = LLM('model.gguf')")
    print("   agent = ReActAgent(llm=llm, tools=[calculator])")
    print("   result = agent.run('What is 5 + 3?')")

    print("\n2. LangChain integration:")
    print("   from cyllama.integrations import cyllama_tool_to_langchain")
    print("   ")
    print("   lc_tool = cyllama_tool_to_langchain(calculator)")
    print("   # Use with LangChain agents, chains, etc.")

    print("\n3. OpenAI-compatible function calling:")
    print("   from cyllama.integrations import OpenAIFunctionCallingClient")
    print("   ")
    print("   client = OpenAIFunctionCallingClient('model.gguf', tools=[calculator])")
    print("   response = client.chat_completion_with_functions(messages)")

    print("\n4. Constrained agent (best reliability):")
    print("   from cyllama.agents import ConstrainedAgent")
    print("   ")
    print("   agent = ConstrainedAgent(llm=llm, tools=[calculator])")
    print("   result = agent.run('Calculate 10 * 5')")
    print("   # Guaranteed valid JSON tool calls")


def main():
    """Run all framework integration examples."""
    print("\n" + "="*70)
    print(" "*15 + "FRAMEWORK INTEGRATION EXAMPLES")
    print("="*70)

    # 1. OpenAI function calling
    example_openai_function_calling()

    # 2. LangChain tool conversion
    example_langchain_tool_conversion()

    # 3. Cyllama agent with LangChain
    example_cyllama_agent_with_langchain()

    # 4. Framework comparison
    example_framework_comparison()

    print("\n" + "="*70)
    print("EXAMPLES COMPLETE")
    print("="*70)

    print("\nKey Takeaways:")
    print("  - Cyllama agents work with LangChain, OpenAI-style APIs")
    print("  - Tool conversion is bidirectional (cyllama ↔ LangChain)")
    print("  - OpenAI function calling format supported")
    print("  - Choose the right tool for your use case:")
    print("    * Pure cyllama: Maximum control, zero dependencies")
    print("    * LangChain: Ecosystem compatibility")
    print("    * OpenAI-compatible: Drop-in replacement")
    print("    * ConstrainedAgent: Production reliability")


if __name__ == "__main__":
    main()
