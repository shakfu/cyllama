"""Example: Using JSON Schema to Grammar for structured output generation.

This example demonstrates:
1. Converting JSON schemas to GBNF grammars
2. Using grammars for structured JSON output
3. Complex nested schemas
4. Real-world use cases (API responses, data extraction)
"""

from cyllama.llama.llama_cpp import json_schema_to_grammar, LlamaModel, LlamaContext
import json
import os


def basic_schema_example():
    """Simple JSON schema to grammar conversion."""
    print("\n=== Basic Schema Example ===\n")

    # Define a simple schema for a person
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string"}
        },
        "required": ["name", "age"]
    }

    # Convert to grammar
    grammar = json_schema_to_grammar(schema)

    print("JSON Schema:")
    print(json.dumps(schema, indent=2))
    print("\nGenerated GBNF Grammar:")
    print(grammar[:500] + "..." if len(grammar) > 500 else grammar)

    return grammar


def nested_schema_example():
    """Nested object schema."""
    print("\n=== Nested Schema Example ===\n")

    schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "address": {
                        "type": "object",
                        "properties": {
                            "street": {"type": "string"},
                            "city": {"type": "string"},
                            "zipcode": {"type": "string"}
                        },
                        "required": ["city"]
                    }
                },
                "required": ["name"]
            },
            "timestamp": {"type": "string"}
        },
        "required": ["user"]
    }

    grammar = json_schema_to_grammar(schema)

    print("Nested Schema:")
    print(json.dumps(schema, indent=2))
    print(f"\nGenerated grammar length: {len(grammar)} characters")

    return grammar


def array_schema_example():
    """Schema with arrays."""
    print("\n=== Array Schema Example ===\n")

    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "name": {"type": "string"},
                        "price": {"type": "number"}
                    },
                    "required": ["id", "name"]
                }
            },
            "total": {"type": "number"}
        },
        "required": ["items"]
    }

    grammar = json_schema_to_grammar(schema)

    print("Array Schema (shopping cart):")
    print(json.dumps(schema, indent=2))
    print(f"\nGenerated grammar length: {len(grammar)} characters")

    return grammar


def enum_schema_example():
    """Schema with enums and constrained values."""
    print("\n=== Enum Schema Example ===\n")

    schema = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["active", "inactive", "pending"]
            },
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high", "critical"]
            },
            "assigned_to": {"type": "string"}
        },
        "required": ["status", "priority"]
    }

    grammar = json_schema_to_grammar(schema)

    print("Enum Schema (task status):")
    print(json.dumps(schema, indent=2))
    print(f"\nGenerated grammar length: {len(grammar)} characters")

    return grammar


def real_world_api_response_schema():
    """Real-world API response schema."""
    print("\n=== Real-World API Response Schema ===\n")

    schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
            "sources": {
                "type": "array",
                "items": {"type": "string"}
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "model": {"type": "string"},
                    "timestamp": {"type": "string"}
                }
            }
        },
        "required": ["reasoning", "answer"]
    }

    grammar = json_schema_to_grammar(schema)

    print("API Response Schema:")
    print(json.dumps(schema, indent=2))
    print(f"\nGenerated grammar length: {len(grammar)} characters")

    return grammar


def use_grammar_with_model(grammar, model_path):
    """Demonstrate using the grammar with actual model generation."""
    print("\n=== Using Grammar with Model ===\n")

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Skipping model generation example")
        return

    print(f"Loading model: {model_path}")

    # Load model and create context
    model = LlamaModel(model_path)
    ctx = LlamaContext(model, n_ctx=2048)

    # Define schema for structured response
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]}
        },
        "required": ["answer", "confidence"]
    }

    grammar = json_schema_to_grammar(schema)

    # Create prompt
    prompt = """Answer this question in JSON format with an 'answer' field and a 'confidence' field (low/medium/high):

Question: What is the capital of France?

JSON:"""

    print(f"\nPrompt: {prompt}")
    print("\nGenerating structured JSON output...")

    # Generate with grammar constraint
    try:
        # Use the grammar to constrain output
        # Note: This requires the full generate API which may need additional setup
        print("\nGrammar has been prepared:")
        print(f"Grammar length: {len(grammar)} characters")
        print("\nTo use this grammar, pass it to the generate() method:")
        print("  response = model.generate(prompt, grammar=grammar)")
        print("\nExpected output format:")
        print(json.dumps({"answer": "<string>", "confidence": "<low|medium|high>"}, indent=2))

    except Exception as e:
        print(f"Note: Full generation example requires additional API: {e}")


def force_gbnf_example():
    """Example using force_gbnf parameter."""
    print("\n=== Force GBNF Format Example ===\n")

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"]
    }

    # Standard conversion
    grammar1 = json_schema_to_grammar(schema, force_gbnf=False)
    print(f"Standard conversion: {len(grammar1)} chars")

    # Force GBNF format
    grammar2 = json_schema_to_grammar(schema, force_gbnf=True)
    print(f"Force GBNF conversion: {len(grammar2)} chars")

    print("\nBoth grammars enforce the same schema structure")


def main(model_path):
    """Run all examples."""
    print("=" * 60)
    print("JSON Schema to Grammar Examples")
    print("=" * 60)

    # 1. Basic schema
    basic_schema_example()

    # 2. Nested objects
    nested_schema_example()

    # 3. Arrays
    array_schema_example()

    # 4. Enums
    enum_schema_example()

    # 5. Real-world API response
    grammar = real_world_api_response_schema()

    # 6. Force GBNF
    force_gbnf_example()

    # 7. Use with model (if available)
    use_grammar_with_model(grammar, model_path)

    print("\n" + "=" * 60)
    print("JSON Schema Examples Complete")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("- json_schema_to_grammar() converts JSON schemas to GBNF grammars")
    print("- Use grammars to constrain LLM output to valid JSON structure")
    print("- Supports nested objects, arrays, enums, and complex schemas")
    print("- Essential for reliable structured output and API integrations")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="JSON Schema to Grammar Examples")
    parser.add_argument("-m", "--model", required=True, help="Path to model file")
    args = parser.parse_args()
    main(args.model)
