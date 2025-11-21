"""Tests for JSON schema to grammar conversion."""

import pytest
from cyllama.llama.llama_cpp import json_schema_to_grammar


def test_json_schema_basic_object():
    """Test converting a basic object schema to grammar."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }

    grammar = json_schema_to_grammar(schema)

    # Should return a non-empty string
    assert isinstance(grammar, str)
    assert len(grammar) > 0

    # Grammar should contain some GBNF-like patterns
    # (exact format depends on llama.cpp implementation)
    print(f"\nGenerated grammar:\n{grammar[:200]}...")


def test_json_schema_string_input():
    """Test that schema can be passed as JSON string."""
    schema_str = '{"type": "object", "properties": {"name": {"type": "string"}}}'

    grammar = json_schema_to_grammar(schema_str)

    assert isinstance(grammar, str)
    assert len(grammar) > 0


def test_json_schema_nested_object():
    """Test schema with nested objects."""
    schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"}
                }
            },
            "timestamp": {"type": "integer"}
        }
    }

    grammar = json_schema_to_grammar(schema)

    assert isinstance(grammar, str)
    assert len(grammar) > 0


def test_json_schema_array():
    """Test schema with array type."""
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    }

    grammar = json_schema_to_grammar(schema)

    assert isinstance(grammar, str)
    assert len(grammar) > 0


def test_json_schema_multiple_types():
    """Test schema with various data types."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "score": {"type": "number"},
            "active": {"type": "boolean"},
            "tags": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    }

    grammar = json_schema_to_grammar(schema)

    assert isinstance(grammar, str)
    assert len(grammar) > 0
    print(f"\nComplex schema grammar length: {len(grammar)} chars")


def test_json_schema_force_gbnf():
    """Test force_gbnf parameter."""
    schema = {
        "type": "object",
        "properties": {
            "message": {"type": "string"}
        }
    }

    # Test with force_gbnf=False (default)
    grammar1 = json_schema_to_grammar(schema, force_gbnf=False)
    assert isinstance(grammar1, str)

    # Test with force_gbnf=True
    grammar2 = json_schema_to_grammar(schema, force_gbnf=True)
    assert isinstance(grammar2, str)

    # Both should produce valid grammars
    # (they may or may not be identical depending on llama.cpp version)
    assert len(grammar1) > 0
    assert len(grammar2) > 0


def test_json_schema_invalid_json_string():
    """Test error handling for invalid JSON string."""
    invalid_json = '{"type": "object", invalid}'

    with pytest.raises(ValueError, match="Invalid JSON schema string"):
        json_schema_to_grammar(invalid_json)


def test_json_schema_invalid_type():
    """Test error handling for invalid schema type."""
    with pytest.raises(TypeError, match="Schema must be dict or str"):
        json_schema_to_grammar(123)


def test_json_schema_dict_and_string_equivalence():
    """Test that dict and string schemas produce same grammar."""
    schema_dict = {
        "type": "object",
        "properties": {
            "value": {"type": "integer"}
        }
    }

    import json
    schema_str = json.dumps(schema_dict)

    grammar1 = json_schema_to_grammar(schema_dict)
    grammar2 = json_schema_to_grammar(schema_str)

    # Should produce identical grammars
    assert grammar1 == grammar2


def test_json_schema_enum():
    """Test schema with enum constraint."""
    schema = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["active", "inactive", "pending"]
            }
        }
    }

    grammar = json_schema_to_grammar(schema)

    assert isinstance(grammar, str)
    assert len(grammar) > 0


def test_json_schema_real_world_example():
    """Test a real-world schema example."""
    schema = {
        "type": "object",
        "properties": {
            "response": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "answer": {"type": "string"},
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["reasoning", "answer", "confidence"]
            }
        },
        "required": ["response"]
    }

    grammar = json_schema_to_grammar(schema)

    assert isinstance(grammar, str)
    assert len(grammar) > 0

    print(f"\nReal-world schema grammar:\n{grammar[:300]}...")


if __name__ == "__main__":
    # Run tests manually
    print("Testing JSON schema to grammar conversion...")
    test_json_schema_basic_object()
    test_json_schema_string_input()
    test_json_schema_nested_object()
    test_json_schema_array()
    test_json_schema_multiple_types()
    test_json_schema_force_gbnf()
    test_json_schema_invalid_json_string()
    test_json_schema_invalid_type()
    test_json_schema_dict_and_string_equivalence()
    test_json_schema_enum()
    test_json_schema_real_world_example()
    print("\nAll JSON schema tests passed!")
