"""Tests for the Response class."""

import json
import pytest
from cyllama import Response, GenerationStats, complete, LLM


class TestResponseClass:
    """Tests for the Response dataclass."""

    def test_response_str(self):
        """Test that Response can be used as a string via __str__."""
        response = Response(text="Hello, world!")
        assert str(response) == "Hello, world!"
        assert f"{response}" == "Hello, world!"

    def test_response_print(self, capsys):
        """Test that print(response) works correctly."""
        response = Response(text="Test output")
        print(response)
        captured = capsys.readouterr()
        assert captured.out.strip() == "Test output"

    def test_response_equality_with_string(self):
        """Test Response equality comparison with strings."""
        response = Response(text="hello")
        assert response == "hello"
        assert "hello" == response.text  # Direct text comparison
        assert response != "world"

    def test_response_equality_with_response(self):
        """Test Response equality comparison with other Response objects."""
        r1 = Response(text="hello")
        r2 = Response(text="hello")
        r3 = Response(text="world")
        assert r1 == r2
        assert r1 != r3

    def test_response_len(self):
        """Test len(response) returns text length."""
        response = Response(text="hello")
        assert len(response) == 5

    def test_response_iter(self):
        """Test iterating over response yields characters."""
        response = Response(text="abc")
        chars = list(response)
        assert chars == ["a", "b", "c"]

    def test_response_contains(self):
        """Test 'in' operator works with Response."""
        response = Response(text="hello world")
        assert "hello" in response
        assert "world" in response
        assert "foo" not in response

    def test_response_add(self):
        """Test string concatenation with Response."""
        response = Response(text="hello")
        assert response + " world" == "hello world"
        assert "say " + response == "say hello"

    def test_response_hash(self):
        """Test Response can be used in sets/dicts."""
        r1 = Response(text="hello")
        r2 = Response(text="hello")
        r3 = Response(text="world")

        # Same text = same hash
        assert hash(r1) == hash(r2)

        # Can be used in a set
        s = {r1, r2, r3}
        assert len(s) == 2  # r1 and r2 are considered equal

    def test_response_repr(self):
        """Test Response repr for debugging."""
        response = Response(text="hello", finish_reason="stop")
        repr_str = repr(response)
        assert "Response" in repr_str
        assert "hello" in repr_str
        assert "stop" in repr_str

    def test_response_repr_truncation(self):
        """Test Response repr truncates long text."""
        long_text = "a" * 100
        response = Response(text=long_text)
        repr_str = repr(response)
        assert "..." in repr_str
        assert len(repr_str) < len(long_text) + 50  # Reasonable length

    def test_response_to_dict_basic(self):
        """Test Response.to_dict() without stats."""
        response = Response(text="hello", finish_reason="stop", model="test.gguf")
        d = response.to_dict()
        assert d["text"] == "hello"
        assert d["finish_reason"] == "stop"
        assert d["model"] == "test.gguf"
        assert "stats" not in d

    def test_response_to_dict_with_stats(self):
        """Test Response.to_dict() with stats."""
        stats = GenerationStats(
            prompt_tokens=10,
            generated_tokens=20,
            total_time=1.5,
            tokens_per_second=13.33
        )
        response = Response(text="hello", stats=stats, finish_reason="stop", model="test.gguf")
        d = response.to_dict()
        assert d["text"] == "hello"
        assert d["stats"]["prompt_tokens"] == 10
        assert d["stats"]["generated_tokens"] == 20
        assert d["stats"]["total_time"] == 1.5
        assert d["stats"]["tokens_per_second"] == 13.33

    def test_response_to_json(self):
        """Test Response.to_json() output."""
        response = Response(text="hello", finish_reason="stop", model="test.gguf")
        json_str = response.to_json()
        parsed = json.loads(json_str)
        assert parsed["text"] == "hello"
        assert parsed["finish_reason"] == "stop"

    def test_response_to_json_indent(self):
        """Test Response.to_json() with indentation."""
        response = Response(text="hello")
        json_str = response.to_json(indent=2)
        assert "\n" in json_str  # Indented output has newlines


class TestResponseIntegration:
    """Integration tests for Response with generation functions."""

    def test_complete_returns_response(self, model_path):
        """Test that complete() returns a Response object."""
        response = complete("Say 'hi'", model_path=model_path, max_tokens=10)
        assert isinstance(response, Response)
        assert isinstance(response.text, str)
        assert len(response.text) > 0
        # Backward compatible: can still print directly
        assert str(response) == response.text

    def test_complete_response_has_stats(self, model_path):
        """Test that complete() Response includes statistics."""
        response = complete("Say 'test'", model_path=model_path, max_tokens=10)
        assert response.stats is not None
        assert response.stats.prompt_tokens > 0
        assert response.stats.generated_tokens >= 0
        assert response.stats.total_time > 0

    def test_llm_returns_response(self, model_path):
        """Test that LLM() returns a Response object."""
        with LLM(model_path, max_tokens=10) as llm:
            response = llm("Say 'hello'")
            assert isinstance(response, Response)
            assert response.stats is not None
            assert response.model == model_path

    def test_response_backward_compatible_string_usage(self, model_path):
        """Test that Response works in string contexts."""
        response = complete("Say 'ok'", model_path=model_path, max_tokens=5)

        # These should all work without errors
        text = str(response)
        assert isinstance(text, str)

        # String concatenation
        combined = "Result: " + response
        assert "Result: " in combined

        # f-string
        formatted = f"Output: {response}"
        assert "Output: " in formatted

        # Length check
        assert len(response) == len(response.text)
