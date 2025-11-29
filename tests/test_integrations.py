"""
Tests for integration helpers.
"""

import pytest
from cyllama.integrations.openai_compat import (
    OpenAICompatibleClient,
    create_openai_client,
    ChatCompletion,
    ChatCompletionChunk,
)


class TestOpenAICompatibleClient:
    """Tests for OpenAI-compatible client."""

    @pytest.mark.slow
    def test_client_initialization(self, model_path):
        """Test client initialization."""
        client = OpenAICompatibleClient(model_path, verbose=False)
        assert client is not None
        assert client.generator is not None

    @pytest.mark.slow
    def test_chat_completion(self, model_path):
        """Test chat completion."""
        client = OpenAICompatibleClient(model_path, temperature=0.0)

        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": "What is 2+2?"}
            ],
            max_tokens=30
        )

        assert isinstance(response, ChatCompletion)
        assert len(response.choices) == 1
        assert response.choices[0].message.role == "assistant"
        assert len(response.choices[0].message.content) > 0
        assert response.choices[0].finish_reason == "stop"
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens >= 0

    @pytest.mark.slow
    def test_chat_completion_streaming(self, model_path):
        """Test streaming chat completion."""
        client = OpenAICompatibleClient(model_path, temperature=0.0)

        chunks = list(client.chat.completions.create(
            messages=[
                {"role": "user", "content": "Count to 3"}
            ],
            max_tokens=20,
            stream=True
        ))

        assert len(chunks) > 0
        assert all(isinstance(chunk, ChatCompletionChunk) for chunk in chunks)

        # First chunk should have role
        assert chunks[0].choices[0].delta.role == "assistant"

        # Middle chunks should have content
        content_chunks = [c for c in chunks if c.choices[0].delta.content]
        assert len(content_chunks) > 0

        # Last chunk should have finish_reason
        assert chunks[-1].choices[0].finish_reason == "stop"

        # Reconstruct full response
        full_content = "".join(
            chunk.choices[0].delta.content or ""
            for chunk in chunks
        )
        assert len(full_content) > 0

    @pytest.mark.slow
    def test_multiple_messages(self, model_path):
        """Test chat with multiple messages."""
        client = OpenAICompatibleClient(model_path, temperature=0.0)

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "What is Python?"}
            ],
            max_tokens=50
        )

        assert isinstance(response, ChatCompletion)
        assert len(response.choices[0].message.content) > 0

    @pytest.mark.slow
    def test_custom_parameters(self, model_path):
        """Test custom generation parameters."""
        client = OpenAICompatibleClient(model_path)

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Test"}],
            temperature=0.5,
            top_p=0.9,
            max_tokens=20
        )

        assert isinstance(response, ChatCompletion)

    @pytest.mark.slow
    def test_stop_sequences(self, model_path):
        """Test stop sequences."""
        client = OpenAICompatibleClient(model_path, temperature=0.0)

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Count: 1, 2, 3"}],
            max_tokens=50,
            stop=["3"]
        )

        assert isinstance(response, ChatCompletion)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.slow
    def test_create_openai_client(self, model_path):
        """Test create_openai_client convenience function."""
        client = create_openai_client(model_path, temperature=0.5)

        assert client is not None
        assert client.generator is not None

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=10
        )

        assert isinstance(response, ChatCompletion)


class TestLangChainIntegration:
    """Tests for LangChain integration."""

    def test_import_without_langchain(self):
        """Test that importing without LangChain raises helpful error."""
        # This test verifies error handling when LangChain is not installed
        try:
            from cyllama.integrations import CyllamaLLM
            # If LangChain is installed, this should work
            assert CyllamaLLM is not None
        except ImportError as e:
            # If LangChain is not installed, should get helpful error
            assert "langchain" in str(e).lower()

    @pytest.mark.skipif(
        not pytest.importorskip("langchain", reason="langchain not installed"),
        reason="Requires langchain"
    )
    @pytest.mark.slow
    def test_langchain_basic_generation(self, model_path):
        """Test basic LangChain generation."""
        from cyllama.integrations import CyllamaLLM

        llm = CyllamaLLM(
            model_path=model_path,
            temperature=0.0,
            max_tokens=30,
            verbose=False
        )

        result = llm.invoke("What is 2+2? Answer briefly.")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.skipif(
        not pytest.importorskip("langchain", reason="langchain not installed"),
        reason="Requires langchain"
    )
    @pytest.mark.slow
    def test_langchain_with_stop_sequences(self, model_path):
        """Test LangChain generation with stop sequences."""
        from cyllama.integrations import CyllamaLLM

        llm = CyllamaLLM(
            model_path=model_path,
            temperature=0.0,
            max_tokens=50,
            stop_sequences=["\n\n"],
            verbose=False
        )

        result = llm.invoke("Count to 5")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.skipif(
        not pytest.importorskip("langchain", reason="langchain not installed"),
        reason="Requires langchain"
    )
    @pytest.mark.slow
    def test_langchain_streaming(self, model_path):
        """Test LangChain streaming generation."""
        from cyllama.integrations import CyllamaLLM

        llm = CyllamaLLM(
            model_path=model_path,
            temperature=0.0,
            max_tokens=30,
            verbose=False
        )

        chunks = list(llm.stream("Say hello"))
        assert len(chunks) > 0
        # Chunks should be strings (LangChain extracts .text automatically in newer versions)
        full_text = "".join(str(chunk) for chunk in chunks)
        assert len(full_text) > 0

    @pytest.mark.skipif(
        not pytest.importorskip("langchain", reason="langchain not installed"),
        reason="Requires langchain"
    )
    @pytest.mark.slow
    def test_langchain_identifying_params(self, model_path):
        """Test that LangChain can identify the LLM."""
        from cyllama.integrations import CyllamaLLM

        llm = CyllamaLLM(
            model_path=model_path,
            temperature=0.5,
            max_tokens=100,
            verbose=False
        )

        params = llm._identifying_params
        assert params["model_path"] == model_path
        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 100

    @pytest.mark.skipif(
        not pytest.importorskip("langchain", reason="langchain not installed"),
        reason="Requires langchain"
    )
    @pytest.mark.slow
    def test_langchain_llm_type(self, model_path):
        """Test LLM type identifier."""
        from cyllama.integrations import CyllamaLLM

        llm = CyllamaLLM(model_path=model_path, verbose=False)
        assert llm._llm_type == "cyllama"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
