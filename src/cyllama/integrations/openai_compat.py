"""
OpenAI-Compatible Client

This module provides an OpenAI-API-compatible interface for cyllama,
allowing drop-in replacement in code that uses the OpenAI Python client.

Example:
    >>> from cyllama.integrations.openai_compat import OpenAICompatibleClient
    >>>
    >>> client = OpenAICompatibleClient(model_path="models/llama.gguf")
    >>>
    >>> # Chat completions (OpenAI-style)
    >>> response = client.chat.completions.create(
    >>>     messages=[
    >>>         {"role": "system", "content": "You are a helpful assistant."},
    >>>         {"role": "user", "content": "What is Python?"}
    >>>     ],
    >>>     temperature=0.7,
    >>>     max_tokens=100
    >>> )
    >>> print(response.choices[0].message.content)
    >>>
    >>> # Streaming
    >>> for chunk in client.chat.completions.create(
    >>>     messages=[{"role": "user", "content": "Count to 5"}],
    >>>     stream=True
    >>> ):
    >>>     print(chunk.choices[0].delta.content, end="")
"""

from typing import List, Dict, Any, Iterator, Optional, Union
from dataclasses import dataclass, field
import time
import uuid

from ..generate import Generator, GenerationConfig


@dataclass
class Message:
    """Chat message."""
    role: str
    content: str
    name: Optional[str] = None


@dataclass
class Choice:
    """Completion choice."""
    index: int
    message: Message
    finish_reason: Optional[str] = None


@dataclass
class Usage:
    """Token usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ChatCompletion:
    """Chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = "cyllama"
    choices: List[Choice] = field(default_factory=list)
    usage: Optional[Usage] = None


@dataclass
class DeltaMessage:
    """Streaming message delta."""
    role: Optional[str] = None
    content: Optional[str] = None


@dataclass
class StreamChoice:
    """Streaming completion choice."""
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None


@dataclass
class ChatCompletionChunk:
    """Streaming chat completion chunk."""
    id: str
    object: str = "chat.completion.chunk"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = "cyllama"
    choices: List[StreamChoice] = field(default_factory=list)


class ChatCompletions:
    """Chat completions API."""

    def __init__(self, generator: Generator):
        self.generator = generator

    def create(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.95,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """
        Create a chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 = greedy)
            max_tokens: Maximum tokens to generate
            top_p: Top-p (nucleus) sampling
            stream: If True, return streaming response
            stop: List of stop sequences
            **kwargs: Additional parameters

        Returns:
            ChatCompletion or iterator of ChatCompletionChunk if streaming
        """
        # Format messages into prompt
        prompt = self._format_messages(messages)

        # Create config
        config = GenerationConfig(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop_sequences=stop or [],
            **{k: v for k, v in kwargs.items() if hasattr(GenerationConfig, k)}
        )

        if stream:
            return self._create_stream(messages, prompt, config)
        else:
            return self._create_completion(messages, prompt, config)

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a prompt string."""
        # Simple implementation - could be enhanced with model-specific templates
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

    def _create_completion(
        self,
        messages: List[Dict[str, str]],
        prompt: str,
        config: GenerationConfig
    ) -> ChatCompletion:
        """Create a non-streaming completion."""
        # Tokenize for usage stats
        prompt_tokens = self.generator.vocab.tokenize(prompt, add_special=True, parse_special=True)

        # Generate
        response_text = self.generator(prompt, config=config, stream=False)

        # Count response tokens
        response_tokens = self.generator.vocab.tokenize(response_text, add_special=False, parse_special=False)

        # Create response
        completion = ChatCompletion(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=len(prompt_tokens),
                completion_tokens=len(response_tokens),
                total_tokens=len(prompt_tokens) + len(response_tokens)
            )
        )

        return completion

    def _create_stream(
        self,
        messages: List[Dict[str, str]],
        prompt: str,
        config: GenerationConfig
    ) -> Iterator[ChatCompletionChunk]:
        """Create a streaming completion."""
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        # First chunk with role
        yield ChatCompletionChunk(
            id=completion_id,
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant"),
                    finish_reason=None
                )
            ]
        )

        # Content chunks
        for chunk in self.generator(prompt, config=config, stream=True):
            yield ChatCompletionChunk(
                id=completion_id,
                choices=[
                    StreamChoice(
                        index=0,
                        delta=DeltaMessage(content=chunk),
                        finish_reason=None
                    )
                ]
            )

        # Final chunk
        yield ChatCompletionChunk(
            id=completion_id,
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(),
                    finish_reason="stop"
                )
            ]
        )


class Chat:
    """Chat API."""

    def __init__(self, generator: Generator):
        self.completions = ChatCompletions(generator)


class OpenAICompatibleClient:
    """
    OpenAI-API-compatible client for cyllama.

    This client mimics the OpenAI Python client API, allowing drop-in
    replacement in existing code.

    Example:
        >>> client = OpenAICompatibleClient(model_path="models/llama.gguf")
        >>>
        >>> response = client.chat.completions.create(
        >>>     messages=[{"role": "user", "content": "Hello!"}]
        >>> )
        >>> print(response.choices[0].message.content)
    """

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.7,
        n_gpu_layers: int = 99,
        verbose: bool = False
    ):
        """
        Initialize the OpenAI-compatible client.

        Args:
            model_path: Path to GGUF model file
            temperature: Default sampling temperature
            n_gpu_layers: Number of layers to offload to GPU
            verbose: Print detailed information
        """
        config = GenerationConfig(
            temperature=temperature,
            n_gpu_layers=n_gpu_layers
        )

        self._generator = Generator(model_path, config=config, verbose=verbose)
        self.chat = Chat(self._generator)

    @property
    def generator(self) -> Generator:
        """Access underlying generator."""
        return self._generator


# Convenience function

def create_openai_client(
    model_path: str,
    **kwargs
) -> OpenAICompatibleClient:
    """
    Create an OpenAI-compatible client.

    Args:
        model_path: Path to GGUF model file
        **kwargs: Additional configuration parameters

    Returns:
        OpenAICompatibleClient instance

    Example:
        >>> client = create_openai_client("models/llama.gguf")
        >>> response = client.chat.completions.create(
        >>>     messages=[{"role": "user", "content": "Hi!"}]
        >>> )
    """
    return OpenAICompatibleClient(model_path, **kwargs)
